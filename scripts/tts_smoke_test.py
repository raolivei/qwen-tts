#!/usr/bin/env python3
"""
Voice cloning smoke test - generates a short audio sample.

Usage:
    python scripts/tts_smoke_test.py                    # Uses audio/voice_ref.wav
    python scripts/tts_smoke_test.py --ref examples/rafael/voice_ref.wav --transcript examples/rafael/voice_ref_transcript.txt
    
Environment variables:
    TTS_MODEL_PATH - Path to Qwen3-TTS model (default: ~/models/qwen3-tts)
    TTS_VOICE_REF - Path to reference audio
    TTS_VOICE_REF_TRANSCRIPT - Path to reference transcript
    TTS_OUTPUT_PATH - Output path (default: audio/output/tts_smoke_test.wav)
"""
import os
import sys
import argparse
import torch
import numpy as np
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Defaults
HOME = os.path.expanduser("~")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL = os.path.join(HOME, "models", "qwen3-tts")
DEFAULT_REF = os.path.join(PROJECT_DIR, "audio", "voice_ref.wav")
DEFAULT_TRANSCRIPT = os.path.join(PROJECT_DIR, "audio", "voice_ref_transcript.txt")
DEFAULT_OUTPUT = os.path.join(PROJECT_DIR, "audio", "output", "tts_smoke_test.wav")

# Test text (~10 seconds)
DEFAULT_TEXT = "Goldie runs on EKS in our shipyard dev cluster. We run a single pod per environment. The app talks to Slack over Socket Mode, a long-lived WebSocket connection."


def normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        y = x.astype(np.float32) / max(abs(info.min), info.max) if info.min < 0 else (x.astype(np.float32) - (info.max + 1) / 2.0) / ((info.max + 1) / 2.0)
    else:
        y = x.astype(np.float32)
        if np.max(np.abs(y)) > 1.0 + 1e-6:
            y = y / (np.max(np.abs(y)) + eps)
    if clip:
        y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return y


def load_audio(path):
    wav, sr = sf.read(path)
    return normalize_audio(wav), int(sr)


def main():
    parser = argparse.ArgumentParser(description="Voice cloning smoke test")
    parser.add_argument("--ref", help="Path to reference audio WAV file")
    parser.add_argument("--transcript", help="Path to reference transcript file")
    parser.add_argument("--text", help="Text to synthesize", default=DEFAULT_TEXT)
    parser.add_argument("--output", "-o", help="Output WAV path")
    parser.add_argument("--model", help="Path to Qwen3-TTS model")
    args = parser.parse_args()

    # Resolve paths (env vars > args > defaults)
    model_path = args.model or os.environ.get("TTS_MODEL_PATH", DEFAULT_MODEL)
    ref_path = args.ref or os.environ.get("TTS_VOICE_REF", DEFAULT_REF)
    transcript_path = args.transcript or os.environ.get("TTS_VOICE_REF_TRANSCRIPT", DEFAULT_TRANSCRIPT)
    output_path = args.output or os.environ.get("TTS_OUTPUT_PATH", DEFAULT_OUTPUT)

    # Validate inputs
    if not os.path.exists(ref_path):
        print(f"ERROR: Reference audio not found: {ref_path}")
        print("\nTo get started:")
        print("  1. Record 10-15 seconds of your voice")
        print("  2. Save as audio/voice_ref.wav (24kHz mono)")
        print("  3. Create audio/voice_ref_transcript.txt with exact words spoken")
        print("\nOr use an example:")
        print("  python scripts/tts_smoke_test.py --ref examples/rafael/voice_ref.wav --transcript examples/rafael/voice_ref_transcript.txt")
        sys.exit(1)

    if not os.path.exists(transcript_path):
        print(f"ERROR: Transcript not found: {transcript_path}")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Device selection
    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Reference: {ref_path}")

    # Load model
    print("\nLoading model...")
    model = Qwen3TTSModel.from_pretrained(model_path, device_map=device, dtype=torch.float32)

    # Load reference
    ref_audio = load_audio(ref_path)
    with open(transcript_path) as f:
        ref_text = f.read().strip()
    print(f"Reference duration: {len(ref_audio[0]) / ref_audio[1]:.1f}s")
    print(f"Reference transcript: {ref_text[:60]}...")

    # Generate
    print(f"\nGenerating: {args.text[:60]}...")
    wavs, sr = model.generate_voice_clone(
        text=args.text,
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=False,
        max_new_tokens=180,        # ~15 seconds max
        repetition_penalty=1.4,    # Prevents loops
        temperature=0.6,           # Natural prosody
        subtalker_temperature=0.55,
        top_p=0.9,
    )

    # Save
    sf.write(output_path, wavs[0], sr)
    duration = len(wavs[0]) / sr
    print(f"\nSaved: {output_path}")
    print(f"Duration: {duration:.1f}s")


if __name__ == "__main__":
    main()
