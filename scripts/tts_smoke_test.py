#!/usr/bin/env python3
"""
Local smoke test: short ref + one short sentence.
Run with: source ~/qwen-tts-env/bin/activate && python scripts/tts_smoke_test.py
Uses ~/models/qwen3-tts, audio/voice_ref_short.wav, audio/voice_ref_transcript.txt.
Output: audio/output/tts_smoke_test.wav (slow on Mac MPS, but validates no-noise).
"""
import os
import torch
import numpy as np
import soundfile as sf
from qwen_tts import Qwen3TTSModel

HOME = os.path.expanduser("~")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(HOME, "models", "qwen3-tts")
VOICE_REF = os.path.join(PROJECT_DIR, "audio", "voice_ref.wav")
VOICE_REF_TRANSCRIPT = os.path.join(PROJECT_DIR, "audio", "voice_ref_transcript.txt")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "audio", "output", "tts_smoke_test.wav")

# One short sentence ~3 sec — fast to generate, enough to hear if it's clean
TARGET_TEXT = "Hello. This is a quick voice clone test."


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


if __name__ == "__main__":
    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    print("Device:", device)
    print("Loading model from", MODEL_PATH)
    model = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=device, dtype=torch.float32)
    ref_audio = load_audio(VOICE_REF)
    with open(VOICE_REF_TRANSCRIPT, "r") as f:
        ref_text = f.read().strip()
    print("Generating:", TARGET_TEXT)
    # WINNING CONFIG (iteration 9): 15s reference for excellent voice fidelity
    wavs, sr = model.generate_voice_clone(
        text=TARGET_TEXT,
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=False,  # Full voice cloning
        max_new_tokens=80,         # For short test sentence
        repetition_penalty=1.4,    # Prevents loops without artifacts
        temperature=0.6,           # Natural prosody
        subtalker_temperature=0.55,
        top_p=0.9,
    )
    sf.write(OUTPUT_PATH, wavs[0], sr)
    print("Saved:", OUTPUT_PATH)
