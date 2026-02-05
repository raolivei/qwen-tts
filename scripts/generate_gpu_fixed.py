#!/usr/bin/env python3
"""
Qwen3-TTS voice cloning - FIXED version based on official HuggingFace implementation
"""
import os
import torch
import numpy as np
import soundfile as sf
from qwen_tts import Qwen3TTSModel

MODEL_PATH = os.environ.get("TTS_MODEL_PATH", "/models/qwen3-tts")
# Use 10-15s reference for best voice fidelity
VOICE_REF = os.environ.get("TTS_VOICE_REF", "/data/voice_ref.wav")
VOICE_REF_TRANSCRIPT = os.environ.get("TTS_VOICE_REF_TRANSCRIPT", "/data/voice_ref_transcript.txt")
OUTPUT_PATH = os.environ.get("TTS_OUTPUT_PATH", "/data/tts_output.wav")
# x_vector_only_mode=False gives best voice fidelity with tuned params
X_VECTOR_ONLY = os.environ.get("X_VECTOR_ONLY", "0").strip().lower() in ("1", "true", "yes")

# Tuned parameters from iteration 9 (excellent results with 15s reference)
def _float_env(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None else default

REPETITION_PENALTY = _float_env("TTS_REPETITION_PENALTY", 1.4)
TEMPERATURE = _float_env("TTS_TEMPERATURE", 0.6)
SUBTALKER_TEMPERATURE = _float_env("TTS_SUBTALKER_TEMPERATURE", 0.55)
TOP_P = _float_env("TTS_TOP_P", 0.9)


def _int_env(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default


# Cap tokens to limit repetition runaways; ~12 tokens/s. Short clip: 96–120; long text: 2048.
MAX_NEW_TOKENS = _int_env("TTS_MAX_NEW_TOKENS", 2048)


def normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range - from official code."""
    x = np.asarray(wav)
    
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")
    
    if clip:
        y = np.clip(y, -1.0, 1.0)
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    
    return y


def load_audio(path):
    """Load audio and return as (wav, sr) tuple."""
    wav, sr = sf.read(path)
    wav = normalize_audio(wav)
    return wav, int(sr)


print("=" * 60)
print("Qwen3-TTS Voice Clone - FIXED VERSION")
print("=" * 60)

# Load model with bfloat16 (as per official code)
print("\n[1/4] Loading model...")
model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    device_map="cuda",
    dtype=torch.bfloat16,  # Changed from float16!
)
print("Model loaded!")

# Load reference audio (short clip, 3–10s recommended)
print("\n[2/4] Loading reference audio (short clip)...")
ref_audio = load_audio(VOICE_REF)
print(f"Audio loaded: {len(ref_audio[0])} samples, {ref_audio[1]} Hz")
print(f"Duration: {len(ref_audio[0]) / ref_audio[1]:.1f} seconds")

# Load transcript (must match the short clip)
print("\n[3/4] Loading reference transcript...")
with open(VOICE_REF_TRANSCRIPT, "r") as f:
    ref_text = f.read().strip()
print(f"Transcript: {ref_text[:80]}...")

# Load target text from file or use default
TARGET_TEXT_FILE = os.environ.get("TTS_TARGET_TEXT_FILE", "/data/target_text.txt")
DEFAULT_TEXT = """
This is a sample text for voice cloning demonstration. The system uses a neural network model to learn the characteristics of your voice from a short reference recording.

Once trained, it can synthesize new speech that sounds like you, reading any text you provide. The quality depends on the reference audio clarity, the accuracy of the transcript, and the generation parameters.

For best results, use a fifteen second reference with clear speech and minimal background noise. Make sure the transcript matches exactly what was spoken in the reference.

You can customize this text by creating your own target text file. The model works well with conversational content and natural phrasing.
""".strip()

# Try to load from file, fall back to default
if os.path.exists(TARGET_TEXT_FILE):
    with open(TARGET_TEXT_FILE, "r") as f:
        target_text = f.read().strip()
    print(f"Loaded target text from: {TARGET_TEXT_FILE}")
else:
    target_text = DEFAULT_TEXT
    print("Using default target text")

print(f"\n[4/4] Generating audio for {len(target_text.split())} words...")
print("This may take a few minutes...")

# Generate with tuned params from iteration 9
wavs, sr = model.generate_voice_clone(
    text=target_text,
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
    x_vector_only_mode=X_VECTOR_ONLY,
    max_new_tokens=MAX_NEW_TOKENS,
    repetition_penalty=REPETITION_PENALTY,
    temperature=TEMPERATURE,
    subtalker_temperature=SUBTALKER_TEMPERATURE,
    top_p=TOP_P,
)

# Save output
sf.write(OUTPUT_PATH, wavs[0], sr)
print(f"\nAudio saved to: {OUTPUT_PATH}")
print(f"Duration: {len(wavs[0]) / sr:.1f} seconds")
print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
