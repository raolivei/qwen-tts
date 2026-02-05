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
OUTPUT_PATH = os.environ.get("TTS_OUTPUT_PATH", "/data/goldie_demo.wav")
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

# Goldie demo script
target_text = """
Goldie runs on EKS in our shipyard-general dev and stage clusters. We run a single Goldie pod per environment. The app talks to Slack over Socket Mode, a long-lived WebSocket that the pod opens to Slack, so events like mentions and messages get pushed to us and we reply on the same connection. No public URL or ingress needed. Embeddings are done via Bedrock batch jobs, so the app stays simple and we follow the same Shipyard pattern as our other Cloud Solutions apps.

We build and push the Docker image with GitHub Actions to ECR. To deploy, we update the image tag in the env-specific Helm values, and ArgoCD syncs and rolls out to EKS. Our workflow can also open a PR with the new tag so you merge and ArgoCD picks it up.

The interesting piece is cross-account Bedrock. We use IAM role assumption so the pod can call Claude and run batch embedding jobs. No need to move the app. For data, we use PostgreSQL with pgvector for RAG and conversation history, and S3 for vectors and batch inference I/O. Secrets come from Vault via External Secrets, no long-lived creds in the app. Repo access is through our GitHub App with short-lived tokens, no SSH keys.

Today it is dev and stage. We are set up to take it to prod when the team is ready.
""".strip()

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
