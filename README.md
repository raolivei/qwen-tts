# Qwen3-TTS Voice Cloning

Generate natural-sounding speech in your own voice using [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS).

## How to Use (Step by Step)

### Step 1: Clone and Setup

```bash
git clone https://github.com/raolivei/qwen-tts.git
cd qwen-tts

# Create Python environment
python -m venv venv
source venv/bin/activate   # On Mac/Linux
pip install torch soundfile numpy qwen-tts

# Download the model (~5GB, one-time)
mkdir -p ~/models
huggingface-cli download Qwen/Qwen3-TTS --local-dir ~/models/qwen3-tts
```

### Step 2: Record Your Voice Sample

Record **10-15 seconds** of yourself speaking clearly. Tips:
- Speak naturally at your normal pace
- Use clear audio (minimal background noise)
- Read something with varied intonation (not monotone)

### Step 3: Prepare Reference Files

```bash
# Convert your recording to correct format (24kHz mono WAV)
ffmpeg -i your_recording.m4a -ar 24000 -ac 1 audio/voice_ref.wav

# Create transcript - MUST match EXACTLY what you said
echo "The exact words you spoke in the recording go here..." > audio/voice_ref_transcript.txt
```

### Step 4: Test with Short Sample

```bash
# Quick test (~10 seconds output)
python scripts/tts_smoke_test.py

# Listen to: audio/output/tts_smoke_test.wav
```

### Step 5: Generate Full Demo (Optional - needs GPU)

For longer content, use the GPU script on EKS:

```bash
# 1. Create your demo script
echo "Your full demo text goes here..." > audio/target_text.txt

# 2. Create GPU pod (VPN required)
kubectl apply -f k8s/qwen-tts-pod.yaml
kubectl wait --for=condition=Ready pod/qwen-tts --timeout=300s

# 3. One-time setup (downloads model, ~10 min)
./scripts/setup_eks_tts.sh default

# 4. Generate (~2 min per minute of audio)
./scripts/run_tts_eks.sh default

# Output: audio/output/goldie_demo.wav
```

## Quick Reference

### Test with Example Voice (Rafael's)

```bash
python scripts/tts_smoke_test.py \
  --ref examples/rafael/voice_ref.wav \
  --transcript examples/rafael/voice_ref_transcript.txt \
  --text "Hello, this is a test of voice cloning."
```

### Custom Text Generation

```bash
python scripts/tts_smoke_test.py --text "Your custom text here"
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_MODEL_PATH` | `~/models/qwen3-tts` | Path to model |
| `TTS_VOICE_REF` | `audio/voice_ref.wav` | Reference audio |
| `TTS_OUTPUT_PATH` | `audio/output/...` | Output file |

## Optimal Parameters

These settings produce the best results:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Reference length | 10-15s | More speech content = better |
| repetition_penalty | 1.4 | Prevents loops |
| temperature | 0.6 | Natural prosody |
| max_new_tokens | 180 (short) / 2048 (long) | ~12 tokens/sec |

## Troubleshooting

### Output is repeating/looping
- Increase `repetition_penalty` to 1.5
- Decrease `temperature` to 0.5
- Reduce `max_new_tokens`

### Output doesn't sound like me
- Use longer reference (15s instead of 10s)
- **Ensure transcript matches audio EXACTLY**
- Try a different segment of your recording

### Output is too short/cut off
- Increase `max_new_tokens` (use 2048 for full demos)

## Project Structure

```
qwen-tts/
├── audio/
│   ├── voice_ref.wav           # YOUR voice reference
│   ├── voice_ref_transcript.txt # YOUR transcript
│   ├── target_text.txt         # (optional) Custom demo script
│   └── output/                 # Generated audio
├── examples/
│   └── rafael/                 # Example voice sample
├── scripts/
│   ├── tts_smoke_test.py       # Quick test (Mac/local)
│   ├── generate_gpu_fixed.py   # Full generation (GPU)
│   ├── run_tts_eks.sh          # EKS helper
│   └── analyze_tts_wav.py      # Audio quality check
└── k8s/
    └── qwen-tts-pod.yaml       # GPU pod spec
```

## For Goldie Demo

1. Record 10-15 seconds of yourself speaking
2. Save as `audio/voice_ref.wav` with matching `audio/voice_ref_transcript.txt`
3. Test locally: `python scripts/tts_smoke_test.py`
4. If good, generate full demo on EKS GPU

## License

MIT
