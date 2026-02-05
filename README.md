# Qwen3-TTS Voice Cloning

Voice cloning using [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS) for generating natural-sounding speech from text.

## Quick Start

### 1. Setup Environment

```bash
python -m venv qwen-tts-env
source qwen-tts-env/bin/activate
pip install torch soundfile numpy qwen-tts
```

### 2. Download Model

```bash
mkdir -p ~/models
huggingface-cli download Qwen/Qwen3-TTS --local-dir ~/models/qwen3-tts
```

### 3. Prepare Reference Audio

Place your reference audio in `audio/`:
- `voice_ref.wav` - 10-15 seconds of clear speech (24kHz mono recommended)
- `voice_ref_transcript.txt` - Exact transcript of the reference audio

### 4. Run

```bash
python scripts/tts_smoke_test.py
# Output: audio/output/tts_smoke_test.wav
```

## Optimal Parameters

After extensive tuning, these parameters produce excellent results:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Reference length | 10-15s | More speech content = better voice capture |
| x_vector_only_mode | False | Full voice cloning |
| repetition_penalty | 1.4 | Prevents loops without artifacts |
| temperature | 0.6 | Natural prosody |
| subtalker_temperature | 0.55 | Consistent tone |
| top_p | 0.9 | Nucleus sampling |
| max_new_tokens | 80 (short) / 2048 (long) | ~12 tokens/sec |

## Project Structure

```
audio/
  voice_ref.wav              # Your reference audio (10-15s)
  voice_ref_transcript.txt   # Transcript of reference
  output/                    # Generated audio files
scripts/
  tts_smoke_test.py          # Quick local test
  generate_gpu_fixed.py      # GPU generation (for EKS)
  analyze_tts_wav.py         # Audio quality analysis
k8s/
  qwen-tts-pod.yaml          # Kubernetes pod spec for GPU
```

## GPU Generation (EKS)

For faster generation on GPU:

```bash
# Create pod
kubectl apply -f k8s/qwen-tts-pod.yaml

# Setup (one-time)
./scripts/setup_tts.sh

# Generate
./scripts/run_tts_eks.sh default
```

## Troubleshooting

**Output too long (repetition/looping)**
- Increase `repetition_penalty` (try 1.5)
- Decrease `temperature` (try 0.5)
- Reduce `max_new_tokens`

**Output doesn't sound like reference voice**
- Use longer reference audio (15s instead of 5s)
- Ensure transcript matches audio exactly
- Keep `x_vector_only_mode=False`

**Output too short/truncated**
- Increase `max_new_tokens`
- Decrease `repetition_penalty` (try 1.3)

## License

MIT
