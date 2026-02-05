# Qwen3-TTS Voice Cloning

Generate natural-sounding speech in your own voice using [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS).

## Quick Start

### 1. Setup

```bash
# Clone repo
git clone https://github.com/raolivei/qwen-tts.git
cd qwen-tts

# Create environment
python -m venv venv
source venv/bin/activate
pip install torch soundfile numpy qwen-tts

# Download model (~5GB)
mkdir -p ~/models
huggingface-cli download Qwen/Qwen3-TTS --local-dir ~/models/qwen3-tts
```

### 2. Prepare Your Voice Sample

Record 10-15 seconds of yourself speaking clearly, then:

```bash
# Convert to correct format (24kHz mono WAV)
ffmpeg -i your_recording.m4a -ar 24000 -ac 1 audio/voice_ref.wav

# Create transcript (MUST match exactly what you said)
echo "The exact words you spoke in the recording..." > audio/voice_ref_transcript.txt
```

**Tips for good results:**
- Speak naturally at your normal pace
- Use clear audio (minimal background noise)
- 10-15 seconds works best (longer isn't always better)
- Transcript must match audio exactly

### 3. Generate

```bash
# Test with short sample
python scripts/tts_smoke_test.py

# Custom text
python scripts/tts_smoke_test.py --text "Your custom text here"

# Output: audio/output/tts_smoke_test.wav
```

### 4. Generate Full Demo

For longer content, use the GPU script on EKS:

```bash
# Edit scripts/generate_gpu_fixed.py with your target text
# Then run on GPU pod:
./scripts/run_tts_eks.sh default
```

## Examples

Use Rafael's voice sample as reference:

```bash
python scripts/tts_smoke_test.py \
  --ref examples/rafael/voice_ref.wav \
  --transcript examples/rafael/voice_ref_transcript.txt \
  --text "Hello, this is a test of voice cloning."
```

## Optimal Parameters

These work well for most voices:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Reference length | 10-15s | More speech = better voice capture |
| repetition_penalty | 1.4 | Prevents loops without artifacts |
| temperature | 0.6 | Natural prosody |
| subtalker_temperature | 0.55 | Consistent tone |
| top_p | 0.9 | Nucleus sampling |
| max_new_tokens | 180 | ~15 sec output max |

## Troubleshooting

**Output is repeating/looping**
- Increase `repetition_penalty` to 1.5
- Decrease `temperature` to 0.5

**Output doesn't sound like me**
- Use longer reference (15s instead of 10s)
- Ensure transcript matches audio exactly
- Try different segment of your recording

**Output is cut off**
- Increase `max_new_tokens`
- For 2-minute demo, use `max_new_tokens=2048`

## Project Structure

```
audio/
  voice_ref.wav              # YOUR reference audio
  voice_ref_transcript.txt   # YOUR transcript
  output/                    # Generated files
examples/
  rafael/                    # Example voice sample
scripts/
  tts_smoke_test.py          # Quick test (Mac/local)
  generate_gpu_fixed.py      # Full generation (GPU/EKS)
  analyze_tts_wav.py         # Audio quality check
k8s/
  qwen-tts-pod.yaml          # Kubernetes pod for GPU
```

## For Goldie Demo

1. Record yourself reading the demo script (or any ~15s sample)
2. Save as `audio/voice_ref.wav` with matching transcript
3. Test: `python scripts/tts_smoke_test.py`
4. Generate full demo on EKS GPU for speed

## License

MIT
