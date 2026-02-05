#!/bin/bash
set -ex

echo "=== Installing system dependencies ==="
apt-get update && apt-get install -y python3 python3-pip python3-venv ffmpeg sox git

echo "=== Creating Python venv ==="
python3 -m venv /opt/venv
source /opt/venv/bin/activate

echo "=== Installing Python packages ==="
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install qwen-tts transformers accelerate soundfile huggingface_hub

echo "=== Verifying GPU ==="
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo "=== Downloading Qwen3-TTS model ==="
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='/models/qwen3-tts')"

echo "=== Setup complete! ==="
