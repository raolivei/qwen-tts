#!/usr/bin/env bash
# Copy short-ref files to EKS qwen-tts pod and run voice-clone generation.
#
# Prereqs:
#   - VPN to reach EKS (e.g. shipyard-general-us-west-dev).
#   - Pod running: kubectl apply -f qwen-tts-pod.yaml
#   - One-time setup already done in pod: copy setup_tts.sh + run it (creates /opt/venv, /models/qwen3-tts).
#
# Usage: ./run_tts_eks.sh [namespace]
# Example: ./run_tts_eks.sh default
set -e

NAMESPACE="${1:-default}"
POD="qwen-tts"
SCRIPT_DIR="$(dirname "$0")"
AUDIO_DIR="$(dirname "$SCRIPT_DIR")/audio"

echo "Using pod $POD in namespace $NAMESPACE"
kubectl get pod -n "$NAMESPACE" "$POD" -o wide 2>/dev/null || { echo "Pod $POD not found. Create it with: kubectl apply -f qwen-tts-pod.yaml"; exit 1; }

echo "Copying reference files and script to /data..."
kubectl cp "$AUDIO_DIR/voice_ref.wav"             "$NAMESPACE/$POD:/data/voice_ref.wav"
kubectl cp "$AUDIO_DIR/voice_ref_transcript.txt"  "$NAMESPACE/$POD:/data/voice_ref_transcript.txt"
kubectl cp "$SCRIPT_DIR/generate_gpu_fixed.py"        "$NAMESPACE/$POD:/data/generate_gpu_fixed.py"

echo "Running generation (this may take 15–20 min)..."
# Optional: X_VECTOR_ONLY=1 ./run_tts_eks.sh to use speaker embedding only (if output still noisy)
export X_VECTOR_ONLY="${X_VECTOR_ONLY:-0}"
kubectl exec -n "$NAMESPACE" "$POD" -- env X_VECTOR_ONLY="$X_VECTOR_ONLY" bash -c 'source /opt/venv/bin/activate && python /data/generate_gpu_fixed.py'

echo "Copying output back..."
OUTPUT_DIR="$AUDIO_DIR/output"
mkdir -p "$OUTPUT_DIR"
kubectl cp "$NAMESPACE/$POD:/data/goldie_demo.wav" "$OUTPUT_DIR/goldie_demo.wav"

echo "Done. Listen to: $OUTPUT_DIR/goldie_demo.wav"
