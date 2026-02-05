#!/usr/bin/env bash
# Generate voice clone on EKS GPU pod.
#
# Usage:
#   ./scripts/run_tts_eks.sh [namespace]
#   ./scripts/run_tts_eks.sh default
#
# To use custom target text:
#   1. Create audio/target_text.txt with your script
#   2. Run this script - it will copy and use that file
#
# Prereqs:
#   - kubectl configured to reach your GPU-enabled cluster
#   - Pod running: kubectl apply -f k8s/qwen-tts-pod.yaml
#   - One-time setup: ./scripts/setup_eks_tts.sh default
set -e

NAMESPACE="${1:-default}"
POD="qwen-tts"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
AUDIO_DIR="$PROJECT_DIR/audio"

echo "=== Qwen3-TTS EKS Generator ==="
echo "Pod: $POD"
echo "Namespace: $NAMESPACE"

# Check pod exists
kubectl get pod -n "$NAMESPACE" "$POD" -o wide 2>/dev/null || {
    echo "ERROR: Pod $POD not found."
    echo "Create it with: kubectl apply -f k8s/qwen-tts-pod.yaml"
    exit 1
}

# Check reference files exist
if [[ ! -f "$AUDIO_DIR/voice_ref.wav" ]]; then
    echo "ERROR: Reference audio not found: $AUDIO_DIR/voice_ref.wav"
    echo ""
    echo "To set up:"
    echo "  1. Record 10-15 seconds of your voice"
    echo "  2. Save as audio/voice_ref.wav"
    echo "  3. Create audio/voice_ref_transcript.txt"
    exit 1
fi

echo ""
echo "Copying files to pod..."
kubectl cp "$AUDIO_DIR/voice_ref.wav"             "$NAMESPACE/$POD:/data/voice_ref.wav"
kubectl cp "$AUDIO_DIR/voice_ref_transcript.txt"  "$NAMESPACE/$POD:/data/voice_ref_transcript.txt"
kubectl cp "$SCRIPT_DIR/generate_gpu_fixed.py"    "$NAMESPACE/$POD:/data/generate_gpu_fixed.py"

# Copy target text if exists
if [[ -f "$AUDIO_DIR/target_text.txt" ]]; then
    echo "Using custom target text from audio/target_text.txt"
    kubectl cp "$AUDIO_DIR/target_text.txt" "$NAMESPACE/$POD:/data/target_text.txt"
else
    echo "Using default target text"
fi

echo ""
echo "Running generation (15-20 min on GPU)..."
export X_VECTOR_ONLY="${X_VECTOR_ONLY:-0}"
kubectl exec -n "$NAMESPACE" "$POD" -- env X_VECTOR_ONLY="$X_VECTOR_ONLY" bash -c 'source /opt/venv/bin/activate && python /data/generate_gpu_fixed.py'

echo ""
echo "Copying output..."
OUTPUT_DIR="$AUDIO_DIR/output"
mkdir -p "$OUTPUT_DIR"
kubectl cp "$NAMESPACE/$POD:/data/tts_output.wav" "$OUTPUT_DIR/tts_output.wav"

echo ""
echo "=== Done! ==="
echo "Output: $OUTPUT_DIR/tts_output.wav"
