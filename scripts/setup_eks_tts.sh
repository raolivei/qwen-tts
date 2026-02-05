#!/usr/bin/env bash
# One-time setup: copy setup_tts.sh into the EKS qwen-tts pod and run it.
# Run this after creating the pod (kubectl apply -f qwen-tts-pod.yaml) and before run_tts_eks.sh.
# Usage: ./setup_eks_tts.sh [namespace]
set -e

NAMESPACE="${1:-default}"
POD="qwen-tts"
SCRIPT_DIR="$(dirname "$0")"

echo "Using pod $POD in namespace $NAMESPACE"
kubectl get pod -n "$NAMESPACE" "$POD" -o wide 2>/dev/null || { echo "Pod $POD not found. Create it with: kubectl apply -f qwen-tts-pod.yaml"; exit 1; }

echo "Copying setup_tts.sh to pod..."
kubectl cp "$SCRIPT_DIR/setup_tts.sh" "$NAMESPACE/$POD:/tmp/setup_tts.sh"

echo "Running setup (installs venv, PyTorch CUDA, qwen-tts, downloads model; ~5–10 min)..."
kubectl exec -n "$NAMESPACE" "$POD" -- bash -c "chmod +x /tmp/setup_tts.sh && /tmp/setup_tts.sh"

echo "Setup complete. You can now run: ./run_tts_eks.sh $NAMESPACE"
