#!/usr/bin/env bash
# =============================================================================
# Generate an Eldertree podcast episode on EKS GPU pod using Qwen3-TTS
#
# Usage:
#   ./scripts/run_podcast_eks.sh <episode-number> [namespace]
#
# Examples:
#   ./scripts/run_podcast_eks.sh 1            # Generate episode 1
#   ./scripts/run_podcast_eks.sh 1 default    # Explicit namespace
#
# Prerequisites:
#   - VPN connected to reach EKS
#   - GPU pod running: kubectl apply -f k8s/qwen-tts-pod.yaml
#   - One-time setup done: ./scripts/setup_eks_tts.sh default
#   - Voice reference: audio/voice_ref.wav + audio/voice_ref_transcript.txt
#   - Episode script: audio/eldertree/episode-NN-script.txt
# =============================================================================
set -euo pipefail

# --- Args ---
EPISODE_NUM="${1:-}"
NAMESPACE="${2:-default}"
POD="qwen-tts"

if [[ -z "$EPISODE_NUM" ]]; then
    echo "Usage: $0 <episode-number> [namespace]"
    echo ""
    echo "Available episodes:"
    ls -1 audio/eldertree/episode-*-script.txt 2>/dev/null | sed 's/.*episode-0*\([0-9]*\).*/  Episode \1/' || echo "  (none found)"
    exit 1
fi

# Zero-pad episode number
EP_NUM=$(printf "%02d" "$EPISODE_NUM")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
AUDIO_DIR="$PROJECT_DIR/audio"
ELDERTREE_DIR="$AUDIO_DIR/eldertree"

# --- Episode slug mapping ---
declare -A EPISODE_SLUGS=(
    [01]="the-beginning"
    [02]="nvme-migration"
    [03]="network-nightmares"
    [04]="flux-bootstrap"
    [05]="storage-wars"
    [06]="secrets-vaults"
    [07]="ha-quest"
    [08]="troubleshooting"
    [09]="monitoring-stack"
    [10]="production-ready"
)

SLUG="${EPISODE_SLUGS[$EP_NUM]:-episode-$EP_NUM}"
SCRIPT_FILE="$ELDERTREE_DIR/episode-${EP_NUM}-script.txt"
OUTPUT_FILE="$ELDERTREE_DIR/output/episode-${EP_NUM}-${SLUG}.wav"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  ELDERTREE PODCAST GENERATOR                                       ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Episode:   $EP_NUM — $SLUG"
echo "║  Pod:       $POD ($NAMESPACE)"
echo "║  Script:    $SCRIPT_FILE"
echo "║  Output:    $OUTPUT_FILE"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# --- Validate inputs ---
if [[ ! -f "$SCRIPT_FILE" ]]; then
    echo "ERROR: Episode script not found: $SCRIPT_FILE"
    echo ""
    echo "Create it at: audio/eldertree/episode-${EP_NUM}-script.txt"
    exit 1
fi

if [[ ! -f "$AUDIO_DIR/voice_ref.wav" ]]; then
    echo "ERROR: Voice reference not found: $AUDIO_DIR/voice_ref.wav"
    echo ""
    echo "Setup your voice reference:"
    echo "  1. Record 10-15 seconds of your voice"
    echo "  2. Convert: ffmpeg -i recording.m4a -ar 24000 -ac 1 audio/voice_ref.wav"
    echo "  3. Create transcript: audio/voice_ref_transcript.txt"
    exit 1
fi

if [[ ! -f "$AUDIO_DIR/voice_ref_transcript.txt" ]]; then
    echo "ERROR: Voice reference transcript not found: $AUDIO_DIR/voice_ref_transcript.txt"
    exit 1
fi

# Word count for time estimate
WORD_COUNT=$(wc -w < "$SCRIPT_FILE" | tr -d ' ')
EST_MINUTES=$(( WORD_COUNT / 140 ))
echo "Script: $WORD_COUNT words (~${EST_MINUTES} min audio)"
echo ""

# --- Check pod ---
echo "Checking pod status..."
kubectl get pod -n "$NAMESPACE" "$POD" -o wide 2>/dev/null || {
    echo ""
    echo "ERROR: Pod $POD not found in namespace $NAMESPACE"
    echo "Create it: kubectl apply -f k8s/qwen-tts-pod.yaml"
    exit 1
}
echo ""

# --- Copy files to pod ---
echo "Copying files to pod..."
kubectl cp "$AUDIO_DIR/voice_ref.wav"            "$NAMESPACE/$POD:/data/voice_ref.wav"
kubectl cp "$AUDIO_DIR/voice_ref_transcript.txt" "$NAMESPACE/$POD:/data/voice_ref_transcript.txt"
kubectl cp "$SCRIPT_FILE"                        "$NAMESPACE/$POD:/data/episode_script.txt"
kubectl cp "$SCRIPT_DIR/generate_podcast.py"     "$NAMESPACE/$POD:/data/generate_podcast.py"

# Create segments and output dirs on pod
kubectl exec -n "$NAMESPACE" "$POD" -- mkdir -p /data/segments

echo "Files copied."
echo ""

# --- Run generation ---
echo "Starting generation (this will take a while — ~2 min per minute of audio)..."
echo "Estimated total: ~$((EST_MINUTES * 2)) minutes"
echo ""

START_TIME=$(date +%s)

kubectl exec -n "$NAMESPACE" "$POD" -- bash -c \
    'source /opt/venv/bin/activate && python /data/generate_podcast.py'

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
ELAPSED_MIN=$(( ELAPSED / 60 ))

echo ""
echo "Generation completed in ${ELAPSED_MIN} minutes (${ELAPSED}s)"
echo ""

# --- Copy output back ---
echo "Copying output..."
mkdir -p "$(dirname "$OUTPUT_FILE")"
kubectl cp "$NAMESPACE/$POD:/data/episode_output.wav" "$OUTPUT_FILE"

# Also copy segments for debugging/re-editing
SEGMENTS_LOCAL="$ELDERTREE_DIR/output/segments-ep${EP_NUM}"
mkdir -p "$SEGMENTS_LOCAL"
# Get list of segment files on pod
SEGMENT_FILES=$(kubectl exec -n "$NAMESPACE" "$POD" -- ls /data/segments/ 2>/dev/null || echo "")
for seg in $SEGMENT_FILES; do
    kubectl cp "$NAMESPACE/$POD:/data/segments/$seg" "$SEGMENTS_LOCAL/$seg" 2>/dev/null || true
done

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  DONE!                                                             ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Episode:    $OUTPUT_FILE"
echo "║  Segments:   $SEGMENTS_LOCAL/"
echo "║  Gen time:   ${ELAPSED_MIN}m ${ELAPSED}s"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Listen to the output: open $OUTPUT_FILE"
echo "  2. If good, convert to MP3: ffmpeg -i '$OUTPUT_FILE' -codec:a libmp3lame -b:a 128k '${OUTPUT_FILE%.wav}.mp3'"
echo "  3. Upload to Spotify for Podcasters"
