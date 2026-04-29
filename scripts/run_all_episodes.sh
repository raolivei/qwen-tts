#!/usr/bin/env bash
# =============================================================================
# Generate ALL Eldertree podcast episodes (2-10) sequentially on EKS GPU pod
#
# Usage:
#   ./scripts/run_all_episodes.sh [start_episode] [end_episode]
#
# Examples:
#   ./scripts/run_all_episodes.sh          # Episodes 2-10
#   ./scripts/run_all_episodes.sh 5 7      # Episodes 5-7 only
# =============================================================================
set -euo pipefail

START_EP="${1:-2}"
END_EP="${2:-10}"
NAMESPACE="default"
POD="qwen-tts"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
AUDIO_DIR="$PROJECT_DIR/audio"
ELDERTREE_DIR="$AUDIO_DIR/eldertree"
LOG_DIR="$ELDERTREE_DIR/output/logs"
mkdir -p "$LOG_DIR"

TOTAL_START=$(date +%s)

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  ELDERTREE BATCH PODCAST GENERATOR                                 ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Episodes:  $START_EP through $END_EP                              "
echo "║  Pod:       $POD ($NAMESPACE)                                      "
echo "║  Started:   $(date '+%Y-%m-%d %H:%M:%S')                          "
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Episode slug mapping
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

# --- Pre-flight: copy shared files once ---
echo "[PRE-FLIGHT] Copying voice reference and generator script to pod..."
kubectl cp "$AUDIO_DIR/voice_ref.wav"            "$NAMESPACE/$POD:/data/voice_ref.wav"
kubectl cp "$AUDIO_DIR/voice_ref_transcript.txt" "$NAMESPACE/$POD:/data/voice_ref_transcript.txt"
kubectl cp "$SCRIPT_DIR/generate_podcast.py"     "$NAMESPACE/$POD:/data/generate_podcast.py"
kubectl exec -n "$NAMESPACE" "$POD" -- mkdir -p /data/segments
echo "[PRE-FLIGHT] Done."
echo ""

COMPLETED=0
FAILED=0
RESULTS=()

for ep_num in $(seq "$START_EP" "$END_EP"); do
    EP_NUM=$(printf "%02d" "$ep_num")
    SLUG="${EPISODE_SLUGS[$EP_NUM]:-episode-$EP_NUM}"
    SCRIPT_FILE="$ELDERTREE_DIR/episode-${EP_NUM}-script.txt"
    OUTPUT_FILE="$ELDERTREE_DIR/output/episode-${EP_NUM}-${SLUG}.wav"
    LOG_FILE="$LOG_DIR/episode-${EP_NUM}.log"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  EPISODE $EP_NUM — $SLUG"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ ! -f "$SCRIPT_FILE" ]]; then
        echo "  SKIP: Script not found: $SCRIPT_FILE"
        RESULTS+=("Episode $EP_NUM ($SLUG): SKIPPED — no script")
        continue
    fi

    WORD_COUNT=$(wc -w < "$SCRIPT_FILE" | tr -d ' ')
    EST_MIN=$(( WORD_COUNT / 140 ))
    echo "  Words: $WORD_COUNT | Est. audio: ~${EST_MIN} min | Est. gen time: ~$((EST_MIN * 2)) min"
    echo "  Started: $(date '+%H:%M:%S')"

    # Copy episode script to pod
    kubectl cp "$SCRIPT_FILE" "$NAMESPACE/$POD:/data/episode_script.txt"

    # Clean previous segments
    kubectl exec -n "$NAMESPACE" "$POD" -- bash -c 'rm -f /data/segments/*.wav /data/episode_output.wav' 2>/dev/null || true

    EP_START=$(date +%s)

    # Run generation
    if kubectl exec -n "$NAMESPACE" "$POD" -- bash -c \
        'source /opt/venv/bin/activate && python /data/generate_podcast.py' 2>&1 | tee "$LOG_FILE"; then

        EP_END=$(date +%s)
        EP_ELAPSED=$(( EP_END - EP_START ))
        EP_ELAPSED_MIN=$(( EP_ELAPSED / 60 ))

        # Copy output back
        mkdir -p "$(dirname "$OUTPUT_FILE")"
        kubectl cp "$NAMESPACE/$POD:/data/episode_output.wav" "$OUTPUT_FILE"

        # Copy segments
        SEGMENTS_LOCAL="$ELDERTREE_DIR/output/segments-ep${EP_NUM}"
        mkdir -p "$SEGMENTS_LOCAL"
        SEGMENT_FILES=$(kubectl exec -n "$NAMESPACE" "$POD" -- ls /data/segments/ 2>/dev/null || echo "")
        for seg in $SEGMENT_FILES; do
            kubectl cp "$NAMESPACE/$POD:/data/segments/$seg" "$SEGMENTS_LOCAL/$seg" 2>/dev/null || true
        done

        # Convert to MP3
        if command -v ffmpeg &>/dev/null && [[ -f "$OUTPUT_FILE" ]]; then
            MP3_FILE="${OUTPUT_FILE%.wav}.mp3"
            ffmpeg -y -i "$OUTPUT_FILE" -codec:a libmp3lame -b:a 128k "$MP3_FILE" 2>/dev/null
            MP3_SIZE=$(du -h "$MP3_FILE" 2>/dev/null | cut -f1)
            echo "  MP3: $MP3_FILE ($MP3_SIZE)"
        fi

        WAV_SIZE=$(du -h "$OUTPUT_FILE" 2>/dev/null | cut -f1)
        echo "  ✅ DONE in ${EP_ELAPSED_MIN}m (${EP_ELAPSED}s) — $WAV_SIZE"
        RESULTS+=("Episode $EP_NUM ($SLUG): ✅ ${EP_ELAPSED_MIN}m — $WAV_SIZE")
        COMPLETED=$((COMPLETED + 1))
    else
        EP_END=$(date +%s)
        EP_ELAPSED=$(( EP_END - EP_START ))
        echo "  ❌ FAILED after ${EP_ELAPSED}s — see $LOG_FILE"
        RESULTS+=("Episode $EP_NUM ($SLUG): ❌ FAILED")
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))
TOTAL_MIN=$(( TOTAL_ELAPSED / 60 ))

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  BATCH GENERATION COMPLETE                                         ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Completed:  $COMPLETED episodes                                   "
echo "║  Failed:     $FAILED episodes                                      "
echo "║  Total time: ${TOTAL_MIN} minutes (${TOTAL_ELAPSED}s)             "
echo "║  Finished:   $(date '+%Y-%m-%d %H:%M:%S')                         "
echo "╠══════════════════════════════════════════════════════════════════════╣"
for result in "${RESULTS[@]}"; do
    echo "║  $result"
done
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Output files: $ELDERTREE_DIR/output/"
echo "Logs: $LOG_DIR/"
