#!/usr/bin/env python3
"""
Batch Eldertree podcast generator — runs ALL episodes on-pod.

Copies episode scripts are expected at /data/episodes/episode-NN-script.txt
Voice reference at /data/voice_ref.wav and /data/voice_ref_transcript.txt
Outputs go to /data/output/episode-NN.wav

Usage (on GPU pod):
    nohup python /data/generate_all_episodes.py &> /data/batch.log &
"""
import os
import sys
import time
import glob
import numpy as np
import soundfile as sf
import torch
import re


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("TTS_MODEL_PATH", "/models/qwen3-tts")
VOICE_REF = os.environ.get("TTS_VOICE_REF", "/data/voice_ref.wav")
VOICE_REF_TRANSCRIPT = os.environ.get("TTS_VOICE_REF_TRANSCRIPT", "/data/voice_ref_transcript.txt")
EPISODES_DIR = "/data/episodes"
OUTPUT_DIR = "/data/output"
SEGMENTS_DIR = "/data/segments"

PAUSE_SECONDS = 1.2
MAX_WORDS_PER_SEGMENT = 350
REPETITION_PENALTY = 1.4
TEMPERATURE = 0.7
SUBTALKER_TEMPERATURE = 0.6
TOP_P = 0.9
MAX_NEW_TOKENS = 2048


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------
def normalize_audio(wav, eps=1e-12):
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=-1).astype(np.float32)
    m = np.max(np.abs(x)) if x.size else 0.0
    if m > 1.0 + 1e-6:
        x = x / (m + eps)
    return np.clip(x, -1.0, 1.0)


def load_audio(path):
    wav, sr = sf.read(path)
    return normalize_audio(wav), int(sr)


def make_silence(duration_sec, sr):
    return np.zeros(int(duration_sec * sr), dtype=np.float32)


# ---------------------------------------------------------------------------
# Transcript segmentation
# ---------------------------------------------------------------------------
def split_transcript(text, max_words=350):
    raw_segments = text.split("---")
    raw_segments = [s.strip() for s in raw_segments if s.strip()]

    final_segments = []
    for seg in raw_segments:
        words = seg.split()
        if len(words) <= max_words:
            final_segments.append(seg)
        else:
            paragraphs = seg.split("\n\n")
            current_chunk = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                test = (current_chunk + "\n\n" + para).strip() if current_chunk else para
                if len(test.split()) <= max_words:
                    current_chunk = test
                else:
                    if current_chunk:
                        final_segments.append(current_chunk)
                    if len(para.split()) > max_words:
                        sentences = re.split(r'(?<=[.!?])\s+', para)
                        current_chunk = ""
                        for sent in sentences:
                            sent = sent.strip()
                            if not sent:
                                continue
                            test2 = (current_chunk + " " + sent).strip() if current_chunk else sent
                            if len(test2.split()) <= max_words:
                                current_chunk = test2
                            else:
                                if current_chunk:
                                    final_segments.append(current_chunk)
                                current_chunk = sent
                    else:
                        current_chunk = para
            if current_chunk:
                final_segments.append(current_chunk)

    return final_segments


# ---------------------------------------------------------------------------
# Generate one episode
# ---------------------------------------------------------------------------
def generate_episode(model, ref_audio, ref_text, script_path, output_path, ep_segments_dir):
    ep_name = os.path.basename(script_path)
    print(f"\n{'='*70}")
    print(f"  GENERATING: {ep_name}")
    print(f"{'='*70}")

    with open(script_path, "r") as f:
        full_script = f.read().strip()

    total_words = len(full_script.split())
    segments = split_transcript(full_script, max_words=MAX_WORDS_PER_SEGMENT)
    print(f"  Words: {total_words} | Segments: {len(segments)}")

    os.makedirs(ep_segments_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Clean old segments
    for old in glob.glob(os.path.join(ep_segments_dir, "*.wav")):
        os.remove(old)

    segment_files = []
    sample_rate = None
    total_gen_time = 0
    total_audio_duration = 0

    for i, seg_text in enumerate(segments):
        seg_path = os.path.join(ep_segments_dir, f"segment-{i+1:02d}.wav")
        seg_words = len(seg_text.split())
        print(f"  [{i+1}/{len(segments)}] {seg_words} words...", end=" ", flush=True)

        t0 = time.time()
        try:
            wavs, sr = model.generate_voice_clone(
                text=seg_text,
                language="English",
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=False,
                max_new_tokens=MAX_NEW_TOKENS,
                repetition_penalty=REPETITION_PENALTY,
                temperature=TEMPERATURE,
                subtalker_temperature=SUBTALKER_TEMPERATURE,
                top_p=TOP_P,
            )
            gen_time = time.time() - t0
            duration = len(wavs[0]) / sr
            total_gen_time += gen_time
            total_audio_duration += duration

            sf.write(seg_path, wavs[0], sr)
            segment_files.append(seg_path)
            if sample_rate is None:
                sample_rate = sr

            print(f"{duration:.1f}s audio in {gen_time:.1f}s ({duration/gen_time:.2f}x RT)")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    if not segment_files:
        print(f"  FAILED: No segments generated for {ep_name}")
        return False

    # Concatenate
    silence = make_silence(PAUSE_SECONDS, sample_rate)
    intro_silence = make_silence(0.5, sample_rate)
    outro_silence = make_silence(1.5, sample_rate)

    all_audio = [intro_silence]
    for i, seg_path in enumerate(segment_files):
        seg_wav, _ = sf.read(seg_path)
        seg_wav = normalize_audio(seg_wav)
        all_audio.append(seg_wav)
        if i < len(segment_files) - 1:
            all_audio.append(silence)
    all_audio.append(outro_silence)

    final_wav = np.concatenate(all_audio)
    peak = np.max(np.abs(final_wav))
    if peak > 0:
        target_peak = 0.707  # ~-3dB
        final_wav = final_wav * (target_peak / peak)

    sf.write(output_path, final_wav, sample_rate)
    final_duration = len(final_wav) / sample_rate

    print(f"  ✅ {ep_name}: {final_duration/60:.1f} min | Gen: {total_gen_time/60:.1f} min | {total_audio_duration/total_gen_time:.2f}x RT")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("="*70)
    print("  ELDERTREE BATCH PODCAST GENERATOR")
    print("="*70)

    # Find episode scripts
    scripts = sorted(glob.glob(os.path.join(EPISODES_DIR, "episode-*-script.txt")))
    if not scripts:
        print(f"ERROR: No episode scripts found in {EPISODES_DIR}")
        sys.exit(1)

    print(f"\nFound {len(scripts)} episodes:")
    for s in scripts:
        wc = len(open(s).read().split())
        print(f"  {os.path.basename(s)}: {wc} words")

    # Load model (once!)
    print(f"\nLoading model from {MODEL_PATH}...")
    t0 = time.time()
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cuda",
        dtype=torch.bfloat16,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Load voice reference (once!)
    print(f"Loading voice reference...")
    ref_audio = load_audio(VOICE_REF)
    with open(VOICE_REF_TRANSCRIPT, "r") as f:
        ref_text = f.read().strip()
    print(f"Reference: {len(ref_audio[0])/ref_audio[1]:.1f}s")

    # Generate each episode
    batch_start = time.time()
    results = []

    for script_path in scripts:
        basename = os.path.basename(script_path)
        # Extract episode number: episode-02-script.txt -> 02
        ep_num = basename.split("-")[1]
        output_path = os.path.join(OUTPUT_DIR, f"episode-{ep_num}.wav")
        ep_segments_dir = os.path.join(OUTPUT_DIR, f"segments-ep{ep_num}")

        success = generate_episode(model, ref_audio, ref_text, script_path, output_path, ep_segments_dir)
        results.append((basename, success))

    batch_elapsed = time.time() - batch_start

    # Summary
    print(f"\n{'='*70}")
    print(f"  BATCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {batch_elapsed/60:.1f} minutes")
    print(f"  Results:")
    for name, ok in results:
        status = "✅" if ok else "❌"
        print(f"    {status} {name}")

    # List output files
    print(f"\n  Output files in {OUTPUT_DIR}/:")
    for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "episode-*.wav"))):
        size_mb = os.path.getsize(f) / (1024*1024)
        print(f"    {os.path.basename(f)}: {size_mb:.1f} MB")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
