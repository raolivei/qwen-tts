#!/usr/bin/env python3
"""
Qwen3-TTS Podcast Generator — Segmented long-form audio generation.

Splits a long transcript into segments (by "---" markers or word count),
generates each segment separately with the same voice reference, inserts
natural pauses between segments, and concatenates into a final episode WAV.

Usage (GPU pod):
    python generate_podcast.py

Environment variables:
    TTS_MODEL_PATH              Model path (default: /models/qwen3-tts)
    TTS_VOICE_REF               Voice reference WAV (default: /data/voice_ref.wav)
    TTS_VOICE_REF_TRANSCRIPT    Reference transcript (default: /data/voice_ref_transcript.txt)
    TTS_SCRIPT_FILE             Episode script (default: /data/episode_script.txt)
    TTS_OUTPUT_PATH             Final output WAV (default: /data/episode_output.wav)
    TTS_SEGMENTS_DIR            Per-segment WAVs (default: /data/segments/)
    TTS_PAUSE_SECONDS           Silence between segments (default: 1.2)
    TTS_MAX_WORDS_PER_SEGMENT   Max words per segment when no --- markers (default: 350)
    TTS_REPETITION_PENALTY      Repetition penalty (default: 1.4)
    TTS_TEMPERATURE             Temperature (default: 0.7)
    TTS_SUBTALKER_TEMPERATURE   Subtalker temperature (default: 0.6)
    TTS_TOP_P                   Top-p sampling (default: 0.9)
    TTS_MAX_NEW_TOKENS          Max new tokens per segment (default: 2048)
"""
import os
import sys
import time
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


# ---------------------------------------------------------------------------
# Configuration (env vars with sane defaults)
# ---------------------------------------------------------------------------

def _env(name, default):
    return os.environ.get(name, default)

def _float_env(name, default):
    v = os.environ.get(name)
    return float(v) if v is not None else default

def _int_env(name, default):
    v = os.environ.get(name)
    return int(v) if v is not None else default


MODEL_PATH              = _env("TTS_MODEL_PATH", "/models/qwen3-tts")
VOICE_REF               = _env("TTS_VOICE_REF", "/data/voice_ref.wav")
VOICE_REF_TRANSCRIPT    = _env("TTS_VOICE_REF_TRANSCRIPT", "/data/voice_ref_transcript.txt")
SCRIPT_FILE             = _env("TTS_SCRIPT_FILE", "/data/episode_script.txt")
OUTPUT_PATH             = _env("TTS_OUTPUT_PATH", "/data/episode_output.wav")
SEGMENTS_DIR            = _env("TTS_SEGMENTS_DIR", "/data/segments")
PAUSE_SECONDS           = _float_env("TTS_PAUSE_SECONDS", 1.2)
MAX_WORDS_PER_SEGMENT   = _int_env("TTS_MAX_WORDS_PER_SEGMENT", 350)
REPETITION_PENALTY      = _float_env("TTS_REPETITION_PENALTY", 1.4)
# Slightly higher temperature for more dynamic/engaging delivery
TEMPERATURE             = _float_env("TTS_TEMPERATURE", 0.7)
SUBTALKER_TEMPERATURE   = _float_env("TTS_SUBTALKER_TEMPERATURE", 0.6)
TOP_P                   = _float_env("TTS_TOP_P", 0.9)
MAX_NEW_TOKENS          = _int_env("TTS_MAX_NEW_TOKENS", 2048)


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def normalize_audio(wav, eps=1e-12):
    """Normalize audio to float32 in [-1, 1]."""
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=-1).astype(np.float32)
    m = np.max(np.abs(x)) if x.size else 0.0
    if m > 1.0 + 1e-6:
        x = x / (m + eps)
    return np.clip(x, -1.0, 1.0)


def load_audio(path):
    """Load audio and return (wav, sr)."""
    wav, sr = sf.read(path)
    return normalize_audio(wav), int(sr)


def make_silence(duration_sec, sr):
    """Create a silence array."""
    return np.zeros(int(duration_sec * sr), dtype=np.float32)


# ---------------------------------------------------------------------------
# Transcript segmentation
# ---------------------------------------------------------------------------

def split_transcript(text, max_words=350):
    """
    Split transcript into segments.
    
    Strategy:
    1. First try splitting on '---' markers (explicit segment boundaries)
    2. If a segment is still too long, split at paragraph boundaries
    3. Last resort: split at sentence boundaries near the word limit
    """
    # Step 1: Split on --- markers
    raw_segments = text.split("---")
    raw_segments = [s.strip() for s in raw_segments if s.strip()]
    
    # Step 2: Further split any segment that's too long
    final_segments = []
    for seg in raw_segments:
        words = seg.split()
        if len(words) <= max_words:
            final_segments.append(seg)
        else:
            # Split on paragraph boundaries (double newline)
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
                    # If a single paragraph is too long, split at sentences
                    if len(para.split()) > max_words:
                        sentences = _split_sentences(para)
                        current_chunk = ""
                        for sent in sentences:
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


def _split_sentences(text):
    """Rough sentence splitter for English text."""
    import re
    # Split on sentence-ending punctuation followed by space or newline
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  ELDERTREE PODCAST GENERATOR — Segmented Long-Form Audio")
    print("=" * 70)
    
    # --- Validate inputs ---
    for label, path in [("Model", MODEL_PATH), ("Voice ref", VOICE_REF),
                        ("Transcript", VOICE_REF_TRANSCRIPT), ("Script", SCRIPT_FILE)]:
        if not os.path.exists(path):
            print(f"\nERROR: {label} not found: {path}")
            sys.exit(1)
    
    os.makedirs(SEGMENTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    
    # --- Load model ---
    print(f"\n[1/5] Loading model from {MODEL_PATH}...")
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cuda",
        dtype=torch.bfloat16,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    
    # --- Load voice reference ---
    print(f"\n[2/5] Loading voice reference...")
    ref_audio = load_audio(VOICE_REF)
    with open(VOICE_REF_TRANSCRIPT, "r") as f:
        ref_text = f.read().strip()
    print(f"  Reference: {len(ref_audio[0]) / ref_audio[1]:.1f}s")
    print(f"  Transcript: \"{ref_text[:80]}...\"")
    
    # --- Load and segment script ---
    print(f"\n[3/5] Loading episode script...")
    with open(SCRIPT_FILE, "r") as f:
        full_script = f.read().strip()
    
    total_words = len(full_script.split())
    segments = split_transcript(full_script, max_words=MAX_WORDS_PER_SEGMENT)
    print(f"  Total words: {total_words}")
    print(f"  Segments: {len(segments)}")
    est_minutes = total_words / 140  # ~140 wpm for natural podcast pace
    print(f"  Estimated duration: ~{est_minutes:.0f} minutes")
    
    for i, seg in enumerate(segments):
        words = len(seg.split())
        preview = seg[:80].replace("\n", " ")
        print(f"  Segment {i+1}: {words} words — \"{preview}...\"")
    
    # --- Generate each segment ---
    print(f"\n[4/5] Generating audio segments...")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Repetition penalty: {REPETITION_PENALTY}")
    print(f"  Max tokens/segment: {MAX_NEW_TOKENS}")
    print(f"  Pause between segments: {PAUSE_SECONDS}s")
    print()
    
    segment_files = []
    sample_rate = None
    total_gen_time = 0
    total_audio_duration = 0
    
    for i, seg_text in enumerate(segments):
        seg_path = os.path.join(SEGMENTS_DIR, f"segment-{i+1:02d}.wav")
        seg_words = len(seg_text.split())
        
        print(f"  [{i+1}/{len(segments)}] Generating segment ({seg_words} words)...")
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
            
            print(f"         Duration: {duration:.1f}s | Gen time: {gen_time:.1f}s | "
                  f"Ratio: {duration/gen_time:.2f}x realtime")
            
        except Exception as e:
            print(f"  ERROR generating segment {i+1}: {e}")
            print(f"  Skipping segment and continuing...")
            continue
    
    if not segment_files:
        print("\nERROR: No segments were generated successfully.")
        sys.exit(1)
    
    # --- Concatenate segments ---
    print(f"\n[5/5] Concatenating {len(segment_files)} segments...")
    
    silence = make_silence(PAUSE_SECONDS, sample_rate)
    # Longer pause at the very beginning (intro breath)
    intro_silence = make_silence(0.5, sample_rate)
    # Longer pause at the very end
    outro_silence = make_silence(1.5, sample_rate)
    
    all_audio = [intro_silence]
    for i, seg_path in enumerate(segment_files):
        seg_wav, _ = sf.read(seg_path)
        seg_wav = normalize_audio(seg_wav)
        all_audio.append(seg_wav)
        
        # Add pause between segments (not after the last one)
        if i < len(segment_files) - 1:
            all_audio.append(silence)
    
    all_audio.append(outro_silence)
    
    # Concatenate
    final_wav = np.concatenate(all_audio)
    
    # Light normalization on final output — target -3dB peak
    peak = np.max(np.abs(final_wav))
    if peak > 0:
        target_peak = 0.707  # ~-3dB
        final_wav = final_wav * (target_peak / peak)
    
    sf.write(OUTPUT_PATH, final_wav, sample_rate)
    
    final_duration = len(final_wav) / sample_rate
    
    # --- Summary ---
    print()
    print("=" * 70)
    print("  GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Output:          {OUTPUT_PATH}")
    print(f"  Duration:        {final_duration/60:.1f} minutes ({final_duration:.1f}s)")
    print(f"  Segments:        {len(segment_files)}")
    print(f"  Total gen time:  {total_gen_time/60:.1f} minutes")
    print(f"  Audio/gen ratio: {total_audio_duration/total_gen_time:.2f}x realtime")
    print(f"  Sample rate:     {sample_rate} Hz")
    print(f"  Segment files:   {SEGMENTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
