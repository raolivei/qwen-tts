#!/usr/bin/env python3
"""
Analyze TTS output WAV: duration, clipping, energy.
Used to tune params when we can't listen (duration/energy as quality proxies).
"""
import sys
import numpy as np
import soundfile as sf

def analyze(path: str) -> dict:
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    wav = wav.astype(np.float64)
    n = len(wav)
    duration_sec = n / sr
    max_abs = float(np.max(np.abs(wav)))
    rms = float(np.sqrt(np.mean(wav ** 2)))
    # "Speech" = frames where energy above 5% of max (rough voice activity)
    frame_len = int(sr * 0.02)  # 20 ms
    if frame_len < 1:
        frame_len = 1
    n_frames = n // frame_len
    energy = np.array([np.sqrt(np.mean(wav[i * frame_len:(i + 1) * frame_len] ** 2)) for i in range(n_frames)])
    thresh = max(energy) * 0.05
    speech_frames = np.sum(energy >= thresh)
    speech_duration_sec = speech_frames * frame_len / sr
    # Heuristic "quality" for a short sentence (~3–5 sec expected): no clip, not too long (repetition), not too short
    expected_min, expected_max = 2.0, 8.0
    duration_ok = expected_min <= duration_sec <= expected_max
    no_clip = max_abs <= 1.01
    has_energy = rms >= 0.005
    score = 0.0
    if no_clip:
        score += 1.0
    if has_energy:
        score += 1.0
    if duration_ok:
        score += 1.0
    if 2.5 <= speech_duration_sec <= 9.0:
        score += 0.5
    return {
        "path": path,
        "sr": sr,
        "duration_sec": round(duration_sec, 2),
        "speech_duration_sec": round(speech_duration_sec, 2),
        "max_abs": round(max_abs, 4),
        "rms": round(rms, 4),
        "duration_ok": duration_ok,
        "no_clip": no_clip,
        "has_energy": has_energy,
        "score": score,
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python analyze_tts_wav.py <file.wav>")
        sys.exit(1)
    r = analyze(path)
    print(f"Duration: {r['duration_sec']} s (speech ~{r['speech_duration_sec']} s)")
    print(f"Max |amplitude|: {r['max_abs']}  RMS: {r['rms']}")
    print(f"Duration OK (2–8 s): {r['duration_ok']}  No clip: {r['no_clip']}  Has energy: {r['has_energy']}")
    print(f"Score (0–3.5): {r['score']}")
