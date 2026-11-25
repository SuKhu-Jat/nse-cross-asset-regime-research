"""Synthesize a clean, subtle minimalist ambient underscore for the video resume."""

from __future__ import annotations

import json
import wave
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt


SR = 48000
ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "assets" / "audio"


def lowpass(sig: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
    b, a = butter(order, cutoff / (0.5 * SR), btype="low")
    return filtfilt(b, a, sig)


def pad_chord(freqs: list[float], duration: float, t_offset: float = 0.0) -> np.ndarray:
    n = int(duration * SR)
    t = np.linspace(0.0, duration, n, endpoint=False)
    env = np.sin(np.pi * np.clip(t / duration, 0, 1)) ** 0.55
    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)
    for idx, f in enumerate(freqs):
        detune = 0.0028 * (1 if idx % 2 == 0 else -1)
        ph1 = 2 * np.pi * (f * (1.0 + detune)) * (t + t_offset)
        ph2 = 2 * np.pi * (f * (1.0 - detune)) * (t + t_offset)
        shimmer = 0.18 * np.sin(2 * np.pi * (f * 2.0) * (t + t_offset))
        left += 0.32 * np.sin(ph1) + 0.14 * np.sin(ph2) + shimmer * 0.5
        right += 0.32 * np.sin(ph2) + 0.14 * np.sin(ph1) + shimmer * 0.5
    left = lowpass(left * env, 1150.0)
    right = lowpass(right * env, 1150.0)
    return np.stack([left, right], axis=1)


def soft_pulse(n_total: int, duration: float, bpm: float = 96.0) -> np.ndarray:
    n = n_total
    beat_period = 60.0 / bpm
    out = np.zeros((n, 2), dtype=np.float64)
    t_beat = 0.0
    i = 0
    while t_beat < duration - 0.15:
        start = int(t_beat * SR)
        click_len = int(0.11 * SR)
        if start + click_len < n:
            tc = np.linspace(0, 0.11, click_len, endpoint=False)
            freq = 110.0 if i % 4 == 0 else 165.0
            amp = 0.045 if i % 4 == 0 else 0.022
            hit = amp * np.sin(2 * np.pi * freq * tc) * np.exp(-32.0 * tc)
            pan = 0.45 if i % 2 == 0 else 0.55
            out[start : start + click_len, 0] += hit * pan
            out[start : start + click_len, 1] += hit * (1.0 - pan)
        t_beat += beat_period
        i += 1
    return out


def build_ambient(total_duration: float, transition_times: list[float], out_wav: Path) -> None:
    # Warm contemplative progression: Dm9 -> Bbmaj7 -> Fmaj7 -> C6
    chords = [
        [146.83, 220.00, 261.63, 329.63],  # Dm9
        [116.54, 174.61, 220.00, 293.66],  # Bbmaj7
        [174.61, 220.00, 261.63, 329.63],  # Fmaj7
        [130.81, 196.00, 220.00, 261.63],  # C6
    ]
    n_total = int(np.ceil(total_duration * SR))
    mix = np.zeros((n_total, 2), dtype=np.float64)

    bar_len = 8.0
    crossfade = 2.0
    cur_t = 0.0
    idx = 0
    while cur_t < total_duration:
        chord = chords[idx % len(chords)]
        seg = pad_chord(chord, bar_len + crossfade, t_offset=cur_t)
        start = int(cur_t * SR)
        end = min(start + len(seg), n_total)
        mix[start:end] += seg[: end - start]
        cur_t += bar_len
        idx += 1

    # Subtle rhythmic momentum pulse
    mix += soft_pulse(n_total, total_duration, bpm=92.0)

    # Subtle clean whoosh chime at scene transitions
    for tt in transition_times:
        if 0.5 < tt < total_duration - 1.0:
            dur = 0.55
            ns = int(dur * SR)
            ts = np.linspace(0, dur, ns, endpoint=False)
            chime = 0.035 * np.sin(2 * np.pi * (440.0 + 220.0 * ts) * ts) * np.exp(-6.5 * ts)
            s_idx = int((tt - 0.12) * SR)
            if s_idx >= 0 and s_idx + ns < n_total:
                mix[s_idx : s_idx + ns, 0] += chime
                mix[s_idx : s_idx + ns, 1] += chime

    # Normalize gently to quiet background level (-23 dBFS peak ~ 0.07)
    peak = np.max(np.abs(mix)) + 1e-9
    mix = (mix / peak) * 0.11

    # Smooth fade in / fade out
    fi = int(1.5 * SR)
    fo = int(3.0 * SR)
    mix[:fi] *= np.linspace(0.0, 1.0, fi)[:, None]
    mix[-fo:] *= np.linspace(1.0, 0.0, fo)[:, None]

    pcm = np.clip(mix * 32767.0, -32767, 32767).astype(np.int16)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    print(f"Saved ambient underscore: {out_wav} ({total_duration:.2f}s)")


if __name__ == "__main__":
    timeline_path = AUDIO_DIR / "timeline.json"
    meta = json.loads(timeline_path.read_text(encoding="utf-8"))
    transitions = [s["start_time"] for s in meta["scenes"][1:]]
    build_ambient(meta["total_duration"], transitions, AUDIO_DIR / "ambient_pad.wav")
