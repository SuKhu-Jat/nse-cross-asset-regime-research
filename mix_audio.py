"""Mix per-scene neural voiceover tracks with ambient underscore and normalize loudness."""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "assets" / "audio"
FFMPEG_BIN = r"C:\ffmpeg\bin\ffmpeg.exe"
SR = 48000


def read_wav_float(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 2
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == SR
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    return arr.reshape(-1, 2)


def main() -> None:
    timeline = json.loads((AUDIO_DIR / "timeline.json").read_text(encoding="utf-8"))
    total_dur = timeline["total_duration"]
    n_total = int(np.ceil(total_dur * SR))

    voice_mix = np.zeros((n_total, 2), dtype=np.float64)
    speech_env = np.zeros(n_total, dtype=np.float64)

    for sc in timeline["scenes"]:
        wav = read_wav_float(Path(sc["wav_path"]))
        start_s = sc["start_time"] + sc["pause_before"]
        idx = int(round(start_s * SR))
        end = min(idx + len(wav), n_total)
        voice_mix[idx:end] += wav[: end - idx]
        # Track active speech region for smooth ducking
        speech_env[idx:end] = 1.0

    # Smooth ducking envelope (~300ms O(N) moving average via cumsum)
    win = int(0.30 * SR)
    pad_env = np.pad(speech_env, (win // 2, win - 1 - win // 2), mode="edge")
    cs = np.cumsum(np.insert(pad_env, 0, 0.0))
    smooth_env = (cs[win:] - cs[:-win]) / float(win)
    # During speech: ambient at 0.30x; during pauses/outro: ambient rises gently to 0.85x
    duck_gain = 0.85 - 0.55 * np.clip(smooth_env, 0.0, 1.0)

    ambient = read_wav_float(AUDIO_DIR / "ambient_pad.wav")
    if len(ambient) < n_total:
        pad = np.zeros((n_total - len(ambient), 2), dtype=np.float64)
        ambient = np.vstack([ambient, pad])
    ambient = ambient[:n_total]

    # Normalize voice cleanly before mixing so voice dominates
    v_peak = np.max(np.abs(voice_mix)) + 1e-9
    voice_norm = (voice_mix / v_peak) * 0.82

    combined = voice_norm + ambient * duck_gain[:, None]
    combined = np.clip(combined, -0.98, 0.98)

    raw_mix_wav = AUDIO_DIR / "raw_master_audio.wav"
    pcm = (combined * 32767.0).astype(np.int16)
    with wave.open(str(raw_mix_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())

    final_wav = AUDIO_DIR / "master_audio.wav"
    subprocess.run(
        [
            FFMPEG_BIN, "-y", "-v", "error",
            "-i", str(raw_mix_wav),
            "-af", "loudnorm=I=-14:TP=-1.5:LRA=9",
            "-ar", "48000", "-ac", "2",
            str(final_wav),
        ],
        check=True,
    )
    print(f"Master audio created: {final_wav} ({total_dur:.2f}s)")


if __name__ == "__main__":
    main()
