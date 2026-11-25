"""Generate per-scene neural voiceover tracks and exact word-boundary timings via edge-tts."""

from __future__ import annotations

import asyncio
import json
import subprocess
import wave
from pathlib import Path
import edge_tts

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "assets" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
FFMPEG_BIN = r"C:\ffmpeg\bin\ffmpeg.exe"

VOICE = "en-US-AndrewMultilingualNeural"
RATE = "+6%"

SCENES = [
    {
        "id": "scene_01",
        "title": "Hook & Identity",
        "pause_before": 0.4,
        "pause_after": 0.6,
        "text": (
            "Most people look at markets and see noise. "
            "Sunil Parswal looks at noise and asks: what system sits underneath? "
            "An engineer at IIT BHU Varanasi graduating in twenty twenty-seven, "
            "Sunil turns messy uncertainty into structured models — "
            "ranking in the top six point four percent globally on LeetCode, "
            "with an eighteen seventy-two rating."
        ),
    },
    {
        "id": "scene_02",
        "title": "Why House of EdTech",
        "pause_before": 0.3,
        "pause_after": 0.6,
        "text": (
            "That is why the Junior Quantitative Research Analyst role at House of EdTech excites him. "
            "Beyond reaching five million learners through Profit Uni and AI for Techies, "
            "the Mumbai quant desk builds AI-led algorithmic trading across NSE, BSE, and MCX — "
            "rewarding first-principles thinking over conventional labels."
        ),
    },
    {
        "id": "scene_03",
        "title": "Built Start-to-Finish",
        "pause_before": 0.3,
        "pause_after": 0.6,
        "text": (
            "He ships end-to-end. At fintech startup NEOFIRST, Sunil engineered lending discovery "
            "and Surepass-Equifax APIs, cutting verification time by sixty percent. "
            "In his IIT BHU machine learning research, he paired Genetic Algorithm feature selection "
            "with Grad-CAM interpretability, optimizing for ninety-eight percent recall "
            "alongside Python and Power BI time-series forecasting."
        ),
    },
    {
        "id": "scene_04",
        "title": "NSE Cross-Asset Backtest",
        "pause_before": 0.3,
        "pause_after": 0.7,
        "text": (
            "To test that rigor on Indian markets, Sunil built a walk-forward study "
            "across NIFTY fifty, Gold ETF, and India VIX from twenty eighteen to twenty twenty-six. "
            "First came data hygiene: fixing an unadjusted split glitch, "
            "enforcing next-bar execution, and deducting brokerage and STT costs. "
            "A naive momentum crossover drops from a one point zero Sharpe in-sample "
            "to zero point five out-of-sample. "
            "An inverse-volatility and India VIX regime model holds "
            "a one point three-two out-of-sample Sharpe and fourteen point three percent CAGR."
        ),
    },
    {
        "id": "scene_05",
        "title": "Think On Your Feet",
        "pause_before": 0.3,
        "pause_after": 0.6,
        "text": (
            "One market question Sunil wants to explore next: weekly index options expiry on the NSE. "
            "Every expiry afternoon, dense retail flow meets dealer gamma hedging, "
            "driving sharp intraday volatility swings. "
            "Does realized intraday volatility systematically overshoot or mean-revert against India VIX "
            "in the final two hours before expiry — and after bid-ask spread and STT, "
            "does any edge actually survive out-of-sample?"
        ),
    },
    {
        "id": "scene_06",
        "title": "First 90 Days & Closing",
        "pause_before": 0.3,
        "pause_after": 1.6,
        "text": (
            "In his first ninety days, Sunil brings a tight loop: "
            "Days one to thirty, audit NSE, BSE, and MCX data for look-ahead and survivorship bias. "
            "Days thirty-one to sixty, stress-test volatility and mean-reversion hypotheses under realistic costs. "
            "Days sixty-one to ninety, ship clean Python and C++ modules. "
            "Sunil Parswal — ready to turn data into edge."
        ),
    },
]


def probe_duration(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


async def synth_scene(scene: dict) -> dict:
    mp3_path = AUDIO_DIR / f"{scene['id']}.mp3"
    wav_path = AUDIO_DIR / f"{scene['id']}.wav"
    meta_path = AUDIO_DIR / f"{scene['id']}.json"

    word_events = []
    for attempt in range(5):
        try:
            comm = edge_tts.Communicate(scene["text"], VOICE, rate=RATE, boundary="WordBoundary")
            word_events = []
            with open(mp3_path, "wb") as f:
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        f.write(chunk["data"])
                    elif chunk["type"] == "WordBoundary":
                        word_events.append({
                            "offset_s": chunk["offset"] / 1e7,
                            "duration_s": chunk["duration"] / 1e7,
                            "text": chunk["text"],
                        })
            break
        except Exception as exc:
            if attempt == 4:
                raise
            print(f"Retrying {scene['id']} after error ({exc})...")
            await asyncio.sleep(2.0 * (attempt + 1))

    subprocess.run(
        [FFMPEG_BIN, "-y", "-v", "error", "-i", str(mp3_path), "-ar", "48000", "-ac", "2", str(wav_path)],
        check=True,
    )
    audio_dur = probe_duration(wav_path)
    info = {
        "id": scene["id"],
        "title": scene["title"],
        "text": scene["text"],
        "wav_path": str(wav_path),
        "audio_duration": round(audio_dur, 3),
        "pause_before": scene["pause_before"],
        "pause_after": scene["pause_after"],
        "scene_duration": round(scene["pause_before"] + audio_dur + scene["pause_after"], 3),
        "words": word_events,
    }
    meta_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    await asyncio.sleep(1.0)
    return info


async def main() -> None:
    results = []
    cursor = 0.0
    for sc in SCENES:
        info = await synth_scene(sc)
        info["start_time"] = round(cursor, 3)
        cursor += info["scene_duration"]
        info["end_time"] = round(cursor, 3)
        results.append(info)
        print(f"{info['id']} ({info['title']}): speech={info['audio_duration']:.2f}s, scene={info['scene_duration']:.2f}s, words={len(info['words'])}")

    timeline = {
        "voice": VOICE,
        "rate": RATE,
        "total_duration": round(cursor, 3),
        "scenes": results,
    }
    (AUDIO_DIR / "timeline.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    print(f"TOTAL VIDEO DURATION: {cursor:.2f}s ({cursor/60:.2f} min)")


if __name__ == "__main__":
    asyncio.run(main())
