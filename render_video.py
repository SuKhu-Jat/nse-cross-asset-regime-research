"""
Render full 1920x1080 @ 30fps MP4 Video Resume for Sunil Parswal
Target Role: Junior Quantitative Research Analyst — House of EdTech
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
PLATES = ASSETS / "plates"
CHARTS = ASSETS / "charts"
AUDIO = ASSETS / "audio"
RESULTS = ROOT / "research" / "results"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FFMPEG_BIN = r"C:\ffmpeg\bin\ffmpeg.exe"
FINAL_MP4 = OUT_DIR / "Sunil_Parswal_House_of_EdTech_Quant_Video_Resume.mp4"

W, H = 1920, 1080
FPS = 30

# Palette
BG = (5, 8, 17)
CARD_BG = (11, 18, 34, 222)
CARD_ALT = (14, 24, 44, 232)
BORDER_MUTED = (36, 52, 82, 210)
BORDER_CYAN = (0, 212, 255, 235)
BORDER_EMERALD = (16, 185, 129, 230)
TEXT_PRIMARY = (248, 250, 252, 255)
TEXT_SECONDARY = (148, 163, 184, 255)
TEXT_MUTED = (100, 116, 139, 255)
CYAN = (0, 212, 255, 255)
EMERALD = (16, 185, 129, 255)
AMBER = (245, 158, 11, 255)
ROSE = (244, 63, 94, 255)

FONT_BOLD = r"C:\Windows\Fonts\segoeuib.ttf"
FONT_REG = r"C:\Windows\Fonts\segoeui.ttf"
FONT_LIGHT = r"C:\Windows\Fonts\segoeuil.ttf"
FONT_MONO = r"C:\Windows\Fonts\consola.ttf"
FONT_MONO_B = r"C:\Windows\Fonts\consolab.ttf"


def get_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


FONTS = {
    "h1": get_font(FONT_BOLD, 68),
    "h2": get_font(FONT_BOLD, 48),
    "h3": get_font(FONT_BOLD, 34),
    "card_title": get_font(FONT_BOLD, 27),
    "body": get_font(FONT_REG, 24),
    "body_sm": get_font(FONT_REG, 20),
    "kicker": get_font(FONT_BOLD, 17),
    "kpi": get_font(FONT_BOLD, 44),
    "kpi_sm": get_font(FONT_BOLD, 32),
    "mono": get_font(FONT_MONO, 19),
    "mono_b": get_font(FONT_MONO_B, 19),
    "sub": get_font(FONT_BOLD, 25),
    "hdr": get_font(FONT_BOLD, 16),
}


def ease_out_cubic(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 1.0 - (1.0 - x) ** 3


def ease_in_out(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 3 * x * x - 2 * x * x * x


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def load_plate(name: str) -> Image.Image:
    p = PLATES / name
    img = Image.open(p).convert("RGB")
    # Pre-scale slightly oversized (2080x1170) for smooth zero-recompute Ken Burns crop
    return img.resize((2080, 1170), Image.Resampling.LANCZOS)


def ken_burns(plate: Image.Image, progress: float, zoom_dir: int = 1, pan_x: float = 1.0) -> Image.Image:
    # Subtle 5% zoom + gentle drift over scene
    t = clamp01(progress)
    if zoom_dir < 0:
        t = 1.0 - t
    cw = int(2080 - 160 * t)
    ch = int(cw * 9 / 16)
    max_dx = 2080 - cw
    max_dy = 1170 - ch
    x0 = int(max_dx * (0.25 + 0.5 * t) if pan_x > 0 else max_dx * (0.75 - 0.5 * t))
    y0 = int(max_dy * 0.5)
    cropped = plate.crop((x0, y0, x0 + cw, y0 + ch))
    return cropped.resize((W, H), Image.Resampling.BILINEAR).convert("RGBA")


def draw_rounded_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int, int] = CARD_BG,
    outline: tuple[int, int, int, int] = BORDER_MUTED,
    radius: int = 18,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def alpha_col(col: tuple[int, int, int, int], a_mult: float) -> tuple[int, int, int, int]:
    return (col[0], col[1], col[2], int(col[3] * clamp01(a_mult)))


def make_squircle_portrait(size: int, radius: int) -> Image.Image:
    p = Image.open(PLATES / "sunil_portrait_studio.png").convert("RGBA")
    p = p.resize((size, size), Image.Resampling.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    p.putalpha(mask)
    return p


def paste_portrait(
    ov: Image.Image,
    d: ImageDraw.ImageDraw,
    portrait: Image.Image,
    x: int,
    y: int,
    alpha_mult: float,
    radius: int,
    border_col: tuple[int, int, int, int],
) -> None:
    if alpha_mult <= 0.02:
        return
    sz = portrait.size[0]
    p_mod = portrait.copy()
    if alpha_mult < 0.999:
        a_band = p_mod.getchannel("A").point(lambda v: int(v * clamp01(alpha_mult)))
        p_mod.putalpha(a_band)
    ov.paste(p_mod, (x, y), p_mod)
    d.rounded_rectangle([x, y, x + sz, y + sz], radius=radius, outline=alpha_col(border_col, alpha_mult), width=2)


def draw_header(draw: ImageDraw.ImageDraw, scene_idx: int, title: str, global_t: float, total_dur: float) -> None:
    # Top sleek bar
    draw.rectangle([0, 0, W, 72], fill=(5, 8, 17, 195))
    draw.line([(0, 72), (W, 72)], fill=(30, 46, 74, 180), width=1)
    draw.text((72, 26), "SUNIL PARSWAL  ·  IIT (BHU) VARANASI '27", font=FONTS["hdr"], fill=TEXT_SECONDARY)

    # Progress segments
    seg_x = 755
    seg_w = 56
    gap = 10
    for i in range(6):
        x0 = seg_x + i * (seg_w + gap)
        col = CYAN if i == scene_idx else (EMERALD if i < scene_idx else (42, 58, 86, 200))
        draw.rounded_rectangle([x0, 31, x0 + seg_w, 39], radius=4, fill=col)

    chapter_txt = f"0{scene_idx + 1} / 06  ·  {title.upper()}"
    draw.text((1170, 26), chapter_txt, font=FONTS["hdr"], fill=CYAN)

    right_txt = "JUNIOR QUANT RESEARCH ANALYST"
    rw = draw.textlength(right_txt, font=FONTS["hdr"])
    draw.text((W - 72 - rw, 26), right_txt, font=FONTS["hdr"], fill=TEXT_SECONDARY)


def build_caption_chunks(text: str, words: list[dict], max_words: int = 8) -> list[dict]:
    """Align punctuated script clauses onto WordBoundary timestamps so sentences never bleed together."""
    if not words:
        return []
    import re
    raw_tokens = text.split()
    # Clean tokens for matching word count; edge-tts sometimes splits hyphenated words
    phrases: list[list[str]] = []
    cur: list[str] = []
    for tok in raw_tokens:
        cur.append(tok)
        if len(cur) >= max_words or (len(cur) >= 3 and re.search(r"[.,;:—\?]$", tok)):
            phrases.append(cur)
            cur = []
    if cur:
        phrases.append(cur)

    # Map each phrase proportionally/sequentially to word timestamps
    total_raw_parts = sum(len(re.findall(r"[A-Za-z0-9']+", " ".join(p))) for p in phrases) or 1
    chunks = []
    w_idx = 0
    n_words = len(words)
    for pi, p_toks in enumerate(phrases):
        part_count = max(1, len(re.findall(r"[A-Za-z0-9']+", " ".join(p_toks))))
        est_span = max(1, round(part_count / total_raw_parts * n_words))
        start_i = min(w_idx, n_words - 1)
        end_i = n_words - 1 if pi == len(phrases) - 1 else min(n_words - 1, start_i + est_span - 1)
        w_idx = min(n_words - 1, end_i + 1)

        s_t = words[start_i]["offset_s"]
        e_t = words[end_i]["offset_s"] + words[end_i]["duration_s"] + 0.16
        # Assign individual word timings evenly across display tokens in this phrase
        sub_words = []
        n_disp = len(p_toks)
        span_t = max(0.2, (words[end_i]["offset_s"] + words[end_i]["duration_s"]) - s_t)
        for ti, tok in enumerate(p_toks):
            sub_words.append({
                "text": tok,
                "offset_s": s_t + (ti / n_disp) * span_t,
                "duration_s": span_t / n_disp,
            })
        chunks.append({"start": s_t, "end": e_t, "words": sub_words})
    return chunks


def draw_captions(draw: ImageDraw.ImageDraw, chunks: list[dict], speech_t: float) -> None:
    active = None
    for ch in chunks:
        if ch["start"] <= speech_t <= ch["end"]:
            active = ch
            break
    if not active:
        return

    words = active["words"]
    line_str = " ".join(w["text"] for w in words)
    tw = draw.textlength(line_str, font=FONTS["sub"])
    pad_x = 28
    box_w = min(W - 180, int(tw) + pad_x * 2)
    x0 = (W - box_w) // 2
    y0 = 980
    draw.rounded_rectangle([x0, y0, x0 + box_w, y0 + 54], radius=14, fill=(7, 12, 24, 228), outline=(42, 65, 98, 210), width=1)

    cx = (W - tw) / 2
    cy = y0 + 12
    space_w = draw.textlength(" ", font=FONTS["sub"])
    for w in words:
        is_now = w["offset_s"] <= speech_t <= (w["offset_s"] + w["duration_s"] + 0.12)
        col = CYAN if is_now else TEXT_PRIMARY
        draw.text((cx, cy), w["text"], font=FONTS["sub"], fill=col)
        cx += draw.textlength(w["text"], font=FONTS["sub"]) + space_w


# --- Preload Scene Assets ---
def prepare_backtest_curves() -> dict:
    df = pd.read_csv(RESULTS / "daily_backtest.csv", parse_dates=["Date"], index_col="Date")
    spec = json.loads((CHARTS / "backtest_axes_spec.json").read_text(encoding="utf-8"))
    n = len(df)

    def map_series(vals: np.ndarray, ax_spec: dict) -> list[tuple[float, float]]:
        x0, x1 = ax_spec["x0"], ax_spec["x1"]
        yt, yb = ax_spec["y_top"], ax_spec["y_bot"]
        ymin, ymax = ax_spec["ylim"]
        xs = np.linspace(x0, x1, n)
        ys = yb - ((vals - ymin) / (ymax - ymin)) * (yb - yt)
        ys = np.clip(ys, yt, yb)
        return list(zip(xs.tolist(), ys.tolist()))

    return {
        "base_img": Image.open(CHARTS / "backtest_axes_base.png").convert("RGBA"),
        "heatmap_img": Image.open(CHARTS / "sensitivity_heatmap.png").convert("RGBA"),
        "expiry_img": Image.open(CHARTS / "expiry_hypothesis_chart.png").convert("RGBA"),
        "spec": spec,
        "n": n,
        "dates": [str(d.date()) for d in df.index],
        "eq_bh": map_series(df["eq_bh"].values, spec["ax1"]),
        "eq_naive": map_series(df["eq_naive"].values, spec["ax1"]),
        "eq_regime": map_series(df["eq_regime"].values, spec["ax1"]),
        "dd_bh": map_series(df["dd_bh"].values, spec["ax2"]),
        "dd_naive": map_series(df["dd_naive"].values, spec["ax2"]),
        "dd_regime": map_series(df["dd_regime"].values, spec["ax2"]),
        "raw_eq_regime": df["eq_regime"].values,
        "raw_dd_regime": df["dd_regime"].values,
    }


def render_scene_01(t: float, dur: float, plates: dict) -> Image.Image:
    # 0 -> 7.2s: Hook Lattice; 7.2 -> dur: Varanasi Campus + Identity Card
    if t < 7.2:
        frame = ken_burns(plates["hook"], t / 7.2, zoom_dir=1, pan_x=1.0)
        if t > 6.5:
            p2 = ken_burns(plates["campus"], 0.0, zoom_dir=1, pan_x=-1.0)
            frame = Image.blend(frame, p2, clamp01((t - 6.5) / 0.7))
    else:
        frame = ken_burns(plates["campus"], (t - 7.2) / max(1.0, dur - 7.2), zoom_dir=1, pan_x=-1.0)

    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    if t < 6.9:
        a1 = ease_out_cubic(t / 0.8)
        fade_out = 1.0 - clamp01((t - 6.2) / 0.7)
        a = a1 * fade_out
        d.text((92, 235), "STAGE 01  ·  HOUSE OF EDTECH CREATIVE CHALLENGE", font=FONTS["kicker"], fill=alpha_col(CYAN, a))
        d.text((92, 290), "Most people look at markets", font=FONTS["h1"], fill=alpha_col(TEXT_PRIMARY, a))
        d.text((92, 372), "and see noise.", font=FONTS["h1"], fill=alpha_col(TEXT_SECONDARY, a))
        if t > 2.4:
            a2 = ease_out_cubic((t - 2.4) / 0.7) * fade_out
            d.text((92, 495), "Sunil asks:", font=FONTS["h2"], fill=alpha_col(TEXT_PRIMARY, a2))
            d.text((92, 562), "What system sits underneath?", font=FONTS["h1"], fill=alpha_col(CYAN, a2))
    else:
        a = ease_out_cubic((t - 7.2) / 0.9)
        y_off = int((1.0 - a) * 28)
        draw_rounded_card(d, (88, 170 + y_off, 1058, 860 + y_off), fill=(9, 15, 30, int(225 * a)), outline=alpha_col(BORDER_CYAN, a))
        d.text((134, 212 + y_off), "CANDIDATE PROFILE", font=FONTS["kicker"], fill=alpha_col(CYAN, a))
        d.text((134, 246 + y_off), "Sunil Parswal", font=FONTS["h1"], fill=alpha_col(TEXT_PRIMARY, a))
        d.text((134, 332 + y_off), "IDD Civil Engineering  ·  IIT (BHU), Varanasi", font=FONTS["body"], fill=alpha_col(TEXT_SECONDARY, a))
        d.text((134, 366 + y_off), "Class of 2027  ·  Khejroli, Jaipur", font=FONTS["body_sm"], fill=alpha_col(TEXT_MUTED, a))
        paste_portrait(ov, d, plates["portrait_sm"], 832, 198 + y_off, a, radius=24, border_col=CYAN)
        d.line([(134, 408 + y_off), (1012, 408 + y_off)], fill=alpha_col(BORDER_MUTED, a), width=2)

        # Animated KPI Tiles
        prog = ease_out_cubic((t - 8.5) / 3.0)
        rating_val = int(1872 * prog)
        kpis = [
            ("LEETCODE PEAK", f"{rating_val}", "Knight · Top 6.42% Globally", CYAN),
            ("BIWEEKLY CONTEST 190", f"Rank #{int(507 + (1 - prog) * 400)}", "Global Competitive Programming", EMERALD),
            ("ACADEMIC RIGOR", "8.14 CGPA", "99.00% Class XII · RBSE", AMBER),
        ]
        for idx, (lbl, big, sub, col) in enumerate(kpis):
            ka = ease_out_cubic((t - (8.4 + idx * 1.4)) / 0.8)
            ky = 434 + idx * 132 + y_off
            draw_rounded_card(d, (134, ky, 1012, ky + 110), fill=(15, 25, 48, int(235 * ka)), outline=alpha_col(col, 0.55 * ka), radius=14)
            d.text((164, ky + 15), lbl, font=FONTS["kicker"], fill=alpha_col(col, ka))
            d.text((164, ky + 43), big, font=FONTS["kpi_sm"], fill=alpha_col(TEXT_PRIMARY, ka))
            d.text((435, ky + 51), sub, font=FONTS["body_sm"], fill=alpha_col(TEXT_SECONDARY, ka))

    return Image.alpha_composite(frame, ov)


def render_scene_02(t: float, dur: float, plates: dict) -> Image.Image:
    frame = ken_burns(plates["mumbai"], t / max(1.0, dur), zoom_dir=1, pan_x=1.0)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    a = ease_out_cubic(t / 0.8)
    d.text((88, 128), "WHY HOUSE OF EDTECH  ·  MUMBAI QUANT DESK", font=FONTS["kicker"], fill=alpha_col(CYAN, a))
    d.text((88, 166), "Where Massive Scale Meets First-Principles Alpha", font=FONTS["h2"], fill=alpha_col(TEXT_PRIMARY, a))

    # Left Card: EdTech Scale
    a_left = ease_out_cubic((t - 1.2) / 0.9)
    draw_rounded_card(d, (88, 275, 760, 740), fill=(11, 18, 34, int(225 * a_left)), outline=alpha_col(BORDER_MUTED, a_left))
    d.text((126, 315), "PROVEN CONSUMER ENGINE", font=FONTS["kicker"], fill=alpha_col(TEXT_SECONDARY, a_left))
    d.text((126, 355), "5M+ Learners", font=FONTS["kpi"], fill=alpha_col(TEXT_PRIMARY, a_left))
    d.text((126, 412), "₹600 Cr+ ARR  ·  2,500+ Team", font=FONTS["body"], fill=alpha_col(EMERALD, a_left))
    d.line([(126, 468), (720, 468)], fill=alpha_col(BORDER_MUTED, a_left), width=2)
    brands = [
        ("Profit Uni", "Applied equity & trading strategies"),
        ("AI for Techies", "Hands-on Python, SQL & AI pipelines"),
        ("Dr. Finance & be10X", "Data-driven wealth & productivity"),
    ]
    for i, (bname, bdesc) in enumerate(brands):
        by = 500 + i * 72
        d.ellipse([128, by + 10, 140, by + 22], fill=alpha_col(CYAN, a_left))
        d.text((156, by), bname, font=FONTS["card_title"], fill=alpha_col(TEXT_PRIMARY, a_left))
        d.text((156, by + 33), bdesc, font=FONTS["body_sm"], fill=alpha_col(TEXT_SECONDARY, a_left))

    # Animated bridge arrow
    a_arr = ease_out_cubic((t - 4.0) / 1.0)
    if a_arr > 0.02:
        ax0, ax1, ay = 785, int(785 + 115 * a_arr), 505
        d.line([(ax0, ay), (ax1, ay)], fill=alpha_col(CYAN, a_arr), width=4)
        if a_arr > 0.7:
            d.polygon([(ax1, ay), (ax1 - 16, ay - 10), (ax1 - 16, ay + 10)], fill=alpha_col(CYAN, a_arr))

    # Right Card: Quant Research Desk
    a_right = ease_out_cubic((t - 4.5) / 1.0)
    draw_rounded_card(d, (925, 275, 1830, 740), fill=(11, 22, 42, int(232 * a_right)), outline=alpha_col(BORDER_CYAN, a_right), width=2)
    d.text((965, 315), "JUNIOR QUANTITATIVE RESEARCH ANALYST", font=FONTS["kicker"], fill=alpha_col(CYAN, a_right))
    d.text((965, 355), "AI-Led Algo Trading & Research", font=FONTS["h3"], fill=alpha_col(TEXT_PRIMARY, a_right))
    d.text((965, 408), "Mumbai Hybrid  ·  Full Research Lifecycle", font=FONTS["body_sm"], fill=alpha_col(TEXT_SECONDARY, a_right))

    pillars = [
        ("NSE / BSE EQUITIES", "Momentum, mean reversion, cross-asset signals", 6.5, CYAN),
        ("F&O DERIVATIVES", "Volatility surface, options flow, risk-adjusted sizing", 8.5, EMERALD),
        ("MCX COMMODITIES", "Macro correlations, regime filtering, slippage rigor", 10.5, AMBER),
    ]
    for i, (pt, pdsc, onset, pcol) in enumerate(pillars):
        pa = ease_out_cubic((t - onset) / 0.8) * a_right
        py = 468 + i * 82
        draw_rounded_card(d, (965, py, 1790, py + 68), fill=(17, 30, 56, int(230 * pa)), outline=alpha_col(pcol, 0.7 * pa), radius=12)
        d.text((990, py + 11), pt, font=FONTS["kicker"], fill=alpha_col(pcol, pa))
        d.text((990, py + 34), pdsc, font=FONTS["body_sm"], fill=alpha_col(TEXT_PRIMARY, pa))

    # Bottom thesis pill
    a_bot = ease_out_cubic((t - 13.0) / 1.0)
    draw_rounded_card(d, (88, 790, 1830, 882), fill=(13, 21, 38, int(220 * a_bot)), outline=alpha_col(BORDER_EMERALD, 0.7 * a_bot), radius=14)
    d.text((130, 822), "Core Alignment:  Independent research hypotheses  ·  Out-of-sample skepticism  ·  Mathematical & code velocity", font=FONTS["body"], fill=alpha_col(TEXT_PRIMARY, a_bot))

    return Image.alpha_composite(frame, ov)


def render_scene_03(t: float, dur: float, plates: dict) -> Image.Image:
    frame = ken_burns(plates["network"], t / max(1.0, dur), zoom_dir=-1, pan_x=1.0)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    a = ease_out_cubic(t / 0.7)
    d.text((88, 122), "TRACK RECORD  ·  BUILT END-TO-END", font=FONTS["kicker"], fill=alpha_col(CYAN, a))
    d.text((88, 158), "From Production Fintech APIs to Explainable ML", font=FONTS["h2"], fill=alpha_col(TEXT_PRIMARY, a))

    cards = [
        {
            "onset": 0.8,
            "x0": 88, "x1": 640,
            "tag": "FINTECH INTERNSHIP · MAY–JUN '26",
            "title": "NEOFIRST",
            "col": CYAN,
            "metrics": [("-60%", "KYC Verification (Surepass & Equifax)"),
                        ("-35%", "Catalog Search (Node.js Geo-APIs)"),
                        ("+45%", "Page Load Speed (Next.js SSR)")],
            "footer": "Low-latency financial discovery & KYC.",
        },
        {
            "onset": 8.0,
            "x0": 684, "x1": 1236,
            "tag": "IIT (BHU) RESEARCH · EXPLAINABLE AI",
            "title": "GA-SVM + Grad-CAM",
            "col": EMERALD,
            "metrics": [("98.0%", "Recall (Cost-Sensitive Objective)"),
                        ("94.7%", "Test Accuracy · 93% Precision"),
                        ("ResNet50", "+ Genetic Algorithm Selection")],
            "footer": "Recall over vanity accuracy + Grad-CAM.",
        },
        {
            "onset": 15.5,
            "x0": 1280, "x1": 1832,
            "tag": "TIME-SERIES & BI ANALYTICS",
            "title": "Demand Forecasting",
            "col": AMBER,
            "metrics": [("RF vs OLS", "Dynamic Menu & Price Modeling"),
                        ("MAE / RMSE", "+ R² Error Benchmarking"),
                        ("Power BI", "Python + SQL Sales Dashboards")],
            "footer": "Noisy series turned into decisions.",
        },
    ]

    for c in cards:
        ca = ease_out_cubic((t - c["onset"]) / 0.9)
        is_focused = c["onset"] <= t < (c["onset"] + 8.0)
        border = c["col"] if is_focused else BORDER_MUTED
        dy = int((1.0 - ca) * 32)
        draw_rounded_card(d, (c["x0"], 255 + dy, c["x1"], 885 + dy), fill=(11, 18, 34, int(232 * ca)), outline=alpha_col(border, ca), width=2 if is_focused else 1)
        d.text((c["x0"] + 32, 290 + dy), c["tag"], font=FONTS["kicker"], fill=alpha_col(c["col"], ca))
        d.text((c["x0"] + 32, 326 + dy), c["title"], font=FONTS["h3"], fill=alpha_col(TEXT_PRIMARY, ca))
        d.line([(c["x0"] + 32, 384 + dy), (c["x1"] - 32, 384 + dy)], fill=alpha_col(BORDER_MUTED, ca), width=2)

        for mi, (mval, mdesc) in enumerate(c["metrics"]):
            my = 415 + mi * 126 + dy
            draw_rounded_card(d, (c["x0"] + 32, my, c["x1"] - 32, my + 106), fill=(16, 28, 50, int(230 * ca)), outline=alpha_col(BORDER_MUTED, ca), radius=12)
            d.text((c["x0"] + 54, my + 14), mval, font=FONTS["kpi_sm"], fill=alpha_col(c["col"], ca))
            d.text((c["x0"] + 54, my + 58), mdesc, font=FONTS["body_sm"], fill=alpha_col(TEXT_PRIMARY, ca))

        d.text((c["x0"] + 32, 824 + dy), c["footer"], font=FONTS["body_sm"], fill=alpha_col(TEXT_SECONDARY, ca))

    return Image.alpha_composite(frame, ov)


def render_scene_04(t: float, dur: float, bt: dict) -> Image.Image:
    # Dark quant IDE + live animated Matplotlib Tear-Sheet!
    base = Image.new("RGBA", (W, H), (5, 8, 17, 255))
    d = ImageDraw.Draw(base)

    a = ease_out_cubic(t / 0.6)
    d.text((72, 102), "ORIGINAL NSE QUANT RESEARCH  ·  research/nifty_regime_research.py", font=FONTS["kicker"], fill=alpha_col(CYAN, a))
    d.text((72, 134), "Walk-Forward Cross-Asset Study: NIFTY 50 + Gold ETF + India VIX (2018–2026)", font=FONTS["h3"], fill=alpha_col(TEXT_PRIMARY, a))

    # --- LEFT COLUMN (x=72..708, y=195..935) ---
    # Phase A (0..18s): Code & Hygiene Controls; Phase B (18..dur): 5x5 Sensitivity Heatmap + Metrics Table
    if t < 18.2:
        ca = a * (1.0 - clamp01((t - 17.4) / 0.8))
        draw_rounded_card(d, (72, 198, 712, 922), fill=(10, 16, 30, int(245 * ca)), outline=alpha_col(BORDER_CYAN, 0.7 * ca), radius=16)
        d.text((100, 224), "RESEARCH HYGIENE & ANTI-BIAS PIPELINE", font=FONTS["kicker"], fill=alpha_col(CYAN, ca))

        code_blocks = [
            (2.0, "1. Data Hygiene (Bad-Tick Audit)", [
                "# Caught Yahoo unadjusted 100:1 split",
                "# glitch in GOLDBEES.NS (Dec 19–20, 2019)",
                "glitch = (g.index >= '2019-12-19') & (g < 2.0)",
                "g.loc[glitch] *= 100.0",
            ], AMBER),
            (6.5, "2. Zero Look-Ahead Execution", [
                "# Strictly lag signal(t) -> execute(t+1)",
                "w_eq = (rp_eq * mom_eq * vix_tilt).shift(1)",
                "w_au = (rp_au * mom_au * (2 - vix_tilt)).shift(1)",
            ], CYAN),
            (10.5, "3. Realistic NSE Cost & Turnover Drag", [
                "# 5 bps one-way (STT + brokerage + slippage)",
                "# + 12% portfolio rebalance deadband",
                "turn = w_eq_b.diff().abs() + w_au_b.diff().abs()",
                "net_ret = gross_ret - turn * (5.0 / 10000.0)",
            ], EMERALD),
        ]
        cy = 272
        for onset, heading, lines, hcol in code_blocks:
            ba = ease_out_cubic((t - onset) / 0.8) * ca
            active_now = onset <= t < onset + 5.0
            bh = 52 + len(lines) * 28 + 20
            draw_rounded_card(d, (96, cy, 688, cy + bh), fill=(15, 23, 42, int(240 * ba)), outline=alpha_col(hcol if active_now else BORDER_MUTED, ba), radius=12)
            d.text((116, cy + 14), heading, font=FONTS["kicker"], fill=alpha_col(hcol, ba))
            for li, line in enumerate(lines):
                col_l = TEXT_MUTED if line.startswith("#") else TEXT_PRIMARY
                d.text((116, cy + 44 + li * 28), line, font=FONTS["mono"], fill=alpha_col(col_l, ba))
            cy += bh + 22
    else:
        ha = ease_out_cubic((t - 18.0) / 0.9)
        draw_rounded_card(d, (72, 198, 712, 922), fill=(11, 18, 34, int(245 * ha)), outline=alpha_col(BORDER_EMERALD, 0.8 * ha), radius=16)
        hm = bt["heatmap_img"]
        base.paste(hm, (88, 214), hm)
        d = ImageDraw.Draw(base)

        # Out-of-Sample Comparison Table below heatmap
        d.text((100, 622), "OUT-OF-SAMPLE SUMMARY (2023–2026 · 960 DAYS)", font=FONTS["kicker"], fill=alpha_col(CYAN, ha))
        rows = [
            ("Strategy", "Sharpe", "CAGR", "Max DD", "p-val", TEXT_SECONDARY),
            ("NIFTY 50 B&H", "0.78", "9.63%", "-15.8%", "0.064", TEXT_SECONDARY),
            ("Naive 20/60 EMA", "0.50", "4.67%", "-13.0%", "0.165", AMBER),
            ("Regime Model", "1.32", "14.29%", "-12.0%", "0.005", EMERALD),
        ]
        cols_x = [104, 322, 416, 514, 614]
        for ri, row in enumerate(rows):
            ry = 660 + ri * 54
            if ri == 3:
                draw_rounded_card(d, (92, ry - 8, 692, ry + 38), fill=(12, 48, 42, 255), outline=alpha_col(EMERALD, ha), radius=8, width=2)
            elif ri > 0:
                d.line([(96, ry - 6), (688, ry - 6)], fill=(30, 46, 74, 255), width=1)
            fnt = FONTS["kicker"] if ri == 0 else (FONTS["mono_b"] if ri == 3 else FONTS["body_sm"])
            for ci in range(5):
                txt_col = TEXT_PRIMARY if (ri == 3 and ci == 0) else row[5]
                d.text((cols_x[ci], ry + 4), row[ci], font=fnt, fill=alpha_col(txt_col, ha))

        # Statistical note
        d.text((100, 882), "All 25 EMA pairs achieve 1.29–1.43 OOS Sharpe (stable plateau)", font=FONTS["body_sm"], fill=alpha_col(TEXT_SECONDARY, ha))

    # --- RIGHT COLUMN: LIVE ANIMATED EQUITY & DRAWDOWN CHART (x=740..1848, y=198..922) ---
    chart_bg = bt["base_img"].copy()
    cd = ImageDraw.Draw(chart_bg)
    # Animate reveal from t=3.0s to t=29.0s across full 2018-2026 history
    curve_prog = ease_in_out((t - 2.5) / 25.0)
    k = max(2, int(curve_prog * bt["n"]))

    # Draw Equity curves
    cd.line(bt["eq_bh"][:k], fill=(100, 116, 139, 210), width=2)
    cd.line(bt["eq_naive"][:k], fill=(245, 158, 11, 225), width=2)
    cd.line(bt["eq_regime"][:k], fill=(0, 212, 255, 255), width=4)

    # Draw Drawdown curves
    cd.line(bt["dd_bh"][:k], fill=(100, 116, 139, 175), width=2)
    cd.line(bt["dd_naive"][:k], fill=(245, 158, 11, 180), width=2)
    cd.line(bt["dd_regime"][:k], fill=(16, 185, 129, 245), width=3)

    # Glowing leading dot on Regime Equity Curve
    lx, ly = bt["eq_regime"][k - 1]
    cd.ellipse([lx - 9, ly - 9, lx + 9, ly + 9], fill=(0, 212, 255, 95))
    cd.ellipse([lx - 5, ly - 5, lx + 5, ly + 5], fill=(248, 250, 252, 255))

    # Legend inside chart top-left
    draw_rounded_card(cd, (94, 72, 470, 172), fill=(11, 18, 34, 215), outline=BORDER_MUTED, radius=10, width=1)
    leg_items = [
        ("Cross-Asset Regime (NIFTY+Gold+VIX)", (0, 212, 255, 255)),
        ("Naive 20/60 EMA Crossover", (245, 158, 11, 255)),
        ("NIFTY 50 Buy & Hold", (148, 163, 184, 255)),
    ]
    for li, (lname, lcol) in enumerate(leg_items):
        ly_i = 84 + li * 28
        cd.line([(112, ly_i + 10), (144, ly_i + 10)], fill=lcol, width=4 if li == 0 else 3)
        cd.text((156, ly_i), lname, font=FONTS["hdr"], fill=TEXT_PRIMARY if li == 0 else TEXT_SECONDARY)

    # Paste animated chart onto frame cleanly above cursor readout
    draw_rounded_card(d, (736, 198, 1848, 926), fill=(10, 16, 30, 255), outline=BORDER_MUTED, radius=16)
    base.paste(chart_bg, (742, 204), chart_bg)
    d = ImageDraw.Draw(base)

    # Live date & equity cursor pill at bottom of chart card
    cur_date = bt["dates"][k - 1]
    cur_eq = bt["raw_eq_regime"][k - 1]
    cur_dd = bt["raw_dd_regime"][k - 1] * 100.0
    d.text((766, 892), f"Timeline: {cur_date}   ·   Regime Equity: {cur_eq:.2f}x   ·   Current DD: {cur_dd:.1f}%", font=FONTS["mono_b"], fill=CYAN)

    return base


def render_scene_05(t: float, dur: float, plates: dict, bt: dict) -> Image.Image:
    frame = ken_burns(plates["vol"], t / max(1.0, dur), zoom_dir=1, pan_x=-1.0)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    a = ease_out_cubic(t / 0.7)
    d.text((88, 122), "THINK ON YOUR FEET  ·  ~30 SECONDS", font=FONTS["kicker"], fill=alpha_col(AMBER, a))
    d.text((88, 158), "NSE Weekly Index Options Expiry Microstructure", font=FONTS["h2"], fill=alpha_col(TEXT_PRIMARY, a))

    # Left structured hypothesis cards
    steps = [
        ("01  FLOW CATALYST", "Expiry afternoon retail option volume collides with dealer gamma hedging, driving sharp intraday vol spikes.", 1.5, AMBER),
        ("02  TESTABLE QUESTION", "Does realized intraday vol systematically overshoot or mean-revert vs India VIX in the final 2 hours (13:30–15:30)?", 7.5, CYAN),
        ("03  POST-COST REALITY CHECK", "After bid-ask spread, slippage, and STT, does the residual spread survive OOS — or is it pure tail-gamma risk?", 15.0, EMERALD),
    ]
    for idx, (stitle, sdesc, onset, scol) in enumerate(steps):
        sa = ease_out_cubic((t - onset) / 0.8)
        sy = 265 + idx * 205
        draw_rounded_card(d, (88, sy, 920, sy + 178), fill=(11, 18, 34, int(232 * sa)), outline=alpha_col(scol, 0.8 * sa), radius=16)
        d.text((120, sy + 22), stitle, font=FONTS["kicker"], fill=alpha_col(scol, sa))
        # Wrap text cleanly across 2 lines
        words = sdesc.split()
        line1, line2 = [], []
        for w in words:
            cand = " ".join(line1 + [w])
            if d.textlength(cand, font=FONTS["body"]) < 730 and not line2:
                line1.append(w)
            else:
                line2.append(w)
        d.text((120, sy + 60), " ".join(line1), font=FONTS["body"], fill=alpha_col(TEXT_PRIMARY, sa))
        if line2:
            d.text((120, sy + 98), " ".join(line2), font=FONTS["body"], fill=alpha_col(TEXT_PRIMARY, sa))

    # Right panel: Conceptual Intraday Expiry Vol Chart (progressive crop wipe reveal!)
    ca = ease_out_cubic((t - 4.0) / 1.0)
    if ca > 0.02:
        draw_rounded_card(d, (960, 265, 1848, 860), fill=(10, 16, 30, int(242 * ca)), outline=alpha_col(BORDER_CYAN, 0.7 * ca), radius=16)
        exp_img = bt["expiry_img"]
        rev = ease_in_out((t - 5.0) / 13.0)
        rev_w = max(1, int(860 * rev))
        cropped_exp = exp_img.crop((0, 0, rev_w, 480))
        ov.paste(cropped_exp, (974, 285), cropped_exp)
        d = ImageDraw.Draw(ov)
        d.text((995, 792), "CONCEPTUAL RESEARCH HYPOTHESIS (TO BE TESTED ON TICK DATA)", font=FONTS["kicker"], fill=alpha_col(TEXT_SECONDARY, ca))

    return Image.alpha_composite(frame, ov)


def render_scene_06(t: float, dur: float, plates: dict) -> Image.Image:
    frame = ken_burns(plates["horizon"], t / max(1.0, dur), zoom_dir=1, pan_x=1.0)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    # Phase A (0..18.5s): First 90 Days Roadmap; Phase B (18.5..dur): Closing Brand Slate
    if t < 18.8:
        a = ease_out_cubic(t / 0.7) * (1.0 - clamp01((t - 18.0) / 0.8))
        d.text((88, 122), "EXECUTION PLAN  ·  FIRST 90 DAYS AT HOUSE OF EDTECH", font=FONTS["kicker"], fill=alpha_col(CYAN, a))
        d.text((88, 158), "From Data Hygiene to Live Quantitative Modules", font=FONTS["h2"], fill=alpha_col(TEXT_PRIMARY, a))

        roadmap = [
            ("DAYS 1 – 30", "Audit & Foundations", [
                "Inspect NSE, BSE & MCX data",
                "Eliminate look-ahead & split errors",
                "Benchmark baseline desk strategies",
            ], 1.2, CYAN),
            ("DAYS 31 – 60", "Hypotheses & Stress Test", [
                "Build volatility & mean-reversion models",
                "Enforce walk-forward OOS validation",
                "Run slippage & parameter plateaus",
            ], 6.0, EMERALD),
            ("DAYS 61 – 90", "Productionize & Ship", [
                "Convert research to Python / C++ modules",
                "Automate reproducible tear-sheets",
                "Support live deployment & monitoring",
            ], 11.2, AMBER),
        ]
        for idx, (day_lbl, rtitle, bullets, onset, rcol) in enumerate(roadmap):
            ra = ease_out_cubic((t - onset) / 0.8) * (1.0 - clamp01((t - 18.0) / 0.8))
            rx0 = 88 + idx * 592
            rx1 = rx0 + 556
            draw_rounded_card(d, (rx0, 275, rx1, 795), fill=(11, 18, 34, int(232 * ra)), outline=alpha_col(rcol, 0.85 * ra), radius=18, width=2)
            d.text((rx0 + 34, 318), day_lbl, font=FONTS["kicker"], fill=alpha_col(rcol, ra))
            d.text((rx0 + 34, 354), rtitle, font=FONTS["h3"], fill=alpha_col(TEXT_PRIMARY, ra))
            d.line([(rx0 + 34, 412), (rx1 - 34, 412)], fill=alpha_col(BORDER_MUTED, ra), width=2)
            for bi, b in enumerate(bullets):
                by = 450 + bi * 92
                d.ellipse([rx0 + 36, by + 10, rx0 + 48, by + 22], fill=alpha_col(rcol, ra))
                d.text((rx0 + 66, by), b, font=FONTS["body"], fill=alpha_col(TEXT_PRIMARY, ra))
    else:
        ca = ease_out_cubic((t - 18.4) / 0.9)
        draw_rounded_card(d, (160, 240, 1760, 810), fill=(8, 13, 26, int(222 * ca)), outline=alpha_col(BORDER_CYAN, 0.75 * ca), radius=24, width=2)
        d.text((230, 305), "HOUSE OF EDTECH  ·  STAGE 1 SUBMISSION", font=FONTS["kicker"], fill=alpha_col(CYAN, ca))
        d.text((230, 355), "Sunil Parswal", font=FONTS["h1"], fill=alpha_col(TEXT_PRIMARY, ca))
        d.text((230, 445), "Applying for Junior Quantitative Research Analyst", font=FONTS["h3"], fill=alpha_col(EMERALD, ca))
        d.line([(230, 515), (1285, 515)], fill=alpha_col(BORDER_MUTED, ca), width=2)
        d.text((230, 555), "IIT (BHU) Varanasi  ·  IDD '27  ·  LeetCode Knight (1872)  ·  Fintech & ML Builder", font=FONTS["body"], fill=alpha_col(TEXT_SECONDARY, ca))
        d.text((230, 625), "sunilparswal.work@gmail.com   ·   GitHub: SuKhu-Jat/nse-cross-asset-regime-research", font=FONTS["mono_b"], fill=alpha_col(CYAN, ca))
        d.text((230, 705), "\"Ready to turn raw market noise into disciplined edge.\"", font=FONTS["h3"], fill=alpha_col(TEXT_PRIMARY, ca))
        paste_portrait(ov, d, plates["portrait_lg"], 1345, 305, ca, radius=28, border_col=CYAN)

    return Image.alpha_composite(frame, ov)


def main() -> None:
    timeline = json.loads((AUDIO / "timeline.json").read_text(encoding="utf-8"))
    total_dur = timeline["total_duration"]
    scenes = timeline["scenes"]

    plates = {
        "hook": load_plate("hook_lattice.jpg"),
        "campus": load_plate("campus_varanasi.jpg"),
        "mumbai": load_plate("mumbai_quant_skyline.png"),
        "network": load_plate("ai_pipeline_network.png"),
        "vol": load_plate("vol_surface_3d.png"),
        "horizon": load_plate("horizon_beam.jpg"),
        "portrait_sm": make_squircle_portrait(188, radius=24),
        "portrait_lg": make_squircle_portrait(350, radius=28),
    }
    bt = prepare_backtest_curves()

    for sc in scenes:
        sc["chunks"] = build_caption_chunks(sc["text"], sc.get("words", []))

    renderers = {
        "scene_01": lambda lt, sd: render_scene_01(lt, sd, plates),
        "scene_02": lambda lt, sd: render_scene_02(lt, sd, plates),
        "scene_03": lambda lt, sd: render_scene_03(lt, sd, plates),
        "scene_04": lambda lt, sd: render_scene_04(lt, sd, bt),
        "scene_05": lambda lt, sd: render_scene_05(lt, sd, plates, bt),
        "scene_06": lambda lt, sd: render_scene_06(lt, sd, plates),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="Render representative preview PNGs for each scene")
    args = parser.parse_args()

    if args.preview:
        preview_dir = OUT_DIR / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        samples = [
            (0, 14.0, "01_identity"),
            (1, 14.5, "02_why_hoe"),
            (2, 18.0, "03_projects"),
            (3, 12.0, "04a_backtest_code"),
            (3, 28.0, "04b_backtest_heatmap"),
            (4, 19.0, "05_think_on_feet"),
            (5, 14.0, "06a_90days"),
            (5, 22.0, "06b_closing"),
        ]
        for s_idx, lt, tag in samples:
            sc = scenes[s_idx]
            frame = renderers[sc["id"]](lt, sc["scene_duration"])
            d = ImageDraw.Draw(frame)
            draw_header(d, s_idx, sc["title"], sc["start_time"] + lt, total_dur)
            draw_captions(d, sc["chunks"], lt - sc["pause_before"])
            out_p = preview_dir / f"{tag}.png"
            frame.convert("RGB").save(out_p)
            print(f"Preview saved: {out_p}")
        return

    cmd = [
        FFMPEG_BIN, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-i", str(AUDIO / "master_audio.wav"),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
        str(FINAL_MP4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    total_frames = int(round(total_dur * FPS))
    fade_frames = 10  # 0.33s smooth cross-dissolve at scene boundaries

    for f_idx in range(total_frames):
        gt = f_idx / FPS
        # Find active scene
        s_idx = len(scenes) - 1
        for i, sc in enumerate(scenes):
            if sc["start_time"] <= gt < sc["end_time"]:
                s_idx = i
                break
        sc = scenes[s_idx]
        lt = max(0.0, gt - sc["start_time"])
        frame = renderers[sc["id"]](lt, sc["scene_duration"])

        # Smooth cross-dissolve from previous scene at first 10 frames of new scene
        if s_idx > 0 and lt < (fade_frames / FPS):
            prev_sc = scenes[s_idx - 1]
            prev_lt = prev_sc["scene_duration"] - 0.05
            prev_frame = renderers[prev_sc["id"]](prev_lt, prev_sc["scene_duration"])
            alpha = clamp01(lt / (fade_frames / FPS))
            frame = Image.blend(prev_frame, frame, alpha)

        # Overlay header & synchronized captions
        d = ImageDraw.Draw(frame)
        draw_header(d, s_idx, sc["title"], gt, total_dur)
        speech_t = lt - sc["pause_before"]
        if 0.0 <= speech_t <= sc["audio_duration"] + 0.3:
            draw_captions(d, sc["chunks"], speech_t)

        # Fade in from black (0.5s) and fade out to black (0.8s)
        if gt < 0.5:
            black = Image.new("RGBA", (W, H), (0, 0, 0, 255))
            frame = Image.blend(black, frame, gt / 0.5)
        elif gt > total_dur - 0.8:
            black = Image.new("RGBA", (W, H), (0, 0, 0, 255))
            frame = Image.blend(frame, black, clamp01((gt - (total_dur - 0.8)) / 0.8))

        proc.stdin.write(frame.convert("RGB").tobytes())
        if f_idx % 300 == 0:
            print(f"Rendered {f_idx}/{total_frames} frames ({100.0 * f_idx / total_frames:.1f}%)")

    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"FFmpeg exited with code {ret}")
    print(f"SUCCESS: Rendered final video -> {FINAL_MP4} ({FINAL_MP4.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
