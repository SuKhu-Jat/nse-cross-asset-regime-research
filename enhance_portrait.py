"""Enhance candidate portrait with studio matting, spill suppression, tone balance, and dark editorial backdrop."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent
RAW_PHOTO = Path(r"C:\Users\sunil\.gemini\antigravity\brain\a0c85d2a-5b59-446b-9a6e-b0c16bf71b0b\.user_uploaded\media_1795600099837_2c38e9f6.jpg")
OUT_PORTRAIT = ROOT / "assets" / "plates" / "sunil_portrait_studio.png"
OUT_ARTIFACT = Path(r"C:\Users\sunil\.gemini\antigravity\brain\a0c85d2a-5b59-446b-9a6e-b0c16bf71b0b\sunil_portrait_studio.png")


def build_clean_alpha(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(bgr.astype(np.int16))

    # Blue curtain has noticeably higher B than R and G, and blue hue in [98..140]
    is_blue_bg = (
        (hsv[:, :, 0] >= 95)
        & (hsv[:, :, 0] <= 142)
        & (hsv[:, :, 1] >= 42)
        & ((b - r) >= 22)
        & ((b - g) >= 18)
    )

    # Trimap for GrabCut
    mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)
    mask[is_blue_bg] = cv2.GC_BGD

    # Definite background strips on outer top/left/right margins where blue is dominant
    mask[:35, :] = np.where(is_blue_bg[:35, :], cv2.GC_BGD, cv2.GC_PR_BGD)
    # Definite foreground in central face & upper chest region (non-blue pixels only)
    cy0, cy1 = int(0.18 * h), int(0.85 * h)
    cx0, cx1 = int(0.32 * w), int(0.68 * w)
    center_non_blue = ~is_blue_bg[cy0:cy1, cx0:cx1]
    mask[cy0:cy1, cx0:cx1][center_non_blue] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, None, bgd_model, fgd_model, 7, cv2.GC_INIT_WITH_MASK)

    fg_bin = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)

    # Keep only largest connected foreground component (eliminates stray curtain islands)
    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats((fg_bin > 0.5).astype(np.uint8))
    if num_lbl > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg_bin = (labels == largest).astype(np.float32)

    # Gentle morphological clean + edge feather
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Slightly erode 1px on edge before blurring to eliminate blue fringe
    fg_erode = cv2.erode(fg_bin, kernel, iterations=1)
    alpha = cv2.GaussianBlur(fg_erode, (5, 5), sigmaX=1.4)
    # Keep solid core opaque
    core = cv2.erode(fg_bin, kernel, iterations=3)
    alpha = np.maximum(alpha, core)
    return np.clip(alpha, 0.0, 1.0)


def studio_grade(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    # 1. Blue spill suppression on boundary pixels (0 < alpha < 0.98)
    out = bgr.astype(np.float32).copy()
    fringe = (alpha > 0.02) & (alpha < 0.98)
    b_ch, g_ch, r_ch = out[:, :, 0], out[:, :, 1], out[:, :, 2]
    max_rg = np.maximum(r_ch, g_ch)
    b_ch[fringe] = np.minimum(b_ch[fringe], max_rg[fringe] * 0.96)
    out[:, :, 0] = b_ch

    # 2. Gentle CLAHE on L channel in LAB for micro-clarity (preserves skin texture naturally)
    lab = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_lab = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    l_mix = cv2.addWeighted(l_ch, 0.45, l_eq, 0.55, 0)

    # Slight cooling of yellow flash cast on shirt/skin (-2 on b*, -1 on a*)
    b_lab = np.clip(b_lab.astype(np.int16) - 3, 0, 255).astype(np.uint8)
    a_ch = np.clip(a_ch.astype(np.int16) - 1, 0, 255).astype(np.uint8)
    graded = cv2.cvtColor(cv2.merge([l_mix, a_ch, b_lab]), cv2.COLOR_LAB2BGR).astype(np.float32)

    # 3. Subtle S-curve contrast + gentle shadow darkening on bright shirt so it harmonizes with dark UI
    norm = np.clip(graded / 255.0, 0.0, 1.0)
    s_curve = norm * norm * (3.0 - 2.0 * norm)
    blended = norm * 0.65 + s_curve * 0.35
    # Slightly pull down extreme highlights (white/cream shirt glare)
    blended = np.power(blended, 1.06) * 255.0

    # 4. Gentle unsharp mask for eyes & hair crispness
    blur = cv2.GaussianBlur(blended, (0, 0), sigmaX=1.3)
    sharp = np.clip(blended + 0.42 * (blended - blur), 0, 255).astype(np.uint8)
    return sharp


def main() -> None:
    bgr = cv2.imread(str(RAW_PHOTO))
    if bgr is None:
        raise FileNotFoundError(RAW_PHOTO)

    alpha = build_clean_alpha(bgr)
    graded_bgr = studio_grade(bgr, alpha)
    rgb = cv2.cvtColor(graded_bgr, cv2.COLOR_BGR2RGB)

    # Crop nicely to head + upper shoulders (660 x 680 from top) so shoulders fill full 720px width
    h, w = rgb.shape[:2]
    crop_h = min(h, 680)
    rgb_c = rgb[:crop_h, :]
    alpha_c = alpha[:crop_h, :]

    # Create sleek dark editorial studio backdrop (720x720)
    S = 720
    bg = Image.new("RGBA", (S, S), (7, 12, 24, 255))
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    # Soft cyan + deep slate studio rim-backlight centered behind head/shoulders
    gd.ellipse([90, 50, 630, 620], fill=(16, 46, 82, 175))
    gd.ellipse([175, 100, 545, 480], fill=(0, 190, 240, 54))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=65))
    bg = Image.alpha_composite(bg, glow)

    # Scale up slightly so shoulders reach full width (>= 736px) with zero side seam
    scale = (S + 20) / float(w)
    nw = int(round(w * scale))
    nh = int(round(crop_h * scale))
    fg_pil = Image.fromarray(rgb_c).resize((nw, nh), Image.Resampling.LANCZOS)
    a_pil = Image.fromarray((alpha_c * 255).astype(np.uint8)).resize((nw, nh), Image.Resampling.LANCZOS)

    # Soft bottom & subtle outer-shoulder edge fade into dark frame
    a_arr = np.array(a_pil, dtype=np.float32)
    fade_start = int(nh * 0.78)
    for y in range(fade_start, nh):
        t = (y - fade_start) / max(1, nh - fade_start)
        a_arr[y, :] *= max(0.0, 1.0 - t ** 1.45)
    a_pil = Image.fromarray(np.clip(a_arr, 0, 255).astype(np.uint8))

    fg_rgba = fg_pil.convert("RGBA")
    fg_rgba.putalpha(a_pil)

    px = (S - nw) // 2
    py = 14
    canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    canvas.paste(fg_rgba, (px, py), fg_rgba)
    final = Image.alpha_composite(bg, canvas).convert("RGB")

    OUT_PORTRAIT.parent.mkdir(parents=True, exist_ok=True)
    final.save(OUT_PORTRAIT, quality=95)
    final.save(OUT_ARTIFACT, quality=95)
    print(f"Saved improved studio portrait -> {OUT_PORTRAIT}")


if __name__ == "__main__":
    main()
