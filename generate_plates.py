"""Generate high-resolution mathematical & dark-mode visual plates for scenes without AI plates."""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFilter

PLATES_DIR = ROOT / "assets" / "plates"
PLATES_DIR.mkdir(parents=True, exist_ok=True)


def make_vol_surface(out_path: Path) -> None:
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#050811")
    ax = fig.add_subplot(111, projection="3d", facecolor="#050811")

    # Realistic volatility smile + term structure (short expiry gamma spike!)
    m_grid = np.linspace(-0.18, 0.18, 80)   # log-moneyness
    t_grid = np.linspace(0.01, 0.25, 65)    # years to expiry (weekly -> quarterly)
    M, T = np.meshgrid(m_grid, t_grid)

    # Skewed SVI-inspired implied volatility surface with steep short-dated expiry wing
    iv = 0.125 + 0.095 * np.exp(-16.0 * T) * (M ** 2) * 18.0 - 0.18 * M * np.exp(-4.0 * T) + 0.032 * np.exp(-22.0 * T)
    iv = np.clip(iv, 0.10, 0.42)

    cmap = LinearSegmentedColormap.from_list(
        "quant_glow", ["#0A2540", "#0088CC", "#00D4FF", "#10B981", "#F59E0B"]
    )

    surf = ax.plot_surface(
        M * 100, T * 365, iv * 100,
        cmap=cmap, edgecolor=(0.0, 0.9, 1.0, 0.18), linewidth=0.35,
        antialiased=True, alpha=0.88, shade=True
    )

    # Position camera nicely on right half so left side stays clean for text
    ax.view_init(elev=24, azim=-52)
    ax.set_position([0.28, -0.05, 0.78, 1.08])
    ax.set_axis_off()
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor((0, 0, 0, 0))

    fig.savefig(out_path, dpi=100, facecolor="#050811")
    plt.close(fig)

    # Subtle bloom post-process with PIL
    img = Image.open(out_path).convert("RGB")
    glow = img.filter(ImageFilter.GaussianBlur(radius=8))
    blended = Image.blend(img, glow, alpha=0.22)
    # Darken left side smoothly for clean text legibility
    grad = Image.new("L", blended.size, 0)
    gd = ImageDraw.Draw(grad)
    w, h = blended.size
    for x in range(w):
        alpha = int(235 * max(0.0, min(1.0, 1.0 - (x / (w * 0.72)))))
        gd.line([(x, 0), (x, h)], fill=alpha)
    dark = Image.new("RGB", blended.size, (5, 8, 17))
    final = Image.composite(dark, blended, grad)
    final.save(out_path, quality=95)
    print(f"Saved {out_path}")


def make_quant_skyline_plate(out_path: Path) -> None:
    w, h = 1920, 1080
    rng = np.random.default_rng(42)
    base = Image.new("RGB", (w, h), (6, 10, 20))
    draw = ImageDraw.Draw(base, "RGBA")

    # Subtle radial glow on right
    glow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow_layer)
    gd.ellipse([900, 140, 2050, 1050], fill=(0, 150, 220, 35))
    gd.ellipse([1100, 280, 1920, 960], fill=(16, 185, 129, 25))
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=90))
    base = Image.alpha_composite(base.convert("RGBA"), glow_layer)

    # Subtle algorithmic grid
    grid = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gr = ImageDraw.Draw(grid)
    for x in range(860, w, 60):
        gr.line([(x, 80), (x, h - 80)], fill=(0, 212, 255, 16), width=1)
    for y in range(120, h - 60, 60):
        gr.line([(860, y), (w, y)], fill=(0, 212, 255, 16), width=1)
    base = Image.alpha_composite(base, grid)

    # Stylized financial skyline / volume-profile bars rising on the right
    sky = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sky)
    x_pos = 920
    while x_pos < w - 40:
        bw = int(rng.integers(24, 58))
        bh = int(rng.integers(140, 620) * (0.65 + 0.55 * np.sin((x_pos - 900) / 360.0) ** 2))
        top = h - bh - 90
        col = (14, 34, 62, 180) if rng.random() > 0.3 else (10, 50, 72, 175)
        sd.rectangle([x_pos, top, x_pos + bw, h], fill=col, outline=(0, 212, 255, 45), width=1)
        # Lit window ticks
        for wy in range(top + 16, h - 100, 22):
            if rng.random() > 0.45:
                wc = (0, 212, 255, 95) if rng.random() > 0.25 else (245, 158, 11, 110)
                sd.rectangle([x_pos + 5, wy, x_pos + bw - 5, wy + 4], fill=wc)
        x_pos += bw + int(rng.integers(6, 16))
    base = Image.alpha_composite(base, sky)

    # Floating alpha trajectory arcs
    arcs = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ad = ImageDraw.Draw(arcs)
    xs = np.linspace(780, w, 400)
    for amp, freq, phase, col in [
        (95, 0.0055, 0.0, (0, 212, 255, 185)),
        (65, 0.0072, 1.3, (16, 185, 129, 165)),
        (45, 0.0041, 2.4, (245, 158, 11, 145)),
    ]:
        ys = 470 - 0.22 * (xs - 780) + amp * np.sin(freq * (xs - 780) + phase)
        pts = list(zip(xs.tolist(), ys.tolist()))
        ad.line(pts, fill=col, width=3)
    arcs_glow = arcs.filter(ImageFilter.GaussianBlur(radius=6))
    base = Image.alpha_composite(base, arcs_glow)
    base = Image.alpha_composite(base, arcs)

    base.convert("RGB").save(out_path)
    print(f"Saved {out_path}")


def make_ai_network_plate(out_path: Path) -> None:
    w, h = 1920, 1080
    rng = np.random.default_rng(7)
    base = Image.new("RGBA", (w, h), (5, 9, 19, 255))

    # Neural / Genetic Algorithm node layers on right half
    layers = [
        (1060, [240, 380, 520, 660, 800]),
        (1320, [200, 340, 480, 620, 760, 880]),
        (1580, [290, 450, 610, 770]),
        (1800, [420, 620]),
    ]
    net = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    nd = ImageDraw.Draw(net)
    for i in range(len(layers) - 1):
        x1, ys1 = layers[i]
        x2, ys2 = layers[i + 1]
        for y1 in ys1:
            for y2 in ys2:
                active = rng.random() > 0.35
                col = (0, 212, 255, 110) if active else (16, 185, 129, 32)
                width = 2 if active else 1
                nd.line([(x1, y1), (x2, y2)], fill=col, width=width)

    for x, ys in layers:
        for y in ys:
            r = 11
            nd.ellipse([x - r - 6, y - r - 6, x + r + 6, y + r + 6], fill=(0, 212, 255, 45))
            nd.ellipse([x - r, y - r, x + r, y + r], fill=(10, 24, 46, 255), outline=(0, 212, 255, 230), width=2)

    glow = net.filter(ImageFilter.GaussianBlur(radius=9))
    base = Image.alpha_composite(base, glow)
    base = Image.alpha_composite(base, net)
    base.convert("RGB").save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    make_vol_surface(PLATES_DIR / "vol_surface_3d.png")
    make_quant_skyline_plate(PLATES_DIR / "mumbai_quant_skyline.png")
    make_ai_network_plate(PLATES_DIR / "ai_pipeline_network.png")
