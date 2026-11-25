"""Pre-render crisp dark-mode Matplotlib chart layers & heatmap for animated compositing."""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

RESULTS_DIR = ROOT / "research" / "results"
CHARTS_DIR = ROOT / "assets" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def make_backtest_canvas() -> None:
    """Create static grid/axes base AND progressive full-frame chart specs for Scene 4."""
    df = pd.read_csv(RESULTS_DIR / "daily_backtest.csv", parse_dates=["Date"], index_col="Date")
    grid_df = pd.read_csv(RESULTS_DIR / "sensitivity_grid.csv")

    # 1. Pre-render 5x5 Sensitivity Heatmap (for Scene 4 Phase B) -> 600x400 px
    piv = grid_df.pivot(index="fast_ema", columns="slow_ema", values="oos_sharpe")
    fig, ax = plt.subplots(figsize=(6.1, 3.85), dpi=100, facecolor="#0B1220")
    ax.set_facecolor("#0B1220")
    cmap = LinearSegmentedColormap.from_list("sharpe_cmap", ["#0F2942", "#0088B8", "#00D4FF", "#10B981"])
    im = ax.imshow(piv.values, cmap=cmap, vmin=1.15, vmax=1.48, aspect="auto")

    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c}d" for c in piv.columns], color="#CBD5E1", fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{r}d" for r in piv.index], color="#CBD5E1", fontsize=10, fontweight="bold")
    ax.set_xlabel("Slow EMA Window", color="#94A3B8", fontsize=10, labelpad=6)
    ax.set_ylabel("Fast EMA Window", color="#94A3B8", fontsize=10, labelpad=6)
    ax.set_title("OOS Parameter Stability (2023–2026 Sharpe)", color="#F8FAFC", fontsize=12, fontweight="bold", pad=10)

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="#04101D", fontsize=11, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_edgecolor("#1E293B")
    plt.tight_layout(pad=1.0)
    fig.savefig(CHARTS_DIR / "sensitivity_heatmap.png", dpi=100, facecolor="#0B1220")
    plt.close(fig)

    # 2. Pre-render Scene 4 Static Chart Base (1100 x 680 px) + save exact pixel plot coordinates
    # so our frame renderer draws razor-sharp animated curves inside exact axes bounds!
    fig = plt.figure(figsize=(11.0, 6.8), dpi=100, facecolor="#0A101E")
    gs = fig.add_gridspec(2, 1, height_ratios=[2.1, 1.0], hspace=0.14, left=0.075, right=0.97, top=0.90, bottom=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    for ax in (ax1, ax2):
        ax.set_facecolor("#0A101E")
        ax.grid(True, color="#1E293B", alpha=0.55, linewidth=0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("bottom", "left"):
            ax.spines[sp].set_color("#334155")
        ax.tick_params(colors="#94A3B8", labelsize=10)

    oos_dt = pd.Timestamp("2023-01-01")
    ax1.set_xlim(df.index.min(), df.index.max())
    y1_min = 0.85
    y1_max = max(df["eq_regime"].max(), df["eq_bh"].max()) * 1.07
    ax1.set_ylim(y1_min, y1_max)
    ax1.set_ylabel("Cumulative Growth (1.0 = Start)", color="#CBD5E1", fontsize=10.5)
    ax1.set_title("Walk-Forward Equity & Drawdown (Net of 10 bps Round-Trip Costs)", color="#F8FAFC", fontsize=13, fontweight="bold", loc="left", pad=10)

    # Shaded OOS zone
    ax1.axvspan(oos_dt, df.index.max(), color="#00D4FF", alpha=0.055)
    ax2.axvspan(oos_dt, df.index.max(), color="#00D4FF", alpha=0.055)
    ax1.axvline(oos_dt, color="#00D4FF", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.axvline(oos_dt, color="#00D4FF", linestyle="--", linewidth=1.2, alpha=0.7)

    ax1.text(pd.Timestamp("2020-06-01"), y1_max * 0.93, "IN-SAMPLE (2018–2022)", color="#64748B", fontsize=10, fontweight="bold", ha="center")
    ax1.text(pd.Timestamp("2024-11-01"), y1_max * 0.93, "OUT-OF-SAMPLE (2023–2026)", color="#00D4FF", fontsize=10, fontweight="bold", ha="center")
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.set_ylim(-0.42, 0.02)
    ax2.set_ylabel("Drawdown", color="#CBD5E1", fontsize=10.5)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.canvas.draw()
    # Save exact data->pixel mapping bbox for ax1 and ax2
    b1 = ax1.get_window_extent().bounds  # x0, y0_from_bottom, w, h
    b2 = ax2.get_window_extent().bounds
    fig.savefig(CHARTS_DIR / "backtest_axes_base.png", dpi=100, facecolor="#0A101E")
    plt.close(fig)

    # Save JSON layout coordinates (PIL y = 680 - mpl_y)
    import json
    spec = {
        "width": 1100,
        "height": 680,
        "ax1": {
            "x0": b1[0], "x1": b1[0] + b1[2],
            "y_top": 680 - (b1[1] + b1[3]), "y_bot": 680 - b1[1],
            "ylim": [float(y1_min), float(y1_max)],
        },
        "ax2": {
            "x0": b2[0], "x1": b2[0] + b2[2],
            "y_top": 680 - (b2[1] + b2[3]), "y_bot": 680 - b2[1],
            "ylim": [-0.42, 0.02],
        },
        "n_points": int(len(df)),
    }
    (CHARTS_DIR / "backtest_axes_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    # 3. Pre-render Scene 5 Conceptual Expiry Intraday Vol Curve Base (860x480)
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=100, facecolor="#0A101E")
    ax.set_facecolor("#0A101E")
    ax.grid(True, color="#1E293B", alpha=0.55, linewidth=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color("#334155")
    ax.set_xlim(0, 120)
    ax.set_ylim(9, 24)
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_xticklabels(["13:30", "14:00", "14:30", "15:00", "15:30 Close"], color="#94A3B8", fontsize=10)
    ax.set_yticks([10, 14, 18, 22])
    ax.set_yticklabels(["10%", "14%", "18%", "22%"], color="#94A3B8", fontsize=10)
    ax.set_ylabel("Annualized Volatility (%)", color="#CBD5E1", fontsize=10.5)
    ax.set_title("Research Hypothesis: NSE Weekly Expiry Final 2-Hour Vol Spread", color="#F8FAFC", fontsize=12.5, fontweight="bold", loc="left", pad=10)

    # Conceptual curve: Implied India VIX baseline + gamma-spike realized vol + post-cost threshold band
    x_m = np.linspace(0, 120, 240)
    iv_base = 13.8 + 0.3 * np.sin(x_m / 40.0)
    rv_spike = 14.0 + 5.6 * np.exp(-((x_m - 74.0) / 22.0) ** 2) + 0.45 * np.sin(x_m * 0.35)
    ax.fill_between(x_m, iv_base - 1.1, iv_base + 1.1, color="#64748B", alpha=0.18, label="Bid-Ask + STT Cost Band")
    ax.plot(x_m, iv_base, color="#94A3B8", linestyle="--", linewidth=2.0, label="Implied India VIX Baseline")
    ax.plot(x_m, rv_spike, color="#F59E0B", linewidth=2.6, label="Expiry Gamma / Flow Spike (Hypothesized)")
    ax.annotate("Overshoot vs VIX:\nAlpha or Tail-Gamma Risk?", xy=(74, 19.6), xytext=(22, 20.8),
                arrowprops=dict(arrowstyle="->", color="#00D4FF", lw=1.6),
                color="#00D4FF", fontsize=10, fontweight="bold")
    ax.legend(loc="lower left", frameon=True, facecolor="#0F172A", edgecolor="#1E293B", labelcolor="#E2E8F0", fontsize=9.5)
    plt.tight_layout(pad=1.0)
    fig.savefig(CHARTS_DIR / "expiry_hypothesis_chart.png", dpi=100, facecolor="#0A101E")
    plt.close(fig)

    # 4. Also save a clean static tearsheet for the GitHub README
    print("Saved charts:", list(p.name for p in CHARTS_DIR.glob("*.png")))


if __name__ == "__main__":
    make_backtest_canvas()
