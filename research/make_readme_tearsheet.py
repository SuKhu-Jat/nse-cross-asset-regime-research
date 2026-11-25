"""Render a static GitHub README tearsheet overview from backtest results."""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

RESULTS = ROOT / "results"


def main() -> None:
    df = pd.read_csv(RESULTS / "daily_backtest.csv", parse_dates=["Date"], index_col="Date")
    grid_df = pd.read_csv(RESULTS / "sensitivity_grid.csv")
    piv = grid_df.pivot(index="fast_ema", columns="slow_ema", values="oos_sharpe")

    fig = plt.figure(figsize=(15.0, 6.8), dpi=140, facecolor="#070B14")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.65, 1.0], height_ratios=[2.0, 1.0], wspace=0.22, hspace=0.16, left=0.06, right=0.97, top=0.88, bottom=0.10)

    ax_eq = fig.add_subplot(gs[0, 0])
    ax_dd = fig.add_subplot(gs[1, 0], sharex=ax_eq)
    ax_hm = fig.add_subplot(gs[:, 1])

    for ax in (ax_eq, ax_dd):
        ax.set_facecolor("#0A101E")
        ax.grid(True, color="#1E293B", alpha=0.55, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#334155")
        ax.spines["left"].set_color("#334155")
        ax.tick_params(colors="#94A3B8", labelsize=9.5)

    oos_dt = pd.Timestamp("2023-01-01")
    ax_eq.axvspan(oos_dt, df.index.max(), color="#00D4FF", alpha=0.06)
    ax_dd.axvspan(oos_dt, df.index.max(), color="#00D4FF", alpha=0.06)
    ax_eq.axvline(oos_dt, color="#00D4FF", linestyle="--", linewidth=1.1, alpha=0.75)
    ax_dd.axvline(oos_dt, color="#00D4FF", linestyle="--", linewidth=1.1, alpha=0.75)

    ax_eq.plot(df.index, df["eq_bh"], color="#64748B", linewidth=1.5, label="NIFTY 50 Buy & Hold (OOS Sharpe 0.78)")
    ax_eq.plot(df.index, df["eq_naive"], color="#F59E0B", linewidth=1.6, label="Naive 20/60 EMA Crossover (OOS Sharpe 0.50)")
    ax_eq.plot(df.index, df["eq_regime"], color="#00D4FF", linewidth=2.3, label="Cross-Asset Regime Model (OOS Sharpe 1.32)")
    ax_eq.set_ylabel("Cumulative Growth (1.0 = Start)", color="#CBD5E1", fontsize=10)
    ax_eq.legend(loc="upper left", frameon=True, facecolor="#0F172A", edgecolor="#1E293B", labelcolor="#F8FAFC", fontsize=9)
    plt.setp(ax_eq.get_xticklabels(), visible=False)

    ax_dd.plot(df.index, df["dd_bh"] * 100, color="#64748B", linewidth=1.2)
    ax_dd.plot(df.index, df["dd_naive"] * 100, color="#F59E0B", linewidth=1.3)
    ax_dd.plot(df.index, df["dd_regime"] * 100, color="#10B981", linewidth=1.8)
    ax_dd.set_ylabel("Drawdown (%)", color="#CBD5E1", fontsize=10)
    ax_dd.xaxis.set_major_locator(mdates.YearLocator())
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax_hm.set_facecolor("#0A101E")
    cmap = LinearSegmentedColormap.from_list("sharpe_cmap", ["#0F2942", "#0088B8", "#00D4FF", "#10B981"])
    ax_hm.imshow(piv.values, cmap=cmap, vmin=1.15, vmax=1.48, aspect="auto")
    ax_hm.set_xticks(range(len(piv.columns)))
    ax_hm.set_xticklabels([f"{c}d" for c in piv.columns], color="#CBD5E1", fontsize=9.5, fontweight="bold")
    ax_hm.set_yticks(range(len(piv.index)))
    ax_hm.set_yticklabels([f"{r}d" for r in piv.index], color="#CBD5E1", fontsize=9.5, fontweight="bold")
    ax_hm.set_xlabel("Slow EMA Window", color="#94A3B8", fontsize=10)
    ax_hm.set_ylabel("Fast EMA Window", color="#94A3B8", fontsize=10)
    ax_hm.set_title("OOS Parameter Stability (2023–2026 Sharpe)", color="#F8FAFC", fontsize=11.5, fontweight="bold", pad=10)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax_hm.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", color="#04101D", fontsize=10.5, fontweight="bold")
    for sp in ax_hm.spines.values():
        sp.set_edgecolor("#1E293B")

    fig.suptitle("NSE Walk-Forward Cross-Asset Regime Study (NIFTY 50 + Gold ETF + India VIX, Net of 10 bps RT Cost)", color="#F8FAFC", fontsize=13.5, fontweight="bold", y=0.955)
    out_p = RESULTS / "tearsheet_overview.png"
    fig.savefig(out_p, dpi=140, facecolor="#070B14")
    plt.close(fig)
    print(f"Saved {out_p}")


if __name__ == "__main__":
    main()
