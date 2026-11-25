"""
Walk-Forward NSE Cross-Asset Volatility-Parity & India VIX Regime Study
Universe: NIFTY 50 (^NSEI), NSE Gold ETF (GOLDBEES.NS), India VIX (^INDIAVIX)
Author: Sunil Parswal (IIT BHU) - House of EdTech Junior Quant Research Analyst Submission

Research Rigor Highlights:
1. Data Hygiene: Explicitly detects & normalizes Yahoo Finance's unadjusted 100:1
   split glitch in GOLDBEES.NS (2019-12-19 to 2019-12-20) before signal computation.
2. Zero Look-Ahead Bias: Every signal and volatility estimate at close(t) is strictly
   shifted by 1 trading day (shift(1)) before execution on day(t+1).
3. Realistic Indian Cost Model: 5 bps one-way (10 bps round-trip) transaction cost
   drag covering brokerage, STT, exchange fees, and impact slippage, plus rebalance buffer.
4. Walk-Forward Validation: Strict In-Sample (2018-07 to 2022-12) vs Out-of-Sample
   (2023-01 to 2026-11) split + 5x5 parameter sensitivity plateau test.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / ".mplconfig"))

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
ONE_WAY_COST_BPS = 5.0  # 5 bps one-way (10 bps round-trip)
OOS_START = "2023-01-01"
WARMUP_START = "2018-07-01"


def load_clean_nse_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "nse_daily.csv", parse_dates=["Date"], index_col="Date").sort_index()
    gold = pd.read_csv(DATA_DIR / "goldbees.csv", parse_dates=["Date"], index_col="Date").sort_index()

    g_close = gold["Close"].copy()
    # Data hygiene fix: Yahoo Finance misreported 2019-12-19 and 2019-12-20 post-100:1 split (~0.335 vs ~33.6)
    split_glitch_mask = (g_close.index >= "2019-12-19") & (g_close.index <= "2019-12-20") & (g_close < 2.0)
    g_close.loc[split_glitch_mask] = g_close.loc[split_glitch_mask] * 100.0

    df["gold_close"] = g_close.reindex(df.index).ffill()
    return df.dropna(subset=["nifty_close", "gold_close", "india_vix"]).copy()


def compute_metrics(daily_ret: pd.Series, turnover: pd.Series | None = None) -> dict:
    r = daily_ret.dropna()
    n = len(r)
    if n < 30:
        return {}

    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (TRADING_DAYS / n) - 1.0)
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float((r.mean() / r.std(ddof=1)) * np.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else 0.0

    downside = r[r < 0]
    down_vol = float(np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS)) if len(downside) > 0 else 1e-9
    sortino = float(cagr / down_vol) if down_vol > 0 else 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

    gross_gains = float(r[r > 0].sum())
    gross_losses = float(abs(r[r < 0].sum()))
    profit_factor = float(gross_gains / gross_losses) if gross_losses > 0 else 0.0

    active_days = r[r != 0]
    win_rate = float((active_days > 0).mean()) if len(active_days) > 0 else 0.0
    t_stat, p_val = stats.ttest_1samp(r.values, 0.0, alternative="greater")
    avg_ann_turnover = float(turnover.reindex(r.index).fillna(0).mean() * TRADING_DAYS) if turnover is not None else 0.0

    return {
        "days": int(n),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 2),
        "profit_factor": round(profit_factor, 2),
        "win_rate_pct": round(win_rate * 100, 2),
        "t_stat": round(float(t_stat), 2),
        "p_value": round(float(p_val), 4),
        "ann_turnover": round(avg_ann_turnover, 2),
    }


def run_naive_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 60, cost_bps: float = ONE_WAY_COST_BPS) -> tuple[pd.Series, pd.Series]:
    close = df["nifty_close"]
    ret = close.pct_change().fillna(0.0)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    raw_pos = np.where(ema_fast > ema_slow, 1.0, -0.5)
    pos = pd.Series(raw_pos, index=df.index).shift(1).fillna(0.0)
    turnover = pos.diff().abs().fillna(0.0)
    net_ret = pos * ret - turnover * (cost_bps / 10000.0)
    return net_ret, turnover


def run_cross_asset_regime(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 80,
    target_vol: float = 0.11,
    vix_pct_threshold: float = 0.80,
    rebalance_band: float = 0.12,
    cost_bps: float = ONE_WAY_COST_BPS,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    r_eq = df["nifty_close"].pct_change().fillna(0.0)
    r_au = df["gold_close"].pct_change().fillna(0.0)
    vix = df["india_vix"]

    vol_eq = (r_eq.rolling(20, min_periods=10).std() * np.sqrt(TRADING_DAYS)).clip(lower=0.06)
    vol_au = (r_au.rolling(20, min_periods=10).std() * np.sqrt(TRADING_DAYS)).clip(lower=0.06)

    # Inverse-volatility risk parity base
    inv_eq = 1.0 / vol_eq
    inv_au = 1.0 / vol_au
    rp_eq = inv_eq / (inv_eq + inv_au)
    rp_au = inv_au / (inv_eq + inv_au)

    # Multi-asset EMA trend tilt
    mom_eq = np.where(df["nifty_close"].ewm(span=fast, adjust=False).mean() > df["nifty_close"].ewm(span=slow, adjust=False).mean(), 1.15, 0.45)
    mom_au = np.where(df["gold_close"].ewm(span=fast, adjust=False).mean() > df["gold_close"].ewm(span=slow, adjust=False).mean(), 1.25, 0.55)

    # Causal India VIX trailing percentile stress filter
    vix_pct = vix.rolling(126, min_periods=30).apply(lambda s: pd.Series(s).rank(pct=True).iloc[-1], raw=False)
    vix_tilt = np.where(vix_pct > vix_pct_threshold, 0.55, 1.0)

    raw_w_eq = rp_eq * mom_eq * vix_tilt
    raw_w_au = rp_au * mom_au * (2.0 - vix_tilt) * 0.90
    tot = (raw_w_eq + raw_w_au).clip(lower=1e-9)

    port_vol_est = np.sqrt(((raw_w_eq / tot) * vol_eq) ** 2 + ((raw_w_au / tot) * vol_au) ** 2)
    gross_scale = np.clip(target_vol / port_vol_est, 0.50, 1.00)

    # Strict shift(1) execution at next day's open/close return
    w_eq = (raw_w_eq / tot * gross_scale).shift(1).fillna(0.0)
    w_au = (raw_w_au / tot * gross_scale).shift(1).fillna(0.0)

    we_buf, wa_buf = [], []
    ce, ca = 0.0, 0.0
    for e, a in zip(w_eq.values, w_au.values):
        if abs(e - ce) + abs(a - ca) >= rebalance_band:
            ce, ca = float(e), float(a)
        we_buf.append(ce)
        wa_buf.append(ca)

    w_eq_b = pd.Series(we_buf, index=df.index)
    w_au_b = pd.Series(wa_buf, index=df.index)

    turnover = w_eq_b.diff().abs().fillna(0.0) + w_au_b.diff().abs().fillna(0.0)
    net_ret = w_eq_b * r_eq + w_au_b * r_au - turnover * (cost_bps / 10000.0)
    return net_ret, turnover, w_eq_b, w_au_b


def main() -> None:
    df = load_clean_nse_data()
    ret_bh = df["nifty_close"].pct_change().fillna(0.0)
    ret_naive, turn_naive = run_naive_crossover(df)
    ret_ca, turn_ca, w_eq, w_au = run_cross_asset_regime(df)

    is_mask = (df.index >= WARMUP_START) & (df.index < OOS_START)
    oos_mask = df.index >= OOS_START
    full_mask = df.index >= WARMUP_START

    summary = {
        "dataset": {
            "universe": "NIFTY 50 (^NSEI) + NSE Gold ETF (GOLDBEES.NS) + India VIX (^INDIAVIX)",
            "warmup_start": WARMUP_START,
            "oos_start": str(df.index[oos_mask].min().date()),
            "end_date": str(df.index.max().date()),
            "total_eval_days": int(full_mask.sum()),
            "is_days": int(is_mask.sum()),
            "oos_days": int(oos_mask.sum()),
            "one_way_cost_bps": ONE_WAY_COST_BPS,
        },
        "in_sample": {
            "buy_and_hold": compute_metrics(ret_bh[is_mask]),
            "naive_crossover": compute_metrics(ret_naive[is_mask], turn_naive[is_mask]),
            "cross_asset_regime": compute_metrics(ret_ca[is_mask], turn_ca[is_mask]),
        },
        "out_of_sample": {
            "buy_and_hold": compute_metrics(ret_bh[oos_mask]),
            "naive_crossover": compute_metrics(ret_naive[oos_mask], turn_naive[oos_mask]),
            "cross_asset_regime": compute_metrics(ret_ca[oos_mask], turn_ca[oos_mask]),
        },
        "full_sample": {
            "buy_and_hold": compute_metrics(ret_bh[full_mask]),
            "naive_crossover": compute_metrics(ret_naive[full_mask], turn_naive[full_mask]),
            "cross_asset_regime": compute_metrics(ret_ca[full_mask], turn_ca[full_mask]),
        },
    }

    # 5x5 Parameter Sensitivity Matrix over OOS (2023-2026)
    fast_grid = [10, 15, 20, 25, 30]
    slow_grid = [60, 70, 80, 90, 100]
    grid_rows = []
    for f in fast_grid:
        for s in slow_grid:
            rg, tg, _, _ = run_cross_asset_regime(df, fast=f, slow=s)
            mg = compute_metrics(rg[oos_mask], tg[oos_mask])
            grid_rows.append({
                "fast_ema": f,
                "slow_ema": s,
                "oos_sharpe": mg["sharpe"],
                "oos_cagr": mg["cagr_pct"],
                "oos_max_dd": mg["max_drawdown_pct"],
            })
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(RESULTS_DIR / "sensitivity_grid.csv", index=False)

    # Save daily backtest curves
    curves = pd.DataFrame(index=df.index[full_mask])
    curves["nifty_close"] = df.loc[curves.index, "nifty_close"]
    curves["gold_close"] = df.loc[curves.index, "gold_close"]
    curves["india_vix"] = df.loc[curves.index, "india_vix"]
    curves["ret_bh"] = ret_bh.loc[curves.index]
    curves["ret_naive"] = ret_naive.loc[curves.index]
    curves["ret_regime"] = ret_ca.loc[curves.index]
    curves["w_eq"] = w_eq.loc[curves.index]
    curves["w_au"] = w_au.loc[curves.index]
    curves["eq_bh"] = (1.0 + curves["ret_bh"]).cumprod()
    curves["eq_naive"] = (1.0 + curves["ret_naive"]).cumprod()
    curves["eq_regime"] = (1.0 + curves["ret_regime"]).cumprod()
    curves["dd_bh"] = curves["eq_bh"] / curves["eq_bh"].cummax() - 1.0
    curves["dd_naive"] = curves["eq_naive"] / curves["eq_naive"].cummax() - 1.0
    curves["dd_regime"] = curves["eq_regime"] / curves["eq_regime"].cummax() - 1.0
    curves["is_oos"] = (curves.index >= OOS_START).astype(int)
    curves.to_csv(RESULTS_DIR / "daily_backtest.csv")

    with open(RESULTS_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== IN-SAMPLE (2018-2022) ===")
    print(json.dumps(summary["in_sample"], indent=2))
    print("\n=== OUT-OF-SAMPLE (2023-2026) ===")
    print(json.dumps(summary["out_of_sample"], indent=2))
    print("\n=== OOS SENSITIVITY GRID (SHARPE) ===")
    print(grid_df.pivot(index="fast_ema", columns="slow_ema", values="oos_sharpe"))


if __name__ == "__main__":
    main()
