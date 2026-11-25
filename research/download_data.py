"""Download historical NSE NIFTY 50 and India VIX daily data via yfinance."""

from pathlib import Path
import yfinance as yf
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    tickers = {"nifty": "^NSEI", "vix": "^INDIAVIX"}
    frames = {}
    for name, symbol in tickers.items():
        df = yf.download(symbol, start="2018-01-01", progress=False, auto_adjust=True)
        if df.empty:
            raise RuntimeError(f"Failed to download data for {symbol}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.to_csv(DATA_DIR / f"{name}.csv")
        frames[name] = df
        print(f"Saved {name} ({symbol}): {len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")

    merged = pd.DataFrame(index=frames["nifty"].index)
    merged["nifty_open"] = frames["nifty"]["Open"]
    merged["nifty_high"] = frames["nifty"]["High"]
    merged["nifty_low"] = frames["nifty"]["Low"]
    merged["nifty_close"] = frames["nifty"]["Close"]
    merged["india_vix"] = frames["vix"]["Close"].reindex(merged.index).ffill()
    merged = merged.dropna(subset=["nifty_close", "india_vix"])
    merged.to_csv(DATA_DIR / "nse_daily.csv")
    print(f"Merged dataset saved: {len(merged)} rows -> {DATA_DIR / 'nse_daily.csv'}")


if __name__ == "__main__":
    main()
