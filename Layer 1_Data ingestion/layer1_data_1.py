"""
=============================================================
LAYER 1 — DATA INGESTION (final)
Gold / A-Share Miner Arbitrage Backtest
=============================================================

ALL data read from local CSVs — no internet connection needed.

Folder structure expected:
    C:/Gold ETF arbitrage/Layer 1_Data ingestion/stock_data_files/
        GC=F.csv          ← gold futures (USD/oz)
        USDCNY=X.csv      ← FX rate for USD → CNY conversion (optional)
        601899.SS.csv
        600547.SS.csv
        600489.SS.csv
        000975.SZ.csv
        002155.SZ.csv
        601069.SS.csv
        002273.SZ.csv
        002237.SZ.csv

CSV format expected (standard yfinance export):
    Date, Open, High, Low, Close, Volume

Output of load_all():
    prices   : pd.DataFrame  — index=Date, columns=[gold_cny, 601899.SS, ...]
    reserves : pd.DataFrame  — static reserves / metadata table

Downstream layers import this file as:
    from layer1_data import load_all
"""

import os
import pandas as pd

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

DATA_DIR      = r"C:\Gold ETF arbitrage\Layer 1_Data ingestion\stock_data_files"
GOLD_TICKER   = "GC=F"
USDCNY_TICKER = "USDCNY=X"

START_DATE  = "2012-01-01"
END_DATE    = "2024-12-31"     # swap to datetime.today().strftime("%Y-%m-%d") for live use

FFILL_LIMIT = 3                # forward-fill up to N days across holiday gaps
MIN_HISTORY = 252 * 5          # drop any series with fewer than 5 years of valid rows


# ─────────────────────────────────────────────────────────────────
# RESERVES TABLE  (static — from annual reports / WGC, approx 2023)
# ─────────────────────────────────────────────────────────────────

RESERVES = {
    #  ticker         name                             exchange   Moz   listed
    "601899.SS": ("Zijin Mining",                      "SSE",    53.0,  2003),
    "600547.SS": ("Shandong Gold",                     "SSE",    35.0,  2000),
    "600489.SS": ("Zhongjin Gold",                     "SSE",    12.0,  2000),
    "000975.SZ": ("Shanjin Intl Gold (ex-Yintai)",     "SZSE",    8.0,  1999),
    "002155.SZ": ("Hunan Gold",                        "SZSE",    7.8,  2007),
    "601069.SS": ("Western Region Gold",               "SSE",     6.0,  2011),
    "002273.SZ": ("Chifeng Jilong Gold",               "SZSE",    9.5,  2010),
    "002237.SZ": ("Shandong Humon Smelting",           "SZSE",    4.0,  2008),
}


def get_reserves_table() -> pd.DataFrame:
    rows = [
        {
            "ticker":       ticker,
            "name":         v[0],
            "exchange":     v[1],
            "reserves_moz": v[2],
            "listed_year":  v[3],
        }
        for ticker, v in RESERVES.items()
    ]
    return pd.DataFrame(rows).set_index("ticker")


# ─────────────────────────────────────────────────────────────────
# CORE LOADER — reads one CSV, returns Close as a Series
# ─────────────────────────────────────────────────────────────────

def _load_csv(filepath: str, col: str = "Close") -> pd.Series:
    """
    Reads a standard yfinance-export CSV.
    Returns the requested column as a pd.Series with a DatetimeIndex.
    Raises clear errors if the file or column is missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found:\n    {filepath}")

    df = pd.read_csv(
        filepath,
        index_col   = "Date",
        parse_dates = True,
    )
    df.columns = df.columns.str.strip()   # remove accidental whitespace

    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' missing in {os.path.basename(filepath)}. "
            f"Found: {list(df.columns)}"
        )

    series = df[col].dropna()
    series.index.name = "Date"
    return series


# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD GOLD  (USD/oz)
# ─────────────────────────────────────────────────────────────────

def load_gold(
    data_dir: str  = DATA_DIR,
    start:    str  = START_DATE,
    end:      str  = END_DATE,
    verbose:  bool = True,
) -> pd.Series:
    filepath = os.path.join(data_dir, f"{GOLD_TICKER}.csv")
    gold_usd = _load_csv(filepath, col="Close").loc[start:end].rename("gold_usd")

    if verbose:
        print(f"  Loaded {'GC=F (gold USD/oz)':<30}  "
              f"rows={len(gold_usd):,}  "
              f"{gold_usd.index[0].date()} → {gold_usd.index[-1].date()}")
    return gold_usd


# ─────────────────────────────────────────────────────────────────
# STEP 2 — LOAD FX  (USD/CNY)
# ─────────────────────────────────────────────────────────────────

def load_fx(
    data_dir: str  = DATA_DIR,
    start:    str  = START_DATE,
    end:      str  = END_DATE,
    verbose:  bool = True,
) -> pd.Series:
    """
    Loads USDCNY=X.csv if present.
    Falls back to a fixed rate of 7.10 with a warning if the file is missing —
    acceptable for a first-pass backtest; replace with real FX data for production.
    """
    filepath = os.path.join(data_dir, f"{USDCNY_TICKER}.csv")

    if os.path.exists(filepath):
        usdcny = _load_csv(filepath, col="Close").loc[start:end].rename("usdcny")
        if verbose:
            print(f"  Loaded {'USDCNY=X':<30}  "
                  f"rows={len(usdcny):,}  "
                  f"avg rate={usdcny.mean():.4f}")
    else:
        print(f"  [WARN] USDCNY=X.csv not found — using fixed fallback rate 7.10")
        usdcny = pd.Series(
            7.10,
            index = pd.bdate_range(start, end),
            name  = "usdcny",
        )

    return usdcny


# ─────────────────────────────────────────────────────────────────
# STEP 3 — CONVERT GOLD USD → CNY
# ─────────────────────────────────────────────────────────────────

def convert_gold_to_cny(
    gold_usd: pd.Series,
    usdcny:   pd.Series,
    verbose:  bool = True,
) -> pd.Series:
    """
    gold_cny (CNY/oz) = gold_usd (USD/oz) × USDCNY rate.
    FX is forward-filled onto gold's date index to cover any gaps.
    """
    fx_aligned = usdcny.reindex(gold_usd.index).ffill().bfill()
    gold_cny   = (gold_usd * fx_aligned).rename("gold_cny")

    if verbose:
        print(f"  Converted gold: "
              f"{gold_usd.iloc[-1]:.1f} USD/oz × "
              f"{fx_aligned.iloc[-1]:.4f} = "
              f"{gold_cny.iloc[-1]:.1f} CNY/oz  (latest date)")

    return gold_cny


# ─────────────────────────────────────────────────────────────────
# STEP 4 — LOAD ALL MINER CSVs
# ─────────────────────────────────────────────────────────────────

def load_miners(
    data_dir: str  = DATA_DIR,
    start:    str  = START_DATE,
    end:      str  = END_DATE,
    verbose:  bool = True,
) -> pd.DataFrame:
    """
    Loads Close prices for all 8 miners.
    Skips missing files with a warning rather than crashing.
    Returns a DataFrame: index=Date, one column per ticker.
    """
    frames = {}

    for ticker in RESERVES:
        filepath = os.path.join(data_dir, f"{ticker}.csv")
        try:
            series          = _load_csv(filepath, col="Close")
            series          = series.loc[start:end].rename(ticker)
            frames[ticker]  = series
            if verbose:
                print(f"  Loaded {ticker:<12}  "
                      f"rows={len(series):,}  "
                      f"{series.index[0].date()} → {series.index[-1].date()}")
        except FileNotFoundError as e:
            print(f"  [WARN] Skipping {ticker} — {e}")
        except ValueError as e:
            print(f"  [WARN] Skipping {ticker} — {e}")

    if not frames:
        raise RuntimeError(
            f"No miner CSV files loaded. Check DATA_DIR:\n    {data_dir}"
        )

    return pd.DataFrame(frames)


# ─────────────────────────────────────────────────────────────────
# STEP 5 — ALIGN EVERYTHING INTO ONE CLEAN DATAFRAME
# ─────────────────────────────────────────────────────────────────

def align_and_clean(
    gold_cny:     pd.Series,
    miner_prices: pd.DataFrame,
    ffill_limit:  int  = FFILL_LIMIT,
    min_history:  int  = MIN_HISTORY,
    verbose:      bool = True,
) -> pd.DataFrame:
    """
    Combines gold_cny + miner Close prices into one DataFrame.

    1. Outer-join on the date index (keeps all dates from any series)
    2. Forward-fill up to ffill_limit days — handles the mismatch between
       A-share trading days (SSE/SZSE) and COMEX trading days
    3. Drop any miner column with fewer than min_history valid rows
    4. Print a summary

    NOTE: rows are NOT dropped where values are NaN — each downstream
    layer handles missing data in its own way using pairwise-complete samples.
    """
    combined = pd.concat([gold_cny, miner_prices], axis=1)
    combined.index.name = "Date"
    combined.sort_index(inplace=True)

    # Forward-fill small gaps from differing holiday calendars
    combined = combined.ffill(limit=ffill_limit)

    # Drop miners with insufficient history
    miner_cols   = [c for c in combined.columns if c != "gold_cny"]
    valid_counts = combined[miner_cols].notna().sum()
    to_drop      = valid_counts[valid_counts < min_history].index.tolist()

    if to_drop:
        print(f"\n  [WARN] Dropping (insufficient history): {to_drop}")
        combined.drop(columns=to_drop, inplace=True)

    if verbose:
        surviving = [c for c in combined.columns if c != "gold_cny"]
        print(f"\n{'='*60}")
        print(f"  LAYER 1 — OUTPUT SUMMARY")
        print(f"{'='*60}")
        print(f"  Date range  : {combined.index[0].date()} → {combined.index[-1].date()}")
        print(f"  Total rows  : {len(combined):,}")
        print(f"  Columns     : gold_cny + {len(surviving)} miners")
        print(f"  Miners      : {surviving}")
        print(f"\n  NaN counts (should all be 0 or very low after ffill):")
        for col in combined.columns:
            n   = combined[col].isna().sum()
            pct = n / len(combined) * 100
            flag = "  ← check this" if pct > 5 else ""
            print(f"    {col:<15}  {n:>5} NaN  ({pct:.1f}%){flag}")
        print(f"{'='*60}\n")

    return combined


# ─────────────────────────────────────────────────────────────────
# MASTER FUNCTION — imported by Layer 2, 3, 4, 5
# ─────────────────────────────────────────────────────────────────

def load_all(
    data_dir:    str  = DATA_DIR,
    start:       str  = START_DATE,
    end:         str  = END_DATE,
    ffill_limit: int  = FFILL_LIMIT,
    min_history: int  = MIN_HISTORY,
    verbose:     bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Master entry point called by all downstream layers.

    Parameters
    ----------
    data_dir     : folder containing all CSV files
    start        : backtest start date  (YYYY-MM-DD)
    end          : backtest end date    (YYYY-MM-DD)
    ffill_limit  : max days to forward-fill across holiday gaps
    min_history  : minimum valid rows to keep a miner in the universe
    verbose      : print progress and summary

    Returns
    -------
    prices : pd.DataFrame
        index   = Date (DatetimeIndex, business-day frequency)
        col[0]  = 'gold_cny'   gold price in CNY/oz
        col[1+] = ticker cols  miner Close prices in CNY

    reserves : pd.DataFrame
        index   = ticker
        columns = name, exchange, reserves_moz, listed_year
    """
    print("=" * 60)
    print("  LAYER 1 — DATA INGESTION")
    print("=" * 60)

    gold_usd = load_gold(data_dir, start, end, verbose)
    usdcny   = load_fx(data_dir, start, end, verbose)
    gold_cny = convert_gold_to_cny(gold_usd, usdcny, verbose)
    miners   = load_miners(data_dir, start, end, verbose)
    prices   = align_and_clean(gold_cny, miners, ffill_limit, min_history, verbose)

    reserves = get_reserves_table()
    surviving = [c for c in prices.columns if c != "gold_cny"]
    reserves  = reserves.loc[reserves.index.isin(surviving)]

    return prices, reserves


# ─────────────────────────────────────────────────────────────────
# STANDALONE SANITY CHECK
# run:  python layer1_data.py
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prices, reserves = load_all()

    print("Reserves table:")
    print(reserves.to_string())

    print("\nFirst 3 rows:")
    print(prices.head(3).to_string())

    print("\nLast 3 rows:")
    print(prices.tail(3).to_string())

    # Save outputs so Layer 2 can load instantly without re-reading CSVs
    out_dir = r"C:\Gold ETF arbitrage\Layer 1_Data ingestion"
    prices.to_parquet(os.path.join(out_dir, "layer1_prices.parquet"))
    reserves.to_csv(os.path.join(out_dir, "layer1_reserves.csv"))
    print(f"\nSaved: layer1_prices.parquet + layer1_reserves.csv → {out_dir}")
