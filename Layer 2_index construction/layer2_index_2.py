"""
=============================================================
LAYER 2 — INDEX CONSTRUCTION
Gold / A-Share Miner Arbitrage Backtest
=============================================================

Inputs  : layer1_prices.parquet  +  layer1_reserves.csv
           (or call load_all() directly from layer1_data.py)

What this layer does
--------------------
1. Reserves-weighted miner index
       w_i = reserves_moz_i / sum(reserves_moz)
       index(t) = sum( w_i * price_i(t) / price_i(t0) ) * 100

2. Rebase gold to 100 at the same start date t0

3. Both series start at exactly 100 on t0 — this is the
   foundation for the spread / lag signal in Layer 3.

4. Saves:
       layer2_indices.parquet   — date-indexed DataFrame:
                                  gold_idx, miner_idx, + per-miner rebased cols
       layer2_weights.csv       — ticker weights used

Downstream layers import:
    from layer2_index import build_indices
"""

import os
import sys
sys.path.insert(0, r"C:\Gold ETF arbitrage\Layer 1_Data ingestion")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

L1_DIR  = r"C:\Gold ETF arbitrage\Layer 1_Data ingestion"
OUT_DIR = r"C:\Gold ETF arbitrage\Layer 2_Index construction"

# 7-stock universe (601069.SS dropped — insufficient history)
UNIVERSE = {
    "601899.SS": 53.0,   # Zijin Mining
    "600547.SS": 35.0,   # Shandong Gold
    "600489.SS": 12.0,   # Zhongjin Gold
    "000975.SZ":  8.0,   # Shanjin Intl Gold (ex-Yintai)
    "002155.SZ":  7.8,   # Hunan Gold
    "002273.SZ":  9.5,   # Chifeng Jilong Gold
    "002237.SZ":  4.0,   # Shandong Humon Smelting
}

REBASE_DATE = "2012-01-04"   # first trading day with full miner data


# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD LAYER 1 OUTPUT
# ─────────────────────────────────────────────────────────────────

def load_layer1(l1_dir: str = L1_DIR) -> pd.DataFrame:
    """
    Loads the parquet saved by Layer 1.
    Falls back to calling load_all() directly if parquet is missing.
    """
    parquet_path = os.path.join(l1_dir, "layer1_prices.parquet")

    if os.path.exists(parquet_path):
        prices = pd.read_parquet(parquet_path)
        print(f"  Loaded layer1_prices.parquet  "
              f"({len(prices):,} rows, {len(prices.columns)} cols)")
    else:
        print("  layer1_prices.parquet not found — running Layer 1 now ...")
        import sys
        sys.path.insert(0, r"C:\Gold ETF arbitrage\Layer 1_Data ingestion\stock_data_files")
        sys.path.insert(0, r"C:\Gold ETF arbitrage\Layer 1_Data ingestion")
        from layer1_data import load_all
        prices, _ = load_all(verbose=False)

    # Keep only tickers in our 7-stock universe + gold_cny
    keep = ["gold_cny"] + [t for t in UNIVERSE if t in prices.columns]
    missing = [t for t in UNIVERSE if t not in prices.columns]
    if missing:
        print(f"  [WARN] These universe tickers not found in Layer 1 data: {missing}")

    return prices[keep]


# ─────────────────────────────────────────────────────────────────
# STEP 2 — COMPUTE RESERVES WEIGHTS
# ─────────────────────────────────────────────────────────────────

def compute_weights(
    available_tickers: list[str],
    verbose: bool = True,
) -> pd.Series:
    """
    Normalises reserves_moz to sum-to-1 weights.
    Only includes tickers that are actually in the price data.

    Returns a pd.Series indexed by ticker.
    """
    raw = {t: UNIVERSE[t] for t in available_tickers if t in UNIVERSE}
    total = sum(raw.values())
    weights = pd.Series({t: v / total for t, v in raw.items()}, name="weight")

    if verbose:
        print(f"\n  Reserves weights (sum = {weights.sum():.4f}):")
        names = {
            "601899.SS": "Zijin Mining",
            "600547.SS": "Shandong Gold",
            "600489.SS": "Zhongjin Gold",
            "000975.SZ": "Shanjin Intl Gold",
            "002155.SZ": "Hunan Gold",
            "002273.SZ": "Chifeng Jilong Gold",
            "002237.SZ": "Shandong Humon Smelting",
        }
        for ticker, w in weights.sort_values(ascending=False).items():
            print(f"    {ticker}  {names.get(ticker, ticker):<28}  "
                  f"{UNIVERSE[ticker]:5.1f} Moz  →  {w*100:.2f}%")

    return weights


# ─────────────────────────────────────────────────────────────────
# STEP 3 — REBASE EACH SERIES TO 100 AT t0
# ─────────────────────────────────────────────────────────────────

def rebase(series: pd.Series, base_date: str = REBASE_DATE) -> pd.Series:
    """
    Divides every value by the value on base_date, then multiplies by 100.
    Handles the case where base_date is not in the index by using the
    next available date.
    """
    # Find the first valid date on or after base_date
    valid_idx = series.dropna()
    t0_candidates = valid_idx.index[valid_idx.index >= base_date]
    if len(t0_candidates) == 0:
        raise ValueError(f"No valid data on or after rebase date {base_date} for {series.name}")
    t0 = t0_candidates[0]
    t0_value = series.loc[t0]
    return (series / t0_value * 100).rename(series.name)


# ─────────────────────────────────────────────────────────────────
# STEP 4 — BUILD THE WEIGHTED MINER INDEX
# ─────────────────────────────────────────────────────────────────

def build_miner_index(
    prices:    pd.DataFrame,
    weights:   pd.Series,
    base_date: str = REBASE_DATE,
    verbose:   bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Constructs the reserves-weighted miner index.

    Method:
        1. Rebase each miner to 100 at base_date
        2. Weighted sum:  index(t) = sum( w_i * rebased_i(t) )
           This gives a level-100 index at t0 regardless of price scales.

    Returns
    -------
    miner_idx     : pd.Series    — the composite index
    rebased_miners: pd.DataFrame — individual rebased series (useful for Layer 5 charts)
    """
    miner_cols = [c for c in prices.columns if c != "gold_cny"]
    rebased    = pd.DataFrame(index=prices.index)

    for ticker in miner_cols:
        rebased[ticker] = rebase(prices[ticker], base_date)

    # Weighted sum — NaN rows in a constituent are excluded from that day's weight
    # by normalising weights on available tickers each day
    weighted_sum  = pd.Series(0.0, index=prices.index)
    weight_actual = pd.Series(0.0, index=prices.index)

    for ticker in miner_cols:
        w = weights.get(ticker, 0.0)
        mask = rebased[ticker].notna()
        weighted_sum[mask]  += rebased[ticker][mask] * w
        weight_actual[mask] += w

    # Re-normalise for days where some tickers were NaN
    miner_idx = (weighted_sum / weight_actual).rename("miner_idx")

    if verbose:
        first_valid = miner_idx.first_valid_index()
        print(f"\n  Miner index built:")
        print(f"    First valid date : {first_valid.date()}")
        print(f"    Value at t0      : {miner_idx.loc[first_valid]:.2f}  (should be ~100)")
        print(f"    Latest value     : {miner_idx.iloc[-1]:.2f}")
        print(f"    Return t0→latest : {(miner_idx.iloc[-1]/100 - 1)*100:.1f}%")

    return miner_idx, rebased


# ─────────────────────────────────────────────────────────────────
# STEP 5 — REBASE GOLD TO 100
# ─────────────────────────────────────────────────────────────────

def build_gold_index(
    prices:    pd.DataFrame,
    base_date: str  = REBASE_DATE,
    verbose:   bool = True,
) -> pd.Series:
    gold_idx = rebase(prices["gold_cny"], base_date).rename("gold_idx")

    if verbose:
        first_valid = gold_idx.first_valid_index()
        print(f"\n  Gold index built:")
        print(f"    First valid date : {first_valid.date()}")
        print(f"    Value at t0      : {gold_idx.loc[first_valid]:.2f}  (should be 100)")
        print(f"    Latest value     : {gold_idx.iloc[-1]:.2f}")
        print(f"    Return t0→latest : {(gold_idx.iloc[-1]/100 - 1)*100:.1f}%")

    return gold_idx


# ─────────────────────────────────────────────────────────────────
# STEP 6 — ASSEMBLE FINAL OUTPUT DATAFRAME
# ─────────────────────────────────────────────────────────────────

def assemble(
    gold_idx:      pd.Series,
    miner_idx:     pd.Series,
    rebased_miners: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combines gold_idx, miner_idx, and per-miner rebased series
    into one DataFrame. Rows before both indices are valid are dropped.
    """
    out = pd.concat([gold_idx, miner_idx, rebased_miners], axis=1)

    # Drop rows where either main index is NaN
    out = out.dropna(subset=["gold_idx", "miner_idx"])

    return out


# ─────────────────────────────────────────────────────────────────
# STEP 7 — PLOT  (optional but useful for a quick sanity check)
# ─────────────────────────────────────────────────────────────────

def plot_indices(df: pd.DataFrame, out_dir: str = OUT_DIR) -> None:
    """
    Two-panel chart:
      Top    : gold_idx vs miner_idx rebased to 100
      Bottom : spread = gold_idx - miner_idx
    """
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )
    fig.suptitle("Layer 2 — Rebased Indices (base = 100)", fontsize=13)

    # Top panel
    ax1.plot(df.index, df["gold_idx"],  label="Gold (GC=F, CNY)",         color="#D4AF37", lw=1.8)
    ax1.plot(df.index, df["miner_idx"], label="Miner Index (reserves-wtd)", color="#2196F3", lw=1.8)
    ax1.axhline(100, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax1.set_ylabel("Index Level (base = 100)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bottom panel — spread
    spread = df["gold_idx"] - df["miner_idx"]
    ax2.fill_between(df.index, spread, 0,
                     where=spread >= 0, color="#D4AF37", alpha=0.35, label="Gold ahead")
    ax2.fill_between(df.index, spread, 0,
                     where=spread < 0,  color="#2196F3", alpha=0.35, label="Miners ahead")
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_ylabel("Spread (gold − miners)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    save_path = os.path.join(out_dir, "layer2_indices.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved → {save_path}")


# ─────────────────────────────────────────────────────────────────
# MASTER FUNCTION — called by Layer 3+
# ─────────────────────────────────────────────────────────────────

def build_indices(
    l1_dir:    str  = L1_DIR,
    out_dir:   str  = OUT_DIR,
    base_date: str  = REBASE_DATE,
    save:      bool = True,
    plot:      bool = True,
    verbose:   bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Master entry point for downstream layers.

    Returns
    -------
    df : pd.DataFrame
        index   = Date
        columns = gold_idx, miner_idx, 601899.SS, 600547.SS, ... (rebased)

    weights : pd.Series
        index   = ticker
        values  = float  (sum to 1.0)
    """
    print("=" * 60)
    print("  LAYER 2 — INDEX CONSTRUCTION")
    print("=" * 60)

    prices   = load_layer1(l1_dir)
    miner_cols = [c for c in prices.columns if c != "gold_cny"]
    weights  = compute_weights(miner_cols, verbose)
    gold_idx = build_gold_index(prices, base_date, verbose)
    miner_idx, rebased = build_miner_index(prices, weights, base_date, verbose)
    df       = assemble(gold_idx, miner_idx, rebased)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  LAYER 2 — OUTPUT SUMMARY")
        print(f"{'='*60}")
        print(f"  Rows        : {len(df):,}")
        print(f"  Date range  : {df.index[0].date()} → {df.index[-1].date()}")
        print(f"  Columns     : {list(df.columns)}")
        spread = df["gold_idx"] - df["miner_idx"]
        print(f"\n  Spread stats (gold_idx - miner_idx):")
        print(f"    Mean   : {spread.mean():+.2f} pts")
        print(f"    Std    : {spread.std():.2f} pts")
        print(f"    Min    : {spread.min():+.2f} pts  ({spread.idxmin().date()})")
        print(f"    Max    : {spread.max():+.2f} pts  ({spread.idxmax().date()})")
        print(f"{'='*60}\n")

    if save:
        os.makedirs(out_dir, exist_ok=True)
        df.to_parquet(os.path.join(out_dir, "layer2_indices.parquet"))
        weights.to_csv(os.path.join(out_dir, "layer2_weights.csv"), header=True)
        print(f"  Saved: layer2_indices.parquet + layer2_weights.csv → {out_dir}")

    if plot:
        plot_indices(df, out_dir)

    return df, weights


# ─────────────────────────────────────────────────────────────────
# STANDALONE — python layer2_index.py
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df, weights = build_indices()

    print("First 3 rows:")
    print(df[["gold_idx", "miner_idx"]].head(3).to_string())

    print("\nLast 3 rows:")
    print(df[["gold_idx", "miner_idx"]].tail(3).to_string())
