"""
=============================================================
LAYER 3 — SIGNAL GENERATION
Gold / A-Share Miner Arbitrage Backtest
=============================================================

Two complementary signal methods:

A) RETURN CROSS-CORRELATION  — measures the empirical lag in days
   - Computes corr( gold_ret(t), miner_ret(t + lag) ) for lag = -10..+10
   - Positive best_lag → miners lag gold (gold leads)
   - Also computed on a rolling basis to see if the lag is stable over time

B) ROLLING Z-SCORE OF SPREAD  — generates the actual entry/exit signal
   - raw_spread(t)   = gold_idx(t) - miner_idx(t)          [levels]
   - z_score(t)      = (raw_spread - rolling_mean) / rolling_std
   - De-trends the spread so it oscillates around 0 regardless of
     the long-run divergence between gold and miners
   - Entry long  miners : z > +ENTRY_Z   (gold ran ahead of miners)
   - Entry short miners : z < -ENTRY_Z   (miners ran ahead of gold)
   - Exit                : |z| < EXIT_Z

Outputs
-------
layer3_signals.parquet   — date-indexed DataFrame with all signal columns
layer3_crosscorr.csv     — lag vs correlation table (full sample)
layer3_signals.png       — 4-panel diagnostic chart

Downstream layers import:
    from layer3_signal import build_signals
"""

import os
import sys
sys.path.insert(0, r"C:\Gold ETF arbitrage\Layer 1_Data ingestion")
sys.path.insert(0, r"C:\Gold ETF arbitrage\Layer 2_Index construction")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

L2_DIR  = r"C:\Gold ETF arbitrage\Layer 2_Index construction"
OUT_DIR = r"C:\Gold ETF arbitrage\Layer 3_Signal generation"

# Cross-correlation
MAX_LAG    = 10          # test lags from -10 to +10 days
ROLLING_CORR_WINDOW = 252  # 1-year rolling window for lag stability

# Z-score signal
ZSCORE_WINDOW = 60       # rolling window (days) for mean/std of spread
ENTRY_Z       = 1.5      # open position when |z| crosses this
EXIT_Z        = 0.5      # close position when |z| falls below this

# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD LAYER 2 OUTPUT
# ─────────────────────────────────────────────────────────────────

def load_layer2(l2_dir: str = L2_DIR) -> pd.DataFrame:
    parquet_path = os.path.join(l2_dir, "layer2_indices.parquet")

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        print(f"  Loaded layer2_indices.parquet  ({len(df):,} rows)")
    else:
        print("  layer2_indices.parquet not found — running Layer 2 now ...")
        from layer2_index import build_indices  # type: ignore
        df, _ = build_indices(verbose=False)

    return df


# ─────────────────────────────────────────────────────────────────
# STEP 2A — FULL-SAMPLE CROSS-CORRELATION
# ─────────────────────────────────────────────────────────────────

def cross_correlation(
    gold_idx:  pd.Series,
    miner_idx: pd.Series,
    max_lag:   int  = MAX_LAG,
    verbose:   bool = True,
) -> pd.DataFrame:
    """
    Computes Pearson correlation between gold daily returns
    and miner daily returns shifted by 'lag' days.

    lag > 0 : gold_ret(t) vs miner_ret(t + lag)
              → miners REACT to gold with a lag of 'lag' days
    lag < 0 : miners lead gold

    Returns a DataFrame: columns = [lag, correlation]
    """
    gold_ret  = gold_idx.pct_change().dropna()
    miner_ret = miner_idx.pct_change().dropna()

    # Align on common dates
    common = gold_ret.index.intersection(miner_ret.index)
    g = gold_ret.loc[common]
    m = miner_ret.loc[common]

    results = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            # gold leads: align gold[:-lag] with miner[lag:]
            corr = g.iloc[:-lag].values
            base = m.iloc[lag:].values
        elif lag < 0:
            # miners lead: align miner[:-|lag|] with gold[|lag|:]
            corr = g.iloc[-lag:].values
            base = m.iloc[:lag].values
        else:
            corr = g.values
            base = m.values

        r = np.corrcoef(corr, base)[0, 1]
        results.append({"lag": lag, "correlation": r})

    df_corr = pd.DataFrame(results)
    best    = df_corr.loc[df_corr["correlation"].idxmax()]

    if verbose:
        print(f"\n  Full-sample cross-correlation  (window: entire history)")
        print(f"  {'Lag':>6}   {'Correlation':>12}")
        for _, row in df_corr.iterrows():
            marker = "  ← BEST" if row["lag"] == best["lag"] else ""
            print(f"  {int(row['lag']):>6}   {row['correlation']:>12.4f}{marker}")
        print(f"\n  Best lag : {int(best['lag'])} day(s)  "
              f"(corr = {best['correlation']:.4f})")
        if best["lag"] > 0:
            print(f"  Interpretation: miners lag gold by ~{int(best['lag'])} trading day(s)")
        elif best["lag"] < 0:
            print(f"  Interpretation: miners LEAD gold by ~{abs(int(best['lag']))} trading day(s)")
        else:
            print(f"  Interpretation: no lag — miners and gold move simultaneously")

    return df_corr


# ─────────────────────────────────────────────────────────────────
# STEP 2B — ROLLING CROSS-CORRELATION  (lag stability over time)
# ─────────────────────────────────────────────────────────────────

def rolling_best_lag(
    gold_idx:  pd.Series,
    miner_idx: pd.Series,
    window:    int = ROLLING_CORR_WINDOW,
    max_lag:   int = MAX_LAG,
) -> pd.Series:
    """
    For each rolling window, finds the lag with the highest correlation.
    Returns a Series of best_lag values over time.
    Useful to check whether the lag is structurally stable or regime-dependent.
    """
    gold_ret  = gold_idx.pct_change()
    miner_ret = miner_idx.pct_change()

    best_lags = []
    dates     = []

    for end in range(window, len(gold_ret)):
        g = gold_ret.iloc[end - window : end].dropna()
        m = miner_ret.iloc[end - window : end].dropna()
        common = g.index.intersection(m.index)
        g, m   = g.loc[common], m.loc[common]

        best_r   = -999
        best_lag = 0
        for lag in range(-max_lag, max_lag + 1):
            if lag > 0 and len(g) > lag:
                r = np.corrcoef(g.iloc[:-lag].values, m.iloc[lag:].values)[0, 1]
            elif lag < 0 and len(g) > abs(lag):
                r = np.corrcoef(g.iloc[-lag:].values, m.iloc[:lag].values)[0, 1]
            else:
                r = np.corrcoef(g.values, m.values)[0, 1]

            if r > best_r:
                best_r   = r
                best_lag = lag

        best_lags.append(best_lag)
        dates.append(gold_ret.index[end])

    return pd.Series(best_lags, index=dates, name="rolling_best_lag")


# ─────────────────────────────────────────────────────────────────
# STEP 3 — ROLLING Z-SCORE SIGNAL
# ─────────────────────────────────────────────────────────────────

def build_zscore_signal(
    gold_idx:  pd.Series,
    miner_idx: pd.Series,
    window:    int  = ZSCORE_WINDOW,
    entry_z:   float = ENTRY_Z,
    exit_z:    float = EXIT_Z,
    verbose:   bool  = True,
) -> pd.DataFrame:
    """
    Computes the rolling z-score of the spread and derives entry/exit signals.

    Columns returned:
        raw_spread   : gold_idx - miner_idx
        rolling_mean : rolling mean of raw_spread  (window days)
        rolling_std  : rolling std  of raw_spread  (window days)
        z_score      : (raw_spread - rolling_mean) / rolling_std
        signal       : +1 (long miners), -1 (short miners), 0 (flat)

    Signal logic:
        z > +entry_z  → long  miners  (gold ran ahead, miners to catch up)
        z < -entry_z  → short miners  (miners ran ahead, expect mean-revert)
        |z| < exit_z  → close position
    """
    raw_spread   = (gold_idx - miner_idx).rename("raw_spread")
    rolling_mean = raw_spread.rolling(window).mean().rename("rolling_mean")
    rolling_std  = raw_spread.rolling(window).std().rename("rolling_std")
    z_score      = ((raw_spread - rolling_mean) / rolling_std).rename("z_score")

    # State-machine signal generation
    signal = pd.Series(0, index=z_score.index, name="signal", dtype=int)
    pos    = 0

    for i in range(1, len(z_score)):
        z = z_score.iloc[i]
        if np.isnan(z):
            signal.iloc[i] = 0
            pos = 0
            continue

        if pos == 0:
            if z > entry_z:
                pos = 1     # long miners
            elif z < -entry_z:
                pos = -1    # short miners
        else:
            if abs(z) < exit_z:
                pos = 0     # exit

        signal.iloc[i] = pos

    # Assemble
    sig_df = pd.concat([raw_spread, rolling_mean, rolling_std, z_score, signal], axis=1)

    if verbose:
        n_long  = (signal ==  1).sum()
        n_short = (signal == -1).sum()
        n_flat  = (signal ==  0).sum()
        total   = len(signal)
        print(f"\n  Z-score signal  (window={window}d, entry=±{entry_z}, exit=±{exit_z})")
        print(f"    Long  days : {n_long:,}  ({n_long/total*100:.1f}%)")
        print(f"    Short days : {n_short:,}  ({n_short/total*100:.1f}%)")
        print(f"    Flat  days : {n_flat:,}  ({n_flat/total*100:.1f}%)")

        # Count round-trip trades
        trades = signal.diff().abs()
        n_trades = int((trades > 0).sum() / 2)
        print(f"    Round-trip trades : ~{n_trades}")

        print(f"\n  Z-score stats:")
        print(f"    Mean   : {z_score.mean():+.4f}  (should be ~0)")
        print(f"    Std    : {z_score.std():.4f}   (should be ~1)")
        print(f"    Min    : {z_score.min():+.4f}  ({z_score.idxmin().date()})")
        print(f"    Max    : {z_score.max():+.4f}  ({z_score.idxmax().date()})")

    return sig_df


# ─────────────────────────────────────────────────────────────────
# STEP 4 — PLOT DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────

def plot_signals(
    idx_df:     pd.DataFrame,
    sig_df:     pd.DataFrame,
    corr_df:    pd.DataFrame,
    roll_lag:   pd.Series,
    out_dir:    str = OUT_DIR,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 14))
    gs  = GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    gold_color  = "#D4AF37"
    miner_color = "#2196F3"
    long_color  = "#2ECC71"
    short_color = "#E74C3C"

    # ── Panel 1 (top full-width): rebased indices ─────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(idx_df.index, idx_df["gold_idx"],  color=gold_color,  lw=1.6, label="Gold index")
    ax1.plot(idx_df.index, idx_df["miner_idx"], color=miner_color, lw=1.6, label="Miner index")
    ax1.set_title("Rebased Indices (base = 100)", fontsize=10)
    ax1.set_ylabel("Index Level")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2 (2nd row full-width): z-score + signals ──────────
    ax2 = fig.add_subplot(gs[1, :])
    z   = sig_df["z_score"]
    ax2.plot(sig_df.index, z, color="gray", lw=0.9, alpha=0.85)
    ax2.fill_between(sig_df.index, z, ENTRY_Z,
                     where=z >= ENTRY_Z,  color=long_color,  alpha=0.3, label=f"Long signal (z>{ENTRY_Z})")
    ax2.fill_between(sig_df.index, z, -ENTRY_Z,
                     where=z <= -ENTRY_Z, color=short_color, alpha=0.3, label=f"Short signal (z<-{ENTRY_Z})")
    ax2.axhline( ENTRY_Z, color=long_color,  lw=0.8, ls="--")
    ax2.axhline(-ENTRY_Z, color=short_color, lw=0.8, ls="--")
    ax2.axhline( EXIT_Z,  color="gray",      lw=0.5, ls=":")
    ax2.axhline(-EXIT_Z,  color="gray",      lw=0.5, ls=":")
    ax2.axhline(0,        color="black",     lw=0.6)
    ax2.set_title(f"Rolling Z-Score of Spread  (window={ZSCORE_WINDOW}d)", fontsize=10)
    ax2.set_ylabel("Z-Score")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)

    # ── Panel 3 (bottom-left): full-sample cross-correlation ──────
    ax3 = fig.add_subplot(gs[2, 0])
    best_lag = corr_df.loc[corr_df["correlation"].idxmax(), "lag"]
    bar_colors = [gold_color if l == best_lag else "#AAAAAA" for l in corr_df["lag"]]
    ax3.bar(corr_df["lag"], corr_df["correlation"], color=bar_colors, width=0.7, alpha=0.85)
    ax3.axvline(0, color="black", lw=0.7, ls="--")
    ax3.set_title("Full-Sample Cross-Correlation\n(+lag = miners lag gold)", fontsize=10)
    ax3.set_xlabel("Lag (days)")
    ax3.set_ylabel("Pearson Correlation")
    ax3.grid(True, alpha=0.3, axis="y")

    # ── Panel 4 (bottom-right): rolling best lag over time ────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(roll_lag.index, roll_lag.values, color=miner_color, lw=1.0, alpha=0.8)
    ax4.axhline(0, color="black", lw=0.7, ls="--")
    ax4.fill_between(roll_lag.index, roll_lag.values, 0,
                     where=roll_lag.values > 0, color=gold_color,  alpha=0.25, label="Miners lag")
    ax4.fill_between(roll_lag.index, roll_lag.values, 0,
                     where=roll_lag.values < 0, color=miner_color, alpha=0.25, label="Miners lead")
    ax4.set_title(f"Rolling Best Lag Over Time\n(window={ROLLING_CORR_WINDOW}d)", fontsize=10)
    ax4.set_ylabel("Best Lag (days)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── Panel 5 (bottom full-width): position timeline ────────────
    ax5 = fig.add_subplot(gs[3, :])
    pos = sig_df["signal"]
    ax5.fill_between(pos.index, pos.values,  0,
                     where=pos.values > 0,  color=long_color,  alpha=0.6, label="Long miners")
    ax5.fill_between(pos.index, pos.values, 0,
                     where=pos.values < 0, color=short_color, alpha=0.6, label="Short miners")
    ax5.axhline(0, color="black", lw=0.6)
    ax5.set_yticks([-1, 0, 1])
    ax5.set_yticklabels(["Short", "Flat", "Long"], fontsize=8)
    ax5.set_title("Position Signal Over Time", fontsize=10)
    ax5.legend(fontsize=8, loc="upper left")
    ax5.grid(True, alpha=0.2)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax5.xaxis.set_major_locator(mdates.YearLocator())

    fig.suptitle("Layer 3 — Signal Generation", fontsize=13, y=1.01)

    save_path = os.path.join(out_dir, "layer3_signals.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved → {save_path}")


# ─────────────────────────────────────────────────────────────────
# MASTER FUNCTION — called by Layer 4+
# ─────────────────────────────────────────────────────────────────

def build_signals(
    l2_dir:        str   = L2_DIR,
    out_dir:       str   = OUT_DIR,
    zscore_window: int   = ZSCORE_WINDOW,
    entry_z:       float = ENTRY_Z,
    exit_z:        float = EXIT_Z,
    max_lag:       int   = MAX_LAG,
    save:          bool  = True,
    plot:          bool  = True,
    verbose:       bool  = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Master entry point for downstream layers.

    Returns
    -------
    signals  : pd.DataFrame
        index   = Date
        columns = raw_spread, rolling_mean, rolling_std, z_score, signal

    idx_df   : pd.DataFrame
        index   = Date
        columns = gold_idx, miner_idx  (+ per-miner rebased cols)

    corr_df  : pd.DataFrame
        columns = lag, correlation  (full-sample cross-corr table)
    """
    print("=" * 60)
    print("  LAYER 3 — SIGNAL GENERATION")
    print("=" * 60)

    idx_df   = load_layer2(l2_dir)
    gold_idx  = idx_df["gold_idx"]
    miner_idx = idx_df["miner_idx"]

    # Cross-correlation
    print("\n  [A] Cross-correlation analysis ...")
    corr_df  = cross_correlation(gold_idx, miner_idx, max_lag, verbose)

    print("\n  [B] Rolling best-lag stability ...")
    roll_lag = rolling_best_lag(gold_idx, miner_idx,
                                window=ROLLING_CORR_WINDOW, max_lag=max_lag)
    mode_lag = int(roll_lag.mode().iloc[0])
    print(f"      Most common rolling best lag: {mode_lag} day(s)")

    # Z-score signal
    print("\n  [C] Z-score signal ...")
    signals  = build_zscore_signal(
                    gold_idx, miner_idx,
                    window=zscore_window,
                    entry_z=entry_z,
                    exit_z=exit_z,
                    verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  LAYER 3 — OUTPUT SUMMARY")
        print(f"{'='*60}")
        print(f"  Rows        : {len(signals):,}")
        print(f"  Date range  : {signals.index[0].date()} → {signals.index[-1].date()}")
        print(f"  Best lag    : {int(corr_df.loc[corr_df['correlation'].idxmax(), 'lag'])} day(s)  (full sample)")
        print(f"  Mode lag    : {mode_lag} day(s)  (rolling)")
        print(f"  Signal col  : 'signal'  values = {{-1, 0, +1}}")
        print(f"{'='*60}\n")

    if save:
        os.makedirs(out_dir, exist_ok=True)
        # Merge signals with index levels for convenient downstream access
        out = pd.concat([idx_df[["gold_idx", "miner_idx"]], signals], axis=1)
        out.to_parquet(os.path.join(out_dir, "layer3_signals.parquet"))
        corr_df.to_csv(os.path.join(out_dir, "layer3_crosscorr.csv"), index=False)
        print(f"  Saved: layer3_signals.parquet + layer3_crosscorr.csv → {out_dir}")

    if plot:
        plot_signals(idx_df, signals, corr_df, roll_lag, out_dir)

    return signals, idx_df, corr_df


# ─────────────────────────────────────────────────────────────────
# STANDALONE — python layer3_signal.py
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    signals, idx_df, corr_df = build_signals()

    print("Signal tail (last 10 rows):")
    print(signals[["z_score", "signal"]].tail(10).to_string())
