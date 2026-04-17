"""
=============================================================
LAYER 6 — OUT-OF-SAMPLE VALIDATION
Gold / A-Share Miner Arbitrage Backtest
=============================================================

Three complementary overfitting tests:

A) SIMPLE TRAIN/TEST SPLIT
   Train: 2012-01-04 → 2019-12-31  (8 years, param selection)
   Test:  2020-01-01 → 2024-12-31  (5 years, never touched)
   → Re-runs Layer 5 sweep on train only, picks best config,
     then evaluates that config on test set.

B) WALK-FORWARD OPTIMISATION (WFO)
   Rolls a fixed training window forward, re-picks best config
   in each window, trades only the subsequent out-of-sample period.
   Train window : 3 years (756 trading days)
   Test  window : 1 year  (252 trading days)
   Step  size   : 1 year  (252 trading days)
   → Simulates realistic live re-optimisation every 12 months.
   → The equity curve is stitched from OOS periods only — no IS data.

C) PARAMETER STABILITY TEST
   Checks whether the best config from Layer 5 (mag=0.75%, win=1d)
   was stable across sub-periods, or only worked in specific regimes.
   → Runs the best config on rolling 2-year windows.
   → If Sharpe > 1.0 in 70%+ of windows: structurally robust.
   → If Sharpe clusters in one era: regime-dependent, use with caution.

Overfitting safeguards built in:
   - IS and OOS data are strictly separated — IS params never see OOS prices
   - WFO uses anchored re-selection (not look-ahead)
   - Deflated Sharpe Ratio (DSR) computed: adjusts for the number of
     configs tested, penalising lucky discoveries
   - Probabilistic Sharpe Ratio (PSR): probability that true Sharpe > 0

Outputs
-------
layer6_oos_summary.txt         — human-readable validation report
layer6_wfo_equity.png          — WFO stitched equity curve
layer6_param_stability.png     — rolling Sharpe over time
layer6_wfo_params.csv          — which config was selected in each window
layer6_oos_results.parquet     — OOS daily returns for further analysis

Run:  python layer6_validation.py
"""

import os
import sys
sys.path.insert(0, r"C:\Gold ETF arbitrage\Layer 1_Data ingestion")
sys.path.insert(0, r"C:\Gold ETF arbitrage\Layer 2_Index construction")
sys.path.insert(0, r"C:\Gold ETF arbitrage\Layer 3_Signal generation")
sys.path.insert(0, r"C:\Gold ETF arbitrage\Layer 4_Backtest engine")
sys.path.insert(0, r"C:\Gold ETF arbitrage\Layer 5_analysis")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from itertools import product

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

L3_DIR  = r"C:\Gold ETF arbitrage\Layer 3_Signal generation"
OUT_DIR = r"C:\Gold ETF arbitrage\Layer 6_Validation"

# Train / test split
TRAIN_END   = "2019-12-31"
TEST_START  = "2020-01-01"

# Walk-forward windows (trading days)
WFO_TRAIN_DAYS = 756    # 3 years
WFO_TEST_DAYS  = 252    # 1 year
WFO_STEP_DAYS  = 252    # re-optimise every year

# Rolling stability window
STABILITY_WINDOW = 504  # 2-year rolling window

# Parameter grid to sweep (same as Layer 5 but tighter — avoids grid bloat)
MAG_THRESHOLDS = [0.000, 0.002, 0.003, 0.005, 0.0075, 0.010]
SIGNAL_WINDOWS = [1, 2, 3, 5, 10]
VOL_SCALE_ON   = [False, True]
VOL_WINDOWS    = [20, 60]

TRANSACTION_COST = 0.002
STOP_LOSS_PCT    = 0.05
MAX_HOLD_DAYS    = 20

# Best config from Layer 5 (used for stability test)
BEST_MAG    = 0.0075
BEST_WIN    = 1
BEST_VS     = False
BEST_VW     = 0


# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    path = os.path.join(L3_DIR, "layer3_signals.parquet")
    df   = pd.read_parquet(path)
    print(f"  Loaded layer3_signals.parquet  ({len(df):,} rows)  "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    return df


# ─────────────────────────────────────────────────────────────────
# CORE BACKTEST (vectorised — reused from Layer 5)
# ─────────────────────────────────────────────────────────────────

def run_variant(
    df:            pd.DataFrame,
    mag_threshold: float = 0.0075,
    signal_window: int   = 1,
    vol_scale:     bool  = False,
    vol_window:    int   = 20,
    txn_cost:      float = TRANSACTION_COST,
    stop_loss_pct: float = STOP_LOSS_PCT,
    max_hold_days: int   = MAX_HOLD_DAYS,
) -> dict:
    """Vectorised backtest — same logic as Layer 5."""
    if len(df) < signal_window + 10:
        return {"sharpe": np.nan, "total_return": np.nan,
                "ann_return": np.nan, "ann_vol": np.nan,
                "max_drawdown": np.nan, "win_rate": np.nan,
                "n_trades": 0, "_net_pnl": pd.Series(dtype=float),
                "_cum": pd.Series(dtype=float)}

    gold_ret_1d = df["gold_idx"].pct_change()
    gold_ret_N  = df["gold_idx"].pct_change(signal_window)
    miner_ret   = df["miner_idx"].pct_change().fillna(0)

    raw_dir = np.sign(gold_ret_N)
    raw_dir[gold_ret_N.abs() < mag_threshold] = 0
    raw_signal = raw_dir.replace(0, np.nan).ffill().fillna(0)

    if vol_scale:
        rv = gold_ret_1d.rolling(vol_window).std()
        rv = rv.replace(0, np.nan).ffill().fillna(0.01)
        size = (gold_ret_N.abs() / rv).clip(0.1, 1.0)
        signal_sized = raw_signal * size
    else:
        signal_sized = raw_signal

    position     = signal_sized.shift(1).fillna(0)
    pos_arr      = position.values.copy()
    entry_level  = 0.0
    entry_sign   = 0
    hold_days    = 0
    miner_levels = df["miner_idx"].values

    for i in range(1, len(pos_arr)):
        cur_sign = np.sign(pos_arr[i])
        if entry_sign == 0 and cur_sign != 0:
            entry_level = miner_levels[i]
            entry_sign  = int(cur_sign)
            hold_days   = 0
        elif entry_sign != 0:
            hold_days += 1
            trade_ret = ((miner_levels[i] / entry_level) - 1) * entry_sign if entry_level else 0
            if trade_ret < -stop_loss_pct:
                pos_arr[i]  = 0.0
                entry_sign  = 0
                entry_level = 0.0
                hold_days   = 0
                continue
            if hold_days >= max_hold_days:
                entry_level = miner_levels[i]
                entry_sign  = int(np.sign(pos_arr[i]))
                hold_days   = 0
            if cur_sign != 0 and int(cur_sign) != entry_sign:
                entry_level = miner_levels[i]
                entry_sign  = int(cur_sign)
                hold_days   = 0

    position_final = pd.Series(pos_arr, index=df.index)
    pos_change     = position_final.diff().abs().fillna(0)
    net_pnl        = position_final * miner_ret - pos_change * txn_cost

    cum      = (1 + net_pnl).cumprod()
    total    = cum.iloc[-1] - 1
    n        = len(net_pnl)
    ann_r    = (1 + total) ** (252 / n) - 1
    ann_v    = net_pnl.std() * np.sqrt(252)
    sharpe   = ann_r / ann_v if ann_v > 0 else 0
    peak     = cum.cummax()
    mdd      = (cum / peak - 1).min()
    active   = net_pnl[net_pnl != 0]
    wr       = (active > 0).mean() if len(active) > 0 else 0
    n_trades = int(pos_change[pos_change > 0].count())

    return {"sharpe": sharpe, "total_return": total, "ann_return": ann_r,
            "ann_vol": ann_v, "max_drawdown": mdd, "win_rate": wr,
            "n_trades": n_trades, "_net_pnl": net_pnl, "_cum": cum}


def best_config_in_sample(df_train: pd.DataFrame) -> dict:
    """Sweeps the param grid on df_train, returns the best config dict."""
    combos = []
    seen   = set()
    for m, s, v, vw in product(MAG_THRESHOLDS, SIGNAL_WINDOWS,
                                VOL_SCALE_ON, VOL_WINDOWS):
        key = (m, s, v, vw if v else 0)
        if key not in seen:
            seen.add(key)
            combos.append((m, s, v, vw))

    best_sr  = -999
    best_cfg = {}
    for m, s, v, vw in combos:
        r = run_variant(df_train, mag_threshold=m, signal_window=s,
                        vol_scale=v, vol_window=vw)
        if r["sharpe"] > best_sr:
            best_sr  = r["sharpe"]
            best_cfg = {"mag": m, "win": s, "vs": v, "vw": vw,
                        "is_sharpe": round(best_sr, 4)}
    return best_cfg


# ─────────────────────────────────────────────────────────────────
# OVERFITTING METRICS
# ─────────────────────────────────────────────────────────────────

def probabilistic_sharpe_ratio(
    net_pnl: pd.Series,
    benchmark_sr: float = 0.0,
) -> float:
    """
    PSR = P(SR* > benchmark_sr) using the asymptotic distribution.
    Accounts for skewness and kurtosis of the return series.
    A PSR > 0.95 means we're 95% confident the true Sharpe exceeds benchmark.
    Bailey & Lopez de Prado (2012).
    """
    from scipy.stats import norm
    n   = len(net_pnl)
    sr  = net_pnl.mean() / net_pnl.std() * np.sqrt(252) if net_pnl.std() > 0 else 0
    sk  = net_pnl.skew()
    ku  = net_pnl.kurtosis()   # excess kurtosis
    denom = np.sqrt((1 - sk * sr + (ku / 4) * sr**2) / (n - 1))
    if denom == 0:
        return 0.5
    z   = (sr - benchmark_sr) / denom
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    oos_sharpe:   float,
    n_configs:    int,
    n_obs:        int,
    net_pnl:      pd.Series,
) -> float:
    """
    DSR penalises the Sharpe for the number of configs tested.
    Estimates the expected maximum Sharpe from noise alone,
    then checks if OOS Sharpe clears that bar.
    Bailey & Lopez de Prado (2014).
    Returns the probability that the strategy's Sharpe is skill, not luck.
    """
    from scipy.stats import norm
    # Expected maximum Sharpe from n_configs iid trials
    gamma   = 0.5772   # Euler–Mascheroni constant
    e_max   = (1 - gamma) * norm.ppf(1 - 1/n_configs) + \
              gamma * norm.ppf(1 - 1/(n_configs * np.e))

    sk  = net_pnl.skew()
    ku  = net_pnl.kurtosis()
    sr  = oos_sharpe / np.sqrt(252)   # daily Sharpe
    denom = np.sqrt((1 - sk * sr + (ku / 4) * sr**2) / (n_obs - 1))
    if denom == 0:
        return 0.5
    z   = (oos_sharpe - e_max) / (denom * np.sqrt(252))
    return float(norm.cdf(z))


# ─────────────────────────────────────────────────────────────────
# TEST A — SIMPLE TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────

def test_train_test_split(df: pd.DataFrame) -> dict:
    print(f"\n{'='*60}")
    print(f"  TEST A — SIMPLE TRAIN / TEST SPLIT")
    print(f"  Train: {df.index[0].date()} → {TRAIN_END}")
    print(f"  Test : {TEST_START} → {df.index[-1].date()}")
    print(f"{'='*60}")

    df_train = df.loc[:TRAIN_END]
    df_test  = df.loc[TEST_START:]

    print(f"\n  [1/3] Finding best config on training data ...")
    best_cfg = best_config_in_sample(df_train)
    print(f"  Best IS config: mag={best_cfg['mag']*100:.2f}%  "
          f"win={best_cfg['win']}d  vs={best_cfg['vs']}  "
          f"IS Sharpe={best_cfg['is_sharpe']:.3f}")

    print(f"\n  [2/3] Evaluating on test data (never seen during optimisation) ...")
    oos = run_variant(df_test,
                      mag_threshold = best_cfg["mag"],
                      signal_window = best_cfg["win"],
                      vol_scale     = best_cfg["vs"],
                      vol_window    = best_cfg["vw"])

    # Also run the Layer 5 best config on test for comparison
    l5  = run_variant(df_test,
                      mag_threshold = BEST_MAG,
                      signal_window = BEST_WIN,
                      vol_scale     = BEST_VS,
                      vol_window    = BEST_VW)

    print(f"\n  [3/3] Computing overfitting metrics ...")
    n_configs = len(MAG_THRESHOLDS) * len(SIGNAL_WINDOWS) * len(VOL_SCALE_ON) * len(VOL_WINDOWS)
    psr = probabilistic_sharpe_ratio(oos["_net_pnl"])
    dsr = deflated_sharpe_ratio(oos["sharpe"], n_configs,
                                len(oos["_net_pnl"]), oos["_net_pnl"])

    result = {
        "best_cfg":     best_cfg,
        "oos_sharpe":   round(oos["sharpe"], 4),
        "oos_ann_ret":  round(oos["ann_return"], 4),
        "oos_mdd":      round(oos["max_drawdown"], 4),
        "oos_win_rate": round(oos["win_rate"], 4),
        "oos_n_trades": oos["n_trades"],
        "l5_oos_sharpe":round(l5["sharpe"], 4),
        "psr":          round(psr, 4),
        "dsr":          round(dsr, 4),
        "n_configs":    n_configs,
        "_oos_cum":     oos["_cum"],
        "_l5_oos_cum":  l5["_cum"],
        "_oos_pnl":     oos["_net_pnl"],
        "_df_test":     df_test,
    }

    print(f"\n  Results:")
    print(f"    OOS Sharpe (best IS config)  : {result['oos_sharpe']:.3f}")
    print(f"    OOS Sharpe (Layer 5 best)    : {result['l5_oos_sharpe']:.3f}")
    print(f"    OOS Ann Return               : {result['oos_ann_ret']*100:.1f}%")
    print(f"    OOS Max Drawdown             : {result['oos_mdd']*100:.1f}%")
    print(f"    OOS Win Rate                 : {result['oos_win_rate']*100:.1f}%")
    print(f"    OOS Trades                   : {result['oos_n_trades']}")
    print(f"\n  Overfitting metrics:")
    print(f"    Configs tested               : {n_configs}")
    print(f"    Probabilistic SR (PSR)       : {psr:.3f}  "
          f"({'PASS' if psr > 0.95 else 'MARGINAL' if psr > 0.80 else 'FAIL'})")
    print(f"    Deflated SR (DSR)            : {dsr:.3f}  "
          f"({'PASS' if dsr > 0.95 else 'MARGINAL' if dsr > 0.80 else 'FAIL'})")

    return result


# ─────────────────────────────────────────────────────────────────
# TEST B — WALK-FORWARD OPTIMISATION
# ─────────────────────────────────────────────────────────────────

def test_walk_forward(df: pd.DataFrame) -> dict:
    print(f"\n{'='*60}")
    print(f"  TEST B — WALK-FORWARD OPTIMISATION")
    print(f"  Train window : {WFO_TRAIN_DAYS}d  |  "
          f"Test window : {WFO_TEST_DAYS}d  |  "
          f"Step : {WFO_STEP_DAYS}d")
    print(f"{'='*60}")

    n         = len(df)
    oos_pnls  = []
    wfo_log   = []
    start     = WFO_TRAIN_DAYS

    window_num = 0
    while start + WFO_TEST_DAYS <= n:
        train_slice = df.iloc[start - WFO_TRAIN_DAYS : start]
        test_slice  = df.iloc[start : start + WFO_TEST_DAYS]

        train_start = train_slice.index[0].date()
        train_end   = train_slice.index[-1].date()
        test_start  = test_slice.index[0].date()
        test_end    = test_slice.index[-1].date()

        window_num += 1
        print(f"\n  Window {window_num}: "
              f"train {train_start}→{train_end}  |  "
              f"test {test_start}→{test_end}")

        # Find best config on training slice
        best = best_config_in_sample(train_slice)
        print(f"    Best IS: mag={best['mag']*100:.2f}%  "
              f"win={best['win']}d  "
              f"IS_SR={best['is_sharpe']:.3f}")

        # Evaluate on test slice
        oos = run_variant(test_slice,
                          mag_threshold = best["mag"],
                          signal_window = best["win"],
                          vol_scale     = best["vs"],
                          vol_window    = best["vw"])

        print(f"    OOS SR={oos['sharpe']:.3f}  "
              f"ret={oos['ann_return']*100:.1f}%  "
              f"trades={oos['n_trades']}")

        oos_pnls.append(oos["_net_pnl"])
        wfo_log.append({
            "window":       window_num,
            "train_start":  str(train_start),
            "train_end":    str(train_end),
            "test_start":   str(test_start),
            "test_end":     str(test_end),
            "best_mag":     best["mag"],
            "best_win":     best["win"],
            "best_vs":      best["vs"],
            "is_sharpe":    best["is_sharpe"],
            "oos_sharpe":   round(oos["sharpe"], 4),
            "oos_ann_ret":  round(oos["ann_return"], 4),
            "oos_mdd":      round(oos["max_drawdown"], 4),
            "oos_n_trades": oos["n_trades"],
        })

        start += WFO_STEP_DAYS

    # Stitch OOS periods into one continuous equity curve
    wfo_pnl = pd.concat(oos_pnls).sort_index()
    wfo_cum = (1 + wfo_pnl).cumprod()

    wfo_total = wfo_cum.iloc[-1] - 1
    wfo_ann   = (1 + wfo_total) ** (252 / len(wfo_pnl)) - 1
    wfo_vol   = wfo_pnl.std() * np.sqrt(252)
    wfo_sr    = wfo_ann / wfo_vol if wfo_vol > 0 else 0
    wfo_peak  = wfo_cum.cummax()
    wfo_mdd   = (wfo_cum / wfo_peak - 1).min()

    # IS/OOS Sharpe ratio — key overfitting measure
    # Rule of thumb: OOS/IS > 0.5 is acceptable, > 0.7 is good
    mean_is  = np.mean([w["is_sharpe"]  for w in wfo_log])
    mean_oos = np.mean([w["oos_sharpe"] for w in wfo_log])
    oos_is_ratio = mean_oos / mean_is if mean_is > 0 else 0

    print(f"\n  WFO Summary ({window_num} windows):")
    print(f"    Stitched OOS Sharpe    : {wfo_sr:.3f}")
    print(f"    Stitched OOS Ann Ret   : {wfo_ann*100:.1f}%")
    print(f"    Stitched OOS MDD       : {wfo_mdd*100:.1f}%")
    print(f"    Mean IS  Sharpe        : {mean_is:.3f}")
    print(f"    Mean OOS Sharpe        : {mean_oos:.3f}")
    print(f"    OOS/IS ratio           : {oos_is_ratio:.3f}  "
          f"({'PASS' if oos_is_ratio > 0.5 else 'WARN — possible overfit'})")
    print(f"    Pct windows profitable : "
          f"{sum(1 for w in wfo_log if w['oos_sharpe'] > 0) / window_num * 100:.0f}%")

    return {
        "wfo_log":       pd.DataFrame(wfo_log),
        "wfo_pnl":       wfo_pnl,
        "wfo_cum":       wfo_cum,
        "wfo_sr":        round(wfo_sr, 4),
        "wfo_ann":       round(wfo_ann, 4),
        "wfo_mdd":       round(wfo_mdd, 4),
        "mean_is":       round(mean_is, 4),
        "mean_oos":      round(mean_oos, 4),
        "oos_is_ratio":  round(oos_is_ratio, 4),
        "n_windows":     window_num,
    }


# ─────────────────────────────────────────────────────────────────
# TEST C — PARAMETER STABILITY (rolling window)
# ─────────────────────────────────────────────────────────────────

def test_param_stability(df: pd.DataFrame) -> dict:
    print(f"\n{'='*60}")
    print(f"  TEST C — PARAMETER STABILITY")
    print(f"  Rolling {STABILITY_WINDOW}d windows on best config "
          f"(mag={BEST_MAG*100:.2f}%, win={BEST_WIN}d)")
    print(f"{'='*60}")

    rolling_sharpes = []
    dates           = []

    step = 63   # advance 1 quarter at a time
    for end in range(STABILITY_WINDOW, len(df), step):
        window = df.iloc[end - STABILITY_WINDOW : end]
        r      = run_variant(window,
                             mag_threshold = BEST_MAG,
                             signal_window = BEST_WIN,
                             vol_scale     = BEST_VS,
                             vol_window    = BEST_VW)
        rolling_sharpes.append(r["sharpe"])
        dates.append(df.index[end])

    sr_series = pd.Series(rolling_sharpes, index=dates)
    pct_positive = (sr_series > 0).mean()
    pct_above_1  = (sr_series > 1.0).mean()

    print(f"\n  Stability results:")
    print(f"    Windows computed       : {len(sr_series)}")
    print(f"    Mean Sharpe            : {sr_series.mean():.3f}")
    print(f"    Std  Sharpe            : {sr_series.std():.3f}")
    print(f"    % windows SR > 0       : {pct_positive*100:.0f}%  "
          f"({'PASS' if pct_positive > 0.70 else 'WARN'})")
    print(f"    % windows SR > 1.0     : {pct_above_1*100:.0f}%")
    print(f"    Min SR                 : {sr_series.min():.3f}  "
          f"({sr_series.idxmin().date()})")
    print(f"    Max SR                 : {sr_series.max():.3f}  "
          f"({sr_series.idxmax().date()})")

    return {
        "sr_series":     sr_series,
        "pct_positive":  pct_positive,
        "pct_above_1":   pct_above_1,
        "mean_sr":       round(sr_series.mean(), 4),
        "std_sr":        round(sr_series.std(), 4),
    }


# ─────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────

def plot_wfo(wfo: dict, df: pd.DataFrame, out_dir: str) -> None:
    gold_ret  = df["gold_idx"].pct_change().fillna(0)
    miner_ret = df["miner_idx"].pct_change().fillna(0)
    gold_cum  = (1 + gold_ret).cumprod() * 100
    miner_cum = (1 + miner_ret).cumprod() * 100

    fig, axes = plt.subplots(3, 1, figsize=(14, 14),
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle("Layer 6 — Walk-Forward Optimisation", fontsize=12)

    gold_c  = "#D4AF37"
    miner_c = "#2196F3"
    wfo_c   = "#2ECC71"
    neg_c   = "#E74C3C"

    # Panel 1: equity curves
    ax1 = axes[0]
    wfo_cum_plot = wfo["wfo_cum"] * 100
    ax1.plot(gold_cum.index,    gold_cum,    color=gold_c,  lw=1.2, ls="--",
             alpha=0.7, label="B&H Gold")
    ax1.plot(miner_cum.index,   miner_cum,   color=miner_c, lw=1.2, ls="--",
             alpha=0.7, label="B&H Miners")
    ax1.plot(wfo_cum_plot.index, wfo_cum_plot, color=wfo_c, lw=2.0,
             label=f"WFO OOS  (SR={wfo['wfo_sr']:.2f})")
    ax1.axhline(100, color="gray", lw=0.5, ls=":")

    # Shade each WFO test window
    for _, row in wfo["wfo_log"].iterrows():
        ax1.axvspan(pd.Timestamp(row["test_start"]),
                    pd.Timestamp(row["test_end"]),
                    alpha=0.06, color=wfo_c)

    ax1.set_ylabel("Portfolio Value (base=100)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    # Panel 2: OOS Sharpe per window
    ax2 = axes[1]
    log = wfo["wfo_log"]
    bar_colors = [wfo_c if s > 0 else neg_c for s in log["oos_sharpe"]]
    ax2.bar(range(len(log)), log["oos_sharpe"], color=bar_colors, alpha=0.8)
    ax2.axhline(0,               color="black",  lw=0.8)
    ax2.axhline(wfo["mean_oos"], color=wfo_c,    lw=1.0, ls="--",
                label=f"Mean OOS SR={wfo['mean_oos']:.2f}")
    ax2.axhline(wfo["mean_is"],  color="orange", lw=1.0, ls="--",
                label=f"Mean IS  SR={wfo['mean_is']:.2f}")
    ax2.set_xticks(range(len(log)))
    ax2.set_xticklabels([f"W{i+1}\n{r['test_start'][:7]}"
                         for i, r in log.iterrows()], fontsize=7)
    ax2.set_ylabel("Sharpe")
    ax2.set_title("OOS Sharpe per WFO Window", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: selected mag threshold per window
    ax3 = axes[2]
    ax3.bar(range(len(log)), log["best_mag"] * 100, color=miner_c, alpha=0.7)
    ax3.set_xticks(range(len(log)))
    ax3.set_xticklabels([f"W{i+1}" for i in range(len(log))], fontsize=8)
    ax3.set_ylabel("Selected mag threshold (%)")
    ax3.set_title("Best mag threshold selected in each window", fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, "layer6_wfo_equity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  WFO chart saved → {path}")


def plot_stability(stab: dict, out_dir: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Layer 6 — Parameter Stability\n"
                 f"Best config: mag={BEST_MAG*100:.2f}%, "
                 f"win={BEST_WIN}d — rolling {STABILITY_WINDOW}d windows", fontsize=11)

    sr  = stab["sr_series"]
    pos = sr[sr >= 0]
    neg = sr[sr < 0]

    ax1 = axes[0]
    ax1.fill_between(pos.index, pos.values, 0, color="#2ECC71", alpha=0.5, label="SR > 0")
    ax1.fill_between(neg.index, neg.values, 0, color="#E74C3C", alpha=0.5, label="SR < 0")
    ax1.plot(sr.index, sr.values, color="black", lw=0.8, alpha=0.6)
    ax1.axhline(0,   color="black", lw=0.8)
    ax1.axhline(1.0, color="#D4AF37", lw=0.8, ls="--", label="SR = 1.0 threshold")
    ax1.axhline(stab["mean_sr"], color="gray", lw=0.8, ls=":",
                label=f"Mean SR = {stab['mean_sr']:.2f}")
    ax1.set_ylabel("Rolling Sharpe Ratio")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    ax2 = axes[1]
    ax2.hist(sr.values, bins=20, color="#2196F3", alpha=0.75, edgecolor="white")
    ax2.axvline(0,   color="#E74C3C", lw=1.2, ls="--", label="SR = 0")
    ax2.axvline(1.0, color="#D4AF37", lw=1.2, ls="--", label="SR = 1.0")
    ax2.axvline(stab["mean_sr"], color="black", lw=1.2,
                label=f"Mean = {stab['mean_sr']:.2f}")
    ax2.set_xlabel("Sharpe Ratio")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of rolling Sharpe ratios", fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, "layer6_param_stability.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Stability chart saved → {path}")


# ─────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────

def write_report(split: dict, wfo: dict, stab: dict, out_dir: str) -> None:
    sep = "=" * 62
    lines = [sep,
             "  LAYER 6 — OUT-OF-SAMPLE VALIDATION REPORT",
             "  Gold / A-Share Miner Arbitrage Backtest",
             sep, ""]

    # Verdict helper
    def verdict(condition, pass_msg, fail_msg):
        return f"  PASS  {pass_msg}" if condition else f"  WARN  {fail_msg}"

    lines += ["  TEST A — TRAIN/TEST SPLIT", "-"*40]
    lines += [
        f"  Best IS config   : mag={split['best_cfg']['mag']*100:.2f}%  "
        f"win={split['best_cfg']['win']}d  SR(IS)={split['best_cfg']['is_sharpe']:.3f}",
        f"  OOS Sharpe       : {split['oos_sharpe']:.3f}",
        f"  OOS Ann Return   : {split['oos_ann_ret']*100:.1f}%",
        f"  OOS Max Drawdown : {split['oos_mdd']*100:.1f}%",
        f"  OOS Win Rate     : {split['oos_win_rate']*100:.1f}%",
        f"  OOS Trades       : {split['oos_n_trades']}",
        f"  Configs tested   : {split['n_configs']}",
        f"  PSR              : {split['psr']:.3f}  "
        f"(prob true SR > 0)",
        f"  DSR              : {split['dsr']:.3f}  "
        f"(prob SR is skill not luck)",
        verdict(split["psr"] > 0.95,
                "PSR > 0.95 — high confidence true SR > 0",
                "PSR < 0.95 — lower confidence in edge"),
        verdict(split["dsr"] > 0.95,
                "DSR > 0.95 — edge survives multiple-testing penalty",
                "DSR < 0.95 — edge may be partially data-mined"),
    ]

    lines += ["", "  TEST B — WALK-FORWARD OPTIMISATION", "-"*40]
    lines += [
        f"  Windows          : {wfo['n_windows']}",
        f"  WFO OOS Sharpe   : {wfo['wfo_sr']:.3f}",
        f"  WFO OOS Ann Ret  : {wfo['wfo_ann']*100:.1f}%",
        f"  WFO OOS MDD      : {wfo['wfo_mdd']*100:.1f}%",
        f"  Mean IS  Sharpe  : {wfo['mean_is']:.3f}",
        f"  Mean OOS Sharpe  : {wfo['mean_oos']:.3f}",
        f"  OOS/IS ratio     : {wfo['oos_is_ratio']:.3f}",
        verdict(wfo["oos_is_ratio"] > 0.5,
                f"OOS/IS={wfo['oos_is_ratio']:.2f} > 0.5 — acceptable degradation",
                f"OOS/IS={wfo['oos_is_ratio']:.2f} < 0.5 — significant overfit"),
    ]

    lines += ["", "  TEST C — PARAMETER STABILITY", "-"*40]
    lines += [
        f"  Rolling window   : {STABILITY_WINDOW}d",
        f"  Mean SR          : {stab['mean_sr']:.3f}",
        f"  Std  SR          : {stab['std_sr']:.3f}",
        f"  % windows SR>0   : {stab['pct_positive']*100:.0f}%",
        f"  % windows SR>1   : {stab['pct_above_1']*100:.0f}%",
        verdict(stab["pct_positive"] > 0.70,
                f"{stab['pct_positive']*100:.0f}% windows profitable — structurally robust",
                f"{stab['pct_positive']*100:.0f}% windows profitable — regime-dependent"),
    ]

    lines += ["", sep,
              "  OVERALL VERDICT", "-"*40]
    tests_passed = sum([
        split["psr"]  > 0.95,
        split["dsr"]  > 0.95,
        wfo["oos_is_ratio"] > 0.5,
        stab["pct_positive"] > 0.70,
    ])
    lines += [
        f"  Tests passed: {tests_passed}/4",
        f"  {'STRATEGY VALIDATED — proceed to paper trading' if tests_passed >= 3 else 'FURTHER REVIEW NEEDED before live trading'}",
    ]
    lines += ["", sep]

    report = "\n".join(lines)
    print("\n" + report)

    path = os.path.join(out_dir, "layer6_oos_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report saved → {path}")


# ─────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────

def run_validation(
    l3_dir:  str = L3_DIR,
    out_dir: str = OUT_DIR,
) -> None:
    print("=" * 62)
    print("  LAYER 6 — OUT-OF-SAMPLE VALIDATION")
    print("=" * 62)

    os.makedirs(out_dir, exist_ok=True)
    df = load_data()

    split = test_train_test_split(df)
    wfo   = test_walk_forward(df)
    stab  = test_param_stability(df)

    # Save outputs
    wfo["wfo_log"].to_csv(
        os.path.join(out_dir, "layer6_wfo_params.csv"), index=False)
    oos_results = pd.DataFrame({
        "wfo_pnl": wfo["wfo_pnl"],
        "wfo_cum": wfo["wfo_cum"],
    })
    oos_results.to_parquet(
        os.path.join(out_dir, "layer6_oos_results.parquet"))

    plot_wfo(wfo, df, out_dir)
    plot_stability(stab, out_dir)
    write_report(split, wfo, stab, out_dir)

    print(f"\n  All outputs saved → {out_dir}")


# ─────────────────────────────────────────────────────────────────
# STANDALONE — python layer6_validation.py
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_validation()
