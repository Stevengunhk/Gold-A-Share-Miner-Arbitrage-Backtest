"""
=============================================================
LAYER 5 — ANALYSIS & SENSITIVITY SWEEP
Gold / A-Share Miner Arbitrage Backtest
=============================================================

Three enhancements tested systematically:

A) MAGNITUDE FILTER
   Only trade when |gold_ret(T)| > threshold
   Tests: 0.0% (no filter), 0.2%, 0.3%, 0.5%, 0.75%, 1.0%

B) MULTI-DAY SIGNAL
   Use N-day rolling gold return instead of 1-day
   Tests: 1d, 2d, 3d, 5d, 10d

C) VOL-SCALED POSITION SIZING  (replaces circuit breaker)
   position_size = clip( |gold_ret| / rolling_vol(N) , 0, 1 )
   Tests: on / off
   rolling_vol windows: 10d, 20d, 60d

Full grid sweep: 6 × 5 × 2 × 3 = 180 combinations
Best configs ranked by Sharpe ratio.

Outputs
-------
layer5_sweep_results.csv    — full grid results
layer5_best_configs.csv     — top 10 by Sharpe
layer5_sweep_heatmap.png    — Sharpe heatmap: magnitude vs signal window
layer5_best_equity.png      — equity curves: top 5 configs vs benchmarks
layer5_report.txt           — human-readable summary report

Run:  python layer5_analysis.py
"""

import os
import sys
sys.path.insert(0, r"C:\Gold ETF arbitrage\Layer 1_Data ingestion")
sys.path.insert(0, r"C:\Gold ETF arbitrage\Layer 2_Index construction")
sys.path.insert(0, r"C:\Gold ETF arbitrage\Layer 3_Signal generation")
sys.path.insert(0, r"C:\Gold ETF arbitrage\Layer 4_Backtest engine")

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
L4_DIR  = r"C:\Gold ETF arbitrage\Layer 4_Backtest engine"
OUT_DIR = r"C:\Gold ETF arbitrage\Layer 5_Analysis"

TRANSACTION_COST = 0.002    # keep constant across all sweep runs
STOP_LOSS_PCT    = 0.05
MAX_HOLD_DAYS    = 20

# Sweep grid
MAG_THRESHOLDS   = [0.000, 0.002, 0.003, 0.005, 0.0075, 0.010]  # |gold_ret| filter
SIGNAL_WINDOWS   = [1, 2, 3, 5, 10]                               # N-day gold return window
VOL_SCALE_ON     = [False, True]                                   # vol-scaled sizing
VOL_WINDOWS      = [10, 20, 60]                                    # rolling vol window (days)


# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    parquet_path = os.path.join(L3_DIR, "layer3_signals.parquet")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        print(f"  Loaded layer3_signals.parquet  ({len(df):,} rows)")
    else:
        raise FileNotFoundError(
            f"layer3_signals.parquet not found in {L3_DIR}\n"
            "Run layer3_signal.py first."
        )
    return df


# ─────────────────────────────────────────────────────────────────
# STEP 2 — SINGLE BACKTEST RUNNER  (fast, no state machine overhead)
# ─────────────────────────────────────────────────────────────────

def run_variant(
    df:              pd.DataFrame,
    mag_threshold:   float = 0.003,
    signal_window:   int   = 1,
    vol_scale:       bool  = False,
    vol_window:      int   = 20,
    txn_cost:        float = TRANSACTION_COST,
    stop_loss_pct:   float = STOP_LOSS_PCT,
    max_hold_days:   int   = MAX_HOLD_DAYS,
) -> dict:
    """
    Vectorised single-pass backtest for one parameter combination.
    Returns a dict of performance metrics.

    Signal construction:
        gold_ret_N  = gold_idx.pct_change(signal_window)
        raw_signal  = sign(gold_ret_N)   if |gold_ret_N| >= mag_threshold else 0
        position    = raw_signal (float if vol-scaled, int otherwise)

    Vol scaling:
        vol(t)      = gold_ret_1d.rolling(vol_window).std()
        size(t)     = clip( |gold_ret_N(t)| / vol(t) , 0.1, 1.0 )
        position(t) = raw_signal(t) * size(t)

    T+1 execution:
        actual_position(t) = position(t-1)   [yesterday's signal]

    P&L:
        gross(t) = actual_position(t) * miner_ret(t)
        cost(t)  = |actual_position(t) - actual_position(t-1)| * txn_cost
        net(t)   = gross(t) - cost(t)

    Stop-loss / max-hold applied per-trade via a lightweight loop.
    """
    gold_ret_1d = df["gold_idx"].pct_change()
    gold_ret_N  = df["gold_idx"].pct_change(signal_window)
    miner_ret   = df["miner_idx"].pct_change().fillna(0)

    # ── Raw directional signal ─────────────────────────────────
    raw_dir = np.sign(gold_ret_N)
    # Zero out signals below magnitude threshold
    below_threshold         = gold_ret_N.abs() < mag_threshold
    raw_dir[below_threshold] = 0
    # Forward-fill through zeros so we keep last valid direction
    # but only when it was zeroed by threshold (not a genuine flat)
    raw_signal = raw_dir.replace(0, np.nan).ffill().fillna(0)

    # ── Vol scaling ────────────────────────────────────────────
    if vol_scale:
        rolling_vol = gold_ret_1d.rolling(vol_window).std()
        # Avoid division by zero
        rolling_vol = rolling_vol.replace(0, np.nan).ffill().fillna(0.01)
        size = (gold_ret_N.abs() / rolling_vol).clip(0.1, 1.0)
        signal_sized = raw_signal * size
    else:
        signal_sized = raw_signal

    # ── T+1 shift: use yesterday's signal ──────────────────────
    position = signal_sized.shift(1).fillna(0)

    # ── Apply stop-loss and max-hold via loop ──────────────────
    pos_arr       = position.values.copy()
    entry_level   = 0.0
    entry_sign    = 0
    hold_days     = 0
    miner_levels  = df["miner_idx"].values

    for i in range(1, len(pos_arr)):
        cur_sign = np.sign(pos_arr[i])

        # New trade entry
        if entry_sign == 0 and cur_sign != 0:
            entry_level = miner_levels[i]
            entry_sign  = int(cur_sign)
            hold_days   = 0

        elif entry_sign != 0:
            hold_days += 1
            trade_ret = (miner_levels[i] / entry_level - 1) * entry_sign if entry_level else 0

            # Stop-loss: zero out position
            if trade_ret < -stop_loss_pct:
                pos_arr[i]  = 0.0
                entry_sign  = 0
                entry_level = 0.0
                hold_days   = 0
                continue

            # Max hold: carry signal but reset entry reference
            if hold_days >= max_hold_days:
                entry_level = miner_levels[i]
                entry_sign  = int(np.sign(pos_arr[i]))
                hold_days   = 0

            # Direction flip
            if cur_sign != 0 and int(cur_sign) != entry_sign:
                entry_level = miner_levels[i]
                entry_sign  = int(cur_sign)
                hold_days   = 0

    position_final = pd.Series(pos_arr, index=df.index)

    # ── P&L ────────────────────────────────────────────────────
    pos_change   = position_final.diff().abs().fillna(0)
    gross_pnl    = position_final * miner_ret
    cost_pnl     = pos_change * txn_cost
    net_pnl      = gross_pnl - cost_pnl

    cum          = (1 + net_pnl).cumprod()
    total_ret    = cum.iloc[-1] - 1
    ann_ret      = (1 + total_ret) ** (252 / len(net_pnl)) - 1
    ann_vol      = net_pnl.std() * np.sqrt(252)
    sharpe       = ann_ret / ann_vol if ann_vol > 0 else 0
    peak         = cum.cummax()
    mdd          = (cum / peak - 1).min()
    active       = net_pnl[net_pnl != 0]
    win_rate     = (active > 0).mean() if len(active) > 0 else 0
    n_trades     = int(pos_change[pos_change > 0].count())
    total_cost   = cost_pnl.sum()

    return {
        "mag_threshold":  mag_threshold,
        "signal_window":  signal_window,
        "vol_scale":      vol_scale,
        "vol_window":     vol_window if vol_scale else 0,
        "total_return":   round(total_ret, 4),
        "ann_return":     round(ann_ret,   4),
        "ann_vol":        round(ann_vol,   4),
        "sharpe":         round(sharpe,    4),
        "max_drawdown":   round(mdd,       4),
        "win_rate":       round(win_rate,  4),
        "n_trades":       n_trades,
        "total_cost_pct": round(total_cost * 100, 2),
        "_net_pnl":       net_pnl,     # kept for equity curve plotting
        "_cum":           cum,
    }


# ─────────────────────────────────────────────────────────────────
# STEP 3 — FULL GRID SWEEP
# ─────────────────────────────────────────────────────────────────

def run_sweep(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Runs all parameter combinations and returns a results DataFrame.
    """
    combos = list(product(
        MAG_THRESHOLDS,
        SIGNAL_WINDOWS,
        VOL_SCALE_ON,
        VOL_WINDOWS,
    ))

    # De-duplicate: when vol_scale=False, vol_window doesn't matter
    seen    = set()
    combos2 = []
    for m, s, v, vw in combos:
        key = (m, s, v, vw if v else 0)
        if key not in seen:
            seen.add(key)
            combos2.append((m, s, v, vw))

    total = len(combos2)
    print(f"\n  Running {total} parameter combinations ...")

    rows = []
    for idx, (m, s, v, vw) in enumerate(combos2):
        if verbose and (idx % 20 == 0):
            print(f"    {idx+1}/{total}  mag={m:.3f}  win={s}d  "
                  f"vol_scale={v}  vol_win={vw}")
        result = run_variant(df, mag_threshold=m, signal_window=s,
                             vol_scale=v, vol_window=vw)
        rows.append({k: v2 for k, v2 in result.items()
                     if not k.startswith("_")})

    sweep_df = pd.DataFrame(rows)
    sweep_df.sort_values("sharpe", ascending=False, inplace=True)
    sweep_df.reset_index(drop=True, inplace=True)

    print(f"\n  Sweep complete.")
    print(f"  Best Sharpe : {sweep_df['sharpe'].max():.3f}  "
          f"(mag={sweep_df.iloc[0]['mag_threshold']:.3f}  "
          f"win={int(sweep_df.iloc[0]['signal_window'])}d  "
          f"vol_scale={sweep_df.iloc[0]['vol_scale']})")
    print(f"  Worst Sharpe: {sweep_df['sharpe'].min():.3f}")

    return sweep_df


# ─────────────────────────────────────────────────────────────────
# STEP 4 — BENCHMARK METRICS
# ─────────────────────────────────────────────────────────────────

def benchmark_metrics(df: pd.DataFrame) -> dict:
    gold_ret  = df["gold_idx"].pct_change().fillna(0)
    miner_ret = df["miner_idx"].pct_change().fillna(0)

    def m(ret):
        total = (1 + ret).prod() - 1
        ann   = (1 + total) ** (252 / len(ret)) - 1
        vol   = ret.std() * np.sqrt(252)
        sh    = ann / vol if vol > 0 else 0
        pk    = (1 + ret).cumprod().cummax()
        mdd   = ((1 + ret).cumprod() / pk - 1).min()
        return {"total": total, "ann": ann, "vol": vol,
                "sharpe": sh, "mdd": mdd}

    return {"gold": m(gold_ret), "miners": m(miner_ret)}


# ─────────────────────────────────────────────────────────────────
# STEP 5 — PLOTS
# ─────────────────────────────────────────────────────────────────

def plot_heatmap(sweep_df: pd.DataFrame, out_dir: str) -> None:
    """
    Sharpe heatmap: rows = magnitude threshold, cols = signal window.
    One panel for vol_scale=False, one for vol_scale=True (best vol_window).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Layer 5 — Sharpe Ratio Heatmap\n"
                 "(magnitude threshold × signal window)", fontsize=12)

    for ax, vs in zip(axes, [False, True]):
        sub = sweep_df[sweep_df["vol_scale"] == vs].copy()
        if vs:
            # Keep best vol_window per (mag, signal) pair
            sub = (sub.sort_values("sharpe", ascending=False)
                      .drop_duplicates(["mag_threshold", "signal_window"]))

        pivot = sub.pivot_table(
            index="mag_threshold", columns="signal_window",
            values="sharpe", aggfunc="max"
        )
        pivot.index = [f"{v*100:.1f}%" for v in pivot.index]

        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                       vmin=-0.5, vmax=1.0)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c}d" for c in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_xlabel("Signal window (N-day gold return)", fontsize=9)
        ax.set_ylabel("|gold_ret| filter threshold", fontsize=9)
        ax.set_title(f"Vol scaling: {'ON' if vs else 'OFF'}", fontsize=10)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if abs(val) > 0.4 else "black")

        plt.colorbar(im, ax=ax, shrink=0.8, label="Sharpe")

    plt.tight_layout()
    path = os.path.join(out_dir, "layer5_sweep_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved → {path}")


def plot_best_equity(df: pd.DataFrame, sweep_df: pd.DataFrame,
                     benchmarks: dict, out_dir: str, top_n: int = 5) -> None:
    """
    Equity curves for the top N configs vs both benchmarks.
    """
    gold_cum  = (1 + df["gold_idx"].pct_change().fillna(0)).cumprod() * 100
    miner_cum = (1 + df["miner_idx"].pct_change().fillna(0)).cumprod() * 100

    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"Layer 5 — Top {top_n} Configs vs Benchmarks", fontsize=12)

    colors = ["#2ECC71", "#3498DB", "#9B59B6", "#E67E22", "#1ABC9C"]

    ax1, ax2 = axes
    ax1.plot(df.index, gold_cum,  color="#D4AF37", lw=1.2, alpha=0.7,
             ls="--", label="B&H Gold")
    ax1.plot(df.index, miner_cum, color="#2196F3", lw=1.2, alpha=0.7,
             ls="--", label="B&H Miners")
    ax1.axhline(100, color="gray", lw=0.5, ls=":")

    sharpe_vals = []

    for rank, (_, row) in enumerate(sweep_df.head(top_n).iterrows()):
        result = run_variant(
            df,
            mag_threshold = row["mag_threshold"],
            signal_window = int(row["signal_window"]),
            vol_scale     = row["vol_scale"],
            vol_window    = int(row["vol_window"]) if row["vol_scale"] else 20,
        )
        cum = result["_cum"] * 100
        label = (f"#{rank+1}  mag={row['mag_threshold']*100:.1f}%  "
                 f"win={int(row['signal_window'])}d  "
                 f"vs={'on' if row['vol_scale'] else 'off'}  "
                 f"SR={row['sharpe']:.2f}")
        ax1.plot(df.index, cum, color=colors[rank], lw=1.6, label=label)
        sharpe_vals.append(row["sharpe"])

    ax1.set_ylabel("Portfolio Value (base=100)")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    # Bottom panel: Sharpe bar chart for top configs
    labels_bar = [f"#{i+1}" for i in range(len(sharpe_vals))]
    bar_colors = ["#2ECC71" if s > 0 else "#E74C3C" for s in sharpe_vals]
    ax2.bar(labels_bar, sharpe_vals, color=bar_colors, alpha=0.8)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.axhline(benchmarks["gold"]["sharpe"],   color="#D4AF37", lw=1.0,
                ls="--", label=f"B&H Gold SR={benchmarks['gold']['sharpe']:.2f}")
    ax2.axhline(benchmarks["miners"]["sharpe"], color="#2196F3", lw=1.0,
                ls="--", label=f"B&H Miners SR={benchmarks['miners']['sharpe']:.2f}")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Sharpe: top configs vs benchmarks")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, "layer5_best_equity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Equity chart saved → {path}")


# ─────────────────────────────────────────────────────────────────
# STEP 6 — TEXT REPORT
# ─────────────────────────────────────────────────────────────────

def write_report(sweep_df: pd.DataFrame,
                 benchmarks: dict,
                 out_dir: str) -> None:
    lines = []
    sep   = "=" * 62

    lines += [sep,
              "  LAYER 5 — SENSITIVITY SWEEP REPORT",
              f"  Gold / A-Share Miner Arbitrage Backtest",
              sep, ""]

    lines += ["  BENCHMARKS",
              f"  {'':30} {'Total':>8} {'Ann Ret':>8} {'Sharpe':>8} {'MDD':>8}",
              f"  {'-'*60}"]
    for name, bm in benchmarks.items():
        lines.append(
            f"  {'B&H '+name.capitalize():<30} "
            f"{bm['total']*100:>7.1f}%"
            f"{bm['ann']*100:>8.1f}%"
            f"{bm['sharpe']:>8.2f}"
            f"{bm['mdd']*100:>7.1f}%"
        )

    lines += ["", "  TOP 10 CONFIGURATIONS (by Sharpe)",
              f"  {'Rank':>4} {'Mag%':>5} {'Win':>4} {'VS':>3} "
              f"{'VW':>3} {'Total':>7} {'Ann%':>6} "
              f"{'SR':>6} {'MDD%':>6} {'WR%':>5} {'Trades':>7}",
              f"  {'-'*60}"]

    for i, row in sweep_df.head(10).iterrows():
        lines.append(
            f"  {i+1:>4} "
            f"{row['mag_threshold']*100:>4.1f}%"
            f" {int(row['signal_window']):>3}d"
            f" {'Y' if row['vol_scale'] else 'N':>3}"
            f" {int(row['vol_window']) if row['vol_scale'] else 0:>3}"
            f" {row['total_return']*100:>6.1f}%"
            f" {row['ann_return']*100:>5.1f}%"
            f" {row['sharpe']:>6.2f}"
            f" {row['max_drawdown']*100:>5.1f}%"
            f" {row['win_rate']*100:>4.1f}%"
            f" {int(row['n_trades']):>7}"
        )

    lines += ["",
              "  DIMENSION ANALYSIS",
              f"  {'-'*60}"]

    # Average Sharpe by magnitude threshold
    lines.append("  Avg Sharpe by magnitude filter:")
    for mag, grp in sweep_df.groupby("mag_threshold"):
        lines.append(f"    |gold_ret| > {mag*100:.1f}%  →  "
                     f"avg SR={grp['sharpe'].mean():.3f}  "
                     f"best SR={grp['sharpe'].max():.3f}")

    lines.append("\n  Avg Sharpe by signal window:")
    for win, grp in sweep_df.groupby("signal_window"):
        lines.append(f"    {win}d return  →  "
                     f"avg SR={grp['sharpe'].mean():.3f}  "
                     f"best SR={grp['sharpe'].max():.3f}")

    lines.append("\n  Avg Sharpe: vol scaling on vs off:")
    for vs, grp in sweep_df.groupby("vol_scale"):
        lines.append(f"    vol_scale={'ON ' if vs else 'OFF'}  →  "
                     f"avg SR={grp['sharpe'].mean():.3f}  "
                     f"best SR={grp['sharpe'].max():.3f}")

    lines += ["", sep,
              "  FILES SAVED",
              f"  {out_dir}",
              "    layer5_sweep_results.csv",
              "    layer5_best_configs.csv",
              "    layer5_sweep_heatmap.png",
              "    layer5_best_equity.png",
              "    layer5_report.txt",
              sep]

    report = "\n".join(lines)
    print("\n" + report)

    path = os.path.join(out_dir, "layer5_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report saved → {path}")


# ─────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────

def run_analysis(
    l3_dir:  str  = L3_DIR,
    out_dir: str  = OUT_DIR,
    top_n:   int  = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Master entry point.

    Returns
    -------
    sweep_df : pd.DataFrame — full sweep results sorted by Sharpe
    """
    print("=" * 62)
    print("  LAYER 5 — ANALYSIS & SENSITIVITY SWEEP")
    print("=" * 62)

    os.makedirs(out_dir, exist_ok=True)

    df          = load_data()
    benchmarks  = benchmark_metrics(df)
    sweep_df    = run_sweep(df, verbose)

    # Save CSVs
    sweep_df.to_csv(os.path.join(out_dir, "layer5_sweep_results.csv"),
                    index=False)
    sweep_df.head(10).to_csv(
        os.path.join(out_dir, "layer5_best_configs.csv"), index=False)
    print(f"\n  Saved CSVs → {out_dir}")

    # Plots
    plot_heatmap(sweep_df, out_dir)
    plot_best_equity(df, sweep_df, benchmarks, out_dir, top_n)

    # Report
    write_report(sweep_df, benchmarks, out_dir)

    return sweep_df


# ─────────────────────────────────────────────────────────────────
# STANDALONE — python layer5_analysis.py
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sweep_df = run_analysis()

    print("\nTop 5 configs:")
    cols = ["mag_threshold", "signal_window", "vol_scale",
            "vol_window", "sharpe", "total_return", "max_drawdown", "n_trades"]
    print(sweep_df[cols].head(5).to_string(index=False))
