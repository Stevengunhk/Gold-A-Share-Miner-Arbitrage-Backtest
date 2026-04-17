"""
=============================================================
PERMUTATION TEST — SHUFFLED GOLD TRADING DAYS
Gold / A-Share Miner Arbitrage Backtest
=============================================================

Hypothesis:
    If the strategy's edge comes from the T+1 lag signal
    (gold moves → miners follow next day), then destroying
    the temporal order of gold returns should collapse the
    Sharpe ratio toward zero.

    If the edge comes purely from gold's bull trend
    (just being long gold-related assets most of the time),
    then the shuffled strategy will perform similarly to
    the real strategy — meaning the lag adds no value.

Method:
    1. Take the real gold return series
    2. Randomly shuffle the dates (bootstrap without replacement)
    3. Rebuild the signal using shuffled gold returns
    4. Run the backtest against REAL miner returns
       (miner dates are NOT shuffled — only gold is)
    5. Repeat N_PERMUTATIONS times
    6. Compare distribution of shuffled Sharpe ratios
       against the real strategy's Sharpe

Key insight:
    Shuffled gold returns have the same mean, std, skew,
    and bull/bear ratio as the real series — but zero
    autocorrelation and zero lead-lag relationship with miners.
    Any persistent edge in shuffled runs = trend/distribution effect.
    The GAP between real Sharpe and shuffled mean Sharpe = pure lag edge.

Outputs:
    permutation_test.png    — distribution of shuffled Sharpes
                              vs real Sharpe (p-value annotated)
    permutation_results.csv — all N shuffled run metrics

Run:  python permutation_test.py
"""

import os
import sys
sys.path.insert(0, r"C:\Gold ETF arbitrage\Layer 3_Signal generation")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

L3_DIR       = r"C:\Gold ETF arbitrage\Layer 3_Signal generation"
OUT_DIR      = r"C:\Gold ETF arbitrage\Final Backtest"

N_PERMUTATIONS  = 1000        # number of shuffled runs
RANDOM_SEED     = 42

# Best config (same as final_backtest.py)
MAG_THRESHOLD    = 0.0075
SIGNAL_WINDOW    = 1
TRANSACTION_COST = 0.002
STOP_LOSS_PCT    = 0.05
MAX_HOLD_DAYS    = 20

# Colours
BG    = "#0E1117"
BG2   = "#161B22"
BG3   = "#1C2333"
LTGRAY= "#30363D"
GOLD  = "#D4AF37"
GREEN = "#2ECC71"
RED   = "#E74C3C"
BLUE  = "#2196F3"
GRAY  = "#8B949E"
WHITE = "#E6EDF3"


# ─────────────────────────────────────────────────────────────────
# CORE BACKTEST (same vectorised logic as Layer 5/Final)
# ─────────────────────────────────────────────────────────────────

def run_backtest(
    gold_ret_series: pd.Series,    # gold returns (may be shuffled)
    miner_ret:       pd.Series,    # miner returns (always real, unshuffled)
    miner_levels:    np.ndarray,
    mag_threshold:   float = MAG_THRESHOLD,
    txn_cost:        float = TRANSACTION_COST,
    stop_loss_pct:   float = STOP_LOSS_PCT,
    max_hold_days:   int   = MAX_HOLD_DAYS,
) -> dict:
    """
    Runs backtest using gold_ret_series as the signal source.
    miner_ret and miner_levels are always the real historical values.
    """
    raw_dir = np.sign(gold_ret_series)
    raw_dir[gold_ret_series.abs() < mag_threshold] = 0
    raw_signal = raw_dir.replace(0, np.nan).ffill().fillna(0)
    position   = raw_signal.shift(1).fillna(0)

    pos_arr    = position.values.copy()
    entry_lvl  = 0.0
    entry_sign = 0
    hold_days  = 0

    for i in range(1, len(pos_arr)):
        cur = int(np.sign(pos_arr[i]))
        if entry_sign == 0 and cur != 0:
            entry_lvl  = miner_levels[i]
            entry_sign = cur
            hold_days  = 0
        elif entry_sign != 0:
            hold_days += 1
            tr = (miner_levels[i] / entry_lvl - 1) * entry_sign if entry_lvl else 0
            if tr < -stop_loss_pct:
                pos_arr[i] = 0.0
                entry_sign = 0; entry_lvl = 0.0; hold_days = 0
                continue
            if hold_days >= max_hold_days:
                entry_lvl  = miner_levels[i]
                entry_sign = int(np.sign(pos_arr[i]))
                hold_days  = 0
            if cur != 0 and cur != entry_sign:
                entry_lvl  = miner_levels[i]
                entry_sign = cur
                hold_days  = 0

    pos_final  = pd.Series(pos_arr, index=miner_ret.index)
    pos_change = pos_final.diff().abs().fillna(0)
    net_pnl    = pos_final * miner_ret - pos_change * txn_cost

    cum  = (1 + net_pnl).cumprod()
    tot  = cum.iloc[-1] - 1
    n    = len(net_pnl)
    annr = (1 + tot) ** (252 / n) - 1
    annv = net_pnl.std() * np.sqrt(252)
    sr   = annr / annv if annv > 0 else 0
    peak = cum.cummax()
    mdd  = (cum / peak - 1).min()
    act  = net_pnl[net_pnl != 0]
    wr   = (act > 0).mean() if len(act) > 0 else 0
    n_tr = int((pos_change > 0).sum())

    return {
        "sharpe":       sr,
        "ann_return":   annr,
        "ann_vol":      annv,
        "total_return": tot,
        "max_drawdown": mdd,
        "win_rate":     wr,
        "n_trades":     n_tr,
        "_net_pnl":     net_pnl,
        "_cum":         cum,
    }


# ─────────────────────────────────────────────────────────────────
# MAIN TEST
# ─────────────────────────────────────────────────────────────────

def run_permutation_test():
    print("=" * 62)
    print("  PERMUTATION TEST — SHUFFLED GOLD TRADING DAYS")
    print("=" * 62)

    # Load data
    path = os.path.join(L3_DIR, "layer3_signals.parquet")
    df   = pd.read_parquet(path)
    print(f"\n  Loaded {len(df):,} rows  "
          f"{df.index[0].date()} → {df.index[-1].date()}")

    miner_ret    = df["miner_idx"].pct_change().fillna(0)
    miner_levels = df["miner_idx"].values
    gold_ret_real= df["gold_idx"].pct_change().fillna(0)

    # ── REAL strategy run ──────────────────────────────────────
    print("\n  [1/3] Running real strategy ...")
    real = run_backtest(gold_ret_real, miner_ret, miner_levels)
    print(f"  Real Sharpe      : {real['sharpe']:.4f}")
    print(f"  Real Ann Return  : {real['ann_return']:.1%}")
    print(f"  Real Win Rate    : {real['win_rate']:.1%}")
    print(f"  Real Trades      : {real['n_trades']}")

    # ── SHUFFLED runs ──────────────────────────────────────────
    print(f"\n  [2/3] Running {N_PERMUTATIONS} shuffled permutations ...")
    rng     = np.random.default_rng(RANDOM_SEED)
    results = []
    gold_values = gold_ret_real.values.copy()

    for i in range(N_PERMUTATIONS):
        # Shuffle gold returns — preserves distribution, destroys temporal order
        shuffled_vals = gold_values.copy()
        rng.shuffle(shuffled_vals)
        shuffled_gold = pd.Series(shuffled_vals, index=gold_ret_real.index)

        r = run_backtest(shuffled_gold, miner_ret, miner_levels)
        results.append({
            "run":          i + 1,
            "sharpe":       r["sharpe"],
            "ann_return":   r["ann_return"],
            "max_drawdown": r["max_drawdown"],
            "win_rate":     r["win_rate"],
            "n_trades":     r["n_trades"],
        })

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{N_PERMUTATIONS} done  "
                  f"(running avg SR={np.mean([x['sharpe'] for x in results]):.3f})")

    res_df = pd.DataFrame(results)

    # ── STATISTICS ────────────────────────────────────────────
    print(f"\n  [3/3] Computing statistics ...")

    sharpes       = res_df["sharpe"].values
    real_sr       = real["sharpe"]
    mean_shuf     = sharpes.mean()
    std_shuf      = sharpes.std()
    p_value       = (sharpes >= real_sr).mean()          # fraction of shuffled >= real
    z_score       = (real_sr - mean_shuf) / std_shuf     # how many std above shuffled mean
    pct_positive  = (sharpes > 0).mean()
    pct_above_1   = (sharpes > 1.0).mean()
    lag_edge      = real_sr - mean_shuf                  # Sharpe attributable to lag

    # Interpret
    if p_value < 0.01:
        verdict = "STRONG — p < 0.01  edge is NOT explained by bull trend alone"
    elif p_value < 0.05:
        verdict = "SIGNIFICANT — p < 0.05  edge largely driven by lag signal"
    elif p_value < 0.10:
        verdict = "MARGINAL — p < 0.10  partial lag edge, partial trend effect"
    else:
        verdict = "WEAK — p > 0.10  edge mainly explained by gold bull trend"

    print(f"\n{'='*62}")
    print(f"  PERMUTATION TEST RESULTS  ({N_PERMUTATIONS} shuffled runs)")
    print(f"{'='*62}")
    print(f"  Real strategy Sharpe        : {real_sr:>8.4f}")
    print(f"  Shuffled mean Sharpe        : {mean_shuf:>8.4f}")
    print(f"  Shuffled std  Sharpe        : {std_shuf:>8.4f}")
    print(f"  Shuffled min  Sharpe        : {sharpes.min():>8.4f}")
    print(f"  Shuffled max  Sharpe        : {sharpes.max():>8.4f}")
    print(f"  Z-score (real vs shuffled)  : {z_score:>8.2f}")
    print(f"  P-value (one-sided)         : {p_value:>8.4f}")
    print(f"  % shuffled runs SR > 0      : {pct_positive*100:>7.1f}%")
    print(f"  % shuffled runs SR > 1.0    : {pct_above_1*100:>7.1f}%")
    print(f"\n  Lag-signal Sharpe component : {lag_edge:>8.4f}")
    print(f"  Trend Sharpe component      : {mean_shuf:>8.4f}")
    if real_sr > 0:
        print(f"  Lag % of total edge         : {lag_edge/real_sr*100:>7.1f}%")
        print(f"  Trend % of total edge       : {mean_shuf/real_sr*100:>7.1f}%")
    print(f"\n  Verdict: {verdict}")
    print(f"{'='*62}\n")

    # Save results CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    res_df.to_csv(os.path.join(OUT_DIR, "permutation_results.csv"), index=False)

    return real, res_df, real_sr, mean_shuf, std_shuf, p_value, z_score, lag_edge


# ─────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────

def plot_permutation(real, res_df, real_sr, mean_shuf, std_shuf,
                     p_value, z_score, lag_edge):

    sharpes = res_df["sharpe"].values

    fig = plt.figure(figsize=(16, 12), facecolor=BG)
    gs  = plt.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                       top=0.88, bottom=0.08, left=0.08, right=0.96)

    fig.text(0.5, 0.94,
             "Permutation Test — Shuffled Gold Trading Days",
             ha="center", color=GOLD, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.915,
             f"{N_PERMUTATIONS} permutations  ·  "
             f"Real SR={real_sr:.3f}  ·  "
             f"Shuffled mean SR={mean_shuf:.3f}  ·  "
             f"p-value={p_value:.4f}  ·  "
             f"z={z_score:.2f}",
             ha="center", color=GRAY, fontsize=9)

    def ax_style(ax, title=""):
        ax.set_facecolor(BG2)
        ax.tick_params(colors=GRAY, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(LTGRAY)
        ax.yaxis.grid(True, color=LTGRAY, lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        if title:
            ax.set_title(title, color=WHITE, fontsize=9,
                         fontweight="bold", pad=6)

    # ── Panel 1: Sharpe distribution histogram ─────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax_style(ax1, "Distribution of Shuffled Sharpe Ratios  vs  Real Strategy")

    # Histogram of shuffled
    n_bins = 60
    counts, bin_edges, patches = ax1.hist(
        sharpes, bins=n_bins, color=BLUE, alpha=0.7,
        edgecolor=BG, linewidth=0.3,
        label=f"Shuffled ({N_PERMUTATIONS} runs)"
    )

    # Colour bars to the right of real SR in red
    for patch, left in zip(patches, bin_edges[:-1]):
        if left >= real_sr:
            patch.set_facecolor(RED)
            patch.set_alpha(0.8)

    # Real SR line
    ax1.axvline(real_sr,    color=GREEN, lw=2.5, zorder=5,
                label=f"Real strategy  SR={real_sr:.3f}")
    ax1.axvline(mean_shuf,  color=GOLD,  lw=1.5, ls="--", zorder=4,
                label=f"Shuffled mean  SR={mean_shuf:.3f}")
    ax1.axvline(0,          color=GRAY,  lw=0.8, ls=":", alpha=0.6)

    # Shade ±1 std of shuffled
    ax1.axvspan(mean_shuf - std_shuf, mean_shuf + std_shuf,
                alpha=0.08, color=BLUE, label="±1 std shuffled")

    # Annotation: lag edge
    ymax = counts.max()
    mid_x = (mean_shuf + real_sr) / 2
    ax1.annotate(
        "",
        xy=(real_sr, ymax * 0.55),
        xytext=(mean_shuf, ymax * 0.55),
        arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.5)
    )
    ax1.text(mid_x, ymax * 0.60,
             f"Lag edge\n+{lag_edge:.2f} SR",
             ha="center", color=GREEN, fontsize=8, fontweight="bold")

    # p-value annotation
    p_label = f"p = {p_value:.4f}"
    color_p = GREEN if p_value < 0.05 else (GOLD if p_value < 0.10 else RED)
    ax1.text(0.98, 0.92, p_label,
             transform=ax1.transAxes,
             ha="right", color=color_p,
             fontsize=13, fontweight="bold")

    ax1.set_xlabel("Sharpe Ratio", color=GRAY, fontsize=9)
    ax1.set_ylabel("Frequency", color=GRAY, fontsize=9)
    ax1.legend(facecolor=BG3, labelcolor=WHITE, fontsize=8,
               framealpha=0.9, loc="upper left")

    # ── Panel 2: Shuffled cumulative returns sample ─────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax_style(ax2, "Sample Shuffled Equity Curves  (20 random runs)")

    # Load data to re-run sample equity curves
    path = os.path.join(L3_DIR, "layer3_signals.parquet")
    df   = pd.read_parquet(path)
    miner_ret    = df["miner_idx"].pct_change().fillna(0)
    miner_levels = df["miner_idx"].values
    gold_ret_real= df["gold_idx"].pct_change().fillna(0)
    gold_values  = gold_ret_real.values.copy()
    rng2         = np.random.default_rng(RANDOM_SEED + 99)

    for _ in range(20):
        sv = gold_values.copy()
        rng2.shuffle(sv)
        sg = pd.Series(sv, index=gold_ret_real.index)
        r  = run_backtest(sg, miner_ret, miner_levels)
        ax2.plot(df.index, r["_cum"] * 100,
                 color=BLUE, lw=0.6, alpha=0.25)

    ax2.plot(df.index, real["_cum"] * 100,
             color=GREEN, lw=2.2, zorder=5,
             label=f"Real  (SR={real_sr:.2f})")
    ax2.axhline(100, color=LTGRAY, lw=0.6, ls=":")
    ax2.legend(facecolor=BG3, labelcolor=WHITE, fontsize=8)
    ax2.set_ylabel("Portfolio Value (base=100)", color=GRAY, fontsize=8)
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    import matplotlib.dates as mdates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), color=GRAY, fontsize=7)

    # ── Panel 3: Edge decomposition bar chart ───────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax_style(ax3, "Sharpe Decomposition — Lag Signal vs Bull Trend")

    components = ["Bull trend\n(shuffled mean)", "Lag signal\n(T+1 edge)", "Total\n(real strategy)"]
    values     = [mean_shuf, lag_edge, real_sr]
    clrs       = [GOLD, GREEN, WHITE]

    bars = ax3.bar(components, values, color=clrs, alpha=0.85, width=0.5)
    ax3.axhline(0, color=LTGRAY, lw=0.8)
    for bar, val in zip(bars, values):
        yoff = 0.03 if val >= 0 else -0.08
        ax3.text(bar.get_x() + bar.get_width()/2,
                 val + yoff,
                 f"{val:.3f}",
                 ha="center", color=WHITE, fontsize=10, fontweight="bold")

    ax3.set_ylabel("Sharpe Ratio contribution", color=GRAY, fontsize=8)
    ax3.tick_params(axis="x", colors=GRAY, labelsize=8)

    # Add percentage labels
    if real_sr > 0:
        ax3.text(0, mean_shuf / 2,
                 f"{mean_shuf/real_sr*100:.0f}%\nof total",
                 ha="center", color=BG, fontsize=7.5, fontweight="bold")
        ax3.text(1, mean_shuf + lag_edge / 2,
                 f"{lag_edge/real_sr*100:.0f}%\nof total",
                 ha="center", color=BG, fontsize=7.5, fontweight="bold")

    # Verdict box
    p_label = f"p = {p_value:.4f}"
    if p_value < 0.01:
        verdict_short = "STRONG LAG SIGNAL"
        vc = GREEN
    elif p_value < 0.05:
        verdict_short = "SIGNIFICANT EDGE"
        vc = GREEN
    elif p_value < 0.10:
        verdict_short = "MARGINAL EDGE"
        vc = GOLD
    else:
        verdict_short = "TREND-DRIVEN"
        vc = RED

    fig.text(0.75, 0.08, f"{verdict_short}\n{p_label}",
             ha="center", va="bottom",
             color=vc, fontsize=13, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor=BG3, edgecolor=vc, linewidth=1.5))

    path = os.path.join(OUT_DIR, "permutation_test.png")
    fig.savefig(path, dpi=180, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  Chart saved → {path}")


# ─────────────────────────────────────────────────────────────────
# STANDALONE
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    real, res_df, real_sr, mean_shuf, std_shuf, p_value, z_score, lag_edge = \
        run_permutation_test()
    plot_permutation(real, res_df, real_sr, mean_shuf, std_shuf,
                     p_value, z_score, lag_edge)
    print("Done.")
