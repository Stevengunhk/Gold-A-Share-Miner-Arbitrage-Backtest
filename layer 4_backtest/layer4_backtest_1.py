"""
=============================================================
LAYER 4 — BACKTEST ENGINE  (directional signal, revised)
Gold / A-Share Miner Arbitrage Backtest
=============================================================

Signal logic (T+1 execution):
    gold_ret(T)  > 0  →  LONG  miners on Day T+1
    gold_ret(T)  < 0  →  SHORT miners on Day T+1
    gold_ret(T) == 0  →  hold previous position

    Always in market (long or short) unless circuit breaker is active.
    Flip direction the moment gold's daily return reverses sign.

Why this makes sense:
    COMEX gold settles ~18:30 UTC on Day T.
    A-shares open at 01:30 UTC on Day T+1.
    gold_ret(T) is fully observable before the T+1 A-share open.
    → zero lookahead bias.

Risk controls (all optional):
    stop_loss        : exit (go flat) if trade P&L < -stop_loss_pct from entry
                       re-enters on next gold signal after the stop
    max_hold         : force flip/exit after max_hold_days regardless of signal
    circuit_breaker  : pause ALL trading for cooldown_days after strategy
                       drawdown breaches dd_threshold; re-enters on resume

Outputs
-------
layer4_results.parquet   — full daily results DataFrame
layer4_trades.csv        — completed trade log
layer4_backtest.png      — 5-panel performance chart

Downstream layers import:
    from layer4_backtest import run_full_backtest
"""

import os
import sys
sys.path.insert(0, r"C:\Gold ETF arbitrage\Layer 1_Data ingestion")
sys.path.insert(0, r"C:\Gold ETF arbitrage\Layer 2_Index construction")
sys.path.insert(0, r"C:\Gold ETF arbitrage\Layer 3_Signal generation")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

L3_DIR  = r"C:\Gold ETF arbitrage\Layer 3_Signal generation"
OUT_DIR = r"C:\Gold ETF arbitrage\Layer 4_Backtest engine"

TRANSACTION_COST    = 0.002   # 0.20% per leg (one-way)

USE_STOP_LOSS       = True
STOP_LOSS_PCT       = 0.05    # exit flat if trade loses 5% from entry

USE_MAX_HOLD        = True
MAX_HOLD_DAYS       = 20      # force flip after 20 days in same direction

USE_CIRCUIT_BREAKER = True
DD_THRESHOLD        = 0.10    # pause after -10% strategy drawdown
COOLDOWN_DAYS       = 10      # resume after 10 trading days


# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD LAYER 3 OUTPUT
# ─────────────────────────────────────────────────────────────────

def load_layer3(l3_dir: str = L3_DIR) -> pd.DataFrame:
    parquet_path = os.path.join(l3_dir, "layer3_signals.parquet")

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        print(f"  Loaded layer3_signals.parquet  ({len(df):,} rows)")
    else:
        print("  layer3_signals.parquet not found — running Layer 3 ...")
        from layer3_signal import build_signals  # type: ignore
        signals, idx_df, _ = build_signals(verbose=False)
        df = pd.concat([idx_df[["gold_idx", "miner_idx"]], signals], axis=1)

    return df


# ─────────────────────────────────────────────────────────────────
# STEP 2 — BUILD DIRECTIONAL SIGNAL FROM GOLD RETURNS
# ─────────────────────────────────────────────────────────────────

def build_directional_signal(df: pd.DataFrame) -> pd.Series:
    """
    signal(T) = +1 if gold_ret(T) > 0
                -1 if gold_ret(T) < 0
                 0 (carry forward) if gold_ret(T) == 0

    This signal is applied to miner returns on Day T+1.
    Using .shift(1) inside the backtest loop handles the T+1 execution.
    """
    gold_ret = df["gold_idx"].pct_change()
    signal   = np.sign(gold_ret)

    # On flat days (gold_ret == 0) carry the previous signal
    signal   = signal.replace(0, np.nan).ffill().fillna(1)
    signal   = signal.astype(int).rename("dir_signal")

    n_long  = (signal ==  1).sum()
    n_short = (signal == -1).sum()
    print(f"\n  Directional signal built:")
    print(f"    Long  days : {n_long:,}  ({n_long/len(signal)*100:.1f}%)")
    print(f"    Short days : {n_short:,}  ({n_short/len(signal)*100:.1f}%)")
    print(f"    (always in market before risk controls)")

    return signal


# ─────────────────────────────────────────────────────────────────
# STEP 3 — BACKTEST STATE MACHINE
# ─────────────────────────────────────────────────────────────────

def run_backtest(
    df:               pd.DataFrame,
    dir_signal:       pd.Series,
    transaction_cost: float = TRANSACTION_COST,
    use_stop_loss:    bool  = USE_STOP_LOSS,
    stop_loss_pct:    float = STOP_LOSS_PCT,
    use_max_hold:     bool  = USE_MAX_HOLD,
    max_hold_days:    int   = MAX_HOLD_DAYS,
    use_circuit_breaker: bool  = USE_CIRCUIT_BREAKER,
    dd_threshold:     float = DD_THRESHOLD,
    cooldown_days:    int   = COOLDOWN_DAYS,
    verbose:          bool  = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Day-by-day backtest loop.

    Execution model:
        Day T  : observe gold_ret(T) → signal(T) known after market close
        Day T+1: enter/flip/hold miner position at T+1 close
        P&L(T+1) = position(T) × miner_ret(T+1) − cost_if_traded
    """
    miner_ret = df["miner_idx"].pct_change().fillna(0)
    gold_ret  = df["gold_idx"].pct_change().fillna(0)
    n         = len(df)

    # Output arrays
    position       = np.zeros(n, dtype=int)
    trade_cost_arr = np.zeros(n)
    daily_pnl      = np.zeros(n)
    cum_pnl        = np.ones(n)

    # State
    pos           = 0
    entry_level   = 0.0
    hold_days     = 0
    cooldown_left = 0
    peak_cum      = 1.0
    stopped_out   = False   # flat after stop-loss; wait for next signal flip to re-enter

    trade_log = []

    for i in range(1, n):
        date          = df.index[i]
        new_signal    = int(dir_signal.iloc[i - 1])   # T+1: yesterday's signal
        miner_level   = df["miner_idx"].iloc[i]
        miner_ret_now = miner_ret.iloc[i]

        # ── Circuit breaker: stay flat during cooldown ─────────
        if use_circuit_breaker and cooldown_left > 0:
            cooldown_left -= 1
            if pos != 0:
                trade_log.append(_close_trade(
                    pos, df.index[i-1], date,
                    entry_level, miner_level, "circuit_breaker"))
                trade_cost_arr[i] += transaction_cost
                pos = 0; hold_days = 0; entry_level = 0.0
            position[i]  = 0
            daily_pnl[i] = -trade_cost_arr[i]
            cum_pnl[i]   = cum_pnl[i-1] * (1 + daily_pnl[i])
            peak_cum      = max(peak_cum, cum_pnl[i])
            stopped_out   = False   # reset stop flag during cooldown
            continue

        # ── If stopped out, wait for gold signal to flip ───────
        if stopped_out:
            # Re-enter only when signal changes direction
            if new_signal != 0 and new_signal != pos:
                stopped_out = False
            else:
                position[i]  = 0
                daily_pnl[i] = 0.0
                cum_pnl[i]   = cum_pnl[i-1]
                peak_cum      = max(peak_cum, cum_pnl[i])
                continue

        # ── Risk checks on open position ───────────────────────
        if pos != 0:
            hold_days += 1
            trade_ret  = (miner_level / entry_level - 1) * pos

            # Stop-loss
            if use_stop_loss and trade_ret < -stop_loss_pct:
                trade_log.append(_close_trade(
                    pos, df.index[i-1], date,
                    entry_level, miner_level, "stop_loss"))
                trade_cost_arr[i] += transaction_cost
                pos = 0; hold_days = 0; entry_level = 0.0
                stopped_out = True

            # Max hold → force flip to match current signal
            elif use_max_hold and hold_days >= max_hold_days:
                trade_log.append(_close_trade(
                    pos, df.index[i-1], date,
                    entry_level, miner_level, "max_hold"))
                trade_cost_arr[i] += transaction_cost
                pos = 0; hold_days = 0; entry_level = 0.0

        # ── Enter / flip based on directional signal ───────────
        if not stopped_out:
            if pos == 0:
                # Enter fresh
                pos         = new_signal
                entry_level = miner_level
                hold_days   = 0
                trade_cost_arr[i] += transaction_cost
                trade_log.append({
                    "entry_date":  date,
                    "exit_date":   None,
                    "direction":   "LONG" if pos == 1 else "SHORT",
                    "entry_level": entry_level,
                    "exit_level":  None,
                    "pnl_pct":     None,
                    "reason":      "signal_entry",
                })

            elif pos != new_signal:
                # Flip: close current, open opposite
                trade_log.append(_close_trade(
                    pos, df.index[i-1], date,
                    entry_level, miner_level, "signal_flip"))
                trade_cost_arr[i] += transaction_cost * 2   # close + reopen
                pos         = new_signal
                entry_level = miner_level
                hold_days   = 0
                trade_log.append({
                    "entry_date":  date,
                    "exit_date":   None,
                    "direction":   "LONG" if pos == 1 else "SHORT",
                    "entry_level": entry_level,
                    "exit_level":  None,
                    "pnl_pct":     None,
                    "reason":      "signal_entry",
                })
            # else: same direction — hold, no cost

        # ── Daily P&L ──────────────────────────────────────────
        position[i]  = pos if not stopped_out else 0
        gross_pnl    = position[i] * miner_ret_now
        daily_pnl[i] = gross_pnl - trade_cost_arr[i]
        cum_pnl[i]   = cum_pnl[i-1] * (1 + daily_pnl[i])

        # ── Circuit breaker trigger ────────────────────────────
        peak_cum = max(peak_cum, cum_pnl[i])
        if use_circuit_breaker and cooldown_left == 0:
            current_dd = (cum_pnl[i] / peak_cum) - 1
            if current_dd < -dd_threshold:
                cooldown_left = cooldown_days
                if verbose:
                    print(f"  [!] Circuit breaker on {date.date()}  "
                          f"DD={current_dd*100:.1f}%  pausing {cooldown_days}d")

    # ── Assemble results ───────────────────────────────────────
    results = df[["gold_idx", "miner_idx", "z_score"]].copy()
    results["gold_ret"]    = gold_ret
    results["dir_signal"]  = dir_signal
    results["position"]    = position
    results["trade_cost"]  = trade_cost_arr
    results["daily_pnl"]   = daily_pnl
    results["cum_strat"]   = cum_pnl * 100
    results["cum_bh_miner"]= (1 + miner_ret).cumprod() * 100
    results["cum_bh_gold"] = (1 + gold_ret).cumprod() * 100

    peak = results["cum_strat"].cummax()
    results["drawdown"] = (results["cum_strat"] / peak) - 1

    trades_df = _finalise_trade_log(
        trade_log, df["miner_idx"].iloc[-1], df.index[-1])

    if verbose:
        _print_summary(results, trades_df, transaction_cost,
                       use_stop_loss, stop_loss_pct,
                       use_max_hold, max_hold_days,
                       use_circuit_breaker, dd_threshold)

    return results, trades_df


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _close_trade(pos, entry_date, exit_date, entry_level, exit_level, reason):
    pnl = (exit_level / entry_level - 1) * pos if entry_level else 0
    return {
        "entry_date":  entry_date,
        "exit_date":   exit_date,
        "direction":   "LONG" if pos == 1 else "SHORT",
        "entry_level": entry_level,
        "exit_level":  exit_level,
        "pnl_pct":     round(pnl * 100, 3),
        "reason":      "signal_entry",
        "exit_reason": reason,
    }


def _finalise_trade_log(trade_log: list,
                        last_level: float,
                        last_date) -> pd.DataFrame:
    completed  = []
    open_trade = None

    for row in trade_log:
        if row.get("reason") == "signal_entry" and row.get("exit_date") is None:
            open_trade = row
        else:
            completed.append(row)
            open_trade = None

    if open_trade is not None:
        pos = 1 if open_trade["direction"] == "LONG" else -1
        pnl = (last_level / open_trade["entry_level"] - 1) * pos
        open_trade.update({
            "exit_date":   last_date,
            "exit_level":  last_level,
            "pnl_pct":     round(pnl * 100, 3),
            "exit_reason": "end_of_backtest",
        })
        completed.append(open_trade)

    df = pd.DataFrame(completed)
    if not df.empty and "exit_date" in df.columns:
        df["hold_days"] = (
            pd.to_datetime(df["exit_date"]) -
            pd.to_datetime(df["entry_date"])
        ).dt.days
    return df


def _print_summary(results, trades_df, txn_cost,
                   use_sl, sl_pct, use_mh, mh_days, use_cb, dd_thr):
    ann = 252

    def metrics(ret):
        total  = (1 + ret).prod() - 1
        ann_r  = (1 + total) ** (ann / max(len(ret), 1)) - 1
        vol    = ret.std() * np.sqrt(ann)
        sharpe = ann_r / vol if vol > 0 else 0
        peak   = (1 + ret).cumprod().cummax()
        mdd    = ((1 + ret).cumprod() / peak - 1).min()
        wins   = (ret > 0).sum()
        active = (ret != 0).sum()
        wr     = wins / active if active > 0 else 0
        return total, ann_r, vol, sharpe, mdd, wr

    st = metrics(results["daily_pnl"])
    bm = metrics(results["miner_ret"] if "miner_ret" in results.columns
                 else results["gold_ret"])
    bg = metrics(results["gold_ret"])

    # Recalc B&H miner properly
    miner_ret_s = results["cum_bh_miner"].pct_change().fillna(0)
    bm = metrics(miner_ret_s)

    print(f"\n{'='*60}")
    print(f"  LAYER 4 — PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Metric':<24} {'Strategy':>12} {'B&H Miners':>12} {'B&H Gold':>12}")
    print(f"  {'-'*60}")
    labels = ["Total Return", "Ann. Return", "Ann. Volatility",
              "Sharpe Ratio", "Max Drawdown", "Win Rate"]
    fmts   = ["{:.1%}", "{:.1%}", "{:.1%}", "{:.2f}", "{:.1%}", "{:.1%}"]
    for label, fmt, s, b, g in zip(labels, fmts, st, bm, bg):
        print(f"  {label:<24} {fmt.format(s):>12} {fmt.format(b):>12} {fmt.format(g):>12}")

    if not trades_df.empty and "pnl_pct" in trades_df.columns:
        pnl = trades_df["pnl_pct"].dropna()
        print(f"\n  Trade Statistics:")
        print(f"    Total trades      : {len(trades_df)}")
        if "hold_days" in trades_df.columns:
            print(f"    Avg hold (days)   : {trades_df['hold_days'].mean():.1f}")
        print(f"    Win rate          : {(pnl > 0).mean()*100:.1f}%")
        print(f"    Avg win  (%)      : {pnl[pnl>0].mean():.2f}%")
        print(f"    Avg loss (%)      : {pnl[pnl<0].mean():.2f}%")
        print(f"    Best trade (%)    : {pnl.max():.2f}%")
        print(f"    Worst trade (%)   : {pnl.min():.2f}%")

        if "direction" in trades_df.columns:
            by_dir = trades_df.groupby("direction")["pnl_pct"].agg(["count","mean"])
            print(f"\n    By direction:")
            for d, row in by_dir.iterrows():
                print(f"      {d:<8}  count={int(row['count'])}  avg_pnl={row['mean']:.2f}%")

        if "exit_reason" in trades_df.columns:
            by_exit = trades_df.groupby("exit_reason")["pnl_pct"].agg(["count","mean"])
            print(f"\n    By exit reason:")
            for r, row in by_exit.iterrows():
                print(f"      {r:<22}  count={int(row['count'])}  avg_pnl={row['mean']:.2f}%")

    print(f"\n  Risk controls:")
    print(f"    Stop-loss        : {'ON  (-'+str(sl_pct*100)+'%)' if use_sl else 'OFF'}")
    print(f"    Max hold         : {'ON  ('+str(mh_days)+' days)' if use_mh else 'OFF'}")
    print(f"    Circuit breaker  : {'ON  (-'+str(dd_thr*100)+'% DD)' if use_cb else 'OFF'}")
    print(f"    Transaction cost : {txn_cost*100:.2f}% per leg")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────
# STEP 4 — PLOT
# ─────────────────────────────────────────────────────────────────

def plot_results(results: pd.DataFrame, trades_df: pd.DataFrame,
                 out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)

    gold_c  = "#D4AF37"
    miner_c = "#2196F3"
    strat_c = "#2ECC71"
    short_c = "#E74C3C"

    fig = plt.figure(figsize=(16, 16))
    gs  = GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    # ── Panel 1: cumulative returns ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(results.index, results["cum_strat"],     color=strat_c, lw=2.0, label="Arb Strategy")
    ax1.plot(results.index, results["cum_bh_miner"],  color=miner_c, lw=1.4, alpha=0.7, label="B&H Miners")
    ax1.plot(results.index, results["cum_bh_gold"],   color=gold_c,  lw=1.4, alpha=0.7, label="B&H Gold")
    ax1.axhline(100, color="gray", lw=0.6, ls="--")
    ax1.set_title("Cumulative Performance (base = 100)", fontsize=10)
    ax1.set_ylabel("Portfolio Value")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: drawdown ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(results.index, results["drawdown"] * 100, 0,
                     color=short_c, alpha=0.5)
    ax2.set_title("Strategy Drawdown (%)", fontsize=10)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: position timeline ────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :])
    pos = results["position"]
    ax3.fill_between(results.index, pos, 0,
                     where=pos > 0, color=strat_c, alpha=0.55, label="Long miners")
    ax3.fill_between(results.index, pos, 0,
                     where=pos < 0, color=short_c, alpha=0.55, label="Short miners")
    ax3.axhline(0, color="black", lw=0.6)
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(["Short", "Flat", "Long"], fontsize=8)
    ax3.set_title("Position Over Time  (flat = circuit breaker / stop-out)", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)

    # ── Panel 4: trade P&L distribution ──────────────────────────
    ax4 = fig.add_subplot(gs[3, 0])
    if not trades_df.empty and "pnl_pct" in trades_df.columns:
        pnl_sorted = trades_df["pnl_pct"].dropna().sort_values().values
        colors     = [strat_c if p > 0 else short_c for p in pnl_sorted]
        ax4.bar(range(len(pnl_sorted)), pnl_sorted, color=colors, alpha=0.75)
        ax4.axhline(0, color="black", lw=0.8)
        ax4.set_title("Trade P&L Distribution (sorted)", fontsize=10)
        ax4.set_xlabel("Trade #")
        ax4.set_ylabel("P&L (%)")
        ax4.grid(True, alpha=0.3, axis="y")

    # ── Panel 5: rolling 1-year Sharpe ───────────────────────────
    ax5 = fig.add_subplot(gs[3, 1])
    roll_sharpe = (
        results["daily_pnl"]
        .rolling(252)
        .apply(lambda x: (x.mean() / x.std() * np.sqrt(252))
               if x.std() > 0 else 0, raw=True)
    )
    ax5.plot(results.index, roll_sharpe, color=strat_c, lw=1.2)
    ax5.axhline(0, color="black", lw=0.8, ls="--")
    ax5.axhline(1, color=gold_c,  lw=0.6, ls=":")
    ax5.set_title("Rolling 1-Year Sharpe Ratio", fontsize=10)
    ax5.set_ylabel("Sharpe")
    ax5.grid(True, alpha=0.3)

    for ax in [ax1, ax2, ax3, ax5]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    fig.suptitle("Layer 4 — Backtest Results (Directional Gold Signal)", fontsize=13, y=1.005)
    save_path = os.path.join(out_dir, "layer4_backtest.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {save_path}")


# ─────────────────────────────────────────────────────────────────
# MASTER FUNCTION — called by Layer 5
# ─────────────────────────────────────────────────────────────────

def run_full_backtest(
    l3_dir:              str   = L3_DIR,
    out_dir:             str   = OUT_DIR,
    transaction_cost:    float = TRANSACTION_COST,
    use_stop_loss:       bool  = USE_STOP_LOSS,
    stop_loss_pct:       float = STOP_LOSS_PCT,
    use_max_hold:        bool  = USE_MAX_HOLD,
    max_hold_days:       int   = MAX_HOLD_DAYS,
    use_circuit_breaker: bool  = USE_CIRCUIT_BREAKER,
    dd_threshold:        float = DD_THRESHOLD,
    cooldown_days:       int   = COOLDOWN_DAYS,
    save:                bool  = True,
    plot:                bool  = True,
    verbose:             bool  = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Master entry point for Layer 5.

    Returns
    -------
    results   : pd.DataFrame — daily backtest results
    trades_df : pd.DataFrame — completed trade log
    """
    print("=" * 60)
    print("  LAYER 4 — BACKTEST ENGINE  (directional signal)")
    print("=" * 60)

    df         = load_layer3(l3_dir)
    dir_signal = build_directional_signal(df)

    results, trades_df = run_backtest(
        df, dir_signal,
        transaction_cost    = transaction_cost,
        use_stop_loss       = use_stop_loss,
        stop_loss_pct       = stop_loss_pct,
        use_max_hold        = use_max_hold,
        max_hold_days       = max_hold_days,
        use_circuit_breaker = use_circuit_breaker,
        dd_threshold        = dd_threshold,
        cooldown_days       = cooldown_days,
        verbose             = verbose,
    )

    if save:
        os.makedirs(out_dir, exist_ok=True)
        results.to_parquet(os.path.join(out_dir, "layer4_results.parquet"))
        trades_df.to_csv(os.path.join(out_dir, "layer4_trades.csv"), index=False)
        print(f"  Saved: layer4_results.parquet + layer4_trades.csv → {out_dir}")

    if plot:
        plot_results(results, trades_df, out_dir)

    return results, trades_df


# ─────────────────────────────────────────────────────────────────
# STANDALONE — python layer4_backtest.py
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results, trades = run_full_backtest()

    print("\nResults tail (last 5 rows):")
    print(results[["gold_idx", "miner_idx", "gold_ret",
                   "position", "daily_pnl", "cum_strat"]].tail(5).to_string())

    print(f"\nLast 5 trades:")
    print(trades.tail(5).to_string())
