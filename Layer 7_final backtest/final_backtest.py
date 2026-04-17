"""
=============================================================
FINAL STRATEGY BACKTEST — FULL PERFORMANCE REPORT
Gold / A-Share Miner Arbitrage
=============================================================

Best config (validated in Layer 5 + Layer 6):
    Magnitude filter : |gold_ret| > 0.75%
    Signal window    : 1-day gold return
    Vol scaling      : OFF
    Transaction cost : 0.20% per leg
    Stop-loss        : 5%
    Max hold         : 20 days

Outputs (all saved as PNG):
    final_01_equity_curve.png       — cumulative returns vs benchmarks
    final_02_drawdown.png           — underwater equity chart
    final_03_monthly_returns.png    — monthly returns heatmap
    final_04_annual_returns.png     — annual bar chart vs benchmarks
    final_05_trade_analysis.png     — trade P&L distribution + hold days
    final_06_rolling_metrics.png    — rolling Sharpe, vol, win rate
    final_07_signal_analysis.png    — gold return distribution + signal hit rate
    final_08_performance_table.png  — full metrics table as image
    final_09_regime_analysis.png    — performance by market regime
    final_10_summary_dashboard.png  — single-page summary dashboard

Run:  python final_backtest.py
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
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

L3_DIR  = r"C:\Gold ETF arbitrage\Layer 3_Signal generation"
OUT_DIR = r"C:\Gold ETF arbitrage\Final Backtest"

# Best validated config
MAG_THRESHOLD    = 0.0075   # 0.75%
SIGNAL_WINDOW    = 1
VOL_SCALE        = False
VOL_WINDOW       = 0
TRANSACTION_COST = 0.002
STOP_LOSS_PCT    = 0.05
MAX_HOLD_DAYS    = 20

# Colour palette — dark professional theme
BG       = "#0E1117"
BG2      = "#161B22"
BG3      = "#1C2333"
GOLD     = "#D4AF37"
BLUE     = "#2196F3"
GREEN    = "#2ECC71"
RED      = "#E74C3C"
ORANGE   = "#F39C12"
PURPLE   = "#9B59B6"
GRAY     = "#8B949E"
WHITE    = "#E6EDF3"
LTGRAY   = "#30363D"

DPI = 180


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=GRAY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(LTGRAY)
    ax.yaxis.grid(True, color=LTGRAY, linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=WHITE, fontsize=9, fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=GRAY, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=GRAY, fontsize=8)


def date_fmt(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, color=GRAY)


def pct_fmt(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"{x:.0f}%")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  Saved → {name}")


def metrics(ret: pd.Series, label="") -> dict:
    ann  = 252
    cum  = (1 + ret).cumprod()
    tot  = cum.iloc[-1] - 1
    ann_r = (1 + tot) ** (ann / max(len(ret), 1)) - 1
    ann_v = ret.std() * np.sqrt(ann)
    sr    = ann_r / ann_v if ann_v > 0 else 0
    peak  = cum.cummax()
    mdd   = (cum / peak - 1).min()
    wins  = ret[ret > 0]
    loss  = ret[ret < 0]
    active = ret[ret != 0]
    wr    = (active > 0).mean() if len(active) > 0 else 0
    profit_factor = wins.sum() / abs(loss.sum()) if loss.sum() != 0 else np.inf
    calmar = ann_r / abs(mdd) if mdd != 0 else 0
    sortino_v = ret[ret < 0].std() * np.sqrt(ann)
    sortino = ann_r / sortino_v if sortino_v > 0 else 0
    return {
        "label":          label,
        "total_return":   tot,
        "ann_return":     ann_r,
        "ann_vol":        ann_v,
        "sharpe":         sr,
        "sortino":        sortino,
        "calmar":         calmar,
        "max_drawdown":   mdd,
        "win_rate":       wr,
        "profit_factor":  profit_factor,
        "avg_win":        wins.mean() if len(wins) > 0 else 0,
        "avg_loss":       loss.mean() if len(loss) > 0 else 0,
        "best_day":       ret.max(),
        "worst_day":      ret.min(),
    }


# ─────────────────────────────────────────────────────────────────
# LOAD & RUN
# ─────────────────────────────────────────────────────────────────

def load_and_run() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = os.path.join(L3_DIR, "layer3_signals.parquet")
    df   = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} rows  "
          f"{df.index[0].date()} → {df.index[-1].date()}")

    gold_ret_1d = df["gold_idx"].pct_change()
    gold_ret_N  = df["gold_idx"].pct_change(SIGNAL_WINDOW)
    miner_ret   = df["miner_idx"].pct_change().fillna(0)
    gold_ret    = df["gold_idx"].pct_change().fillna(0)

    # Signal
    raw_dir = np.sign(gold_ret_N)
    raw_dir[gold_ret_N.abs() < MAG_THRESHOLD] = 0
    raw_signal = raw_dir.replace(0, np.nan).ffill().fillna(0)
    position   = raw_signal.shift(1).fillna(0)

    # Stop-loss / max-hold loop
    pos_arr      = position.values.copy()
    entry_level  = 0.0
    entry_sign   = 0
    hold_days    = 0
    miner_levels = df["miner_idx"].values

    trade_log = []
    for i in range(1, len(pos_arr)):
        cur_sign = int(np.sign(pos_arr[i]))
        if entry_sign == 0 and cur_sign != 0:
            entry_level = miner_levels[i]
            entry_sign  = cur_sign
            hold_days   = 0
            trade_log.append({"entry_i": i, "entry_sign": cur_sign,
                               "entry_level": entry_level})
        elif entry_sign != 0:
            hold_days += 1
            tr = (miner_levels[i] / entry_level - 1) * entry_sign if entry_level else 0
            if tr < -STOP_LOSS_PCT:
                if trade_log and trade_log[-1].get("exit_i") is None:
                    trade_log[-1].update({"exit_i": i, "exit_level": miner_levels[i],
                                          "exit_reason": "stop_loss"})
                pos_arr[i] = 0.0
                entry_sign = 0; entry_level = 0.0; hold_days = 0
                continue
            if hold_days >= MAX_HOLD_DAYS:
                entry_level = miner_levels[i]
                entry_sign  = int(np.sign(pos_arr[i]))
                hold_days   = 0
            if cur_sign != 0 and cur_sign != entry_sign:
                if trade_log and trade_log[-1].get("exit_i") is None:
                    trade_log[-1].update({"exit_i": i, "exit_level": miner_levels[i],
                                          "exit_reason": "flip"})
                entry_level = miner_levels[i]
                entry_sign  = cur_sign
                hold_days   = 0
                trade_log.append({"entry_i": i, "entry_sign": cur_sign,
                                   "entry_level": entry_level})

    position_final = pd.Series(pos_arr, index=df.index)
    pos_change     = position_final.diff().abs().fillna(0)
    net_pnl        = position_final * miner_ret - pos_change * TRANSACTION_COST
    cum_strat      = (1 + net_pnl).cumprod() * 100
    cum_gold       = (1 + gold_ret).cumprod() * 100
    cum_miner      = (1 + miner_ret).cumprod() * 100
    peak           = cum_strat.cummax()
    drawdown       = (cum_strat / peak) - 1

    results = pd.DataFrame({
        "gold_idx":    df["gold_idx"],
        "miner_idx":   df["miner_idx"],
        "gold_ret":    gold_ret,
        "miner_ret":   miner_ret,
        "position":    position_final,
        "net_pnl":     net_pnl,
        "cum_strat":   cum_strat,
        "cum_gold":    cum_gold,
        "cum_miner":   cum_miner,
        "drawdown":    drawdown,
        "trade_cost":  pos_change * TRANSACTION_COST,
    })

    # Build clean trade log
    trades = []
    for t in trade_log:
        if "exit_i" not in t:
            t["exit_i"] = len(miner_levels) - 1
            t["exit_level"] = miner_levels[-1]
            t["exit_reason"] = "open"
        pnl = (t["exit_level"] / t["entry_level"] - 1) * t["entry_sign"]
        trades.append({
            "entry_date":  df.index[t["entry_i"]],
            "exit_date":   df.index[min(t["exit_i"], len(df)-1)],
            "direction":   "LONG" if t["entry_sign"] == 1 else "SHORT",
            "entry_level": t["entry_level"],
            "exit_level":  t["exit_level"],
            "pnl_pct":     pnl * 100,
            "exit_reason": t.get("exit_reason", "flip"),
            "hold_days":   t["exit_i"] - t["entry_i"],
        })
    trades_df = pd.DataFrame(trades)

    return results, trades_df


# ─────────────────────────────────────────────────────────────────
# PLOT 1 — EQUITY CURVE
# ─────────────────────────────────────────────────────────────────

def plot_equity(res: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax_style(ax, "Cumulative Performance  (base = 100, log scale)")
    ax.semilogy(res.index, res["cum_strat"],  color=GREEN,  lw=2.2,
                label=f"Strategy  (SR={metrics(res['net_pnl'])['sharpe']:.2f})")
    ax.semilogy(res.index, res["cum_miner"],  color=BLUE,   lw=1.4,
                alpha=0.75, ls="--", label="B&H Miners")
    ax.semilogy(res.index, res["cum_gold"],   color=GOLD,   lw=1.4,
                alpha=0.75, ls="--", label="B&H Gold")
    ax.axhline(100, color=LTGRAY, lw=0.7, ls=":")
    ax.legend(facecolor=BG3, labelcolor=WHITE, fontsize=9, framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}"))
    date_fmt(ax)
    fig.suptitle("", y=0)
    fig.tight_layout()
    save(fig, "final_01_equity_curve.png")


# ─────────────────────────────────────────────────────────────────
# PLOT 2 — DRAWDOWN
# ─────────────────────────────────────────────────────────────────

def plot_drawdown(res: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)
    ax_style(ax, "Underwater Equity (Drawdown from Peak)")
    dd = res["drawdown"] * 100
    ax.fill_between(res.index, dd, 0, color=RED, alpha=0.55)
    ax.plot(res.index, dd, color=RED, lw=0.7, alpha=0.8)
    ax.axhline(0, color=LTGRAY, lw=0.8)

    # Annotate worst drawdown
    min_idx = dd.idxmin()
    ax.annotate(f"  Max DD: {dd.min():.1f}%\n  {min_idx.date()}",
                xy=(min_idx, dd.min()),
                xytext=(min_idx, dd.min() - 3),
                color=WHITE, fontsize=8,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))

    pct_fmt(ax)
    date_fmt(ax)
    ax.set_ylabel("Drawdown (%)", color=GRAY, fontsize=8)
    fig.tight_layout()
    save(fig, "final_02_drawdown.png")


# ─────────────────────────────────────────────────────────────────
# PLOT 3 — MONTHLY RETURNS HEATMAP
# ─────────────────────────────────────────────────────────────────

def plot_monthly_heatmap(res: pd.DataFrame) -> None:
    monthly = (1 + res["net_pnl"]).resample("ME").prod() - 1
    monthly.index = monthly.index.to_period("M")
    df_m = monthly.to_frame("ret")
    df_m["year"]  = df_m.index.year
    df_m["month"] = df_m.index.month

    pivot = df_m.pivot(index="year", columns="month", values="ret") * 100
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
    ax_style(ax, "Monthly Returns Heatmap (%)")

    cmap = LinearSegmentedColormap.from_list(
        "rg", [RED, BG3, GREEN], N=256)

    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 1)
    im   = ax.imshow(pivot.values, cmap=cmap, aspect="auto",
                     vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(12))
    ax.set_xticklabels(pivot.columns, color=GRAY, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, color=GRAY, fontsize=8)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                txt_color = WHITE if abs(v) > vmax * 0.4 else GRAY
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=7.5, color=txt_color, fontweight="bold")

    cb = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.ax.yaxis.set_tick_params(color=GRAY, labelsize=7)
    cb.ax.set_ylabel("Return (%)", color=GRAY, fontsize=7)

    fig.tight_layout()
    save(fig, "final_03_monthly_returns.png")


# ─────────────────────────────────────────────────────────────────
# PLOT 4 — ANNUAL RETURNS
# ─────────────────────────────────────────────────────────────────

def plot_annual(res: pd.DataFrame) -> None:
    annual_s = (1 + res["net_pnl"]).resample("YE").prod() - 1
    annual_g = (1 + res["gold_ret"]).resample("YE").prod() - 1
    annual_m = (1 + res["miner_ret"]).resample("YE").prod() - 1
    years    = annual_s.index.year

    x   = np.arange(len(years))
    w   = 0.26
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax_style(ax, "Annual Returns (%)")

    bars_s = ax.bar(x - w, annual_s * 100, w, label="Strategy",
                    color=[GREEN if v >= 0 else RED for v in annual_s], alpha=0.9)
    bars_g = ax.bar(x,     annual_g * 100, w, label="B&H Gold",
                    color=GOLD, alpha=0.65)
    bars_m = ax.bar(x + w, annual_m * 100, w, label="B&H Miners",
                    color=BLUE, alpha=0.65)

    ax.axhline(0, color=LTGRAY, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years, color=GRAY, fontsize=8)
    pct_fmt(ax)

    # Value labels on strategy bars
    for bar, val in zip(bars_s, annual_s * 100):
        ypos = bar.get_height() if val >= 0 else bar.get_height() - 1
        ax.text(bar.get_x() + bar.get_width()/2, ypos + (1 if val >= 0 else -4),
                f"{val:+.0f}%", ha="center", va="bottom",
                color=WHITE, fontsize=6.5)

    ax.legend(facecolor=BG3, labelcolor=WHITE, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    save(fig, "final_04_annual_returns.png")


# ─────────────────────────────────────────────────────────────────
# PLOT 5 — TRADE ANALYSIS
# ─────────────────────────────────────────────────────────────────

def plot_trade_analysis(trades: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    pnl = trades["pnl_pct"].dropna()

    # 5a: sorted P&L waterfall
    ax1 = fig.add_subplot(gs[0, :2])
    ax_style(ax1, "Trade P&L — Sorted (each bar = one trade)")
    sorted_pnl = pnl.sort_values().values
    colors     = [GREEN if v > 0 else RED for v in sorted_pnl]
    ax1.bar(range(len(sorted_pnl)), sorted_pnl, color=colors, alpha=0.8, width=1.0)
    ax1.axhline(0, color=WHITE, lw=0.8)
    ax1.set_xlabel("Trade #  (sorted by P&L)", color=GRAY, fontsize=8)
    ax1.set_ylabel("P&L (%)", color=GRAY, fontsize=8)
    pct_fmt(ax1)

    # 5b: P&L histogram
    ax2 = fig.add_subplot(gs[0, 2])
    ax_style(ax2, "P&L Distribution")
    ax2.hist(pnl[pnl >= 0], bins=25, color=GREEN, alpha=0.7, label="Wins")
    ax2.hist(pnl[pnl <  0], bins=25, color=RED,   alpha=0.7, label="Losses")
    ax2.axvline(pnl.mean(), color=WHITE, lw=1.2, ls="--",
                label=f"Mean={pnl.mean():.2f}%")
    ax2.axvline(0, color=LTGRAY, lw=0.8)
    ax2.legend(facecolor=BG3, labelcolor=WHITE, fontsize=7)
    ax2.set_xlabel("P&L (%)", color=GRAY, fontsize=8)

    # 5c: Hold days distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax_style(ax3, "Hold Days Distribution")
    ax3.hist(trades["hold_days"].dropna(), bins=20,
             color=BLUE, alpha=0.8, edgecolor=BG)
    ax3.axvline(trades["hold_days"].mean(), color=WHITE, lw=1.2, ls="--",
                label=f"Mean={trades['hold_days'].mean():.1f}d")
    ax3.legend(facecolor=BG3, labelcolor=WHITE, fontsize=7)
    ax3.set_xlabel("Days held", color=GRAY, fontsize=8)

    # 5d: Win/Loss by direction
    ax4 = fig.add_subplot(gs[1, 1])
    ax_style(ax4, "Avg P&L by Direction")
    dirs   = trades.groupby("direction")["pnl_pct"]
    labels = list(dirs.groups.keys())
    means  = [dirs.get_group(d).mean() for d in labels]
    clrs   = [GREEN if m > 0 else RED for m in means]
    ax4.bar(labels, means, color=clrs, alpha=0.85, width=0.5)
    ax4.axhline(0, color=LTGRAY, lw=0.8)
    pct_fmt(ax4)
    for lbl, m in zip(labels, means):
        ax4.text(lbl, m + (0.05 if m >= 0 else -0.15),
                 f"{m:+.2f}%", ha="center", color=WHITE, fontsize=8)

    # 5e: Exit reason breakdown
    ax5 = fig.add_subplot(gs[1, 2])
    ax_style(ax5, "Exit Reasons")
    reason_cnt = trades["exit_reason"].value_counts()
    clrs_pie   = [GREEN, GOLD, RED, BLUE, PURPLE][:len(reason_cnt)]
    wedges, texts, autotexts = ax5.pie(
        reason_cnt.values,
        labels=reason_cnt.index,
        autopct="%1.0f%%",
        colors=clrs_pie,
        startangle=90,
        textprops={"color": WHITE, "fontsize": 7},
    )
    for at in autotexts:
        at.set_color(BG)
        at.set_fontsize(7)
    ax5.set_facecolor(BG2)

    fig.tight_layout()
    save(fig, "final_05_trade_analysis.png")


# ─────────────────────────────────────────────────────────────────
# PLOT 6 — ROLLING METRICS
# ─────────────────────────────────────────────────────────────────

def plot_rolling(res: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor=BG,
                             sharex=True)
    fig.subplots_adjust(hspace=0.3)

    pnl  = res["net_pnl"]
    w252 = 252

    # Rolling Sharpe
    roll_sr = pnl.rolling(w252).apply(
        lambda x: x.mean() / x.std() * np.sqrt(w252) if x.std() > 0 else 0,
        raw=True)
    ax1 = axes[0]
    ax_style(ax1, "Rolling 1-Year Sharpe Ratio")
    ax1.plot(res.index, roll_sr, color=GREEN, lw=1.3)
    ax1.fill_between(res.index, roll_sr, 0,
                     where=roll_sr >= 0, color=GREEN, alpha=0.15)
    ax1.fill_between(res.index, roll_sr, 0,
                     where=roll_sr < 0,  color=RED,   alpha=0.15)
    ax1.axhline(0, color=LTGRAY, lw=0.8)
    ax1.axhline(1, color=GOLD,   lw=0.7, ls="--", alpha=0.7,
                label="SR = 1.0")
    ax1.legend(facecolor=BG3, labelcolor=WHITE, fontsize=8)
    ax1.set_ylabel("Sharpe", color=GRAY, fontsize=8)

    # Rolling Volatility
    roll_vol = pnl.rolling(w252).std() * np.sqrt(w252) * 100
    ax2 = axes[1]
    ax_style(ax2, "Rolling 1-Year Annualised Volatility (%)")
    ax2.plot(res.index, roll_vol, color=ORANGE, lw=1.3)
    ax2.fill_between(res.index, roll_vol, 0, color=ORANGE, alpha=0.12)
    pct_fmt(ax2)
    ax2.set_ylabel("Volatility (%)", color=GRAY, fontsize=8)

    # Rolling Win Rate (63-day)
    active = (res["net_pnl"] != 0).astype(float)
    wins   = (res["net_pnl"] > 0).astype(float)
    roll_wr = wins.rolling(63).sum() / active.rolling(63).sum().replace(0, np.nan) * 100
    ax3 = axes[2]
    ax_style(ax3, "Rolling 63-Day Win Rate (%)")
    ax3.plot(res.index, roll_wr, color=PURPLE, lw=1.3)
    ax3.axhline(50, color=GOLD, lw=0.7, ls="--", alpha=0.7, label="50%")
    ax3.fill_between(res.index, roll_wr, 50,
                     where=roll_wr >= 50, color=GREEN,  alpha=0.12)
    ax3.fill_between(res.index, roll_wr, 50,
                     where=roll_wr <  50, color=RED,    alpha=0.12)
    ax3.legend(facecolor=BG3, labelcolor=WHITE, fontsize=8)
    pct_fmt(ax3)
    ax3.set_ylabel("Win Rate (%)", color=GRAY, fontsize=8)
    date_fmt(ax3)

    save(fig, "final_06_rolling_metrics.png")


# ─────────────────────────────────────────────────────────────────
# PLOT 7 — SIGNAL ANALYSIS
# ─────────────────────────────────────────────────────────────────

def plot_signal_analysis(res: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs  = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    gold_ret = res["gold_ret"] * 100

    # 7a: Gold return distribution with signal threshold
    ax1 = fig.add_subplot(gs[0, :])
    ax_style(ax1, f"Gold Daily Return Distribution  "
             f"(signal triggered when |ret| > {MAG_THRESHOLD*100:.2f}%)")
    traded   = gold_ret[gold_ret.abs() > MAG_THRESHOLD * 100]
    filtered = gold_ret[gold_ret.abs() <= MAG_THRESHOLD * 100]
    bins = np.linspace(gold_ret.quantile(0.01), gold_ret.quantile(0.99), 60)
    ax1.hist(filtered, bins=bins, color=GRAY,  alpha=0.6, label="Filtered (no trade)")
    ax1.hist(traded[traded > 0],  bins=bins, color=GREEN, alpha=0.7, label="Long signal")
    ax1.hist(traded[traded < 0],  bins=bins, color=RED,   alpha=0.7, label="Short signal")
    ax1.axvline( MAG_THRESHOLD * 100, color=WHITE, lw=1.0, ls="--")
    ax1.axvline(-MAG_THRESHOLD * 100, color=WHITE, lw=1.0, ls="--")
    ax1.legend(facecolor=BG3, labelcolor=WHITE, fontsize=9)
    ax1.set_xlabel("Gold Daily Return (%)", color=GRAY, fontsize=8)
    ax1.set_ylabel("Frequency", color=GRAY, fontsize=8)

    # 7b: Signal hit rate — does the sign of gold_ret predict miner_ret next day?
    ax2 = fig.add_subplot(gs[1, 0])
    ax_style(ax2, "Predictive Hit Rate by Gold Return Bucket")
    buckets = pd.cut(gold_ret,
                     bins=[-np.inf, -1.5, -0.75, 0, 0.75, 1.5, np.inf],
                     labels=["<-1.5%", "-1.5 to -0.75%", "-0.75% to 0",
                              "0 to 0.75%", "0.75% to 1.5%", ">1.5%"])
    miner_next = res["miner_ret"].shift(-1) * 100
    hit_rate = []
    bucket_labels = []
    for b in buckets.cat.categories:
        mask = buckets == b
        if mask.sum() == 0:
            continue
        gr = gold_ret[mask]
        mr = miner_next[mask]
        # hit = gold direction matches miner next-day direction
        aligned = (np.sign(gr) == np.sign(mr)).mean() * 100
        hit_rate.append(aligned)
        bucket_labels.append(str(b))

    colors_hr = [GREEN if h > 50 else RED for h in hit_rate]
    ax2.barh(bucket_labels, hit_rate, color=colors_hr, alpha=0.85)
    ax2.axvline(50, color=WHITE, lw=1.0, ls="--", label="50% baseline")
    ax2.legend(facecolor=BG3, labelcolor=WHITE, fontsize=8)
    ax2.set_xlabel("Hit Rate (%)", color=GRAY, fontsize=8)
    pct_fmt(ax2, axis="x")

    # 7c: Signal frequency over time
    ax3 = fig.add_subplot(gs[1, 1])
    ax_style(ax3, "Signal Frequency — Annual Trade Count")
    trades_per_year = (res["position"].diff().abs() > 0).resample("YE").sum()
    ax3.bar(trades_per_year.index.year, trades_per_year.values,
            color=BLUE, alpha=0.8)
    ax3.set_xlabel("Year", color=GRAY, fontsize=8)
    ax3.set_ylabel("Trades", color=GRAY, fontsize=8)
    for x, y in zip(trades_per_year.index.year, trades_per_year.values):
        ax3.text(x, y + 1, str(int(y)), ha="center", color=WHITE, fontsize=7)

    fig.tight_layout()
    save(fig, "final_07_signal_analysis.png")


# ─────────────────────────────────────────────────────────────────
# PLOT 8 — PERFORMANCE TABLE
# ─────────────────────────────────────────────────────────────────

def plot_performance_table(res: pd.DataFrame, trades: pd.DataFrame) -> None:
    m_s = metrics(res["net_pnl"],  "Strategy")
    m_g = metrics(res["gold_ret"], "B&H Gold")
    m_m = metrics(res["miner_ret"],"B&H Miners")

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    rows = [
        ("Total Return",      "{:.1%}", "total_return"),
        ("Ann. Return",       "{:.1%}", "ann_return"),
        ("Ann. Volatility",   "{:.1%}", "ann_vol"),
        ("Sharpe Ratio",      "{:.2f}", "sharpe"),
        ("Sortino Ratio",     "{:.2f}", "sortino"),
        ("Calmar Ratio",      "{:.2f}", "calmar"),
        ("Max Drawdown",      "{:.1%}", "max_drawdown"),
        ("Win Rate",          "{:.1%}", "win_rate"),
        ("Profit Factor",     "{:.2f}", "profit_factor"),
        ("Avg Win",           "{:.2%}", "avg_win"),
        ("Avg Loss",          "{:.2%}", "avg_loss"),
        ("Best Day",          "{:.2%}", "best_day"),
        ("Worst Day",         "{:.2%}", "worst_day"),
    ]

    col_labels = ["Metric", "Strategy", "B&H Gold", "B&H Miners"]
    table_data = []
    for label, fmt, key in rows:
        vs = fmt.format(m_s[key])
        vg = fmt.format(m_g[key])
        vm = fmt.format(m_m[key])
        table_data.append([label, vs, vg, vm])

    # Extra trade stats
    table_data.append(["— — —", "— — —", "— — —", "— — —"])
    table_data.append(["Total Trades",   str(len(trades)), "—", "—"])
    table_data.append(["Avg Hold (days)",f"{trades['hold_days'].mean():.1f}", "—", "—"])
    table_data.append(["Best Trade",     f"{trades['pnl_pct'].max():.2f}%", "—", "—"])
    table_data.append(["Worst Trade",    f"{trades['pnl_pct'].min():.2f}%", "—", "—"])

    tbl = ax.table(
        cellText   = table_data,
        colLabels  = col_labels,
        cellLoc    = "center",
        loc        = "center",
        bbox       = [0.02, 0.0, 0.96, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(LTGRAY)
        if row == 0:
            cell.set_facecolor(BG3)
            cell.set_text_props(color=GOLD, fontweight="bold")
        elif table_data[row-1][0].startswith("—"):
            cell.set_facecolor(LTGRAY)
            cell.set_text_props(color=LTGRAY)
        elif col == 0:
            cell.set_facecolor(BG3)
            cell.set_text_props(color=GRAY)
        elif col == 1:
            # Strategy column — colour-code positive metrics
            cell.set_facecolor(BG2)
            txt = table_data[row-1][col]
            try:
                val = float(txt.replace("%","").replace(",",""))
                is_negative_metric = table_data[row-1][0] in [
                    "Max Drawdown", "Avg Loss", "Worst Day", "Worst Trade"]
                if is_negative_metric:
                    cell.set_text_props(color=RED if val < 0 else WHITE)
                else:
                    cell.set_text_props(color=GREEN if val > 0 else RED)
            except:
                cell.set_text_props(color=WHITE)
        else:
            cell.set_facecolor(BG2)
            cell.set_text_props(color=GRAY)

    ax.set_title("Performance Metrics — Final Validated Strategy\n"
                 f"Config: mag={MAG_THRESHOLD*100:.2f}%  win=1d  "
                 f"txn={TRANSACTION_COST*100:.2f}%/leg  "
                 f"stop={STOP_LOSS_PCT*100:.0f}%  max_hold={MAX_HOLD_DAYS}d",
                 color=WHITE, fontsize=11, pad=12)

    fig.tight_layout()
    save(fig, "final_08_performance_table.png")


# ─────────────────────────────────────────────────────────────────
# PLOT 9 — REGIME ANALYSIS
# ─────────────────────────────────────────────────────────────────

def plot_regime_analysis(res: pd.DataFrame) -> None:
    regimes = {
        "Bear gold\n2012–2015":   ("2012-01-04", "2015-12-31"),
        "Flat gold\n2016–2018":   ("2016-01-01", "2018-12-31"),
        "Gold rally\n2019–2020":  ("2019-01-01", "2020-12-31"),
        "Volatile\n2021–2022":    ("2021-01-01", "2022-12-31"),
        "New ATH\n2023–2024":     ("2023-01-01", "2024-12-31"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor=BG)

    # 9a: Sharpe per regime
    ax1 = axes[0]
    ax_style(ax1, "Sharpe Ratio by Gold Regime")
    names, sharpes, ann_rets, mdds = [], [], [], []
    for name, (s, e) in regimes.items():
        sub  = res.loc[s:e]
        if len(sub) < 50:
            continue
        m    = metrics(sub["net_pnl"])
        names.append(name)
        sharpes.append(m["sharpe"])
        ann_rets.append(m["ann_return"] * 100)
        mdds.append(m["max_drawdown"] * 100)

    colors_r = [GREEN if s > 0 else RED for s in sharpes]
    bars = ax1.bar(names, sharpes, color=colors_r, alpha=0.85, width=0.55)
    ax1.axhline(0, color=LTGRAY, lw=0.8)
    for bar, v in zip(bars, sharpes):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 v + (0.05 if v >= 0 else -0.15),
                 f"{v:.2f}", ha="center", color=WHITE, fontsize=8)
    ax1.set_ylabel("Sharpe Ratio", color=GRAY, fontsize=8)
    ax1.tick_params(axis="x", labelsize=7, colors=GRAY)

    # 9b: Ann Return vs MDD per regime (scatter)
    ax2 = axes[1]
    ax_style(ax2, "Ann Return vs Max Drawdown by Regime")
    for i, (name, ret, mdd) in enumerate(zip(names, ann_rets, mdds)):
        color = GREEN if ret > 0 else RED
        ax2.scatter(mdd, ret, color=color, s=120, zorder=3)
        ax2.annotate(name, (mdd, ret),
                     textcoords="offset points", xytext=(6, 3),
                     fontsize=7, color=WHITE)
    ax2.axhline(0, color=LTGRAY, lw=0.8, ls="--")
    ax2.axvline(0, color=LTGRAY, lw=0.8, ls="--")
    ax2.set_xlabel("Max Drawdown (%)", color=GRAY, fontsize=8)
    ax2.set_ylabel("Ann. Return (%)", color=GRAY, fontsize=8)
    pct_fmt(ax2, axis="x")
    pct_fmt(ax2, axis="y")

    fig.tight_layout()
    save(fig, "final_09_regime_analysis.png")


# ─────────────────────────────────────────────────────────────────
# PLOT 10 — SUMMARY DASHBOARD
# ─────────────────────────────────────────────────────────────────

def plot_dashboard(res: pd.DataFrame, trades: pd.DataFrame) -> None:
    m = metrics(res["net_pnl"])

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    gs  = GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.35,
                   top=0.88, bottom=0.06, left=0.06, right=0.97)

    # Title
    fig.text(0.5, 0.945, "Gold / A-Share Miner Arbitrage — Final Strategy",
             ha="center", va="top", color=GOLD, fontsize=16, fontweight="bold")
    fig.text(0.5, 0.915,
             f"Config: |gold_ret| > {MAG_THRESHOLD*100:.2f}%  ·  "
             f"1-day signal  ·  T+1 execution  ·  "
             f"txn={TRANSACTION_COST*100:.2f}%/leg  ·  "
             f"stop={STOP_LOSS_PCT*100:.0f}%  ·  "
             f"2012–2024  ·  7 A-share miners",
             ha="center", va="top", color=GRAY, fontsize=9)

    # KPI boxes (top row)
    kpis = [
        ("Sharpe Ratio",   f"{m['sharpe']:.2f}",      GREEN),
        ("Ann. Return",    f"{m['ann_return']:.1%}",   GREEN),
        ("Max Drawdown",   f"{m['max_drawdown']:.1%}", RED),
        ("Win Rate",       f"{m['win_rate']:.1%}",     GREEN if m['win_rate'] > 0.5 else ORANGE),
        ("Total Trades",   str(len(trades)),            BLUE),
        ("Profit Factor",  f"{m['profit_factor']:.2f}", GREEN),
        ("Sortino",        f"{m['sortino']:.2f}",       GREEN),
        ("Calmar",         f"{m['calmar']:.2f}",        GREEN),
    ]

    for i, (label, value, color) in enumerate(kpis):
        ax_k = fig.add_subplot(gs[0, i % 4])
        ax_k.set_facecolor(BG3)
        for spine in ax_k.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.5)
        ax_k.set_xticks([]); ax_k.set_yticks([])
        ax_k.text(0.5, 0.62, value, ha="center", va="center",
                  transform=ax_k.transAxes,
                  color=color, fontsize=20, fontweight="bold")
        ax_k.text(0.5, 0.22, label, ha="center", va="center",
                  transform=ax_k.transAxes,
                  color=GRAY, fontsize=8)
        if i == 3:   # start second row of KPIs
            pass     # handled by i % 4

    # Row 2 — equity + drawdown
    ax_eq = fig.add_subplot(gs[1, :3])
    ax_style(ax_eq, "Cumulative Return (log scale)")
    ax_eq.semilogy(res.index, res["cum_strat"],  color=GREEN, lw=2.0, label="Strategy")
    ax_eq.semilogy(res.index, res["cum_miner"],  color=BLUE,  lw=1.2, alpha=0.6,
                   ls="--", label="B&H Miners")
    ax_eq.semilogy(res.index, res["cum_gold"],   color=GOLD,  lw=1.2, alpha=0.6,
                   ls="--", label="B&H Gold")
    ax_eq.axhline(100, color=LTGRAY, lw=0.5, ls=":")
    ax_eq.legend(facecolor=BG3, labelcolor=WHITE, fontsize=8)
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    date_fmt(ax_eq)

    ax_dd = fig.add_subplot(gs[1, 3])
    ax_style(ax_dd, "Drawdown")
    ax_dd.fill_between(res.index, res["drawdown"]*100, 0,
                       color=RED, alpha=0.55)
    pct_fmt(ax_dd)
    date_fmt(ax_dd)

    # Row 3 — monthly heatmap condensed + rolling SR + trade dist
    ax_ann = fig.add_subplot(gs[2, :2])
    ax_style(ax_ann, "Annual Returns (%)")
    ann_s = (1 + res["net_pnl"]).resample("YE").prod() - 1
    ann_g = (1 + res["gold_ret"]).resample("YE").prod() - 1
    x  = np.arange(len(ann_s))
    ax_ann.bar(x - 0.2, ann_s * 100, 0.35,
               color=[GREEN if v >= 0 else RED for v in ann_s],
               alpha=0.9, label="Strategy")
    ax_ann.bar(x + 0.2, ann_g * 100, 0.35, color=GOLD, alpha=0.55,
               label="B&H Gold")
    ax_ann.set_xticks(x)
    ax_ann.set_xticklabels(ann_s.index.year, color=GRAY, fontsize=7)
    ax_ann.axhline(0, color=LTGRAY, lw=0.8)
    pct_fmt(ax_ann)
    ax_ann.legend(facecolor=BG3, labelcolor=WHITE, fontsize=7)

    ax_sr = fig.add_subplot(gs[2, 2])
    ax_style(ax_sr, "Rolling 1Y Sharpe")
    roll_sr = res["net_pnl"].rolling(252).apply(
        lambda x: x.mean()/x.std()*np.sqrt(252) if x.std()>0 else 0, raw=True)
    ax_sr.plot(res.index, roll_sr, color=GREEN, lw=1.2)
    ax_sr.fill_between(res.index, roll_sr, 0,
                       where=roll_sr >= 0, color=GREEN, alpha=0.12)
    ax_sr.fill_between(res.index, roll_sr, 0,
                       where=roll_sr < 0,  color=RED,   alpha=0.12)
    ax_sr.axhline(0, color=LTGRAY, lw=0.8)
    date_fmt(ax_sr)

    ax_td = fig.add_subplot(gs[2, 3])
    ax_style(ax_td, "Trade P&L Distribution")
    pnl = trades["pnl_pct"].dropna()
    ax_td.hist(pnl[pnl >= 0], bins=20, color=GREEN, alpha=0.75)
    ax_td.hist(pnl[pnl <  0], bins=20, color=RED,   alpha=0.75)
    ax_td.axvline(0, color=WHITE, lw=0.8)
    ax_td.set_xlabel("P&L (%)", color=GRAY, fontsize=8)

    save(fig, "final_10_summary_dashboard.png")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  FINAL STRATEGY BACKTEST")
    print(f"  Config: mag={MAG_THRESHOLD*100:.2f}%  win={SIGNAL_WINDOW}d  "
          f"txn={TRANSACTION_COST*100:.2f}%  stop={STOP_LOSS_PCT*100:.0f}%")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n  Running backtest ...")
    res, trades = load_and_run()

    m = metrics(res["net_pnl"])
    print(f"\n  Quick stats:")
    print(f"    Sharpe      : {m['sharpe']:.3f}")
    print(f"    Ann Return  : {m['ann_return']:.1%}")
    print(f"    Max DD      : {m['max_drawdown']:.1%}")
    print(f"    Win Rate    : {m['win_rate']:.1%}")
    print(f"    Trades      : {len(trades)}")

    print("\n  Generating charts ...")
    plot_equity(res)
    plot_drawdown(res)
    plot_monthly_heatmap(res)
    plot_annual(res)
    plot_trade_analysis(trades)
    plot_rolling(res)
    plot_signal_analysis(res)
    plot_performance_table(res, trades)
    plot_regime_analysis(res)
    plot_dashboard(res, trades)

    print(f"\n  All 10 PNGs saved to:\n  {OUT_DIR}")
    print("\n  Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith(".png"):
            size = os.path.getsize(os.path.join(OUT_DIR, f)) // 1024
            print(f"    {f:<45} {size:>4} KB")


if __name__ == "__main__":
    main()
