# Gold / A-Share Miner Arbitrage Backtest

A systematic backtesting framework that exploits the **T+1 lag** between COMEX gold futures and A-share gold mining stocks. When gold closes in New York, A-share miners in Shanghai have not yet had a chance to react — this creates a structurally exploitable one-day lag.

---

## Strategy Summary

| Parameter | Value |
|---|---|
| Signal | `sign(gold_return_T)` → long/short miners on Day T+1 |
| Magnitude filter | Only trade when `|gold_ret| > 0.75%` |
| Universe | 7 A-share gold miners (reserves-weighted index) |
| Execution | T+1 close, no lookahead bias |
| Transaction cost | 0.20% per leg |
| Stop-loss | 5% per trade |
| Backtest period | 2012–2024 (13 years) |

### Why the lag exists

COMEX gold settles at ~13:30 ET (18:30 UTC). A-shares open at 09:30 CST the next morning (01:30 UTC). There is a structural 6+ hour gap — gold's Day T close is fully observable before A-share miners open on Day T+1.

---

## Results
<img width="2503" height="1036" alt="final_01_equity_curve" src="https://github.com/user-attachments/assets/663801bf-2d2a-4c6d-8401-bcd538e428ef" />
<img width="2502" height="1063" alt="final_04_annual_returns" src="https://github.com/user-attachments/assets/fe0d6c52-deed-4e99-8c61-6511d2acdfa3" />
<img width="2259" height="1243" alt="final_03_monthly_returns" src="https://github.com/user-attachments/assets/2ef59dc3-afea-4f8e-8c5e-5b4f8f86db2c" />
<img width="2503" height="886" alt="final_02_drawdown" src="https://github.com/user-attachments/assets/dbbb77e9-7509-45f1-8916-d3db0dfd7e08" />
<img width="2080" height="1491" alt="final_06_rolling_metrics" src="https://github.com/user-attachments/assets/18c538d0-e4fd-44b7-a860-866af6f04864" />
<img width="2503" height="1244" alt="final_09_regime_analysis" src="https://github.com/user-attachments/assets/9ade8f82-982e-4cb7-b2d0-f5722829b8c5" />
<img width="2501" height="1243" alt="final_08_performance_table" src="https://github.com/user-attachments/assets/19661cd4-98ce-4c30-af16-124ec4c3c780" />
<img width="2153" height="1243" alt="final_07_signal_analysis" src="https://github.com/user-attachments/assets/d68def95-b6aa-4d61-afb3-dbf0bab47761" />

| Metric | Strategy | B&H Miners | B&H Gold |
|---|---|---|---|
| Sharpe Ratio | **2.41** | 0.34 | 0.25 |
| Ann. Return | **70.7%** | 10.8% | 3.8% |
| Max Drawdown | **-25.9%** | -55.8% | -41.2% |
| Win Rate | **52.9%** | 50.7% | 51.8% |

### Validation (Layer 6)

| Test | Result | Verdict |
|---|---|---|
| Train/Test split (2012–2019 train, 2020–2024 test) | OOS Sharpe = 3.44 | ✅ Pass |
| Probabilistic Sharpe Ratio (PSR) | 1.000 | ✅ Pass |
| Deflated Sharpe Ratio (DSR) | 0.977 | ✅ Pass |
| Walk-forward OOS/IS ratio | 0.919 | ✅ Pass |
| Rolling stability (% windows SR > 0) | 100% | ✅ Pass |
| Permutation test (p-value) | 0.0000, Z = 10.28 | ✅ Pass |

**4/4 overfitting tests passed. Permutation test confirms the edge is the T+1 lag signal, not gold's bull trend.**

---

## Project Structure

```
.
├── layer1_data.py          # Data ingestion — local CSVs + USDCNY FX
├── layer2_index.py         # Reserves-weighted miner index construction
├── layer3_signal.py        # Cross-correlation lag analysis + z-score signal
├── layer4_backtest.py      # Backtest engine — directional signal, T+1 execution
├── layer5_analysis.py      # Parameter sweep — magnitude filter, signal window
├── layer6_validation.py    # OOS validation — train/test split, WFO, stability
├── final_backtest.py       # Final strategy run — 10 performance charts
├── permutation_test.py     # Permutation test — isolates lag edge vs trend
├── requirements.txt
└── data/                   # CSV files (gitignored — see Data section)
```

Each layer is fully standalone and saves its output as a `.parquet` file for the next layer to consume. You can run them in order or jump to any layer if the upstream parquet already exists.

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Data

Place the following CSV files in your data folder (standard yfinance export format: `Date, Open, High, Low, Close, Volume`):

```
GC=F.csv          # COMEX gold futures (USD/oz)
USDCNY=X.csv      # USD/CNY FX rate (optional — falls back to 7.10)
601899.SS.csv     # Zijin Mining
600547.SS.csv     # Shandong Gold
600489.SS.csv     # Zhongjin Gold
000975.SZ.csv     # Shanjin Intl Gold (ex-Yintai)
002155.SZ.csv     # Hunan Gold
002273.SZ.csv     # Chifeng Jilong Gold
002237.SZ.csv     # Shandong Humon Smelting
```

Download via yfinance:

```python
import yfinance as yf

tickers = [
    "GC=F", "USDCNY=X",
    "601899.SS", "600547.SS", "600489.SS",
    "000975.SZ", "002155.SZ", "002273.SZ", "002237.SZ"
]
for t in tickers:
    yf.download(t, start="2012-01-01").to_csv(f"{t}.csv")
```

### Configure paths

Each layer file has a `DATA_DIR` / `OUT_DIR` constant at the top. Update these to match your folder structure before running.

---

## Running the Pipeline

Run layers in order. Each layer reads the previous layer's `.parquet` output.

```bash
python layer1_data.py        # → layer1_prices.parquet
python layer2_index.py       # → layer2_indices.parquet
python layer3_signal.py      # → layer3_signals.parquet
python layer4_backtest.py    # → layer4_results.parquet  (directional signal reference)
python layer5_analysis.py    # → layer5_sweep_results.csv  (parameter sweep)
python layer6_validation.py  # → layer6_oos_summary.txt  (validation report)
python final_backtest.py     # → 10 performance PNGs
python permutation_test.py   # → permutation_test.png
```

---

## Universe

| Ticker | Name | Exchange | Gold Reserves (Moz) | Listed |
|---|---|---|---|---|
| 601899.SS | Zijin Mining | SSE | 53.0 | 2003 |
| 600547.SS | Shandong Gold | SSE | 35.0 | 2000 |
| 600489.SS | Zhongjin Gold | SSE | 12.0 | 2000 |
| 000975.SZ | Shanjin Intl Gold | SZSE | 8.0 | 1999 |
| 002155.SZ | Hunan Gold | SZSE | 7.8 | 2007 |
| 002273.SZ | Chifeng Jilong Gold | SZSE | 9.5 | 2010 |
| 002237.SZ | Shandong Humon Smelting | SZSE | 4.0 | 2008 |

The index is weighted by gold mineral reserves (million troy oz). Weights rebalance only when the reserves table is updated manually.

---

## Layer Details

### Layer 1 — Data Ingestion
Loads all CSVs, converts gold from USD/oz to CNY/oz using the FX rate, forward-fills up to 3 days across holiday mismatches between COMEX and A-share calendars. Any miner with fewer than 5 years of data is dropped automatically.

### Layer 2 — Index Construction
Computes reserves weights, rebases each miner and gold to 100 at the common start date (2012-01-04), and builds the weighted miner composite index. Saves the spread stats which reveal the long-run structural miner outperformance (+285% vs +63% for gold over 13 years).

### Layer 3 — Signal Generation
Two analyses run in parallel:
- **Cross-correlation**: confirms the best lag is 1 trading day (corr = 0.228 at lag=1 vs 0.174 at lag=0), stable across all rolling windows
- **Z-score of spread**: alternative mean-reversion signal (superseded by directional signal in Layer 4)

### Layer 4 — Backtest Engine
Directional T+1 signal: `sign(gold_ret_T)` → position on Day T+1. Always long or short. Includes stop-loss (5%), max hold (20 days), and circuit breaker (pause after -10% drawdown). Transaction costs charged at each position change.

### Layer 5 — Parameter Sweep
Tests 120 parameter combinations across magnitude filter (0%–1%), signal window (1–10 days), and vol scaling (on/off). Sharpe heatmap identifies `mag=0.75%, 1d, no vol scaling` as the optimal config. Key finding: magnitude filter above 0.5% dramatically improves Sharpe by filtering noise trades.

### Layer 6 — Out-of-Sample Validation
Three overfitting tests:
- **Train/test split**: 2012–2019 train, 2020–2024 test. Best IS config selected on training data only, then evaluated cold on test set.
- **Walk-forward optimisation**: re-optimises every 12 months on rolling 3-year window. OOS/IS ratio = 0.919.
- **Rolling stability**: 100% of 45 rolling 2-year windows showed positive Sharpe.

### Permutation Test
Shuffles gold return dates 1,000 times, preserving the return distribution but destroying temporal order. Real Sharpe = 2.41. Shuffled mean = -0.12. Z-score = 10.28, p = 0.0000. The lag signal contributes 105% of the edge — the bull trend alone is actually a slight drag due to transaction costs on noise trades.

---

## Key Findings

1. **The lag is real and structural** — confirmed by cross-correlation, stable at 1 day across 13 years and all rolling windows
2. **Magnitude filtering is the biggest lever** — filtering trades below 0.75% gold moves improves Sharpe from ~1.0 to 2.4 by removing noise
3. **The edge survives out-of-sample** — OOS Sharpe (3.44) actually exceeded IS Sharpe (1.78) on the 2020–2024 test period
4. **The edge is the lag, not the trend** — permutation test proves this definitively (p = 0.0000)
5. **Vol scaling adds nothing** — the magnitude filter already selects high-conviction signals

---

## Limitations & Next Steps

- **Currency**: gold uses a fixed FX fallback (7.10) if `USDCNY=X.csv` is not provided. For production, use real daily FX data — the CNY/USD rate moved from 6.3 to 7.3 over the backtest window.
- **Instrument**: miner index is a synthetic construct. Consider replacing with `518880.SS` (Huaxia Gold ETF) for a single liquid tradeable instrument.
- **Re-optimisation**: the WFO shows the optimal magnitude threshold shifts over time (0.30%–0.75%). Re-run Layer 5 on a rolling 3-year window every 12 months to update the threshold.
- **Shorting constraints**: A-share short selling has restrictions. Short legs may need to be replaced with inverse ETF positions or simply skipped (long-only variant).
- **Execution**: backtest uses Close prices. In live trading, T+1 entry at the Open is more realistic and should be tested separately.

---

## Disclaimer

This project is for research and educational purposes only. Past backtest performance does not guarantee future results. This is not financial advice.

---

## License

MIT
