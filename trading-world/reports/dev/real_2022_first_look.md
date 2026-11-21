# First look at real data (development only; inside the frozen "train" split, 2022)

Written 2026-11-21. These are development checks, not a stage test. The formal stage-7 test uses 2023 (validation),
2024–mid-2025 (test) and the sealed holdout, per `configs/data/splits.yaml`. Scripts: `real_probe.py` and `real_dev.py`
in this folder.

## 1. Predictability probe

Mid prices, January–October 2022. Linear prediction from the last three returns, fitted on the first 70% and checked
on the rest.

- **Lags barely predict.** Out-of-sample R² is about 0 (−0.006 to +0.002). Predicted moves are 1–3% of the trading
  cost at 1-minute decisions and below the cost at 15 minutes. At 1 hour they reach 0.8–1.1× the cost for EUR/USD,
  USD/JPY and Nasdaq, but with R² of about 0 or negative.
- **Time of day.** Hourly mean returns repeat between the two periods for EUR/USD (correlation 0.54) and USD/JPY
  (0.61). 2022 was a strong-dollar year, so part of this may be the trend concentrated in some sessions.
- **Costs.** Half the spread relative to the per-bar volatility at 1 minute is 0.15 (EUR/USD), 0.50 (gold) and 1.79
  (bitcoin).

## 2. Version 5 agents, hourly decisions

Each agent learned from January–July 2022 and traded August–October 2022 in Environment A, with real spreads,
slippage and swaps. Runs are 500 decisions each.

| Instrument | Regimes | Median size | Growth per run (each run) |
|---|---|---|---|
| EUR/USD | 3 | 20× | −0.27 (+0.46, −0.63, −0.41, −0.48) |
| USD/JPY | 3 | 28× | +0.53 (−0.33, +0.47, +1.29, +0.70) |
| Gold | 2 | 16× | −0.60 (all four negative) |
| Nasdaq | 2 | 21× | −0.98 (all three negative) |

**Finding: version 5 is not safe on real data.** It sizes 15–28× on effects that are mostly noise.

- Per-regime trends and patterns are not corrected for estimation error. Only the single-regime model is.
- The single-regime model's significance measure (Wald statistic about 10⁶) is wildly overconfident on heavy-tailed,
  clustered real returns. The model treats its return mixture as exact.

The synthetic markets could not reveal this, because their effects were large and real.

**Next:** version 6 with estimation risk measured robustly on the agent's own experience (block bootstrap) and
applied to every regime. The development loop runs on real 2022 data before the formal stage 7.

## 3. Version 6, same check (hourly; same learn and trade periods)

Version 6 adds a held-out check of predicted edges with block-bootstrap uncertainty, applied to every model
(branch `claude/phase3-m6`).

| Instrument | Held-out slope (se) | Kept | Result |
|---|---|---|---|
| EUR/USD | +0.55 (0.62) | 0 | flat, 0.00 per run |
| USD/JPY | +0.01 (0.44) | 0 | flat, 0.00 |
| Gold | −0.65 (0.48) | 0 | flat, 0.00 |
| Nasdaq | −0.99 (0.44) | 0 | flat, 0.00 |

No predicted edge held up out of sample, so version 6 stays out and no longer loses money. It also no longer catches
USD/JPY's 2022 trend, which version 5 rode by luck. The honest state: safe on real data, but with the last three
hourly returns as its only information it finds no edge in 2022 worth trading. Real opportunities, if any, need richer
information (time of day, longer horizons, other markets) and more history.
