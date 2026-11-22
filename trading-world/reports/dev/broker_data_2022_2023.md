# Owner's broker data (Exness via MT5), development on 2022–2023

Written 2026-11-22. This is development only: frozen `train` (2022) and `validation` (2023) splits. Test (2024 to
mid-2025) and the sealed holdout (mid-2025 onward) are untouched.

## The data

The owner exported it on 2026-11-22 with `scripts/mt5_export.py`; it is read only.

- **Broker symbols:** EURUSDm, GBPUSDm, USDJPYm, XAUUSDm, BTCUSDm, USTECm (Exness Standard).
- **Coverage:** M1 bars for every month from 2022-01 to 2026-10, with the broker's spread for each minute. Months hold
  about 29–31k bars (BTC about 45k).
- **Median spreads:** EUR/USD 0.75 bp, GBP/USD 0.74, USD/JPY 0.67, gold 0.85, Nasdaq 2.9, bitcoin 4.4.
- **Contract specifications:** saved per instrument in `symbol_info.json`.

## Rolling walk-forward

Version 6 with the forward check and the multi-timeframe view, hourly decisions. Each month from 2022-04 to 2023-12:
learn from all earlier data, then trade the month once.

| Instrument | Total log growth | Months up / down / flat |
|---|---|---|
| GBP/USD | −7.32 | 1 / 8 / 12 |
| EUR/USD | −3.43 | 6 / 10 / 5 |
| Gold | −0.40 | 3 / 2 / 16 |
| USD/JPY | +0.06 | 2 / 0 / 19 |
| Nasdaq | −0.30 | 2 / 2 / 17 |
| Bitcoin | −1.47 | 2 / 3 / 16 |

Over the six markets the total is −12.88: five lose and one is flat. On Nasdaq and bitcoin the forward check kept the robot out of most months; it traded only when an edge seemed to hold up, and most of those months still lost.

## Cost breakdown of one losing month

GBP/USD, May 2022, started with $10:

| Item | Amount |
|---|---|
| P&L before costs | −$3.88 |
| Spread paid | $2.30 |
| Slippage charged (Environment A's *assumed* model) | $3.20 |
| Total | −$9.38 |

At about 42× with hourly re-sizing, the edge did not persist and the costs alone used up 55% of the account.

## Facts from the owner's Exness demo trade (2026-11-22, account D-trial)

- **Spread.** Sold 0.06 BTCUSDm at 59,882.39. The immediate −$0.60 is the spread: about $9.9, or 1.7 bp at that moment.
- **Stop-out.** A 0.28% move against about 360× leverage closed the position after 3.5 minutes, at 09:35:53. Stop-out
  is effectively at about 0% margin level.
- **Negative balance protection is active.** A "D-NULL" credit of +$0.04 reset the negative balance to 0.
- **Commission and swap:** 0 on this trade.

## Conclusions and next design

On hourly decisions no edge survives both a changing market and costs. Next steps:

1. **Learned costs.** The agent learns its true cost per unit traded, slippage included, from its own fills, rather than
   assuming the spread is the only cost.
2. **Self-chosen timeframe.** The agent tries several decision speeds and keeps the one that held up best in its own
   forward check. There is no imposed trading frequency: owner's instruction, 2026-11-22.
3. **Pooled learning.** One agent learns across all six markets, giving more evidence per decision.
4. **Measured slippage.** Environment A's slippage is an assumption; real Exness slippage will be measured on the demo
   and replace it.

## Version 7: self-chosen speed, learned costs (2023, validation split)

Each month the agent replays its last 3 months as if live. It does this at 15-minute, hourly, 4-hour and daily speed,
first measuring the cost it actually paid per unit traded (spread + slippage), then replaying again charging that
cost. It keeps the speed that grew most. If none grew, it stays out. It then learns from all its history at that speed
and trades the month. There is no rule on how often to trade.

Fixes found while building it:

- Inputs that never vary at a speed (time of day at daily speed) are dropped.
- Inputs are put on a common scale.
- Inputs are held inside the range the agent has seen. Daily predictions of +156% a day had produced 175× bets.

| Market | Hourly version, 2023 | Version 7, 2023 | Months traded |
|---|---|---|---|
| EUR/USD | −1.784 | +0.443 | 1 (February, 15m, ~17×) |
| GBP/USD | −0.226 | −0.046 | 1 (December, 15m, ~2×) |
| USD/JPY | +0.026 | 0 | 0 |
| Gold | −0.515 | 0 | 0 |
| Bitcoin | −0.814 | 0 | 0 |
| Nasdaq | +0.049 | 0 | 0 |

Totals: the hourly version −3.26, version 7 +0.40. Version 7 stops the losses, mostly by staying out. It traded 2 of 72 market-months. Most months its replay found a
speed that grew, but when it relearned on all its history, the forward check rejected the edge. The edges are too weak
to trust.

## Pooled edge check across all six markets (`pool_check.py`)

Each market's returns and inputs are scaled by its own causal volatility, using the last 1440 quoted minutes. One ridge
model is fitted on 2022 and predicts 2023 one bar ahead.

| Speed | Out-of-sample slope ± block-bootstrap SE | Median predicted edge / median half spread (in σ) |
|---|---|---|
| 60m | −0.73 ± 0.42 | 0.022 / 0.051 |
| 240m | +0.13 ± 0.25 | 0.045 / 0.025 |
| 1440m | +0.20 ± 0.39 | 0.116 / 0.010 |

At hourly speed the edge is smaller than the cost and points the wrong way out of sample. At 4-hour and daily speed the
cost is small next to the predicted edge, but one year of training (about 1,900 pooled daily bars) is too little
evidence to separate an edge from noise. **The limiting factor at slow speeds is the length of history.** The next data
step is `scripts/mt5_export_history.py`: H1/H4/D1 bars for as many years as the broker holds. Everything before 2023 is
inside the frozen train split.

## Long history from the owner's MT5 (H1/H4/D1), pooled edge check (`pool_hist.py`)

The data arrived on 2026-11-22. FX and gold run from 2014-01, bitcoin from 2018-02, Nasdaq from 2019-07, all to
2026-11. There were no malformed bars. D1 bar spreads are often 0 (a bar records the minimum spread), so costs use the
median M1 spreads instead. The model learns on 2014–2022 (frozen train split) and predicts 2023 (validation) one bar
ahead. 2024 and later are never read. Each market's moves are in units of its own recent bar volatility. Inputs are the
last 3 moves, longer-window moves, the volatility ratio and the clock. The model is ridge (λ = 1), and the standard
error is a moving-block bootstrap (block 50, 400 resamples).

| Bars | Train / test bars | Out-of-sample slope ± SE (t) | Median pred / half spread (σ units) | By market (slope) |
|---|---|---|---|---|
| D1 | 11,754 / 1,911 | +0.56 ± 0.75 (0.7) | 0.034 / 0.009 | BTC +4.4, NAS +1.2, XAU +0.1, FX −0.8 to −1.5 |
| H4 | 54,010 / 10,159 | **+1.22 ± 0.57 (2.1)** | 0.016 / 0.021 | BTC +3.7, NAS +2.0, XAU +1.8, FX −0.6 to −1.0 |
| H1 | 202,765 / 39,047 | +0.94 ± 0.49 (1.9) | 0.010 / 0.041 | XAU +1.8, BTC +2.8, EUR +0.9, others ≤ 0.1 |

This is the first pooled prediction that holds up on unseen data, at 4-hour bars. Caveats:

- Three speeds were tested, so t ≈ 2 is weak.
- The edge comes from bitcoin, Nasdaq and gold, which show short-term continuation. 2023 was a strong trend year for
  bitcoin and Nasdaq.
- On FX the 2023 slope is negative.
- The typical 4-hour prediction is a little below the half spread, so only the stronger predictions are worth trading.

The model learned to keep trends going over 1–3 bars (positive coefficients on the last three moves). The 2023
validation year has now been looked at for this family. Further choices must not be tuned on it repeatedly: use
walk-forward inside 2014–2022 for development, and keep 2023 as a check.

### Year-by-year walk-forward inside the train split (`pool_walk.py`)

Same model, learned on all years before Y, predicting year Y (2017–2022; 2023 from the previous table).
Out-of-sample slopes:

| Bars | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | Positive years |
|---|---|---|---|---|---|---|---|---|
| D1 | −0.15 | −0.67 | +1.11 | +0.14 | −0.62 | −0.45 | +0.56 | 3/7 |
| H4 | −0.03 | −0.12 | +0.05 | +1.15 | +0.32 | −0.07 | +1.22 | 4/7 (strong only in 2020 and 2023) |
| H1 | n/a | −0.29 | +0.45 | +0.91 | +0.25 | +0.19 | +0.94 | 5/6 |

The 2023 result at 4-hour bars is not a stable edge: it appears only in the two strongest trend years. At hourly bars
there is a small, fairly consistent continuation effect, with correlations around 0.005–0.012. Its typical
prediction is about one quarter of the half spread, so it cannot pay for trading. Linear price-pattern inputs at these
speeds give no edge worth trading after costs.

Remaining sources of return to test next, all learned by the agent itself, with no imposed style:

- the long-run drift of each market, learned over the long history with estimation risk;
- volatility timing (size by predicted risk);
- the broker's swap (carry) on held positions.
