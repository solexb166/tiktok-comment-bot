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
