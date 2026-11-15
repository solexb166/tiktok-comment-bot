# Simulator specification (Phase 1, as implemented)

Code references are to `src/tradeworld/`. All money is float64 USD internally; Cent accounts
display USC = USD x 100.

## Time and data
- Time = int64 UTC minutes since epoch. `MarketData` is a regular M1 grid; `times[t]` is the
  bar's open minute. `quoted[t, i]` marks real quotes; unquoted bars are flat at the last close
  (valuation only; never filled). (`data/market_data.py`)
- Observation bars (default 5 min) aggregate only completed M1 bars; decision at the close of
  row d sees bars up to `j = (d+1)//r - 1`. Window = last W bars (default 120). (`replay/replay.py`)
- Market features per bar: mid log return, (high-low)/close, (close-open)/close, spread/mid,
  quoted fraction, window-relative volume.

## Actions (`interface/actions.py`)
`G = 10^(log10_min + (u+1)/2 (log10_max - log10_min))`, primary range [-4, 4];
`w_i = G d_i / max(1, sum|d_j|)`. NaN/Inf or wrong shape -> invalid (positions unchanged).
Flags: `clipped_action`, `near_ceiling` (u > 0.95).

## Execution (`execution/engine.py`)
- Decision at close of row d; fill at the open of row d+1.
- Target units = `w * equity_d / (mid_d * conv_d)`; lot profiles round toward zero to `lot_step`,
  `< min_lot` -> 0; fractional deltas below `min_trade_notional_usd` are skipped (numerical).
- Buy at `ask_open * (1 + s)`, sell at `bid_open * (1 - s)`;
  `s = (base_bps + |N(0,1)| noise_bps)/1e4 + range_coef * range_frac`. Spread multiplier widens
  bid/ask around mid. Slippage draws happen in a fixed order (reductions, then increases).
- Closed market (unquoted fill bar) -> `market_closed` rejection.
- Reductions execute first. Increases execute only if post-trade used margin <= post-trade equity
  (atomic); else all are rejected (`insufficient_margin`). Broker max volume -> `max_volume`.
- Commission per side: `lots * commission_per_lot + notional * commission_bps / 1e4`.

## Accounts (`accounts/account.py`)
- Net position per instrument; average entry price; realized P&L on reduction converted at the
  exit conversion rate. Longs valued at bid, shorts at ask.
- `equity = balance + sum(unrealized)`; balance changes only through realized P&L, commissions,
  swaps and NBP resets.

## Margin (`accounts/margin.py`)
- Leverage-based rate `1 / min(account_leverage, tier_cap(equity), instrument max, HMR cap)`;
  fixed instruments use `fixed_margin_rate` (raised to the HMR minimum in windows).
- `account_leverage = inf` (Unlimited) is granted only if `UnlimitedEligibility` holds
  (equity below threshold, closed trades, volume); otherwise the fallback leverage applies.
- Margin uses mid price; margin level = equity / used margin.

## Swaps
Charged at the 17:00 New York rollover bar (21:00 UTC summer / 22:00 UTC winter), Monday to
Friday, on positions held after that bar's fills: `|units| * mid * conv * rate / 360 * mult`,
`mult = 3` on the instrument's triple-swap weekday. Swap-free instruments are skipped.

## Liquidation order per M1 bar (`execution/engine.py::_intrabar_checks`)
Each held instrument moves linearly from its open mark to its adverse extreme (longs: bid low;
shorts: ask high); conversion fixed at the bar open (approximation; conservative because the
extremes are assumed simultaneous).
1. Death if equity at the open <= 5% of initial -> close all at the open (`gap_through`).
2. Stop-out at the open (margin level <= level) -> close the most unprofitable position.
3. Otherwise bisection finds the earliest alpha of death or stop-out; death wins ties
   (`equity_floor`); stop-outs close one position at a time and the walk resumes.
4. After a full liquidation with negative balance: NBP profiles reset to 0 (`nbp_reset`);
   others keep the debt (`negative_balance`).
A 0% stop-out level can never fire before death, so it is skipped.

## Known approximations
Netting (not hedging) positions; intra-bar path linear and simultaneous; death liquidation on a
closed market fills at the valuation price (only reachable through swap charges); swaps from a
static snapshot; Exness spread calibration not yet applied (data spreads are used).
