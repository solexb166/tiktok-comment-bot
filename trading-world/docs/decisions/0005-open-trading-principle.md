# 0005: Open trading principle, and audit for hidden style bias

Status: principle adopted 2026-11-17 (owner instruction). Audit completed the same day. **No code,
rule or setting was changed by this record.** Findings that need changes are proposals and await
separate approval. The frozen rules (decisions 0001-0004) are unchanged.

## The principle

Agents operate in an open trading environment, like a person who enters the trading world and
learns to trade. We do not prescribe a trading style or strategy. No agent is told to scalp, swing
trade, use precise ("sniper") entries, trade particular sessions or timeframes, make a set number
of trades, hold for predetermined durations, follow indicators or strategies, or use mandatory
entry or exit patterns.

Agents learn for themselves when to enter and exit; whether to buy, sell or stay out; how long to
hold; how often to trade; how much exposure to take; how to respond to changing conditions; and
how to handle winning and losing positions. Different agents may develop different trading
personalities.

What we do provide, and keep: realistic market mechanics (prices, spreads, slippage, commissions,
swaps, gaps, margin, stop-out, lot rules), the survival rule (death at 5%), declared reward
functions, and evaluation. These are the world, not a strategy. Restrictions that indirectly force
a style are treated as defects: they are documented here and proposed for removal, never added.

Test for any future change: does it give the agent information, tools or realistic consequences
(acceptable), or does it decide for the agent what to do (not acceptable)?

## Audit (2026-11-17)

Scope: action space (`interface/actions.py`), simulator (`execution/engine.py`,
`accounts/account.py`), training environment (`learning/env.py`, `learning/features.py`,
`learning/train.py`), reward functions (`learning/rewards.py`) and evaluation
(`learning/gate.py`, `learning/quality.py`, `learning/diagnosis.py`).

### Findings that do shape trading style (proposals required)

| # | Finding | Evidence | Style it favours | Proposal (not implemented) |
|---|---|---|---|---|
| H1 | **Exploration noise forces churn during training.** PPO adds Gaussian noise to every action, every decision, so every position is re-sized at every decision while learning. | Real data, Jan 2022, 1 week of 5-minute decisions, 1x EUR/USD intended: deterministic 14 fills in 2,016 decisions and $0.004 costs; with training-level noise 12,843 fills and **$6.14 of $10 lost to costs**. | Teaches every agent that being in the market is ruinous whatever its intent: a bias toward not trading and against holding. Absent from evaluation (deterministic), so training and testing see different worlds. Irrelevant for S6 (no costs), important for real data. | P1: temporally correlated exploration (Stable-Baselines3 generalized state-dependent exploration, `use_sde`), which keeps the exploratory change in place for several decisions. This is a learning-algorithm setting, so it comes with a separate proposal. |
| H2 | **Positions cannot outlive an episode.** Every episode ends and resets the account. | Synthetic: 1,000 decisions (about 17 hours at 1 minute). Production plan: 2,016 decisions (1 week). | Agents never experience holding for weeks or months, which rules out position-trading styles. (Truncation is bootstrapped correctly, so this is a limit on experience, not a distorted reward.) | P2: much longer or continuing episodes (one continuous life, as already planned for evaluation, architecture §8.2), or random episode lengths. |
| H3 | **Short learning horizon.** Discount factor gamma = 0.99. | Influence halves every 69 decisions: 5.7 hours at 5-minute decisions, 69 minutes at 1-minute. | Consequences beyond about a day barely shape learning. This favours short-horizon behaviour over swing and position trading. | P3: a discount defined per unit of market time (for example a half-life of days, not hours), compared in an approved experiment. |
| H4 | **Short memory of the market.** The observation is the last 120 five-minute bars (10 hours); the policy has no memory. | `replay.py`, `features.py`; MLP policy. | The agent cannot see daily or weekly structure, so multi-day ideas cannot be formed. | P4: multi-timescale observation (for example 120 x M5 plus 120 x H1 plus 60 x D1 bars) and/or a recurrent policy, offered as an option rather than a requirement. |

### Limitations that narrow the strategy space (proposals required)

| # | Finding | Effect | Proposal |
|---|---|---|---|
| M1 | Fixed decision interval (5 minutes, approved decision 6). | No reaction faster than 5 minutes and no exit inside the interval. Any holding period that is a multiple of the interval is possible. | Let the interval differ between agents (1/5/15/60 minutes already configurable), or event-driven decisions. |
| M2 | Market orders only, filled at the next bar. No limit, stop or take-profit orders. | Price-level entries and exits (resting orders) are impossible; protection between decisions is impossible. | P6: optional order types in the action (limit and stop offsets the agent chooses itself). These are tools, not rules. |
| M3 | No time-of-day or day-of-week information (dates excluded by design). | Session behaviour can only be inferred indirectly from volume and spread. | Optional cyclical time-of-week features (no dates, so no leakage). |
| M4 | Absolute price level removed (window-local normalisation, by design). | Level-based behaviour (round numbers, older highs and lows) cannot be learned. | Relative-distance features (to recent daily or weekly highs and lows) rather than raw prices. |
| M5 | "Stay out" is not a distinct choice. Continuous outputs are almost never exactly zero. | Residual positions and constant small trades (S6 v1 agents changed position in 100% of decisions). | An exact flat region in the action mapping, or a policy that can choose "no position" or "keep position" as discrete options. |
| M6 | One learner type: PPO with a single-peaked Gaussian policy. | Tends to average between alternatives ("long big" versus "flat" becomes "long small"). | Already planned: SAC and evolutionary learners (architecture §8.3). |
| M7 | All agents share one network design, observation and hyperparameters. | Personalities can differ only through seeds and reward. | Planned population diversity (architecture §4.4): varied observations, decision intervals, learners. |

### Checked and found neutral (keep)

| Area | Finding |
|---|---|
| Rewards | All are computed from net-of-cost, marked-to-market equity. There are no trade-count, win-rate, market-presence or realized-only rewards, so there is no forced disposition (no pressure to close winners or hold losers). R3 (drawdown) and R4 (Sharpe) express declared risk attitudes. Different agents may be given different rewards, which supports different personalities. R1 (log growth) expresses a growth-optimal risk attitude. That is a risk preference, not a trading style. |
| Market frictions | Spread, slippage, commission, swaps at rollover, weekend gaps, margin, stop-out and lot rounding (environment B) are realistic consequences every trader faces. |
| Leverage | No behavioural cap. The log range of 0.0001x to 10,000x is a numerical safeguard well above any margin-feasible exposure. |
| Survival rule | Death at 5% is an approved experimental rule. It is a consequence, not a strategy. |
| Small-trade threshold | Differences below $0.001 are not traded. At $10 a constant position needed only 14 fills a week. At larger account sizes it would allow small rebalancing trades (negligible cost). Note: the threshold is absolute, so this behaviour depends on account size. |
| Evaluation | The synthetic-market criteria (S1-S7, including S6's leverage band and the decision-quality score) test the learning machinery against known right answers. They are not prescriptions for real markets. Real-market success criteria (architecture §9.4) judge outcomes against matched random agents and are style-neutral. |
| Diagnosis | Analysis only; never fed back into training or rewards. |

## Effect on Phase 2.5

S6 has no trading costs, uses 1-minute decisions, and its returns are independent from one decision
to the next. H1 to H4 therefore do not bias the leverage question Phase 2.5 asks, and Phase 2.5
can proceed unchanged. They do matter before agents train on real market data (Phase 3), where
H1 in particular would push every agent toward inactivity.
