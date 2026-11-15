# Autonomous Multi-Agent Trading World — Phase 0 Architecture & Plan

Status: **Revision 2, for review.** It includes the decisions of 2026-11-15 (§12) and a researched Exness
specification (§5). No implementation code until this revision is approved.

Change log
- **r1**: initial architecture, critique and plan.
- **r2**: recorded decisions 1–7. Replaced generic broker assumptions with researched Exness
  specifications and a verification status for each fact (§5). Revised the $10-account analysis
  (Cent accounts change it substantially). Added the "Dukascopy price path + Exness cost model"
  data design (§5.6), a hardware-assessment step (§13), Stage D account-mixing issues (§5.5) and
  new open questions (§12.2).

---

## 0. Summary

The vision is buildable. The key conclusions after the Exness research:

1. **On an Exness Standard Cent account, a $10 agent can trade the three FX pairs even at L = 3.**
   One Cent lot is 0.01 of a standard lot, and the minimum is 0.01 Cent lots, so the smallest EUR/USD
   position is about €10 (≈ $11.6) of notional. With a $30 cap (L = 3 × $10), the agent can hold
   0, 1 or 2 minimum units per FX pair. **Position sizing is therefore very coarse: about 1.2× leverage
   per step.** Gold (smallest ≈ 0.01 oz ≈ $40) does not fit at L = 3 but does at L = 100. BTC and
   NAS100 availability on Cent accounts is contradictory in Exness's own documentation and must be
   verified on a demo account (§5.2).
2. **Exness's stop-out level is 0%, so the project's death rule (equity ≤ 5% of initial) always fires
   first.** Broker stop-out and negative balance protection only matter when a price gap jumps the
   account straight past the 5% floor.
3. **Agents train on fractional sizes but Stage D uses coarse Cent lots.** That is a deliberate
   train/evaluation mismatch. Stage D must measure how much of a learned behaviour survives
   quantization. A quantization-aware fine-tuning step is an optional experiment (§6.5).
4. **On historical data this is not a multi-agent market.** It is N independent price-taker
   experiments on a shared tape. Behavioural diversity comes from seeds, learners, rewards,
   observation settings and population-based evolution (§3.4).
5. **"No predefined trading knowledge" can only be approximated.** The observation encoding,
   architecture, decision interval and reward function are inductive biases. They are documented and
   varied experimentally (§4).
6. **With many agents, some will look profitable by luck.** Success criteria are pre-registered and
   tested against null distributions (§7.4).

---

## 1. Critique of the specification

| # | Issue | Why it matters | Resolution |
|---|---|---|---|
| 1 | $10 + L = 3 + real lot sizes | With standard lots nothing is tradable | **Decided:** fractional sizes for training; Exness Cent specs for Stage D (§5). FX is tradable on Cent at L = 3, coarsely |
| 2 | Normalized equity 1.0 | Lot minimums, per-lot costs and swaps are not scale-invariant | The simulator accounts in **real USD (and USC for Cent)**. Normalization applies only in observations and rewards |
| 3 | Multi-agent "world" on replay | Price-takers cannot interact | Accept the price-taker model. Diversity through seeds, learners, rewards and evolution |
| 4 | "No risk rules" vs. reward design | The reward encodes risk preference | Log-wealth baseline; penalty variants labelled as shaping (§6) |
| 5 | Target exposure re-evaluated every step | Implicit rebalancing rule plus churn costs | Execute only deltas ≥ one lot step (§5.7) |
| 6 | Decision frequency | M1 decisions are cost-dominated | **Decided:** 5-minute decisions by default; 1, 15 and 60 minutes configurable. M1 remains the fill and valuation clock |
| 7 | Observation window | 120 × M1 = only 2 hours | Proposed default: 120 bars **at the decision resolution** (5m → 10 hours), built from completed M1 bars. Ablations: 120 × M1, multi-resolution (§12.2) |
| 8 | Different trading hours | BTC 24/7; FX 24/5; indices and gold have daily breaks | UTC minute clock + per-instrument `tradeable` mask from Exness session tables; no forward-filled tradeable prices |
| 9 | Death at 5% vs. broker stop-out | **Exness stop-out = 0%**, so death fires first | Death rule as specified; Exness stop-out and NBP modelled for gaps (§5.4) |
| 10 | "Zombie" accounts | Alive, but cannot afford the minimum lot on any instrument | Track an `incapacitated` status and duration |
| 11 | Negative balance protection | A free option to gamble if combined with resets | **Decided:** Exness NBP in the broker profile, plus a separate no-NBP stress scenario |
| 12 | Financing | Historical swap rates unavailable | Current Exness swap tables from the demo account (§5.6), Exness swap-free status for majors modelled as a variant; documented approximation |
| 13 | Success criterion | Undefined | Pre-registered (§7.4) |
| 14 | Compute | Cloud session is ephemeral | **Decided:** develop locally first; hardware assessment before long runs (§13) |
| 15 | No real orders | Safety | No order-sending code path; the MetaApi adapter is restricted to read-only endpoints by an allowlist and a test |
| 16 | Train/serve skew | Live bars ≠ historical bars | Parity test on an overlapping period |
| 17 | Bid/ask data | Mid-only data makes costs fictional | **Decided:** Dukascopy bid/ask M1 from 2018, with Exness data for cost calibration |
| 18 | **New:** Stage D "Cent where supported, otherwise other Exness specs" | A real $10 account is **one** account of **one** type; it cannot be Cent for FX and Standard for BTC at once | Proposal in §5.5; needs your decision (§12.2) |
| 19 | **New:** Higher Margin Requirement (HMR) periods | Exness raises margins around high-impact news; simulating that needs a historical news calendar | Configurable HMR windows; calendar data is a later addition. At L = 3 margin never binds on Exness leverage, so this only affects L = 100 tests |
| 20 | **New:** Exness legal entity | Kenya/Jordan entities differ (stop-out 20%, leverage cap 1:400, BTCUSDc blocked for KE) | The broker profile is per entity; which entity applies to you is needed (§12.2) |

---

## 2. Is the experiment likely to "work"? (setting expectations)

Most agents will probably learn to stay flat, die, or capture drift (beta) that disappears out of
sample. At 5-minute decisions with Cent-account spreads, costs remain a large share of typical
moves. A null result is a valid finding, so the design makes it **measurable and trustworthy**: the
learner is first proven on synthetic markets with planted structure (§8.3), so that failure on real
data can be attributed to the market and not to a bug.

---

## 3. Architecture

### 3.1 Component map

```
                ┌──────────────── offline ───────────────┐
 Dukascopy M1 → │ 1 Market Data Engine (download, clean, │ → Parquet store (immutable, hashed)
 bid/ask        │   align, validate, session masks)      │
 Exness ticks → │   Cost-model calibration (spreads)     │ → calibrated cost tables
 Exness demo  → │   Spec snapshot (contract, swaps, ...) │ → versioned broker profile
 (read-only)    └────────────────────────────────────────┘
                                   │
                                   ▼
            ┌──────────── deterministic core (no learning) ───────────┐
            │ 2 Replay Engine: UTC clock, cursor, past-only windows   │
            │ 3 Execution Engine: bid/ask fills, latency, slippage,   │
            │   lot rounding, commissions                             │
            │ 4 Account Sim: ledger, margin, financing, stop-out,     │
            │   NBP, death, FX/cent conversion; one Account per agent │
            │   World: one clock, N accounts, event log               │
            └─────────────────────────────────────────────────────────┘
                     ▲ actions                    │ observations
            ┌────────┴─────────────┐   ┌──────────▼───────────┐
            │ 6 Action Interface   │   │ 5 Observation Iface  │   ← the ONLY surface agents see
            └────────▲─────────────┘   └──────────┬───────────┘
            ┌────────┴────────────────────────────▼────────────┐
            │ 7 Gymnasium Env (one agent ↔ one account)        │
            │   + pluggable reward module                      │
            └────────▲─────────────────────────────────────────┘
   8 Training (PPO → SAC, ES; per-agent params, optimizer, seed)
   9 Evaluation (walk-forward, null distributions, stress tests, behaviour analysis)
  10 Tracking (run registry: config + git hash + data hash + broker-profile hash + seeds)
  11 Dashboard (read-only over the tracking store; Phase 6)
```

### 3.2 Key design principles

- **Pure, deterministic core.** All randomness goes through injected, seeded generators. Same
  config + data hash + broker-profile hash + seed ⇒ bit-identical event log.
- **Event-sourced accounting.** Fills, commissions, swaps, stop-outs, NBP resets and deaths are
  immutable events. Balances are derived from them, so every run can be audited.
- **Hard isolation boundary.** Agents only receive an `Observation` and return an `Action`. Each
  agent has its own parameters, optimizer, buffers and RNG stream
  (`SeedSequence(experiment_seed).spawn(n)`). **No parameter sharing.**
- **Broker profiles are data, not code.** Exness specs live in versioned YAML snapshots with a
  source and verification status per field, so specs can be updated without code changes.
- **Simulator first.** The simulator has no Gymnasium or PyTorch dependency.
- **Performance by design.** Pre-aligned `float32` arrays `[time, instrument, field]`; zero-copy
  window views; numba only if profiling shows the need.

### 3.3 Technology stack

| Concern | Choice | Reason |
|---|---|---|
| Language | Python 3.11+ | Spec requirement |
| Arrays | NumPy (+ numba if needed) | Deterministic and fast |
| Data prep | Polars/pandas → Parquet (pyarrow) | Columnar, hashed files |
| Log queries | DuckDB over Parquet | SQL without a server |
| Config | Pydantic + YAML | Validated experiment and broker configs |
| RL env | Gymnasium | Standard API, `check_env` |
| Learning | PyTorch; Stable-Baselines3 for the first PPO/SAC; own code later if needed | Well tested, which lowers RL bug risk |
| Evolution | Own ES/GA loop in PyTorch (or `evotorch`) | Gradient-free; fits the "survival" framing |
| Tests | pytest + Hypothesis | Property tests for accounting invariants |
| Tracking | Own run registry + TensorBoard (MLflow optional) | Few dependencies, reproducible |
| Dashboard | Streamlit → FastAPI + frontend if needed | Read-only |

### 3.4 What "multi-agent" means here

All agents are price-takers on the same tape. Running them together is operationally convenient and
mathematically identical to running them alone, which a test enforces. Diversity comes from seeds,
learners (PPO / SAC / ES), reward variants, observation and decision settings, and
population-based evolution where survivors reproduce with mutation. Selection pressure comes only
from account outcomes.

---

## 4. How agents learn without predefined strategies

**Agents get:**
- market information per instrument over the window: bid/ask OHLC transformed into scale-free
  quantities (mid log returns, range/mid, body/mid, spread/mid, relative tick volume, tradeable flag);
- their own account state, as shown on a trading terminal: log equity / initial, free-margin
  fraction, margin level, current exposure per instrument, unrealized PnL / equity, time since each
  position was opened, and the margin-call flag (Exness margin call is at 60%);
- optionally, time-of-day and day-of-week (public information; ablated).

**Agents do not get:** indicators, signals, absolute price levels, dates, other agents' data, or
simulator internals (fill model, slippage RNG, future bars, cost-model parameters).

**Remaining inductive biases (documented and ablated):** feature transform, network architecture
(per-instrument 1D-CNN/GRU encoder + account MLP vs. plain MLP), window and decision interval,
reward function, learning algorithm.

Nothing in the engine opens, closes or resizes positions for an agent, **except** enforced account
constraints: margin rejection, broker stop-out, NBP and death. Each is logged with a reason.

---

## 5. Exness broker specification (researched 2026-11-15)

### 5.0 How these facts were verified, and the limits

- Exness's help center (`get.exness.help`) and site **refuse automated page fetches (HTTP 403)**
  from this environment. Facts below come from search-engine extracts of Exness's official pages
  plus third-party sources, cross-checked. **Exness's own pages contradict each other on several
  points**, which are marked as such.
- Status legend: **✅ confirmed** = official Exness source, consistent across sources ·
  **⚠️ conflicting** = official sources disagree · **❓ unverified** = third-party only, or derived.
- **Nothing marked ⚠️ or ❓ is used in Stage D until it is confirmed** against an Exness MT5
  Standard Cent **demo account**, through the read-only MetaApi connection already available to
  this project (symbol specification: contract size, min/max volume, step, digits, swaps,
  triple-swap day, margin rates, trading sessions; plus live bid/ask for spreads). See step 1.0b in §11.

### 5.1 Account-level specifications

| Item | Standard Cent | Standard | Status |
|---|---|---|---|
| Account currency | US cents (USC); $10 = 1,000 USC | USD | ✅ |
| Lot definition | 1 Cent lot = 0.01 standard lot (≈ 1,000 units of base currency) | 1 lot = 100,000 units (FX) | ✅ |
| Min volume | 0.01 lot (ETHUSDc: 0.1) | 0.01 lot | ✅ |
| Max volume | 200 lots (ETHUSDc 2,000) | 200 lots typical | ✅ (Cent) |
| Commission | none; costs in spread | none | ✅ |
| Spread | "from 0.3 pips" | "from 0.2 pips" (typical marketing figure) | ✅ / ❓ |
| Margin call | 60% | 60% | ✅ |
| Stop-out | **0%** (KE and JO entities: 20%; stocks up to 100% in high-risk periods) | **0%** (same exceptions) | ✅ |
| Stop-out closing order | reported as "most unprofitable position first" | same | ❓ |
| Negative balance protection | balance restored to 0 **only** when all orders were closed by stop-out and balance is negative; a deposit made while negative equity persists is not covered | same | ✅ |
| Max leverage (leverage-based instruments) | equity < $5,000: 1:2000 default, 1:Unlimited if eligible · $5k–30k: 1:2000 · $30k–100k: 1:1000 · ≥ $100k: 1:500 · KE entity: 1:400 | same | ✅ tiers / ⚠️ Unlimited-eligibility conditions |
| Platforms | MT4, MT5, Exness Terminal (BTC and indices: MT5 / Exness Terminal only) | MT4, MT5 | ✅ / ⚠️ |
| Swap timing | 21:00 GMT+0 (summer) / 22:00 GMT+0 (winter) | same | ✅ (Exness help) / ❓ (some third-party sources say 00:00 server time) |
| Triple swap | FX Wednesday (USDCAD Thursday); indices Friday; crypto Friday; energies none | same | ✅ |
| Swap-free | Exness states the seven major FX pairs (and most minors) can be traded swap-free; Islamic-country accounts are swap-free by default; third parties say swap-free status can depend on trading behaviour | same | ✅ / ❓ |
| Higher Margin Requirements (HMR) | from 15 min before a high-impact release to **90 s or 5 min after** (two official pages disagree); gold band 1:200–1:1000 | same | ⚠️ window |

### 5.2 Instrument mapping

| Project instrument | Exness Cent symbol | Availability on Cent | Smallest position (real money) | Margin type | Status |
|---|---|---|---|---|---|
| EUR/USD | `EURUSDc` | MT4/MT5 | 0.01 lot ≈ €10 ≈ $11.6 | leverage-based (equity tiers) | ✅ availability, lot maths ✅ |
| GBP/USD | `GBPUSDc` | MT4/MT5 | ≈ £10 ≈ $13 | leverage-based | ✅ |
| USD/JPY | `USDJPYc` | MT4/MT5 | $10; P&L in JPY → USC | leverage-based | ✅ |
| XAU/USD | `XAUUSDc` | MT4/MT5 | standard XAUUSD contract = 100 oz, so 0.01 Cent lot ≈ 0.01 oz ≈ $40 at $4,000/oz | leverage-based; HMR band 1:200–1:1000 | ✅ availability / ❓ derived contract |
| BTC/USD | `BTCUSDc` | **MT5 only. One official statement says "Trading on BTCUSDc is unavailable"; blocked for the KE entity** | reported contract 0.01 BTC per lot (unclear whether real or cent-denominated) | **fixed margin** (independent of account leverage); HMR cap 1:200–1:400 | ⚠️ |
| NAS100 | `USTECc` | **Listed for MT5 / Exness Terminal, but the Indices page also says indices are not offered on Standard Cent** | standard `USTEC` contract = 1 (reported min 0.05 lot); Cent contract unknown | **fixed** 0.25% (1:400) for USTEC; HMR 1:50–1:100 reported | ⚠️ |

Fallback specs (Standard account) per decision 2, if a Cent symbol is unavailable:
`BTCUSD` 1 lot = 1 BTC, min 0.01 (≈ $650 notional) · `USTEC` contract 1, min reported 0.05
(≈ $1,000–1,250 notional) · `XAUUSD` 100 oz, min 0.01 (1 oz).

### 5.3 Consequences for a $10 account (Stage D, Cent profile)

Max notional: **L = 3 → $30; L = 100 → $1,000.**

| Instrument | Smallest position | Tradable at L = 3? | Exposure steps at L = 3 | Tradable at L = 100? |
|---|---|---|---|---|
| EUR/USD (c) | ≈ $11.6 | yes | 0, ±1.16×, ±2.3× | yes, ≈ 0.116× steps |
| GBP/USD (c) | ≈ $13 | yes | 0, ±1.3×, ±2.6× | yes |
| USD/JPY (c) | $10 | yes | 0, ±1×, ±2×, ±3× | yes |
| XAU/USD (c) | ≈ $40 | **no** | — | yes, ≈ 4× steps |
| BTC/USD (c) | pending §5.2 | pending | pending | pending |
| NAS100 (c) | pending §5.2 | pending | pending | pending |
| BTC/USD (Standard fallback) | ≈ $650 | no | — | no ($650 needs 65×, allowed, but it is a single all-or-nothing step) |
| NAS100 (Standard fallback) | ≈ $1,000+ | no | — | no / borderline |

Margin practically never binds at L = 3: with 1:2000 leverage, $30 of FX notional needs $0.015 of
margin. **Under Exness the constraints that matter for a $10 account are the project's L cap, lot
quantization, spreads, swaps and the 5% death floor**, not broker margin.

Costs: with no commission, cost = spread, and on a Cent account it scales with position size like on
any account. The minimum EUR/USD Cent position (0.01 Cent lot = €10 real) moves $0.001 (0.1 USC) per
pip. A 1-pip spread is ≈ 0.9 bps of notional, so a round trip at 2× leverage costs ≈ 1.8 bps of
equity. That is small per trade, but **turnover compounds it**: at 5-minute decisions, 100 round
trips a day would cost ≈ 1.8% of equity per day. (This corrects r1, where the "1–6% of the account
per round trip" figure applied to *standard* 0.01 lots on a $10 account.) **Exact pip values are
verified numerically on the demo account in step 1.0b.** Cent-account pip values are a classic
source of factor-of-100 errors, so this is also a golden test (§8.2).

### 5.4 Stop-out, NBP and death under Exness

- **Death (project rule):** equity ≤ 5% of initial ($0.50 = 50 USC) ⇒ all positions closed at
  market (bid/ask + slippage), account terminated, reason recorded.
- **Exness stop-out (0%)** therefore only triggers if one bar **gaps** the account from above 5%
  straight to ≤ 0 (weekend gap, illiquid session open). Positions are then closed at the gap price
  (most unprofitable first, pending verification).
- **NBP (Exness profile):** if the balance is negative after a full stop-out, it is reset to 0 and an
  `nbp_reset` event is logged. The agent is dead either way.
- **No-NBP stress profile:** the negative balance is kept and reported as a debt.
- The intra-bar check uses each instrument's adverse extreme within the bar. This is conservative,
  because extremes on different instruments may not be simultaneous; the approximation is documented.
- **Margin call (60%)** is a flag in the observation only. It is not a forced action.
- The entity-specific variant (KE/JO: stop-out 20%, leverage 1:400) is a separate profile.

### 5.5 Stage D account-mixing problem (needs a decision)

Decision 2 says: "Cent specs where supported, otherwise the applicable Exness specs". A real $10
account is one account of one type, so a Cent account cannot also hold Standard-account BTC.
Options:

- **(a) Cent-only universe (recommended):** Stage D trades only the instruments the Cent account
  actually offers (verified on demo). Unsupported instruments are not tradable, and the
  observation still contains them, flagged `tradeable = 0`.
- **(b) Hypothetical hybrid account:** one $10 ledger that uses Cent specs for some symbols and
  Standard specs for others. This is not a real account; label it as hypothetical.
- **(c) Two sub-accounts:** $10 split across a Cent account and a Standard account. Realistic but
  changes the "$10 per agent" premise.

### 5.6 Data design: Dukascopy price path + Exness cost model

Prices and costs come from different sources, on purpose:

- **Price path:** Dukascopy M1 **bid and ask** candles, 2018 → present, for EURUSD, GBPUSD, USDJPY,
  XAUUSD, BTCUSD and USATECHIDXUSD (Nasdaq-100 CFD). It gives a long, consistent, gap-annotated
  history. Dukascopy is reachable from this environment but rate-limits it (HTTP 429), so the bulk
  download runs locally (decision 7). BTC history start date on Dukascopy to be verified on download.
- **Exness cost model:** Exness publishes official **tick history** (bid/ask ticks; monthly zip
  files per symbol, downloadable from its Tick History page, columns: Exness, Symbol, Timestamp,
  Bid, Ask). From a calibration sample (e.g. the last 12–24 months) we estimate each instrument's
  **Exness spread distribution by hour-of-week and volatility regime**. The simulator then
  replaces Dukascopy's spread with this calibrated Exness spread, centred on the Dukascopy mid.
  Whether Cent symbols have their own tick files, or match Standard spreads, is **❓ to verify
  against the demo**.
- **Exness specs snapshot:** contract sizes, volume limits, digits, swap long/short, triple-swap
  day and trading sessions, read from the demo account through MetaApi (read-only endpoints).
  Saved as a dated, hashed YAML profile.
- **Calibration test:** simulated Exness spreads vs. the held-out Exness tick months, compared on
  distribution quantiles per hour-of-week.
- **Caveat:** swap rates are only available as current snapshots, not history. Historical swaps
  are approximated by the snapshot (documented). Exness swap-free majors are a profile variant.

### 5.7 From action to orders (no hidden rebalancing)

```
a ∈ [-1,1]^n  →  w_i = L·a_i / max(1, Σ|a_j|)               (gross exposure ≤ L)
target_units_i = w_i · equity / (price_i · contract_size_i · fx_to_account_i)
fractional profile: no rounding (min trade size threshold only, e.g. $0.01 notional)
exness profiles   : round toward zero to lot_step; |target| < min_lot ⇒ 0
delta_i = target_i − current_i ; execute only if |delta_i| ≥ lot_step_i
ordering: reductions first, then increases
margin check: increases beyond free margin are reduced to the largest affordable size (logged)
```

Fills: decision at the close of the decision bar → fill at the next **M1** bar's open; buys at ask,
sells at bid; adverse slippage (base + range-scaled + seeded noise). Closed market ⇒ rejected and
logged. Swaps at Exness rollover times, with triple days as above.

---

## 6. Reward design

### 6.1 Variants (all on **net-of-cost** equity)

| ID | Reward per decision step | Expected behaviour |
|---|---|---|
| R1 (baseline) | `log(max(E_t, ε·E_0) / E_{t-1})`, ε = 0.01 | Growth-optimal; penalizes variance and ruin. The ε floor keeps the reward finite at death and under no-NBP negative equity |
| R2 (negative control) | `(E_t − E_{t-1}) / E_0` | Risk-neutral; expected to drive max leverage and death |
| R3 | R1 − λ·Δ(drawdown) | Conservative; a labelled risk preference |
| R4 | Differential Sharpe ratio | Smooth; can be gamed by near-zero variance |
| R5 | Sparse terminal `log(E_T / E_0)` | No shaping; hard credit assignment |

Not used: trade-count bonuses, win-rate rewards, rewards for market presence, realized-PnL-only rewards.

### 6.2 Reward-hacking surfaces

| Exploit | Closure |
|---|---|
| Mid fills / stale prices / closed sessions | bid/ask fills, tradeable mask, rejection |
| Favourable lot rounding | round toward zero |
| Margin abuse between bars | checks at order time and intra-bar |
| NBP as a free option | NBP only in the Exness profile; training default = no NBP + ε-floored log reward; no-NBP stress run |
| Episode reset as a "free life" | episode ends at death; evaluation = one continuous life |
| Weekend gaps | valuation at the first real quote |
| Memorizing dates / levels | no absolute levels or dates in observations |
| Optimizing the reward instead of the outcome | all variants judged on the same out-of-sample metrics |
| Exploiting a cost-model artefact (e.g. hours where the calibrated spread is too low) | cost-robustness tests at 1.5× and 2×, and calibration tests (§5.6) |

### 6.3 Learner choice

PPO first (stable, continuous actions; use a tanh-squashed or Beta policy to avoid bound bias).
Weaknesses: on-policy sample inefficiency and noisy rewards. Then SAC (off-policy, entropy
regularized) and ES/GA (gradient-free, population-based) in Phase 3.

### 6.4 Baselines (never shown to agents)

Flat, uniform random, equal-weight buy-and-hold, and a turnover-matched random agent per learned
agent. Together they form the null distribution.

### 6.5 Train (fractional) → Stage D (Cent lots) transfer

The agent is evaluated, without retraining, on the Exness Cent profile. Reported: the performance
gap between fractional and quantized execution, how often targets round to zero, and turnover
change. Optional experiment: short quantization-aware fine-tuning on **training-period** data only.

---

## 7. Preventing data leakage and overfitting

### 7.1 Splits (chronological, embargoed)

| Split | Period | Use |
|---|---|---|
| Train | 2018-01 → 2022-12 | Stage A |
| Embargo | 2 weeks | |
| Validation | 2023-01 → 2023-12 | hyperparameters, model selection |
| Embargo | 2 weeks | |
| Test | 2024-01 → 2025-06 | Stage B, milestone-only access, every access logged |
| Final holdout | 2025-07 → present | touched once, before Stage C |
| Exness calibration sample | recent 12–24 months of Exness ticks | **cost-model only**; prices from these months are never used as training inputs |

Plus walk-forward folds inside the training range (2020 COVID, 2022 rate cycle, …).

Note: the Exness calibration months overlap the test/holdout period. This is acceptable because only
**spread statistics** are taken from them, not price paths, but it is a documented leak of cost
information into evaluation. An alternative is calibrating on older Exness months only, if
available.

### 7.2 Leakage controls

Observations only from bars closed ≤ decision time; a future-perturbation test; window-local
normalization; aggregation only from completed M1 bars; the test split is physically separate and
the training loader refuses it.

### 7.3 Overfitting controls

Random episode starts; validation-only model selection; beta vs. skill (drift-adjusted reporting,
optional sign-flip augmentation); selection-bias correction (deflated Sharpe, null distributions,
block bootstrap); cost robustness at 1.5× and 2×.

### 7.4 Pre-registered success criteria

On the test split, **all** of:
1. net log-growth > 0 with a 95% block-bootstrap CI excluding 0;
2. above the 95th percentile of turnover-matched random agents;
3. not explained by buy-and-hold drift;
4. still positive at 1.5× costs;
5. consistent across ≥ 2 non-overlapping sub-periods;
6. **(Stage D)** reported separately under the Exness Cent profile, including quantization effects.

---

## 8. Tests required before reinforcement learning (Phase 1 exit gate)

### 8.1 Data quality
Monotonic unique UTC timestamps; bid ≤ ask; OHLC consistency per side; gap classification; DST;
spread outlier detection; Dukascopy-vs-Exness mid divergence report on overlapping months.

### 8.2 Simulator correctness
1. No look-ahead (future-perturbation test).
2. Fills: ask/bid at the next M1 open, adverse slippage, exact commissions (zero for Exness Cent).
3. Accounting invariants (Hypothesis): equity = balance + unrealized; Δbalance = realized − costs −
   swaps; ledger reconciliation; zero-cost round trip ⇒ zero PnL.
4. **Golden scenarios with Exness numbers:** EURUSDc, USDJPYc (JPY → USC conversion) and XAUUSDc
   trades with values checked against the demo terminal's profit calculation. **USC/USD
   factor-of-100 test.**
5. Action mapping: formula exact, gross ≤ L, zero ⇒ flat, reversal, rounding toward zero, min lot.
6. Margin: fixed-margin instruments (crypto, indices) vs. leverage-based (FX, gold) vs. equity-tier
   leverage; HMR windows raise margin.
7. Death at 5%; Exness stop-out at 0% on a gap; closing order; no actions after death.
8. NBP: reset only after a full stop-out with negative balance; no-NBP profile keeps the debt.
9. Swaps: only for positions held across Exness rollover; triple days per instrument class;
   swap-free variant.
10. Sessions: no fills when closed; daily breaks for gold and indices; 24/7 BTC.
11. Determinism: identical event-log hash for the same seed.
12. Isolation: agent results identical alone vs. with others.
13. Baselines match analytic expectations.
14. No order path: the MetaApi adapter calls only allowlisted read-only endpoints; no trade
    endpoint string exists in the package.
15. Performance benchmark (steps/s), recorded per machine.

### 8.3 Phase 2 gate
`check_env` passes; on synthetic markets with planted structure the agent learns a profitable
policy, and on a costly random walk it converges to ≈ flat.

---

## 9. Project directory structure

```
trading-world/
├── pyproject.toml
├── README.md
├── docs/
│   ├── ARCHITECTURE.md
│   ├── simulator-spec.md            ← exact formulas
│   ├── exness-spec-verification.md  ← each broker field: value, source, date, status
│   └── decisions/                   ← ADRs
├── configs/
│   ├── instruments/
│   ├── brokers/
│   │   ├── fractional.yaml
│   │   ├── exness_standard_cent.yaml      ← generated from the demo snapshot
│   │   ├── exness_standard.yaml
│   │   ├── exness_standard_cent_no_nbp.yaml
│   │   └── exness_entity_ke.yaml          ← only if relevant
│   └── experiments/
├── src/tradeworld/
│   ├── data/          # sources/{dukascopy, exness_ticks}, align, quality, store
│   ├── calibration/   # Exness spread model fitting + validation
│   ├── brokerspec/    # read-only MetaApi spec snapshot → broker YAML
│   ├── replay/  market/  execution/  accounts/  world/
│   ├── interface/  envs/  rewards/  agents/  training/
│   ├── evaluation/  tracking/  live/  dashboard/
├── tests/  unit/ property/ golden/ integration/ benchmarks/
├── scripts/       # hardware_check, download_data, calibrate_costs, snapshot_specs, run_world
├── data/          # gitignored
└── runs/          # gitignored
```

The earlier `btc-trading-bot/` folder (an indicator bot, excluded by this spec) shares no code with
this project. Its fate is still open (§12.2).

---

## 10. Metrics tracked per agent

Everything in spec §11, plus log-growth, Sharpe/Sortino, deflated Sharpe, drawdown duration,
turnover, gross/net leverage, cost share of gross PnL, swap share, time in market, long/short bias,
allocation entropy, holding-time distribution, stop-out and NBP counts, time to death,
incapacitated duration, margin-limited orders, **rounded-to-zero target rate (Stage D)**. Behaviour
embedding and clustering compare emergent diversity against seed-only diversity.

---

## 11. Milestone 1 (Phase 0 + Phase 1)

| Step | Deliverable | Exit criteria |
|---|---|---|
| 1.0 | ADRs for the decisions; `simulator-spec.md` | approved |
| **1.0a** | **Hardware assessment** on your machine (§13) | resource plan agreed |
| **1.0b** | **Exness spec verification:** read-only snapshot from an Exness MT5 Standard Cent demo via MetaApi; resolve every ⚠️/❓ in §5 | `exness-spec-verification.md` complete; broker YAMLs generated |
| 1.1 | Instrument and broker configs; `market/`; FX and cent conversion | unit + golden tests |
| 1.2 | `accounts/` + `execution/` on **synthetic** prices | §8.2 accounting, margin, stop-out, NBP, death tests green |
| 1.3 | Data pipeline: Dukascopy bid/ask M1 (local download) + Exness tick sample; quality report | §8.1 green |
| 1.4 | Exness spread calibration + validation | calibration test passes |
| 1.5 | `replay/` + `interface/` + `world/` on real data | look-ahead, determinism, isolation green; throughput benchmark recorded |
| 1.6 | Baselines (flat, random, buy-and-hold) on train/validation under the fractional and Exness Cent profiles | numbers match analytic expectations; report reviewed |

Phase 2 starts only after Milestone 1 sign-off.

---

## 12. Decisions

### 12.1 Recorded (2026-11-15)

1. **Training sizing:** fractional, with realistic costs.
2. **Stage D:** Exness Cent specs where the instrument is supported, otherwise the applicable Exness
   account specs and real minimum lots (see §5.5 for the open sub-question).
3. **Margin and leverage:** verified Exness requirements, configurable per instrument and account type.
4. **NBP:** Exness NBP in the broker-specific profile, plus a separate no-NBP stress scenario.
5. **Data:** Dukascopy M1 bid/ask from 2018; Exness data for calibration.
6. **Decision interval:** 5 minutes default; 1, 15 and 60 minutes configurable.
7. **Compute:** develop locally first; assess hardware before long training runs.

### 12.2 Still open

1. **Stage D account mixing (§5.5):** (a) Cent-only universe (recommended), (b) hypothetical hybrid
   ledger, or (c) two sub-accounts.
2. **Exness entity/region** your account would be registered under (affects stop-out, leverage and
   BTCUSDc availability).
3. **Demo account for verification:** an Exness **MT5 Standard Cent demo** connected to MetaApi, and
   its MetaApi account ID (used read-only).
4. **Observation window at 5-minute decisions:** 120 × M5 bars (recommended) vs. 120 × M1 bars.
5. **`btc-trading-bot/`:** keep separately or delete.

---

## 13. Hardware assessment (step 1.0a)

To size the work, please run this on the machine that will do the training and paste the output:

```bash
python -c "import os,platform,shutil;print(platform.platform());print('cpu',platform.processor(),os.cpu_count(),'threads');print('disk free GB',round(shutil.disk_usage('.').free/1e9))"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```
and state your **RAM size** and **GPU model / VRAM** (Task Manager on Windows, About This Mac,
or `free -h` / `nvidia-smi` on Linux).

Rough requirements (estimates, to be replaced by measured benchmarks in step 1.5):

| Resource | Minimum | Comfortable | Driver |
|---|---|---|---|
| Disk | 10 GB | 50 GB | Dukascopy raw files (~40k daily files), Parquet store (~0.5–1 GB), Exness tick sample (large; zipped monthly ticks), run logs |
| RAM | 8 GB | 16–32 GB | aligned M1 arrays ≈ 1 GB float32 for 6 instruments × 9 years, plus parallel environments |
| CPU | 4 cores | 8+ cores | the simulator is CPU-bound; parallel envs scale with cores |
| GPU | none (CPU training is possible for small nets) | any CUDA GPU with ≥ 6 GB | larger encoders and many agents |

Ballpark: at 5-minute decisions, the 2018–2022 training range is ≈ 0.3–0.5 M decision steps per pass
per agent. Typical PPO budgets of 10–50 M steps per agent mean hours per agent on a mid-range
CPU. A population of agents multiplies that, which is why benchmarks come before any long run.

---

## Sources (Exness research, 2026-11-15)

Official Exness pages (content obtained via search-engine extracts, because direct fetches returned 403):
- Standard Cent account — https://get.exness.help/hc/en-us/articles/17537782786588-Standard-Cent-account
- Standard account — https://get.exness.help/hc/en-us/articles/17537782738460-Standard-account
- Margin call and stop out levels by account type — https://get.exness.help/hc/en-us/articles/360014693340-Margin-call-and-stop-out-levels-by-account-type
- 0% stop out / Stop Out Protection — https://get.exness.help/hc/en-us/articles/17685808714140-0-stop-out-Stop-Out-Protection
- Negative Balance Protection — https://get.exness.help/hc/en-us/articles/18055126006044-Negative-Balance-Protection
- Margin — https://get.exness.help/hc/en-us/articles/360014504992-Margin
- Leverage — https://get.exness.help/hc/en-us/articles/360014529380-Leverage
- Higher Margin Requirements (HMR) — https://get.exness.help/hc/en-us/articles/20618327917596-Higher-Margin-Requirements-HMR
- Commodities — https://get.exness.help/hc/en-us/articles/17854173039388-Commodities
- Indices — https://get.exness.help/hc/en-us/articles/17854383867548-Indices
- Cryptocurrencies — https://get.exness.help/hc/en-us/articles/17854191888540-Cryptocurrencies
- About swap — https://get.exness.help/hc/en-us/articles/360014709151-About-swap
- Account type suffixes — https://get.exness.help/hc/en-us/articles/360014560220-Account-type-suffixes
- Forex (swap-free majors) — https://www.exness.com/forex/
- Indices product page (USTEC 1:400) — https://www.exness.com/indices/us-tech-100/
- Tick history — https://www.exness.com/tick-history/

Third-party (used only for items marked ❓):
- https://tradersunion.com/brokers/forex/view/exness/stop-out-level-and-trailing-stop/
- https://onforex.co/en/exness/leverage/
- https://www.brokerxplorer.com/article/exness-tick-history-report-a-beginners-guide-4624
- https://pypi.org/project/exness-data-preprocess/
