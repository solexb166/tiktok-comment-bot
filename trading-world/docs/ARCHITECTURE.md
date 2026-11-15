# Autonomous Multi-Agent Trading World — Phase 0 Architecture & Plan

Status: **DRAFT for review.** No implementation code until the open decisions in §12 are answered.

---

## 0. Summary

The vision is buildable. Several assumptions in the spec conflict with each other or with how markets and brokers actually work, and they need decisions before Phase 1. The most important ones:

1. **A $10 account at L = 3 cannot trade any of the six instruments at a real broker.** The smallest
   standard position (0.01 lot) is $200–$4,000 of notional, and L = 3 allows $30. Under EU-style margin
   rules, a single EUR/USD micro-lot needs ~$38 of margin. So "realistic $10 account" and "L = 3"
   cannot both hold unless the broker model allows fractional sizes. See §5.
2. **On historical data this is not a multi-agent market. It is N independent single-agent experiments
   that share a price tape.** $10 agents are price-takers and cannot affect each other. That is
   realistic, but it means "competition" is a leaderboard, not an interaction, and any behavioural
   diversity comes from seeds, hyperparameters, learners and reward variants, not from agents
   reacting to each other. See §3.4.
3. **"No predefined trading knowledge" can only be approximated.** The observation encoding, network
   architecture, decision frequency and above all the **reward function** are inductive biases. The
   reward function *is* a risk preference. The honest goal is *no trading rules*, with every
   remaining bias written down and varied experimentally. See §4.
4. **Target-exposure actions secretly create a rebalancing rule.** If the engine re-sizes positions to
   `w·equity` every minute, it performs constant-leverage rebalancing for the agent (an implicit
   risk rule) and charges costs for it. The execution layer must avoid this. See §5.3.
5. **M1 bars with a 120-bar window (2 hours) is a hostile setting.** Per-minute moves are of the same
   order as the spread. The most likely emergent behaviour is "stay flat", and that is a correct,
   reportable result, not a bug. Decision interval and window length should be experiment variables.
6. **With many agents, some will look profitable by luck.** Claims need a null distribution
   (random/flat agents and shuffled-time controls) and multiple-testing correction. See §7.

---

## 1. Critique of the specification

| # | Issue | Why it matters | Proposal |
|---|---|---|---|
| 1 | $10 + L=3 + real lot sizes are incompatible | No position can be opened; the agent learns nothing | Two broker profiles: `fractional` (continuous size, realistic bps costs) for learning, `retail_lots` / `retail_cent` for Stage D. Decision needed (§12). |
| 2 | Normalized equity 1.0 "for numerical stability" | Lot minimums, fixed commissions and per-lot swaps are **not scale-invariant**. A $10 account behaves differently from a $10,000 one | The simulator accounts in **real dollars** with real contract specs. Normalization (equity / initial) is applied **only in observations and rewards**. |
| 3 | Multi-agent "world" on historical replay | No interaction is possible without an endogenous price process | Accept the price-taker model (realistic for $10). Generate diversity through seeds, learners, reward variants, and optionally population-based evolution. A synthetic agent-based market could be a separate, later research track. |
| 4 | "Agents must not be given risk rules" vs. reward design | The reward encodes risk preference | Use log-wealth growth as the baseline (the most neutral choice, explained in §6). Treat any penalty term as an experimental variant and label it as such. |
| 5 | Target-exposure re-evaluated every step | Implicit auto-deleveraging plus churn costs | Execute only position *deltas* that exceed the lot step. Rebalance only when the agent's output changes the target materially. See §5.3. |
| 6 | M1 decisions every minute | ~525k decisions/year per instrument, cost-dominated, poor credit assignment (γ=0.99 → ~100-minute horizon) | Decision interval `k ∈ {1, 5, 15, 60}` minutes as an experiment variable. M1 data remains the fill and valuation clock. |
| 7 | 120-bar window only | 2 hours of M1 context | Optional multi-resolution observation (120 × M1, M15 and H4 bars aggregated *only from past M1*). This is information, not a strategy. Ablate it. |
| 8 | Instruments trade different hours | BTC 24/7, FX 24/5, NAS100/XAU with daily breaks; weekend gaps | Unified UTC minute clock with a per-instrument `tradeable` mask. Never forward-fill a price as tradeable. Weekend and session gaps are valued at the first real quote. |
| 9 | Death at 5% vs. broker stop-out | Real brokers stop out at ~50% margin level, usually before 5% equity | Model both. Stop-out liquidates positions and the agent lives on. Death = equity ≤ 5% of initial ($0.50), which terminates the account. |
| 10 | "Zombie" accounts | An account with $2 is alive but may be unable to afford any minimum position | Track an `incapacitated` status and duration as a metric. Don't kill it; report it. |
| 11 | Negative balance | Gaps can push equity below zero | Configurable `negative_balance_protection`. **Warning:** NBP plus episode resets makes high leverage a free option and rewards gambling. Default off in training, studied separately. |
| 12 | Financing data | Historical swap rates are hard to source | Model swaps as (rate differential + broker markup), triple on Wednesday for FX. Use dividend-adjusted financing for NAS100 and broker-style daily funding for BTC. Document it as an approximation. |
| 13 | No success criterion | "Discover profitable behaviour" is undefined | Pre-register criteria (§7.4) before training starts. |
| 14 | Compute and persistence | This cloud container is ephemeral; RL on M1 is millions of steps per agent | Training runs on a persistent machine (the user's PC/server or a rented VM). This environment is fine for building and testing the simulator. |
| 15 | Live stage must never trade | My earlier `btc-trading-bot` contains real order-sending code | This project contains **no code path that can send an order to a broker**. The live adapter is read-only market data. Enforced by a test. |
| 16 | Train/serve skew in live mode | Live bars built differently from historical bars → silent distribution shift | Live feed adapter must emit the identical bar schema (bid/ask OHLC, same aggregation, same session mask). A parity test compares live-built bars against the historical source for an overlapping period. |
| 17 | Bid/ask data | Midpoint-only M1 data makes costs fictional | Require **bid and ask** M1 candles (Dukascopy provides both for all six instruments). |

---

## 2. Is the experiment likely to "work"? (setting expectations)

Plainly: most agents will learn to stay flat, die, or capture market drift (beta) that disappears out of
sample. RL in markets has a very low signal-to-noise ratio, and at M1 resolution costs dominate. This
is not a reason to stop. It is a reason to design the evaluation so that a **null result is
measurable and trustworthy**, and so that the learning machinery is proven to work *where there is
signal* (synthetic markets, §8.3). Otherwise a failure can't be attributed to the market or to a bug.

---

## 3. Architecture

### 3.1 Component map

```
                ┌──────────────── offline ───────────────┐
 Data sources → │ 1 Market Data Engine (download, clean, │ → Parquet store (immutable, hashed)
 (Dukascopy,    │   align, validate, session masks)      │
  Coinbase)     └────────────────────────────────────────┘
                                   │
                                   ▼
            ┌──────────── deterministic core (no learning) ───────────┐
            │ 2 Replay Engine: UTC clock, cursor, past-only windows   │
            │ 3 Execution Engine: bid/ask fills, latency, slippage,   │
            │   lot rounding, commissions                             │
            │ 4 Account Sim: ledger, margin, financing, stop-out,     │
            │   death, FX conversion — one isolated Account per agent │
            │   World: one clock, N accounts, event log               │
            └─────────────────────────────────────────────────────────┘
                     ▲ actions                    │ observations
            ┌────────┴─────────────┐   ┌──────────▼───────────┐
            │ 6 Action Interface   │   │ 5 Observation Iface  │   ← the ONLY surface agents see
            └────────▲─────────────┘   └──────────┬───────────┘
                     │                            │
            ┌────────┴────────────────────────────▼────────────┐
            │ 7 Gymnasium Env (one agent ↔ one account)        │
            │   + reward module (pluggable)                    │
            └────────▲─────────────────────────────────────────┘
                     │
   8 Training (PPO / SAC / ES; per-agent params, optimizer, seed)
   9 Evaluation (walk-forward, null distributions, stress tests, behaviour analysis)
  10 Tracking (run registry: config + git hash + data hash + seeds + logs + checkpoints)
  11 Dashboard (read-only over tracking store; Phase 6)
```

### 3.2 Key design principles

- **Pure, deterministic core.** The replay, execution and account modules contain no randomness
  except through an injected, seeded `numpy.random.Generator`. Same config + data hash + seed ⇒
  bit-identical event log.
- **Event-sourced accounting.** Every state change (fill, commission, swap, stop-out, death) is an
  immutable event. Balance and equity are derived from events, so the ledger can be replayed and
  audited. This also feeds the dashboard and the behaviour analysis.
- **Hard isolation boundary.** Agents receive an `Observation` dataclass and return an `Action`
  array. They hold no reference to the World, other accounts, the replay cursor, or the data
  store. In multi-agent runs each agent has its own parameters, optimizer, rollout buffer and
  RNG stream (derived via `numpy.random.SeedSequence(experiment_seed).spawn(n_agents)`). There is
  **no parameter sharing**, because shared parameters would mean the agents are not independent.
- **Simulator first, learning second.** The `envs/` layer is a thin adapter. The simulator has no
  Gymnasium or PyTorch dependency.
- **Performance by design.** Data is pre-aligned into contiguous `float32` arrays
  `[time, instrument, field]`; observation windows are strided views (zero-copy). Target throughput
  for the core is ≥ 50k agent-steps/s per CPU core, with `numba` reserved for hot loops if profiling
  shows the need.

### 3.3 Technology stack

| Concern | Choice | Reason |
|---|---|---|
| Language | Python 3.11+ | Spec requirement; ecosystem |
| Arrays | NumPy (+ numba if needed) | Deterministic, fast, simple |
| Data prep | Polars or pandas → Parquet (pyarrow) | Columnar, compressed, fast; a hash per file for reproducibility |
| Log queries | DuckDB over Parquet | SQL over large decision logs without a database server |
| Config | Pydantic models + YAML | Validated, typed experiment configs |
| RL env | Gymnasium | Standard API; `check_env` validation |
| Learning | PyTorch; Stable-Baselines3 for the first PPO/SAC baseline, own CleanRL-style code later if needed | SB3 is well tested, which reduces the risk of RL bugs in Phase 2. Own code later gives full control |
| Evolution | Own ES / genetic loop in PyTorch (or `evotorch`) | Gradient-free; fits the "survive" framing (§6.4) |
| Tests | pytest + Hypothesis (property-based) | Accounting invariants are best tested as properties |
| Tracking | Own lightweight run registry (directory per run) + TensorBoard; MLflow optional | Few dependencies; fully reproducible |
| Dashboard | Streamlit (prototype) → FastAPI + small frontend if needed | Read-only over the tracking store |

### 3.4 What "multi-agent" means here

On historical replay all agents are price-takers on the same tape. Running them "simultaneously" in
one World is useful for operations (one data pass, a shared clock, a live dashboard) and is
mathematically identical to running them separately. A test enforces that. Behavioural diversity can
come from:

- different random seeds (weakest source; often converges to similar policies),
- different learners (PPO vs. SAC vs. evolution),
- different reward variants (§6),
- different observation resolutions and decision intervals,
- **population-based evolution**, where agents that survive and grow reproduce with mutation. This
  is the closest match to the "world" metaphor, and selection pressure comes only from account
  outcomes.

---

## 4. How agents learn without predefined strategies

**Agents get:**
- raw market information for each instrument over the past window: bid/ask OHLC, transformed into
  scale-free quantities (log returns of mid relative to the previous bar, bar range / mid,
  body / mid, spread / mid, relative tick volume, a tradeable flag);
- their own account state, as any trader would see it on a terminal: equity / initial (log),
  free-margin fraction, margin level, current exposure per instrument, unrealized PnL per
  position / equity, and time since each position was opened;
- optionally, time-of-day and day-of-week (public information; ablated).

**Agents do not get:** indicators, signals, absolute price levels, dates, other agents' data, or
simulator internals such as the fill model, the slippage RNG, or future bars.

**Inductive biases that remain and are documented:**

| Bias | Mitigation |
|---|---|
| Feature transform (log returns, window-relative scaling) | Minimal, monotone, no trading meaning; documented |
| Network architecture (e.g. per-instrument 1D-CNN/GRU encoder shared across instruments within one agent, plus an account MLP) | Ablate against a plain MLP |
| Window length / decision interval | Experiment variables |
| Reward function | Several variants compared (§6) |
| Learning algorithm | Several families compared |

Entry, exit, sizing, holding period, reversal and loss response all have to emerge from the policy
mapping `observation → a ∈ [-1, 1]^6`. Nothing in the engine opens, closes or resizes a position on the
agent's behalf. The **only** exceptions are enforced account constraints (margin rejection,
stop-out, death). These happen to the agent as market consequences, and every one is logged with a
reason.

---

## 5. Simulator mechanics (realistic small accounts)

### 5.1 Instruments (typical retail CFD specs, to be calibrated to one named broker)

| Instrument | 0.01-lot notional (approx.) | Typical spread | Notes |
|---|---|---|---|
| EUR/USD | €1,000 ≈ $1,100–1,200 | 0.1–1.0 pip | 24/5 |
| GBP/USD | £1,000 ≈ $1,250–1,350 | 0.3–1.5 pip | 24/5 |
| USD/JPY | $1,000 | 0.2–1.2 pip | P&L in JPY → converted to USD |
| XAU/USD | 1 oz ≈ $3,000–4,500 | $0.15–0.40 | daily break |
| BTC/USD | 0.01 BTC ≈ $650 (at $65k) | $15–60 | 24/7, high financing |
| NAS100 | broker-dependent, ~$200–250 | 1–2 points | daily break, dividend adj. |

**Consequences at $10 equity:**
- Max notional: L=3 → $30 (nothing is tradable with standard lots); L=100 → $1,000 (only BTC and
  NAS100 fit; the FX majors sit at or just above the cap).
- **One round trip of one minimum lot costs ~1–6% of the account** (spread + commission). A $10
  account is a cost-dominated environment.

**Broker profiles (configs):**
- `fractional` — continuous position sizes, spreads/commissions in bps, real financing, stop-out
  and death. Used for learning, so that the action → exposure mapping is continuous.
- `retail_lots` — real min lot / lot step / per-lot commission and margin tables (e.g. 1:30 FX EU
  retail, or 1:500 offshore). Used for Stage D.
- `retail_cent` — cent-account lot sizes (100× smaller units), which is how real $10 accounts usually
  trade.

### 5.2 Execution model

- Decision at the close of bar *t* → order fills at the **open of bar t+1** (one-bar latency).
  Buys fill at **ask**, sells at **bid**.
- Slippage: adverse, `base_bps + k·(bar range)` plus seeded noise. Optional size impact (irrelevant at
  $10, present for completeness).
- Market closed / no quote ⇒ order rejected for that bar (logged), never filled at a stale price.
- Commission per lot or per notional by profile. Financing at the rollover time (17:00 New York),
  triple for FX on Wednesday, daily for BTC.
- Multi-currency: account in USD; JPY-denominated P&L converted at the current USD/JPY quote.

### 5.3 From action to orders (no hidden rebalancing)

```
a ∈ [-1,1]^n  →  w_i = L·a_i / max(1, Σ|a_j|)               (gross exposure ≤ L)
target_units_i = w_i · equity / (price_i · contract_size_i · fx_to_usd_i)
target_units_i = round_toward_zero(target_units_i, lot_step_i);  |target| < min_lot ⇒ 0
delta_i = target_units_i − current_units_i ;  execute only if |delta_i| ≥ lot_step_i
ordering: reductions first (frees margin), then increases
margin check: increases that exceed free margin are reduced to the largest affordable size
              (logged as `margin_limited`), never silently ignored
```

Open question for review: should the agent express *hold* explicitly? With the mapping above,
holding means outputting the target it already has. The agent can learn that because its current
exposure is in the observation. An alternative is an extra action dimension, "rebalance
gate" ∈ [0, 1]. I recommend **no gate initially**, plus a tracked turnover metric.

### 5.4 Margin, stop-out, death

- `margin_level = equity / used_margin`. Margin-call flag at 100% (visible to the agent, as on a real
  terminal). **Stop-out at 50%**: close the position with the largest loss first (MT-style), one at a
  time, until margin level > 50%.
- Intra-bar check uses each held instrument's adverse extreme (low bid for longs, high ask for
  shorts). This is conservative, because the extremes may not have happened at the same time;
  the approximation is documented. Fill at the trigger price + slippage, or at the open if the bar
  gapped through it.
- **Death:** equity ≤ 5% of initial ($0.50) ⇒ close everything, account terminated, no further
  actions accepted. Recorded reasons: `equity_floor`, `stop_out_cascade`, `gap_through`,
  `negative_balance`.

---

## 6. Reward design

### 6.1 Variants to compare (all computed on **net-of-cost equity**, never on gross PnL)

| ID | Reward per decision step | Behaviour it tends to produce |
|---|---|---|
| R1 (baseline) | `log(E_t / E_{t-1})` | Growth-optimal (Kelly). Variance is penalized automatically (≈ −σ²/2), and ruin is penalized by `log(0.05) ≈ −3.0` |
| R2 (negative control) | `(E_t − E_{t-1}) / E_0` | Risk-neutral. Expected to push towards max leverage and death; included to show the failure mode |
| R3 | R1 − λ·Δ(drawdown) | More conservative; **this is a risk preference**, so labelled as shaping |
| R4 | Differential Sharpe ratio (Moody & Saffell) | Smooth risk-adjusted reward; can be gamed by near-zero variance |
| R5 | Sparse: `log(E_T / E_0)` at episode end | No per-step shaping; hard credit assignment |

**Not used:** trade-count bonuses, win-rate rewards, rewards for "being in the market", or rewards on
realized PnL only (that teaches holding losers so they stay unrealized).

### 6.2 Reward-hacking surfaces and how they are closed

| Exploit | Closure |
|---|---|
| Fill at mid / stale prices / during closed sessions | bid/ask fills, tradeable mask, rejection logged |
| Lot rounding in the agent's favour | always round toward zero |
| Margin over-use between bars | margin checked at order time and intra-bar |
| Negative balance protection as a free option | NBP off in training; studied as an explicit variant |
| Episode reset as a "free life" | episode ends at death; evaluation uses one continuous life per test period |
| Weekend gap ignorance | valuation at the first real post-gap quote |
| Learning dates / price levels (memorization) | no absolute levels or dates in observations |
| Optimizing the reward rather than the outcome | every variant is evaluated on the **same** out-of-sample metrics, never on its own reward |

### 6.3 PPO suitability

PPO is a reasonable first learner. It is stable, handles continuous `Box` actions, and is well
tested in SB3. Its weaknesses here: it is on-policy, so sample efficiency is poor on expensive M1
rollouts; Gaussian policies clipped to [-1, 1] bias actions at the bounds (use a tanh-squashed or
Beta distribution); and it struggles with very low reward signal-to-noise. **SAC** (off-policy,
entropy-regularized) is a strong second candidate. **Evolution strategies / genetic algorithms**
are the third: they ignore per-step reward noise, optimize episode outcomes directly, parallelize
trivially, and give a natural "population that survives" world. Recommendation: PPO first (Phase 2
gate), then compare SAC and ES in Phase 3.

### 6.4 Baselines (reference points only; never given to agents)

Flat (always 0), uniform random, equal-weight buy-and-hold, and a random agent with turnover matched
to each learned agent. These define the **null distribution** that learned agents must beat.

---

## 7. Preventing data leakage and overfitting

### 7.1 Data splits (chronological, with embargo)

Proposed (adjustable to the available data):

| Split | Period | Use |
|---|---|---|
| Train | 2018-01 → 2022-12 | Stage A |
| Embargo | 2 weeks | buffer |
| Validation | 2023-01 → 2023-12 | hyperparameters, early stopping, model selection |
| Embargo | 2 weeks | |
| Test | 2024-01 → 2025-06 | Stage B. Evaluated **only at milestones**; every access logged |
| Final holdout | 2025-07 → present | touched once, before Stage C |

Plus **walk-forward** folds inside the training range to measure stability across regimes (2020
COVID, 2022 rate cycle, etc.).

### 7.2 Leakage controls

- Observations are built only from bars whose **close time ≤ decision time**. Enforced by a
  *future-perturbation test*: randomizing all data after *t* must leave every observation, action
  and fill up to *t* bit-identical.
- Normalization is **window-local** (relative to the observation window), so no global statistics
  are computed over future data.
- Multi-resolution bars are aggregated only from completed M1 bars.
- The test split is physically separate (different files, hash-recorded). The training config loader
  refuses to open it.

### 7.3 Overfitting controls

- Random episode start points; agents never see the same sequence in the same order.
- Model selection on validation only.
- **Beta vs. skill:** BTC, NAS100 and gold trended strongly over parts of this period. An agent that
  learns "be long" has learned drift, not trading. Report performance against buy-and-hold and as
  drift-adjusted return. Optional augmentation experiment: train on sign-flipped price paths so that
  drift can't be memorized.
- **Selection bias:** with N agents, the best one is partly lucky. Use the deflated Sharpe ratio
  and compare the *distribution* of learned agents against the null distribution from §6.4, with
  block-bootstrap confidence intervals.
- **Cost robustness:** re-evaluate every test agent at 1.5× and 2× spreads and commissions.
  An edge that disappears under +50% costs is not an edge.

### 7.4 Pre-registered success criteria (proposal)

An agent counts as having "discovered profitable behaviour" only if **all** hold on the test split:
1. net log-growth > 0, with a 95% block-bootstrap CI excluding 0;
2. it beats the 95th percentile of the turnover-matched random-agent distribution;
3. it is not explained by buy-and-hold drift;
4. it survives at 1.5× costs;
5. it is consistent across ≥ 2 non-overlapping test sub-periods.

---

## 8. Tests required before reinforcement learning (Phase 1 exit gate)

### 8.1 Data quality
- Monotonic, unique UTC timestamps; bid ≤ ask; low ≤ open, close ≤ high for both sides.
- Gaps detected and classified (weekend, session break, outage); DST transitions correct.
- Spread distribution per instrument within expected ranges (flags bad ticks).

### 8.2 Simulator correctness
1. **No look-ahead:** future-perturbation test (above).
2. **Fills:** buy at ask, sell at bid, at the next bar's open; slippage always adverse; commission exact.
3. **Accounting invariants (Hypothesis, random action sequences):**
   equity = balance + unrealized PnL; Δbalance = realized PnL − commission − financing;
   Σ events reconciles with final equity; zero-cost round trip at the same price ⇒ zero PnL.
4. **Golden scenarios:** hand-computed multi-step trades per instrument, including USD/JPY
   conversion and an XAU stop-out.
5. **Action mapping:** formula exact; gross leverage ≤ L for all inputs; zero action ⇒ flat;
   reversal; rounding toward zero; min-lot handling.
6. **Margin:** order larger than free margin is reduced/rejected and logged; no trading on
   non-existent capital.
7. **Stop-out & death:** scripted scenarios hit 50% margin level and the 5% floor; correct order of
   liquidation, prices and reasons; no actions accepted after death.
8. **Gaps:** weekend gap through stop-out ⇒ fill at gap price; NBP on/off behaviour.
9. **Financing:** charged only for positions held over rollover; triple Wednesday.
10. **Closed markets:** no fills, rejection logged, valuation unchanged until the next real quote.
11. **Determinism:** same seed ⇒ identical event-log hash; different seed ⇒ different slippage only.
12. **Isolation:** agent B's results are bit-identical whether it runs alone or alongside agents A, C, D.
13. **Baselines:** flat agent equity constant; buy-and-hold equals the analytic return minus costs
    and financing; the random agent bleeds roughly the expected cost.
14. **No order path:** static test that the package imports no broker trading client and contains no
    order-sending endpoint.
15. **Performance benchmark:** steps/s recorded; regressions fail CI.

### 8.3 Phase 2 gate (learning loop works)
- Gymnasium `check_env` passes.
- **Synthetic markets:** on a market with planted structure (e.g. an AR(1) mean-reverting series,
  or a drift switched by a visible regime flag), the agent must learn a profitable policy. On a
  pure random walk with costs it must converge to ≈ flat. This proves the learner works before
  real data is involved, and it doesn't hand the agent any strategy.

---

## 9. Project directory structure

```
trading-world/
├── pyproject.toml
├── README.md
├── docs/
│   ├── ARCHITECTURE.md            ← this document
│   ├── simulator-spec.md          ← exact formulas (fills, margin, financing, stop-out)
│   └── decisions/                 ← short ADRs, one per approved decision
├── configs/
│   ├── instruments/               ← contract size, lot step, min lot, sessions, pip, currency
│   ├── brokers/                   ← fractional.yaml, retail_lots_eu.yaml, retail_cent.yaml
│   └── experiments/               ← one YAML per experiment (data split, agents, reward, learner, seeds)
├── src/tradeworld/
│   ├── data/          # 1 market data engine: sources/, schema, align, quality, store
│   ├── replay/        # 2 clock, cursor, past-only window views
│   ├── market/        #   instrument specs, sessions, FX conversion
│   ├── execution/     # 3 fill model, slippage, commissions, lot rounding
│   ├── accounts/      # 4 ledger (event-sourced), margin, financing, stop-out, death
│   ├── world/         #   multi-account orchestrator + event log
│   ├── interface/     # 5+6 Observation builder, Action mapper (the agent boundary)
│   ├── envs/          # 7 Gymnasium env + wrappers
│   ├── rewards/       #   R1–R5
│   ├── agents/        # 8 policies, learners (ppo, sac, es), baselines
│   ├── training/      #   loops, schedules, checkpointing
│   ├── evaluation/    # 9 metrics, walk-forward, null distributions, behaviour clustering
│   ├── tracking/      # 10 run registry, artifacts, hashing
│   ├── live/          #   Phase 5: read-only live data adapters (no order code)
│   └── dashboard/     # 11 Phase 6
├── tests/
│   ├── unit/  property/  golden/  integration/  benchmarks/
├── scripts/           # download_data, validate_data, run_world, evaluate
├── data/              # gitignored: raw/, processed/ (Parquet, hashed)
└── runs/              # gitignored: one directory per experiment run
```

The earlier `btc-trading-bot/` folder is an indicator-based bot, which this spec explicitly excludes.
It shares no code with this project. Keep it or delete it (decision §12).

---

## 10. Metrics tracked per agent

Everything listed in spec §11, plus: log-growth, Sharpe/Sortino, deflated Sharpe, max drawdown and
duration, turnover, gross/net leverage over time, cost share of gross PnL, time in market,
long/short bias, instrument allocation entropy, holding-time distribution, stop-out count, time to
death, incapacitated duration, and margin-limited order count.

To answer "do different behaviours emerge?", agents are embedded by behaviour features (turnover,
holding time, leverage, allocation, long/short bias, loss-response) and clustered. This is compared
against the clustering you'd get from seeds alone.

Storage: per-step decisions as Parquet (`float32`, partitioned by run/agent), executions and
account events as separate event tables, and summary metrics as JSON per run. Every run records the
config, git commit, data file hashes, seeds and package versions.

---

## 11. Phased plan and first milestone

**Milestone 1 = Phase 0 + Phase 1: "a deterministic, validated simulator with baseline agents".**

| Step | Deliverable | Exit criteria |
|---|---|---|
| 1.0 | ADRs for the §12 decisions; `simulator-spec.md` with exact formulas | approved by you |
| 1.1 | Instrument & broker configs; `market/` specs; FX conversion | unit + golden tests |
| 1.2 | `accounts/` + `execution/` on **synthetic** price series | all §8.2 accounting/margin/stop-out/death tests green |
| 1.3 | `data/` pipeline: download, clean and align bid/ask M1 for the six instruments; quality report | §8.1 green; quality report reviewed |
| 1.4 | `replay/` + `interface/` + `world/` on real data | look-ahead, determinism and isolation tests green; ≥ 50k steps/s |
| 1.5 | Baseline agents (flat, random, buy-and-hold) across train/validation; cost calibration report | numbers match analytic expectations; report reviewed |

Only after Milestone 1 is signed off: Phase 2 (Gym env, synthetic-market learning gate, first PPO
agent), then Phases 3–6 as specified.

Data note: Dukascopy (bid and ask M1 for all six instruments) is reachable from this environment but
rate-limits it heavily. The bulk download (≈6 instruments × 7–8 years) is best run on your own
machine with the provided script, or slowly in the background here. Coinbase M1 works here for BTC.

---

## 12. Decisions needed from you before Phase 1

1. **Account model for learning:** `fractional` (recommended) vs. real lots from day one.
2. **Stage D broker profile:** standard 0.01 lots, cent account, or fractional. With standard lots a
   $10 account cannot trade at L = 3, so either Stage D uses L = 100 / cent accounts or it is
   expected to be untradeable.
3. **Margin regime:** EU retail (1:30 FX, 1:20 gold/indices, 1:2 crypto) or offshore (up to
   1:500). This interacts with your L = 3 / L = 100 tests.
4. **Negative balance protection** in training: off (recommended) or on.
5. **Data source and range:** Dukascopy bid/ask M1, 2018 → present (recommended).
6. **Decision interval:** start at 5 minutes on M1 fills (recommended), with 1-minute as an experiment.
7. **Learners to compare:** PPO first, then SAC and evolution (recommended).
8. **Compute:** where long training will run (this cloud session is temporary).
9. **`btc-trading-bot/`:** keep it separate or delete it.
