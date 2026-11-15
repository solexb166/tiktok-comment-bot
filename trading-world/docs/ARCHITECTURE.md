# Autonomous Multi-Agent Trading World — Phase 0 Architecture & Plan

Status: **Revision 3, for review.** Phase 1 does not start until this revision is approved.

Change log
- **r1**: initial architecture, critique and plan.
- **r2**: decisions 1–7; researched Exness specifications with verification status; Cent-account
  analysis; Dukascopy + Exness cost-model design; hardware assessment.
- **r3**: decisions 8–14 (§12.1). Formal definition of the three environments A/B/C (§1).
  Cent-only Stage D. Modular regional configs until the legal entity is verified. MetaApi
  deferred, so Phase 1 needs no credentials. Observation = 120 × M5 (configurable).
  Exness 1:Unlimited account leverage and its conditions (§6.3). Verified-spec guard (§6.6).
  Simulator test groups and synthetic-market experiments made explicit (§9).

---

## 0. Summary

1. **Three environments, never mixed** (§1): **A** fractional training (all six instruments, 3×
   cap); **B** Exness Cent Stage D (only verified Cent instruments, real specs, regional rules);
   **C** live virtual monitoring (live prices, simulated execution, **no order path at all**).
2. **On an Exness Cent account, a $10 agent can trade FX even at 3×**, but very coarsely:
   0, ±1.2× or ±2.3× of equity per FX pair. Gold needs ~$40 of notional. BTC and NAS100 availability
   on Cent is contradictory in Exness's own documentation and stays **excluded from B until verified**.
3. **The 5% death threshold is the experiment's survival rule.** Exness stop-out (0%, or 20% for
   some entities) and negative balance protection are separate broker mechanics. Because death
   fires at 5%, broker stop-out matters only on gaps.
4. **Exness 1:Unlimited leverage affects only broker margin in B. It does not change the agent's
   exposure cap.** It is conditional (equity threshold, eligibility, entity, instrument type,
   Higher Margin Requirement periods), so B models those conditions explicitly (§6.3).
5. **Phase 1 needs no broker credentials.** It uses synthetic data, Dukascopy history, official
   Exness tick history and documented specs. Environment B refuses to run while any required spec
   is unverified (§6.6).
6. On historical data this is N independent price-takers on a shared tape, not an interacting
   market (§4.4). "No trading knowledge" is approximated and every remaining bias is documented
   (§5). Success claims are pre-registered and tested against null distributions (§8.4).

---

## 1. The three environments

| | **A. Fractional training** | **B. Exness Cent Stage D** | **C. Live virtual monitoring** |
|---|---|---|---|
| Purpose | learning (Stage A) and unseen-history tests (Stage B) | realistic small-account evaluation (spec Stage D) | forward test on incoming prices (spec Stage C) |
| Instruments | all six: XAU/USD, EUR/USD, GBP/USD, USD/JPY, BTC/USD, NAS100 | **only instruments verified as available on the Exness Standard Cent account** for the verified entity; unverified ones are `tradeable = 0` | the instruments of the environment the agent is being monitored under (A or B rules) |
| Starting capital | $10 (normalized 1.0 internally) | $10 = 1,000 USC | $10 virtual |
| Position sizing | fractional (continuous), tiny minimum trade threshold | Exness Cent min lot 0.01, lot step 0.01, contract specs; round toward zero | per A or B profile |
| Exposure cap | **L = 3** (stress: L = 100) | the agent's cap is inherited from its experiment (3 or 100); **broker leverage 1:Unlimited** affects margin only | per profile |
| Costs | realistic spread (calibrated to Exness), slippage, swaps; commission 0 (Exness Standard Cent has none) | verified Exness Cent spreads (calibrated), commissions (none), verified swaps and triple days | per profile |
| Margin / stop-out | margin model not binding at 3×; **no broker stop-out** | verified Exness margin rules: leverage-based vs. fixed-margin instruments, equity tiers, HMR, stop-out level of the verified entity | per profile |
| Negative balance | **no NBP** (gap losses stay negative; reward ε-floored) | **Exness NBP** per verified rule + separate **no-NBP stress** variant | per profile |
| Death | equity ≤ 5% of initial | equity ≤ 5% of initial (evaluated before broker stop-out) | equity ≤ 5% of initial |
| Data | Dukascopy M1 bid/ask 2018 → present; Exness-calibrated spreads | same price path; Exness-calibrated spreads; Cent specs | live price feed (source TBD, §7.3); same bar schema |
| Real orders | impossible | impossible | **impossible**: no broker trading client is imported; enforced by a test |

Rules:
- An agent trained in A is evaluated in B **without retraining**. The gap is reported as the
  quantization/realism penalty (§7.5).
- Profiles are data (YAML), not code. One simulated account always uses exactly **one** profile;
  mixing Cent and Standard specs in one account is rejected by config validation.

---

## 2. Critique of the specification (status after r3)

| # | Issue | Resolution |
|---|---|---|
| 1 | $10 + L = 3 + real lots | A is fractional; B is Cent-only, where FX is tradable at 3× (coarsely) |
| 2 | Normalized equity is not scale-invariant | Ledger in real USD/USC; normalization only in observations and rewards |
| 3 | Multi-agent world on replay | Price-takers; diversity via seeds, learners, rewards and evolution (§4.4) |
| 4 | Reward = risk preference | Log-wealth baseline; shaping variants labelled (§7) |
| 5 | Hidden rebalancing in target exposure | Only deltas ≥ one lot step execute (§6.7) |
| 6 | Decision frequency | 5-minute default; 1, 15, 60 configurable |
| 7 | Observation window | **120 × M5 bars (10 h), configurable** (length and resolution) |
| 8 | Trading sessions | UTC clock + per-instrument tradeable mask |
| 9 | Death vs. stop-out | Death at 5% is the experimental criterion; stop-out/NBP modelled separately (B only) |
| 10 | Zombie accounts | `incapacitated` status tracked (relevant in B with min lots) |
| 11 | NBP as a free option | A: no NBP. B: Exness NBP + no-NBP stress |
| 12 | Historical swaps unavailable | Documented snapshot approximation; swap-free majors as a variant |
| 13 | Success criteria | Pre-registered (§8.4) |
| 14 | Compute | Local first; hardware assessment pending (§14) |
| 15 | No real orders | No order path in the package, test-enforced |
| 16 | Train/serve skew in C | Bar-schema parity test |
| 17 | Mid-only data | Dukascopy bid/ask |
| 18 | Mixing account types | **Resolved: Cent-only B** |
| 19 | HMR periods need a news calendar | Configurable HMR windows; calendar source unresolved (§13) |
| 20 | Legal entity unknown | **Modular regional profiles; B is blocked until verified** (§6.5) |
| 21 | **New:** "Unlimited leverage" vs. exposure cap | Interpreted as the Exness **account leverage setting** in B. The agent's exposure cap L stays 3 (stress 100). Needs your confirmation (§12.2) |

---

## 3. Expectations

Most agents will likely learn to stay flat, die, or capture drift that does not generalize.
At 5-minute decisions, turnover costs dominate small edges. The design makes a null result
measurable and trustworthy, and proves the learner on synthetic markets first (§9.3).

---

## 4. Architecture

### 4.1 Component map

```
                ┌──────────────── offline ─────────────────┐
 Dukascopy M1 → │ 1 Market Data Engine (download, clean,   │ → Parquet store (immutable, hashed)
 bid/ask        │   align, validate, session masks)        │
 Exness ticks → │   Spread calibration (Exness cost model) │ → calibrated cost tables (hashed)
 Exness specs → │   Broker profile build + verification    │ → profile YAML with per-field status
 (documented)   └──────────────────────────────────────────┘
                                   │
            ┌──────────── deterministic core (no learning) ───────────┐
            │ 2 Replay: UTC clock, cursor, past-only windows          │
            │ 3 Execution: bid/ask fills, latency, slippage, lots     │
            │ 4 Accounts: ledger, margin, swaps, stop-out, NBP, death,│
            │   USD/USC/JPY conversion; World: one clock, N accounts  │
            └─────────────────────────────────────────────────────────┘
                     ▲ actions                    │ observations
            ┌────────┴─────────────┐   ┌──────────▼───────────┐
            │ 6 Action Interface   │   │ 5 Observation Iface  │   ← only surface agents see
            └────────▲─────────────┘   └──────────┬───────────┘
            ┌────────┴────────────────────────────▼────────────┐
            │ 7 Gymnasium Env + pluggable reward               │
            └────────▲─────────────────────────────────────────┘
   8 Training (PPO → SAC, ES)    9 Evaluation (A → B transfer, walk-forward, null distributions)
  10 Tracking (config, git, data, cost-table and profile hashes, seeds)   11 Dashboard (Phase 6)
  12 Live virtual feed (Phase 5, environment C; read-only market data only)
```

### 4.2 Design principles

- **Deterministic core**: seeded generators only; same inputs ⇒ bit-identical event log.
- **Event-sourced accounting**: fills, costs, swaps, stop-outs, NBP resets and deaths are immutable events.
- **Isolation**: agents see only `Observation` → `Action`. Each agent has its own parameters, optimizer,
  buffers and RNG stream (`SeedSequence(seed).spawn(n)`); no parameter sharing.
- **Profiles as data**: broker, region and environment profiles are versioned YAML, and every field
  carries `value`, `source`, `retrieved`, `status`.
- **Simulator first**: no Gymnasium or PyTorch imports in the core.
- **Performance**: pre-aligned `float32` arrays, zero-copy windows, numba only if profiling shows the need.

### 4.3 Technology stack

Python 3.11+, NumPy (+ numba if needed), Polars/pandas → Parquet, DuckDB for log queries,
Pydantic + YAML configs, Gymnasium, PyTorch with Stable-Baselines3 for the first PPO/SAC, own ES/GA
loop, pytest + Hypothesis, own run registry + TensorBoard, Streamlit → FastAPI for the dashboard.
Rationale as in r2: well-tested components where bugs are costly (RL), own code where control matters
(simulator, ledger).

### 4.4 What "multi-agent" means

Price-takers on a shared tape. Running them together is mathematically identical to running them
alone, and a test enforces that. Diversity comes from seeds, learners, rewards,
observation/decision settings and population-based evolution, with selection only on account outcomes.

---

## 5. Learning without predefined strategies

- **Observation (default):** for each of the six instruments, the last **120 × M5 bars** (configurable
  length and resolution), built only from completed M1 bars. Per bar: mid log return, range/mid,
  body/mid, spread/mid, relative tick volume, tradeable flag. Plus own account state (log equity /
  initial, free-margin fraction, margin level, exposure per instrument, unrealized PnL / equity, time
  in position, margin-call flag where the profile has one). Optional time-of-day / day-of-week (ablated).
- **Not given:** indicators, signals, absolute prices, dates, other agents, simulator internals,
  cost-model parameters, future bars.
- **Documented biases:** feature transform, network architecture, window and decision interval,
  reward, learner. All are ablated.
- The engine never trades for the agent except for enforced constraints (margin rejection, broker
  stop-out in B, NBP in B, death), each logged with a reason.

---

## 6. Broker specification: Exness (environment B)

### 6.0 Verification policy

- Direct fetches of Exness's help center return HTTP 403 from this environment. r2 facts came from
  search-engine extracts of official pages, cross-checked, and Exness's pages **contradict each
  other** on several points.
- Status per field: **✅ verified** (official source, consistent, recorded with URL and date) ·
  **⚠️ conflicting** · **❓ unverified/derived** · **⛔ unknown**.
- **Verification sources for Phase 1 (no credentials needed):** (1) official Exness help-center and
  product pages, copied into `docs/exness-spec-verification.md` with retrieval date. Since automated
  fetches are blocked, these can be saved by you from a browser, or pasted. (2) Official Exness
  tick history (bid/ask), which empirically confirms symbol existence, digits, trading sessions and
  spreads. (3) **Later, optional:** a read-only Exness demo snapshot via MetaApi (deferred, decision 10).

### 6.1 Account-level facts (Standard Cent)

| Item | Value | Status |
|---|---|---|
| Account currency | USC; $10 = 1,000 USC | ✅ |
| Lot | 1 Cent lot = 0.01 standard lot (≈ 1,000 units of base currency) | ✅ |
| Volume | min 0.01 (ETHUSDc 0.1); max 200 lots | ✅ |
| Commission | none | ✅ |
| Spread | "from 0.3 pips" (marketing floor; real distribution from tick calibration) | ✅ floor / ⛔ distribution |
| Margin call | 60% | ✅ |
| Stop-out | 0% default; **20% for Exness (KE) and Exness (JO)**; stocks up to 100% in high-risk periods | ✅ |
| Stop-out closing order | reportedly most unprofitable first | ❓ |
| NBP | balance reset to 0 only after all orders were closed by stop-out and balance < 0 | ✅ |
| Swap timing | 21:00 GMT+0 summer / 22:00 GMT+0 winter | ✅ (Exness) / ⚠️ (third-party: 00:00 server) |
| Triple swap | FX Wednesday (USDCAD Thursday); indices Friday; crypto Friday | ✅ |
| Swap-free | majors (and most minors) swap-free per Exness; status may depend on behaviour | ✅ / ❓ |
| HMR window | 15 min before high-impact news → 90 s or 5 min after | ⚠️ |

### 6.2 Instruments on the Cent account

| Project instrument | Cent symbol | In B? | Smallest position | Margin type | Status |
|---|---|---|---|---|---|
| EUR/USD | EURUSDc | **yes, once spec fields are verified** | ≈ €10 | leverage-based | availability ✅ |
| GBP/USD | GBPUSDc | yes, once verified | ≈ £10 | leverage-based | ✅ |
| USD/JPY | USDJPYc | yes, once verified | $10 | leverage-based | ✅ |
| XAU/USD | XAUUSDc | yes, once verified | ≈ 0.01 oz (derived from a 100 oz standard contract) | leverage-based; HMR band | availability ✅ / contract ❓ |
| BTC/USD | BTCUSDc | **excluded until verified** | reported 0.01 BTC contract (denomination unclear) | fixed margin | ⚠️ (MT5-only listing vs. "unavailable"; blocked for KE) |
| NAS100 | USTECc | **excluded until verified** | ⛔ | fixed 0.25% for USTEC (standard) | ⚠️ (listed vs. "indices not offered on Cent") |

### 6.3 Leverage: 1:Unlimited (decision 14) and its conditions

**Interpretation (to confirm, §12.2):** "unlimited leverage" is the **Exness account leverage
setting** used to compute broker margin in environment B. It does **not** remove the agent's
exposure cap: `w_i = L·a_i / max(1, Σ|a_j|)` needs a finite L, and A is defined with L = 3. The cap
for B runs is the one the agent was trained or stress-tested with (3 or 100).

What Exness documents about Unlimited (to be verified field by field):

| Condition | Value | Status |
|---|---|---|
| Equity tiers | < $5,000: 1:2000 default, 1:Unlimited **if eligible**; $5k–30k 1:2000; $30k–100k 1:1000; ≥ $100k 1:500 | ✅ tiers |
| Eligibility for Unlimited | equity below a threshold **plus** trading-history requirements; threshold reported as $5,000 by some sources and ~$1,000 by others | ⚠️ |
| Entity caps | Exness (KE): max 1:400 | ✅ |
| Instruments not affected | fixed-margin instruments (crypto, indices, some others) use a fixed margin regardless of account leverage | ✅ |
| HMR periods | higher margin (e.g. gold 1:200–1:1000 band, crypto 1:200–1:400, indices 1:50–1:100 reported) overrides account leverage around high-impact news | ✅ existence / ⚠️ windows and bands |
| Margin formula under Unlimited | ⛔ (how Exness computes "unlimited" margin, e.g. minimal vs. zero, and margin level under it) | ⛔ |

Modelling consequence: in B with Unlimited, margin will essentially never bind for FX at L = 3 or
L = 100. The binding constraints are the agent's L cap, lot quantization, costs and the 5% death
floor. HMR can still bind during news windows (only in L = 100 stress tests), so an HMR calendar
is needed for B's high-leverage runs (§13).

The profile supports all leverage values (`1:Unlimited`, `1:2000`, `1:400`, …) so sensitivity runs
can show whether results depend on Unlimited being available.

### 6.4 $10 account in B (Cent-only)

Max notional: L = 3 → $30; L = 100 → $1,000.

| Instrument | Smallest position | L = 3 exposure steps | L = 100 |
|---|---|---|---|
| EURUSDc | ≈ $11.6 | 0, ±1.16×, ±2.3× | ≈ 0.12× steps |
| GBPUSDc | ≈ $13 | 0, ±1.3×, ±2.6× | ≈ 0.13× steps |
| USDJPYc | $10 | 0, ±1×, ±2×, ±3× | 0.1× steps |
| XAUUSDc | ≈ $40 | not tradable | ≈ 4× steps |
| BTCUSDc / USTECc | excluded until verified | — | — |

Costs: no commission. The minimum EURUSDc position moves $0.001 (0.1 USC) per pip. A 1-pip spread is
≈ 0.9 bps of notional, so a round trip at 2× ≈ 1.8 bps of equity; 100 round trips/day ≈ 1.8%/day.
Exact pip values are golden-tested (factor-of-100 USC/USD risk).

### 6.5 Regional configuration (modular until verified)

```
configs/brokers/exness/
  account_types/standard_cent.yaml       # account-level facts (§6.1), per-field status
  instruments/{EURUSDc,...}.yaml         # per-symbol specs (§6.2)
  regions/
    global.yaml                          # stop-out 0%, leverage tiers incl. Unlimited
    ke.yaml                              # stop-out 20%, max 1:400, BTCUSDc blocked
    jo.yaml                              # stop-out 20%, leverage cap (to verify)
    unverified.yaml                      # ACTIVE DEFAULT: marks all region-dependent fields ⛔
```
A B profile = account type ⊕ region ⊕ instruments. **The legal entity serving your account is not
assumed** (decision 9). Until you confirm it, the active region is `unverified`: B can be built and
unit-tested, and can run explicitly labelled **sensitivity runs across regions**, but no B result is
reported as "the Exness Stage D result".

### 6.6 Verified-spec guard

Loading a B profile for an evaluation fails if any field the run needs is not ✅. The error lists
the fields, sources and statuses. A `--sensitivity` flag allows running over candidate values,
and the results are tagged `unverified-spec` in tracking.

### 6.7 From action to orders (no hidden rebalancing)

```
a ∈ [-1,1]^n  →  w_i = L·a_i / max(1, Σ|a_j|)               (gross ≤ L)
target_units_i = w_i · equity / (price_i · contract_size_i · fx_to_account_i)
A: fractional (no rounding; minimum trade threshold)
B: round toward zero to lot_step; |target| < min_lot ⇒ 0
execute delta_i only if |delta_i| ≥ lot_step_i ; reductions first, then increases
B margin check: increases beyond free margin reduced to the largest affordable size (logged)
```
Decision at the close of the M5 decision bar → fill at the next **M1** open; buys at ask, sells at
bid; adverse slippage (base + range-scaled + seeded noise); closed market ⇒ rejected and logged.

### 6.8 Death, stop-out and NBP: order of evaluation (B)

At every M1 step:
1. Revalue positions (intra-bar adverse extremes, documented as conservative).
2. **Death check:** equity ≤ 5% of initial ⇒ close all, terminate (`equity_floor`).
3. **Broker stop-out check** (only reachable if a gap skipped past the 5% floor): margin level ≤
   the verified stop-out level ⇒ close positions at the gap price (order per verified rule) ⇒ agent
   dead (`gap_through`).
4. **NBP:** balance < 0 after a full stop-out ⇒ reset to 0 (`nbp_reset`) in the NBP profile; kept as
   debt in the no-NBP stress profile.

In environment A, steps 3–4 are replaced by: negative equity is kept, and the reward is ε-floored.

---

## 7. Rewards and learners

### 7.1 Reward variants (net-of-cost equity)

| ID | Reward | Expected behaviour |
|---|---|---|
| R1 baseline | `log(max(E_t, ε·E_0)/E_{t-1})`, ε = 0.01 | growth-optimal; penalizes variance and ruin |
| R2 negative control | `(E_t − E_{t-1})/E_0` | risk-neutral; expected to drive max leverage and death |
| R3 | R1 − λ·Δdrawdown | labelled risk preference |
| R4 | differential Sharpe ratio | smooth; gameable by near-zero variance |
| R5 | sparse terminal log-wealth | no shaping; hard credit assignment |

Never: trade-count bonuses, win-rate rewards, market-presence rewards, realized-only PnL rewards.

### 7.2 Reward-hacking closures

Bid/ask fills and tradeable mask; rounding toward zero; margin checks; no NBP in A; one continuous
life in evaluation; gap valuation; no absolute levels or dates; all variants judged on the same
out-of-sample metrics; cost-robustness tests at 1.5× and 2×; calibration tests against held-out
Exness tick months.

### 7.3 Learners and baselines

PPO first (tanh-squashed or Beta policy), then SAC and ES/GA. Baselines (flat, uniform random,
equal-weight buy-and-hold, turnover-matched random) form the null distribution and are never shown
to agents.

### 7.4 Live feed for environment C (Phase 5, unresolved source)

Read-only price sources only: candidates are Exness via a read-only MetaApi/MT5 connection (when
re-introduced), a public crypto feed for BTC, or another FX data vendor. Whatever is chosen must
produce the identical bar schema (bar-parity test). Execution is always the simulator.

### 7.5 A → B transfer

B evaluation without retraining. Reported: performance gap vs. A, the share of targets rounded to
zero, turnover change, and the effect of excluded instruments. Optional: quantization-aware
fine-tuning on training-period data only.

---

## 8. Leakage and overfitting

### 8.1 Splits

| Split | Period | Use |
|---|---|---|
| Train | 2018-01 → 2022-12 | Stage A (environment A) |
| Embargo | 2 weeks | |
| Validation | 2023-01 → 2023-12 | hyperparameters, model selection |
| Embargo | 2 weeks | |
| Test | 2024-01 → 2025-06 | unseen history (A and B); milestone-only, access logged |
| Final holdout | 2025-07 → present | touched once, before C |
| Exness cost calibration | Exness tick months (as old as available; preferably pre-test) | spread statistics only, never price inputs |

Walk-forward folds inside the training range. If Exness tick history only covers recent months, the
calibration overlaps the test period. That is a documented leak of **cost statistics only**.

### 8.2 Leakage controls
Only bars closed at or before decision time; future-perturbation test; window-local normalization;
M5 aggregation only from completed M1 bars; physically separate test files that the training loader
refuses.

### 8.3 Overfitting controls
Random episode starts; validation-only selection; drift-adjusted reporting (optional sign-flip
augmentation); deflated Sharpe and null distributions with block bootstrap; cost robustness.

### 8.4 Pre-registered success criteria (test split, all required)
1. Net log-growth > 0, 95% block-bootstrap CI excluding 0.
2. Above the 95th percentile of turnover-matched random agents.
3. Not explained by buy-and-hold drift.
4. Still positive at 1.5× costs.
5. Consistent across ≥ 2 non-overlapping sub-periods.
6. Reported separately in B (verified profile only), including quantization and excluded-instrument effects.

---

## 9. Validation plan (retained from r1/r2, now explicit)

### 9.1 Data-quality checks
Monotonic unique UTC timestamps; bid ≤ ask; OHLC consistency per side; gap classification
(weekend, session break, outage); DST; spread outliers; Dukascopy-vs-Exness mid divergence on
overlapping months.

### 9.2 The 15 simulator test groups (Phase 1 exit gate)

1. **No look-ahead:** future-perturbation test; M5 bars only from completed M1 bars.
2. **Fills:** ask/bid at the next M1 open; adverse slippage; exact commissions (0 for Exness Cent).
3. **Accounting invariants** (Hypothesis): equity = balance + unrealized; Δbalance = realized −
   costs − swaps; ledger reconciliation; zero-cost round trip ⇒ zero PnL.
4. **Golden scenarios:** hand-computed trades per instrument, including USD/JPY → USD/USC conversion,
   the USC/USD factor-of-100 check, and XAUUSDc pip values.
5. **Action mapping:** formula exact; gross ≤ L; zero ⇒ flat; reversal; rounding toward zero; min lot.
6. **Margin:** leverage-based vs. fixed-margin instruments; equity tiers; Unlimited; HMR windows;
   orders beyond free margin reduced and logged.
7. **Death and stop-out:** death at 5% before broker stop-out; gap-through stop-out at the verified
   level and closing order; no actions after death.
8. **NBP:** reset only after a full stop-out with negative balance; no-NBP profile keeps the debt;
   A keeps negative equity with an ε-floored reward.
9. **Swaps:** charged only across rollover; triple days per instrument class; swap-free variant.
10. **Sessions:** no fills when closed; daily breaks for gold and indices; 24/7 BTC; weekend-gap valuation.
11. **Determinism:** same seed ⇒ identical event-log hash.
12. **Isolation:** agent results identical alone vs. with other agents.
13. **Baselines:** flat, buy-and-hold and random agents match analytic expectations.
14. **No order path / profile integrity:** no broker trading client or trade endpoint in the
    package; one profile per account (no Cent/Standard mixing); the verified-spec guard rejects
    unverified B fields.
15. **Performance benchmark:** steps/s recorded per machine; regressions flagged.

### 9.3 Synthetic-market learning experiments (Phase 2 gate)

Before any real data is used for learning, the learning loop must pass:

| Experiment | Synthetic market | Expected outcome |
|---|---|---|
| S1 | AR(1) mean-reverting returns with costs | profitable policy learned; beats the null distribution |
| S2 | Drift that switches sign with a visible regime flag | learns to follow the flag's regime |
| S3 | Pure random walk with costs | converges to ≈ flat (no turnover) |
| S4 | Random walk, no costs | no systematic profit (sanity: zero-edge market) |
| S5 | Leakage canary: future returns hidden in a field the agent **cannot** see | no profit; a profit means a leak |
| S6 | High-leverage trap: positive drift but fat-tailed crashes; L = 100 | R1 agents survive with moderate exposure; R2 agents die (validates reward behaviour) |

Plus Gymnasium `check_env`. These are tests of the learning machinery, not strategies given to agents.

---

## 10. Directory structure

```
trading-world/
├── pyproject.toml
├── README.md
├── docs/
│   ├── ARCHITECTURE.md
│   ├── simulator-spec.md
│   ├── exness-spec-verification.md   ← each field: value, source URL, retrieved date, status
│   └── decisions/                    ← ADRs
├── configs/
│   ├── environments/{A_fractional, B_exness_cent, C_live_virtual}.yaml
│   ├── instruments/
│   ├── brokers/
│   │   ├── fractional.yaml
│   │   └── exness/{account_types, instruments, regions}/   (see §6.5)
│   └── experiments/
├── src/tradeworld/
│   ├── data/          # sources/{dukascopy, exness_ticks}, align, quality, store
│   ├── calibration/   # Exness spread model + validation
│   ├── brokerspec/    # profile loading, composition, verified-spec guard
│   ├── replay/  market/  execution/  accounts/  world/
│   ├── interface/  envs/  rewards/  agents/  training/
│   ├── evaluation/  tracking/
│   ├── live/          # Phase 5: read-only feeds only
│   └── dashboard/     # Phase 6
├── tests/  unit/ property/ golden/ integration/ synthetic/ benchmarks/
├── scripts/  hardware_check, download_dukascopy, import_exness_ticks, calibrate_costs, run_world
├── data/   # gitignored
└── runs/   # gitignored
```

`btc-trading-bot/` stays in the repository, untouched, and is not part of this project (decision 12).

---

## 11. Metrics per agent

Everything in spec §11, plus log-growth, Sharpe/Sortino, deflated Sharpe, drawdown duration,
turnover, gross/net leverage, cost and swap share of gross PnL, time in market, long/short bias,
allocation entropy, holding-time distribution, stop-out/NBP counts, time to death, incapacitated
duration, margin-limited orders, rounded-to-zero rate (B). Behaviour clustering compares emergent
diversity against seed-only diversity.

---

## 12. Decisions

### 12.1 Recorded

| # | Date | Decision |
|---|---|---|
| 1 | 2026-11-15 | Fractional position sizing during training, with realistic costs |
| 2 | 2026-11-15 | Exness specs for Stage D (refined by 8) |
| 3 | 2026-11-15 | Verified Exness margin and leverage, configurable per instrument and account type |
| 4 | 2026-11-15 | Exness NBP in the broker profile, plus a no-NBP stress scenario |
| 5 | 2026-11-15 | Dukascopy M1 bid/ask from 2018; Exness data for calibration |
| 6 | 2026-11-15 | 5-minute decisions; 1, 15 and 60 configurable |
| 7 | 2026-11-15 | Develop locally first; hardware assessment before long runs |
| 8 | 2026-11-15 | **Stage D = Exness Cent-only account.** No Cent/Standard mixing. Only instruments verified as available on Cent. A keeps all six instruments |
| 9 | 2026-11-15 | **Legal entity is not assumed** (not from nationality or residence). Regional config is modular until verified |
| 10 | 2026-11-15 | **MetaApi deferred.** Start with official Exness tick history and verified specs. Phase 1 needs no credentials; read-only demo integration later |
| 11 | 2026-11-15 | **Observation = 120 × M5 bars (10 h)**, configurable for experiments |
| 12 | 2026-11-15 | **Keep `btc-trading-bot/`** unmodified; inspect separately later |
| 13 | 2026-11-15 | **Three environments A/B/C** as defined in §1. **5% death is the experimental survival criterion**; broker stop-out and NBP modelled separately per verified Exness rules. C has absolutely no real order execution |
| 14 | 2026-11-15 | **Unlimited leverage** (interpretation in §6.3, to confirm) |

### 12.2 Needs your confirmation

1. **Decision 14 interpretation:** Exness account leverage = 1:Unlimited for broker margin in B,
   while the agent's exposure cap stays L = 3 (stress L = 100)? Or did you mean removing the
   exposure cap? (Removing it would need a different action mapping, because the formula requires
   a finite L.)
2. **Legal entity:** to be supplied by you once verified (Personal Area / account agreement).

---

## 13. Unresolved specifications (blocking for B results, not for Phase 1)

| # | Item | Needed for | How to resolve |
|---|---|---|---|
| U1 | Legal entity → stop-out level, leverage cap, instrument restrictions | B | you verify; then select `regions/*.yaml` |
| U2 | BTCUSDc availability on Cent (MT5) and contract denomination | B instrument set | official page copy; Exness tick history symbol list; later demo |
| U3 | USTECc availability and contract size on Cent | B instrument set | same |
| U4 | XAUUSDc contract size (derived 1 oz per Cent lot) and pip value | B golden tests | official spec copy; tick data digits |
| U5 | Unlimited-leverage eligibility conditions and margin formula | B margin | official Leverage page copy |
| U6 | HMR windows (90 s vs. 5 min after news), bands per instrument, and a historical high-impact news calendar | B at L = 100 | official HMR page; choose a calendar source |
| U7 | Stop-out closing order | B gap scenarios | official page or support |
| U8 | Swap values (long/short) per symbol, swap-free eligibility | A and B costs | official swap tables (snapshot), later demo |
| U9 | Whether Exness tick history includes Cent symbols, or whether Cent spreads equal Standard | cost calibration | inspect the tick-history symbol list |
| U10 | Rollover time (21/22:00 GMT vs. 00:00 server) and server time zone | swaps, sessions | official About-swap page copy; tick timestamps |
| U11 | Trading sessions and daily breaks per symbol | session mask | official pages; inferred from tick gaps and cross-checked |
| U12 | Live price source for C | Phase 5 | decide in Phase 5 |
| U13 | Dukascopy BTC history start date | A data range | on download |

Assumptions in force until resolved: Cent spreads ≈ Standard spreads (U9); swaps from a current
snapshot applied to all history (U8); conservative intra-bar stop evaluation; fractional A uses
Exness-calibrated spreads with no commission.

---

## 14. Milestone 1 (Phase 0 + Phase 1), no credentials required

| Step | Deliverable | Exit criteria |
|---|---|---|
| 1.0 | ADRs; `simulator-spec.md`; environment configs A/B/C | approved by you |
| 1.0a | Hardware assessment (§15) | resource plan agreed |
| 1.1 | Profile system: fractional + modular Exness profiles, per-field status, verified-spec guard, `exness-spec-verification.md` skeleton | unit tests; guard tests |
| 1.2 | `accounts/` + `execution/` on synthetic prices (A and B mechanics) | test groups 2–9, 11, 12, 14 green |
| 1.3 | Dukascopy pipeline (local download) + data-quality report | §9.1 green |
| 1.4 | Exness tick import + spread calibration (first symbols: EURUSD, GBPUSD, USDJPY, XAUUSD) | calibration validation passes |
| 1.5 | `replay/` + `interface/` (120 × M5) + `world/` on real data | test groups 1, 10, 15 green |
| 1.6 | Baselines in A (and B with the `unverified` region, sensitivity-tagged) | group 13 green; report reviewed |

Phase 2 (Gym env + synthetic experiments S1–S6 + first PPO agent) starts only after sign-off.

---

## 15. Hardware assessment (step 1.0a)

Please run on the training machine and paste the output, plus RAM size and GPU model/VRAM:

```bash
python -c "import os,platform,shutil;print(platform.platform());print('cpu',platform.processor(),os.cpu_count(),'threads');print('disk free GB',round(shutil.disk_usage('.').free/1e9))"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

| Resource | Minimum | Comfortable | Driver |
|---|---|---|---|
| Disk | 10 GB | 50 GB+ | Dukascopy raw files, Parquet store (~0.5–1 GB), Exness tick months (large), run logs |
| RAM | 8 GB | 16–32 GB | aligned M1 arrays ≈ 1 GB float32 + parallel envs |
| CPU | 4 cores | 8+ cores | the simulator is CPU-bound |
| GPU | none | CUDA GPU ≥ 6 GB | larger encoders, many agents |

Ballpark: the 2018–2022 range at 5-minute decisions is ≈ 0.3–0.5 M decision steps per pass. PPO
budgets of 10–50 M steps per agent mean hours per agent on a mid-range CPU. Benchmarks come before
any long run.

---

## Sources (Exness research, 2026-11-15)

Official Exness pages (search-engine extracts; direct fetches returned 403):
- Standard Cent account — https://get.exness.help/hc/en-us/articles/17537782786588-Standard-Cent-account
- Standard account — https://get.exness.help/hc/en-us/articles/17537782738460-Standard-account
- Margin call and stop out levels — https://get.exness.help/hc/en-us/articles/360014693340-Margin-call-and-stop-out-levels-by-account-type
- 0% stop out / Stop Out Protection — https://get.exness.help/hc/en-us/articles/17685808714140-0-stop-out-Stop-Out-Protection
- Negative Balance Protection — https://get.exness.help/hc/en-us/articles/18055126006044-Negative-Balance-Protection
- Margin — https://get.exness.help/hc/en-us/articles/360014504992-Margin
- Leverage — https://get.exness.help/hc/en-us/articles/360014529380-Leverage
- Higher Margin Requirements — https://get.exness.help/hc/en-us/articles/20618327917596-Higher-Margin-Requirements-HMR
- Commodities — https://get.exness.help/hc/en-us/articles/17854173039388-Commodities
- Indices — https://get.exness.help/hc/en-us/articles/17854383867548-Indices
- Cryptocurrencies — https://get.exness.help/hc/en-us/articles/17854191888540-Cryptocurrencies
- About swap — https://get.exness.help/hc/en-us/articles/360014709151-About-swap
- Account type suffixes — https://get.exness.help/hc/en-us/articles/360014560220-Account-type-suffixes
- Forex — https://www.exness.com/forex/
- US Tech 100 — https://www.exness.com/indices/us-tech-100/
- Tick history — https://www.exness.com/tick-history/

Third-party (only for ❓ items):
- https://tradersunion.com/brokers/forex/view/exness/stop-out-level-and-trailing-stop/
- https://onforex.co/en/exness/leverage/
- https://www.brokerxplorer.com/article/exness-tick-history-report-a-beginners-guide-4624
- https://pypi.org/project/exness-data-preprocess/
