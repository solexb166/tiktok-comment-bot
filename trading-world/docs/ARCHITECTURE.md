# Autonomous Multi-Agent Trading World — Phase 0 Architecture & Plan

Status: **Revision 4, for review.** Phase 1 does not start until this revision is approved.

Change log
- **r1**: initial architecture, critique and plan.
- **r2**: decisions 1–7; researched Exness specifications; Cent analysis; Dukascopy + Exness cost model.
- **r3**: decisions 8–14; environments A/B/C; Cent-only Stage D; modular regions; MetaApi deferred;
  120 × M5 observation; verified-spec guard; explicit test groups and synthetic experiments.
- **r4**: decision 14 clarified (decision 15): **no fixed exposure ceiling in any environment.**
  New uncapped action space (§5). Environment A gets an explicit margin and liquidation model.
  3×/10×/50×/100× become benchmark experiments only. Numerical safeguards and per-decision
  leverage recording. Environment B uses only verified Exness leverage and margin rules
  (Unlimited never assumed). Environment C gets configurable leverage experiments. New synthetic
  experiment S7 (does the agent find growth-optimal leverage?).

---

## 0. Summary

1. **Agents choose their own exposure, with no behavioural ceiling.** The action space sets
   direction/allocation and **log-scaled gross exposure** independently, spanning 0.0001× to
   10,000× of equity (§5). Exposure above what the market allows is limited the way a real
   broker limits it (margin rejection, stop-out, death), not by a cap inside the agent interface.
2. **Three kinds of limits, kept strictly separate (§5.1):**
   *behavioural caps*: **none** in the primary experiments (3×, 10×, 50× and 100× are benchmark
   variants only); *market/broker constraints*: equity, margin, costs, maximum volume, stop-out,
   death; *numerical safeguards*: finite value ranges that sit far above anything margin-feasible.
   They are logged whenever touched and are never silent.
3. **Environment A now has a margin and liquidation model.** Without margin, "uncapped" would mean
   infinite exposure. The default is a documented **A-reference margin schedule** (§5.4), with an
   "unlimited-margin" variant as an experiment.
4. **Environment B uses only verified Exness Cent leverage and margin rules.** Unlimited is never
   assumed from a small balance. It is enabled only if the verified eligibility conditions are met
   by the simulated account (§7.3).
5. **Every decision records requested vs. executed exposure, effective gross and net leverage,
   margin level and any rejection** (§5.7).
6. Three environments: **A** fractional training · **B** Exness Cent Stage D · **C** live virtual
   (no real orders, ever) (§1). The 5% death rule is the experiment's survival criterion; broker
   stop-out and NBP are modelled separately (§7.8).
7. Phase 1 needs no broker credentials. Price-taker agents, documented biases, pre-registered
   success criteria and null distributions, as in r3.

---

## 1. The three environments

| | **A. Fractional training** | **B. Exness Cent Stage D** | **C. Live virtual monitoring** |
|---|---|---|---|
| Purpose | learning (Stage A) and unseen-history tests (Stage B) | realistic small-account evaluation (Stage D) | forward test on incoming prices (Stage C) |
| Instruments | all six: XAU/USD, EUR/USD, GBP/USD, USD/JPY, BTC/USD, NAS100 | only instruments **verified** as available on Exness Standard Cent for the verified entity | per chosen profile (A or B rules) |
| Starting capital | $10 (normalized 1.0 internally) | $10 = 1,000 USC | $10 virtual |
| Sizing | fractional; sub-cent positions treated as zero (numerical) | Cent lots: min 0.01, step 0.01; round toward zero | per profile |
| **Exposure ceiling** | **none** (benchmark variants: 3×, 10×, 50×, 100×) | **none** beyond verified Exness margin rules, lot limits and max volume | **configurable leverage experiments** (uncapped, benchmark caps, or B margin settings) |
| Margin | **A-reference margin schedule** (§5.4); variant: unlimited-margin | verified Exness Cent rules: account leverage setting, equity tiers, fixed-margin instruments, HMR, entity limits (§7.3) | per profile |
| Margin rejection | atomic: increases rejected if they don't fit free margin (§5.5) | same, per Exness behaviour (to verify, U14) | per profile |
| Stop-out | A-reference stop-out level (configurable; default 0%, as Exness global) | verified entity level (0% or 20%) | per profile |
| Negative balance | no NBP (debt kept; reward ε-floored) | Exness NBP + separate no-NBP stress | per profile |
| Death | equity ≤ 5% of initial | equity ≤ 5% of initial | equity ≤ 5% of initial |
| Costs | Exness-calibrated spreads, slippage, swaps; no commission | verified Cent spreads (calibrated), commission (none), swaps | per profile |
| Data | Dukascopy M1 bid/ask 2018 →; Exness-calibrated spreads | same price path; Cent specs | live feed (source TBD), identical bar schema |
| Real orders | impossible | impossible | **impossible** (no trading client in the package; test-enforced) |

Rules: one account = one profile (no Cent/Standard mixing). A-trained agents are evaluated in B
without retraining.

---

## 2. Critique of the specification (status after r4)

| # | Issue | Resolution |
|---|---|---|
| 1 | $10 + real lots | A fractional; B Cent-only (FX tradable; gold needs ≈ $40 notional) |
| 2 | Normalized equity not scale-invariant | Ledger in USD/USC; normalization only for observations and rewards |
| 3 | Multi-agent world on replay | Price-takers; diversity via seeds, learners, rewards, evolution |
| 4 | Reward = risk preference | Log-wealth baseline; shaping variants labelled (§8) |
| 5 | Hidden rebalancing in target exposure | Documented action semantics plus a delta-action ablation (§5.6) |
| 6 | Decision frequency | 5-minute default; 1, 15, 60 configurable |
| 7 | Observation window | 120 × M5 (configurable) |
| 8 | Sessions | UTC clock + tradeable mask |
| 9 | Death vs. stop-out | 5% death = experimental criterion; stop-out and NBP separate |
| 10 | Zombie accounts | `incapacitated` tracked |
| 11 | NBP as a free option | A no NBP; B Exness NBP + no-NBP stress |
| 12 | Historical swaps | Snapshot approximation, documented |
| 13 | Success criteria | Pre-registered (§9.4) |
| 14 | Compute | Local first; hardware assessment pending |
| 15 | No real orders | No order path; test-enforced |
| 16 | Train/serve skew in C | Bar-parity test |
| 17 | Mid-only data | Bid/ask |
| 18 | Account mixing | Resolved: Cent-only B |
| 19 | HMR needs a news calendar | Configurable windows; calendar source unresolved (U6) |
| 20 | Legal entity | Modular regions; B blocked until verified |
| 21 | Fixed exposure caps | **Resolved: removed (decision 15); uncapped action space §5** |
| 22 | **New:** "uncapped" needs a margin model in A | A-reference margin schedule + unlimited-margin variant (§5.4); default needs approval (§13.2) |
| 23 | **New:** uncapped exposure makes early training death-heavy and rewards heavy-tailed | Numerical safeguards, log-scaled exposure features, ε-floored log reward, entropy/initialization documented (§5.8, §8) |

---

## 3. Expectations

Many agents will likely learn to stay flat, die from leverage, or capture drift that does not
generalize. With uncapped exposure, **early training will be dominated by deaths**. That is the
point: agents discover the consequences of leverage themselves. Under the log-wealth reward,
leverage beyond the growth-optimal level lowers expected reward through variance, so moderate
leverage *can* emerge without any rule. Synthetic experiment S7 (§10.3) tests whether it actually does.

---

## 4. Architecture

### 4.1 Component map

```
                ┌──────────────── offline ─────────────────┐
 Dukascopy M1 → │ 1 Market Data Engine (download, clean,   │ → Parquet store (immutable, hashed)
 bid/ask        │   align, validate, session masks)        │
 Exness ticks → │   Spread calibration (Exness cost model) │ → calibrated cost tables (hashed)
 Exness specs → │   Broker profile build + verification    │ → profile YAML with per-field status
                └──────────────────────────────────────────┘
            ┌──────────── deterministic core (no learning) ───────────┐
            │ 2 Replay: UTC clock, cursor, past-only windows          │
            │ 3 Execution: bid/ask fills, latency, slippage, lots,    │
            │   atomic margin check, max-volume check                 │
            │ 4 Accounts: ledger, margin model (A-reference / Exness),│
            │   swaps, stop-out, NBP, death, USD/USC/JPY conversion;  │
            │   World: one clock, N accounts, event log               │
            └─────────────────────────────────────────────────────────┘
                     ▲ actions                    │ observations
            ┌────────┴─────────────┐   ┌──────────▼───────────┐
            │ 6 Action Interface   │   │ 5 Observation Iface  │   ← only surface agents see
            │ (uncapped, §5)       │   │                      │
            └────────▲─────────────┘   └──────────┬───────────┘
            ┌────────┴────────────────────────────▼────────────┐
            │ 7 Gymnasium Env + pluggable reward               │
            └────────▲─────────────────────────────────────────┘
   8 Training (PPO → SAC, ES)    9 Evaluation (A → B transfer, leverage benchmarks, null distributions)
  10 Tracking (config, git, data, cost-table and profile hashes, seeds; per-decision exposure log)
  11 Dashboard (Phase 6)   12 Live virtual feed (Phase 5, read-only market data)
```

### 4.2 Design principles

Deterministic core (seeded RNG only); event-sourced accounting; strict isolation (own
parameters, optimizer, buffers and RNG per agent; no parameter sharing); profiles as versioned
data with per-field `value/source/retrieved/status`; simulator independent of Gymnasium and
PyTorch; float64 ledger, float32 observations; pre-aligned arrays and zero-copy windows.

### 4.3 Technology stack

Python 3.11+, NumPy (+ numba if needed), Polars/pandas → Parquet, DuckDB, Pydantic + YAML,
Gymnasium, PyTorch + Stable-Baselines3 (first PPO/SAC), own ES/GA loop, pytest + Hypothesis,
own run registry + TensorBoard, Streamlit → FastAPI.

### 4.4 Multi-agent meaning

Price-takers on a shared tape. Running agents together is identical to running them alone, which a
test enforces. Diversity comes from seeds, learners, rewards, observation and decision settings, and
population-based evolution.

---

## 5. Action space, exposure and leverage (new in r4)

### 5.1 Three kinds of limits

| Kind | Examples | Where enforced | Agent-visible? |
|---|---|---|---|
| **Behavioural cap** (an artificial limit on the agent's choice) | "max 3× exposure" | **Nowhere in the primary experiments.** Only in labelled benchmark variants (§5.9) | n/a |
| **Market / broker constraint** (what a real participant faces) | equity; margin requirements; atomic margin rejection; broker max volume per order and position; lot step (B); stop-out; NBP (B); 5% death; costs | Execution and account engines, logged with a reason | Yes: free margin, margin level, rejection flag in the observation, as on a trading terminal |
| **Numerical safeguard** (keeps computation valid) | finite action range; representational exposure range 10⁻⁴…10⁴×; NaN/Inf handling; float64 ledger; sub-cent position threshold | Action interface and engine; every activation is logged and counted | Indirectly (a rejected/invalid action is visible as "no change") |

A safeguard that binds in practice would act as a hidden cap, so it is **monitored**. If more than
a small fraction of decisions sit at the representational ceiling, the run is flagged and repeated
with a wider range (sensitivity check).

### 5.2 Action definition

Each decision the agent outputs `n + 1` numbers, squashed to [-1, 1] by the policy (tanh):

```
d ∈ [-1,1]^n     direction and allocation per instrument
u ∈ [-1,1]       gross-exposure level, log-scaled

G(u)   = 10^(4·u)                         → 0.0001× … 1× (u = 0) … 10,000× of equity
w_i    = G(u) · d_i / max(1, Σ_j |d_j|)   → signed target exposure (notional / equity) per instrument
gross  = Σ|w_i| ≤ G(u) ;  net = Σ w_i
```

Properties:
- **No behavioural ceiling:** any exposure from exactly 0 (d = 0) to 10⁴× equity is expressible,
  far above the largest margin-feasible exposure (≈ 2,000× at 1:2000), so the market, not the
  interface, decides what is possible.
- **Log scale:** each 0.25 step of `u` is one order of magnitude. Exploration covers 0.1× and 100×
  with equal ease, so agents can actually *discover* very high exposure. A linear scale up to 10⁴
  would make low exposures practically unreachable, which would be a hidden bias.
- **Neutral initialization:** a freshly initialized policy outputs u ≈ 0, i.e. ≈ 1× gross, with
  log-symmetric exploration around it. This is an initialization choice; documented and ablated
  (e.g. init at 0.1× and 10×).
- **Zero exposure, long, short, partial, reversal and multi-instrument allocation** are all
  continuous regions of the same space.
- **Exposure semantics are explicit:** `w` is a *target* fraction of current equity. The agent
  sees its current exposure, so "hold unchanged" means re-outputting the current target. This
  is documented rather than hidden. A **delta-action ablation** (actions = change in exposure,
  §5.6) tests whether the semantics shape behaviour.

### 5.3 From target exposure to units

```
target_units_i = w_i · equity / (price_i · contract_size_i · fx_to_account_i)
A: fractional; |target notional| < $0.001 ⇒ 0  (numerical threshold, logged)
B: round toward zero to lot_step; |target| < min_lot ⇒ 0 (broker rule)
delta_i = target_units_i − current_units_i
```
Fills: decision at the M5 bar close → next M1 open; buys at ask, sells at bid; adverse slippage;
closed market ⇒ rejected and logged.

### 5.4 Margin model in environment A

Uncapped exposure needs a margin model; otherwise exposure would be unbounded except numerically.

- **Default: A-reference margin schedule.** Per-instrument initial margin rates modelled on
  documented Exness retail conditions *without* Unlimited: leverage-based instruments
  (FX, gold) at 1:2000 (0.05%), and fixed-margin instruments (BTC, NAS100) at documented
  Exness-style fixed margins (placeholder values marked ❓ until verified; e.g. NAS100 0.25%).
  This is a realistic, very permissive retail regime: up to ≈ 2,000× on FX.
- **Variant: unlimited-margin.** Margin rate ≈ 0 for leverage-based instruments; exposure then
  limited only by the numerical range (10⁴×), equity, costs and death. It is labelled as an
  experiment, since no broker works this way for all instruments at all times.
- **Stop-out** in A: configurable, default 0% (Exness global). Death at 5% therefore fires first.
- Margin uses float64 notional; margin level is reported as `equity / used_margin` (∞ when flat,
  stored as a sentinel).

### 5.5 Margin rejection (atomic, order-independent)

At each decision: all **reductions** execute first (they free margin). Then all **increases**
execute together if their total initial margin fits in free margin; **otherwise all increases in
that decision are rejected** (`insufficient_margin`, as MT5 rejects with "not enough money"). There
is no partial scaling and no instrument ordering, so the engine never chooses a size for the
agent. The agent sees the rejection flag next step. A **scale-to-fit variant** (proportional
reduction) exists only as a labelled experiment, because it is a sizing rule. Broker max volume per
position (e.g. 200 lots) is applied the same way.

### 5.6 Ablations of the action space

| Variant | Definition | Question |
|---|---|---|
| X1 (primary) | factorized direction + log-gross (§5.2) | — |
| X2 | signed-log per instrument: `w_i = sign(x_i)·(10^(4|x_i|) − 1)`, range 0…≈10⁴× per instrument, exactly 0 at x_i = 0 | does coupling allocation and scale matter? |
| X3 | delta actions: `Δw_i` per decision, log-scaled | do target semantics shape holding behaviour? |
| X4 | benchmark caps (§5.9) | what does the cap itself change? |

### 5.7 Exposure and leverage recording (every decision)

Per agent, per decision step:

| Field | Meaning |
|---|---|
| `action_raw` | network outputs before squashing (for NaN diagnostics) |
| `d`, `u`, `G` | squashed actions and gross level |
| `w_target` | requested signed exposure per instrument |
| `w_executed` | exposure after fills (after rounding, rejection, closed markets) |
| `gross_leverage`, `net_leverage` | Σ|notional| / equity, Σ notional / equity, after execution |
| `max_gross_leverage_interval` | highest mark-to-market gross leverage between this decision and the next (from M1 revaluation) |
| `margin_used`, `free_margin`, `margin_level` | account state after execution |
| `rejections` | insufficient margin, max volume, closed market, below minimum, invalid action |
| `safeguards` | ceiling-proximity flag (u > 0.95), sub-threshold zeroing, NaN replacement |
| `costs` | spread, slippage, commission, swap attributed to this step |
| `equity`, `balance`, `status` | including death / stop-out / NBP events |

Stored as Parquet (float32 where safe, float64 for money), partitioned by run/agent.

### 5.8 Numerical safeguards

| Risk | Safeguard |
|---|---|
| NaN/Inf actions | action ignored (positions unchanged), `invalid_action` event; training raises after K consecutive events (bug signal) |
| Exposure overflow | representational range 10⁻⁴…10⁴×; notional and units computed in float64; asserts on finiteness |
| Network inputs from huge leverage or negative equity | observation features use signed-log transforms: `sign(x)·log1p(|x|)` for leverage, PnL/equity, margin level |
| Reward blow-ups | R1 uses `log(max(E_t, ε·E_0)/max(E_{t-1}, ε·E_0))`, ε = 0.01; optional running reward normalization |
| Negative equity (A, no NBP) | allowed in the ledger; account is dead; reward floored |
| Precision at extreme leverage | float64 ledger; golden tests at 2,000× and 10,000× |
| Silent safeguard binding | counts per run; run flagged if the ceiling binds in > 0.1% of decisions |

### 5.9 Benchmark leverage experiments (not limits)

Same action space with the upper end of the log range set to `G_max ∈ {3, 10, 50, 100}`:
`log10 G(u) = −4 + (u+1)/2 · (4 + log10 G_max)`, so u = −1 ⇒ 10⁻⁴× and u = +1 ⇒ G_max. This
rescales the range instead of clipping it, which would create a flat, gradient-free region at the cap.
They are trained and evaluated alongside uncapped agents. Reported: survival, log-growth,
drawdown, turnover, and learned leverage distribution vs. uncapped. Optional: evaluate uncapped
agents with a cap imposed at test time, to measure how much they rely on extreme exposure.

### 5.10 Observation (default)

Per instrument, the last **120 × M5 bars** (configurable), built only from completed M1 bars: mid log
return, range/mid, body/mid, spread/mid, relative tick volume, tradeable flag. Account state:
signed-log equity / initial, free-margin fraction, signed-log margin level, signed-log exposure per
instrument, signed-log gross and net leverage, unrealized PnL / equity, time in position,
last-decision rejection flags, margin-call flag (where the profile has one). Optional time
features (ablated). **Not given:** indicators, signals, absolute prices, dates, other agents,
simulator internals, cost-model parameters, the margin schedule as a parameter (agents feel it
through rejections and margin level, as traders do), future bars.

---

## 6. Learning without predefined strategies

Nothing in the engine opens, closes or resizes positions for the agent except enforced
market/broker constraints (margin rejection, max volume, stop-out, NBP, death), each logged.
There are no stop-losses, take-profits, sizing rules or leverage caps in the primary experiments.
Remaining inductive biases (feature transforms, architecture, window, decision interval, action
parametrization, initialization, reward, learner) are documented and ablated.

---

## 7. Broker specification: Exness (environment B)

### 7.0 Verification policy

Direct fetches of Exness pages return HTTP 403 here. r2/r3 facts came from search-engine extracts
of official pages, and Exness pages **contradict each other** on several points. Status per field:
✅ verified · ⚠️ conflicting · ❓ unverified/derived · ⛔ unknown. Verification sources for Phase 1
(no credentials): official pages saved or pasted by you into `docs/exness-spec-verification.md`;
official Exness tick history. Later and optional: a read-only demo snapshot via MetaApi.

### 7.1 Account-level facts (Standard Cent)

| Item | Value | Status |
|---|---|---|
| Currency | USC; $10 = 1,000 USC | ✅ |
| Lot | 1 Cent lot = 0.01 standard lot (≈ 1,000 base units) | ✅ |
| Volume | min 0.01 (ETHUSDc 0.1); max 200 lots | ✅ |
| Commission | none | ✅ |
| Spread | "from 0.3 pips" (distribution from calibration) | ✅ / ⛔ |
| Margin call | 60% | ✅ |
| Stop-out | 0%; 20% for KE and JO entities; stocks up to 100% | ✅ |
| Stop-out closing order | reportedly most unprofitable first | ❓ |
| NBP | reset to 0 only after a full stop-out with negative balance | ✅ |
| Swap timing | 21:00 / 22:00 GMT+0 (summer/winter) | ✅ / ⚠️ |
| Triple swap | FX Wed (USDCAD Thu); indices Fri; crypto Fri | ✅ |
| Swap-free | majors and most minors swap-free; may depend on behaviour | ✅ / ❓ |
| HMR window | 15 min before → 90 s or 5 min after high-impact news | ⚠️ |
| Insufficient-margin behaviour | order rejected (MT5 standard) | ❓ (U14) |

### 7.2 Instruments on Cent

| Instrument | Symbol | In B? | Smallest position | Margin type | Status |
|---|---|---|---|---|---|
| EUR/USD | EURUSDc | yes, once fields verified | ≈ €10 | leverage-based | ✅ availability |
| GBP/USD | GBPUSDc | yes, once verified | ≈ £10 | leverage-based | ✅ |
| USD/JPY | USDJPYc | yes, once verified | $10 | leverage-based | ✅ |
| XAU/USD | XAUUSDc | yes, once verified | ≈ 0.01 oz (≈ $40) | leverage-based + HMR | ✅ / contract ❓ |
| BTC/USD | BTCUSDc | **excluded until verified** | ❓ | fixed | ⚠️ |
| NAS100 | USTECc | **excluded until verified** | ⛔ | fixed | ⚠️ |

### 7.3 Leverage and margin in B (decision 15, point 8)

B uses only verified rules. **Unlimited is never assumed because the balance is small.**

- **Account leverage setting** is an experiment parameter, chosen only from values **verified as
  available** to the verified entity: candidates 1:200, 1:400, 1:1000, 1:2000, 1:Unlimited.
- **1:Unlimited** is enabled only if (a) its eligibility conditions are verified (U5: equity
  threshold, which sources give as $5,000 or ≈ $1,000, plus trading-history requirements), and
  (b) the simulated account meets them **at that moment**. The simulator tracks the conditions
  (equity, and history such as trade count/volume if those are the verified criteria). A fresh $10
  account that lacks the history therefore does **not** get Unlimited.
- **Equity tiers** (✅): < $5k 1:2000 (Unlimited if eligible); $5k–30k 1:2000; $30k–100k 1:1000;
  ≥ $100k 1:500, applied as the verified rule specifies (how tier changes affect open positions is U15).
- **Fixed-margin instruments** (crypto, indices) ignore account leverage (✅). **HMR** windows raise
  margins (✅ existence, ⚠️ details). **Entity caps** apply (e.g. KE 1:400, ✅).
- Until U1/U5 are resolved, B runs are **sensitivity-tagged** across candidate leverage settings and
  regions. No B result is reported as "the" Stage D result.

### 7.4 $10 account in B: what is feasible

| Instrument | Smallest exposure step at $10 | Max exposure, margin-limited (1:2000 example) |
|---|---|---|
| EURUSDc | ≈ 1.16× equity | ≈ 2,000× (≈ 1,700 minimum units) |
| GBPUSDc | ≈ 1.3× | ≈ 2,000× |
| USDJPYc | 1.0× | ≈ 2,000× |
| XAUUSDc | ≈ 4× | per verified gold margin / HMR band |

**Resolution is coarse at low exposure** (no 0.5× EUR/USD possible at $10) and fine at high
exposure. Costs scale with exposure: a 1-pip EUR/USD spread is ≈ 0.9 bps of notional, i.e. ≈ 0.9%
of equity per round trip at 100× and ≈ 18% at 2,000×. Agents will feel this directly. Pip values are
golden-tested (USC/USD factor-of-100 risk).

### 7.5 Regional configuration (modular until verified)

```
configs/brokers/exness/
  account_types/standard_cent.yaml
  instruments/{EURUSDc,...}.yaml
  leverage/{tiers.yaml, unlimited_eligibility.yaml, hmr.yaml}
  regions/{global.yaml, ke.yaml, jo.yaml, unverified.yaml (ACTIVE DEFAULT)}
```
B profile = account type ⊕ region ⊕ leverage setting ⊕ instruments. The legal entity is not
assumed (decision 9).

### 7.6 Verified-spec guard

B evaluations fail to load if any required field is not ✅ (the error lists the fields). The
`--sensitivity` flag runs over candidate values and tags results `unverified-spec`.

### 7.7 Spread and cost model

Dukascopy bid/ask M1 is the price path. The Exness spread distribution is calibrated from official
Exness tick history by hour-of-week and volatility regime, and replaces the Dukascopy spread around
the Dukascopy mid. Calibration is validated on held-out Exness months. Swaps come from a dated
snapshot (approximation).

### 7.8 Death, stop-out, NBP: order of evaluation (A and B)

At every M1 step: (1) revalue at intra-bar adverse extremes (conservative, documented);
(2) **death check**: equity ≤ 5% of initial ⇒ close all, terminate (`equity_floor`);
(3) **broker stop-out** at the profile's level (reachable before death only if the stop-out level
is above 5% equity-equivalent, or on gaps) ⇒ liquidation per verified order (`stop_out` /
`gap_through`); (4) **NBP** (B profile): negative balance after a full stop-out ⇒ reset to 0;
no-NBP profiles keep the debt. At extreme leverage, death can occur within one M1 bar. The fill
then happens at the trigger price + slippage, or at the bar open if it gapped through.

---

## 8. Rewards and learners

### 8.1 Reward variants (net-of-cost equity)

| ID | Reward | Expected behaviour with uncapped exposure |
|---|---|---|
| R1 baseline | ε-floored log-wealth growth | growth-optimal: leverage above the log-optimal level lowers expected reward (variance drag ≈ −G²σ²/2), so high leverage is discouraged by outcomes, not rules |
| R2 negative control | linear equity change / E₀ | risk-neutral: expected to push exposure towards the margin limit and die |
| R3 | R1 − λ·Δdrawdown | labelled risk preference |
| R4 | differential Sharpe | leverage-invariant in theory; gameable by tiny variance |
| R5 | sparse terminal log-wealth | hard credit assignment |

Never: trade-count bonuses, win-rate rewards, market-presence rewards, realized-only rewards.

### 8.2 Reward-hacking closures

Bid/ask fills and tradeable mask; rounding toward zero (B); atomic margin rejection (no
engine-chosen sizes); no NBP in A; one continuous life in evaluation; gap valuation; no absolute
levels or dates; same out-of-sample metrics for every variant; cost robustness at 1.5× and 2×;
safeguard-binding monitor (§5.8), so the numerical ceiling cannot quietly become the strategy.

### 8.3 Learners and baselines

PPO first (tanh-squashed policy), then SAC and ES/GA. Entropy and initialization settings are
recorded, because uncapped exposure makes exploration consequential. Baselines (flat, uniform
random, equal-weight buy-and-hold at 1×, turnover-and-leverage-matched random) form the null
distribution.

### 8.4 Environment C leverage experiments

C runs an agent under any profile: A-uncapped, A-benchmark caps, or B with any verified leverage
setting. Several such virtual accounts can follow the same agent in parallel (e.g. one agent at
1:2000 vs. 1:400), all on simulated execution.

### 8.5 A → B transfer

Evaluation without retraining. Reported: gap vs. A, share of targets rounded to zero, rejection
rates, turnover and leverage changes, and the effect of excluded instruments.

---

## 9. Leakage and overfitting

### 9.1 Splits

| Split | Period | Use |
|---|---|---|
| Train | 2018-01 → 2022-12 | Stage A |
| Embargo | 2 weeks | |
| Validation | 2023-01 → 2023-12 | hyperparameters, selection |
| Embargo | 2 weeks | |
| Test | 2024-01 → 2025-06 | unseen history (A, B); milestone-only, access logged |
| Final holdout | 2025-07 → present | once, before C |
| Exness cost calibration | Exness tick months (oldest available preferred) | spread statistics only |

### 9.2 Leakage controls
Bars closed at or before decision time only; future-perturbation test; window-local normalization;
M5 only from completed M1; physically separate test files.

### 9.3 Overfitting controls
Random episode starts; validation-only selection; drift-adjusted reporting; deflated Sharpe, null
distributions, block bootstrap; cost robustness; **leverage robustness** (results re-evaluated under
alternative margin settings).

### 9.4 Pre-registered success criteria (test split, all required)
1. Net log-growth > 0, 95% block-bootstrap CI excluding 0.
2. Above the 95th percentile of turnover-and-leverage-matched random agents.
3. Not explained by buy-and-hold drift.
4. Positive at 1.5× costs.
5. Consistent across ≥ 2 non-overlapping sub-periods.
6. Reported separately in B (verified profile only).

---

## 10. Validation plan

### 10.1 Data-quality checks
Monotonic unique UTC timestamps; bid ≤ ask; OHLC consistency; gap classification; DST; spread
outliers; Dukascopy-vs-Exness divergence.

### 10.2 The 15 simulator test groups (Phase 1 exit gate)

1. **No look-ahead:** future-perturbation test; M5 from completed M1 only.
2. **Fills:** ask/bid at the next M1 open; adverse slippage; exact commissions.
3. **Accounting invariants** (Hypothesis, including random actions across the full 10⁻⁴…10⁴ range).
4. **Golden scenarios:** per instrument, USD/JPY → USD/USC, USC factor-of-100, XAUUSDc pip value,
   **extreme-leverage trades at 2,000× and 10,000×** (precision).
5. **Action mapping:** G(u) exact; d = 0 ⇒ flat; gross ≤ G; reversal; B rounding toward zero; min
   lot; NaN/Inf handling; ceiling-proximity flag; benchmark-cap mapping.
6. **Margin:** A-reference schedule and unlimited variant; Exness leverage-based vs. fixed margin;
   equity tiers; Unlimited eligibility (granted only when verified conditions hold); HMR; **atomic
   rejection** (all increases rejected, reductions executed, order-independent); max volume.
7. **Death and stop-out:** death at 5%; stop-out at the profile level; gap-through; intra-bar death
   at extreme leverage; no actions after death.
8. **NBP:** B reset rule; no-NBP debt; A negative equity with ε-floored reward.
9. **Swaps:** across rollover only; triple days; swap-free variant.
10. **Sessions:** closed-market rejection; daily breaks; 24/7 BTC; weekend gaps.
11. **Determinism:** identical event-log hash for the same seed.
12. **Isolation:** agent results identical alone vs. with others.
13. **Baselines:** analytic checks for flat, buy-and-hold and random agents.
14. **No order path / profile integrity:** no trading client; no Cent/Standard mixing; verified-spec
    guard; **no behavioural cap active in primary configs** (config test).
15. **Performance benchmark** + **recording completeness** (every decision row has all §5.7 fields).

### 10.3 Synthetic-market learning experiments (Phase 2 gate)

| Exp. | Synthetic market | Expected outcome |
|---|---|---|
| S1 | AR(1) mean-reverting returns with costs | profitable policy; beats the null distribution |
| S2 | sign-switching drift with a visible regime flag | follows the regime |
| S3 | random walk with costs | converges to ≈ flat |
| S4 | random walk, no costs | no systematic profit |
| S5 | leakage canary (future returns in a field the agent cannot see) | no profit; profit ⇒ leak |
| S6 | positive drift with fat-tailed crashes, uncapped | R1 survives with moderate exposure; R2 drives to high leverage and dies |
| **S7** | **GBM with known drift μ and volatility σ, no costs** | **R1 agent's learned leverage approaches the growth-optimal μ/σ² (within tolerance); R2 does not.** This tests whether agents can discover sensible leverage without any rule |

Plus Gymnasium `check_env`.

---

## 11. Directory structure

```
trading-world/
├── pyproject.toml
├── README.md
├── docs/  ARCHITECTURE.md · simulator-spec.md · exness-spec-verification.md · decisions/
├── configs/
│   ├── environments/{A_fractional, B_exness_cent, C_live_virtual}.yaml
│   ├── margin/{a_reference.yaml, a_unlimited.yaml}
│   ├── instruments/
│   ├── brokers/{fractional.yaml, exness/{account_types, instruments, leverage, regions}/}
│   └── experiments/  (primary uncapped; benchmark caps 3/10/50/100; ablations X2–X3)
├── src/tradeworld/
│   ├── data/  calibration/  brokerspec/
│   ├── replay/  market/  execution/  accounts/  world/
│   ├── interface/      # observation builder, uncapped action mapper, safeguards
│   ├── envs/  rewards/  agents/  training/
│   ├── evaluation/  tracking/  live/  dashboard/
├── tests/  unit/ property/ golden/ integration/ synthetic/ benchmarks/
├── scripts/  hardware_check, download_dukascopy, import_exness_ticks, calibrate_costs, run_world
├── data/  runs/   (gitignored)
```

`btc-trading-bot/` remains untouched and separate (decision 12).

---

## 12. Metrics per agent

Spec §11 metrics, plus log-growth, Sharpe/Sortino, deflated Sharpe, drawdown duration, turnover,
**leverage profile** (distribution of gross/net leverage, time at > 10×/100×/1,000×, leverage
before death, leverage response after losses and gains), cost and swap share, time in market,
long/short bias, allocation entropy, holding times, rejection counts, safeguard activations,
stop-out/NBP counts, time to death, incapacitated duration, rounded-to-zero rate (B). Behaviour
clustering includes leverage features.

---

## 13. Decisions

### 13.1 Recorded

| # | Date | Decision |
|---|---|---|
| 1 | 2026-11-15 | Fractional sizing during training, realistic costs |
| 2 | 2026-11-15 | Exness specs for Stage D (refined by 8) |
| 3 | 2026-11-15 | Verified Exness margin and leverage, configurable per instrument and account type |
| 4 | 2026-11-15 | Exness NBP in the broker profile + no-NBP stress |
| 5 | 2026-11-15 | Dukascopy M1 bid/ask from 2018; Exness data for calibration |
| 6 | 2026-11-15 | 5-minute decisions; 1/15/60 configurable |
| 7 | 2026-11-15 | Local development first; hardware assessment before long runs |
| 8 | 2026-11-15 | Stage D = Exness Cent-only; no mixing; only verified instruments; A keeps all six |
| 9 | 2026-11-15 | Legal entity not assumed; modular regional config |
| 10 | 2026-11-15 | MetaApi deferred; Phase 1 needs no credentials |
| 11 | 2026-11-15 | Observation 120 × M5, configurable |
| 12 | 2026-11-15 | Keep `btc-trading-bot/` unmodified |
| 13 | 2026-11-15 | Environments A/B/C; 5% death is the survival criterion; stop-out/NBP separate; C never executes real orders |
| 14 | 2026-11-15 | ~~Unlimited leverage (r3 interpretation: account setting only)~~, **superseded by 15** |
| 15 | 2026-11-15 | **No fixed exposure ceiling.** A: agents discover exposure themselves; no hidden sizing or risk rules; equity, costs, margin and liquidation preserved; numerical safeguards; exposure and leverage recorded every decision; 3×/10×/50×/100× are benchmark experiments only. B: only verified Exness Cent leverage and margin rules; Unlimited not assumed from a small balance. C: configurable leverage experiments |

### 13.2 Needs your approval

1. **Action space X1 (§5.2):** factorized direction + log-scaled gross, range 10⁻⁴…10⁴×, init ≈ 1×.
2. **A margin default (§5.4):** A-reference schedule (1:2000 for FX/gold, fixed margins for
   BTC/NAS100), with unlimited-margin as a variant. Or should the unlimited-margin variant be the
   primary?
3. **Margin rejection (§5.5):** atomic rejection of all increases (recommended) vs. proportional
   scale-to-fit.
4. **Legal entity:** supply it once verified.

---

## 14. Unresolved specifications (block B results, not Phase 1)

| # | Item | Resolve via |
|---|---|---|
| U1 | Legal entity → stop-out, leverage options, restrictions | you |
| U2 | BTCUSDc availability and contract | official page copy; tick-history symbol list |
| U3 | USTECc availability and contract | same |
| U4 | XAUUSDc contract and pip value | official spec; tick digits |
| U5 | Unlimited eligibility conditions and margin formula | official Leverage page copy |
| U6 | HMR windows, bands, historical news calendar | official HMR page; choose calendar source |
| U7 | Stop-out closing order | official page / support |
| U8 | Swap values, swap-free eligibility | official tables (snapshot) |
| U9 | Cent symbols in tick history vs. Standard spreads | tick-history symbol list |
| U10 | Rollover time and server time zone | official page; tick timestamps |
| U11 | Sessions and daily breaks | official pages; tick gaps |
| U12 | Live price source for C | Phase 5 |
| U13 | Dukascopy BTC history start | on download |
| U14 | Exness behaviour on insufficient margin (reject vs. partial) | official page / later demo |
| U15 | How equity-tier leverage changes apply to open positions | official Leverage page |
| U16 | A-reference fixed margins for BTC and NAS100 | official Cryptocurrencies/Indices pages |

Assumptions until resolved: Cent spreads ≈ Standard spreads; swaps from a current snapshot;
conservative intra-bar evaluation; MT5-style rejection on insufficient margin.

---

## 15. Milestone 1 (Phase 0 + Phase 1), no credentials required

| Step | Deliverable | Exit criteria |
|---|---|---|
| 1.0 | ADRs; `simulator-spec.md`; environment, margin and experiment configs | approved |
| 1.0a | Hardware assessment (§16) | resource plan agreed |
| 1.1 | Profile system (fractional, A-reference margin, modular Exness), verified-spec guard | unit + guard tests |
| 1.2 | `accounts/` + `execution/` + uncapped `interface/` actions on synthetic prices | groups 2–9, 11, 12, 14 green |
| 1.3 | Dukascopy pipeline (local) + quality report | §10.1 green |
| 1.4 | Exness tick import + spread calibration (EURUSD, GBPUSD, USDJPY, XAUUSD first) | calibration validation |
| 1.5 | `replay/` + observation (120 × M5) + `world/` on real data | groups 1, 10, 15 green |
| 1.6 | Baselines in A (uncapped and benchmark caps) and B (`unverified`, sensitivity-tagged) | group 13 green; report reviewed |

Phase 2 (Gym env, S1–S7, first PPO agent) starts after sign-off.

---

## 16. Hardware assessment (step 1.0a)

Run on the training machine and paste the output, plus RAM size and GPU model/VRAM:

```bash
python -c "import os,platform,shutil;print(platform.platform());print('cpu',platform.processor(),os.cpu_count(),'threads');print('disk free GB',round(shutil.disk_usage('.').free/1e9))"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

| Resource | Minimum | Comfortable | Driver |
|---|---|---|---|
| Disk | 10 GB | 50 GB+ | raw Dukascopy files, Parquet (~0.5–1 GB), Exness ticks, per-decision logs |
| RAM | 8 GB | 16–32 GB | aligned arrays ≈ 1 GB + parallel envs |
| CPU | 4 cores | 8+ cores | CPU-bound simulator |
| GPU | none | CUDA ≥ 6 GB | larger encoders, many agents |

Note: uncapped training produces many short (death-terminated) episodes, and per-decision
logging (§5.7) adds roughly 1–2 GB per 100 M decision rows (compressed). Log retention per run is
configurable.

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
