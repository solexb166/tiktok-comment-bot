# Proposal 0007: Investigation of the reward system (for approval)

Status: **proposal only, 2026-11-18.** The reward functions, code, agents, configurations and records were not changed and
no training was run. All numbers below come from exact calculation and offline simulation in
`docs/proposals/0007-reward-analysis/` (script `analysis.py`; results `exact.json`, `values.json`, `hack.json`,
`learn.json`, `learn_long.json`). The script only reads the market definition; it uses no agent and writes no record.

---

## 1. Executive summary

The current growth reward R1 has the right *global* optimum: of all constant position sizes in S6, the growth-optimal
2.93× scores best. Its *landscape*, however, contains a second, inferior hill beyond the ruin line. Above 9.9×, holding
a bigger position scores **better**, up to about 55×.

**The cause is one specific line of the reward.** R1 floors equity at 1% of the starting equity before taking the log,
so every loss deeper than 99% of the account counts as the same −4.6. Above 9.9×, a crash costs the same whatever the
size, while every normal bar pays more for more size up to the no-crash optimum μ/σ² = 55.6×.

**Environment A has no negative-balance protection**, so the losses the floor hides are real: a 55× position in a 10%
crash leaves a debt of 4.5 times the account. R1 therefore rewards a limited liability that the simulated account does
not have.

**Larger death penalties and longer horizons do not remove the hill.** A larger death penalty (ε = 10⁻⁶), γ = 0.999
and an undiscounted 1000-step objective all keep the uphill slope from 10× to 55×; they only lower the whole hill.
Longer horizons even make "gambling to recover" slightly better than optimal sizing under R1.

**A minimal correction removes the hill: full-liability log reward FL(c).** For any one-bar multiplier M ≥ c of the
account's equity, the reward is the log return ln M, exactly as R1. For deeper losses, including losses beyond the
account (M < 0, which environment A records as debt), it uses the tangent line of ln at c.

- With c = 0.01 (R1's own floor) it is identical to R1 on every bar that does not end the account. There is no new
  penalty, no exposure limit and no new risk parameter.
- Its expected-reward landscape has a single peak, at 2.93×, and falls steeply beyond the ruin line.
- In a learning model it lets agents that start inside the ruinous basin walk back out (0% ruinous states in every
  seed, against 56–81% under R1).
- From safe starts it learns exactly like R1, so it adds no conservatism.

**Recommendation:** a pre-registered experiment that trains new agents with FL(0.01) (primary) and FL(0.05)
(sensitivity) against a concurrent R1 control:

- otherwise identical settings;
- six paired seeds per group;
- the already-approved success criteria W1–W7 and E;
- added checks for timidity, gambling-to-recover, stability and out-of-distribution behaviour;
- an S7 no-crash sanity check.

Nothing is implemented until you approve.

## 2. Evidence base

| Source | Ruinous decisions (growth agents, S6) | Behaviour |
|---|---|---|
| Phase 2 (R1, seeds 0–2) | 10.9%, 0.3% (timid, 0.17×), 13.2% | spikes to 17–21× from a ~2× base; deaths 40–45% of test runs for seeds 0 and 2 |
| Phase 2 (R2 linear, seeds 0–2) | 3.1%, 3.6%, 11.5% | the "reckless" control was *less* ruinous than R1 in 2 of 3 seeds |
| Phase 2.5 (continued training) | 9.9%, 0.2%, 15.8% | ruinous sizes drift **upward** with more training (seed 2: median ruinous size 17× → 22×; control 19× → 29×) |
| Exp 1 control (6 seeds) | 2.3–24.7% (mean 11.1%) | ruinous share rises during training: first 20% → last 20% of training 0.1–4.4% → 3–30% |
| Exp 1 A (γ 0.999) | 0.5–7.1% (mean 3.9%) | ruin cut by about two-thirds, but timid (median 0.18–0.85×) |
| Exp 1 D (steadier updates) | 6.4–22.9% | the drift into ruin is slower but the same |
| Exp 1 C1/C2 | 0.0% | safe only because they explored ~×1.05 per sd (Exp 1b) |
| Exp 1b C1m / C2-up / C2-tail | 5.6% / 20.8% / 27.6% (all 18 seeds ≥ 1%) | start safe (0–0.1% at 500k steps), then sizes and ruin rise steadily; C2 agents grow through 2.93× into the ruinous zone |

Across all of these:

- **The consequence is immediate.** 99% of deaths happened on the bar right after the fatal decision (Phase 2.5
  analysis).
- **Agents did not learn from deaths.** They died 100–1,700 times each in training, and their ruinous share at the end
  of training equals their test share.

## 3. The current reward, exactly

`rewards.py`: R1 at each bar is r = ln max(E₁, εE₀) − ln max(E₀′, εE₀), with ε = 0.01.

- E₀′ and E₁ are equity before and after the bar; E₀ is the initial equity.
- The episode ends when equity falls to 5% of E₀ or below (the death rule), or after 1,000 decisions; truncation is
  bootstrapped by PPO.
- Discount γ = 0.99.
- Rewards telescope, so an undiscounted episode returns ln max(E_T, 0.01 E₀)/E₀: the log of final wealth, floored at
  1% of the start.

S6 (synthetic_tasks): each bar, with probability p = 1/300 a crash of J = 10%; otherwise a log return m + σZ with
μ = 5×10⁻⁴, σ = 3×10⁻³.

A position of G× equity multiplies equity by M = 1 + G(eˣ − 1). For G ≥ 1/J = 10, a crash gives M = 1 − GJ ≤ 0. The
synthetic profile has `nbp = False`, so the account keeps the negative balance as debt (`account.py`, event
`negative_balance`).

## 4. Why 10× scores worse than 55× (questions 1 and 2)

### 4.1 The mechanism

The expected reward of one decision, from equity E₀′ = E₀, is

  ρ(G) = (1 − p)·g_n(G) + p·max(ln(1 − GJ), ln ε),  where g_n(G) = E[ln(1 + G(e^{m+σZ} − 1))] ≈ Gμ − G²σ²/2.

1. **Normal bars.** g_n is the growth on bars without a crash. It increases with size up to the no-crash optimum
   G = μ/σ² = **55.6×**.
2. **Crash bars.** The true log loss ln(1 − GJ) falls to −∞ as G → 10. R1 replaces it by ln ε = −4.605 for every
   G ≥ (1 − ε)/J = **9.9×**.
3. **Above 9.9×, size costs nothing extra in a crash.** The crash term is constant, so ρ′(G) = (1 − p)(μ − Gσ²) > 0
   for 9.9 < G < 55.6. In plain words: **once a crash wipes the account out anyway, the reward no longer cares how far
   past the wipe-out the position was, and every other bar pays more for more size.**
4. **The valley at the ruin line.** Below 9.9×, ρ follows true Kelly growth and falls steeply towards the ruin line.
   This creates a valley at about 9.9–10× and a second hill whose top is at 55.6×.

Exact values (Gauss–Hermite, 10⁻⁴ per decision; `exact.json`):

| G | 2.93 | 9.5 | 9.9 | 10 | 20 | 55.6 | 100 |
|---|---|---|---|---|---|---|---|
| true Kelly g(G) | +2.65 | −56.6 | −108.6 | −∞ | −∞ | −∞ | −∞ |
| **R1** ρ(G) | +2.65 | −56.6 | −108.6 | −108.2 | −72.0 | **−17.1** | −151.0 |
| R1 with ε = 10⁻⁶ | +2.65 | −56.6 | −108.6 | −415.3 | −379.0 | −324.1 | −475.5 |

### 4.2 With the death rule, wealth paths and discounting

Monte Carlo values of constant policies (4,000 episodes, common random numbers; `values.json`):

| G | 2.93 | 6 | 9.5 | 10 | 15 | 20 | 30 | 55 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| **R1, γ 0.99** | **+0.029** | −0.018 | −0.458 | −0.872 | −0.761 | −0.662 | −0.504 | **−0.342** | −1.050 |

V(10×) = −0.87 and V(55×) = −0.34: holding 10× scores worse than holding 55×. The global maximum is still 2.93×.

## 5. Interaction of the reward, the 5% death rule and sizing (question 3)

1. **Death ends the episode, so ruin costs a one-off, bounded amount.** The reward never sees the true permanent
   consequence of ruin, which for log wealth is −∞. In addition, the floor sits at 1% of *initial* equity while death
   happens at 5%, so every loss between −95% and the debt is cut to the same number.
2. **The floor hides real debt.** Environment A records the negative balance. R1 counts a 55× crash (debt 4.5× the
   account) as the same −4.6 as a 10× crash (debt 0).
3. **Gambling for resurrection is free.** The death cost is ln(0.01 E₀ / E), so the poorer the account, the cheaper
   ruin becomes.
   - Under R1, "55× whenever equity is below half the start" scores +0.028, the same as optimal sizing (+0.028).
   - With γ 0.999 or no discounting it scores better than optimal: +0.170 vs +0.167, and +0.268 vs +0.263.
4. **On-policy, death looks cheap.** With γ = 0.99, the future of even the optimal policy is worth only about 0.03, so
   the opportunity cost of dying is tiny. For an agent already sizing ruinously, its own expected future is negative,
   so ending the episode looks *harmless* to its critic.
5. **The fatal step is visible but bounded.** The consequence arrives on the very next bar (§2), so this is not a
   credit-assignment problem. The signal arrives, but it is capped, and a single bar carries it. PPO limits how far any
   one sample can move the policy, and crashes are 1 bar in 300.

## 6. Does R1 measure sustainable growth? (question 4)

**Globally, yes.** The best constant policy under R1 is the growth-optimal 2.93×, at every horizon tested:

| Objective | V(2.93×) | Best beyond the ruin line (V at 55×) |
|---|---|---|
| γ 0.99 | +0.029 | −0.34 |
| γ 0.999 | +0.262 | −2.77 |
| undiscounted 1000 steps | +0.265 | −3.83 |

**Locally, no.** Once a decision is beyond the ruin line, R1 rewards making it larger. Ruinous spikes are also cheap:

- "2.93× with a 5% chance of 55× each bar" keeps **45% of the optimal value** (+0.013 vs +0.028) while dying in 16.7%
  of episodes (`hack.json`).
- This matches what the agents actually did: a sensible base size with isolated ruinous spikes that drift upward with
  training.

So R1 unintentionally rewards pathological risk-taking *conditional on already being, or briefly going, beyond the
ruin line*. This is also an inaccurate description of the simulated account, because the account is not protected
against negative balances.

## 7. Candidate reward formulations (question 5)

Notation:
- M = E₁/E₀′ is the one-bar equity multiplier (it can be negative);
- the death rule (5% of E₀) and episode handling are unchanged in every candidate.

### C0. Current R1 (reference)

| | |
|---|---|
| Definition | r = ln max(E₁, εE₀) − ln max(E₀′, εE₀), ε = 0.01, γ 0.99 |
| Expected behaviour | finds 2.93× if it never crosses 9.9×; otherwise climbs towards 55× |
| Advantages | equals log growth on every non-ruinous bar; bounded rewards; global optimum correct |
| Disadvantages | limited-liability hill beyond 9.9×; free gambling to recover; cheap spikes |
| Failure modes | the drift into ruin observed in Phase 2, 2.5, Exp 1 and Exp 1b |
| Exploitation | sizes beyond ruin, spikes, gambling to recover (§5, §6) |

### C1. Larger death cost (R1 with ε = 10⁻⁶)

| | |
|---|---|
| Definition | R1 with ε = 10⁻⁶ (ruin costs about −13.8 instead of −4.6) |
| Expected behaviour | more cautious near the line; **the hill beyond 10× is unchanged** (ρ′ is the same, the hill only drops by a constant); 55× is still better than 10× (V −2.60 vs −3.14) |
| Advantages | trivial change |
| Disadvantages | a constant death penalty: the size of the overshoot is still free; a new arbitrary constant |
| Failure modes | agents inside the basin still climb: in the learning model, 31–50% of states stay ruinous after starting at 20× |
| Exploitation | spikes still cheaper than their true cost; gambling to recover +0.022 vs +0.028 optimal |
| Verdict | **not recommended**: precisely the "larger death penalty" that does not address the mechanism |

### C2. Longer horizon (γ 0.999, or undiscounted episodes; R5 is the sparse form)

| | |
|---|---|
| Definition | R1 with γ = 0.999, or γ = 1 over 1000-step episodes (R5: one terminal reward ln max(E_T, εE₀)/E₀) |
| Expected behaviour | the gap between 2.93× and the beyond-ruin hill grows about 10-fold; the slope beyond 10× remains (V −3.28 at 10× vs −2.77 at 55×) |
| Advantages | closer to long-run (time-average) growth; Exp 1 A cut ruin by about two-thirds |
| Disadvantages | 10× noisier value targets; slower learning; Exp 1 A became timid (median 0.18–0.85×) |
| Failure modes | learning model: more states trapped beyond ruin after long training than with γ 0.99 (0–25% vs 0%); timidity |
| Exploitation | gambling to recover becomes **better** than optimal sizing (+0.170 vs +0.167; undiscounted +0.268 vs +0.263) |
| Verdict | **not recommended alone**; possibly later, combined with C3 |

### C3. Full-liability log reward FL(c): recommended

**Definition:** r = ln M if M ≥ c; otherwise r = ln c + (M − c)/c (the tangent line of ln at c, defined for M ≤ 0).
γ, the death rule and the episode handling are as in R1. Two settings:

- **c = 0.01, R1's own floor.** Identical to R1 on every bar where the account loses less than 99%, i.e. on every bar
  that does not end it. This setting introduces no new constant.
- **c = 0.05, a 95% one-bar loss.** A gentler slope (20 instead of 100 per unit of equity lost), as a sensitivity check.

**Why it is not an added penalty:**
- It does not charge for dying: the 5% death rule still simply ends the episode.
- It does not cap or restrict any position, and it does not change the reward for any loss smaller than 1 − c.
- It removes R1's truncation, so that losing more money costs more. Below c it uses the tangent line, which is gentler
  than the true log (−∞). So FL sits between R1 (truncated) and true log utility.

**Expected-reward landscape** (10⁻⁴ per decision; `exact.json`):

| G | 2.93 | 9.9 | 10 | 20 | 55.6 |
|---|---|---|---|---|---|
| FL(0.01) | +2.65 | −108.6 | −141.6 | −3,439 | −15,250 |
| FL(0.05) | +2.65 | −81.7 | −87.9 | −718 | −3,037 |

The landscape is unimodal: there is one peak, at 2.93×, and every size beyond it scores lower. Monte Carlo values
(γ 0.99) for FL(0.05) fall monotonically: 2.93× +0.029, 10× −0.64, 20× −5.3, 55× −22.0.

| | |
|---|---|
| Expected behaviour | same incentives as R1 everywhere below the ruin line; above it, every extra unit of size costs J/c (2 per unit of G for c = 0.05; 10 for c = 0.01) per crash |
| Advantages | removes the mechanism itself; matches environment A's real P&L (debt counts as loss); scale-invariant (per-bar multiplier); no exposure limit, no style |
| Disadvantages | very large negative rewards on rare ruin events (−94 for a 55× crash with c = 0.05; −456 with c = 0.01) may destabilise the critic; for an account *with* negative-balance protection it charges a debt the trader would not owe (a design decision for you, §11) |
| Failure modes | timidity if the critic over-weights rare large losses (not seen in the learning model: same sizes as R1 from safe starts); still needs exploration to find 2.93× |
| Exploitation | none found among the probes (table in §9); every pathological policy now scores below optimal sizing |

**An absolute-wealth variant was tested and rejected.** Using log utility of absolute equity with a tangent below 5%
of E₀ makes the charge grow with the account's wealth: wealthy survivors that crash produce rewards of −10⁸. The
per-bar form above fixes this.

### C4. Linear reward R2 (historical control)

| | |
|---|---|
| Definition | r = (E₁ − E₀′)/E₀ |
| Expected behaviour | risk-neutral: expected reward rises without limit with size (+1.67×10⁻⁴·G per bar) |
| Advantages | full liability (debt counts); bounded variance at small sizes |
| Disadvantages | no preference for survival or long-run growth |
| Exploitation | "55× until equity doubles, then flat" scores 5× more than optimal sizing (+0.263 vs +0.052); spikes beat optimal sizing |
| Verdict | not recommended. Its lower ruin than R1 in Phase 2 (2 of 3 seeds) is consistent with full liability helping, but its objective is wrong |

### C5. Risk-penalised objectives (R3 drawdown penalty, CVaR, variance penalties)

These add a hand-chosen risk preference (λ), which is the "manually imposed risk restriction" you asked to avoid. They
also change the optimum away from growth-optimal sizing. **Not recommended.**

### C6. Sharpe-type rewards (R4 differential Sharpe)

The Sharpe ratio of G·r equals the Sharpe ratio of r for every G > 0. These rewards are therefore **blind to position
size** and cannot teach sizing at all. **Not recommended** for this problem.

### C7. Model-based expected-growth reward (reward = g(G) from the known market, i.e. the Q2 score)

It gives a perfect landscape, but it uses information no real agent has (the true crash probability and size). The
agent would be told the answer, which is not autonomous learning. **Diagnostic upper bound only.**

### C8. Empirical-distribution ("counterfactual over history") reward

The reward would be the average log growth of the chosen size over all past bar returns. It removes rare-event noise,
but it ignores the agent's current information, so it would **penalise correct state-dependent bets** in markets with
predictable structure. **Not recommended.**

### C9. Average-reward formulation with restarts

The long-run growth rate with an account reset after ruin reproduces the same hill: each ruin costs the account, not
more for a larger overshoot. **Not a fix on its own.**

## 8. Learning from catastrophe without excessive conservatism (question 6)

Why agents currently do not learn from their deaths:

- **The reward caps the information** (§4): a crash at 15× and at 55× look the same.
- **Crashes are rare** (1 bar in 300), and PPO's clipping limits how much one sample can change the policy.
- **The critic cannot predict crashes** (explained variance about 0), and, on-policy, dying looks cheap to a reckless
  agent (§5.4).

What would help, in order of directness:

1. **Remove the cap (C3).** Each crash then carries a loss proportional to the overshoot, so the same event teaches
   "smaller is better" all the way down to the ruin line. Below the line nothing changes, so there is no added pressure
   towards timidity. The learning model confirms this: from 1× starts FL agents reach the same sizes and Q2 as R1
   (median 1.1–1.5×, Q2 +0.67).
2. **Counterfactual (full-information) learning, an algorithm change outside this proposal's scope.** The realised bar
   return determines exactly what any other size would have earned on that bar. A learner can use the expected reward
   over its own action distribution for the immediate term, so every crash informs every size at once. This sharply
   reduces the rare-event noise without changing the reward. It would need its own proposal. Note that under R1 it would
   *not* help beyond the ruin line, where the reward's slope is zero; it complements C3.
3. **Keep exploration matched to the control (Exp 1b).** Agents can only learn from consequences they experience.
   Reduced exploration produced the safe-but-timid agents of Experiment 1.

## 9. Offline tests (question 7)

### 9.1 Value of constant policies (Monte Carlo, γ as stated; `values.json`)

| G | R1 | R1 ε 10⁻⁶ | R1 γ 0.999 | R1 undiscounted | **FL(0.01)** | **FL(0.05)** | R2 linear |
|---|---|---|---|---|---|---|---|
| 2.93 | **+0.029** | **+0.029** | **+0.262** | **+0.265** | **+0.029** | **+0.029** | +0.053 |
| 9.5 | −0.458 | −0.490 | −2.226 | −2.830 | −0.490 | −0.490 | +0.196 |
| 10 | −0.872 | −3.138 | −3.281 | −4.229 | −1.033 | −0.637 | +0.209 |
| 20 | −0.662 ↑ | −2.929 ↑ | −3.081 ↑ | −4.078 ↑ | −25.4 ↓ | −5.28 ↓ | +1.06 |
| 55 | −0.342 ↑ | −2.597 ↑ | −2.768 ↑ | −3.827 ↑ | −110.5 ↓ | −21.96 ↓ | +1.9×10⁶ |

↑ = value higher than at 10× (the hill); ↓ = lower.

### 9.2 Learning dynamics (`learn.json`, `learn_long.json`)

The learning model is a deliberately simple stand-in for training:
- 16 noise-only states, each with its own mean ln(size);
- Gaussian exploration starting at 0.756 in ln(size) (the matched value), shrinking six-fold over training as observed;
- REINFORCE with a per-time baseline, 64 episodes × 1,000 bars per iteration;
- four seeds per setting.

| Start | R1 | R1 ε 10⁻⁶ | R1 γ 0.999 | R1 undiscounted | **FL(0.01)** | **FL(0.05)** | FL(0.05) γ 0.999 | R2 |
|---|---|---|---|---|---|---|---|---|
| **Inside the basin (20×)**: share of states ruinous after 300 iterations | 56–81% | 31–50% | 62–81% | 69–88% | **0%** (sizes 3.7–4.4×) | **0%** (4.1–4.6×) | 25–75% | 75–88% |
| Q2 | −14.9 | −10.1 | −16.9 | −17.3 | **+0.39** | **+0.29** | −12.1 | −18.5 |
| **Safe start (1×)**, 300 iterations: median size / Q2 | 1.2–1.5× / +0.67 | same | 1.0–1.2× / +0.53 | 0.9–1.1× / +0.52 | **1.1–1.4× / +0.66** | **1.2–1.5× / +0.67** | 0.9–1.2× / +0.53 | 1.2–2.4× / −1.22 (6–12% ruinous) |
| Long runs (1,500 iterations, from 1× and from 2.93× with state spread): states ending ruinous | 0% | 0% | 0–25% | — | **0%** | **0%** | — | — |

Limits of the model, stated plainly:

- **It reproduces the trap and the escape, but not the entry.** It does *not* reproduce how real PPO agents enter the
  basin from a safe start under R1 at γ 0.99; there R1 stayed at 0%.
- **It drifts downward late in long runs.** All rewards drift to 0.2–1.3× late in long runs, a random walk caused by its
  optimiser normalising noisy gradients. This is a property of the model, not of any reward.
- **What it does establish:** the asymmetry of trapped versus escaping, and that FL adds no conservatism below the line.
  Whether FL prevents *entry* in real training must be tested with real agents (§10).

## 10. Exploitation and unintended incentives (question 8)

Policy values under each candidate. The reference is optimal sizing; any probe scoring at or above it marks an
exploitable incentive (`hack.json`, death rates in brackets).

| Probe | R1 | R1 ε 10⁻⁶ | R1 γ 0.999 | R1 undiscounted | **FL(0.01)** | **FL(0.05)** | R2 |
|---|---|---|---|---|---|---|---|
| optimal 2.93× (reference) | +0.028 | +0.028 | +0.167 | +0.263 | +0.028 | +0.028 | +0.052 |
| flat | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| timid 0.5× | +0.008 | +0.008 | +0.049 | +0.077 | +0.008 | +0.008 | +0.009 |
| 55× constant (96.9% die) | −0.376 | −2.65 | −2.66 | −3.98 | −111.3 | −22.2 | −42.0 |
| 2.93× with 5% spikes to 55× (16.7% die) | **+0.013** | −0.142 | +0.022 | +0.022 | −7.58 | −1.49 | **+0.106** |
| gamble to recover below 0.5 E₀ (2.3% die) | **+0.028** | +0.022 | **+0.170** | **+0.268** | −0.268 | −0.031 | +0.052 |
| 55× until equity doubles, then flat (14.6% die) | −0.002 | −0.823 | +0.018 | +0.022 | −40.2 | −8.0 | **+0.263** |
| 55× in the last 50 bars (15.9% die) | +0.028 | +0.028 | +0.096 | +0.074 | +0.024 | +0.028 | +0.052 |

The last probe matches optimal sizing under γ 0.99 only because those bars are discounted to almost nothing. Agents
cannot see the time remaining, so they cannot use it.

**Bold** marks a probe at or above optimal sizing (exploitable). Under FL there is none. Remaining risks for FL to
monitor:

- **Critic and gradient instability** from rare, very large negative rewards. Watch value loss, explained variance and
  gradient norms; advantage normalisation already rescales them.
- **Timidity:** sizes in non-ruinous decisions must not shrink relative to R1 (pre-registered check below).
- **Short positions:** an extreme up-move charges shorts the same way. This is symmetric and real; check the long share.
- **Bar granularity:** FL charges the full one-bar loss. In real data with intrabar stop-outs, the account's actual
  loss can be smaller than the bar-close calculation; FL should use the account's realised equity, which it does by
  construction.

## 11. Recommendation

The most scientifically defensible direction is **C3, the full-liability log reward**.

- **It targets the identified mechanism directly.** It removes the truncation that creates the hill, not the symptom.
- **It changes nothing where R1 is already right.** It is identical to R1 on every bar that does not wipe the account
  out, so it preserves R1's correct global optimum.
- **It adds no exposure limit, risk preference, style or death penalty.** Agents remain free to choose any size and to
  discover where growth turns into ruin.
- **It is consistent with the simulated account** (no negative-balance protection).

**Decision needed from you.** If future environments *do* protect against negative balances (e.g. environment B),
FL would charge losses that such an account does not bear. The argument for FL there is long-run sustainability rather
than accounting, and you should decide whether to accept it.

### Proposed validation experiment: Experiment R-FL (not started)

**Design**
- Groups, 6 paired fresh seeds each, 3M steps, everything else as the Experiment 1 control (original size output and
  exploration):
  - **R1 control**;
  - **FL(0.01)**, primary;
  - **FL(0.05)**, sensitivity.
- Separate branch, registry and storage; criteria and code checksums frozen before training; final checkpoints only;
  matched random agents; exploration monitor.

**Pre-registered criteria**
- The approved per-seed success criteria **W1–W7 and E**, with the group rule of at least 4 of 6 seeds.
- Planned comparisons against the concurrent R1 control: Q2, ruinous share, distance from the optimum. Exact
  permutation tests, Holm correction, bootstrap intervals.

**Additional checks against unintended incentives**
- **Timidity:** the median size of *non-ruinous* decisions must be no smaller than the R1 control's (interval
  excludes a fall of more than 25%).
- **Gambling to recover:** the ratio of size after a loss of more than 20% to size elsewhere must be at most 1.5.
- **Training stability:** no divergence, and explained variance and value loss recorded and compared.
- **Out-of-distribution:** the final agents tested without retraining on S6 variants with known optimal sizes (crash
  probability 1/150, crash size 15%, halved drift). Sizes should move in the direction of the optimum.
- **S7 sanity check** (no crashes): FL and R1 should behave the same there; any difference indicates a side effect.

**Interpretation, fixed in advance**
- If FL meets the success rule and R1 does not, the reward truncation is confirmed as a cause.
- If FL agents are safe but timid, the remaining problem is learning, i.e. finding 2.93×: the next proposal would
  address the learning signal (§8.2).
- If FL agents still enter the ruin zone, the cause lies outside the reward's shape (e.g. how one network couples sizes
  across states).

**Cost**
- 18 agents: about 3.5 hours of training and 45 minutes of evaluation on 4 cores.
- S7 sanity check: 12 agents, about 2 hours.
- Out-of-distribution tests: about 30 minutes.
- About 200 MB of checkpoints on a dedicated storage branch.

**Preservation**
- R1 stays frozen and unchanged. FL would be a new, separately named reward (R6), available only to new agents.
- All existing agents, versions, checkpoints, registries, evaluations and scoring rules remain untouched.

Awaiting your decision before any implementation.
