# Proposal 0006: Phase 3 learning interventions (for approval)

Status: **proposal only, 2026-11-17.** Nothing described here is implemented. No code, configuration,
agent, checkpoint, evaluation or registry record was changed while preparing it. The investigation
used read-only analyses of the registry (Phase 2 and Phase 2.5 records).

---

## 1. Executive summary

Phase 2.5 showed that more training under the frozen rules does not fix the growth agents' sizing
failures in S6. **0 of 3 growth seeds met the pre-registered criteria.** Ruinous bets rose for seed 2
(12.7% to 15.8%) and barely moved for seed 0 (10.6% to 9.9%). Seed 1 stayed timid (median 0.17x
against an optimum of 2.93x).

The investigation found the failure has several causes. Some are confirmed by measurement. Others
remain hypotheses, and are labelled as such.

**Confirmed:**

- **The penalty for a ruinous bet arrives at once.** 140 of 141 deaths in round 5 happened on the bar
  immediately after a ruinous decision. The learning horizon is therefore *not* the reason agents fail
  to connect ruinous bets with death.
- **The reward the agents optimise has a trap.** Under R1 with the 5% death rule, holding a size of 10x
  is worth less than holding 55x (a death is a bounded loss, and larger sizes earn more on the 299 in
  300 decisions without a crash). Once a decision lands above about 10x, local learning pushes it
  *further up*, not back to about 3x. The trap exists at every horizon tested (gamma 0.99, gamma 0.999,
  undiscounted).
- **The size output is extremely sensitive.** A shift of 0.25 in the network's size output multiplies
  leverage tenfold. Ruinous decisions come from such shifts (median output +0.31, against +0.08 for
  normal decisions), together with the direction output saturating at its limit (|d| = 1.00).
- **Agents react to noise.** S6's next return is independent of past returns, yet seed 2 makes ruinous
  bets four times as often after a strong up-move as after a strong down-move (28.7% against 7.0%).
- **The optimiser thrashes.** PPO moves the policy 5 to 10 times further per update than its normal
  target (median approximate KL 0.09 to 0.15), and about half of every batch is clipped.

**Recommendation:** test, first and one at a time, (C) a redesigned size output that can still reach
any positive size and (D) stabilised optimisation. Run them against a replicated control and with (A) a
longer horizon included as a falsification test. Test (B) persistent exploration separately on a
market with trading costs (S3), where the churn it addresses actually matters. Combine interventions
only after each has been measured alone.

## 2. Evidence-based diagnosis

Sources: `reports/phase25/s6_report.md`; the registry (54 versions, 162 checkpoints, 165
evaluations, training histories); decision records of round 5 (338,338 decisions from the 21 growth
versions); and decision records 0004 and 0005. Analyses were run read-only.

### 2.1 Confirmed findings

| # | Finding | Evidence |
|---|---|---|
| F1 | More training did not reduce ruinous decisions | Round 4/5 ruinous share for v1 against the candidate: seed 0 10.6% to 9.9%; seed 1 0.2% to 0.2%; seed 2 12.7% to 15.8%. Training episodes, first 20% against last 20% of each version: seed 2 B1 14.4% to 21.6%; no version below its starting level by a meaningful margin. |
| F2 | Agents *experienced* the consequences constantly | Seeds 0 and 2 died in 26% to 79% of training episodes (about 18,000 episodes in total). |
| F3 | The consequence is immediate | 140 of 141 deaths (round 5, growth versions) occurred on the bar right after a decision of 9.5x or more; the median delay between the last ruinous bet and death is 0 decisions. |
| F4 | R1 reward landscape has a basin beyond ruin | Expected value of always holding size G (exact expected R1, death ends the episode): gamma 0.99: 2.93x +0.027, 10x -0.81, 20x -0.54, 55x -0.13, 100x -1.14. gamma 0.999: +0.27, -2.50, -1.66, -0.39, -3.49. The optimum is global, but between about 10x and 55x value *increases* with size. |
| F5 | Ruinous sizes drift upward where agents are in the basin | Median / 90th percentile of ruinous sizes, v1 to stage-3 versions: seed 2 17x/51x to 22x/82x; control seed 2 19x/74x to 29x/169x; seed 0 stable (15x to 16x). |
| F6 | Size output sensitivity | G = 10^(4u) on u in [-1, 1]: d ln G / du = 9.2. Normal decisions median u = +0.08 (2.1x, seed 0); ruinous median u = +0.31 (16.9x). Ruinous decisions have the direction output saturated at 1.00 (normal 0.62 to 0.86). |
| F7 | Noise-driven sizing | Share of ruinous decisions by the previous bar's move (in standard deviations), seed 2: below -2: 7.0%; above +2: 28.7%. Seed 0 shows the reverse tilt (12.8% to 5.4%). S6 returns are independent, so neither tilt has any basis. |
| F8 | Spikes are mostly isolated | P(next decision ruinous given this one is): 0.16 to 0.21, against base rates 0.12 to 0.17. |
| F9 | Optimiser updates are very large | Median approximate KL per update 0.086 to 0.147; median clip fraction 0.46 to 0.55 (PPO's usual operating range is about 0.01 to 0.02 KL and 0.1 to 0.2 clip fraction). |
| F10 | Exploration of size did not collapse | Size-exploration standard deviation about 0.017 in u (one standard deviation = x1.17 leverage), steady across all stages. Direction exploration narrowed from 0.12 to 0.04. |
| F11 | Seed 1 never grew toward the optimum | Median size 0.12x to 0.18x in every version; no version met the 0.5x eligibility floor. |

### 2.2 Hypotheses (not confirmed)

| # | Hypothesis | What would test it |
|---|---|---|
| H1 | The basin (F4) is the main reason ruinous bets persist: local policy-gradient steps cannot cross the valley at about 10x | If C (lower sensitivity) reduces *entry* into the basin, ruinous share should fall without any reward change; a large-jump exploration probe should find the optimum from inside the basin. |
| H2 | Output sensitivity (F6) turns noise-driven state responses (F7) into ruinous bets | C should shrink size variance across states for the same network noise; the relation between previous move and size should weaken. |
| H3 | Large, noisy updates (F9) are why agents fit noise (F7) and do not converge | D (smaller, steadier updates) should lower KL and clip fraction, reduce the previous-move tilt, and give steadier learning curves. |
| H4 | Seed 1's timidity is a local optimum found early (small sizes are safe, and the gradient toward 2.9x is weak at 0.17x: g(0.17x) is 16% of the optimum) | Under C, timid seeds should still move toward the optimum if its pull is restored; under D they should move slowly but steadily. |
| H5 | Arm B (new market paths) did worse than arm A because changing data adds noise to already noise-driven learning | Re-test A against B under D; if H3 holds, the gap should shrink. |

### 2.3 Explanations rejected or weakened by the evidence

- **"The ~100-decision horizon hides delayed consequences."** Rejected for S6: the consequence is
  immediate (F3). A longer horizon raises the value of survival, but the basin persists at every
  horizon (F4).
- **"Exploration collapsed, so the agents could not try other sizes."** Rejected: size exploration
  stayed at about ±17% (F10). The problem is that local exploration cannot cross the valley (H1), not
  that exploration stopped.
- **"The critic is broken"** (explained variance about 0). Not a fault in S6: the next return cannot be
  predicted from the market, so explained variance near 0 is expected even for a perfect critic.

## 3. The interventions

All variants below apply only to new agents in a new experiment. Existing agents, records, scoring
rules and criteria stay frozen and untouched.

### A. Longer learning horizon

**What:** raise the discount factor (gamma 0.99 to 0.999, or define it per unit of market time) and/or
lengthen episodes.

**Addresses:** the low value placed on survival (lost future growth counts for about 100 decisions of
growth); real-market styles that need multi-day credit (audit H2/H3).

**Evidence against it as the S6 fix:** consequences are immediate (F3) and the basin survives a longer
horizon (F4).

**Costs:** none in compute per step. With gamma 0.999, value targets are 10x larger and much noisier,
so learning may slow.

**Risks:** higher-variance advantages make the noise problem (F7, F9) worse; possible instability.

**Role in Phase 3:** a falsification arm in S6 (prediction: little or no reduction in ruinous share)
and a candidate for real-market work later.

### B. Persistent exploration

**What:** keep an exploratory change in place for several decisions instead of redrawing it every
decision. Options:

1. generalized state-dependent exploration (Stable-Baselines3 `use_sde`, resampled every k steps);
2. temporally correlated (autoregressive) action noise;
3. exploring in network parameters.

**Addresses:** the training-churn bias (audit H1). On real January 2022 data, per-decision noise caused
12,843 fills and $6.14 of costs in a week for a policy meant to hold one position; persistent noise
would cause a few dozen.

**For S6:** S6 has no trading costs, so churn cannot explain the S6 failure. Persistent exploration
does not change the noise *magnitude*, so it does not cross the valley either.

**Costs:** small (`use_sde` adds a matrix product per step).

**Risks:** gSDE interacts with the action representation (C). With the current log-scale size, noise
that is held for k steps also holds a possibly ruinous size for k steps, so B should not precede C.

**Role in Phase 3:** test on S3 (noise plus trading costs; Phase 2 failed it 2 of 3), where churn
matters; later on real data.

### C. Redesigned position-size output (any positive size remains reachable)

**Problem:** G = 10^(4u), u clipped to [-1, 1]. A 0.25 change in u means x10 leverage; Gaussian
exploration in u is multiplicative on leverage; actions pile up at the clip limits (F6).

**Candidate representations.** None caps exposure; all can express any positive size.

| Option | Size from network output z | Sensitivity at typical sizes | Comment |
|---|---|---|---|
| C0 current | 10^(4u), u in [-1, 1] | x10 per 0.25 | reference |
| **C1 unit-gain log, unbounded** | G = e^z, z unbounded (numerical safeguard only, e.g. |z| <= 25) | x1.28 per 0.25 | keeps scale-free sizing, removes clip saturation, 9.2 times less sensitive |
| **C2 softplus (near-linear)** | G = softplus(z) | +0.25x per 0.25 | large sizes need large outputs: 17x needs z of about 17, far from typical outputs, so spikes are much rarer |
| C3 split direction | direction weights through a softmax with an explicit "no position" share, times a separate size | n/a | gives an exact "stay out" choice (audit M5); bigger change, for later |

**Addresses:** F6, H2, and entry into the basin (H1).

**Costs:** none in compute. Existing checkpoints cannot be reused, because the output meaning changes,
so C requires new agents trained from scratch.

**Risks:**

- C2's additive scale makes very small sizes (below about 0.05x) and very large ones harder to fine-tune.
- C1 still allows large multiplicative jumps under large output shifts.
- Either changes what exploration means, so exploration settings must be recalibrated (an initial
  noise level chosen once, recorded, and identical across arms).

**Role in Phase 3:** the primary intervention for S6.

### D. Stabilised optimisation (found during this investigation; not in the original list)

**What:**

- a smaller learning rate (3e-4 to 1e-4);
- fewer epochs per batch (10 to 4);
- a KL limit per update (`target_kl` of about 0.02);
- optionally larger batches.

**Addresses:** F9 (updates 5 to 10 times larger than intended), H3 (fitting noise), H5.

**Costs:** slower learning per step; may need more steps.

**Risks:** may lock in the timid optimum (seed 1) by making escape slower.

**Role in Phase 3:** a single-factor arm in S6. This was arm "C" of the earlier proposal (not
approved then); it is now supported by evidence.

### Root issue outside the four interventions: the reward basin

The basin (F4) comes from R1 combined with the approved 5% death rule. It is a property of the frozen
reward, not of the agents. None of A to D removes it; C and D can only make agents less likely to
enter it and better able to stay out. Removing it would mean changing how death is valued (for
example, treating ruin with the true log-wealth consequence). That changes the frozen scoring rules
and is **not** proposed here; it is listed so that a future decision can be made with full
information if C and D prove insufficient.

## 4. Comparison of interventions

| Problem (finding) | A horizon | B persistent exploration | C size output | D stable optimiser |
|---|---|---|---|---|
| Ruinous spikes from output sensitivity (F6) | no | no (could make it worse if tested first) | **yes** | partly (smaller policy jumps) |
| Basin beyond ruin (F4, H1) | no (persists at every horizon) | no | **reduces entry** | reduces drift within it |
| Noise-driven sizing (F7) | worse (noisier targets) | no | partly (less amplification) | **yes** (H3) |
| Thrashing updates (F9) | no | no | no | **yes** |
| Timidity (F11, H4) | slight (raises value of growth) | possibly (wider exploration) | possibly (changes pull toward 2.9x) | uncertain |
| Training churn with costs (audit H1) | no | **yes** | no | no |
| Multi-day credit on real markets (audit H2/H3) | **yes** | no | no | no |

Interactions:

- **C before B:** holding noise for k steps on a log-scale output holds possibly ruinous sizes.
- **D with A:** a longer horizon raises advantage variance; D's smaller steps may be needed for A to
  learn at all.
- **C with D:** independent mechanisms (where the size lands, and how far each update moves it). They
  are expected to add up, but this must be measured, not assumed.

## 5. Experimental architecture

- **New identity.** Each arm is a new experiment (for example `phase3-s6-arm-C1`) with its own agents,
  identities, versions, checkpoints and evaluations in the same registry. No existing record is
  touched. The registry seal continues to cover old and new records.
- **New agents trained from scratch** for every arm, including a **replicated control** (current
  settings, new seeds). Interventions C and D change what an agent is, so continuing from v1 would mix
  the old and the new. The original v1 agents remain the historical reference only.
- **Frozen in every arm:** market mechanics, reward R1, death rule, margin, criteria, the decision-
  quality score Q2.0 (exact for S6), and the open trading principle (no style, no cap, no penalties).
- **Each arm changes exactly one thing** against the replicated control; recorded in its manifest.
- **Seeds:** 6 independent seeds per arm, drawn fresh. These are separate from Phase 2 seeds (0 to 2)
  and from each other.
- **Data:** fresh training paths per seed; fresh evaluation rounds from the ledger, identical for all
  arms within a round; sealed final data untouched.

## 6. Experimental sequence

**Prerequisites** (each needs approval; none is started):

1. Storage: about 6 MB of checkpoints per 3M-step agent at the 250k-step interval, about 150 MB per
   25 agents. Move checkpoint blobs to dedicated artifact storage, or adopt a 500k-step interval, before
   Phase 3.
2. Implement the interventions as new, opt-in options recorded in each new agent's identity; defaults
   for existing agents unchanged; tests for each.
3. For S7 work later: an exact decision-quality version for S7 (Q2.0 is exact only for S6).

**Experiment 1 (recommended first): S6 single-factor ablation.**
Arms: Control (replicated current settings), C1, C2, D, A. 5 arms x 6 seeds x 3M steps each (the Phase
2 budget).

**Experiment 2: S3 churn test.** Control against B (persistent exploration), 6 seeds each. S3 has
trading costs and no edge: the right behaviour is to stop trading. Tests whether B removes the churn
bias.

**Experiment 3 (only if justified by Experiments 1 and 2):** the best single interventions combined
(for example C plus D, then plus B), against the same control.

**Experiment 4 (out-of-distribution):** the final agents of Experiments 1 and 3, tested without retraining
on S6 variants whose growth-optimal size is known exactly:

- S6-a: crash probability 1/150 (optimum lower);
- S6-b: crash size 15%;
- S6-c: drift halved;
- S6-d: S7-like market without crashes (optimum higher).

This tests whether sizing responds to the market or is memorised.

Selection rule: **none.** Each agent's final checkpoint is the one evaluated, so there is no
candidate selection and nothing to leak from confirmation. Learning curves (evaluations every 500k steps
on one development round) are descriptive only.

## 7. What counts as genuine learning (pre-registered criteria, Experiment 1)

Unit of analysis: the seed (6 per arm). Per seed: final checkpoint, 2 fresh rounds of 20 test runs.

**Genuine improvement of an arm over the replicated control requires all of:**

| # | Criterion | Guards against |
|---|---|---|
| G1 | Mean Q2.0 higher than control, permutation test across seeds p < 0.05 after Holm correction across arms, effect at least +0.5 | random improvement |
| G2 | Ruinous share below 1% in at least 5 of 6 seeds | ruin |
| G3 | Median size between 0.5x and 2x of the optimum (1.47x to 5.86x) in at least 4 of 6 seeds | excessive conservatism, recklessness |
| G4 | Realised growth over the 40 test runs above zero, with the 95% interval above zero, and above the 95th percentile of leverage- and turnover-matched random agents | luck, drift-only profits |
| G5 | The same direction of effect in both rounds (replication) | one-round flukes |
| G6 | In training history, ruinous share falls from the first to the last 20% of training, in at least 4 of 6 seeds | improvement that never appears while learning |

**Reward-exploitation checks** (must all be clear):

- no gain that appears only on training paths (compare training-path and fresh evaluation);
- no dependence on simulator boundaries (positions near the margin limit or the numerical range);
- the previous-move tilt in sizing shrinks (|seed 2-style tilt| below 1.5x);
- no rise in rejected orders.

**Excessive conservatism** is caught by G3 and by Q2.0 itself (staying out scores 0; 0.17x scores 0.10).

**Learning from catastrophe** (measured from training records): ruinous share in the 50 training
episodes after an agent's death compared with the 50 before, averaged over deaths. It must fall
(paired test) in the arm credited with improvement.

**Adaptation (Experiment 4):** the rank correlation, across the four variants, between each agent's
median size and the variant's growth-optimal size must be positive in at least 4 of 6 seeds, with no
variant producing a ruinous share above 5%.

## 8. Statistical methodology

- **Seeds are the independent units.** Test runs within a seed are not independent. Arm comparisons use
  per-seed means.
- Permutation tests across seeds (all 924 relabellings of 6 against 6 seeds), Holm correction across the
  four intervention arms, and 95% bootstrap intervals for the effect size.
- **Power:** seed-to-seed Q2 spread in Phase 2.5 was large (seed 0 -2.3, seed 1 +0.1, seed 2 -3.0), so
  6 seeds per arm can reliably detect effects of about one between-seed standard deviation. Smaller
  effects would need 10 or more seeds (cost scales linearly).
- **Within-seed comparisons** (paired by episode on identical data, as in Phase 2.5) are reported as
  supporting evidence only.
- **All arms, seeds and results are reported**, including failures and every number of attempts.

## 9. Computational requirements (measured rates: 12.4 min per 1M steps per job, 4 jobs in parallel; 21.5 s per evaluation)

| Experiment | Training | Evaluations | Wall time (4 cores) |
|---|---|---|---|
| 1: S6 ablation, 5 arms x 6 seeds x 3M | 90M steps | 30 agents x (2 rounds + 6 curve points) = 240 | about 4 h 45 min + 22 min |
| 2: S3, 2 arms x 6 seeds x 1M | 12M steps | 12 x 4 = 48 | about 40 min |
| 3: combination, about 2 arms x 6 seeds x 3M | 36M steps | about 100 | about 2 h |
| 4: OOD, about 18 agents x 4 variants x 2 rounds | none | 144 | about 15 min |
| **Total** | **about 138M steps** | **about 530** | **about 8 hours** |

Storage: about 6 MB per 3M-step agent at the 250k-step interval (about 330 MB for the programme), or
about 3 MB at the 500k-step interval. Moving to dedicated artifact storage first is recommended.

## 10. Recommendation

1. **Approve the prerequisites:** a storage decision, and implementing the interventions as opt-in
   options for new agents only, with tests. No existing record changes.
2. **Run Experiment 1 first:** S6 ablation with Control, C1, C2, D and A (A as a falsification test of
   the horizon explanation).
   - Strongest prediction: C1 and C2 reduce ruinous share; D reduces noise-driven sizing and
     thrashing; A changes little.
   - If none of C1, C2 or D meets the criteria, the evidence will point squarely at the reward basin
     (F4), and the next proposal would discuss the frozen reward with you.
3. **Run Experiment 2 (S3, persistent exploration) in parallel or next.** It is cheap and is what the
   audit's churn finding needs before any real-market training.
4. **Combine interventions only after** individual results justify it.

Throughout, agents receive no trading style, no exposure cap and no new penalties. Each change in
this proposal alters how agents represent or learn their choices, never what they are allowed to
choose.
