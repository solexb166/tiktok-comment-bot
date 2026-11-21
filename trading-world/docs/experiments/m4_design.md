# Experiment M4: design notes (written before pre-registration)

**Aim:** stage 6 of the road to demo trading. Markets change; an agent must notice from its own results, adapt, and
not raise false alarms in markets that do not change. Agent: version 4 (`model_kelly4`). Runner: `m4.py`. Frozen
criteria: `m4_analysis.py`.

## Markets (new; earlier markets byte-identical, tested)

| Market | Built on | What switches, at unannounced times |
|---|---|---|
| S1X | S1C (spread 0.4 sigma) | The pattern flips between bounce-back (phi −0.3) and momentum (+0.3). Mean spell 3,000 bars. |
| S2X | S2C (spread 1.0 sigma) | The announced trend flag flips meaning. Mean spell 3,000 bars. |
| S6X | S6 (no costs) | Crash frequency switches between calm (1/300) and storm (1/50). Mean spells 6,000 and 1,500 bars. |

The true regime path is kept in the task's hidden field. No agent sees it; only the labelled oracle reference does.

## Prototypes (fast stand-alone simulator unless stated; seeds below 10,000, so no overlap with agents' ledger seeds)

### 1. A fixed (learn-once) model fails on switching markets

It averages the regimes:
- S1X: averaged pattern about −0.05, so growth is 0.
- S2X: +0.29 with 21 of 60 deaths.
- S6X: +0.02.

### 2. Forgetting models (recursive least squares with exponential forgetting)

- Given the right single feature, they did well: S1X +4.16, S2X +4.94.
- On all features with a short memory they were noisy and overbet: 14–38 deaths in 39 held-out runs.
- They were dropped: the good results needed the feature to be handed to them.

### 3. Learned regime models (hidden Markov model fitted by Baum-Welch on the agent's own experience)

- **S1X:** found both patterns exactly (+0.29 and −0.30) and earned +5.17 with 0 deaths, against the oracle's +4.86.
- **S6X:** a dedicated two-state crash model reached 71% of the oracle. The general model reached about 58% (prototype)
  and 28% (real simulator, one seed) of the oracle.
- **S2X:** found both meanings (flag ±1.5e-4, true ±1.5e-4) but died often (10 of 60).

### 4. Model choice

- The number of regimes chosen on held-out log likelihood alone picked 2–3 regimes on stable markets for negligible
  gains.
- Hence the rule: a larger number of regimes is kept only with a paired t > 2 on held-out per-bar likelihood. Stable
  markets (S1C, S6) then chose one regime, which is the static model.
- Several starting points per fit, keeping the best training fit, fixed a local optimum in S2X.

### 5. Choosing models by held-out decision growth

This was tried and abandoned: with 39 held-out runs it was too noisy to choose reliably.

### 6. Real simulator, one seed per market, 20 fresh test runs

| Market | v4 growth | v4's single-regime model | True-model filter | Oracle |
|---|---|---|---|---|
| S1X | +3.17, 0 deaths | −0.45 | +2.80, 3 deaths | +3.59, 2 deaths |
| S2X | +1.59, 6 deaths | −0.22 | +2.18, 5 deaths | +4.72, 0 deaths |
| S6X | +0.68, 0 deaths | +0.15 | +1.54, 0 deaths | +2.47, 0 deaths |

### Conclusions drawn before freezing

- The fair benchmark is the true-model filter. It knows the rules but, like any trader, must infer the current regime
  from prices. The oracle is reported only as an upper bound.
- S2X's deaths are not a learning failure. Full growth-optimal sizing of a weak signal that can silently reverse is
  inherently risky: the true-model filter died 5 of 20 times. Hence no absolute death ceiling in A4. This is reported
  as a risk-appetite question for the owner (fractional sizing), not imposed.

## Known limitations, stated in advance

- **Linear world model.** Within each regime the model is linear in the last three returns and visible side features.
  Version 3's non-linear network is not used.
- **Tail assumption.** Version 3's tail assumption of about 10 sd per learned component carries over.
- **Short test runs.** A test run is 1,000 bars. About a third of runs contain a regime switch in S1X/S2X, and fewer
  in S6X.
