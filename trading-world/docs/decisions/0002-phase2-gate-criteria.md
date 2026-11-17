# 0002: Phase 2 synthetic-experiment gate criteria (pre-registered)

Status: recorded before any gate results were produced (2026-11-17).

Phase 2 asks one question: **can the learning loop learn at all, and does it learn the right
things?** Each synthetic market (S1–S7, `src/tradeworld/learning/synthetic_tasks.py`) has a known
correct behaviour. An agent is trained on one market path and evaluated on a different,
unseen path from the same process (out-of-sample), with deterministic actions, over 20 fixed
test episodes of 1,000 decisions.

| Task | Reward | Pass criterion (all on the test path) |
|---|---|---|
| S1 AR(1) + costs | R1 | mean log-growth > 0 **and** above the 95th percentile of leverage- and turnover-matched random agents |
| S2 visible regime | R1 | position agrees with the regime flag ≥ 80% of the time it is in the market, **and** mean log-growth > 0 |
| S3 random walk + costs | R1 | ≈ flat: mean gross exposure < 0.2× (an untrained policy sits near 1×) |
| S4 random walk, no costs | R1 | no systematic profit: mean log-growth not significantly > 0 (t < 2) |
| S5 leakage canary | R1 | no systematic profit (t < 2); a significant profit would indicate a leak |
| S6 drift + crashes | R1 | death rate ≤ 10% **and** mean gross leverage between 1× and 6× (growth-optimal ≈ 2.9×) |
| S6 drift + crashes | R2 | mean gross leverage > 2× the R1 agent's **and** a higher death rate than R1 |
| S7 GBM, known μ/σ² = 5 | R1 | mean net leverage between 2.5× and 10× (within a factor of 2 of 5×) |
| S7 GBM | R2 | mean net leverage outside 2.5×–10× or death rate ≥ 50% |

A task passes if the criterion holds for the majority of training seeds. Every result,
including failures, is reported. Hyperparameters are recorded with each run; if they are
changed after a failed attempt, both attempts are reported and the change is explained.

Settings fixed for the gate: one decision per M1 bar, 30-bar observation window (synthetic
validation only; the production default stays 5-minute decisions on 120 × M5 bars),
A-reference margin 1:2000, no slippage, costs from each task's spread, PPO (Stable-Baselines3),
MLP 64-64 tanh, deterministic evaluation.
