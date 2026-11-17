# 0004: S6 decision-quality score (Phase 2.5 analysis measure)

Status: defined and validated 2026-11-17, before any Phase 2.5 training. This is an analysis
measure only. It does not change the reward functions, scoring rules or pass/fail criteria.
Code: `src/tradeworld/learning/quality.py`. Evidence: `reports/phase25/decision_quality_validation.md`
(produced by `scripts/validate_decision_quality.py`).

## The market

One decision = one M1 bar. Each bar's log return is

    r = ln(1 - J)                     with probability p          (crash)
    r = m + sigma * Z,  Z ~ N(0,1)    with probability 1 - p,     m = mu - sigma^2 / 2

with mu = 0.0005, sigma = 0.003, p = 1/300, J = 0.10 (`synthetic_tasks.build("S6")`).
S6 has no spread, slippage, commission or swap and fills at the previous close, so a position of
net exposure G (position value / equity; negative = short) multiplies equity by exactly
M(G) = 1 + G (e^r - 1) over the bar.

## The score

Expected log growth of one decision (the Kelly growth rate, and the expectation of the R1 reward):

    g(G) = (1 - p) * E_Z[ ln(1 + G (e^(m + sigma Z) - 1)) ] + p * ln(1 - G J)

computed with 96-point Gauss-Hermite quadrature. g(G) = minus infinity for G >= 1/J (a crash takes
equity to zero or below).

- Growth-optimal size G* = argmax g = **2.931x**; g(G*) = 2.652e-4 per decision.
- **Ruinous decision:** one ordinary adverse move ends the account (equity at or below the 5%
  death line): G >= (1 - 0.05)/J = **9.5x** long, or |G| >= 0.95/(e^(m + 5 sigma) - 1) = 60.8x short.
- **Decision quality q** = mean of g(G_t)/g(G*) over non-ruinous decisions: 1 = optimal sizing,
  0 = no better than staying out, negative = sizes that lose on average.
- **Ruinous share** = fraction of ruinous decisions, reported separately.
- **Episode decision quality Q:** an episode's recorded sequence of sizes is replayed on 2,000
  simulated S6 paths with death at 5% permanent. The mean final log growth is divided by
  H g(G*) for H decisions. This puts sizing quality and ruin on one scale. Episode k of every
  version uses the same simulated paths (common random numbers), so version comparisons are paired.

All three depend only on the agent's decisions and the market's known parameters. They never use
the returns that actually occurred, so luck cannot raise or lower them.

## Validation (numbers from the report)

| Case | Result |
|---|---|
| Staying out (G = 0) | g = 0, q = 0, Q = 0 exactly |
| Optimal size 2.93x | q = 1.000 (maximum; 2x scores 0.885, 4x 0.820); Q = 0.974 (rare double crashes) |
| Too large: 6x / 8x / 9x | q = -0.86 / -6.30 / -13.4; Q = -0.76 / -4.10 / -6.54; episode death rate 12% / 42% / 64% |
| Ruinous: 9.5x and above | flagged ruinous; Q = -8.1 at 9.5x and -9 to -10 from 10x up (96% of episodes die; differences inside this band are not meaningful) |
| Formula vs Monte Carlo (2 independent samples of 4M draws) | within 2 standard errors at every size (sample 1 held 230 more crashes than expected, moving all sizes together; sample 2 within 0.23 SE) |
| Formula vs the trading simulator (constant-size agents, 60,000 decisions) | 1x, 2.93x, 5x all within 0.32 SE; staying out exact |
| Same decisions, different paths | on one path the account died, on the other it grew +4,844%; scores identical |
| Lucky gambler at 12x on a crash-free path | large realised profit, yet ruinous share 100% and Q = -10.1 |

## A rejected variant

A per-decision score that treated death as a one-step loss capped at the 5% line was rejected.
It counts death as a bounded loss and ignores the growth lost afterwards, so it rises again above
10x (-20.6 at 10x, -6.9 at 20x) and would have rewarded larger ruinous bets.

## Use in Phase 2.5

Candidate selection (rounds 1-3) uses Q. Criterion C1 uses the ruinous share. C2 uses the paired
difference in Q over the 20 identical test runs. C3 uses the median position size against G*.
On real market data the return process is unknown, so this score does not apply; only
behavioural and consequence measures do.
