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

## Revision 2026-11-17: Q2.0 replaces Q (pending owner approval and freezing)

The owner asked for worked examples at 10x, 20x and 100x before approving Q. They exposed the same
kind of flaw in Q (the episode-replay score, here called Q1) as in the first variant: once death is
near certain, the rare crash-free simulated paths decide the score, and those favour larger
positions. Q1 scored 20x (-9.55) above 10x (-10.12). Q1 is therefore rejected as well.

**Q2.0** scores each decision directly from the exact expected log growth g (defined above), which
is concave in the position size:

    s(G) = max(g(G) / g(G*), F)   for a non-ruinous decision,     s(G) = F   for a ruinous one,
    F = g(G_ruin) / g(G*) = -21.353 in S6  (the score of the smallest ruinous position, 9.5x).

Q2 of an episode is the mean of s over its decisions. Q2 of an evaluation is the mean over its
episodes, and the per-episode values are used for paired comparisons. Worked examples: staying
out 0, optimal 2.93x scores 1, and 10x, 20x and 100x all score exactly -21.353 (the worst, tied).

Concavity guarantees that a position further beyond the optimum never scores higher. This is checked
over 400,001 sizes and by a property-based test. Q2 needs no simulation, is exact, and uses
only decisions and known parameters.

Q2 does not count the decisions an agent would have made after an account died. Death timing
depends on the realised path, so counting it would make the score depend on luck. Deaths are
reported separately, and the ruinous decisions that cause them already score the minimum.

Version 1 baselines recalculated with Q2.0 are stored as separate write-once records
(`registry/analyses/decision-quality-Q2.0/`). Full worked examples and the baseline table are in
`reports/phase25/q2_decision_quality.md`. The earlier scores remain in the original evaluation
records, unchanged.

Implementation checksum (to be frozen on approval as Q2.0):
`0d0c6486487edacee4b54a34180d63855d77c413ab79e5bb15f27616aa292147`.

### Verification and scope (before freezing)

Independent verification (`scripts/verify_q2_math.py`, `reports/phase25/q2_math_verification.md`):
36 of 36 checks pass. g matches Simpson integration to about 1e-10 relative and 10-million-draw
Monte Carlo within about 1.4 standard errors at every size tested. An independent search finds the
same optimum (2.93073x, difference 2e-7). The analytic second derivative is negative everywhere. A
crash at 9.5x leaves exactly the 5% death line. The score never rises beyond the optimum, on either side.

Scope. In a normal model any position above 1x can in principle be wiped out by an extreme enough
move, and the quadrature caps the logarithm at its most extreme nodes. For S6 that cap acts only
where the score is already clamped at F (shorts beyond -17.3x, clamped from -15.4x; longs beyond
18.6x, clamped from 9.5x), so every S6 score is exact. For S7 (about three times the volatility) the
cap acts for longs from about 5.9x, below its 19.65x clamp. **Q2.0 is therefore frozen for S6 only.**
The S7 baseline values in `registry/analyses/decision-quality-Q2.0/` are provisional.
