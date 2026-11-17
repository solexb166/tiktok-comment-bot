# Decision-quality score Q2.0 (proposed for freezing)

Implementation checksum: `0d0c6486487edacee4b54a34180d63855d77c413ab79e5bb15f27616aa292147`

## Definition

Per decision with net exposure G (position value / equity): s(G) = max(g(G) / g(G*), F), and s(G) = F for a ruinous decision. g(G) = (1 - p) E_Z[ln(1 + G (e^(m + sigma Z) - 1))] + p ln(1 - G J) is the exact expected log growth of holding G for one decision (96-point Gauss-Hermite quadrature; minus infinity if G >= 1/J). G* = argmax g. F = g(G_ruin) / g(G*), the score of the smallest ruinous long position (G_ruin = 0.95 / J in S6; for a market without crashes, the size at which a 5-sigma fall reaches the 5% death line). Ruinous: G >= G_ruin long, or |G| >= 0.95 / (e^(m + 5 sigma) - 1) short. Q2 of an episode = mean of s over its decisions; Q2 of an evaluation = mean of the episode values. Uses only the decisions and the market's known parameters, never realised returns.

Code: `src/tradeworld/learning/quality.py` (`KnownMarket.growth`, `q2_floor`, `q2_scores`, `q2_metrics`).

S6 constants: mu = 0.0005, sigma = 0.003, p = 1/300, J = 10%. G* = 2.9307x, g(G*) = 2.652324e-04 per decision, G_ruin = 9.50x, F = g(G_ruin)/g(G*) = -21.353.

## Worked examples (S6, a decision holding position size G)

| Position | g(G): expected log growth per decision | Ruinous | Q2 score | Rejected Q1 (episode replay) |
|---|---|---|---|---|
| Staying out (0x) | 0.0000e+00 | no | **0.000** | 0.000 |
| Optimal (2.93x) | 2.6523e-04 | no | **1.000** | 1.057 |
| 10x | minus infinity | yes | **-21.353** | -10.116 |
| 20x | minus infinity | yes | **-21.353** | -9.551 |
| 100x | minus infinity | yes | **-21.353** | -10.805 |

- Staying out: g(0) = ln(1) = 0, so the score is exactly 0.
- Optimal: s = g(G*)/g(G*) = 1, the maximum of the score.
- 10x, 20x, 100x: each is at or above G_ruin = 9.50x. A single crash (probability 1/300 per decision) takes equity to 1 - G x 10% <= 0, so g = minus infinity. All three score exactly F = -21.353, the worst possible value, and tie: a larger ruinous position can never score better.
- The rejected Q1 column shows the flaw that Q1 had: 20x scored above 10x.

## Why a larger excessive position can never score better

g(G) = E[ln(1 + G X)] with X = e^r - 1 is concave in G (the logarithm of a linear function of G, averaged). A concave function rises up to its maximum G* and falls from there on: for G* <= a < b, g(b) <= g(a). Dividing by g(G*) > 0 keeps the order, and max(., F) and the ruinous value F can only flatten it, never reverse it. So s(b) <= s(a): a position further beyond the optimum never scores higher. The same holds for short positions (sizes below zero), where g decreases as the short grows.

Numerical check over 400,001 sizes from 2.93x to 10,000x: largest increase between neighbours = 0.0e+00 (none). Shorts, -10,000x to 0: largest decrease toward 0 = 0.0e+00 (none). Property-based test: `tests/test_quality.py::test_q2_never_rewards_a_bigger_position_beyond_the_optimum`.

## Version 1 baselines recalculated with Q2.0

From the preserved decision records of each agent's original gate evaluation. Stored as separate write-once records (`registry/analyses/decision-quality-Q2.0/`). The previous scores stay in the original evaluation records, unchanged, and are shown for comparison.

| Agent | Q2.0 (new) | Q1 (previous, rejected) | q on non-ruinous decisions (previous) | Ruinous decisions | Deaths | Median size | Original verdict |
|---|---|---|---|---|---|---|---|
| S6-R1-seed0 | **-2.450** | -1.300 | -0.123 | 10.9% | 45% | 1.30x | fail |
| S6-R1-seed1 | **+0.097** | +0.108 | +0.154 | 0.3% | 0% | 0.17x | fail |
| S6-R1-seed2 | **-2.797** | -1.661 | -0.058 | 13.2% | 40% | 1.43x | fail |
| S6-R2-seed0 | **-0.554** | -0.121 | +0.186 | 3.1% | 15% | 0.53x | fail |
| S6-R2-seed1 | **-0.696** | -0.409 | +0.097 | 3.6% | 5% | 0.42x | pass |
| S6-R2-seed2 | **-3.209** | -7.398 | +0.006 | 11.5% | 70% | 1.05x | fail |
| S7-R1-seed0 | **+0.211** | -0.896 | -23.264 | 0.0% | 10% | 1.36x | pass |
| S7-R1-seed1 | **+0.272** | +0.248 | -0.757 | 0.0% | 0% | 0.65x | fail |
| S7-R1-seed2 | **+0.138** | -1.026 | -61.337 | 0.0% | 30% | 0.75x | fail |
| S7-R2-seed0 | **+0.258** | +0.097 | -11.020 | 0.0% | 5% | 0.74x | pass |
| S7-R2-seed1 | **+0.269** | +0.243 | -0.472 | 0.0% | 0% | 0.57x | pass |
| S7-R2-seed2 | **+0.255** | +0.009 | -20.760 | 0.0% | 10% | 0.94x | pass |

S7's growth-optimal size is 4.99x; its floor is the score of the size at which a 5-sigma fall reaches the death line (19.4x).
