# Q2.0 mathematics: independent verification

Implementation checksum verified: `0d0c6486487edacee4b54a34180d63855d77c413ab79e5bb15f27616aa292147`

## 1-2. Expected log growth g(G): implementation vs Simpson integration vs Monte Carlo

| G | Implementation (Gauss-Hermite) | Simpson (independent) | Relative difference | Monte Carlo (10M) | MC difference / SE |
|---|---|---|---|---|---|
| 0.00 | 0.00000000e+00 | 0.00000000e+00 | 0.0e+00 | 0.0000e+00 | +0.00 |
| 0.50 | 7.70983551e-05 | 7.70983551e-05 | 2.1e-11 | 7.8421e-05 | +1.26 |
| 1.00 | 1.42646614e-04 | 1.42646614e-04 | 2.3e-11 | 1.4533e-04 | +1.25 |
| 2.00 | 2.34683306e-04 | 2.34683306e-04 | 2.7e-11 | 2.4022e-04 | +1.23 |
| 2.93 | 2.65232335e-04 | 2.65232335e-04 | 3.5e-11 | 2.7362e-04 | +1.21 |
| 4.00 | 2.17532149e-04 | 2.17532149e-04 | 5.8e-11 | 2.2949e-04 | +1.19 |
| 6.00 | -2.28750179e-04 | -2.28750179e-04 | 8.1e-11 | -2.0875e-04 | +1.13 |
| 8.00 | -1.67033265e-03 | -1.67033265e-03 | 1.4e-11 | -1.6383e-03 | +1.06 |
| 9.00 | -3.55996024e-03 | -3.55996024e-03 | 7.6e-12 | -3.5171e-03 | +1.00 |
| 9.49 | -5.60152021e-03 | -5.60152021e-03 | 5.0e-12 | -5.5488e-03 | +0.96 |
| -1.00 | -1.85376169e-04 | -1.85376169e-04 | 1.8e-11 | -1.8792e-04 | -1.28 |
| -3.00 | -6.62483425e-04 | -6.62483425e-04 | 1.5e-11 | -6.6980e-04 | -1.31 |
| -10.00 | -3.14077377e-03 | -3.14077377e-03 | 1.1e-11 | -3.1628e-03 | -1.38 |

## 3. Optimum

Independent ternary search: G* = 2.93073x; implementation: 2.93073x (difference 1.9e-07). g(G*) = 2.652324e-04 (implementation 2.652324e-04).

## 4. Concavity

g''(G) = -(1-p) E[X^2 / (1+GX)^2] - p J^2 / (1-GJ)^2, a sum of negative terms wherever g is finite, so g is strictly concave (it has a single peak and falls on both sides of it).
Evaluated at 600 sizes from -50x to 9.49x: largest value -1.295e-05 (all negative).

Numerical scope. In a normal model any position above 1x can in principle be wiped out by an extreme enough move; the quadrature caps the logarithm at its most extreme nodes (|z| up to 18.5). For S6 that cap acts only for shorts beyond -17.32x and longs beyond 18.6x, while the score is already clamped at F for shorts beyond -15.36x and longs from 9.50x. Every S6 score Q2 can produce is therefore computed exactly. Second differences of the implementation on 6,001 sizes from -15.36x to 9.49x: largest -2.58e-10 (none positive). Outside that range the score is the constant F.

S7 is NOT covered: its volatility is about three times higher, so the cap acts for longs from about 5.9x, below its 19.65x clamp (an error of up to about 5% near 19x). Q2.0 is frozen for S6 only; S7 values are provisional.

## 5. Ruin threshold and floor

A crash at 9.50x leaves 1 - 9.50 x 0.10 = 0.0500 of equity: the death line. Floor F = g(9.5x)/g(G*): implementation -21.35264, independent -21.35264.

## 6. Score monotonicity

Checked on 400,001 sizes from the optimum to 10,000x, 400,001 sizes from -10,000x to 0, and 100,001 sizes from 0 to the optimum.

## Result

- PASS: g matches Simpson at 0.00x
- PASS: g within 4 SE of Monte Carlo at 0.00x
- PASS: g matches Simpson at 0.50x
- PASS: g within 4 SE of Monte Carlo at 0.50x
- PASS: g matches Simpson at 1.00x
- PASS: g within 4 SE of Monte Carlo at 1.00x
- PASS: g matches Simpson at 2.00x
- PASS: g within 4 SE of Monte Carlo at 2.00x
- PASS: g matches Simpson at 2.93x
- PASS: g within 4 SE of Monte Carlo at 2.93x
- PASS: g matches Simpson at 4.00x
- PASS: g within 4 SE of Monte Carlo at 4.00x
- PASS: g matches Simpson at 6.00x
- PASS: g within 4 SE of Monte Carlo at 6.00x
- PASS: g matches Simpson at 8.00x
- PASS: g within 4 SE of Monte Carlo at 8.00x
- PASS: g matches Simpson at 9.00x
- PASS: g within 4 SE of Monte Carlo at 9.00x
- PASS: g matches Simpson at 9.49x
- PASS: g within 4 SE of Monte Carlo at 9.49x
- PASS: g matches Simpson at -1.00x
- PASS: g within 4 SE of Monte Carlo at -1.00x
- PASS: g matches Simpson at -3.00x
- PASS: g within 4 SE of Monte Carlo at -3.00x
- PASS: g matches Simpson at -10.00x
- PASS: g within 4 SE of Monte Carlo at -10.00x
- PASS: optimum matches independent search
- PASS: analytic g'' < 0 everywhere tested
- PASS: quadrature tail cap never acts where the S6 score is unclamped
- PASS: implementation concave wherever the score is unclamped
- PASS: a crash at the ruin size leaves exactly the 5% death line
- PASS: floor matches independent value
- PASS: score never rises beyond the optimum
- PASS: score never rises as a short grows
- PASS: score rises from 0 to the optimum
- PASS: worked examples

**All 36 checks pass.**
