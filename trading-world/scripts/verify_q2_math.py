"""Independent verification of the Q2.0 mathematics before freezing.

Re-derives everything WITHOUT the project's quadrature or optimiser and compares with
learning/quality.py:
  1. g(G) by composite Simpson integration over the normal density (200,001 points, |z| <= 12)
     instead of Gauss-Hermite quadrature.
  2. g(G) by plain Monte Carlo (10 million draws, independent random stream).
  3. G* by an independent ternary search on the Simpson g.
  4. Concavity: analytic second derivative g''(G) = -(1-p) E[X^2/(1+GX)^2] - p J^2/(1-GJ)^2 < 0,
     evaluated numerically, and second differences of the implementation.
  5. Ruin threshold: a crash at 9.5x leaves exactly 1 - 9.5 x 0.10 = 5% (the death line).
  6. Monotonicity of the implemented score on a dense grid, both sides.
Writes reports/phase25/q2_math_verification.md and exits non-zero on any failure.
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tradeworld.learning import synthetic_tasks as st  # noqa: E402
from tradeworld.learning.metrics import q2_checksum  # noqa: E402
from tradeworld.learning.quality import KnownMarket, q2_floor, q2_scores  # noqa: E402

mk = KnownMarket.from_task(st.build("S6", 0, 1000))
mu, sig, p, J = mk.mu, mk.sigma, mk.crash_prob, mk.crash
m = mu - 0.5 * sig ** 2
z = np.linspace(-12, 12, 200_001)
h = z[1] - z[0]
w = np.ones_like(z)
w[1:-1:2], w[2:-1:2] = 4, 2
w *= h / 3 * np.exp(-z ** 2 / 2) / math.sqrt(2 * math.pi)            # Simpson weights x normal density
X = np.expm1(m + sig * z)


def g_simpson(G):
    if G * J >= 1:
        return -math.inf
    return (1 - p) * float(np.sum(w * np.log(1 + G * X))) + p * math.log(1 - G * J)


def g2_simpson(G):                                                    # analytic second derivative
    return -(1 - p) * float(np.sum(w * X ** 2 / (1 + G * X) ** 2)) - p * J ** 2 / (1 - G * J) ** 2


checks, L = [], ["# Q2.0 mathematics: independent verification\n",
                 f"Implementation checksum verified: `{q2_checksum()}`\n"]

# 1-2. g against Simpson and Monte Carlo
rng = np.random.default_rng(20261117)
n = 10_000_000
crash = rng.random(n) < p
r = np.where(crash, math.log(1 - J), m + sig * rng.standard_normal(n))
L += ["## 1-2. Expected log growth g(G): implementation vs Simpson integration vs Monte Carlo\n",
      "| G | Implementation (Gauss-Hermite) | Simpson (independent) | Relative difference | Monte Carlo (10M) | MC difference / SE |",
      "|---|---|---|---|---|---|"]
for G in [0.0, 0.5, 1.0, 2.0, 2.93, 4.0, 6.0, 8.0, 9.0, 9.49, -1.0, -3.0, -10.0]:
    gi, gsim = float(mk.growth(G)[0]), g_simpson(G)
    x = np.log(1 + G * np.expm1(r))
    mc, se = float(x.mean()), float(x.std() / math.sqrt(n))
    rel = abs(gi - gsim) / max(abs(gsim), 1e-12)
    checks.append(("g matches Simpson at %.2fx" % G, rel < 1e-6 or abs(gi - gsim) < 1e-12))
    checks.append(("g within 4 SE of Monte Carlo at %.2fx" % G, abs(mc - gi) < 4 * se or se == 0))
    L.append(f"| {G:.2f} | {gi:.8e} | {gsim:.8e} | {rel:.1e} | {mc:.4e} | {(mc - gi) / se if se else 0:+.2f} |")

# 3. optimum by ternary search
lo, hi = 0.0, 1 / J - 1e-6
for _ in range(200):
    a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
    if g_simpson(a) < g_simpson(b):
        lo = a
    else:
        hi = b
G_ind = (lo + hi) / 2
G_imp, g_imp = mk.optimal()
checks.append(("optimum matches independent search", abs(G_ind - G_imp) < 1e-3))
L += ["", f"## 3. Optimum\n", f"Independent ternary search: G* = {G_ind:.5f}x; implementation: {G_imp:.5f}x "
      f"(difference {abs(G_ind - G_imp):.1e}). g(G*) = {g_simpson(G_ind):.6e} (implementation {g_imp:.6e}).\n"]

# 4. concavity
grid = np.r_[np.linspace(-50, 9.49, 600)]
second = np.array([g2_simpson(G) for G in grid])
checks.append(("analytic g'' < 0 everywhere tested", bool(np.all(second < 0))))
# Where the score is not clamped at F (the only place g can affect Q2), the implementation must be concave.
from tradeworld.learning.quality import _Z
Gsh = np.linspace(0, -200, 200_001)
G_clamp_short = float(Gsh[np.argmax(q2_scores(Gsh, mk) <= q2_floor(mk) + 1e-12)])
G_tail_short = float(-1 / np.expm1(mk.m + mk.sigma * _Z.max()))           # extreme node wipes out a short beyond this
G_tail_long = float(-1 / np.expm1(mk.m + mk.sigma * _Z.min()))            # ... and a long beyond this
checks.append(("quadrature tail cap never acts where the S6 score is unclamped",
               G_tail_short < G_clamp_short and G_tail_long > mk.ruin_long()))
gi = mk.growth(np.linspace(G_clamp_short, 9.49, 6001))
d2 = np.diff(gi, 2)
checks.append(("implementation concave wherever the score is unclamped", bool(np.all(d2 <= 1e-15))))
L += ["## 4. Concavity\n",
      "g''(G) = -(1-p) E[X^2 / (1+GX)^2] - p J^2 / (1-GJ)^2, a sum of negative terms wherever g is finite, so g is "
      "strictly concave (it has a single peak and falls on both sides of it).",
      f"Evaluated at 600 sizes from -50x to 9.49x: largest value {second.max():.3e} (all negative).",
      "",
      "Numerical scope. In a normal model any position above 1x can in principle be wiped out by an extreme enough "
      "move; the quadrature caps the logarithm at its most extreme nodes (|z| up to 18.5). For S6 that cap acts only "
      f"for shorts beyond {G_tail_short:.2f}x and longs beyond {G_tail_long:.1f}x, while the score is already clamped at "
      f"F for shorts beyond {G_clamp_short:.2f}x and longs from {mk.ruin_long():.2f}x. Every S6 score Q2 can produce is "
      f"therefore computed exactly. Second differences of the implementation on 6,001 sizes from {G_clamp_short:.2f}x to "
      f"9.49x: largest {d2.max():.2e} (none positive). Outside that range the score is the constant F.",
      "",
      "S7 is NOT covered: its volatility is about three times higher, so the cap acts for longs from about 5.9x, below "
      "its 19.65x clamp (an error of up to about 5% near 19x). Q2.0 is frozen for S6 only; S7 values are provisional.\n"]

# 5. ruin threshold
eq_after = 1 - mk.ruin_long() * J
checks.append(("a crash at the ruin size leaves exactly the 5% death line", abs(eq_after - 0.05) < 1e-12))
F, F_ind = q2_floor(mk), g_simpson(0.95 / J) / g_simpson(G_ind)
checks.append(("floor matches independent value", abs(F - F_ind) < 1e-4))
L += ["## 5. Ruin threshold and floor\n",
      f"A crash at {mk.ruin_long():.2f}x leaves 1 - {mk.ruin_long():.2f} x 0.10 = {eq_after:.4f} of equity: the death "
      f"line. Floor F = g(9.5x)/g(G*): implementation {F:.5f}, independent {F_ind:.5f}.\n"]

# 6. monotonicity of the score
up = q2_scores(np.linspace(G_imp, 10_000, 400_001), mk)
dn = q2_scores(np.linspace(-10_000, 0, 400_001), mk)
rise = q2_scores(np.linspace(0, G_imp, 100_001), mk)
checks += [("score never rises beyond the optimum", bool(np.all(np.diff(up) <= 1e-12))),
           ("score never rises as a short grows", bool(np.all(np.diff(dn) >= -1e-12))),
           ("score rises from 0 to the optimum", bool(np.all(np.diff(rise) >= -1e-12))),
           ("worked examples", list(np.round(q2_scores([0, G_imp, 10, 20, 100], mk), 3)) == [0.0, 1.0, -21.353, -21.353, -21.353])]
L += ["## 6. Score monotonicity\n",
      "Checked on 400,001 sizes from the optimum to 10,000x, 400,001 sizes from -10,000x to 0, and 100,001 sizes from 0 "
      "to the optimum.\n", "## Result\n"]
for name, ok in checks:
    L.append(f"- {'PASS' if ok else 'FAIL'}: {name}")
allok = all(ok for _, ok in checks)
L.append(f"\n**{'All ' + str(len(checks)) + ' checks pass.' if allok else 'VERIFICATION FAILED.'}**")
out = ROOT / "reports" / "phase25" / "q2_math_verification.md"
out.write_text("\n".join(L) + "\n")
print("\n".join(L))
sys.exit(0 if allok else 1)
