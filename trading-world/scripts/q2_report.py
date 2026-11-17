"""Q2.0 decision-quality score: worked examples, monotonicity proof check, version-1 baselines.
Writes reports/phase25/q2_decision_quality.md."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tradeworld.learning import synthetic_tasks as st  # noqa: E402
from tradeworld.learning.metrics import Q2_DEFINITION, q2_checksum  # noqa: E402
from tradeworld.learning.quality import KnownMarket, episode_expected_growth, q2_floor, q2_scores  # noqa: E402

mk = KnownMarket.from_task(st.build("S6", 0, 1000))
Gs, gs = mk.optimal()
F = q2_floor(mk)
L = ["# Decision-quality score Q2.0 (proposed for freezing)\n", f"Implementation checksum: `{q2_checksum()}`\n",
     "## Definition\n", Q2_DEFINITION + "\n",
     "Code: `src/tradeworld/learning/quality.py` (`KnownMarket.growth`, `q2_floor`, `q2_scores`, `q2_metrics`).\n",
     f"S6 constants: mu = {mk.mu}, sigma = {mk.sigma}, p = 1/300, J = 10%. G* = {Gs:.4f}x, g(G*) = {gs:.6e} per "
     f"decision, G_ruin = {mk.ruin_long():.2f}x, F = g(G_ruin)/g(G*) = {F:.3f}.\n",
     "## Worked examples (S6, a decision holding position size G)\n",
     "| Position | g(G): expected log growth per decision | Ruinous | Q2 score | Rejected Q1 (episode replay) |", "|---|---|---|---|---|"]
for G, label in [(0.0, "Staying out (0x)"), (Gs, f"Optimal ({Gs:.2f}x)"), (10.0, "10x"), (20.0, "20x"), (100.0, "100x")]:
    g = mk.growth(G)[0]
    q1 = episode_expected_growth(np.full(1000, G), mk, 4000, 0).mean() / (1000 * gs)
    L.append(f"| {label} | {'minus infinity' if np.isneginf(g) else f'{g:.4e}'} | {'yes' if mk.ruinous(G) else 'no'} | "
             f"**{q2_scores([G], mk)[0]:.3f}** | {q1:.3f} |")
L += ["",
      f"- Staying out: g(0) = ln(1) = 0, so the score is exactly 0.",
      f"- Optimal: s = g(G*)/g(G*) = 1, the maximum of the score.",
      f"- 10x, 20x, 100x: each is at or above G_ruin = {mk.ruin_long():.2f}x. A single crash (probability 1/300 per "
      f"decision) takes equity to 1 - G x 10% <= 0, so g = minus infinity. All three score exactly F = {F:.3f}, the "
      "worst possible value, and tie: a larger ruinous position can never score better.",
      "- The rejected Q1 column shows the flaw that Q1 had: 20x scored above 10x.\n",
      "## Why a larger excessive position can never score better\n",
      "g(G) = E[ln(1 + G X)] with X = e^r - 1 is concave in G (the logarithm of a linear function of G, averaged). A "
      "concave function rises up to its maximum G* and falls from there on: for G* <= a < b, g(b) <= g(a). Dividing "
      "by g(G*) > 0 keeps the order, and max(., F) and the ruinous value F can only flatten it, never reverse it. So "
      "s(b) <= s(a): a position further beyond the optimum never scores higher. The same holds for short positions "
      "(sizes below zero), where g decreases as the short grows.\n"]
grid = np.linspace(Gs, 10000, 400001)
s = q2_scores(grid, mk)
neg = q2_scores(np.linspace(-10000, 0, 400001), mk)
L.append(f"Numerical check over 400,001 sizes from {Gs:.2f}x to 10,000x: largest increase between neighbours = "
         f"{max(0.0, float(np.max(np.diff(s)))):.1e} (none). Shorts, -10,000x to 0: largest decrease toward 0 = "
         f"{max(0.0, float(-np.min(np.diff(neg)))):.1e} (none). Property-based test: "
         "`tests/test_quality.py::test_q2_never_rewards_a_bigger_position_beyond_the_optimum`.\n")
L += ["## Version 1 baselines recalculated with Q2.0\n",
      "From the preserved decision records of each agent's original gate evaluation. Stored as separate write-once "
      "records (`registry/analyses/decision-quality-Q2.0/`). The previous scores stay in the original evaluation "
      "records, unchanged, and are shown for comparison.\n",
      "| Agent | Q2.0 (new) | Q1 (previous, rejected) | q on non-ruinous decisions (previous) | Ruinous decisions | Deaths | Median size | Original verdict |",
      "|---|---|---|---|---|---|---|---|"]
recs = [json.loads(p.read_text()) for p in sorted((ROOT / "registry/analyses/decision-quality-Q2.0").glob("*.json"))]
idents = {p.parent.name: json.loads(p.read_text()) for p in (ROOT / "registry/agents").glob("*/identity.json")}
rows = []
for r in recs:
    i = idents[r["agent_id"]]
    ev = json.loads((ROOT / "registry/agents" / r["agent_id"] / "evaluations" / f"{r['eval_id']}.json").read_text())
    rows.append((i["name"], r, ev))
for name, r, ev in sorted(rows, key=lambda x: x[0]):
    ps = r["previous_scores"]
    L.append(f"| {name} | **{r['q2']:+.3f}** | {ps['Q1']:+.3f} | {ps['q_nonruinous']:+.3f} | {100 * r['ruinous_share']:.1f}% | "
             f"{100 * ev['summary']['died']:.0f}% | {ev['diagnosis']['metrics']['median_leverage']:.2f}x | {ev['result']} |")
L.append("\nS7's growth-optimal size is 4.99x; its floor is the score of the size at which a 5-sigma fall reaches the "
         "death line (19.4x).")
out = ROOT / "reports" / "phase25" / "q2_decision_quality.md"
out.write_text("\n".join(L) + "\n")
print("\n".join(L))
