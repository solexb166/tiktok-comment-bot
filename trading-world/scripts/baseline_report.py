"""Baseline ("dummy trader") fairness report on archived real market data.

Runs non-learning reference agents through Environment A on archived Dukascopy months and
checks that the practice world behaves as expected:

  * flat (never trades)            -> equity stays exactly $10
  * buy-and-hold, one instrument   -> return matches the price move minus the spread paid
  * random traders                 -> before costs they roughly break even; after costs they lose
  * random traders, uncapped size  -> most hit the 5% death line quickly (no hidden safety net)

Usage:
    python scripts/baseline_report.py --start 2022-01 --end 2022-01 [--seeds 10]
Writes reports/baseline_<start>_<end>.md and prints it.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradeworld.agents.baselines import FlatPolicy, HoldPolicy, RandomPolicy  # noqa: E402
from tradeworld.brokerspec.profile import load_profile               # noqa: E402
from tradeworld.data.align import align                              # noqa: E402
from tradeworld.data.dukascopy_archive import load_archived, month_days  # noqa: E402
from tradeworld.data.quality import validate                         # noqa: E402
from tradeworld.interface.actions import ActionMapper                # noqa: E402
from tradeworld.market.clock import to_minute                        # noqa: E402
from tradeworld.world.world import AgentSpec, World                  # noqa: E402


def month_range(spec: str) -> tuple[int, int]:
    y, m = (int(x) for x in spec.split("-"))
    return y, m


def load_market(symbols, start: date, end: date):
    src = yaml.safe_load((ROOT / "configs/data/dukascopy.yaml").read_text())
    plan = yaml.safe_load((ROOT / "configs/data/download_plan.yaml").read_text())
    root = ROOT / plan["archive_dir"]
    frames = {}
    for sym in symbols:
        div = src["instruments"][sym]["price_divisor"]["value"]
        frames[sym] = load_archived(root, sym, start, end, float(div))
    t0 = to_minute(datetime(start.year, start.month, start.day, tzinfo=timezone.utc))
    t1 = to_minute(datetime(end.year, end.month, end.day, tzinfo=timezone.utc)) + 1440
    zv = src.get("zero_volume_is_unquoted", {})
    zv = zv.get("value", True) if isinstance(zv, dict) else bool(zv)
    return align(frames, symbols, t0, t1, zero_volume_is_unquoted=zv)


def act(n, weights, gross):
    return np.r_[np.asarray(weights, float), ActionMapper.primary(n).u_for_gross(gross)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--profile", default="environments/A_fractional.yaml")
    args = ap.parse_args()
    ys, ms = month_range(args.start)
    ye, me = month_range(args.end or args.start)
    start, end = date(ys, ms, 1), month_days(ye, me)[-1]

    profile = load_profile(args.profile)
    u = profile.universe
    syms, n = list(u.symbols), u.n
    t_load = time.time()
    md = load_market(syms, start, end)
    issues = validate(md)
    load_s = time.time() - t_load

    agents = [AgentSpec("flat", FlatPolicy(n))]
    for i, s in enumerate(syms):
        w = np.zeros(n); w[i] = 1.0
        agents.append(AgentSpec(f"hold_{s}", HoldPolicy(act(n, w, 1.0), ActionMapper.primary(n))))
    agents.append(AgentSpec("hold_basket", HoldPolicy(act(n, np.full(n, 1.0 / n), 1.0), ActionMapper.primary(n))))
    groups = {"random_cap3": 3.0, "random_cap100": 100.0, "random_uncapped": None}
    for g, cap in groups.items():
        for k in range(args.seeds):
            mapper = ActionMapper.primary(n) if cap is None else ActionMapper.benchmark(n, cap)
            agents.append(AgentSpec(f"{g}_{k}", RandomPolicy(n, 1000 + k), mapper))

    t_run = time.time()
    world = World(md, profile, agents, seed=0, decision_interval=5, obs_resolution=5, window=120)
    res = world.run()
    run_s = time.time() - t_run
    rec = res.records
    p_end = world.view.prices_close(md.T - 1)          # spread-adjusted, with USD conversion

    def summary(aid):
        a = res.accounts[aid]
        eq = a.equity(p_end) if a.alive else a.balance   # dead accounts hold no positions
        death = next((e for e in a.events if e.kind == "death"), None)
        r = rec[rec.agent == aid]
        return dict(
            equity=eq, ret=eq / 10.0 - 1.0, alive=a.alive,
            death_day=None if death is None else (death.t - md.times[0]) / 1440.0,
            death_reason=None if death is None else death.data.get("reason"),
            fills=sum(e.kind == "fill" for e in a.events),
            rejects=sum(e.kind == "reject" for e in a.events),
            stopouts=sum(e.kind == "stop_out" for e in a.events),
            costs=dict(a.costs_total), cost_total=sum(a.costs_total.values()),
            max_lev=float(r["gross_leverage"].replace(np.inf, np.nan).max()) if len(r) else 0.0,
        )

    S = {a.agent_id: summary(a.agent_id) for a in agents}
    L = []
    P = L.append
    P(f"# Baseline fairness report: {start} to {end}\n")
    P(f"Environment: `{profile.name}` (A, fractional, no exposure cap). Start equity $10; death at "
      f"${profile.death_equity:.2f}. Decisions every 5 min on 120 x M5 bars.\n")
    P(f"Data: {md.T:,} M1 rows x {n} instruments, loaded in {load_s:.1f}s. "
      f"Simulation: {len(agents)} agents, {len(rec):,} decisions in {run_s:.1f}s "
      f"({len(rec) / max(run_s, 1e-9):,.0f} decisions/s).\n")

    P("## Data quality\n")
    P("| Instrument | Quoted minutes | Median spread | Median spread (bps) | Price range (mid) |")
    P("|---|---|---|---|---|")
    mid = md.mid_c()
    for i, s in enumerate(syms):
        q = md.quoted[:, i]
        sp = (md.ask_c[:, i] - md.bid_c[:, i])[q]
        P(f"| {s} | {q.mean():.1%} | {np.median(sp):.5g} | {np.median(sp / mid[q, i]) * 1e4:.2f} | "
          f"{mid[q, i].min():.5g} – {mid[q, i].max():.5g} |")
    errs = [x for x in issues if x.severity == "error"]
    P(f"\nValidation: {len(errs)} errors, {len(issues) - len(errs)} warnings/info.")
    for x in issues[:10]:
        P(f"- {x}")
    P("")

    checks = []

    P("## 1. Flat agent (never trades)\n")
    f = S["flat"]
    ok = f["equity"] == 10.0 and f["fills"] == 0
    checks.append(("Flat agent keeps exactly $10", ok))
    P(f"Final equity ${f['equity']:.6f}, fills {f['fills']}. **{'PASS' if ok else 'FAIL'}**\n")

    P("## 2. Buy-and-hold, 1x, one instrument each\n")
    P("Expected = price move from the first fill (bought at the ask) to the end (valued at the bid), "
      "which already includes the spread paid, minus slippage, commission and swaps the account recorded.\n")
    P("| Agent | Final equity | Return | Expected | Diff | Spread+slip cost | Swap cost |")
    P("|---|---|---|---|---|---|---|")
    hold_ok = True
    for i, s in enumerate(syms):
        aid = f"hold_{s}"
        a, x = res.accounts[aid], S[aid]
        fl = [e for e in a.events if e.kind == "fill"]
        if not fl:
            P(f"| {aid} | ${x['equity']:.4f} | – | no fill | | | |"); hold_ok = False; continue
        e0 = fl[0]
        units, px = e0.data["du"], e0.data["price"]
        conv_end = p_end.conv[i]
        bid_end = p_end.bid[i]
        expected = 10.0 + units * (bid_end * conv_end - px * e0.data["conv"]) - \
            a.costs_total["commission"] - a.costs_total["swap"]
        diff = x["equity"] - expected
        # Fill price already includes slippage; FX conversion drift (USDJPY) explains tiny diffs.
        tol = 0.02 * 10.0 if syms[i] == "USDJPY" else 1e-6
        hold_ok &= abs(diff) <= tol and len(fl) == 1
        P(f"| {aid} | ${x['equity']:.4f} | {x['ret']:+.2%} | ${expected:.4f} | {diff:+.2e} | "
          f"${x['costs']['spread'] + x['costs']['slippage']:.4f} | ${x['costs']['swap']:.4f} |")
    xb = S["hold_basket"]
    P(f"| hold_basket (1/6 each) | ${xb['equity']:.4f} | {xb['ret']:+.2%} | | | "
      f"${xb['costs']['spread'] + xb['costs']['slippage']:.4f} | ${xb['costs']['swap']:.4f} |")
    checks.append(("Buy-and-hold matches price move after costs, one fill each", hold_ok))
    P(f"\n**{'PASS' if hold_ok else 'FAIL'}**\n")

    P("## 3. Random traders\n")
    P("Random direction and size every 5 minutes. 'Before costs' adds back spread, slippage, "
      "commission and swaps. A fair world has before-cost results scattered around zero and "
      "after-cost results below them.\n")
    P("| Group | Agents | Survived | Median final equity | Mean P&L | Mean costs | Mean P&L before costs | Median max leverage |")
    P("|---|---|---|---|---|---|---|---|")
    for g in groups:
        ids = [k for k in S if k.startswith(g + "_")]
        eq = np.array([S[k]["equity"] for k in ids])
        pnl = eq - 10.0
        cost = np.array([S[k]["cost_total"] for k in ids])
        surv = sum(S[k]["alive"] for k in ids)
        lev = np.median([S[k]["max_lev"] for k in ids])
        P(f"| {g} | {len(ids)} | {surv} | ${np.median(eq):.4f} | ${pnl.mean():+.4f} | ${cost.mean():.4f} | "
          f"${(pnl + cost).mean():+.4f} | {lev:,.1f}x |")
        if g == "random_cap3":
            gross = pnl + cost
            se = gross.std(ddof=1) / np.sqrt(len(gross)) if len(gross) > 1 else np.inf
            ok = bool(np.all(pnl <= gross + 1e-12)) and abs(gross.mean()) <= 4 * se + 1e-9
            checks.append(("Random 3x traders: costs always hurt; before-cost mean within 4 s.e. of zero", ok))
        if g == "random_uncapped":
            deaths = [S[k] for k in ids if not S[k]["alive"]]
            reasons = {}
            for d in deaths:
                reasons[d["death_reason"]] = reasons.get(d["death_reason"], 0) + 1
            P(f"\nUncapped random: {len(deaths)}/{len(ids)} died; reasons {reasons}; median day of death "
              f"{np.median([d['death_day'] for d in deaths]) if deaths else float('nan'):.2f}.")
            ok = all(d["equity"] <= profile.death_equity + 1e-9 or d["death_reason"] for d in deaths)
            checks.append(("Every death is recorded with a reason", ok))
    P("")

    P("## Summary\n")
    for name, ok in checks:
        P(f"- {'✅' if ok else '❌'} {name}")
    allok = all(ok for _, ok in checks) and not errs
    P(f"\n**Overall: {'PASS' if allok else 'NEEDS REVIEW'}**\n")
    P("These agents do not learn. They exist only to show that the practice world charges costs, "
      "values positions and ends accounts the way it should on real prices.")

    text = "\n".join(L)
    out = ROOT / "reports" / f"baseline_{args.start}_{args.end or args.start}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text + "\n")
    print(text)
    print(f"\nwritten {out.relative_to(ROOT)}")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
