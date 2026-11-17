"""Versioned, freezable analysis metrics and write-once recalculations.

The decision-quality score Q2 is identified by a version ("Q2.0") and the SHA-256 of its exact
implementation (the source of every function and constant it uses). Freezing writes that checksum
to the registry once; afterwards any evaluation refuses to run if the implementation differs.

Recalculations of earlier evaluations with a new metric are stored as separate write-once
analysis records (registry/analyses/<metric>/<eval_id>.json). The evaluation records, and the
scores they already contain, are never modified.
"""
from __future__ import annotations

import hashlib
import inspect

from ..registry.registry import FrozenRuleError, Registry, RegistryError, now_utc
from ..registry.store import read_json, write_once_json
from . import quality

Q2_DEFINITION = (
    "Per decision with net exposure G (position value / equity): s(G) = max(g(G) / g(G*), F), and s(G) = F for a "
    "ruinous decision. g(G) = (1 - p) E_Z[ln(1 + G (e^(m + sigma Z) - 1))] + p ln(1 - G J) is the exact expected log "
    "growth of holding G for one decision (96-point Gauss-Hermite quadrature; minus infinity if G >= 1/J). G* = argmax g. "
    "F = g(G_ruin) / g(G*), the score of the smallest ruinous long position (G_ruin = 0.95 / J in S6; for a market "
    "without crashes, the size at which a 5-sigma fall reaches the 5% death line). Ruinous: G >= G_ruin long, or "
    "|G| >= 0.95 / (e^(m + 5 sigma) - 1) short. Q2 of an episode = mean of s over its decisions; Q2 of an evaluation = "
    "mean of the episode values. Uses only the decisions and the market's known parameters, never realised returns."
)


def q2_checksum() -> str:
    parts = [inspect.getsource(quality.KnownMarket), inspect.getsource(quality.q2_floor),
             inspect.getsource(quality.q2_scores), inspect.getsource(quality.q2_metrics),
             repr(quality.DEATH), repr(len(quality._Z)), quality.Q2_VERSION]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def q2_freeze_path(reg: Registry):
    return reg.root / "metrics" / f"{quality.Q2_VERSION}.json"


def freeze_q2(reg: Registry, approved_by: str, note: str = "", scope=("S6",)) -> dict:
    rec = dict(metric="decision-quality", version=quality.Q2_VERSION, checksum=q2_checksum(),
               definition=Q2_DEFINITION, frozen_at=now_utc(), approved_by=approved_by, note=note,
               scope=list(scope),
               scope_note=("Exact for S6 (verified: reports/phase25/q2_math_verification.md). Not frozen for S7, "
                           "where the quadrature tail cap affects longs from about 5.9x; S7 values are provisional."),
               history=["q0 per-decision score with a one-step death floor: rejected (rises again above 10x)",
                        "Q1 episode replay on simulated paths with permanent death: rejected (scored 20x above 10x)",
                        "Q2.0 clamped Kelly score: frozen"])
    write_once_json(q2_freeze_path(reg), rec)
    return rec


def check_q2(reg: Registry, require_frozen: bool = False) -> dict:
    """Return the Q2 identity; refuse if it was frozen with a different implementation."""
    p = q2_freeze_path(reg)
    cs = q2_checksum()
    if p.exists():
        frozen = read_json(p)
        if frozen["checksum"] != cs:
            raise FrozenRuleError(f"decision-quality score {quality.Q2_VERSION} changed after it was frozen")
        return dict(version=frozen["version"], checksum=cs, frozen=True)
    if require_frozen:
        raise RegistryError(f"decision-quality score {quality.Q2_VERSION} is not frozen yet")
    return dict(version=quality.Q2_VERSION, checksum=cs, frozen=False)


def recalc_q2(reg: Registry, round_id: str = "gate-1", tasks=("S6", "S7"), log=print) -> list[dict]:
    """Write-once Q2 recalculation of every evaluation in a round, from its preserved decisions."""
    from . import synthetic_tasks as st
    from .evaluate import decision_log
    ident_q2 = check_q2(reg)
    out = []
    for a in reg.agents():
        i = reg.identity(a)
        if i["task"] not in tasks:
            continue
        mk = quality.KnownMarket.from_task(st.build(i["task"], 0, 1000))
        for e in reg.evaluations(a):
            if e["round_id"] != round_id:
                continue
            path = reg.root / "analyses" / f"decision-quality-{quality.Q2_VERSION}" / f"{e['eval_id']}.json"
            if path.exists():
                out.append(read_json(path))
                continue
            log_df = decision_log(reg, e)
            m = quality.q2_metrics(log_df["exposure_after_fill"].to_numpy(), mk, log_df["episode"].to_numpy())
            prev = (e.get("diagnosis") or {}).get("metrics", {})
            rec = dict(eval_id=e["eval_id"], agent_id=a, version=e["version"], round_id=round_id,
                       metric=ident_q2, computed_at=now_utc(), decision_log_fingerprint=e.get("decision_log_fingerprint"),
                       q2=m["q2"], q2_by_episode=m["q2_by_episode"], floor=m["floor"],
                       ruinous_share=float(mk.ruinous(log_df["exposure_after_fill"].to_numpy()).mean()),
                       previous_scores=dict(q_nonruinous=prev.get("dq_quality"), Q1=prev.get("dq_episode_quality"),
                                            note="as stored in the original evaluation record (unchanged)"))
            write_once_json(path, rec)
            out.append(rec)
            log(f"{i['name']}: Q2 {m['q2']:+.3f} (Q1 was {prev.get('dq_episode_quality')})")
    return out
