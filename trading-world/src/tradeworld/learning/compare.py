"""Compare an agent's versions, and agents across seeds.

Fair comparison: competing versions are compared on the SAME fresh round (identical unseen test
data). Each version's original evaluations stay as they were and are listed separately.
"""
from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd

from ..registry.registry import Registry
from .evaluate import decision_log, evaluate


def eval_row(reg: Registry, e: dict) -> dict:
    s, m = e["summary"], (e.get("diagnosis") or {}).get("metrics", {})
    row = dict(version=e["version"], round_id=e["round_id"], result=e["result"], reason=e.get("reason", ""),
               growth_pct=100 * (math.exp(s["lg"]) - 1), t=s["t"], death_rate=s["died"], mean_gross=s["gross"],
               mean_net=s["net"], turnover=s["turnover"], median_leverage=m.get("median_leverage"),
               p95_leverage=m.get("p95_leverage"), max_leverage=m.get("max_leverage"),
               share_over_fatal=m.get("share_over_fatal"), trade_frequency=m.get("trade_frequency"),
               time_in_market=m.get("time_in_market"), aligned_with_edge=m.get("aligned_with_edge"),
               legacy=bool(e.get("legacy")),
               diagnosis=",".join(o["code"] for o in (e.get("diagnosis") or {}).get("observations", [])))
    if e.get("decision_log_sha256"):
        log = decision_log(reg, e)
        dd = []
        for _, g in log.groupby("episode"):
            eq = np.r_[g["equity_before"].iloc[0], g["equity_after"].to_numpy()]
            peak = np.maximum.accumulate(eq)
            dd.append(float(np.max(1 - np.maximum(eq, 0) / peak)))
        row["max_drawdown_mean"] = float(np.mean(dd))
    return row


def compare_versions(reg: Registry, agent_id: str, round_id: str | None = None, evaluate_missing: bool = True) -> dict:
    """Original results per version, plus (if `round_id`) every version on that one round."""
    evals = reg.evaluations(agent_id)
    out = dict(agent_id=agent_id, identity=reg.identity(agent_id))
    if round_id:
        rows = []
        for v in reg.versions(agent_id):
            got = [e for e in evals if e["version"] == v["version"] and e["round_id"] == round_id]
            if not got and evaluate_missing:
                got = [evaluate(reg, agent_id, v["version"], round_id)]
            rows += [eval_row(reg, e) for e in got]
        out["same_round"] = pd.DataFrame(rows)
    original = {}
    for e in reg.evaluations(agent_id):                # each version's first evaluation, untouched
        original.setdefault(e["version"], e)
    out["original"] = pd.DataFrame([eval_row(reg, e) for e in sorted(original.values(), key=lambda e: e["version"])])
    out["state"] = reg.state(agent_id)
    out["training"] = training_progress(reg, agent_id)
    return out


def training_progress(reg: Registry, agent_id: str, bins: int = 20) -> pd.DataFrame:
    """Learning curves across all versions, on one continuous step axis."""
    frames = []
    for v in reg.versions(agent_id):
        sha = (v.get("history") or {}).get("episodes_sha256")
        if not sha:
            continue
        ep = pd.read_parquet(io.BytesIO(reg.blobs.get(sha)))
        if not len(ep):
            continue
        ep["version"] = v["version"]
        ep["bin"] = pd.cut(ep["step"], bins=min(bins, max(1, len(ep) // 5)), labels=False)
        g = ep.groupby(["version", "bin"]).agg(step=("step", "max"), growth=("growth", "mean"), died=("died", "mean"),
                                               mean_gross=("mean_gross", "mean"), max_gross=("max_gross", "mean"),
                                               drawdown=("max_drawdown", "mean"), episodes=("growth", "size"))
        frames.append(g.reset_index())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compare_seeds(reg: Registry, task: str, reward: str, round_id: str) -> pd.DataFrame:
    """Consistency across random seeds: every agent of one market/scoring rule on one round."""
    rows = []
    for a in reg.agents():
        i = reg.identity(a)
        if i["task"] != task or i["reward"] != reward:
            continue
        for e in reg.evaluations(a):
            if e["round_id"] == round_id:
                rows.append(dict(agent_id=a, seed=i["original_seed"], **eval_row(reg, e)))
    return pd.DataFrame(rows)
