"""Decision and consequence recorder.

DecisionRecorder wraps a TradeWorldEnv and records one row per decision. It is read-only: it
passes actions, observations and rewards through unchanged, and reads the account and the
simulator's own event log (fills, rejections, stop-outs, deaths) before and after each step.

Observations: the full observation vector is a deterministic function of the market data, the
decision index and the account state, so each row stores its SHA-256 plus a compact market
summary. `regenerate_observation` rebuilds any observation exactly from the stored recipe
(checked against the hash) instead of storing ~200 floats per decision in the repository.
"""
from __future__ import annotations

import hashlib
import math

import gymnasium as gym
import numpy as np

from ..interface.actions import ActionMapper

POSITION_CHANGES = ("stay_flat", "open", "close", "reverse", "increase", "reduce", "hold")


def classify_change(before: float, after: float, tol: float = 1e-12) -> str:
    scale = max(abs(before), abs(after), 1e-300)
    if abs(before) <= tol and abs(after) <= tol:
        return "stay_flat"
    if abs(before) <= tol:
        return "open"
    if abs(after) <= tol:
        return "close"
    if np.sign(before) != np.sign(after):
        return "reverse"
    if abs(abs(after) - abs(before)) <= 1e-9 * scale:
        return "hold"
    return "increase" if abs(after) > abs(before) else "reduce"


class DecisionRecorder(gym.Wrapper):
    def __init__(self, env, expected_return=None):
        """`expected_return` (optional, synthetic tasks only): array over M1 rows giving the
        known expected log return of the bar after each row. Used only for diagnosis, never
        shown to the agent."""
        super().__init__(env)
        self.rows: list[dict] = []
        self.episode = -1
        self.expected_return = expected_return

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.episode += 1
        self._obs = obs
        return obs, info

    def step(self, action):
        core = self.env.unwrapped
        acct = core.account
        mapper: ActionMapper = core.mapper
        k = core.k
        d = int(core.dec[k])
        n_ev = len(acct.events)
        units0 = acct.units.copy()
        costs0 = dict(acct.costs_total)
        eq0 = core._e
        mid = core.view.prices_close(d).mid
        m = mapper.map(action)
        obs, r, te, tr, info = self.env.step(action)
        new = acct.events[n_ev:]
        d1 = int(core.dec[core.k])
        mid1 = core.view.prices_close(d1).mid
        dec_du = np.zeros_like(units0)
        for e in new:
            if e.kind == "fill" and e.data.get("reason") == "decision":
                dec_du[e.data["i"]] += e.data["du"]
        after_fill = units0 + dec_du
        po = core.view.prices_open(min(d + 1, core.md.T - 1))
        exp_fill = after_fill * po.mid * po.conv / eq0 if eq0 > 0 else np.zeros_like(units0)
        rep = core.world.slots[0].last_report
        deaths = [e for e in new if e.kind == "death"]
        a_raw = np.asarray(action, dtype=float).reshape(-1)
        dir_ = 0 if m.d is None else int(np.sign(np.round(m.d[0], 9)))
        row = dict(
            episode=self.episode, step=k + 1, decision_row=d, t=int(core.view.times[d]),
            obs_sha256=hashlib.sha256(np.ascontiguousarray(self._obs).tobytes()).hexdigest(),
            obs_last_return=float(self._obs[(core.encoder.W - 1) * core.encoder.F]),
            obs_vol_level=float(self._obs[core.encoder.n * core.encoder.W * core.encoder.F]),
            action_direction_raw=float(a_raw[0]), action_exposure_raw=float(a_raw[-1]),
            valid_action=bool(m.valid), direction=dir_, gross_selected=float(m.G) if m.G is not None else 0.0,
            target_exposure=float(m.w[0]) if m.w is not None else 0.0,
            exposure_after_fill=float(exp_fill[0]), leverage_after_fill=float(np.sum(np.abs(exp_fill))),
            position_change=";".join(classify_change(b, a) for b, a in zip(units0, after_fill)),
            equity_before=float(eq0), reward=float(r), equity_after=float(info["equity"]),
            pnl=float(info["equity"] - eq0),
            cost_spread=acct.costs_total["spread"] - costs0["spread"],
            cost_slippage=acct.costs_total["slippage"] - costs0["slippage"],
            cost_commission=acct.costs_total["commission"] - costs0["commission"],
            cost_swap=acct.costs_total["swap"] - costs0["swap"],
            price=float(mid[0]), next_price=float(mid1[0]), market_return=float(math.log(mid1[0] / mid[0])),
            expected_return=float(self.expected_return[d]) if self.expected_return is not None else float("nan"),
            net_exposure_end=float(info["net_leverage"]) if info["alive"] else 0.0,
            gross_exposure_end=float(info["gross_leverage"]) if info["alive"] else 0.0,
            rejected=";".join(f"{i}:{x}" for i, x in sorted(rep.rejections.items())) if rep else "",
            stop_outs=sum(e.kind == "stop_out" for e in new),
            died=bool(deaths), death_reason=deaths[0].data.get("reason", "") if deaths else "",
            alive_after=bool(info["alive"]),
        )
        self.rows.append(row)
        self._obs = obs
        return obs, r, te, tr, info


def regenerate_observation(env, start_decision: int, seed: int, actions: list, step: int) -> np.ndarray:
    """Rebuild the observation seen before decision `step` (1-based) of an episode, by replaying
    the recorded actions on the same market and seed."""
    o, _ = env.reset(seed=seed, options={"start_decision": int(start_decision)})
    for a in actions[: step - 1]:
        o, *_ = env.step(a)
    return o
