"""World: N isolated agent accounts on one shared replay clock.

Isolation: each agent has its own Account, RNG stream (derived from the world seed and the
agent's id, not its position in the list), action mapper and policy. Agents receive only an
Observation; the world, other accounts and the data store are never exposed to them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..accounts.account import Account
from ..brokerspec.profile import EnvProfile
from ..data.market_data import MarketData
from ..execution.engine import MarketView, execute, process_bar
from ..interface.actions import ActionMapper
from ..interface.observation import build_observation
from ..replay.replay import Replay
from ..tracking.hashing import event_log_hash, stable_int

RECORD_FIELDS = (
    "t", "agent", "status", "equity", "balance", "action_raw", "d", "u", "G", "w_target",
    "w_executed", "gross_leverage", "net_leverage", "max_gross_leverage_interval", "margin_used",
    "free_margin", "margin_level", "rejections", "safeguards", "cost_spread", "cost_slippage",
    "cost_commission", "cost_swap",
)


@dataclass
class AgentSpec:
    agent_id: str
    policy: Callable
    mapper: ActionMapper | None = None


@dataclass
class _Slot:
    spec: AgentSpec
    acct: Account
    mapper: ActionMapper
    last_report: object = None
    last_row: dict | None = None
    invalid_streak: int = 0


@dataclass
class WorldResult:
    accounts: dict
    records: pd.DataFrame
    profile: EnvProfile

    def event_hash(self, agent_id: str) -> str:
        return event_log_hash(self.accounts[agent_id].events)


class World:
    def __init__(self, md: MarketData, profile: EnvProfile, agents: list[AgentSpec], *,
                 seed: int = 0, decision_interval: int = 5, obs_resolution: int = 5,
                 window: int = 120, record: bool = True, max_invalid_streak: int | None = None,
                 view: MarketView | None = None, replay: Replay | None = None):
        """`view`/`replay` may be passed in to share pre-built, read-only market structures
        between many worlds on the same data (e.g. training episodes); they must match `md`."""
        ids = [a.agent_id for a in agents]
        if len(set(ids)) != len(ids):
            raise ValueError("agent ids must be unique")
        self.view = view if view is not None else MarketView(md, profile)
        self.replay = replay if replay is not None else Replay(md, decision_interval, obs_resolution, window)
        if self.view.md is not md or self.replay.md is not md:
            raise ValueError("shared view/replay must be built on the same MarketData")
        if (self.replay.k, self.replay.r, self.replay.W) != (decision_interval, obs_resolution, window):
            raise ValueError("shared replay settings differ from decision_interval/obs_resolution/window")
        self.profile, self.record, self.max_invalid_streak = profile, record, max_invalid_streak
        n = profile.universe.n
        self.slots = []
        for a in agents:
            rng = np.random.Generator(np.random.PCG64(
                np.random.SeedSequence(entropy=seed, spawn_key=(stable_int(a.agent_id),))))
            self.slots.append(_Slot(a, Account(a.agent_id, profile, rng), a.mapper or ActionMapper.primary(n)))
        self.rows: list[dict] = []

    def decisions(self, start: int | None = None, stop: int | None = None) -> np.ndarray:
        """M1 indices of the decisions whose index lies in [start, stop)."""
        dec = self.replay.decision_indices
        if start is not None:
            dec = dec[dec >= start]
        if stop is not None:
            dec = dec[dec < stop]
        return dec

    def step(self, dec: np.ndarray, k: int) -> int:
        """Decision k of schedule `dec` for every live agent, then process the M1 bars up to and
        including the next decision bar. Returns the last processed M1 index."""
        d = int(dec[k])
        for s in self.slots:
            if s.acct.alive:
                self._decide(s, d)
        nxt = int(dec[k + 1]) if k + 1 < len(dec) else min(d + self.replay.k, self.view.md.T - 1)
        for j in range(d + 1, nxt + 1):
            for s in self.slots:
                if s.acct.alive and s.acct.has_positions:
                    process_bar(s.acct, self.view, j)
        return nxt

    def result(self) -> WorldResult:
        for s in self.slots:
            self._close_interval(s)
        return WorldResult({s.spec.agent_id: s.acct for s in self.slots},
                           pd.DataFrame(self.rows, columns=RECORD_FIELDS), self.profile)

    def run(self, start: int | None = None, stop: int | None = None) -> WorldResult:
        """Run decisions whose M1 index lies in [start, stop)."""
        dec = self.decisions(start, stop)
        for k in range(len(dec)):
            self.step(dec, k)
        return self.result()

    # ------------------------------------------------------------------
    def _close_interval(self, s: _Slot) -> None:
        if s.last_row is not None:
            s.last_row["max_gross_leverage_interval"] = s.acct.max_gross_interval

    def _decide(self, s: _Slot, d: int) -> None:
        acct, view = s.acct, self.view
        self._close_interval(s)
        obs = build_observation(self.replay, view, acct, d, s.last_report)
        equity_dec = acct.equity(view.prices_close(d))
        raw = s.spec.policy(obs)
        m = s.mapper.map(raw)
        safeguards = list(m.safeguards)
        rep = None
        if m.valid:
            s.invalid_streak = 0
            rep = execute(acct, view, m.w, d)
            safeguards += [f"{k}:{acct.u.symbols[i]}" for k, i in rep.safeguards]
        else:
            s.invalid_streak += 1
            acct._event(int(view.times[d + 1]), "invalid_action")
            if self.max_invalid_streak and s.invalid_streak >= self.max_invalid_streak:
                raise RuntimeError(f"agent {s.spec.agent_id}: {s.invalid_streak} consecutive invalid actions")
        s.last_report = rep
        # After-execution account state, valued at the fill bar's open.
        po = view.prices_open(d + 1)
        eq = acct.equity(po)
        notional = acct.units * po.mid * po.conv
        used = acct.margin_used(po, int(view.times[d + 1]), eq) if acct.has_positions else 0.0
        acct.max_gross_interval = float(np.sum(np.abs(notional)) / eq) if eq > 0 else 0.0
        acct.min_equity_interval = eq
        if self.record:
            row = dict(
                t=int(view.times[d]) + 1, agent=s.spec.agent_id, status=acct.status,
                equity=equity_dec, balance=acct.balance,
                action_raw=np.asarray(raw, dtype=float).tolist() if np.size(raw) else [],
                d=None if m.d is None else m.d.tolist(), u=m.u, G=m.G,
                w_target=None if m.w is None else m.w.tolist(),
                w_executed=(notional / eq).tolist() if eq > 0 else [0.0] * acct.u.n,
                gross_leverage=float(np.sum(np.abs(notional)) / eq) if eq > 0 else math.inf,
                net_leverage=float(np.sum(notional) / eq) if eq > 0 else math.inf,
                max_gross_leverage_interval=None, margin_used=used, free_margin=eq - used,
                margin_level=math.inf if used <= 0 else eq / used,
                rejections=";".join(f"{acct.u.symbols[i]}:{r}" for i, r in sorted(rep.rejections.items())) if rep else "",
                safeguards=";".join(safeguards),
                cost_spread=acct.costs["spread"], cost_slippage=acct.costs["slippage"],
                cost_commission=acct.costs["commission"], cost_swap=acct.costs["swap"],
            )
            self.rows.append(row)
            s.last_row = row
        for key in acct.costs:
            acct.costs[key] = 0.0
