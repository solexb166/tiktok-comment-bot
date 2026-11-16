"""Non-learning reference policies (null distribution / sanity baselines). Never given to agents."""
from __future__ import annotations

import numpy as np

from ..interface.actions import ActionMapper


class FlatPolicy:
    def __init__(self, n: int):
        self.n = n

    def __call__(self, obs):
        return np.zeros(self.n + 1)


class ConstantPolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=float)

    def __call__(self, obs):
        return self.action


class RandomPolicy:
    def __init__(self, n: int, seed: int):
        self.n = n
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs):
        return self.rng.uniform(-1.0, 1.0, self.n + 1)


class HoldPolicy:
    """Submits `initial_action` until every instrument it targets holds a position (orders are
    rejected while a market is closed), then re-outputs its current exposure (true buy-and-hold:
    the position is not rebalanced as equity changes)."""

    def __init__(self, initial_action, mapper: ActionMapper):
        self.initial = np.asarray(initial_action, dtype=float)
        self.mapper = mapper
        self.targets = np.flatnonzero(self.initial[:mapper.n] != 0.0)
        self.started = False

    def __call__(self, obs):
        if not self.started:
            held = np.asarray(obs.exposure, dtype=float)[self.targets] != 0.0
            if not held.all():
                return self.initial
            self.started = True
        w = np.asarray(obs.exposure, dtype=float)
        G = float(np.sum(np.abs(w)))
        if G == 0.0:
            return np.zeros(self.mapper.size)
        return np.r_[w / G, self.mapper.u_for_gross(G)]


class ScriptedPolicy:
    """Returns actions from a {decision_number: action} script; holds the last action otherwise."""

    def __init__(self, script: dict, n: int, default=None):
        self.script, self.k = script, 0
        self.last = np.zeros(n + 1) if default is None else np.asarray(default, float)

    def __call__(self, obs):
        if self.k in self.script:
            self.last = np.asarray(self.script[self.k], float)
        self.k += 1
        return self.last
