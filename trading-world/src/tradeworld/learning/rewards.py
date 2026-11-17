"""Reward variants (architecture §8.1). All are computed from net-of-cost account equity only.

Never: trade-count bonuses, win-rate rewards, market-presence rewards, realized-only rewards.

R1  eps-floored log-wealth growth            log(max(E1, eps*E0)) - log(max(E0', eps*E0))
R2  linear equity change (negative control)  (E1 - E0') / E0
R3  R1 minus lambda * increase in log-drawdown
R4  differential Sharpe ratio (Moody & Saffell 1998) on simple per-step returns
R5  sparse terminal log-wealth                0 until the episode ends, then log(max(E_T, eps*E0) / E0)
"""
from __future__ import annotations

import math


class Reward:
    name = "base"

    def reset(self, e0: float) -> None:
        self.e0 = float(e0)

    def __call__(self, e_prev: float, e_now: float, done: bool) -> float:
        raise NotImplementedError

    def _floor(self, e: float, eps: float) -> float:
        return max(e, eps * self.e0)


class LogWealth(Reward):
    name = "R1"

    def __init__(self, eps: float = 0.01):
        self.eps = eps

    def __call__(self, e_prev, e_now, done):
        return math.log(self._floor(e_now, self.eps)) - math.log(self._floor(e_prev, self.eps))


class Linear(Reward):
    name = "R2"

    def __call__(self, e_prev, e_now, done):
        return (e_now - e_prev) / self.e0


class LogWealthDrawdown(Reward):
    name = "R3"

    def __init__(self, lam: float = 1.0, eps: float = 0.01):
        self.lam, self.eps = lam, eps

    def reset(self, e0):
        super().reset(e0)
        self.peak = math.log(e0)
        self.dd = 0.0

    def __call__(self, e_prev, e_now, done):
        lw_now = math.log(self._floor(e_now, self.eps))
        r = lw_now - math.log(self._floor(e_prev, self.eps))
        self.peak = max(self.peak, lw_now)
        dd = self.peak - lw_now
        out = r - self.lam * max(0.0, dd - self.dd)
        self.dd = dd
        return out


class DifferentialSharpe(Reward):
    name = "R4"

    def __init__(self, eta: float = 0.01, clip: float = 10.0):
        self.eta, self.clip = eta, clip

    def reset(self, e0):
        super().reset(e0)
        self.A = 0.0
        self.B = 0.0

    def __call__(self, e_prev, e_now, done):
        R = (e_now - e_prev) / e_prev if e_prev > 0 else -1.0
        dA, dB = R - self.A, R * R - self.B
        var = self.B - self.A ** 2
        D = 0.0 if var <= 1e-12 else (self.B * dA - 0.5 * self.A * dB) / var ** 1.5
        self.A += self.eta * dA
        self.B += self.eta * dB
        return float(max(-self.clip, min(self.clip, D)))


class SparseTerminal(Reward):
    name = "R5"

    def __init__(self, eps: float = 0.01):
        self.eps = eps

    def __call__(self, e_prev, e_now, done):
        return math.log(self._floor(e_now, self.eps) / self.e0) if done else 0.0


REWARDS = {"R1": LogWealth, "R2": Linear, "R3": LogWealthDrawdown, "R4": DifferentialSharpe,
           "R5": SparseTerminal}


def make_reward(name: str, **kwargs) -> Reward:
    try:
        return REWARDS[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown reward {name!r}; choose from {sorted(REWARDS)}") from None
