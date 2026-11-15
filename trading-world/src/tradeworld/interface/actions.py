"""Uncapped action space (architecture §5.2, ablation X1).

action = [d_1..d_n, u], each in [-1, 1]:
    G(u)  = 10 ** (log10_min + (u + 1) / 2 * (log10_max - log10_min))   gross exposure level
    w_i   = G(u) * d_i / max(1, sum|d_j|)                                 target exposure / equity
Primary experiments: log10 range [-4, 4] -> 0.0001x ... 1x (u=0) ... 10,000x. This is a numerical
representational range, far above any margin-feasible exposure; it is NOT a behavioural cap.
Benchmark experiments set log10_max = log10(cap) for caps 3, 10, 50, 100 (range rescaled, not clipped).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CEILING_FLAG_U = 0.95


@dataclass(frozen=True)
class MappedAction:
    valid: bool
    w: np.ndarray | None
    d: np.ndarray | None = None
    u: float | None = None
    G: float | None = None
    safeguards: tuple = ()


@dataclass(frozen=True)
class ActionMapper:
    n: int
    log10_min: float = -4.0
    log10_max: float = 4.0
    cap: float | None = None            # set only for benchmark experiments

    @classmethod
    def primary(cls, n: int) -> "ActionMapper":
        return cls(n)

    @classmethod
    def benchmark(cls, n: int, cap: float) -> "ActionMapper":
        if cap <= 0:
            raise ValueError("cap must be > 0")
        return cls(n, log10_max=math.log10(cap), cap=float(cap))

    @property
    def size(self) -> int:
        return self.n + 1

    def gross(self, u: float) -> float:
        return 10.0 ** (self.log10_min + (u + 1.0) / 2.0 * (self.log10_max - self.log10_min))

    def u_for_gross(self, G: float) -> float:
        """Inverse of gross(): the u that produces gross level G (clipped to [-1, 1])."""
        x = (math.log10(G) - self.log10_min) / (self.log10_max - self.log10_min) * 2.0 - 1.0
        return float(min(1.0, max(-1.0, x)))

    def map(self, action) -> MappedAction:
        try:
            a = np.asarray(action, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return MappedAction(False, None, safeguards=("invalid_action",))
        if a.shape != (self.size,) or not np.all(np.isfinite(a)):
            return MappedAction(False, None, safeguards=("invalid_action",))
        flags = []
        if np.any(np.abs(a) > 1.0 + 1e-9):
            flags.append("clipped_action")
        a = np.clip(a, -1.0, 1.0)
        d, u = a[:-1], float(a[-1])
        G = self.gross(u)
        w = G * d / max(1.0, float(np.sum(np.abs(d))))
        if u > CEILING_FLAG_U:
            flags.append("near_ceiling")
        return MappedAction(True, w, d, u, G, tuple(flags))
