"""Uncapped action space (architecture §5.2, ablation X1).

action = [d_1..d_n, u], each in [-1, 1]:
    G(u)  = 10 ** (log10_min + (u + 1) / 2 * (log10_max - log10_min))   gross exposure level
    w_i   = G(u) * d_i / max(1, sum|d_j|)                                 target exposure / equity
Primary experiments: log10 range [-4, 4] -> 0.0001x ... 1x (u=0) ... 10,000x. This is a numerical
representational range, far above any margin-feasible exposure; it is NOT a behavioural cap.
Benchmark experiments set log10_max = log10(cap) for caps 3, 10, 50, 100 (range rescaled, not clipped).

Optional size representations (Phase 3, new experimental agents only; proposal 0006, option C).
The last action component is then a size output z instead of u; direction is unchanged:
    size_repr="log10"     G = 10 ** (...)  as above (default; every existing agent)
    size_repr="exp"       G = e^z,          z in [-25, 25]    (unit-gain log scale: +0.25 in z = x1.28)
    size_repr="softplus"  G = ln(1 + e^z),  z in [-25, 1e4]   (near-linear: +0.25 in z = +0.25x)
The z ranges are numerical safeguards only (e^25 = 7e10x; softplus reaches 10,000x, the same
largest size as the default range). They are far above any margin-feasible exposure: any
positive size the account can hold remains reachable. No exposure limit is imposed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CEILING_FLAG_U = 0.95
SIZE_REPRS = ("log10", "exp", "softplus")
Z_RANGE = {"exp": (-25.0, 25.0), "softplus": (-25.0, 1.0e4)}


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
    size_repr: str = "log10"            # see module docstring; "log10" for every existing agent

    def __post_init__(self):
        if self.size_repr not in SIZE_REPRS:
            raise ValueError(f"unknown size representation {self.size_repr!r}")

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

    @property
    def size_bounds(self) -> tuple[float, float]:
        return (-1.0, 1.0) if self.size_repr == "log10" else Z_RANGE[self.size_repr]

    @property
    def low(self) -> np.ndarray:
        return np.r_[np.full(self.n, -1.0), self.size_bounds[0]].astype(np.float32)

    @property
    def high(self) -> np.ndarray:
        return np.r_[np.full(self.n, 1.0), self.size_bounds[1]].astype(np.float32)

    def gross(self, u: float) -> float:
        if self.size_repr == "exp":
            return math.exp(u)
        if self.size_repr == "softplus":
            return float(np.logaddexp(0.0, u))
        return 10.0 ** (self.log10_min + (u + 1.0) / 2.0 * (self.log10_max - self.log10_min))

    def u_for_gross(self, G: float) -> float:
        """Inverse of gross(): the size output that produces gross level G (clipped to its range)."""
        lo, hi = self.size_bounds
        if self.size_repr == "exp":
            x = math.log(G)
        elif self.size_repr == "softplus":
            x = G + math.log(-math.expm1(-G))               # ln(e^G - 1), stable for large G
        else:
            x = (math.log10(G) - self.log10_min) / (self.log10_max - self.log10_min) * 2.0 - 1.0
        return float(min(hi, max(lo, x)))

    def map(self, action) -> MappedAction:
        try:
            a = np.asarray(action, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return MappedAction(False, None, safeguards=("invalid_action",))
        if a.shape != (self.size,) or not np.all(np.isfinite(a)):
            return MappedAction(False, None, safeguards=("invalid_action",))
        if self.size_repr != "log10":
            return self._map_z(a)
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

    def _map_z(self, a: np.ndarray) -> MappedAction:
        """Optional size representations: direction in [-1, 1], size output z in its range."""
        lo, hi = self.size_bounds
        flags = []
        if np.any(np.abs(a[:-1]) > 1.0 + 1e-9) or not (lo - 1e-9 <= a[-1] <= hi + 1e-9):
            flags.append("clipped_action")
        d = np.clip(a[:-1], -1.0, 1.0)
        z = float(min(hi, max(lo, a[-1])))
        G = self.gross(z)
        w = G * d / max(1.0, float(np.sum(np.abs(d))))
        if z > hi - 0.05 * (hi - lo):
            flags.append("near_ceiling")
        return MappedAction(True, w, d, z, G, tuple(flags))
