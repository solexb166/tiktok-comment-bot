"""Observation encoder: Observation -> flat float32 vector for a neural policy.

This is a documented inductive bias (architecture §6), applied identically to every agent:
  * Price-shaped features (ret, range, body, spread) are divided by the window's own standard
    deviation of returns (window-local normalisation: no statistics from outside the window, so
    nothing from the future can leak in) and clipped to +-CLIP.
  * One extra feature per instrument gives the volatility level itself, log10(std) scaled, so the
    agent can still sense how volatile a market is (needed to size exposure sensibly).
  * quoted_frac is kept as is; rel_volume is clipped to [0, CLIP].
  * Account features are passed through (already log-scaled by the observation builder).
  * Optional `extra` features (synthetic experiments only, e.g. a visible regime flag).
No indicators, prices, dates or signals are added.
"""
from __future__ import annotations

import numpy as np

CLIP = 10.0
BOUND = 1e3                          # hard bound on every encoded value (observation space)
VOL_FLOOR = 1e-8
PRICE_SHAPED = (0, 1, 2, 3)          # ret, range, body, spread
QUOTED, REL_VOL = 4, 5


class ObsEncoder:
    def __init__(self, n_instruments: int, window: int, n_market_features: int, n_account: int,
                 n_extra: int = 0):
        self.n, self.W, self.F, self.A, self.E = n_instruments, window, n_market_features, n_account, n_extra
        self.dim = self.n * self.W * self.F + self.n + self.A + self.E

    def encode(self, obs, extra: np.ndarray | None = None) -> np.ndarray:
        m = np.array(obs.market, dtype=np.float64)                  # [N, W, F] copy
        ret = m[:, :, 0]
        q = m[:, :, QUOTED] > 0
        std = np.empty(self.n)
        for i in range(self.n):
            r = ret[i][q[i]] if q[i].any() else ret[i]
            std[i] = max(float(np.std(r)), VOL_FLOOR)
        m[:, :, PRICE_SHAPED] /= std[:, None, None]
        m[:, :, PRICE_SHAPED] = np.clip(m[:, :, PRICE_SHAPED], -CLIP, CLIP)
        m[:, :, REL_VOL] = np.clip(m[:, :, REL_VOL], 0.0, CLIP)
        vol_level = np.log10(std) / 4.0 + 1.0                         # ~0 for 1e-4 per bar
        parts = [m.reshape(-1), vol_level, np.asarray(obs.account, dtype=np.float64)]
        if self.E:
            if extra is None or np.shape(extra) != (self.E,):
                raise ValueError(f"expected {self.E} extra features")
            parts.append(np.asarray(extra, dtype=np.float64))
        out = np.concatenate(parts).astype(np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=BOUND, neginf=-BOUND)
        return np.clip(out, -BOUND, BOUND)
