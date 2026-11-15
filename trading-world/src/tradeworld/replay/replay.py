"""Historical replay: decision schedule and past-only observation windows.

Observation bars of `obs_resolution` minutes are aggregated ONLY from completed M1 bars. The
grid start must be hour-aligned, so observation bar b covers M1 rows [b*r, (b+1)*r). At a decision
made at the close of M1 row d, the latest completed observation bar is j = (d + 1) // r - 1; the
window is bars j-W+1 .. j. Nothing at or after row d+1 can enter an observation (test group 1).
"""
from __future__ import annotations

import numpy as np

from ..data.market_data import MarketData

MARKET_FEATURES = ("ret", "range", "body", "spread", "quoted_frac", "rel_volume")


class Replay:
    def __init__(self, md: MarketData, decision_interval: int = 5, obs_resolution: int = 5,
                 window: int = 120):
        if int(md.times[0]) % 60:
            raise ValueError("market data must start on a full hour")
        if decision_interval < 1 or obs_resolution < 1 or window < 1:
            raise ValueError("decision_interval, obs_resolution and window must be >= 1")
        if 60 % obs_resolution and obs_resolution % 60:
            raise ValueError("obs_resolution must divide or be a multiple of 60 minutes")
        self.md, self.k, self.r, self.W = md, decision_interval, obs_resolution, window
        self._aggregate()
        first_d = (window + 1) * obs_resolution - 1          # need W bars plus one for returns
        d = np.arange(first_d, md.T - 1)                    # need a next bar to fill on
        self.decision_indices = d[(md.times[d] + 1) % decision_interval == 0]

    def _aggregate(self) -> None:
        md, r = self.md, self.r
        B = md.T // r
        rows = B * r

        def blk(a):
            return a[:rows].reshape(B, r, md.N)

        mid_o = (md.bid_o + md.ask_o) / 2
        mid_h = (md.bid_h + md.ask_h) / 2
        mid_l = (md.bid_l + md.ask_l) / 2
        mid_c = (md.bid_c + md.ask_c) / 2
        o = blk(mid_o)[:, 0, :]
        h = blk(mid_h).max(axis=1)
        l = blk(mid_l).min(axis=1)
        c = blk(mid_c)[:, -1, :]
        spread = (blk(md.ask_c)[:, -1, :] - blk(md.bid_c)[:, -1, :]) / c
        qfrac = blk(md.quoted.astype(np.float64)).mean(axis=1)
        vol = blk(md.volume).sum(axis=1)
        ret = np.zeros_like(c)
        ret[1:] = np.log(c[1:] / c[:-1])
        feats = np.stack([ret, (h - l) / c, (c - o) / c, spread, qfrac], axis=-1)   # [B, N, 5]
        self.bar_features = feats.astype(np.float32)
        self.bar_volume = vol.astype(np.float32)
        self.bar_features.setflags(write=False)
        self.bar_volume.setflags(write=False)

    def obs_bar(self, d: int) -> int:
        return (int(d) + 1) // self.r - 1

    def market_window(self, d: int) -> np.ndarray:
        """float32 [N, W, F] read-only window for a decision at the close of M1 row d."""
        j = self.obs_bar(d)
        if j - self.W + 1 < 0:
            raise IndexError("not enough history for a full observation window")
        f = self.bar_features[j - self.W + 1: j + 1]                 # [W, N, 5]
        v = self.bar_volume[j - self.W + 1: j + 1]                    # [W, N]
        mean = v.mean(axis=0, keepdims=True)
        rel = np.divide(v, mean, out=np.zeros_like(v), where=mean > 0)   # window-local
        out = np.concatenate([f, rel[..., None]], axis=-1).transpose(1, 0, 2).copy()
        out.setflags(write=False)
        return out
