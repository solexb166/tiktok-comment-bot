"""The only information an agent receives. Market window + its own account state."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..accounts.account import Account
from ..replay.replay import MARKET_FEATURES, Replay

EPS_EQUITY = 0.01
MARGIN_LEVEL_SENTINEL = 1e6


def slog(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass(frozen=True)
class Observation:
    t: int                       # decision time (UTC minute at which the decision bar closed)
    market: np.ndarray           # float32 [N, W, F], read-only
    account: np.ndarray          # float32 [A], read-only
    exposure: np.ndarray         # float64 [N], current signed exposure (notional / equity)
    symbols: tuple

    market_features = MARKET_FEATURES


def account_feature_names(symbols) -> list[str]:
    n = [f"{k}:{s}" for k in ("exposure", "upnl", "time_in_pos", "rejected") for s in symbols]
    return ["log_equity", "free_margin_frac", "margin_level", "gross", "net",
            "margin_call", "increases_rejected"] + n


def build_observation(replay: Replay, view, acct: Account, d: int, last_report=None) -> Observation:
    p = view.prices_close(d)
    t = int(view.times[d])
    eq = acct.equity(p)
    used = acct.margin_used(p, t, eq) if acct.has_positions else 0.0
    ml = math.inf if used <= 0 else eq / used
    w = acct.exposure(p) if eq > 0 else np.zeros(acct.u.n)
    w = np.where(np.isfinite(w), w, 0.0)
    upnl = acct.unrealized(p) / eq if eq > 0 else np.zeros(acct.u.n)
    tip = np.where(acct.open_t >= 0, np.log1p(np.maximum(0, t + 1 - acct.open_t)) / 10.0, 0.0)
    rej = np.zeros(acct.u.n)
    if last_report is not None:
        for i in last_report.rejections:
            rej[i] = 1.0
    mc_level = acct.profile.margin_call_level
    account = np.concatenate([
        [math.log(max(eq, EPS_EQUITY * acct.initial) / acct.initial),
         float(np.clip((eq - used) / eq, -10, 10)) if eq > 0 else -10.0,
         float(slog(MARGIN_LEVEL_SENTINEL if math.isinf(ml) else ml)),
         float(slog(np.sum(np.abs(w)))), float(slog(np.sum(w))),
         1.0 if (mc_level is not None and ml <= mc_level) else 0.0,
         1.0 if (last_report is not None and last_report.increases_rejected) else 0.0],
        slog(w), slog(upnl), tip, rej]).astype(np.float32)
    account.setflags(write=False)
    w = w.copy()
    w.setflags(write=False)
    return Observation(t + 1, replay.market_window(d), account, w, acct.u.symbols)
