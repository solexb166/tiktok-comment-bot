"""Margin model: account leverage setting, equity tiers, Unlimited eligibility, fixed-margin
instruments and Higher Margin Requirement (HMR) windows. All values come from profiles."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..market.instruments import Universe


@dataclass(frozen=True)
class UnlimitedEligibility:
    """Conditions under which an 'unlimited' account leverage setting is actually granted.
    If not met, `fallback_leverage` applies. Values are profile data (Exness: unverified, U5)."""
    equity_below: float
    min_closed_trades: int = 0
    min_volume_lots: float = 0.0
    fallback_leverage: float = 2000.0

    def eligible(self, equity: float, closed_trades: int, volume_lots: float) -> bool:
        return (equity < self.equity_below and closed_trades >= self.min_closed_trades
                and volume_lots >= self.min_volume_lots)


@dataclass(frozen=True)
class HMRWindow:
    start: int                                  # UTC minute, inclusive
    end: int                                    # UTC minute, exclusive
    leverage_cap: dict = field(default_factory=dict)      # asset_class -> max leverage
    fixed_rate_min: dict = field(default_factory=dict)    # asset_class -> min fixed margin rate


@dataclass(frozen=True)
class MarginModel:
    account_leverage: float = 2000.0            # math.inf = 'unlimited' setting
    equity_tiers: tuple = ()                    # ((equity_upper_exclusive, max_leverage), ...)
    unlimited: UnlimitedEligibility | None = None
    hmr_windows: tuple = ()

    def effective_account_leverage(self, equity: float, closed_trades: int = 0,
                                   volume_lots: float = 0.0) -> float:
        lev = self.account_leverage
        if math.isinf(lev) and self.unlimited is not None:
            if not self.unlimited.eligible(equity, closed_trades, volume_lots):
                lev = self.unlimited.fallback_leverage
        for upper, cap in self.equity_tiers:
            if equity < upper:
                lev = min(lev, cap)
                break
        return lev

    def active_hmr(self, t: int) -> HMRWindow | None:
        for w in self.hmr_windows:
            if w.start <= t < w.end:
                return w
        return None

    def rates(self, universe: Universe, equity: float, t: int, closed_trades: int = 0,
              volume_lots: float = 0.0) -> np.ndarray:
        """Initial-margin rate (fraction of notional) per instrument at time t."""
        lev_acct = self.effective_account_leverage(equity, closed_trades, volume_lots)
        hmr = self.active_hmr(t)
        out = np.empty(universe.n)
        for i, s in enumerate(universe.specs):
            if s.margin_mode == "fixed":
                rate = s.fixed_margin_rate
                if hmr is not None and s.asset_class in hmr.fixed_rate_min:
                    rate = max(rate, hmr.fixed_rate_min[s.asset_class])
            else:
                lev = min(lev_acct, s.max_leverage)
                if hmr is not None and s.asset_class in hmr.leverage_cap:
                    lev = min(lev, hmr.leverage_cap[s.asset_class])
                rate = 0.0 if math.isinf(lev) else 1.0 / lev
            out[i] = rate
        return out
