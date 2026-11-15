"""Execution engine and per-bar account processing.

Timing (no look-ahead):
  * A decision is made at the CLOSE of M1 bar `d` using only data up to and including bar d.
  * Orders fill at the OPEN of bar d+1: buys at ask, sells at bid, plus adverse slippage.
  * Target units are sized from decision-time equity and mid (the agent cannot know the next open).
Margin (atomic, order-independent): reductions always execute; all increases of a decision
execute together only if the post-trade used margin fits post-trade equity, otherwise all of
them are rejected ('insufficient_margin'). The engine never picks a size for the agent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..accounts.account import Account, Prices
from ..brokerspec.profile import EnvProfile
from ..data.market_data import MarketData
from ..market.clock import rollover_schedule


class MarketView:
    """Read-only, profile-adjusted view of market data shared by all agents of a world."""

    def __init__(self, md: MarketData, profile: EnvProfile):
        if tuple(md.symbols) != profile.universe.symbols:
            raise ValueError(f"data symbols {md.symbols} != profile symbols {profile.universe.symbols}")
        self.md, self.profile, self.u = md, profile, profile.universe
        m = profile.spread_multiplier
        arrays = {}
        for f in ("o", "h", "l", "c"):
            b, a = getattr(md, f"bid_{f}"), getattr(md, f"ask_{f}")
            mid = (a + b) / 2.0
            arrays[f"bid_{f}"] = mid - m * (mid - b)
            arrays[f"ask_{f}"] = mid + m * (a - mid)
            arrays[f"mid_{f}"] = mid
        for k, v in arrays.items():
            v.setflags(write=False)
            setattr(self, k, v)
        self.conv_o = self.u.quote_to_usd_series(self.mid_o)
        self.conv_c = self.u.quote_to_usd_series(self.mid_c)
        self.range_frac = (md.ask_h - md.bid_l) / self.mid_c - (md.ask_c - md.bid_c) / self.mid_c
        self.range_frac = np.clip(self.range_frac, 0.0, None)
        self.quoted = md.quoted
        self.times = md.times
        self.is_rollover, self.rollover_weekday = rollover_schedule(md.times, profile.rollover_hour_ny)

    def prices_open(self, t: int) -> Prices:
        return Prices(self.bid_o[t], self.ask_o[t], self.conv_o[t])

    def prices_close(self, t: int) -> Prices:
        return Prices(self.bid_c[t], self.ask_c[t], self.conv_c[t])


@dataclass
class ExecReport:
    target_units: np.ndarray
    rejections: dict = field(default_factory=dict)      # instrument index -> reason
    increases_rejected: bool = False
    safeguards: list = field(default_factory=list)


def target_units(view: MarketView, w: np.ndarray, d: int, equity: float) -> np.ndarray:
    p = view.prices_close(d)
    return w * equity / (p.mid * p.conv)


def round_to_lots(spec, units: float) -> float:
    if spec.fractional:
        return units
    lots = abs(units) / spec.contract_size
    lots = math.floor(lots / spec.lot_step + 1e-9) * spec.lot_step     # toward zero
    if lots < spec.min_lot - 1e-12:
        return 0.0
    return math.copysign(round(lots, 10) * spec.contract_size, units) if lots else 0.0


def _split(cur: float, target: float) -> tuple[float, float]:
    """Split a position change into (reduction, increase) parts."""
    if cur == 0.0:
        return 0.0, target
    if target != 0.0 and np.sign(target) == np.sign(cur):
        if abs(target) <= abs(cur):
            return target - cur, 0.0
        return 0.0, target - cur
    return -cur, target


def execute(acct: Account, view: MarketView, w: np.ndarray, d: int) -> ExecReport:
    """Move the account toward target exposure `w` (decided at close of bar d) at the open of d+1."""
    prof, u = acct.profile, acct.u
    f = d + 1
    t_fill = int(view.times[f])
    eq_dec = acct.equity(view.prices_close(d))
    raw = target_units(view, w, d, eq_dec)
    tgt = np.array([round_to_lots(s, x) for s, x in zip(u.specs, raw)])
    rep = ExecReport(target_units=tgt)
    p_dec = view.prices_close(d)

    reds, incs = np.zeros(u.n), np.zeros(u.n)
    for i, spec in enumerate(u.specs):
        cur = acct.units[i]
        delta = tgt[i] - cur
        if delta == 0.0:
            continue
        if spec.fractional and tgt[i] != 0.0 and abs(delta) * p_dec.mid[i] * p_dec.conv[i] < prof.min_trade_notional_usd:
            rep.safeguards.append(("sub_threshold_delta", i))
            continue
        if not view.quoted[f, i]:
            rep.rejections[i] = "market_closed"
            continue
        red, inc = _split(cur, tgt[i])
        if inc != 0.0 and abs(cur + red + inc) / spec.contract_size > spec.max_lots + 1e-9:
            rep.rejections[i] = "max_volume"
            inc = 0.0
        reds[i], incs[i] = red, inc
    for i, spec in enumerate(u.specs):          # flag targets rounded to zero by lot rules
        if not spec.fractional and raw[i] != 0.0 and tgt[i] == 0.0 and acct.units[i] == 0.0:
            rep.rejections.setdefault(i, "below_min_lot")

    # Slippage is drawn up front in a fixed order so the RNG stream does not depend on outcomes.
    slip_red = {i: prof.slippage.draw(acct.rng, view.range_frac[f, i]) for i in np.flatnonzero(reds)}
    slip_inc = {i: prof.slippage.draw(acct.rng, view.range_frac[f, i]) for i in np.flatnonzero(incs)}
    po = view.prices_open(f)

    def fill_price(i, q, s):
        return (po.ask[i] * (1 + s), po.ask[i]) if q > 0 else (po.bid[i] * (1 - s), po.bid[i])

    for i, s in slip_red.items():
        price, ref = fill_price(i, reds[i], s)
        acct.fill(t_fill, int(i), reds[i], price, po.conv[i], po.mid[i], ref, "decision")

    if slip_inc:
        # Hypothetical post-trade state for the atomic margin check.
        units_after = acct.units.copy()
        bal_after = acct.balance
        for i, s in slip_inc.items():
            price, _ = fill_price(i, incs[i], s)
            spec = u.specs[i]
            bal_after -= (abs(incs[i]) / spec.contract_size * spec.commission_per_lot
                          + abs(incs[i]) * price * po.conv[i] * spec.commission_bps / 1e4)
            # average price after adding (increases never cross zero after reductions)
            units_after[i] += incs[i]
        avg_after = acct.avg.copy()
        for i, s in slip_inc.items():
            price, _ = fill_price(i, incs[i], s)
            cur = acct.units[i]
            avg_after[i] = (abs(cur) * acct.avg[i] + abs(incs[i]) * price) / abs(units_after[i])
        mark = np.where(units_after > 0, po.bid, po.ask)
        eq_after = bal_after + float(np.sum(units_after * (mark - avg_after) * po.conv))
        margin_after = float(np.sum(np.abs(units_after) * po.mid * po.conv *
                                    prof.margin.rates(u, eq_after, t_fill, acct.closed_trades, acct.volume_lots)))
        if margin_after > eq_after or eq_after <= 0:
            rep.increases_rejected = True
            for i in slip_inc:
                rep.rejections[int(i)] = "insufficient_margin"
            acct._event(t_fill, "reject", reason="insufficient_margin", margin=float(margin_after),
                        equity=float(eq_after))
        else:
            for i, s in slip_inc.items():
                price, ref = fill_price(i, incs[i], s)
                acct.fill(t_fill, int(i), incs[i], price, po.conv[i], po.mid[i], ref, "decision")
    for i, reason in rep.rejections.items():
        if reason != "insufficient_margin":
            acct._event(t_fill, "reject", reason=reason, i=int(i))
    return rep


# ---------------------------------------------------------------------------- per-bar processing
def process_bar(acct: Account, view: MarketView, j: int) -> None:
    """Swaps, intra-bar death / stop-out, and interval statistics for M1 bar j."""
    if not acct.alive:
        return
    prof = acct.profile
    t = int(view.times[j])
    if acct.has_positions and view.is_rollover[j]:
        acct.apply_swaps(t, view.prices_open(j), int(view.rollover_weekday[j]))
    if acct.has_positions:
        _intrabar_checks(acct, view, j, t)
    if acct.alive:
        pc = view.prices_close(j)
        eq = acct.equity(pc)
        acct.min_equity_interval = min(acct.min_equity_interval, eq)
        if eq > 0:
            acct.max_gross_interval = max(acct.max_gross_interval, float(np.sum(acct.notional(pc))) / eq)
        elif acct.has_positions:
            acct.max_gross_interval = math.inf


def _intrabar_checks(acct: Account, view: MarketView, j: int, t: int) -> None:
    """Walk each held instrument linearly from its open to its adverse extreme (longs: bid low,
    shorts: ask high); find the earliest point (alpha in [0, 1]) where death or stop-out occurs.
    Conservative: assumes all adverse extremes coincide. Conversion fixed at the bar open."""
    prof = acct.profile
    conv = view.conv_o[j]
    mids = view.mid_o[j]
    rf = view.range_frac[j]
    death_eq = prof.death_equity
    open_mark = None
    for _ in range(acct.u.n + 1):               # at most one stop-out per position, then death
        if not acct.has_positions:
            return
        longs = acct.units > 0
        if open_mark is None:                    # after a stop-out, resume from the trigger marks
            open_mark = np.where(longs, view.bid_o[j], view.ask_o[j])
        extreme = np.where(longs, view.bid_l[j], view.ask_h[j])

        def marks(a):
            return open_mark + a * (extreme - open_mark)

        def equity(a):
            m = marks(a)
            return acct.balance + float(np.sum(acct.units * (m - acct.avg) * conv))

        def margin(a, eq):
            m = marks(a)
            rates = prof.margin.rates(acct.u, eq, t, acct.closed_trades, acct.volume_lots)
            return float(np.sum(np.abs(acct.units) * m * conv * rates))

        def death_hit(a):
            return equity(a) <= death_eq

        def so_hit(a):
            if prof.stop_out_level <= 0.0:       # 0% stop-out => equity <= 0 < death floor: death first
                return False
            eq = equity(a)
            mu = margin(a, eq)
            return mu > 0 and eq <= prof.stop_out_level * mu

        # Death has priority at the same alpha (experimental criterion first).
        if death_hit(0.0):
            acct.die(t, "gap_through", open_mark, conv, mids, rf)
            return
        if so_hit(0.0):
            _stop_out_one(acct, t, open_mark, conv, mids, rf, "gap_through")
            continue
        a_d = _first_hit(death_hit) if death_hit(1.0) else None
        a_s = _first_hit(so_hit) if so_hit(1.0) else None
        if a_d is None and a_s is None:
            return
        if a_s is None or (a_d is not None and a_d <= a_s):
            acct.die(t, "equity_floor", marks(a_d), conv, mids, rf)
            return
        m = marks(a_s)
        _stop_out_one(acct, t, m, conv, mids, rf, "stop_out")
        open_mark = m


def _first_hit(hit, iters: int = 50) -> float:
    lo, hi = 0.0, 1.0                            # hit(lo) False, hit(hi) True
    for _ in range(iters):
        mid = (lo + hi) / 2
        if hit(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _stop_out_one(acct: Account, t, marks, conv, mids, rf, reason) -> None:
    """Close the most unprofitable position (MT-style; Exness order unverified, U7)."""
    upnl = acct.units * (marks - acct.avg) * conv
    upnl = np.where(acct.units != 0, upnl, np.inf)
    i = int(np.argmin(upnl))
    acct._event(t, "stop_out", i=i, reason=reason)
    acct._close_at(t, i, float(marks[i]), float(conv[i]), float(mids[i]), float(rf[i]), "stop_out")
    if not acct.has_positions:
        acct._apply_nbp(t)
