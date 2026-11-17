"""Automatic failure diagnosis from an evaluation's decision log.

Output keeps two things apart:
  observations           - facts counted directly from the recorded decisions (confirmed)
  possible_explanations  - hypotheses that would fit those facts (NOT confirmed; each says what
                           would test it)
plus the behaviour metrics they are based on.

Decision quality is judged against the expected return of the next bar where the market's
structure is known (synthetic tasks), not against the realised outcome: a well-founded position
that loses money is not counted as a mistake. Thresholds are derived from the test market itself
(e.g. the leverage at which a single observed extreme move would wipe out the account), not set
as arbitrary limits on the agent.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

ACTIVE_CHANGES = ("open", "close", "reverse", "increase", "reduce")


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def behaviour_metrics(log: pd.DataFrame, task, identity: dict) -> dict:
    lev = log["leverage_after_fill"].to_numpy()
    r = log["market_return"].to_numpy()
    sigma = 1.4826 * float(np.median(np.abs(r - np.median(r)))) or float(np.std(r)) or 1e-12
    worst = float(np.max(np.abs(r))) if len(r) else 0.0
    fatal = 0.95 / worst if worst > 0 else math.inf
    big = np.abs(r) >= 4 * sigma
    active = log["position_change"].isin(ACTIVE_CHANGES).to_numpy()
    in_mkt = lev > 1e-6
    exp = log["expected_return"].to_numpy()
    known = np.isfinite(exp).all()
    edge = known and np.any(np.abs(exp) > 1e-12)
    m = dict(
        decisions=int(len(log)), episodes=int(log["episode"].nunique()),
        deaths=int(log["died"].sum()), death_reasons=log.loc[log["died"], "death_reason"].value_counts().to_dict(),
        median_leverage=float(np.median(lev)), p95_leverage=float(np.percentile(lev, 95)),
        max_leverage=float(lev.max()) if len(lev) else 0.0,
        share_over={f"{b}x": float(np.mean(lev > b)) for b in (1, 3, 10, 100)},
        fatal_leverage=fatal, worst_move=worst, typical_move=sigma,
        share_over_fatal=float(np.mean(lev >= fatal)), time_in_market=float(np.mean(in_mkt)),
        trade_frequency=float(np.mean(active)),
        costs_per_episode=float((log[["cost_spread", "cost_slippage", "cost_commission", "cost_swap"]].sum(axis=1))
                                .groupby(log["episode"]).sum().mean()),
        big_moves=int(big.sum()),
        leverage_before_big_moves=float(np.mean(lev[big])) if big.any() else None,
        leverage_elsewhere=float(np.mean(lev[~big])) if (~big).any() else None,
        losing_decisions=int((log["pnl"] < 0).sum()),
    )
    nz = lev[in_mkt]
    m["spike_ratio"] = float(nz.max() / np.median(nz)) if len(nz) else 0.0
    m["spikes"] = int(np.sum(nz > 10 * np.median(nz))) if len(nz) else 0
    if known:
        pos = log["exposure_after_fill"].to_numpy()
        has = np.abs(exp) > 1e-12
        aligned = (np.sign(pos) == np.sign(exp)) & (np.abs(pos) > 1e-6)
        m["edge_known"] = bool(edge)
        m["aligned_with_edge"] = float(np.mean(aligned[has])) if has.any() else None
        m["flat_when_edge"] = float(np.mean(~in_mkt[has])) if has.any() else None
        m["against_edge"] = float(np.mean((np.sign(pos) == -np.sign(exp))[has] & (np.abs(pos) > 1e-6)[has])) if has.any() else None
        losses = log["pnl"].to_numpy() < 0
        m["losses_on_sound_positions"] = float(np.mean(aligned[losses])) if losses.any() and has.any() else None
    g_opt = task.params.get("growth_optimal_G") if task is not None else None
    m["growth_optimal_leverage"] = g_opt
    from .quality import KnownMarket, instability, quality_metrics
    m.update(instability(log["exposure_after_fill"].to_numpy(), log["episode"].to_numpy()))
    mk = KnownMarket.from_task(task) if task is not None else None
    if mk is not None:
        qm = quality_metrics(log["exposure_after_fill"].to_numpy(), mk, log["episode"].to_numpy())
        m.update({f"dq_{k}": v for k, v in qm.items()})
        m["growth_optimal_leverage"] = qm["optimal_size"]
        from .quality import q2_metrics
        q2 = q2_metrics(log["exposure_after_fill"].to_numpy(), mk, log["episode"].to_numpy())
        m.update(dq2=q2["q2"], dq2_by_episode=q2["q2_by_episode"], dq2_floor=q2["floor"], dq2_version=q2["version"])
    # behaviour after a loss of more than 2% of equity: size at the next decision vs before
    eq0 = log["equity_before"].to_numpy()
    loss = (log["pnl"].to_numpy() / np.where(eq0 > 0, eq0, np.nan)) < -0.02
    same = np.r_[log["episode"].to_numpy()[1:] == log["episode"].to_numpy()[:-1], False]
    sel = loss & same & (lev > 1e-6)
    m["size_after_loss_ratio"] = float(np.median(np.r_[lev[1:], np.nan][sel] / lev[sel])) if sel.any() else None
    m["long_share"] = float(np.mean(log["exposure_after_fill"].to_numpy() > 1e-6))
    if task is not None and task.extra is not None:
        flag = task.extra[log["decision_row"].to_numpy(), 0]
        sw = np.flatnonzero(np.diff(flag) != 0) + 1
        lags = []
        pos = log["exposure_after_fill"].to_numpy()
        ep = log["episode"].to_numpy()
        for s in sw:
            k = s
            while k < len(pos) and ep[k] == ep[s] and np.sign(pos[k]) != np.sign(flag[s]):
                k += 1
            if k < len(pos) and ep[k] == ep[s]:
                lags.append(k - s)
        m["regime_switches"] = int(len(sw))
        m["median_adjustment_lag"] = float(np.median(lags)) if lags else None
    return m


def diagnose(log: pd.DataFrame, task, identity: dict, summary: dict | None = None) -> dict:
    if not len(log):
        return dict(observations=[], possible_explanations=[], metrics={})
    m = behaviour_metrics(log, task, identity)
    obs, why = [], []
    O, W = obs.append, why.append
    eps = m["episodes"]

    # 1. deaths after excessive leverage
    if m["deaths"]:
        dead = log[log["died"]]
        prev_lev = []
        for idx in dead.index:
            same = log[(log["episode"] == log.at[idx, "episode"]) & (log.index <= idx)]
            prev_lev.append(float(same["leverage_after_fill"].iloc[-1]))
        n_fatal = sum(x >= m["fatal_leverage"] for x in prev_lev)
        O(dict(code="deaths_after_excessive_leverage" if n_fatal else "deaths",
               text=f"{m['deaths']} of {eps} test runs ended in account death. In {n_fatal} of them the position "
                    f"just before death was at or above {m['fatal_leverage']:.1f}x, the size at which the largest "
                    f"single move in this test data ({_pct(m['worst_move'])}) wipes out 95% of the account.",
               deaths=m["deaths"], after_fatal_leverage=n_fatal))
        if n_fatal:
            W(dict(code="leverage_not_risk_aware",
                   text="Possible: the agent has not learned that rare large moves make positions above "
                        f"~{m['fatal_leverage']:.0f}x fatal. Test: compare its leverage just before large moves "
                        "with its leverage elsewhere, across versions."))

    # 1b. ruinous decisions judged against the known market (S6/S7)
    if m.get("dq_ruinous_share"):
        O(dict(code="ruinous_decisions",
               text=f"{_pct(m['dq_ruinous_share'])} of decisions were ruinous: positions of "
                    f"{m['dq_ruin_size_long']:.1f}x or more, where one ordinary crash in this market ends the "
                    f"account. Decision-quality score {m['dq_quality']:.2f} on the other decisions "
                    f"(1 = optimal {m['dq_optimal_size']:.2f}x, 0 = staying out); episode quality "
                    f"{m['dq_episode_quality']:.2f}.", share=m["dq_ruinous_share"]))

    # 2. leverage spikes
    if m["spikes"] and m["spike_ratio"] > 50:
        O(dict(code="leverage_spikes",
               text=f"Leverage is usually around {m['median_leverage']:.2f}x but reached {m['max_leverage']:.0f}x; "
                    f"{m['spikes']} decisions used more than 10 times the agent's typical size.",
               spikes=m["spikes"], max_leverage=m["max_leverage"]))
        W(dict(code="exposure_output_unstable",
               text="Possible: exposure is chosen on a log scale (0.0001x to 10,000x), so small shifts in the "
                    "network's output move leverage by large factors. Test: measure how much leverage changes "
                    "between consecutive decisions with near-identical observations."))

    # 3. exposure before sudden moves
    if m["big_moves"] and m["leverage_before_big_moves"] is not None and m["leverage_elsewhere"]:
        ratio = m["leverage_before_big_moves"] / m["leverage_elsewhere"]
        if ratio > 1.5:
            O(dict(code="exposed_before_big_moves",
                   text=f"Before the {m['big_moves']} largest moves (beyond 4 typical moves) the agent averaged "
                        f"{m['leverage_before_big_moves']:.2f}x, versus {m['leverage_elsewhere']:.2f}x otherwise.",
                   ratio=ratio))

    # 4. unnecessary trading in markets with nothing to learn
    if m.get("edge_known") is False and m["costs_per_episode"] > 0:
        if m["trade_frequency"] > 0.05 or m["time_in_market"] > 0.2:
            O(dict(code="trading_without_edge",
                   text=f"This market has no predictable component, yet the agent changed its position in "
                        f"{_pct(m['trade_frequency'])} of decisions and held a position {_pct(m['time_in_market'])} "
                        f"of the time, paying ${m['costs_per_episode']:.2f} in costs per test run on average.",
                   trade_frequency=m["trade_frequency"]))
            W(dict(code="noise_fitting",
                   text="Possible: the agent is reacting to random fluctuations as if they were patterns. Test: "
                        "whether its positions correlate with recent returns on fresh random data."))

    # 5. missed opportunities / acting against the known edge
    if m.get("edge_known"):
        if m["flat_when_edge"] is not None and m["flat_when_edge"] > 0.5:
            O(dict(code="missed_opportunities",
                   text=f"An edge was present at every decision, but the agent had no position "
                        f"{_pct(m['flat_when_edge'])} of the time.", flat_share=m["flat_when_edge"]))
        if m["against_edge"] is not None and m["against_edge"] > 0.25:
            O(dict(code="against_the_edge",
                   text=f"{_pct(m['against_edge'])} of positions pointed against the market's known expected "
                        f"direction.", share=m["against_edge"]))
        if m.get("losses_on_sound_positions") is not None:
            O(dict(code="losses_on_sound_positions",
                   text=f"Of {m['losing_decisions']} losing decisions, {_pct(m['losses_on_sound_positions'])} "
                        f"were positioned in the expected direction: losses that came from market noise, not "
                        f"from the decision.", share=m["losses_on_sound_positions"]))

    # 6. too conservative relative to the known optimum
    g = m.get("growth_optimal_leverage")
    if g and m["median_leverage"] < 0.2 * g and m["deaths"] == 0:
        O(dict(code="too_conservative",
               text=f"Median position size {m['median_leverage']:.2f}x, against a growth-optimal level of "
                    f"{g:.1f}x for this market.", median=m["median_leverage"], optimal=g))
        W(dict(code="risk_aversion_learned_from_exploration",
               text="Possible: early training with very large exploratory positions taught the agent that "
                    "exposure is dangerous before it learned how much is useful. Test: compare training-history "
                    "leverage early versus late."))

    # 7. adjusting to changing conditions (markets with a visible regime)
    if m.get("median_adjustment_lag") is not None and m["median_adjustment_lag"] > 3:
        O(dict(code="slow_to_adjust",
               text=f"After the market's {m['regime_switches']} announced trend switches, the agent took a median "
                    f"of {m['median_adjustment_lag']:.0f} decisions to turn its position around.",
               lag=m["median_adjustment_lag"]))
    return dict(observations=obs, possible_explanations=why, metrics=m)
