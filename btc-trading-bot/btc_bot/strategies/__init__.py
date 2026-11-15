from .base import Action, Decision, Strategy
from .breakout import DonchianBreakout
from .mean_reversion import MeanReversion

STRATEGIES: dict[str, type[Strategy]] = {
    MeanReversion.name: MeanReversion,
    DonchianBreakout.name: DonchianBreakout,
}


def make_strategy(name: str, params: dict | None = None) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy {name!r}; available: {sorted(STRATEGIES)}")
    return STRATEGIES[name](**(params or {}))


__all__ = ["Action", "Decision", "Strategy", "STRATEGIES", "make_strategy",
           "MeanReversion", "DonchianBreakout"]
