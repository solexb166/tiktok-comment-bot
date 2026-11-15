# Trading World (Phase 1: deterministic simulator)

A market simulator in which independent agents trade with $10 virtual accounts. Phase 1 contains
the simulator only: data, replay, accounts, execution, margin, liquidation. **No learning code
and no real-order code** (test-enforced). Design: `docs/ARCHITECTURE.md`; exact formulas:
`docs/simulator-spec.md`.

## Quick start
```bash
pip install -e ".[dev]"          # or: pip install numpy pandas pyarrow pyyaml pytest hypothesis
python -m pytest                 # full suite (all 15 simulator test groups)
python scripts/hardware_check.py # facts for the training resource plan
```

## Environments
| Profile | File | Status |
|---|---|---|
| A fractional (primary) | `configs/environments/A_fractional.yaml` | usable; explicit assumptions |
| A unlimited-margin variant | `configs/environments/A_unlimited_margin.yaml` | usable |
| B Exness Cent | `configs/environments/B_exness_cent.yaml` | **blocked** until specs are verified (`docs/exness-spec-verification.md`) |

## Minimal example
```python
from tradeworld.brokerspec import load_profile
from tradeworld.data.synthetic import generate
from tradeworld.world.world import World, AgentSpec
from tradeworld.agents.baselines import RandomPolicy, FlatPolicy

prof = load_profile("environments/A_fractional.yaml")
md = generate(prof.universe, start_minute=28_411_200, minutes=3 * 1440, seed=1)  # Mon 2024-01-08
res = World(md, prof, [AgentSpec("flat", FlatPolicy(6)), AgentSpec("rand", RandomPolicy(6, 1))], seed=0).run()
print(res.records.groupby("agent")[["equity", "gross_leverage"]].describe())
```

## Historical data (run locally; Dukascopy rate-limits cloud IPs)
```bash
python scripts/download_dukascopy.py --start 2018-01-01 --end 2018-01-31
```
Builds an aligned, hashed Parquet store under `data/dukascopy/aligned/`. Price divisors in
`configs/data/dukascopy.yaml` are assumptions checked against plausible price ranges on import.

## Test groups
| # | Group | File |
|---|---|---|
| 1 | No look-ahead | `tests/test_g01_lookahead.py` |
| 2 | Fills | `tests/test_g02_fills.py` |
| 3 | Accounting invariants (Hypothesis) | `tests/test_g03_accounting.py` |
| 4 | Golden scenarios | `tests/test_g04_golden.py` |
| 5 | Action mapping | `tests/test_g05_actions.py` |
| 6 | Margin | `tests/test_g06_margin.py` |
| 7 | Death and stop-out | `tests/test_g07_death_stopout.py` |
| 8 | NBP | `tests/test_g08_nbp.py` |
| 9 | Swaps | `tests/test_g09_swaps.py` |
| 10 | Sessions | `tests/test_g10_sessions.py` |
| 11 | Determinism | `tests/test_g11_determinism.py` |
| 12 | Isolation | `tests/test_g12_isolation.py` |
| 13 | Baselines | `tests/test_g13_baselines.py` |
| 14 | No order path / profile integrity | `tests/test_g14_integrity.py` |
| 15 | Performance + recording completeness | `tests/test_g15_performance.py` |
Plus data quality (`test_data_quality.py`) and historical data infrastructure (`test_historical_data.py`).
