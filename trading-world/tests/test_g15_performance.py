"""Group 15: throughput benchmark (recorded per machine) and per-decision recording completeness."""
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import pytest

from tradeworld.agents.baselines import ConstantPolicy, FlatPolicy, RandomPolicy
from tradeworld.brokerspec import load_profile
from tradeworld.data.synthetic import generate
from tradeworld.world.world import RECORD_FIELDS
from conftest import MON_2024_01_08, act, run

OUT = Path(__file__).resolve().parents[1] / "benchmarks"


@pytest.mark.benchmark
def test_throughput_and_recording():
    prof = load_profile("environments/A_fractional.yaml")
    days = 5
    md = generate(prof.universe, MON_2024_01_08, days * 1440, seed=1)
    policies = {"flat": FlatPolicy(6), "rand": RandomPolicy(6, 1),
                "long": ConstantPolicy(act(6, [1 / 6] * 6, 1.0)),
                "lev": ConstantPolicy(act(6, [0.5, 0.5, 0, 0, 0, 0], 50.0))}
    t0 = time.perf_counter()
    res = run(md, prof, policies, seed=1, interval=5, resolution=5, window=120)
    dt = time.perf_counter() - t0
    rec = res.records
    stats = {
        "machine": platform.platform(), "cpus": os.cpu_count(), "python": platform.python_version(),
        "instruments": 6, "agents": len(policies), "m1_bars": md.T, "decisions": len(rec),
        "seconds": round(dt, 3), "decisions_per_s": round(len(rec) / dt, 1),
        "agent_m1_steps_per_s": round(md.T * len(policies) / dt, 1),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "throughput.json").write_text(json.dumps(stats, indent=2))
    print(stats)
    assert stats["decisions_per_s"] > 200                           # loose floor; see benchmarks/

    # recording completeness
    assert tuple(rec.columns) == RECORD_FIELDS
    valid = rec["G"].notna()
    for col in RECORD_FIELDS:
        if col in ("d", "u", "G", "w_target"):
            assert rec.loc[valid, col].notna().all(), col
        else:
            assert rec[col].notna().all(), col
    assert (rec["max_gross_leverage_interval"] >= 0).all()
    lev = rec[rec.agent == "lev"]
    assert lev["gross_leverage"].between(40, 60).all()
    assert np.allclose(lev["w_target"].iloc[0], [25, 25, 0, 0, 0, 0])
