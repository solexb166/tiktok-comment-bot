"""Phase 3 Experiment 1b: exploration calibration, starting size-noise setting, exploration monitor,
numerical safeguards, protection of earlier agents and the pre-registered decision rules."""
import io
import json
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
import torch

from tradeworld.interface.actions import ActionMapper
from tradeworld.learning import continual as ct
from tradeworld.learning import evaluate as ev
from tradeworld.learning import phase3b as p3b
from tradeworld.learning import phase3b_analysis as an
from tradeworld.registry.registry import ROOT, Registry, RegistryError
from tradeworld.registry.store import LocalBlobStore

ENV = dict(decision_interval=1, obs_resolution=1, window=30)
LOG10 = dict(log10_min=-4.0, log10_max=4.0, cap=None)


# ---------------------------------------------------------------------------- calibration
def test_calibration_matches_the_control_as_designed():
    c = p3b.calibration()
    assert c["control_multiplier_per_sd"] == pytest.approx(2.130, abs=1e-3)
    assert c["size_log_std"]["C1m"] == pytest.approx(-0.2797, abs=1e-4)
    sp, inv = p3b._softplus, p3b._softplus_inv
    zs = inv(c["g_star"])
    # C2-up: +1 sd at the optimum multiplies size by the control's x2.13
    assert float(sp(zs + c["sigma"]["C2-up"])) / c["g_star"] == pytest.approx(c["control_multiplier_per_sd"], rel=1e-9)
    # C2-tail: same chance as the control of exploring to a ruinous size from the optimum (6.0%)
    p = 1 - NormalDist().cdf((inv(c["ruin"]) - zs) / c["sigma"]["C2-tail"])
    assert p == pytest.approx(c["control_ruin_probability_from_optimum"], rel=1e-9)
    assert c["control_ruin_probability_from_optimum"] == pytest.approx(0.0599, abs=5e-4)


def test_designed_distribution_c1m_is_identical_to_control_at_every_size():
    d = p3b.exploration_design(n=200_000)["by_start_size"]
    for a, b in zip(d["control"], d["C1m"]):
        assert a["lnsize_sd"] == pytest.approx(b["lnsize_sd"], rel=1e-9)
        assert a["p_ge_ruin"] == pytest.approx(b["p_ge_ruin"], abs=1e-12)


def test_group_configs_are_stable_and_distinct():
    g1, g2 = p3b.groups(), p3b.groups()
    assert {k: p3b.config_checksum(v) for k, v in g1.items()} == {k: p3b.config_checksum(v) for k, v in g2.items()}
    assert len({p3b.config_checksum(v) for v in g1.values()}) == 3


# ---------------------------------------------------------------------------- safeguards
@pytest.mark.parametrize("repr_", ["exp", "softplus"])
def test_vectorised_size_mapping_matches_scalar_and_is_unclipped_over_usable_sizes(repr_):
    m = ActionMapper(1, size_repr=repr_)
    sizes = (1e-4, 0.01, 0.5, 2.93, 9.5, 100.0, 2000.0)              # up to the 1:2000 margin limit
    z = np.array([m.u_for_gross(G) for G in sizes])
    lo, hi = m.size_bounds
    assert np.all((z > lo) & (z < hi))                          # every usable size lies strictly inside the range
    assert np.allclose(m.gross_array(z), sizes, rtol=1e-9)
    assert m.gross(hi) >= 10_000.0 - 1e-6                        # top of the range: at least the original 10,000x
    assert np.allclose(m.gross_array(z), [m.gross(float(x)) for x in z], rtol=1e-12)


@pytest.mark.parametrize("group", ["C1m", "C2-up", "C2-tail"])
def test_exploration_from_any_usable_size_essentially_never_reaches_the_numerical_range(group):
    c = p3b.calibration()
    m = ActionMapper(1, size_repr="exp" if group == "C1m" else "softplus")
    lo, hi = m.size_bounds
    sig = c["sigma"][group]
    assert m.gross(lo) < 1e-10                                       # the lower end only affects sizes below 1e-10x
    for G in (1e-3, 2.93, 2000.0):                                  # up to the margin limit
        z = m.u_for_gross(G)
        p_upper = 1 - NormalDist().cdf((hi - z) / sig)               # the only end that could act as a cap
        assert p_upper < 1e-9, (group, G, p_upper)
    for G in (0.1, 2.93, 2000.0):                                   # usable sizes: neither end within reach
        z = m.u_for_gross(G)
        assert NormalDist().cdf((lo - z) / sig) + 1 - NormalDist().cdf((hi - z) / sig) < 1e-6, (group, G)


# ---------------------------------------------------------------------------- training settings and monitor
@pytest.fixture
def reg(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(ct, "TRAIN_STEPS_DATA", 3000)
    monkeypatch.setitem(ct.PPO_DEFAULTS, "n_steps", 256)
    monkeypatch.setitem(ct.PPO_DEFAULTS, "batch_size", 128)
    return Registry(tmp_path / "reg")


@pytest.mark.parametrize("group", ["C1m", "C2-tail"])
def test_size_log_std_applied_recorded_and_exploration_monitored(reg, group):
    cfg = p3b.groups()[group]
    a = reg.create_agent(task="S6", reward="R1", seed=7, env_settings=ENV, action_space=cfg["action_space"],
                         profile="A_synthetic", episode_decisions=50)
    v = ct.train_version(reg, a, steps=512, n_envs=2, checkpoint_every=10_000, seed=7,
                         initial_settings=cfg["initial_settings"], log=lambda m: None)
    assert v["initial_settings"] == cfg["initial_settings"]
    up = pd.read_parquet(io.BytesIO(reg.blobs.get(v["history"]["updates_sha256"])))
    first = up.iloc[0]
    assert float(first["explore/log_std_exposure"]) == pytest.approx(cfg["initial_settings"]["size_log_std"], abs=1e-6)
    for k in ("explore/size_ratio_sd", "explore/position_ratio_sd", "explore/size_clipped_share",
              "explore/size_sample_ge_ruin_share", "explore/band_lt1_share", "explore/size_mean_median"):
        assert k in up and np.isfinite(float(first[k]))
    if group == "C1m":                                   # e^z: the size spread equals the noise sigma exactly
        assert float(first["explore/size_ratio_sd"]) == pytest.approx(math.exp(cfg["initial_settings"]["size_log_std"]), rel=0.08)
    assert float(first["explore/size_clipped_share"]) == 0.0


def test_size_log_std_only_for_new_agents(reg):
    a = reg.create_agent(task="S6", reward="R1", seed=7, env_settings=ENV, action_space=LOG10,
                         profile="A_synthetic", episode_decisions=50)
    ct.train_version(reg, a, steps=256, n_envs=2, checkpoint_every=10_000, log=lambda m: None)
    with pytest.raises(RegistryError):
        ct.train_version(reg, a, steps=256, parent=1, n_envs=2, initial_settings=dict(size_log_std=0.0), log=lambda m: None)


# ---------------------------------------------------------------------------- protection of earlier agents
@pytest.mark.parametrize("root,task_reward_or_group", [
    ("registry", ("S6", "R1", 0)), ("registry", ("S7", "R1", 1)),
    ("registry-phase3", "control"), ("registry-phase3", "C1"), ("registry-phase3", "C2"),
])
def test_earlier_agents_regenerate_identical_decisions(tmp_path, root, task_reward_or_group):
    """Original and Experiment 1 agents, re-run from their checkpoints under the new code, make
    exactly the recorded decisions (full decision-log fingerprint)."""
    reg = Registry(ROOT / root)
    reg.cache = LocalBlobStore(tmp_path / "cache")                   # never write to their caches
    if root == "registry":
        a = reg.find(*task_reward_or_group)
        e = next(x for x in reg.evaluations(a) if x["round_id"] == "gate-1")
    else:
        m = json.loads((ROOT / root / "experiments" / "phase3-exp1" / "manifest.json").read_text())
        a = next(x["agent_id"] for x in m["agents"].values() if x["group"] == task_reward_or_group)
        e = next(x for x in reg.evaluations(a) if x["round_id"] == "exp1-final-a")
    log = ev.decision_log(reg, e)                                     # raises if any decision differs
    assert len(log) == e["decisions"]


# ---------------------------------------------------------------------------- decision rules
def _seed(i, **kw):
    s = dict(seed_index=i, ruinous=0.0, deaths=0, median_size=2.9, q2=0.9, growth_ok=True, tilt=1.1, w7=True,
             train_gap=0.0, boundary=0.0, rejected=0.0, safeguard_share=0.0)
    s.update(kw)
    return s


def test_seed_flags_reject_timid_and_reckless_agents():
    g = 2.93
    assert an.seed_flags(_seed(1), g)["success"]
    timid = an.seed_flags(_seed(1, median_size=0.5, q2=0.3), g)
    assert timid["W1"] and not timid["W3"] and not timid["success"]           # survives at 0.5x: not a success
    reckless = an.seed_flags(_seed(1, ruinous=0.05, deaths=4), g)
    assert reckless["W3"] and not reckless["W1"] and not reckless["success"]  # reaches 3x but catastrophic bets
    assert not an.seed_flags(_seed(1, safeguard_share=0.01), g)["E"]           # leaning on the numerical range


def test_interpretation_rules():
    ok = [an.seed_flags(_seed(i), 2.93) for i in range(6)]
    assert an.interpretation(ok, [_seed(i) for i in range(6)]) == "safe growth"
    ss = [_seed(i, ruinous=0.02 if i < 3 else 0.0, median_size=1.0) for i in range(6)]
    assert an.interpretation([an.seed_flags(s, 2.93) for s in ss], ss) == "earlier safety came from reduced exploration"
    ss = [_seed(i, median_size=0.6, q2=0.3) for i in range(6)]
    assert an.interpretation([an.seed_flags(s, 2.93) for s in ss], ss) == "safe but timid"
    ss = [_seed(i, median_size=2.0 if i < 3 else 0.6, w7=i < 2) for i in range(6)]
    assert an.interpretation([an.seed_flags(s, 2.93) for s in ss], ss) == "mixed"


def test_two_sided_tests_and_verdict_structure():
    assert an.permutation_p_two_sided([10] * 6, [0] * 6) == pytest.approx(2 / 924)
    assert an.signflip_p_two_sided([1] * 6) == pytest.approx(2 / 64)
    seeds = {g: [_seed(i, q2=0.8 + 0.01 * i) for i in range(1, 7)] for g in an.NEW_GROUPS}
    refs = {g: [_seed(i, q2=-2.0, ruinous=0.1, median_size=1.4) for i in range(1, 7)] for g in ("control", "C1", "C2")}
    v = an.verdicts(seeds, refs, 2.93)
    assert len([k for g in v["groups"].values() for k in g["comparisons"]]) == 18
    assert all(g["interpretation"] == "safe growth" for g in v["groups"].values())
    assert all("p_holm" in c for g in v["groups"].values() for c in g["comparisons"].values())
