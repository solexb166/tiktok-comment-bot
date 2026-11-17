"""Phase 2: reward variants R1-R5."""
import math

import pytest

from tradeworld.learning.rewards import make_reward


def run(name, equities, **kw):
    r = make_reward(name, **kw)
    r.reset(equities[0])
    out = []
    for k in range(1, len(equities)):
        out.append(r(equities[k - 1], equities[k], k == len(equities) - 1))
    return out


def test_r1_sums_to_log_growth():
    eq = [10.0, 11.0, 9.0, 12.5]
    assert sum(run("R1", eq)) == pytest.approx(math.log(12.5 / 10.0))


def test_r1_is_floored_and_finite_at_zero_or_negative_equity():
    out = run("R1", [10.0, 0.0, -5.0])
    assert all(math.isfinite(x) for x in out)
    assert out[0] == pytest.approx(math.log(0.01))      # eps * E0 = 0.1
    assert out[1] == 0.0


def test_r2_is_linear():
    assert run("R2", [10.0, 12.0, 11.0]) == pytest.approx([0.2, -0.1])


def test_r3_penalises_only_new_drawdown():
    out = run("R3", [10.0, 8.0, 9.0, 12.0], lam=1.0)
    r1 = run("R1", [10.0, 8.0, 9.0, 12.0])
    assert out[0] == pytest.approx(2 * r1[0])           # new drawdown counted twice
    assert out[1] == pytest.approx(r1[1])               # recovery: no extra penalty or bonus
    assert out[2] == pytest.approx(r1[2])


def test_r4_differential_sharpe_is_bounded_and_rewards_consistent_gains():
    eq = [10.0 * 1.001 ** k for k in range(200)]
    out = run("R4", eq)
    assert all(abs(x) <= 10.0 for x in out)
    eq2 = [10.0]
    for k in range(199):
        eq2.append(eq2[-1] * (1.02 if k % 2 else 0.981))
    assert sum(out) > sum(run("R4", eq2))


def test_r5_is_sparse():
    out = run("R5", [10.0, 12.0, 8.0, 15.0])
    assert out[:-1] == [0.0, 0.0] and out[-1] == pytest.approx(math.log(1.5))


def test_unknown_reward_rejected():
    with pytest.raises(ValueError):
        make_reward("R9")
