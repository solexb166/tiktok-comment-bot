"""Registry core: checksummed write-once storage, permanent identity, versions, lifecycle."""
import json

import pytest

from tradeworld.registry.registry import FrozenRuleError, LifecycleError, Registry, check_changes
from tradeworld.registry.store import IntegrityError, LocalBlobStore, WriteOnceError, write_once_json

ENV = dict(decision_interval=1, obs_resolution=1, window=30)
ACT = dict(log10_min=-4.0, log10_max=4.0, cap=None)


@pytest.fixture
def reg(tmp_path):
    return Registry(tmp_path / "reg")


def new(reg, task="S6", reward="R1", seed=0):
    return reg.create_agent(task=task, reward=reward, seed=seed, env_settings=ENV, action_space=ACT,
                            profile="A_synthetic", episode_decisions=1000)


def ckpt(reg, payload=b"weights", step=100):
    return dict(model_sha256=reg.blobs.put(payload, "zip"), state_sha256=reg.blobs.put(b"rng" + payload, "pkl"),
                step=step)


def version(reg, a, n, parent=None, payload=b"w", changes=None):
    c = ckpt(reg, payload + bytes([n]), 100 * n)
    reg.start_training(a, n, parent, changes or {})
    return reg.record_version(a, dict(version=n, parent_version=parent, changes=changes or {},
                                      checkpoints=[c], final_checkpoint=c))


def test_blob_store_is_content_addressed_deduplicated_and_verified(tmp_path):
    s = LocalBlobStore(tmp_path / "b")
    a = s.put(b"hello", "bin")
    assert s.put(b"hello", "bin") == a and len(list((tmp_path / "b").rglob("*.bin"))) == 1
    assert s.get(a) == b"hello" and s.exists(a) and s.verify_all() == []
    p = next((tmp_path / "b").rglob("*.bin"))
    p.write_bytes(b"tampered")
    with pytest.raises(IntegrityError):
        s.get(a)
    assert s.verify_all() == [a]


def test_write_once_refuses_overwrite(tmp_path):
    write_once_json(tmp_path / "r.json", {"a": 1})
    with pytest.raises(WriteOnceError):
        write_once_json(tmp_path / "r.json", {"a": 2})
    assert json.loads((tmp_path / "r.json").read_text()) == {"a": 1}


def test_identity_is_permanent_and_unique(reg):
    a, b = new(reg), new(reg, seed=1)
    assert a != b and a.startswith("ag-")
    i = reg.identity(a)
    assert i["task"] == "S6" and i["reward"] == "R1" and i["original_seed"] == 0 and i["created_at"]
    with pytest.raises(WriteOnceError):
        write_once_json(reg.root / "agents" / a / "identity.json", {})
    assert reg.find("S6", "R1", 1) == b and reg.state(a)["status"] == "created"


def test_versions_are_numbered_linked_and_immutable(reg):
    a = new(reg)
    version(reg, a, 1)
    v2 = version(reg, a, 2, parent=1, changes={"learning_rate": 1e-4})
    assert v2["parent_version"] == 1 and reg.state(a)["current_version"] == 2
    with pytest.raises(WriteOnceError):
        reg.record_version(a, dict(version=2, parent_version=1, checkpoints=[], final_checkpoint={}))
    with pytest.raises(LifecycleError):
        reg.start_training(a, 4, 2, {})                     # cannot skip a number
    assert [v["version"] for v in reg.versions(a)] == [1, 2]


def test_frozen_rules_cannot_change(reg):
    a = new(reg)
    with pytest.raises(FrozenRuleError):
        check_changes({"reward": "R2"})
    with pytest.raises(FrozenRuleError):
        reg.start_training(a, 1, None, {"window": 60})
    reg.start_training(a, 1, None, {})
    c = ckpt(reg)
    frozen = dict(reg.identity(a)["frozen"], reward="R2")
    with pytest.raises(FrozenRuleError):
        reg.record_version(a, dict(version=1, parent_version=None, checkpoints=[c], final_checkpoint=c, frozen=frozen))


def test_evaluations_are_write_once_and_failed_results_stay_failed(reg):
    a = new(reg)
    v1 = version(reg, a, 1)
    reg.record_evaluation(a, dict(version=1, round_id="gate-1", result="fail",
                                  checkpoint_sha256=v1["final_checkpoint"]["model_sha256"]))
    assert reg.state(a)["status"] == "failed"
    with pytest.raises(WriteOnceError):
        reg.record_evaluation(a, dict(version=1, round_id="gate-1", result="pass"))
    version(reg, a, 2, parent=1)
    assert reg.state(a)["status"] == "awaiting_evaluation"
    reg.record_evaluation(a, dict(version=2, round_id="round-0001", result="pass"))
    s = reg.state(a)
    assert s["status"] == "improved" and s["passed_on_version"] == 2 and s["passed_on_evaluation"] == 2
    assert s["training_attempts"] == 2 and s["failed_evaluations"] == 1 and s["champion"]["version"] == 2
    old = [e for e in reg.evaluations(a) if e["round_id"] == "gate-1"][0]
    assert old["result"] == "fail"                          # original failure is untouched


def test_first_time_pass_is_distinguished_from_pass_after_failures(reg):
    a = new(reg)
    version(reg, a, 1)
    reg.record_evaluation(a, dict(version=1, round_id="gate-1", result="pass"))
    assert reg.state(a)["status"] == "passed" and reg.state(a)["passed_on_evaluation"] == 1


def test_lifecycle_training_states_and_retirement(reg):
    a = new(reg)
    reg.start_training(a, 1, None, {})
    assert reg.state(a)["status"] == "training"
    with pytest.raises(LifecycleError):
        reg.start_training(a, 1, None, {})
    c = ckpt(reg)
    reg.record_version(a, dict(version=1, parent_version=None, checkpoints=[c], final_checkpoint=c))
    reg.start_training(a, 2, 1, {"train_steps": 1000})
    assert reg.state(a)["status"] == "continued_training"
    reg.abandon_training(a, 2, "container restarted")
    assert reg.state(a)["status"] == "awaiting_evaluation"
    reg.retire(a, "superseded")
    assert reg.state(a)["status"] == "retired"
    with pytest.raises(LifecycleError):
        reg.start_training(a, 2, 1, {})


def test_evaluation_must_reference_a_checkpoint_of_its_version(reg):
    a = new(reg)
    version(reg, a, 1)
    with pytest.raises(Exception):
        reg.record_evaluation(a, dict(version=1, round_id="r", result="fail", checkpoint_sha256="0" * 64))
