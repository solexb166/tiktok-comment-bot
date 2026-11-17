"""Training attempts as registry versions: new agents (v1) and continued training (v2, v3, ...).

A continued version starts from its parent's learned weights, optimizer state, learned
exploration noise and random streams (or from any earlier checkpoint), so previous experience
is kept. Only the training settings in registry.ALLOWED_CHANGES may differ, each recorded with
its old and new value. The market, scoring rule, simulator and environment settings come from
the agent's frozen identity and cannot be changed here.

If the container restarts mid-training, the in-flight log keeps every periodic checkpoint taken
so far; `recover` abandons the interrupted attempt and continues from its last checkpoint as a
new version, so at most `checkpoint_every` steps are lost.
"""
from __future__ import annotations

import pickle
import platform
import time

import numpy as np

from ..registry.datasets import DataLedger
from ..registry.registry import ALLOWED_CHANGES, Registry, RegistryError, code_version, check_changes
from ..registry.store import read_jsonl
from . import synthetic_tasks as st
from .checkpoint import load_checkpoint, restore_state
from .env import TradeWorldEnv
from .evaluate import mapper_from
from .history import EpisodeStatsWrapper, checkpoint_callback, frame_bytes, history_callback
from .train import POLICY_DEFAULTS, PPO_DEFAULTS, U_LOG_STD_INIT

TRAIN_STEPS_DATA = 200_000           # as the original gate
HYPER_KEYS = ("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef",
              "vf_coef", "max_grad_norm")


def hyperparameters(model) -> dict:
    h = {k: getattr(model, k) for k in HYPER_KEYS}
    h["learning_rate"] = float(model.lr_schedule(1.0))
    h["clip_range"] = float(model.clip_range(1.0))
    h["log_std"] = [float(x) for x in model.policy.log_std.detach().cpu().numpy()]
    h["policy_kwargs"] = {k: (str(v) if k == "activation_fn" else v) for k, v in model.policy_kwargs.items()}
    return h


def apply_changes(model, changes: dict) -> dict:
    """Apply whitelisted setting changes; return {setting: {from, to}}."""
    import torch
    from stable_baselines3.common.utils import FloatSchedule
    check_changes(changes)
    before = hyperparameters(model)
    for k, v in changes.items():
        if k == "learning_rate":
            model.learning_rate = float(v)
            model.lr_schedule = FloatSchedule(float(v))
        elif k == "clip_range":
            model.clip_range = FloatSchedule(float(v))
        elif k == "log_std":
            with torch.no_grad():
                cur = model.policy.log_std.detach().clone()
                if isinstance(v, dict):                           # {"direction": x, "exposure": y}
                    if "direction" in v:
                        cur[:-1] = float(v["direction"])
                    if "exposure" in v:
                        cur[-1] = float(v["exposure"])
                else:
                    cur[:] = torch.as_tensor(v, dtype=cur.dtype)
                model.policy.log_std.copy_(cur)
        elif k in ("ent_coef", "vf_coef", "max_grad_norm", "n_epochs", "batch_size"):
            setattr(model, k, type(getattr(model, k))(v))
        elif k in ("gamma", "gae_lambda"):
            setattr(model, k, float(v))
            setattr(model.rollout_buffer, k, float(v))
    after = hyperparameters(model)
    return {k: dict(**{"from": before[k], "to": after[k]}) for k in after if before[k] != after[k]}


def build_venv(identity: dict, tasks: list, n_envs: int, seed: int):
    from stable_baselines3.common.vec_env import DummyVecEnv
    f = identity["frozen"]
    mapper = mapper_from(identity)

    def fn(rank):
        def _f():
            t = tasks[rank % len(tasks)]
            env = EpisodeStatsWrapper(TradeWorldEnv(t.md, t.profile, reward=f["reward"], extra=t.extra, mapper=mapper,
                                                    episode_decisions=f["episode_decisions"], **f["env_settings"]))
            env.reset(seed=seed * 1000 + rank)
            return env
        return _f
    v = DummyVecEnv([fn(r) for r in range(n_envs)])
    v.seed(seed)
    return v


def _find_checkpoint(reg: Registry, agent_id: str, sha: str) -> dict:
    for v in reg.versions(agent_id):
        for c in v.get("checkpoints", []) + [v["final_checkpoint"]]:
            if c.get("model_sha256") == sha:
                return c
    inflight = reg.root / "agents" / agent_id / "inflight"
    for p in sorted(inflight.glob("*.jsonl")) if inflight.exists() else []:
        for c in read_jsonl(p):
            if c["model_sha256"] == sha:
                return c
    raise RegistryError(f"{agent_id} has no checkpoint {sha}")


def train_version(reg: Registry, agent_id: str, *, steps: int, parent: int | None = None, changes: dict | None = None,
                  reason: str = "", n_envs: int = 8, checkpoint_every: int = 250_000, seed: int | None = None,
                  log=print, recovered_from: dict | None = None) -> dict:
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList

    changes = dict(changes or {})
    check_changes(changes)
    ident = reg.identity(agent_id)
    n = reg.next_version(agent_id)
    ledger = DataLedger(reg.root)

    # ---- training data: the parent's (default) or fresh, never-used seeds
    if "train_data_seeds" in changes:
        seeds = []
        for s in changes["train_data_seeds"]:
            s = ledger.fresh_seed(ident["task"], "train") if s == "fresh" else int(s)
            ledger.record(ident["task"], s, "train", note=f"{agent_id} v{n}")
            seeds.append(s)
    elif parent is not None:
        seeds = [d["seed"] for d in reg.version(agent_id, parent)["training"]["train_data"]]
    else:
        seeds = [ident["original_seed"]]
    tasks = [st.build(ident["task"], s, TRAIN_STEPS_DATA) for s in seeds]
    n_envs = int(changes.get("n_envs", n_envs))
    seed = int(seed if seed is not None else ident["original_seed"] * 100 + n)
    venv = build_venv(ident, tasks, n_envs, seed)

    # ---- model: fresh, or the parent's checkpoint with its full state
    parent_ckpt = None
    rng_restored = False
    if parent is None:
        torch.manual_seed(seed)
        pk = dict(POLICY_DEFAULTS, activation_fn=torch.nn.Tanh)
        model = PPO("MlpPolicy", venv, seed=seed, policy_kwargs=pk, verbose=0, device="cpu", **PPO_DEFAULTS)
        with torch.no_grad():
            model.policy.log_std[-1] = float(U_LOG_STD_INIT)
    else:
        sha = changes.get("resume_from_checkpoint") or reg.version(agent_id, parent)["final_checkpoint"]["model_sha256"]
        parent_ckpt = _find_checkpoint(reg, agent_id, sha)
        same_envs = parent_ckpt.get("n_envs") == n_envs and parent_ckpt.get("train_data_seeds") == seeds
        model = load_checkpoint(reg.blobs, parent_ckpt, venv, restore=False)
        if parent_ckpt.get("state_sha256"):
            state = pickle.loads(reg.blobs.get(parent_ckpt["state_sha256"]))
            if same_envs:
                restore_state(model, venv, state)
                rng_restored = True
            else:                                   # different data or env count: global streams only
                import random
                torch.set_rng_state(torch.from_numpy(np.frombuffer(state["torch_rng"], dtype=np.uint8).copy()))
                np.random.set_state(state["numpy_global"])
                random.setstate(state["python_random"])
                rng_restored = "global_only"
    applied = apply_changes(model, {k: v for k, v in changes.items()
                                    if k not in ("train_steps", "train_data_seeds", "n_envs", "checkpoint_every",
                                                 "resume_from_checkpoint")})
    checkpoint_every = int(changes.get("checkpoint_every", checkpoint_every))

    # ---- train with periodic checkpoints and full history
    start_info = dict(n_envs=n_envs, train_data_seeds=seeds, seed=seed, step_start=int(model.num_timesteps),
                      parent_checkpoint=None if parent_ckpt is None else
                      {k: parent_ckpt.get(k) for k in ("model_sha256", "state_sha256", "step")})
    reg.start_training(agent_id, n, parent, changes, info=start_info)
    meta = dict(version=n, n_envs=n_envs, train_data_seeds=seeds)
    hist = history_callback()
    inflight = reg.root / "agents" / agent_id / "inflight" / f"v{n:04d}.jsonl"
    ck = checkpoint_callback(reg.blobs, venv, checkpoint_every, inflight, meta, history=hist)
    step0 = int(model.num_timesteps)
    t0 = time.time()
    try:
        model.learn(total_timesteps=int(steps), reset_num_timesteps=False, callback=CallbackList([hist, ck]))
        final = ck.save("final")
    except BaseException as e:
        record_interrupted(reg, agent_id, n, f"{type(e).__name__}: {e}")
        raise
    ep, up = hist.frames()
    rec = dict(
        version=n, parent_version=parent, kind="initial" if parent is None else "continued",
        reason=reason, changes=changes, recovered_from=recovered_from,
        parent_checkpoint=None if parent_ckpt is None else {k: parent_ckpt.get(k) for k in ("model_sha256", "state_sha256", "step")},
        training=dict(
            algorithm="PPO", library=dict(sb3=stable_baselines3.__version__, torch=torch.__version__,
                                          python=platform.python_version(), numpy=np.__version__),
            steps_added=int(model.num_timesteps - step0), step_start=step0, step_end=int(model.num_timesteps),
            wall_seconds=round(time.time() - t0, 1), n_envs=n_envs, seed=seed, rng_restored=rng_restored,
            hyperparameters=hyperparameters(model), settings_changes=applied,
            train_data=[dict(task=t.name, seed=s, data_hash=t.md.content_hash(), steps=TRAIN_STEPS_DATA)
                        for t, s in zip(tasks, seeds)]),
        checkpoints=ck.records, final_checkpoint=final,
        history=dict(episodes_sha256=reg.blobs.put(frame_bytes(ep), "parquet") if len(ep) else None,
                     updates_sha256=reg.blobs.put(frame_bytes(up), "parquet") if len(up) else None,
                     episodes=len(ep), updates=len(up), complete=True),
        code_version=code_version(),
    )
    log(f"{agent_id} v{n}: trained {rec['training']['steps_added']:,} steps in {rec['training']['wall_seconds']:.0f}s")
    return reg.record_version(agent_id, rec)


def record_interrupted(reg: Registry, agent_id: str, version: int, reason: str) -> dict | None:
    """An attempt that stopped early is still an attempt: if it saved checkpoints, record it as
    its own (interrupted) version with everything up to its last checkpoint."""
    if any(v["version"] == version for v in reg.versions(agent_id)):
        return reg.version(agent_id, version)
    recs = read_jsonl(reg.root / "agents" / agent_id / "inflight" / f"v{version:04d}.jsonl")
    started = [e for e in reg.events(agent_id) if e["kind"] == "training_started" and e["version"] == version]
    if not started:
        raise RegistryError(f"{agent_id} v{version} was never started")
    ev = started[-1]
    if not recs:
        reg.abandon_training(agent_id, version, reason + " (no checkpoint was saved)")
        return None
    info, last = ev.get("info", {}), recs[-1]
    return reg.record_version(agent_id, dict(
        version=version, parent_version=ev.get("parent"), kind="interrupted", reason=reason,
        changes=ev.get("changes", {}), parent_checkpoint=info.get("parent_checkpoint"),
        training=dict(algorithm="PPO", step_start=info.get("step_start"), step_end=last["step"],
                      steps_added=None if info.get("step_start") is None else last["step"] - info["step_start"],
                      n_envs=last["n_envs"], seed=info.get("seed"),
                      train_data=[dict(seed=s) for s in last["train_data_seeds"]]),
        checkpoints=recs, final_checkpoint=last,
        history=dict(episodes_sha256=last.get("history_episodes_sha256"),
                     updates_sha256=last.get("history_updates_sha256"), complete=False,
                     note="history up to the last checkpoint before the interruption"),
        code_version=code_version()))


def recover(reg: Registry, agent_id: str, version: int, *, steps: int, reason: str = "resumed after interruption",
            log=print) -> dict:
    """Record an interrupted attempt as its own version (if it is not yet), then continue from
    its last checkpoint as the next version."""
    v = record_interrupted(reg, agent_id, version, "interrupted (detected on recovery)")
    if v is None:
        raise RegistryError(f"{agent_id} v{version} saved no checkpoint before it stopped")
    changes = {k: x for k, x in v.get("changes", {}).items() if k in ALLOWED_CHANGES and k != "resume_from_checkpoint"}
    return train_version(reg, agent_id, steps=steps, parent=version, changes=changes, reason=reason,
                         n_envs=v["training"]["n_envs"], log=log,
                         recovered_from=dict(version=version, model_sha256=v["final_checkpoint"]["model_sha256"],
                                             step=v["final_checkpoint"]["step"]))
