"""Complete training checkpoints: everything needed to continue an agent where it stopped.

A checkpoint is two content-addressed blobs:
  model  - the Stable-Baselines3 archive: policy weights, optimizer (Adam) state, learned
           exploration noise (log_std), hyperparameters, learning-rate schedule, step counter.
  state  - everything SB3 does not save: PyTorch, NumPy and Python random-number generator
           states; each practice-market environment's random stream and its current episode's
           account random stream; SB3 bookkeeping (updates done, episode count).
Plus a JSON record with checksums, the step, the training configuration, the code version and
the training-data fingerprints.

Resuming restores all of it. One thing cannot be restored exactly: an episode that was in
progress when the checkpoint was taken (the environments start new episodes from their restored
random streams). Resumption is therefore faithful in learned state and random streams, but not
bit-for-bit identical to a run that was never interrupted.
"""
from __future__ import annotations

import io
import pickle
import random

import numpy as np

STATE_FORMAT = 1


def _env_core(env):
    return getattr(env, "unwrapped", env)


def capture_state(model, venv) -> dict:
    import torch
    envs = []
    for e in getattr(venv, "envs", []):
        core = _env_core(e)
        world = getattr(core, "world", None)
        envs.append(dict(
            np_random=core.np_random.bit_generator.state if getattr(core, "_np_random", None) is not None else None,
            account_rng=world.slots[0].acct.rng.bit_generator.state if world is not None else None,
            episode_start_decision=int(core.dec_all.searchsorted(core.dec[0])) if world is not None else None,
            decision_in_episode=int(core.k) if world is not None else None,
        ))
    return dict(
        format=STATE_FORMAT,
        torch_rng=torch.get_rng_state().numpy().tobytes(),
        numpy_global=np.random.get_state(),
        python_random=random.getstate(),
        envs=envs,
        sb3=dict(num_timesteps=int(model.num_timesteps), n_updates=int(getattr(model, "_n_updates", 0)),
                 episode_num=int(getattr(model, "_episode_num", 0))),
    )


def restore_state(model, venv, state: dict) -> None:
    """Restore RNG streams after loading a checkpoint, then start fresh episodes from them."""
    import torch
    if state.get("format") != STATE_FORMAT:
        raise ValueError("unknown checkpoint state format")
    torch.set_rng_state(torch.from_numpy(np.frombuffer(state["torch_rng"], dtype=np.uint8).copy()))
    np.random.set_state(state["numpy_global"])
    random.setstate(state["python_random"])
    envs = getattr(venv, "envs", [])
    if len(envs) != len(state["envs"]):
        raise ValueError(f"checkpoint has {len(state['envs'])} environments, resume has {len(envs)}")
    for e, s in zip(envs, state["envs"]):
        if s["np_random"] is not None:
            _env_core(e).np_random.bit_generator.state = s["np_random"]
    obs = venv.reset()                                   # new episodes from the restored streams
    model._last_obs = obs
    model._last_episode_starts = np.ones((venv.num_envs,), dtype=bool)
    model._n_updates = state["sb3"]["n_updates"]
    model._episode_num = state["sb3"]["episode_num"]


def model_bytes(model) -> bytes:
    buf = io.BytesIO()
    model.save(buf)
    return buf.getvalue()


def save_checkpoint(store, model, venv, meta: dict) -> dict:
    """Store model + state blobs; return the checkpoint record (with checksums)."""
    m_sha = store.put(model_bytes(model), "zip")
    s_sha = store.put(pickle.dumps(capture_state(model, venv), protocol=4), "pkl")
    return dict(meta, model_sha256=m_sha, state_sha256=s_sha, step=int(model.num_timesteps))


def load_checkpoint(store, record: dict, venv, device: str = "cpu", restore: bool = True):
    """Load a checkpoint's model onto `venv`; restore its random streams if it has them."""
    from stable_baselines3 import PPO
    model = PPO.load(io.BytesIO(store.get(record["model_sha256"])), env=venv, device=device)
    if restore and record.get("state_sha256"):
        restore_state(model, venv, pickle.loads(store.get(record["state_sha256"])))
    else:
        model._last_obs = venv.reset()
        model._last_episode_starts = np.ones((venv.num_envs,), dtype=bool)
    return model
