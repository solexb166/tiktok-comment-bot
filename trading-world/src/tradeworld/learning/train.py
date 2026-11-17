"""PPO training and out-of-sample evaluation for one isolated agent.

Isolation: every run owns its policy network, optimizer, rollout buffer, environment copies and
RNG seeds; nothing is shared between agents (architecture §4.2).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ..interface.actions import ActionMapper
from .env import TradeWorldEnv

PPO_DEFAULTS = dict(n_steps=1024, batch_size=512, n_epochs=10, learning_rate=3e-4, gamma=0.99,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.0)
POLICY_DEFAULTS = dict(net_arch=[64, 64], log_std_init=-0.5)
# Initial exploration noise (log std) on the gross-exposure dimension u. Because exposure is
# G = 10**(4u), Gaussian noise on u is multiplicative on leverage: std s gives E[G^2] inflated by
# exp(2 * (4 ln10 s)^2). With s = 0.6 that is ~1e26, so early exploration samples absurd leverage
# and the agent learns to stay flat before it can learn anything else. A smaller initial noise on
# u is an exploration setting, not a limit: PPO learns the noise level and can reach any u.
U_LOG_STD_INIT = -2.5


def make_env_fn(task, reward, episode_decisions, rank, seed, mapper=None, start=None, stop=None):
    def _f():
        env = TradeWorldEnv(task.md, task.profile, reward=reward, episode_decisions=episode_decisions,
                            extra=task.extra, mapper=mapper, start=start, stop=stop, **task.env_kwargs)
        env.reset(seed=seed * 1000 + rank)
        return env
    return _f


def train_ppo(task, reward: str, seed: int, total_steps: int, n_envs: int = 8,
              episode_decisions: int = 1000, ppo_kwargs: dict | None = None,
              policy_kwargs: dict | None = None, mapper: ActionMapper | None = None, log=print,
              u_log_std_init: float | None = U_LOG_STD_INIT):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.manual_seed(seed)
    venv = DummyVecEnv([make_env_fn(task, reward, episode_decisions, r, seed, mapper) for r in range(n_envs)])
    venv.seed(seed)
    kw = {**PPO_DEFAULTS, **(ppo_kwargs or {})}
    pk = {**POLICY_DEFAULTS, **(policy_kwargs or {})}
    pk["activation_fn"] = torch.nn.Tanh
    model = PPO("MlpPolicy", venv, seed=seed, policy_kwargs=pk, verbose=0, device="cpu", **kw)
    if u_log_std_init is not None:
        with torch.no_grad():
            model.policy.log_std[-1] = float(u_log_std_init)
    t0 = time.time()
    model.learn(total_timesteps=total_steps)
    log(f"trained {task.name}/{reward}/seed{seed}: {total_steps:,} steps in {time.time() - t0:.0f}s")
    return model


@dataclass
class EpisodeStats:
    log_growth: float
    died: bool
    steps: int
    mean_gross: float
    mean_net: float
    turnover: float
    costs: float
    flag_agreement: float | None


def run_episodes(policy, task, starts, episode_decisions, reward="R1", mapper=None, seed=0):
    """policy(obs_vector, env) -> action. Runs fixed-start episodes on task's market."""
    env = TradeWorldEnv(task.md, task.profile, reward=reward, episode_decisions=episode_decisions,
                        extra=task.extra, mapper=mapper, random_start=False, **task.env_kwargs)
    out = []
    for i, k0 in enumerate(starts):
        o, info = env.reset(seed=seed + i, options={"start_decision": int(k0)})
        e0 = info["equity"]
        gross, net, agree, w_prev, turn = [], [], [], 0.0, 0.0
        done = False
        while not done:
            a = policy(o, env)
            o, r, te, tr, info = env.step(a)
            done = te or tr
            if info["alive"]:
                g, n = info["gross_leverage"], info["net_leverage"]
                gross.append(g)
                net.append(n)
                turn += abs(n - w_prev)
                w_prev = n
                if task.extra is not None:
                    flag = task.extra[int(env.dec[env.k - 1]), 0]
                    if abs(n) > 1e-6:
                        agree.append(float(np.sign(n) == np.sign(flag)))
        e1 = info["equity"]
        out.append(EpisodeStats(
            log_growth=math.log(max(e1, 0.01 * e0) / e0), died=not info["alive"], steps=env.k,
            mean_gross=float(np.mean(gross)) if gross else 0.0, mean_net=float(np.mean(net)) if net else 0.0,
            turnover=turn / max(env.k, 1), costs=float(sum(info["costs"].values())),
            flag_agreement=float(np.mean(agree)) if agree else (None if task.extra is None else 0.0)))
    return out


def model_policy(model, deterministic=True):
    def _p(o, env):
        a, _ = model.predict(o, deterministic=deterministic)
        return a
    return _p


def matched_random_policy(mapper: ActionMapper, gross: float, switch_prob: float, seed: int):
    """Null-distribution agent: random direction, the agent's own average gross exposure and
    direction-change frequency (turnover-and-leverage matched, architecture §8.3)."""
    rng = np.random.default_rng(seed)
    state = {"d": float(rng.choice([-1.0, 1.0]))}
    u = mapper.u_for_gross(max(gross, 1e-4))

    def _p(o, env):
        if rng.random() < switch_prob:
            state["d"] = float(rng.choice([-1.0, 1.0]))
        return np.array([state["d"], u])
    return _p


def trace_episode(policy, task, start_decision, episode_decisions, reward="R1", mapper=None, seed=0):
    """One fixed-start episode, recorded decision by decision for replay: the mid price the agent
    saw, its net and gross exposure after acting, equity, and the regime flag where the task has one."""
    env = TradeWorldEnv(task.md, task.profile, reward=reward, episode_decisions=episode_decisions,
                        extra=task.extra, mapper=mapper, random_start=False, **task.env_kwargs)
    o, info = env.reset(seed=seed, options={"start_decision": int(start_decision)})
    mid = task.md.mid_c()[:, 0]
    rows = []
    done = False
    while not done:
        d = int(env.dec[env.k])
        eq_before = info["equity"]
        a = policy(o, env)
        o, r, te, tr, info = env.step(a)
        done = te or tr
        rows.append(dict(step=env.k, price=float(mid[d]), net=float(info["net_leverage"]) if info["alive"] else 0.0,
                         gross=float(info["gross_leverage"]) if info["alive"] else 0.0,
                         equity=float(info["equity"]), equity_before=float(eq_before), alive=bool(info["alive"]),
                         flag=None if task.extra is None else float(task.extra[d, 0])))
    return rows
