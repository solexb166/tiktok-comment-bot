"""Gymnasium environment: one agent, one account, one decision per step.

The environment drives the SAME code path as the multi-agent World (World.step), so training
episodes obey exactly the tested market and broker mechanics. Nothing here opens, closes or
resizes positions: the agent's action is passed to the execution engine unchanged.

Episode: `episode_decisions` consecutive decisions starting at a random decision (or `start`).
  terminated  -> the account died (equity <= 5% of initial)
  truncated   -> the episode length was reached
Reward: one of R1-R5 (rewards.py), from net-of-cost equity at consecutive decision times.
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from ..brokerspec.profile import EnvProfile
from ..data.market_data import MarketData
from ..execution.engine import MarketView
from ..interface.actions import ActionMapper
from ..interface.observation import account_feature_names, build_observation
from ..replay.replay import MARKET_FEATURES, Replay
from ..world.world import AgentSpec, World
from .features import BOUND, ObsEncoder
from .rewards import make_reward

AGENT_ID = "agent"


class TradeWorldEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, md: MarketData, profile: EnvProfile, *, reward: str = "R1",
                 reward_kwargs: dict | None = None, mapper: ActionMapper | None = None,
                 episode_decisions: int = 2016, decision_interval: int = 5, obs_resolution: int = 5,
                 window: int = 120, extra: np.ndarray | None = None, random_start: bool = True,
                 start: int | None = None, stop: int | None = None, record: bool = False):
        """`extra` (optional, synthetic experiments only): float array [T, E] aligned with M1 rows;
        the agent sees row d at a decision made at the close of row d (never later rows).
        `start`/`stop` restrict decisions to M1 indices in [start, stop) (data splits)."""
        super().__init__()
        self.md, self.profile = md, profile
        n = profile.universe.n
        self.mapper = mapper or ActionMapper.primary(n)
        if self.mapper.n != n:
            raise ValueError("mapper size does not match the universe")
        self.view = MarketView(md, profile)
        self.replay = Replay(md, decision_interval, obs_resolution, window)
        self.settings = dict(decision_interval=decision_interval, obs_resolution=obs_resolution, window=window)
        dec = self.replay.decision_indices
        if start is not None:
            dec = dec[dec >= start]
        if stop is not None:
            dec = dec[dec < stop]
        if len(dec) < episode_decisions + 1:
            raise ValueError(f"only {len(dec)} decisions available, need {episode_decisions + 1}")
        self.dec_all, self.L, self.random_start, self.record = dec, int(episode_decisions), random_start, record
        self.extra = None if extra is None else np.asarray(extra, dtype=np.float64)
        if self.extra is not None and (self.extra.ndim != 2 or self.extra.shape[0] != md.T):
            raise ValueError("extra must have shape [T, E]")
        self.reward_name, self.reward_kwargs = reward, dict(reward_kwargs or {})
        self.reward_fn = make_reward(reward, **self.reward_kwargs)
        self.encoder = ObsEncoder(n, window, len(MARKET_FEATURES), len(account_feature_names(profile.universe.symbols)),
                                  0 if self.extra is None else self.extra.shape[1])
        self.observation_space = gym.spaces.Box(-BOUND, BOUND, (self.encoder.dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.mapper.size,), np.float32)
        self.world = None

    # ------------------------------------------------------------------
    def _policy(self, obs):
        return self._pending

    @property
    def account(self):
        return self.world.slots[0].acct

    def _equity(self, d: int) -> float:
        a = self.account
        return a.equity(self.view.prices_close(d)) if a.alive else a.balance

    def _obs(self) -> np.ndarray:
        d = int(self.dec[self.k])
        o = build_observation(self.replay, self.view, self.account, d, self.world.slots[0].last_report)
        return self.encoder.encode(o, None if self.extra is None else self.extra[d])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        hi = len(self.dec_all) - self.L - 1
        if "start_decision" in options:
            k0 = int(options["start_decision"])
        elif self.random_start:
            k0 = int(self.np_random.integers(0, hi + 1))
        else:
            k0 = 0
        if not 0 <= k0 <= hi:
            raise ValueError("start_decision out of range")
        self.dec = self.dec_all[k0:k0 + self.L + 1]
        self._world_seed = world_seed = int(self.np_random.integers(0, 2 ** 31 - 1))
        self._pending = np.zeros(self.mapper.size)
        self.world = World(self.md, self.profile, [AgentSpec(AGENT_ID, self._policy, self.mapper)],
                           seed=world_seed, record=self.record, view=self.view, replay=self.replay,
                           **self.settings)
        self.k = 0
        self.e0 = self.account.initial
        self.reward_fn.reset(self.e0)
        self._e = self._equity(int(self.dec[0]))
        self.ep_return = 0.0
        return self._obs(), self._info(0.0)

    def step(self, action):
        if self.world is None:
            raise RuntimeError("call reset() first")
        self._pending = np.asarray(action, dtype=np.float64)
        self.world.step(self.dec, self.k)
        self.k += 1
        e_now = self._equity(int(self.dec[self.k]))
        terminated = not self.account.alive
        truncated = (self.k >= self.L) and not terminated
        r = float(self.reward_fn(self._e, e_now, terminated or truncated))
        if not math.isfinite(r):
            raise FloatingPointError(f"non-finite reward {r}")
        self._e = e_now
        self.ep_return += r
        return self._obs(), r, terminated, truncated, self._info(r)

    def _info(self, r: float) -> dict:
        a = self.account
        d = int(self.dec[self.k])
        p = self.view.prices_close(d)
        eq = self._e
        notional = a.units * p.mid * p.conv
        return dict(equity=eq, alive=a.alive, gross_leverage=float(np.sum(np.abs(notional)) / eq) if eq > 0 else math.inf,
                    net_leverage=float(np.sum(notional) / eq) if eq > 0 else math.inf, decision=self.k,
                    t=int(self.view.times[d]), costs=dict(a.costs_total))

    def result(self):
        """Full World result (records only if record=True) for the current episode."""
        return self.world.result()
