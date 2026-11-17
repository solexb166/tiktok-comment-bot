"""Complete training history: every training episode and every learning update, kept in full.

EpisodeStatsWrapper  - wraps a TradeWorldEnv (read-only: it never changes actions, rewards or
                       observations) and summarises each training episode's behaviour.
HistoryCallback      - collects those episode summaries plus SB3's learning metrics (losses,
                       exploration noise, KL, clip fraction, explained variance) at every update.
CheckpointCallback   - saves a full checkpoint every `every` steps and appends it to an
                       in-flight log, so a container restart loses at most `every` steps.
"""
from __future__ import annotations

import io
import math

import gymnasium as gym
import numpy as np
import pandas as pd

from ..registry.store import append_jsonl

LEVERAGE_BANDS = (1.0, 3.0, 10.0, 100.0)


class EpisodeStatsWrapper(gym.Wrapper):
    def __init__(self, env, market=None):
        """`market` (optional, quality.KnownMarket): adds the share of ruinous decisions and the
        decision-quality score to each training episode's summary."""
        super().__init__(env)
        self.market = market

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        a = self.env.unwrapped.account
        self._s = dict(ret=0.0, n=0, gross=[], net=[], turn=0.0, prev_net=0.0, peak=info["equity"],
                       mdd=0.0, e0=info["equity"], fills0=sum(e.kind == "fill" for e in a.events))
        return obs, info

    def step(self, action):
        obs, r, te, tr, info = self.env.step(action)
        s = self._s
        s["ret"] += r
        s["n"] += 1
        if info["alive"]:
            g = info["gross_leverage"]
            n = info["net_leverage"]
            s["gross"].append(g if math.isfinite(g) else 0.0)
            s["net"].append(n if math.isfinite(n) else 0.0)
            s["turn"] += abs(s["net"][-1] - s["prev_net"])
            s["prev_net"] = s["net"][-1]
        eq = info["equity"]
        s["peak"] = max(s["peak"], eq)
        if s["peak"] > 0:
            s["mdd"] = max(s["mdd"], 1.0 - max(eq, 0.0) / s["peak"])
        if te or tr:
            a = self.env.unwrapped.account
            g = np.array(s["gross"]) if s["gross"] else np.zeros(1)
            stats = dict(
                episode_reward=s["ret"], length=s["n"], final_equity=eq,
                growth=math.log(max(eq, 0.01 * s["e0"]) / s["e0"]), died=not info["alive"],
                max_drawdown=s["mdd"], mean_gross=float(g.mean()), max_gross=float(g.max()),
                p95_gross=float(np.percentile(g, 95)), mean_net=float(np.mean(s["net"])) if s["net"] else 0.0,
                turnover=s["turn"] / max(s["n"], 1), time_in_market=float(np.mean(g > 1e-6)),
                trades=sum(e.kind == "fill" for e in a.events) - s["fills0"],
                costs=float(sum(a.costs_total.values())),
                **{f"share_gross_over_{int(b)}x": float(np.mean(g > b)) for b in LEVERAGE_BANDS},
            )
            if self.market is not None and s["net"]:
                net = np.array(s["net"])
                ruin = self.market.ruinous(net)
                g_star = self._g_star()
                ok = ~ruin
                stats["share_ruinous"] = float(np.mean(ruin))
                stats["decision_quality"] = float(np.mean(self.market.growth(net[ok]) / g_star)) if ok.any() else None
            info = dict(info, episode_stats=stats)
        return obs, r, te, tr, info

    def _g_star(self):
        if not hasattr(self, "_gs"):
            self._gs = self.market.optimal()[1]
        return self._gs


def _metric_writer(rows: list):
    from stable_baselines3.common.logger import KVWriter

    class ListWriter(KVWriter):
        def write(self, key_values, key_excluded, step=0):
            rows.append({"step": int(step), **{k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                                                for k, v in key_values.items()}})

        def close(self):
            pass
    return ListWriter()


def history_callback():
    from stable_baselines3.common.callbacks import BaseCallback

    class HistoryCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episodes, self.updates = [], []

        def _on_training_start(self):
            from stable_baselines3.common.logger import Logger
            self.model.set_logger(Logger(folder=None, output_formats=[_metric_writer(self.updates)]))

        def _on_step(self):
            for i, info in enumerate(self.locals.get("infos", [])):
                if "episode_stats" in info:
                    self.episodes.append(dict(step=int(self.num_timesteps), env=i, **info["episode_stats"]))
            return True

        def _on_rollout_end(self):
            ls = self.model.policy.log_std.detach().cpu().numpy()
            self.model.logger.record("explore/log_std_direction_mean", float(ls[:-1].mean()))
            self.model.logger.record("explore/log_std_exposure", float(ls[-1]))

        def frames(self):
            return pd.DataFrame(self.episodes), pd.DataFrame(self.updates)

    return HistoryCallback()


def frame_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, compression="zstd", index=False)
    return buf.getvalue()


def checkpoint_callback(store, venv, every: int, inflight_log, meta: dict, history=None):
    from stable_baselines3.common.callbacks import BaseCallback

    from .checkpoint import save_checkpoint

    class CheckpointCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.records = []
            self._next = None

        def _on_training_start(self):
            self._next = (self.model.num_timesteps // every + 1) * every

        def _on_step(self):
            if self.model.num_timesteps >= self._next:
                self.save("periodic")
                self._next += every
            return True

        def save(self, kind: str):
            rec = save_checkpoint(store, self.model, venv, dict(meta, kind=kind))
            if history is not None:
                ep, up = history.frames()
                rec["history_episodes_sha256"] = store.put(frame_bytes(ep), "parquet") if len(ep) else None
                rec["history_updates_sha256"] = store.put(frame_bytes(up), "parquet") if len(up) else None
            self.records.append(rec)
            if inflight_log is not None:
                append_jsonl(inflight_log, rec)
            return rec

    return CheckpointCallback()
