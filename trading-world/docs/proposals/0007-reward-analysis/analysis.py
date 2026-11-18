"""Proposal 0007: offline analysis of the S6 reward landscape (read only; changes no reward, agent or record).

  python docs/proposals/0007-reward-analysis/analysis.py exact     per-decision expected reward (quadrature)
  python docs/proposals/0007-reward-analysis/analysis.py values    Monte Carlo values of constant-size policies
  python docs/proposals/0007-reward-analysis/analysis.py learn     toy learning dynamics (validated against Exp 1/1b)
  python docs/proposals/0007-reward-analysis/analysis.py hack      exploitation probes for each candidate
Results are written as JSON next to this file.

Market: S6 exactly as synthetic_tasks.build("S6"): each bar, with probability p = 1/300 a crash of J = 10%
(log return ln 0.9), otherwise log return m + sigma Z with m = mu - sigma^2/2, mu = 5e-4, sigma = 3e-3.
Account: position G = exposure / equity, re-set every bar; equity multiplier 1 + G (e^x - 1); no negative-
balance protection (environment A: a wiped-out account keeps its negative balance); death when equity
<= 5% of the initial equity (absorbing; the episode ends).
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
MU, SIGMA, P, J, DELTA = 5e-4, 3e-3, 1 / 300, 0.10, 0.05
M = MU - 0.5 * SIGMA ** 2
Z, W = np.polynomial.hermite_e.hermegauss(96)
W = W / math.sqrt(2 * math.pi)


# ---------------------------------------------------------------------------- rewards
# Each takes the equity before (alive, > DELTA) and the unprotected equity after (may be negative),
# both relative to the initial equity E0 = 1, and returns the step reward.
def r1(eps=0.01):
    return lambda w0, w1: np.log(np.maximum(w1, eps)) - np.log(np.maximum(w0, eps))


def full_liability_absolute(level=DELTA):
    """REJECTED variant (kept for the record): log utility of ABSOLUTE equity above `level`, tangent line
    below. Not scale-invariant: the charge grows with the account's wealth, so wealthy survivors that
    crash produce rewards of order -1e8 (see values_absolute_fl in the report)."""
    def u(w):
        return np.where(w >= level, np.log(np.maximum(w, level)), math.log(level) + (w - level) / level)
    return lambda w0, w1: u(w1) - u(w0)


def full_liability(c=0.01):
    """FL(c): per-step log return of equity, ln(M) with M = equity after / equity before, for M >= c; for a
    one-bar loss deeper than 1 - c (including losses beyond the account's equity, M < 0, which environment A
    keeps as a negative balance), the tangent line of ln at c: ln c + (M - c) / c. Identical to R1 on every
    bar where M >= c and the account is above R1's floor; differs only where R1 truncates the loss."""
    def r(w0, w1):
        m = w1 / w0
        return np.where(m >= c, np.log(np.maximum(m, c)), math.log(c) + (m - c) / c)
    return r


def linear():
    return lambda w0, w1: w1 - w0


REWARDS = {
    "R1 (current, eps 0.01)": dict(f=r1(0.01), gamma=0.99),
    "R1, eps 1e-6 (larger death cost)": dict(f=r1(1e-6), gamma=0.99),
    "R1, gamma 0.999": dict(f=r1(0.01), gamma=0.999),
    "R1, undiscounted 1000-step episode": dict(f=r1(0.01), gamma=1.0),
    "FL(0.01): full-liability log, tangent at R1's floor": dict(f=full_liability(0.01), gamma=0.99),
    "FL(0.05): full-liability log, tangent at a 95% loss": dict(f=full_liability(0.05), gamma=0.99),
    "FL(0.05), gamma 0.999": dict(f=full_liability(0.05), gamma=0.999),
    "R2 linear (historical control)": dict(f=linear(), gamma=0.99),
}


# ---------------------------------------------------------------------------- exact per-decision expectation
def exact(grid=(0.5, 1, 2, 2.93, 4, 6, 8, 9, 9.5, 9.9, 10, 12, 15, 20, 30, 40, 55.6, 70, 100, 150)):
    """Expected one-step reward from equity 1 (no termination effects), by Gauss-Hermite quadrature."""
    out = {}
    normal_x = np.expm1(M + SIGMA * Z)
    for name, spec in REWARDS.items():
        f = spec["f"]
        rows = []
        for G in grid:
            w1n = 1 + G * normal_x
            normal = float((f(np.ones_like(w1n), w1n) * W).sum())
            crash = float(f(np.ones(1), np.array([1 - G * J]))[0])
            p_die_normal = float(((w1n <= DELTA) * W).sum())
            rows.append(dict(G=G, normal_branch=normal, crash_branch=crash, expected=(1 - P) * normal + P * crash,
                             p_death=P * (1 - G * J <= DELTA) + (1 - P) * p_die_normal))
        out[name] = rows
    kelly = []
    for G in grid:
        g = (1 - P) * float((np.log(np.maximum(1 + G * normal_x, 1e-300)) * W).sum())
        g += P * (math.log(1 - G * J) if G * J < 1 else -math.inf)
        kelly.append(dict(G=G, kelly=g))
    out["true Kelly growth g(G) (no floor, no death rule)"] = kelly
    out["no-crash Kelly size mu/sigma^2"] = MU / SIGMA ** 2
    return out


# ---------------------------------------------------------------------------- Monte Carlo constant-policy values
def _value_job(args):
    name, G, gamma, H, N, seed = args
    f = REWARDS[name]["f"]
    rng = np.random.default_rng(seed)
    w = np.ones(N)
    alive = np.ones(N, bool)
    tot = np.zeros(N)
    disc = 1.0
    deaths_t = np.full(N, -1)
    for t in range(H):
        crash = rng.random(N) < P
        x = np.where(crash, math.log(1 - J), M + SIGMA * rng.standard_normal(N))
        w1 = w * (1 + G * np.expm1(x))
        r = f(w, w1)
        tot += disc * np.where(alive, r, 0.0)
        died = alive & (w1 <= DELTA)
        deaths_t[died] = t
        alive &= ~died
        w = np.where(alive, w1, w)
        disc *= gamma
    se = float(tot.std(ddof=1) / math.sqrt(N))
    return name, G, dict(value=float(tot.mean()), se=se, death_rate=float(1 - alive.mean()),
                         median_death_step=float(np.median(deaths_t[deaths_t >= 0])) if (deaths_t >= 0).any() else None,
                         survivor_median_log_wealth=float(np.median(np.log(w[alive]))) if alive.any() else None)


def values(grid=(0.5, 1, 2, 2.93, 4, 6, 8, 9.5, 10, 12, 15, 20, 30, 40, 55, 70, 100), N=4000, workers=4):
    jobs = []
    for name, spec in REWARDS.items():
        g = spec["gamma"]
        H = 1000 if g < 0.995 else (4000 if g < 1 else 1000)
        for G in grid:
            jobs.append((name, G, g, H, N, 12345))            # common random numbers across sizes
    out = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for name, G, res in pool.imap_unordered(_value_job, jobs):
            out.setdefault(name, {})[G] = res
    return {k: [dict(G=G, **v[G]) for G in sorted(v)] for k, v in out.items()}


# ---------------------------------------------------------------------------- toy learning dynamics
def q2_score(G):
    """Q2.0-style per-decision score: g(G)/g(G*) for non-ruinous longs, else the floor (S6: -21.35)."""
    normal_x = np.expm1(M + SIGMA * Z)
    def g(x):
        if x * J >= 1:
            return -math.inf
        return (1 - P) * float((np.log(1 + x * normal_x) * W).sum()) + P * math.log(1 - x * J)
    grid = np.linspace(0.01, 9.99, 4000)
    gs = np.array([g(x) for x in grid])
    gstar = gs.max()
    floor = g(0.95 / J) / gstar
    return np.array([floor if x >= 0.95 / J else max(g(x) / gstar, floor) for x in np.atleast_1d(G)])


def _learn_job(args):
    name, s0, theta0, seed, iters, n_ep, H, K, lr = args[:9]
    spread = args[9] if len(args) > 9 else 0.3
    spec = REWARDS[name]
    f, gamma = spec["f"], spec["gamma"]
    rng = np.random.default_rng(seed)
    theta = theta0 + spread * rng.standard_normal(K)        # per-state mean ln(size); states are pure noise
    m_adam, v_adam = np.zeros(K), np.zeros(K)
    hist = []
    for it in range(iters):
        s = s0 * math.exp(-math.log(6.0) * it / max(iters - 1, 1))   # exploration shrinks ~6x, as observed
        st = rng.integers(0, K, (n_ep, H))
        eps = rng.standard_normal((n_ep, H))
        lnG = theta[st] + s * eps
        G = np.exp(lnG)
        crash = rng.random((n_ep, H)) < P
        x = np.where(crash, math.log(1 - J), M + SIGMA * rng.standard_normal((n_ep, H)))
        w = np.ones(n_ep)
        alive = np.ones(n_ep, bool)
        R = np.zeros((n_ep, H))
        A = np.zeros((n_ep, H), bool)
        for t in range(H):
            w1 = w * (1 + G[:, t] * np.expm1(x[:, t]))
            R[:, t] = np.where(alive, f(w, w1), 0.0)
            A[:, t] = alive
            alive = alive & (w1 > DELTA)
            w = np.where(alive, w1, w)
        ret = np.zeros((n_ep, H))
        acc = np.zeros(n_ep)
        for t in range(H - 1, -1, -1):
            acc = R[:, t] + gamma * acc * A[:, t]
            ret[:, t] = acc
        cnt = np.maximum(A.sum(0), 1)
        base = (ret * A).sum(0) / cnt
        adv = np.where(A, ret - base, 0.0)
        sd = adv[A].std() + 1e-12
        adv /= sd
        score = eps / s                                         # d log pi / d theta for the visited state
        grad = np.bincount(st[A], weights=(adv * score)[A], minlength=K) / A.sum()
        m_adam = 0.9 * m_adam + 0.1 * grad
        v_adam = 0.999 * v_adam + 0.001 * grad ** 2
        theta += lr * (m_adam / (1 - 0.9 ** (it + 1))) / (np.sqrt(v_adam / (1 - 0.999 ** (it + 1))) + 1e-8)
        if it % 25 == 0 or it == iters - 1:
            sizes = np.exp(theta)
            hist.append(dict(it=it, median_size=float(np.median(sizes)), ruinous_states=float(np.mean(sizes >= 0.95 / J)),
                             death_rate=float(1 - alive.mean()), q2=float(q2_score(sizes).mean())))
    sizes = np.exp(theta)
    return name, s0, theta0, seed, dict(final_median_size=float(np.median(sizes)), final_sizes=[float(v) for v in sizes],
                                        ruinous_states=float(np.mean(sizes >= 0.95 / J)), q2=float(q2_score(sizes).mean()),
                                        history=hist)


def learn(seeds=(1, 2, 3, 4), iters=300, n_ep=64, H=1000, K=16, lr=0.03, workers=4):
    jobs = [(name, s0, math.log(1.0), sd, iters, n_ep, H, K, lr)
            for name in REWARDS for s0 in (0.082, 0.756) for sd in seeds]
    jobs += [(name, 0.756, math.log(20.0), sd, iters, n_ep, H, K, lr) for name in REWARDS for sd in seeds]   # start inside the basin
    out = []
    with mp.get_context("fork").Pool(workers) as pool:
        for name, s0, th0, sd, res in pool.imap_unordered(_learn_job, jobs):
            out.append(dict(reward=name, s0=s0, start_size=round(math.exp(th0), 3), seed=sd, **res))
            print(f"  {name:55s} s0={s0:.3f} start={math.exp(th0):5.1f}x seed={sd}: median {res['final_median_size']:6.2f}x "
                  f"ruinous states {100 * res['ruinous_states']:5.1f}% Q2 {res['q2']:+.2f}", flush=True)
    return out


# ---------------------------------------------------------------------------- exploitation probes
def _probe(policy, f, gamma, N=4000, H=1000, seed=7):
    rng = np.random.default_rng(seed)
    w = np.ones(N)
    alive = np.ones(N, bool)
    tot = np.zeros(N)
    disc = 1.0
    for t in range(H):
        G = policy(t, w)
        crash = rng.random(N) < P
        x = np.where(crash, math.log(1 - J), M + SIGMA * rng.standard_normal(N))
        w1 = w * (1 + G * np.expm1(x))
        tot += disc * np.where(alive, f(w, w1), 0.0)
        alive &= w1 > DELTA
        w = np.where(alive, w1, w)
        disc *= gamma
    return float(tot.mean()), float(1 - alive.mean())


def hack():
    """Pathological policies a learner could drift into, valued under each candidate (constant G* = reference)."""
    gs = 2.9307
    probes = {
        "growth-optimal constant 2.93x": lambda t, w: np.full_like(w, gs),
        "flat (never trade)": lambda t, w: np.zeros_like(w),
        "timid 0.5x": lambda t, w: np.full_like(w, 0.5),
        "constant 55x (beyond-ruin optimum)": lambda t, w: np.full_like(w, 55.0),
        "2.93x, but 55x with prob 5% each bar (spikes)": None,
        "wealth-chasing: 55x once equity < 0.5 (gamble to recover)": lambda t, w: np.where(w < 0.5, 55.0, gs),
        "lock-in: 55x until equity doubles, then flat": lambda t, w: np.where(w < 2.0, 55.0, 0.0),
        "end-of-episode gamble: 55x in the last 50 bars": lambda t, w: np.full_like(w, 55.0 if t >= 950 else gs),
    }
    out = {}
    for name, spec in REWARDS.items():
        row = {}
        for pn, pol in probes.items():
            if pol is None:
                rng = np.random.default_rng(3)
                pol = (lambda r: (lambda t, w: np.where(r.random(len(w)) < 0.05, 55.0, gs)))(rng)
            row[pn] = _probe(pol, spec["f"], spec["gamma"])
        out[name] = row
    return out


def learn_long(seeds=(1, 2, 3, 4), iters=1500, n_ep=64, H=1000, K=16, lr=0.05, workers=4):
    """Longer runs: does a learner that starts safe (1x) or near the optimum with state-to-state size
    variation (2.93x, spread 0.6 in ln size, i.e. sizes from ~1x to ~9x) drift into the ruinous basin?"""
    names = ("R1 (current, eps 0.01)", "R1, eps 1e-6 (larger death cost)", "R1, gamma 0.999",
             "FL(0.05): full-liability log, tangent at a 95% loss", "FL(0.01): full-liability log, tangent at R1's floor")
    jobs = [(n, 0.756, math.log(st), sd, iters, n_ep, H, K, lr, spread) for n in names
            for st, spread in ((1.0, 0.3), (2.93, 0.6)) for sd in seeds]
    out = []
    with mp.get_context("fork").Pool(workers) as pool:
        for name, s0, th0, sd, res in pool.imap_unordered(_learn_job, jobs):
            out.append(dict(reward=name, s0=s0, start_size=round(math.exp(th0), 3), seed=sd, **res))
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("exact", "all"):
        (OUT / "exact.json").write_text(json.dumps(exact(), indent=1))
        print("exact done", flush=True)
    if cmd in ("values", "all"):
        (OUT / "values.json").write_text(json.dumps(values(), indent=1))
        print("values done", flush=True)
    if cmd in ("hack", "all"):
        (OUT / "hack.json").write_text(json.dumps(hack(), indent=1))
        print("hack done", flush=True)
    if cmd in ("learn_long", "all"):
        (OUT / "learn_long.json").write_text(json.dumps(learn_long(), indent=1))
        print("learn_long done", flush=True)
    if cmd in ("learn", "all"):
        (OUT / "learn.json").write_text(json.dumps(learn(), indent=1))
        print("learn done", flush=True)


if __name__ == "__main__":
    main()
