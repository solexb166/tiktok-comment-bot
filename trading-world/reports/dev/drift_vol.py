"""Development: can an agent learn each market's long-run drift and size by predicted risk, after the broker's spread,
slippage and overnight swap? Daily bars from the owner's MT5 history. Walk-forward by year: learn from all years before
Y, trade year Y (2017-2022 inside the train split; 2023 = validation, reported separately). 2024+ never read.
World models the agent can hold (it chooses between them itself, from a forward check on its own training history):
  'sharpe' : the next day's move in units of predicted volatility has a constant mean m  -> expected move m*sigma_t
  'level'  : the next day's move has a constant mean mu                                    -> expected move mu
Estimation risk: the mean is shrunk by positive-part James-Stein with a block-bootstrap standard error.
Decision each day (growth-optimal, cost-aware, no rules on frequency): choose exposure G maximising
  G*E[move] - 0.5*G^2*sigma^2 - swap(G) - cost*|G - G_now|
Swap: the broker's CURRENT swap per calendar day (an approximation for past years), long and short separately."""
import json, math, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from tradeworld.learning.model_kelly6 import block_bootstrap
ROOT = Path("/home/user/tiktok-comment-bot/trading-world/datasets")
SYMS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "NAS100"]
COST_BP = dict(EURUSD=1.0, GBPUSD=1.0, USDJPY=1.0, XAUUSD=1.2, NAS100=2.5, BTCUSD=3.5)   # per unit traded, measured by v7 (BTC: half spread 2.2 + slippage)
HL = 20
GMAX = 50.0
VAL_END = 1704067200 // 60
yr = lambda Y: int(datetime(Y, 1, 1, tzinfo=timezone.utc).timestamp()) // 60

def swap_rates(s, price):
    j = json.load(open(ROOT / "mt5" / s / "symbol_info.json"))["spec"]
    per_lot = lambda pts: pts * j["point"] * j["trade_contract_size"]           # account currency per lot per day (USD-quoted)
    notional = j["trade_contract_size"] * price
    if s == "USDJPY":                                                          # quoted in JPY: notional in USD is the contract size
        return (j["swap_long"] * j["point"] * j["trade_contract_size"] / price / j["trade_contract_size"],
                j["swap_short"] * j["point"] * j["trade_contract_size"] / price / j["trade_contract_size"])
    return per_lot(j["swap_long"]) / notional, per_lot(j["swap_short"]) / notional    # fraction of notional per day (negative = cost)

def decide(Er, sig2, G0, cost, sl, ss, days):
    G = np.linspace(-GMAX, GMAX, 4001)
    swap = np.where(G > 0, G * sl, -G * ss) * days                             # sl, ss < 0 mean the position pays
    f = G * Er - 0.5 * G * G * sig2 + swap - cost * np.abs(G - G0)
    return float(G[np.argmax(f)])

def shrunk_mean(z, block=20):
    m = float(z.mean())
    se = float(np.std(block_bootstrap(lambda i: float(z[i].mean()), len(z), block, 300, 1), ddof=1))
    k = max(0.0, 1.0 - se * se / (m * m)) if m != 0 else 0.0
    return k * m, m, se

def run(seg, model, m, cost, sl, ss):
    """Trade the segment once. seg: dict of arrays r (next-day log return), sig (predicted sigma), days (calendar days held)."""
    G0, logw, dead, turn, gsum, n = 0.0, 0.0, False, 0.0, 0.0, 0
    for r, s, d in zip(seg["r"], seg["sig"], seg["days"]):
        Er = m * s if model == "sharpe" else m
        G = decide(Er, s * s, G0, cost, sl, ss, d)
        sw = (G * sl if G > 0 else -G * ss) * d
        gross = 1.0 + G * math.expm1(r) + sw - cost * abs(G - G0)
        turn += abs(G - G0); gsum += abs(G); n += 1
        if gross <= 0.0:
            return dict(g=-math.inf, dead=True, turn=turn, lev=gsum / n)
        logw += math.log(gross); G0 = G
    return dict(g=logw, dead=dead, turn=turn, lev=gsum / max(n, 1))

for s in SYMS:
    df = pd.read_parquet(ROOT / "mt5_history" / s / "D1.parquet").sort_values("time")
    df = df[df.time < VAL_END]
    t = df.time.values; lm = np.log(0.5 * (df.bid_c.values + df.ask_c.values))
    r = np.diff(lm); t0 = t[:-1]
    days = np.diff(t) / 1440.0                                                 # calendar days the position is held (weekends count)
    v = np.empty(len(r)); a = 1 - 0.5 ** (1 / HL); acc = r[:HL].var()
    for i in range(len(r)):                                                    # sigma predicted for return i from returns before it
        v[i] = acc; acc = (1 - a) * acc + a * r[i] ** 2
    sig = np.sqrt(v)
    sl, ss = swap_rates(s, float(np.exp(lm[-1])))
    cost = COST_BP[s] * 1e-4
    ok = np.arange(len(r)) >= 60
    lines, tot = [], {"agent": 0.0, "hold": 0.0}
    for Y in range(2017, 2024):
        tr = ok & (t0 < yr(Y)); te = ok & (t0 >= yr(Y)) & (t0 < yr(Y + 1))
        if tr.sum() < 500 or te.sum() < 50:
            continue
        # the agent's own choice of world model: learn on the first 70% of its history, replay the last 30% as if live
        idx = np.where(tr)[0]; cut = idx[int(0.7 * len(idx))]
        fit, chk = tr & (np.arange(len(r)) < cut), tr & (np.arange(len(r)) >= cut)
        best, score = None, -math.inf
        for model in ("sharpe", "level"):
            z = r[fit] / sig[fit] if model == "sharpe" else r[fit]
            ms, _, _ = shrunk_mean(z)
            g = run(dict(r=r[chk], sig=sig[chk], days=days[chk]), model, ms, cost, sl, ss)["g"]
            if g > score:
                best, score = model, g
        z = r[tr] / sig[tr] if best == "sharpe" else r[tr]
        ms, mraw, se = shrunk_mean(z)
        res = run(dict(r=r[te], sig=sig[te], days=days[te]), best, ms, cost, sl, ss)
        hold = float(np.sum(np.log1p(np.expm1(r[te]) + sl * days[te])))      # buy and hold 1x, paying the long swap
        tag = "VALIDATION " if Y == 2023 else ""
        lines.append(f"  {tag}{Y}: model {best:6s} mean {mraw:+.4f} +- {se:.4f} kept {ms:+.4f} | agent growth {res['g']:+.3f} "
                     f"(avg size {res['lev']:.2f}x, turnover {res['turn']:.1f}) | buy&hold 1x {hold:+.3f}")
        if Y < 2023:
            tot["agent"] += res["g"]; tot["hold"] += hold
    print(f"{s}: swap per day long {sl*1e4:+.2f}bp short {ss*1e4:+.2f}bp, trading cost {COST_BP[s]}bp")
    print("\n".join(lines))
    print(f"  2017-2022 total: agent {tot['agent']:+.3f}, buy&hold {tot['hold']:+.3f}", flush=True)
