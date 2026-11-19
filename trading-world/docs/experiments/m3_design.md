# Experiment M3: design notes (written before pre-registration)

**Aim:** stage 5 of the road to demo trading, profiting from predictable patterns after trading costs. Agent: model-based
growth-optimal agent version 3 (`model_kelly3`). Runner: `m3.py`. Frozen criteria: `m3_analysis.py`.

## Why version 3

Version 2 (Experiment M2) passed the crash, known-optimum and no-edge markets, all without trading costs. Two of its six
no-edge agents still held positions (0.98x, 2.06x) on chance drift. That was harmless without costs but would be costly
with them. Real markets have costs and weak signals, some short-lived and some persistent. Version 3 adds three things:

- **Calibration.** It checks its conditional model on held-out experience and shrinks forecasts that do not hold up.
- **Persistence.** It measures how long its own predicted edge lasts.
- **Cost-aware decision.** It chooses each position for expected log growth after the cost of getting there. The cost
  is half the quoted spread per unit traded, read from its own observation, and is weighed per bar by the edge's
  persistence.

## Prototypes (seeds below 10,000; agents' seeds start at 1,000 for training and 20,000 for testing in the ledger, so they cannot overlap)

All results below come from a fast stand-alone simulator with the same fill-cost rule.

### Original S1/S2/S3 (seed 501)

- Version 2's world models found the patterns:
  - S1: correlation of predicted with true mean 0.988;
  - S2: 0.98;
  - S3: no edge, so the agent stays flat.
- Costs there are small next to the edge, so a cost-blind decision did nearly as well. These markets do not test cost
  discipline, which is why S1C and S2C were made.

### S2C (spread 1.0 sigma)

A one-bar-ahead (myopic) cost rule never trades S2C (growth 0), because a single bar's drift is smaller than the cost.
Spreading the cost over the edge's persistence trades it profitably:

| Policy | Growth per run |
|---|---|
| Agent, persistence-weighed cost | +3.5 |
| Agent, cost-blind | +0.7, with 8 of 20 deaths |
| Reference, true parameters | +5.1 |

### Choice of S1C spread, by the true-parameter reference only (120 runs each)

| Spread | Reference | Cost-blind |
|---|---|---|
| 0.2 sigma | +10.3 | +8.9 |
| 0.3 sigma | +6.3 | +3.0, with 9 deaths |
| 0.4 sigma (chosen) | +3.3 (t = 13), 3 deaths | −2.0, with 81 of 120 deaths |

The chosen spread is wide enough that trading every signal loses money, yet the market stays clearly profitable for
disciplined play.

### Version 3 code (seeds 601, 602; 60 runs each)

Calibration slopes were 0.93–0.99, so the forecasts held up. Learned persistence was −0.33 and −0.29 in S1C (true −0.3)
and 0.95–0.97 in S2C.

| Market | Agent growth per run | Reference growth per run |
|---|---|---|
| S1C | +3.43 / +3.53 (2 deaths each) | +3.56 / +3.65 (1 and 0 deaths) |
| S2C | +4.82 / +3.91 (4 deaths each) | +5.42 / +5.31 (3 and 1 deaths) |
| S3 | flat | flat |

On held-out S2C data, the agent's predicted variance matched the realised one: ratio 0.975, standardised residual
sd 0.99, kurtosis 3.0. So the extra deaths are not miscalibrated risk. Full growth-optimal sizing has a known
drawdown property: the chance of ever falling to a fraction x of the start is about x.

## Criteria choices made from the prototypes (before any M3 agent existed)

- **C4 (deaths).** M2's "reference + 2" rule is too tight at these death rates: Poisson noise alone is about ±2–3 in
  120 runs. It is replaced by an exact paired test on the same runs, plus a 10% ceiling.
- **C5 (direction).** "Position on the edge's side on every decision" would fail even the true-parameter reference (68%),
  because the no-trade band rightly keeps positions through weak opposite signals. "Trades toward the edge" would too
  (70%), because of rebalancing trims. The criterion instead uses decisions where the true edge exceeds the per-bar
  entry cost. On those, the decision's mathematics puts the reference on the edge's side; it measured 99.96%. The bar is
  90%.

## Known limitations, stated in advance

- **Tail assumption.** The decision's 32-node quadrature treats moves up to about 10 standard deviations of each learned
  component as possible, so it never sizes such that a larger move would wipe out the account. In S1C/S2C this binds at
  about 100x, below the textbook optimum (150x and more). The reference uses the same rule.
- **Drawdown risk.** Full growth-optimal sizing accepts roughly x chance of a drawdown to x. Whether a live account
  should use a fraction of it is a risk-appetite decision for the owner, to be raised in the readiness report. It is
  not imposed here.
