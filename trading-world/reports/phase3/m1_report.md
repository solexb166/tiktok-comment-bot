# Experiment M1: model-based growth-optimal agent (phase3-m1)

Pre-registered plan: `registry-m1/experiments/phase3-m1/manifest.json` (planned 2026-11-19T11:11:28Z); configuration and code checksums verified identical before training, evaluation and this analysis.

Each agent watches its own fresh 200,000-bar market (the same amount of market data each PPO agent trained on), learns the distribution of the next bar's return from what it saw (mixture-density network, maximum likelihood, early stopping), and sizes every decision to maximise expected log growth under what it learned. Nothing about the market is given to it. Final checkpoints only; two fresh rounds of 20 test runs.

## S6 (crashes): pre-registered verdict

| W1 ruin<1% | W2 deaths≤1 | W3 size 1.47–5.86x | W4 Q2≥0.5 | W5 growth | W6 tilt<1.5 | W7' learns from experience | E | **Successful seeds** | **Interpretation** |
|---|---|---|---|---|---|---|---|---|---|
| 6/6 | 6/6 | 6/6 | 6/6 | 2/6 | 6/6 | 6/6 | 6/6 | **2/6** | **mixed** |

Group means (95% bootstrap interval): Q2 +0.72 (+0.62, +0.83); ruinous 0.00% (0.00, 0.00); median size 2.32x (2.28, 2.35); test deaths 0 in 240 runs.

Against earlier PPO agents (historical references; Holm over 6 comparisons):

| Comparison | MK-S6 minus reference (95% CI) | Reference mean | p raw / Holm |
|---|---|---|---|
| MK-S6 vs R-FL R1 control: q2 | +2.389 (+1.700, +3.059) | -1.673 | 0.002 / 0.013 |
| MK-S6 vs R-FL R1 control: ruinous | -0.075 (-0.101, -0.047) | +0.075 | 0.002 / 0.013 |
| MK-S6 vs R-FL R1 control: size_distance | -0.979 (-1.355, -0.611) | +1.215 | 0.002 / 0.013 |
| MK-S6 vs Exp1 control: q2 | +3.560 (+1.810, +5.356) | -2.845 | 0.002 / 0.013 |
| MK-S6 vs Exp1 control: ruinous | -0.111 (-0.180, -0.050) | +0.111 | 0.002 / 0.013 |
| MK-S6 vs Exp1 control: size_distance | -0.808 (-1.224, -0.345) | +1.044 | 0.015 / 0.015 |

### Every MK-S6 agent

| Seed | Agent | Q2 | Ruinous | Deaths | Median size | 95th pct | Largest | Long | Growth per run (95% CI) | Beats random A/B | Tilt | Train-path gap | Learning curve: median size / Q2 at 20k → 60k → 200k bars | Success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (1060) | ag-36e975d83e | +0.60 | 0.00% | 0 | 2.30x | 2.39x | 4.22x | 85% | +0.1464 (-0.0015, +0.2911) | y/y | 1.05 | +0.01 | 5.67x / -6.58 → 2.64x / +0.74 → 2.31x / +0.61 | no |
| 2 (1061) | ag-f7d8e2e78c | +0.66 | 0.00% | 0 | 2.37x | 2.78x | 8.42x | 88% | +0.1327 (-0.0251, +0.2839) | y/y | 1.05 | +0.01 | 4.21x / -3.27 → 2.42x / +0.47 → 2.37x / +0.66 | no |
| 3 (1062) | ag-948c641374 | +0.96 | 0.00% | 0 | 2.37x | 2.48x | 3.64x | 100% | +0.2138 (+0.0514, +0.3712) | y/y | 1.01 | +0.00 | 4.27x / -2.71 → 2.43x / +0.66 → 2.37x / +0.96 | **yes** |
| 4 (1063) | ag-b64e1decee | +0.80 | 0.00% | 0 | 2.33x | 2.52x | 2.72x | 94% | +0.1973 (+0.0519, +0.3415) | y/y | 1.16 | -0.00 | 7.50x / -9.37 → 2.58x / +0.63 → 2.33x / +0.80 | **yes** |
| 5 (1064) | ag-5e2b8d4a0c | +0.71 | 0.00% | 0 | 2.27x | 2.35x | 4.49x | 90% | +0.1599 (+0.0137, +0.3070) | y/n | 1.02 | -0.01 | 4.59x / -3.51 → 2.80x / +0.67 → 2.27x / +0.70 | no |
| 6 (1065) | ag-0bccfab57c | +0.56 | 0.00% | 0 | 2.25x | 2.36x | 2.83x | 83% | +0.0770 (-0.0789, +0.2295) | y/n | 1.03 | -0.00 | 4.13x / -3.05 → 2.51x / +0.70 → 2.25x / +0.57 | no |

## S7 (generality check; growth-optimal size 5x; Q2 provisional, not used)

| W1 | W2 | W3 size 2.5–10x | W5 growth | W6 | W7'' | E | **Successful seeds** | **Passes (≥4/6)** |
|---|---|---|---|---|---|---|---|---|
| 6/6 | 4/6 | 6/6 | 6/6 | 5/6 | 4/6 | 6/6 | **2/6** | **no** |

### Every MK-S7 agent

| Seed | Agent | Q2 | Ruinous | Deaths | Median size | 95th pct | Largest | Long | Growth per run (95% CI) | Beats random A/B | Tilt | Train-path gap | Learning curve: median size / Q2 at 20k → 60k → 200k bars | Success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (1000) | ag-adfd7cc212 | +0.75 | 0.00% | 4 | 5.55x | 9.57x | 10.78x | 99% | +0.8903 (+0.2311, +1.5247) | y/y | 1.02 | -0.00 | 3.62x / +0.45 → 4.86x / +0.69 → 5.53x / +0.75 | no |
| 2 (1001) | ag-6e6c4595c6 | +0.83 | 0.00% | 0 | 3.86x | 6.92x | 10.44x | 99% | +1.1471 (+0.6920, +1.5904) | y/y | 1.28 | +0.00 | 2.93x / -0.05 → 5.12x / +0.63 → 3.85x / +0.83 | **yes** |
| 3 (1002) | ag-2f457f8d79 | +0.80 | 0.00% | 3 | 5.66x | 9.20x | 10.62x | 99% | +0.9148 (+0.2767, +1.5247) | y/y | 1.07 | -0.00 | 6.32x / +0.30 → 4.78x / +0.79 → 5.67x / +0.80 | no |
| 4 (1003) | ag-f0bfd525f4 | +0.78 | 0.00% | 0 | 4.33x | 8.06x | 10.69x | 97% | +0.9477 (+0.4833, +1.4041) | y/y | 1.67 | -0.00 | 4.89x / +0.63 → 4.12x / +0.69 → 4.33x / +0.78 | no |
| 5 (1004) | ag-301f092d24 | +0.82 | 0.00% | 1 | 4.75x | 8.30x | 10.71x | 99% | +0.8609 (+0.3247, +1.3711) | y/y | 1.20 | -0.00 | 5.26x / +0.11 → 5.26x / +0.74 → 4.72x / +0.82 | no |
| 6 (1005) | ag-c112cb0b98 | +0.80 | 0.00% | 1 | 4.94x | 8.51x | 10.67x | 99% | +1.0117 (+0.4908, +1.5007) | y/y | 1.28 | -0.00 | 3.64x / +0.35 → 4.72x / +0.64 → 4.93x / +0.80 | **yes** |

## Limits

- Six seeds per market; intervals show the range consistent with the data.
- The PPO references used different test data; Q2 depends only on the decisions.
- S6 and S7 have i.i.d. returns and no trading costs; markets with predictable structure and costs are the next test.
- Verdicts are the pre-registered rules applied mechanically; nothing was changed after training began.
- Every agent, model, checkpoint and evaluation is preserved.

## Protection of earlier work

- original registry: content hash before / now: 1ba10554e4724f90… / 1ba10554e4724f90… (identical)
- original registry: seal verification: all sealed records intact
- Phase 3 Experiment 1 registry: content hash before / now: 90bcdabfa9d5e246… / 90bcdabfa9d5e246… (identical)
- Phase 3 Experiment 1 registry: seal verification: all sealed records intact
- Phase 3 Experiment 1b registry: content hash before / now: cdc3d011e4a294ef… / cdc3d011e4a294ef… (identical)
- Phase 3 Experiment 1b registry: seal verification: all sealed records intact
- Experiment R-FL registry: content hash before / now: 86830d9ee889c6d4… / 86830d9ee889c6d4… (identical)
- Experiment R-FL registry: seal verification: all sealed records intact
- Experiment M1 registry: seal verification: all sealed records intact
