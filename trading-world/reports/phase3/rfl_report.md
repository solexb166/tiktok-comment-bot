# Experiment R-FL: full-liability log reward vs the current reward (phase3-rfl)

Proposal: `docs/proposals/0007-reward-investigation.md`. Pre-registered plan: `registry-rfl/experiments/phase3-rfl/manifest.json` (planned 2026-11-19T08:12:06Z); configuration and code checksums verified identical before training, evaluation and this analysis.

Six paired seeds per group, 3,000,000 steps each, final checkpoints only, two fresh rounds of 20 test runs. Everything except the reward is identical to the Experiment 1 control. G* = 2.93x; ruinous = 9.5x or more.

| Group | Reward | Config SHA-256 |
|---|---|---|
| R1 | current reward R1 (eps-floored log wealth); concurrent control | `863fb9de3fd1…` |
| R6 | full-liability log return, c = 0.01: ln M for M >= 0.01, tangent line below (identical to R1 on every bar that does not end the account) | `3c6910da99eb…` |
| R6-c0.05 | full-liability log return, c = 0.05 (sensitivity) | `f0da48bcb2a2…` |

## Pre-registered verdicts

Per seed: W1–W7 and E (approved in Experiment 1b). A group supports safe growth if at least 4 of 6 seeds meet all.

| Group | W1 ruin<1% | W2 deaths≤1 | W3 size 1.47–5.86x | W4 Q2≥0.5 | W5 growth | W6 tilt<1.5 | W7 progress | E | **Successful seeds** | **Interpretation** |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 0/6 | 0/6 | 1/6 | 0/6 | 0/6 | 1/6 | 0/6 | 6/6 | **0/6** | **still enters the ruin zone** |
| R6 | 0/6 | 1/6 | 2/6 | 0/6 | 0/6 | 2/6 | 0/6 | 5/6 | **0/6** | **still enters the ruin zone** |
| R6-c0.05 | 0/6 | 1/6 | 1/6 | 0/6 | 0/6 | 2/6 | 0/6 | 6/6 | **0/6** | **still enters the ruin zone** |

Reward truncation confirmed as a cause (R6 'safe growth' and R1 not): **no**.

Group means across seeds (95% bootstrap interval):

| Group | Q2 | Ruinous share | Median size | Seeds ruinous ≥1% | Training stable | Explained variance (last 20%) | Value loss (last 20%) |
|---|---|---|---|---|---|---|---|
| R1 | -1.67 (-2.31, -0.96) | 7.46% (4.65, 9.98) | 0.96x (0.65, 1.34) | 6/6 | **yes** | +0.01 | 0.0331 |
| R6 | -2.41 (-4.08, -0.90) | 9.82% (4.13, 16.11) | 1.14x (0.53, 1.89) | 6/6 | **yes** | +0.00 | 0.0298 |
| R6-c0.05 | -1.41 (-2.62, -0.48) | 6.47% (2.64, 11.36) | 0.90x (0.47, 1.44) | 6/6 | **yes** | -0.00 | 0.0114 |

## Checks for unintended incentives

| Group | Timidity: non-ruinous median size vs R1 (95% CI) | T ok (lower bound ≥ 0.75) | Gambling to recover: size after a >20% loss / elsewhere | GR ok (≤ 1.5) |
|---|---|---|---|---|
| R1 | — | n/a | 0.39 (372 decisions) | **yes** |
| R6 | x1.01 (0.53, 1.72) | no | 0.51 (491 decisions) | **yes** |
| R6-c0.05 | x0.90 (0.50, 1.51) | no | 0.30 (305 decisions) | **yes** |

## Comparison with the concurrent R1 control (Holm-corrected over 6 comparisons)

| Comparison | Difference (95% CI) | p raw / Holm |
|---|---|---|
| R6 vs R1: q2 | -0.740 (-2.508, +0.930) | 0.465 / 1.000 |
| R6 vs R1: ruinous | +0.024 (-0.039, +0.091) | 0.537 / 1.000 |
| R6 vs R1: size_distance | +0.043 (-0.703, +0.842) | 0.913 / 1.000 |
| R6-c0.05 vs R1: q2 | +0.262 (-1.112, +1.463) | 0.723 / 1.000 |
| R6-c0.05 vs R1: ruinous | -0.010 (-0.058, +0.045) | 0.775 / 1.000 |
| R6-c0.05 vs R1: size_distance | +0.189 (-0.465, +0.826) | 0.602 / 1.000 |

Paired by seed index (same training data and initial network):

| Difference | Mean | sign-flip p |
|---|---|---|
| R6 - R1 (same seed): q2 | -0.740 | 0.500 |
| R6 - R1 (same seed): ruinous | +0.024 | 0.500 |
| R6 - R1 (same seed): size_distance | +0.043 | 0.688 |
| R6-c0.05 - R1 (same seed): q2 | +0.262 | 0.406 |
| R6-c0.05 - R1 (same seed): ruinous | -0.010 | 0.625 |
| R6-c0.05 - R1 (same seed): size_distance | +0.189 | 0.469 |

## Every agent (final checkpoint, 40 test runs)

| Group | Seed | Agent | Q2 | Ruinous | Test deaths | Median size | Non-ruinous median | 95th pct | Largest | Long | Growth per run (95% CI) | Beats random A/B | Tilt | Train size first→last | Train ruin last 20% | Train deaths | W1–W7,E | Success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 1 (1054) | ag-ef2fc03643 | -2.08 | 9.07% | 14 | 0.87x | 0.70x | 15.65x | 299.47x | 83% | -0.7223 (-1.3020, -0.1573) | n/n | 3.34 | 0.56x→3.08x | 7.62% | 321 | ✗✗✗✗✗✗✗✓ | no |
| R1 | 2 (1055) | ag-2dfd993fa3 | -0.88 | 3.70% | 6 | 0.82x | 0.76x | 8.12x | 64.81x | 81% | -0.1057 (-0.5283, +0.2866) | n/y | 2.11 | 0.71x→1.69x | 2.68% | 107 | ✗✗✗✗✗✗✗✓ | no |
| R1 | 3 (1056) | ag-1a12b8b837 | -0.34 | 2.28% | 4 | 0.39x | 0.37x | 5.62x | 86.51x | 80% | -0.0086 (-0.3801, +0.3077) | y/n | 1.12 | 0.50x→1.39x | 2.42% | 110 | ✗✗✗✗✗✓✗✓ | no |
| R1 | 4 (1057) | ag-04ee4dd2da | -2.86 | 11.96% | 16 | 1.76x | 1.38x | 18.59x | 289.60x | 94% | -0.3716 (-1.0779, +0.3186) | n/n | 1.56 | 0.91x→4.30x | 11.09% | 774 | ✗✗✓✗✗✗✗✓ | no |
| R1 | 5 (1058) | ag-9717099bda | -2.18 | 9.78% | 12 | 1.21x | 0.99x | 16.18x | 929.73x | 94% | -0.1019 (-0.7430, +0.5164) | n/y | 2.57 | 1.12x→3.81x | 9.16% | 1096 | ✗✗✗✗✗✗✗✓ | no |
| R1 | 6 (1059) | ag-0d84ae56a4 | -1.69 | 7.95% | 9 | 0.72x | 0.60x | 13.70x | 462.85x | 84% | +0.0553 (-0.5490, +0.6363) | n/y | 2.76 | 0.67x→3.63x | 8.93% | 445 | ✗✗✗✗✗✗✗✓ | no |
| R6 | 1 (1054) | ag-ba79275e1c | -2.07 | 8.87% | 14 | 0.71x | 0.56x | 15.64x | 597.50x | 79% | -0.5672 (-1.1540, +0.0202) | n/n | 3.62 | 0.57x→3.42x | 8.63% | 373 | ✗✗✗✗✗✗✗✓ | no |
| R6 | 2 (1055) | ag-8c8ffdfdb3 | -0.88 | 3.70% | 6 | 0.82x | 0.76x | 8.12x | 64.81x | 81% | -0.1057 (-0.5283, +0.2866) | n/y | 2.11 | 0.71x→1.69x | 2.68% | 107 | ✗✗✗✗✗✗✗✓ | no |
| R6 | 3 (1056) | ag-d2f0125b5b | -0.18 | 1.40% | 1 | 0.19x | 0.18x | 3.51x | 119.64x | 70% | +0.0676 (-0.1696, +0.2514) | y/n | 2.62 | 0.50x→0.91x | 1.35% | 73 | ✗✓✗✗✗✗✗✓ | no |
| R6 | 4 (1057) | ag-7b163db4eb | -4.53 | 16.79% | 18 | 1.74x | 1.15x | 26.98x | 399.41x | 83% | -0.3044 (-1.0881, +0.4635) | y/y | 1.24 | 0.91x→5.21x | 14.78% | 785 | ✗✗✓✗✗✓✗✗ | no |
| R6 | 5 (1058) | ag-d78ae24739 | -5.64 | 22.45% | 22 | 2.75x | 1.64x | 35.85x | 530.65x | 92% | -0.2105 (-1.1590, +0.7556) | y/y | 1.02 | 1.26x→8.85x | 23.35% | 1598 | ✗✗✓✗✗✓✗✓ | no |
| R6 | 6 (1059) | ag-f118ef4949 | -1.19 | 5.72% | 11 | 0.64x | 0.55x | 10.82x | 276.83x | 90% | -0.3830 (-0.9366, +0.1612) | n/n | 2.54 | 0.67x→2.66x | 5.89% | 399 | ✗✗✗✗✗✗✗✓ | no |
| R6-c0.05 | 1 (1054) | ag-53d43d305c | -0.75 | 3.69% | 3 | 0.30x | 0.28x | 7.44x | 305.91x | 74% | -0.0653 (-0.4226, +0.2561) | y/n | 2.81 | 0.56x→1.52x | 3.22% | 135 | ✗✗✗✗✗✗✗✓ | no |
| R6-c0.05 | 2 (1055) | ag-a6204a4440 | -0.77 | 3.41% | 1 | 0.72x | 0.67x | 7.85x | 106.61x | 71% | +0.0911 (-0.2157, +0.3793) | y/n | 1.57 | 0.71x→1.81x | 2.80% | 105 | ✗✓✗✗✗✗✗✓ | no |
| R6-c0.05 | 3 (1056) | ag-cd7284be94 | -0.38 | 2.44% | 4 | 0.40x | 0.38x | 6.06x | 63.10x | 89% | -0.0604 (-0.4585, +0.2891) | y/n | 1.49 | 0.50x→1.39x | 2.39% | 190 | ✗✗✗✗✗✓✗✓ | no |
| R6-c0.05 | 4 (1057) | ag-2932141bf6 | -4.15 | 17.38% | 23 | 2.12x | 1.41x | 24.00x | 693.13x | 84% | -0.9583 (-1.7001, -0.1651) | n/n | 1.53 | 0.91x→5.49x | 15.75% | 736 | ✗✗✓✗✗✗✗✓ | no |
| R6-c0.05 | 5 (1058) | ag-8ae0b79eda | -2.22 | 10.15% | 12 | 1.26x | 1.00x | 17.45x | 262.26x | 95% | -0.1698 (-0.8199, +0.4636) | n/y | 1.15 | 0.93x→3.46x | 7.81% | 438 | ✗✗✗✗✗✓✗✓ | no |
| R6-c0.05 | 6 (1059) | ag-a8981842f9 | -0.20 | 1.74% | 3 | 0.60x | 0.57x | 5.61x | 89.26x | 91% | +0.0423 (-0.2834, +0.3288) | y/n | 2.16 | 0.65x→1.31x | 1.49% | 72 | ✗✗✗✗✗✗✗✓ | no |

## Learning curves (development round; descriptive only)

Group mean of Q2 / ruinous share / median size by training steps.

| Group | 500k | 1000k | 1500k | 2000k | 2500k | 3000k |
|---|---|---|---|---|---|---|
| R1 | +0.17 / 0.2% / 0.39x | -0.60 / 3.3% / 0.71x | -1.38 / 6.3% / 0.95x | -1.65 / 7.4% / 1.01x | -1.73 / 7.7% / 1.00x | -1.73 / 7.7% / 0.98x |
| R6 | +0.17 / 0.2% / 0.39x | -0.60 / 3.3% / 0.71x | -1.81 / 7.6% / 1.15x | -1.84 / 8.4% / 1.16x | -2.16 / 9.6% / 1.23x | -2.37 / 10.8% / 1.33x |
| R6-c0.05 | +0.19 / 0.1% / 0.36x | -0.15 / 1.5% / 0.55x | -0.53 / 3.0% / 0.68x | -0.86 / 4.2% / 0.72x | -1.13 / 5.2% / 0.80x | -1.44 / 6.6% / 0.91x |

## Limits

- Six seeds per group resolve only large effects; intervals show the range consistent with the data.
- Evaluation records of R6 agents carry the frozen Phase 2 gate's 'fail: no criterion' (the gate has no R6 rule); it is not used.
- Verdicts are the pre-registered rules applied mechanically; nothing was changed after training began.
- Every agent, version (including interrupted attempts), checkpoint and evaluation is preserved.

## Protection of earlier work

- original registry: content hash before / now: 1ba10554e4724f90… / 1ba10554e4724f90… (identical)
- original registry: seal verification: all sealed records intact
- Phase 3 Experiment 1 registry: content hash before / now: 90bcdabfa9d5e246… / 90bcdabfa9d5e246… (identical)
- Phase 3 Experiment 1 registry: seal verification: all sealed records intact
- Phase 3 Experiment 1b registry: content hash before / now: cdc3d011e4a294ef… / cdc3d011e4a294ef… (identical)
- Phase 3 Experiment 1b registry: seal verification: all sealed records intact
- Experiment R-FL registry: seal verification: all sealed records intact
