# Phase 3 Experiment 1: S6 single-factor ablation (phase3-exp1)

Pre-registered plan: `registry-phase3/experiments/phase3-exp1/manifest.json` (planned 2026-11-17T18:02:35Z). Analysis code `src/tradeworld/learning/phase3_analysis.py`, SHA-256 `ded7df4f3402ad0e…`, identical to the frozen version.

Five groups of 6 new agents, each trained from scratch for 3,000,000 steps on its own fresh S6 market path. Every agent is judged on its final checkpoint only, on two fresh evaluation rounds (test runs each, 20005 and 20006 market seeds). Growth-optimal size G* = 2.93x; ruinous = 9.5x or more.

| Group | Configuration |
|---|---|
| control | existing settings: size output u, G = 10^(4u); PPO lr 3e-4, 10 epochs, no KL limit, gamma 0.99 |
| C1 | size output z with G = e^z (z in [-25, 25], numerical safeguard only) |
| C2 | size output z with G = softplus(z) = ln(1 + e^z) (z in [-25, 10,000], numerical safeguard only) |
| D | steadier updates: learning rate 1e-4, 4 epochs per update, KL limit 0.02 per update |
| A | longer learning horizon: discount factor 0.999 |

## Pre-registered verdicts

Q2 effect = group mean minus control mean of per-seed decision quality (Q2.0), with a 95% bootstrap interval across seeds; p = exact one-sided permutation test (924 relabellings), Holm-corrected across the four groups.

| Group | Q2 mean (sd) | Effect vs control (95% CI) | Cohen's d | p raw / Holm | G1 | G2 ruin<1% | G3 size | G4 growth | G5 both rounds | G6 history | E1–E4 clear | K catastrophe | **Genuine improvement** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | -2.84 (2.52) | — | — | — | — | 0/6 | 1/6 | 0/6 | — | 0/6 | no | mean change +0.0014 over 6 seeds, sign-flip p = 1.000 | — |
| C1 | +0.46 (0.05) | +3.31 (+1.53, +5.09) | +1.86 | 0.001 / 0.004 | **yes** | 6/6 **yes** | 0/6 no | 2/6 no | **yes** | 6/6 **yes** | **yes** | not measurable (0 seeds with 5+ training deaths) | no |
| C2 | +0.32 (0.02) | +3.17 (+1.39, +4.94) | +1.78 | 0.001 / 0.004 | **yes** | 6/6 **yes** | 0/6 no | 6/6 **yes** | **yes** | 6/6 **yes** | **yes** | not measurable (0 seeds with 5+ training deaths) | no |
| D | -3.00 (1.77) | -0.16 (-2.39, +2.06) | -0.07 | 0.554 / 0.554 | no | 0/6 no | 3/6 no | 0/6 no | no | 0/6 no | no | mean change +0.0013 over 6 seeds, sign-flip p = 1.000 | no |
| A | -0.76 (0.65) | +2.08 (+0.26, +4.00) | +1.13 | 0.047 / 0.093 | no | 1/6 no | 0/6 no | 0/6 no | **yes** | 0/6 no | **yes** | mean change +0.0003 over 6 seeds, sign-flip p = 0.828 | no |

Exploitation checks per group: E1 training-path advantage (mean Q2 gap, limit 0.5); E2 share of decisions at 1,000x or more (limit 0.1%); E3 median previous-move tilt (limit 1.5); E4 rejected-order share (control + 0.1 pp).

| Group | E1 gap | E2 boundary | E3 tilt | E4 rejected | Ruinous share effect vs control (95% CI) | K: change after deaths (control) |
|---|---|---|---|---|---|---|
| control | +0.03 **yes** | 0.000% **yes** | 1.64 no | 0.000% **yes** | — | +0.0014 (+0.0014) |
| C1 | -0.00 **yes** | 0.000% **yes** | 1.15 **yes** | 0.000% **yes** | -11.1 pp (-18.0, -5.0) | — (+0.0014) |
| C2 | -0.00 **yes** | 0.000% **yes** | 1.17 **yes** | 0.000% **yes** | -11.1 pp (-18.0, -5.0) | — (+0.0014) |
| D | +0.07 **yes** | 0.000% **yes** | 2.16 no | 0.000% **yes** | +2.0 pp (-6.3, +10.3) | +0.0013 (+0.0014) |
| A | -0.02 **yes** | 0.000% **yes** | 1.37 **yes** | 0.000% **yes** | -7.3 pp (-14.4, -0.8) | +0.0003 (+0.0014) |

## Every agent (final checkpoint, rounds A and B pooled: 40 test runs)

| Group | Seed | Agent | Q2 | Q2 A / B | Ruinous | Median size | 95th pct size | Largest | Long | Test deaths | Growth per run (95% CI) | Beats random A/B | Tilt | Q2 own paths | Train deaths | Ruin first→last 20% | Median KL | Clip frac | Gate A/B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | 1 (1018) | ag-b6a8a7a0df | -3.13 | -3.22 / -3.04 | 13.45% | 1.38x | 20.5x | 492x | 82% | 20 | -0.8432 (-1.5458, -0.1140) | y/n | 4.49 | -3.05 | 882 | 0.4%→14.2% | 0.053 | 0.35 | fail/fail |
| control | 2 (1023) | ag-d6c1ecb6e8 | -0.33 | -0.30 / -0.36 | 2.33% | 0.57x | 5.8x | 125x | 88% | 5 | -0.2701 (-0.6555, +0.0812) | n/n | 1.12 | -0.41 | 145 | 0.2%→2.9% | 0.058 | 0.38 | fail/pass |
| control | 3 (1028) | ag-96f65f7cf8 | -5.94 | -5.09 / -6.78 | 18.11% | 1.39x | 39.0x | 810x | 84% | 29 | -1.3466 (-2.1232, -0.4781) | y/y | 1.09 | -5.94 | 1765 | 2.3%→21.7% | 0.062 | 0.37 | fail/fail |
| control | 4 (1033) | ag-7c0627230c | -0.80 | -0.68 / -0.92 | 4.34% | 0.74x | 8.8x | 181x | 93% | 6 | -0.0328 (-0.4749, +0.3671) | y/n | 3.46 | -0.85 | 312 | 0.1%→4.6% | 0.050 | 0.34 | fail/fail |
| control | 5 (1038) | ag-9ae68a71be | -5.76 | -5.81 / -5.71 | 24.68% | 3.27x | 39.3x | 403x | 98% | 32 | -1.8345 (-2.5293, -1.0534) | y/n | 1.14 | -5.56 | 2717 | 4.4%→30.3% | 0.046 | 0.32 | fail/fail |
| control | 6 (1043) | ag-debf4eb4f2 | -1.11 | -1.10 / -1.11 | 3.90% | 0.57x | 7.8x | 355x | 54% | 12 | -0.2020 (-0.8733, +0.4648) | n/n | 2.14 | -1.07 | 487 | 0.3%→3.0% | 0.053 | 0.37 | fail/fail |
| C1 | 1 (1019) | ag-95739a2025 | +0.44 | +0.44 / +0.44 | 0.00% | 0.86x | 1.2x | 2x | 98% | 0 | +0.1300 (+0.0589, +0.1983) | y/n | 1.24 | +0.44 | 0 | 0.0%→0.0% | 0.041 | 0.30 | fail/fail |
| C1 | 2 (1024) | ag-a8de73b5e9 | +0.45 | +0.45 / +0.45 | 0.00% | 0.99x | 1.4x | 2x | 93% | 0 | +0.1297 (+0.0454, +0.2084) | y/n | 1.09 | +0.45 | 0 | 0.0%→0.0% | 0.042 | 0.31 | fail/fail |
| C1 | 3 (1029) | ag-5a642f1564 | +0.42 | +0.42 / +0.42 | 0.00% | 0.80x | 1.1x | 2x | 99% | 0 | +0.1230 (+0.0570, +0.1860) | y/n | 1.14 | +0.41 | 0 | 0.0%→0.0% | 0.034 | 0.27 | fail/fail |
| C1 | 4 (1034) | ag-df6ac981fc | +0.50 | +0.50 / +0.49 | 0.00% | 0.97x | 1.3x | 2x | 99% | 0 | +0.1453 (+0.0623, +0.2249) | y/y | 1.26 | +0.50 | 0 | 0.0%→0.0% | 0.041 | 0.30 | fail/fail |
| C1 | 5 (1039) | ag-65992c2849 | +0.44 | +0.44 / +0.44 | 0.00% | 0.91x | 1.3x | 2x | 96% | 0 | +0.1236 (+0.0468, +0.1945) | y/n | 1.16 | +0.44 | 0 | 0.0%→0.0% | 0.036 | 0.29 | fail/fail |
| C1 | 6 (1044) | ag-98424529d6 | +0.55 | +0.55 / +0.55 | 0.00% | 1.05x | 1.4x | 2x | 100% | 0 | +0.1524 (+0.0602, +0.2407) | y/y | 1.03 | +0.55 | 0 | 0.0%→0.0% | 0.036 | 0.28 | pass/pass |
| C2 | 1 (1020) | ag-414a2bfe2d | +0.29 | +0.30 / +0.29 | 0.00% | 0.61x | 0.8x | 1x | 94% | 0 | +0.0863 (+0.0443, +0.1279) | y/y | 1.10 | +0.29 | 0 | 0.0%→0.0% | 0.031 | 0.26 | fail/fail |
| C2 | 2 (1025) | ag-b3421991b7 | +0.31 | +0.31 / +0.31 | 0.00% | 0.62x | 0.8x | 1x | 95% | 0 | +0.0974 (+0.0519, +0.1409) | y/y | 1.15 | +0.31 | 0 | 0.0%→0.0% | 0.044 | 0.31 | fail/fail |
| C2 | 3 (1030) | ag-adf1943651 | +0.34 | +0.34 / +0.33 | 0.00% | 0.65x | 0.8x | 1x | 98% | 0 | +0.1027 (+0.0507, +0.1509) | y/y | 1.11 | +0.34 | 0 | 0.0%→0.0% | 0.051 | 0.33 | fail/fail |
| C2 | 4 (1035) | ag-66f227f540 | +0.33 | +0.33 / +0.33 | 0.00% | 0.61x | 0.8x | 1x | 99% | 0 | +0.1006 (+0.0546, +0.1438) | y/y | 1.23 | +0.33 | 0 | 0.0%→0.0% | 0.034 | 0.27 | fail/fail |
| C2 | 5 (1040) | ag-e9cf82bfe0 | +0.36 | +0.36 / +0.35 | 0.00% | 0.72x | 0.9x | 1x | 98% | 0 | +0.0971 (+0.0416, +0.1509) | y/y | 1.19 | +0.36 | 0 | 0.0%→0.0% | 0.050 | 0.32 | fail/fail |
| C2 | 6 (1045) | ag-67473dd3f4 | +0.30 | +0.30 / +0.30 | 0.00% | 0.61x | 0.8x | 1x | 96% | 0 | +0.0810 (+0.0383, +0.1231) | y/y | 1.62 | +0.30 | 0 | 0.0%→0.0% | 0.046 | 0.32 | fail/fail |
| D | 1 (1021) | ag-14202002d1 | -1.71 | -1.78 / -1.64 | 8.00% | 2.19x | 12.0x | 79x | 100% | 12 | -0.3610 (-0.9537, +0.2051) | n/n | 2.92 | -1.67 | 1077 | 0.1%→14.9% | 0.005 | 0.04 | fail/fail |
| D | 2 (1026) | ag-62bddcc3ae | -2.50 | -2.54 / -2.46 | 11.28% | 1.10x | 16.7x | 166x | 100% | 15 | -0.6052 (-1.2188, +0.0205) | n/n | 1.68 | -2.33 | 1008 | 0.3%→10.6% | 0.006 | 0.05 | fail/fail |
| D | 3 (1031) | ag-742128fccb | -2.07 | -2.08 / -2.07 | 9.71% | 0.73x | 17.4x | 201x | 100% | 15 | -0.5848 (-1.2217, +0.0422) | y/n | 3.68 | -2.06 | 679 | 0.1%→8.0% | 0.006 | 0.05 | fail/fail |
| D | 4 (1036) | ag-18dc035387 | -4.76 | -4.84 / -4.68 | 20.50% | 2.87x | 24.6x | 179x | 100% | 21 | -0.7462 (-1.5051, +0.0399) | y/n | 2.49 | -4.65 | 1489 | 0.1%→21.9% | 0.005 | 0.04 | fail/fail |
| D | 5 (1041) | ag-5a3bbe467e | -1.33 | -1.30 / -1.36 | 6.36% | 0.44x | 11.5x | 152x | 100% | 6 | -0.0588 (-0.5366, +0.3828) | y/n | 1.82 | -1.35 | 701 | 0.5%→6.2% | 0.006 | 0.05 | pass/fail |
| D | 6 (1046) | ag-9ac072de56 | -5.63 | -5.44 / -5.83 | 22.89% | 2.67x | 30.4x | 151x | 100% | 24 | -0.8314 (-1.6443, +0.0026) | y/n | 1.83 | -5.54 | 1351 | 0.3%→18.4% | 0.006 | 0.05 | fail/fail |
| A | 1 (1022) | ag-9180058cb8 | +0.01 | +0.03 / +0.00 | 0.48% | 0.18x | 2.9x | 59x | 65% | 1 | -0.0189 (-0.2208, +0.1290) | y/n | 1.10 | +0.03 | 60 | 0.1%→0.6% | 0.066 | 0.38 | fail/fail |
| A | 2 (1027) | ag-7599ff4650 | -0.34 | -0.32 / -0.35 | 1.97% | 0.38x | 5.5x | 85x | 76% | 4 | -0.0827 (-0.4295, +0.2191) | y/n | 2.68 | -0.34 | 111 | 0.0%→2.4% | 0.066 | 0.38 | pass/fail |
| A | 3 (1032) | ag-d71260f87c | -0.28 | -0.24 / -0.32 | 2.29% | 0.53x | 5.5x | 103x | 93% | 4 | +0.0472 (-0.3188, +0.3563) | y/n | 2.25 | -0.30 | 164 | 0.3%→2.3% | 0.061 | 0.36 | pass/pass |
| A | 4 (1037) | ag-0c86f02206 | -0.98 | -0.97 / -0.99 | 4.76% | 0.82x | 9.2x | 190x | 88% | 9 | -0.6561 (-1.1604, -0.1700) | n/n | 1.52 | -1.01 | 234 | 0.2%→4.8% | 0.044 | 0.33 | fail/fail |
| A | 5 (1042) | ag-c58090a8d9 | -1.50 | -1.47 / -1.53 | 7.10% | 0.53x | 13.1x | 758x | 94% | 8 | -0.2502 (-0.8062, +0.3085) | n/n | 1.10 | -1.42 | 506 | 0.5%→6.5% | 0.032 | 0.27 | fail/fail |
| A | 6 (1047) | ag-04c0c4b75e | -1.49 | -1.59 / -1.40 | 6.26% | 0.85x | 11.4x | 295x | 76% | 7 | -0.4277 (-0.9191, +0.0361) | n/n | 1.21 | -1.65 | 391 | 0.3%→7.1% | 0.074 | 0.40 | fail/fail |

## Learning curves (development round; descriptive only, never used for any decision)

Group mean across seeds of Q2 / ruinous share / median size, by training steps.

| Group | 500k | 1000k | 1500k | 2000k | 2500k | 3000k |
|---|---|---|---|---|---|---|
| control | +0.06 / 0.9% / 0.51x | -1.86 / 8.4% / 1.42x | -2.59 / 10.4% / 1.47x | -2.66 / 11.0% / 1.40x | -3.21 / 12.5% / 1.73x | -3.01 / 11.8% / 1.34x |
| C1 | +0.28 / 0.0% / 0.53x | +0.40 / 0.0% / 0.81x | +0.43 / 0.0% / 0.88x | +0.44 / 0.0% / 0.90x | +0.46 / 0.0% / 0.93x | +0.47 / 0.0% / 0.93x |
| C2 | +0.16 / 0.0% / 0.30x | +0.24 / 0.0% / 0.47x | +0.28 / 0.0% / 0.57x | +0.30 / 0.0% / 0.61x | +0.31 / 0.0% / 0.62x | +0.32 / 0.0% / 0.64x |
| D | +0.31 / 0.0% / 0.50x | -0.12 / 1.9% / 1.15x | -1.90 / 8.7% / 1.48x | -3.06 / 13.6% / 1.77x | -3.21 / 14.1% / 1.74x | -3.09 / 13.5% / 1.70x |
| A | +0.20 / 0.2% / 0.39x | -0.24 / 1.9% / 0.60x | -0.52 / 3.0% / 0.54x | -0.43 / 2.7% / 0.50x | -0.67 / 3.5% / 0.53x | -0.80 / 3.9% / 0.56x |

## Predictions recorded before training

- P-C (from the diagnosis, hypotheses H1/H2): C1 and C2 reduce the ruinous-decision share relative to control.
- P-D (H3): D lowers per-update KL and clip fraction and the previous-move tilt relative to control.
- P-A (falsification of the horizon explanation): A does not meet G1 and does not reduce the mean ruinous share by more than half relative to control.
- If none of C1, C2 or D meets the criteria, the evidence points at the reward basin (finding F4); any change to the reward would need a separate proposal and the owner's approval.

## How to read this (limits of six seeds)

- Six seeds per group detect only large differences (about one between-seed standard deviation). A group that misses G1 has not been shown to be ineffective; its effect may be smaller than this experiment can resolve. The intervals above show the range of effects consistent with the data.
- The verdict for each group is the pre-registered rule applied mechanically; no threshold was changed after training began.
- Every agent, version, checkpoint and evaluation is preserved, including failures and interrupted attempts.

## Protection of the original system

- Original registry content hash (before / now): 1ba10554e4724f90… / 1ba10554e4724f90… (identical)
- Original registry seal verification: all sealed records intact
- Experiment registry seal verification: all sealed records intact
- Original data ledger: read only (inherited); no seed reused
