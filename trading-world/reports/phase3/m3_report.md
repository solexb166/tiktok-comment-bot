# Experiment M3: model-based growth-optimal agent v3, trading costs (phase3-m3)

Pre-registered plan: `registry-m3/experiments/phase3-m3/manifest.json` (planned 2026-11-19T14:12:45Z); configuration and code checksums verified identical before training, evaluation and this analysis.

Each agent learns from its own fresh 200,000-bar market (version 2's world models and model choice), checks its conditional predictions on held-out experience and shrinks them by how well they held up, measures how long its predicted edge persists, and decides each position for maximum expected log growth after the cost of getting there (half the quoted spread per unit traded, weighed per bar by the edge's persistence). It reads the spread and its position from its own observation; nothing about the market is given to it. Final checkpoints only; six fresh rounds of 20 runs (120 runs per agent). References use the same decision with the market's true parameters (S6: the exact optimal constant size) on the same runs.

**M3 passes (all four markets): YES**

## S1C: bounce-back pattern, trading costs 4x S1's

Successful seeds: **6/6** → market **passes**.

Criterion counts: C1 6/6, C2 6/6, C3 6/6, C4 6/6, C5 6/6, E 6/6.

Models chosen: 0 unconditional, 6 conditional. Group growth per run +3.658 (95% CI +3.611, +3.701); 8 deaths in 720 runs.

Reference (true-parameter cost-aware decision): growth +3.659 per run (95% CI +3.273, +4.044), 0 deaths in 120 runs.

Cost-blind ablation (same learned models, trading treated as free): growth -2.388 per run, 568 deaths in 720 runs. Version 3 minus ablation: +6.046 (95% CI +5.964, +6.139); sign-flip p = 0.031, Holm 0.062.

| Seed | Agent | Model | Calibration slope (k) | Persistence | Median / largest size | In market | Trades | Edge side when edge > cost | Costs/run ($10 start) | Deaths (agent-only / reference-only, p) | Growth per run (95% CI) | t | Rounds beating random | Growth at 20k → 60k → 200k bars | Failed | Success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (1000) | ag-d2a2108487 | conditional | 0.95 (0.95) | -0.305 | 103.4x / 155.4x | 100% | 37% | 95.3% of 48,655 | 12498.062 | 3 (3/0, p=0.12) | +3.670 (+3.223, +4.102) | +16.3 | 6/6 | +3.57 → +3.59 → +3.63 | — | **yes** |
| 2 (1001) | ag-99a1bf095f | conditional | 1.04 (1.00) | -0.286 | 95.5x / 141.5x | 100% | 36% | 95.0% of 49,033 | 7359.781 | 0 (0/0, p=1.00) | +3.648 (+3.241, +4.055) | +17.7 | 6/6 | +3.25 → +3.71 → +3.89 | — | **yes** |
| 3 (1002) | ag-be32093d1f | conditional | 0.97 (0.97) | -0.287 | 99.8x / 155.7x | 100% | 37% | 95.1% of 48,631 | 13013.542 | 2 (2/0, p=0.25) | +3.681 (+3.220, +4.137) | +15.9 | 6/6 | +3.50 → +3.59 → +3.65 | — | **yes** |
| 4 (1003) | ag-4ab37c109c | conditional | 0.91 (0.91) | -0.295 | 94.7x / 149.0x | 100% | 38% | 96.6% of 48,733 | 11045.200 | 1 (1/0, p=0.50) | +3.747 (+3.319, +4.171) | +17.4 | 6/6 | +3.91 → +4.04 → +3.81 | — | **yes** |
| 5 (1004) | ag-cb99f20d01 | conditional | 1.00 (1.00) | -0.296 | 96.3x / 147.7x | 100% | 37% | 95.9% of 48,872 | 11637.632 | 1 (1/0, p=0.50) | +3.643 (+3.209, +4.073) | +16.7 | 6/6 | +3.52 → +3.79 → +3.81 | — | **yes** |
| 6 (1005) | ag-d153151655 | conditional | 0.99 (0.99) | -0.296 | 93.4x / 138.2x | 100% | 36% | 95.4% of 48,819 | 9730.976 | 1 (1/0, p=0.50) | +3.558 (+3.148, +3.963) | +17.1 | 6/6 | +3.12 → +3.55 → +3.74 | — | **yes** |

## S2C: announced trends, trading cost wider than a bar's drift

Successful seeds: **5/6** → market **passes**.

Criterion counts: C1 6/6, C2 6/6, C3 6/6, C4 5/6, C5 6/6, E 6/6.

Models chosen: 0 unconditional, 6 conditional. Group growth per run +4.800 (95% CI +4.612, +5.055); 43 deaths in 720 runs.

Reference (true-parameter cost-aware decision): growth +5.610 per run (95% CI +5.008, +6.196), 4 deaths in 120 runs.

Cost-blind ablation (same learned models, trading treated as free): growth +2.745 per run, 95 deaths in 720 runs. Version 3 minus ablation: +2.055 (95% CI +1.650, +2.567); sign-flip p = 0.031, Holm 0.062.

| Seed | Agent | Model | Calibration slope (k) | Persistence | Median / largest size | In market | Trades | Edge side when edge > cost | Costs/run ($10 start) | Deaths (agent-only / reference-only, p) | Growth per run (95% CI) | t | Rounds beating random | Growth at 20k → 60k → 200k bars | Failed | Success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (1000) | ag-c3b4672937 | conditional | 0.94 (0.94) | +0.950 | 108.8x / 133.0x | 100% | 45% | 100.0% of 116,546 | 86409.935 | 6 (2/0, p=0.25) | +5.401 (+4.770, +6.019) | +17.0 | 6/6 | +6.93 → +6.16 → +5.94 | — | **yes** |
| 2 (1001) | ag-f0dfaa3499 | conditional | 1.03 (1.00) | +0.955 | 110.6x / 136.1x | 100% | 47% | 100.0% of 114,713 | 65240.762 | 10 (6/0, p=0.02) | +4.547 (+3.896, +5.190) | +13.8 | 6/6 | +6.42 → +5.32 → +5.07 | C4 | no |
| 3 (1002) | ag-79a66979c2 | conditional | 0.90 (0.90) | +0.959 | 110.8x / 131.6x | 100% | 48% | 100.0% of 115,195 | 69330.432 | 8 (4/0, p=0.06) | +4.692 (+4.039, +5.328) | +14.3 | 6/6 | +5.92 → +5.23 → +5.15 | — | **yes** |
| 4 (1003) | ag-a256942132 | conditional | 0.86 (0.86) | +0.943 | 108.0x / 135.8x | 100% | 45% | 100.0% of 116,213 | 64939.476 | 6 (2/0, p=0.25) | +4.857 (+4.230, +5.462) | +15.4 | 6/6 | +6.04 → +5.91 → +5.03 | — | **yes** |
| 5 (1004) | ag-359f5eb876 | conditional | 0.95 (0.95) | +0.955 | 108.0x / 134.1x | 100% | 48% | 100.0% of 116,747 | 48677.097 | 7 (3/0, p=0.12) | +4.581 (+3.963, +5.199) | +14.6 | 6/6 | +6.41 → +5.20 → +5.11 | — | **yes** |
| 6 (1005) | ag-a5571c9cbf | conditional | 0.95 (0.95) | +0.953 | 107.7x / 135.6x | 100% | 48% | 100.0% of 116,816 | 61339.905 | 6 (2/0, p=0.25) | +4.724 (+4.099, +5.347) | +14.8 | 6/6 | +6.89 → +5.15 → +5.14 | — | **yes** |

## S3: pure noise with trading costs (right behaviour: do not trade)

Successful seeds: **6/6** → market **passes**.

Criterion counts: N1 6/6, N2 6/6, N3 6/6, E 6/6.

Models chosen: 6 unconditional, 0 conditional. Group growth per run +0.000 (95% CI +0.000, +0.000); 0 deaths in 720 runs.

Reference (true-parameter cost-aware decision): growth +0.000 per run (95% CI +0.000, +0.000), 0 deaths in 120 runs.

| Seed | Agent | Model | Calibration slope (k) | Persistence | Median / largest size | In market | Trades | Edge side when edge > cost | Costs/run ($10 start) | Deaths (agent-only / reference-only, p) | Growth per run (95% CI) | t | Rounds beating random | Growth at 20k → 60k → 200k bars | Failed | Success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (1000) | ag-22f2e455ad | unconditional | — (—) | +0.000 | 0.0x / 0.0x | 0% | 0% | — | 0.000 | 0 | +0.000 (+0.000, +0.000) | +0.0 | 0/6 | +0.00 → +0.00 → +0.00 | — | **yes** |
| 2 (1001) | ag-29b439775f | unconditional | — (—) | +0.000 | 0.0x / 0.0x | 0% | 0% | — | 0.000 | 0 | +0.000 (+0.000, +0.000) | +0.0 | 0/6 | +0.00 → +0.00 → +0.00 | — | **yes** |
| 3 (1002) | ag-c11d6ce64e | unconditional | — (—) | +0.000 | 0.0x / 0.0x | 0% | 0% | — | 0.000 | 0 | +0.000 (+0.000, +0.000) | +0.0 | 0/6 | +0.00 → +0.00 → +0.00 | — | **yes** |
| 4 (1003) | ag-7024820835 | unconditional | — (—) | +0.000 | 0.0x / 0.0x | 0% | 0% | — | 0.000 | 0 | +0.000 (+0.000, +0.000) | +0.0 | 0/6 | +0.00 → +0.00 → +0.00 | — | **yes** |
| 5 (1004) | ag-d3052fbc46 | unconditional | — (—) | +0.000 | 0.0x / 0.0x | 0% | 0% | — | 0.000 | 0 | +0.000 (+0.000, +0.000) | +0.0 | 0/6 | +0.00 → -0.01 → +0.00 | — | **yes** |
| 6 (1005) | ag-11568bd6fa | unconditional | — (—) | +0.000 | 0.0x / 0.0x | 0% | 0% | — | 0.000 | 0 | +0.000 (+0.000, +0.000) | +0.0 | 0/6 | +0.00 → +0.00 → +0.00 | — | **yes** |

## S6: drift plus rare crashes, no costs (regression check against version 2)

Successful seeds: **4/6** → market **passes**.

Criterion counts: W1 6/6, W2 6/6, W3 6/6, W5 6/6, W6 6/6, W4 6/6, W7 4/6, E 6/6.

Models chosen: 6 unconditional, 0 conditional. Group growth per run +0.300 (95% CI +0.296, +0.303); 0 deaths in 720 runs.

Reference (exact optimal constant size): growth +0.302 per run (95% CI +0.190, +0.413), 0 deaths in 120 runs.

| Seed | Agent | Model | Median size | Ruinous | Deaths | Growth per run (95% CI) | Rounds beating random | Q2 | Costs/run | Success |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (1072) | ag-5443c8fd59 | unconditional | +2.59x | 0.00% | 0 | +0.2928 (+0.1956, +0.3893) | 5/6 | +0.98 | 0.0000 | **yes** |
| 2 (1073) | ag-b6e7b7b844 | unconditional | +3.15x | 0.00% | 0 | +0.3041 (+0.1821, +0.4247) | 6/6 | +0.99 | 0.0000 | **yes** |
| 3 (1074) | ag-86d499dfd0 | unconditional | +3.13x | 0.00% | 0 | +0.3041 (+0.1828, +0.4240) | 6/6 | +0.99 | 0.0000 | **yes** |
| 4 (1075) | ag-7afb001f5c | unconditional | +3.22x | 0.00% | 0 | +0.3039 (+0.1788, +0.4277) | 6/6 | +0.99 | 0.0000 | **yes** |
| 5 (1076) | ag-e2fff0a7e1 | unconditional | +2.78x | 0.00% | 0 | +0.2990 (+0.1937, +0.4035) | 6/6 | +1.00 | 0.0000 | no |
| 6 (1077) | ag-73d46f5c33 | unconditional | +2.61x | 0.00% | 0 | +0.2934 (+0.1955, +0.3905) | 6/6 | +0.99 | 0.0000 | no |

## Limits

- Six seeds per market; intervals show the range consistent with the data.
- These are still synthetic markets with one instrument and known structure; real historical data is the next test.
- The decision weighs costs one bar ahead, spread over the edge's measured persistence; it does not plan a full sequence of future trades.
- References are given the true market parameters and are references only, never agents.
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
- Experiment M1 registry: content hash before / now: 30abc076d23a4651… / 30abc076d23a4651… (identical)
- Experiment M1 registry: seal verification: all sealed records intact
- Experiment M2 registry: content hash before / now: d892c95248bd7658… / d892c95248bd7658… (identical)
- Experiment M2 registry: seal verification: all sealed records intact
- Experiment M3 registry: seal verification: all sealed records intact
