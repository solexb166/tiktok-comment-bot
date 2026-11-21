# Experiment M5: model-based growth-optimal agent v5, markets that change (reliable regime fit) (phase3-m5)

Pre-registered plan: `registry-m5/experiments/phase3-m5/manifest.json` (planned 2026-11-21T20:49:32Z); configuration and code checksums verified identical before training, evaluation and this analysis.

Each agent learns from its own fresh 200,000-bar market a regime-switching world model: how many hidden regimes the market has (one = it does not change), how each behaves (pattern, trend response, volatility, crash frequency) and how long each lasts. While trading it updates its belief about the current regime with every bar it sees and decides as version 3 (growth after costs). Nothing about the market is given to it. Final checkpoints only; six fresh rounds of 20 runs (120 runs per agent). The true-model filter knows the market's rules and infers the regime like the agent; the oracle is told the regime (an upper bound).

**M5 passes (all five markets): YES**

## S1X: bounce-back pattern that flips to momentum and back at unannounced times

Successful seeds: **6/6** → market **passes**.

Criterion counts: A1 6/6, A2 6/6, A3 6/6, A4 6/6, A5 6/6, E 6/6.

Regimes found: 6 agents chose 2. Group growth per run +4.422 (95% CI +4.336, +4.500); 15 deaths in 720 runs.

Reference (true-filter): growth +4.223 per run (95% CI +3.640, +4.805), 3 deaths in 120 runs.

Oracle (told the current regime; upper bound): growth +5.268 per run, 1 deaths in 120 runs. Own single-regime model (ablation): growth -0.470 per run, 146 deaths in 720 runs.

| Seed | Agent | Regimes (mean lengths) | Median / largest size | Trades | Deaths (agent-only / ref-only, p) | Growth per run (95% CI) | Gain over own single-regime model (95% CI) | Rounds beating random | Failed | Success |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (1006) | ag-8f9a328252 | 2 (3414, 2375) | 93.4x / 100.0x | 50% | 4 (2/1, p=0.50) | +4.251 (+3.661, +4.848) | +5.044 (+4.526, +5.548) | 6/6 | — | **yes** |
| 2 (1007) | ag-ee5a682eff | 2 (3178, 3033) | 91.6x / 99.6x | 45% | 2 (0/1, p=1.00) | +4.479 (+3.946, +5.024) | +4.814 (+4.250, +5.374) | 6/6 | — | **yes** |
| 3 (1008) | ag-9835eeb35b | 2 (2895, 2084) | 93.5x / 102.3x | 45% | 3 (1/1, p=0.75) | +4.516 (+3.956, +5.083) | +4.977 (+4.465, +5.489) | 6/6 | — | **yes** |
| 4 (1009) | ag-face28a877 | 2 (2598, 2746) | 93.0x / 99.7x | 49% | 2 (0/1, p=1.00) | +4.306 (+3.758, +4.872) | +4.854 (+4.244, +5.473) | 6/6 | — | **yes** |
| 5 (1010) | ag-fe4c67c17d | 2 (2006, 3512) | 93.5x / 102.0x | 47% | 2 (0/1, p=1.00) | +4.524 (+3.979, +5.081) | +4.856 (+4.336, +5.378) | 6/6 | — | **yes** |
| 6 (1011) | ag-f6421fe695 | 2 (2772, 2670) | 91.8x / 99.8x | 45% | 2 (0/1, p=1.00) | +4.455 (+3.925, +4.997) | +4.807 (+4.240, +5.372) | 6/6 | — | **yes** |

## S2X: announced trends whose meaning flips at unannounced times

Successful seeds: **4/6** → market **passes**.

Criterion counts: A1 6/6, A2 6/6, A3 6/6, A4 4/6, A5 6/6, E 6/6.

Regimes found: 6 agents chose 2. Group growth per run +3.849 (95% CI +3.714, +3.976); 97 deaths in 720 runs.

Reference (true-filter): growth +4.016 per run (95% CI +3.324, +4.711), 14 deaths in 120 runs.

Oracle (told the current regime; upper bound): growth +5.625 per run, 2 deaths in 120 runs. Own single-regime model (ablation): growth +0.081 per run, 66 deaths in 720 runs.

| Seed | Agent | Regimes (mean lengths) | Median / largest size | Trades | Deaths (agent-only / ref-only, p) | Growth per run (95% CI) | Gain over own single-regime model (95% CI) | Rounds beating random | Failed | Success |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (1006) | ag-22d5892979 | 2 (3628, 2419) | 97.2x / 98.6x | 54% | 12 (0/2, p=1.00) | +4.064 (+3.381, +4.741) | +4.170 (+3.418, +4.947) | 6/6 | — | **yes** |
| 2 (1007) | ag-ab5744d976 | 2 (3173, 3656) | 99.3x / 100.1x | 54% | 14 (1/1, p=0.75) | +3.974 (+3.268, +4.688) | +4.009 (+3.292, +4.726) | 6/6 | — | **yes** |
| 3 (1008) | ag-f904705ea7 | 2 (3231, 2587) | 101.5x / 107.6x | 54% | 20 (6/0, p=0.02) | +3.604 (+2.864, +4.348) | +3.589 (+2.848, +4.338) | 6/6 | A4 | no |
| 4 (1009) | ag-90874c0b25 | 2 (3260, 1917) | 97.8x / 99.0x | 54% | 19 (5/0, p=0.03) | +3.677 (+2.953, +4.412) | +3.094 (+2.346, +3.886) | 6/6 | A4 | no |
| 5 (1010) | ag-2139c1e282 | 2 (3852, 3724) | 98.5x / 100.1x | 54% | 15 (3/2, p=0.50) | +3.954 (+3.238, +4.672) | +3.951 (+3.235, +4.668) | 6/6 | — | **yes** |
| 6 (1011) | ag-daa2fdce09 | 2 (3045, 3397) | 99.5x / 99.8x | 54% | 17 (3/0, p=0.12) | +3.823 (+3.091, +4.551) | +3.796 (+3.061, +4.528) | 6/6 | — | **yes** |

## S6X: crash frequency switching between calm and storm spells

Successful seeds: **6/6** → market **passes**.

Criterion counts: A1 6/6, A2 6/6, A3 6/6, A4 6/6, A5 6/6, E 6/6.

Regimes found: 5 agents chose 2, 1 agents chose 3. Group growth per run +1.226 (95% CI +1.023, +1.334); 0 deaths in 720 runs.

Reference (true-filter): growth +1.342 per run (95% CI +0.752, +2.048), 0 deaths in 120 runs.

Oracle (told the current regime; upper bound): growth +1.984 per run, 3 deaths in 120 runs. Own single-regime model (ablation): growth +0.060 per run, 0 deaths in 720 runs.

| Seed | Agent | Regimes (mean lengths) | Median / largest size | Trades | Deaths (agent-only / ref-only, p) | Growth per run (95% CI) | Gain over own single-regime model (95% CI) | Rounds beating random | Failed | Success |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (1006) | ag-58168788f9 | 3 (168, 50, 172) | 2.2x / 27.9x | 46% | 0 (0/0, p=1.00) | +0.721 (+0.185, +1.318) | +0.716 (+0.433, +1.018) | 5/6 | — | **yes** |
| 2 (1007) | ag-8f84629fee | 2 (2159, 8782) | 1.9x / 18.6x | 48% | 0 (0/0, p=1.00) | +1.306 (+0.718, +2.008) | +1.242 (+0.864, +1.681) | 6/6 | — | **yes** |
| 3 (1008) | ag-9d2cb479f7 | 2 (7280, 1311) | 2.2x / 20.5x | 55% | 0 (0/0, p=1.00) | +1.337 (+0.756, +2.033) | +1.237 (+0.796, +1.756) | 6/6 | — | **yes** |
| 4 (1009) | ag-dd86ac12ca | 2 (1633, 7557) | 2.8x / 17.4x | 59% | 0 (0/0, p=1.00) | +1.333 (+0.766, +2.005) | +1.248 (+0.771, +1.808) | 6/6 | — | **yes** |
| 5 (1010) | ag-8971557c94 | 2 (1500, 4490) | 2.6x / 19.5x | 50% | 0 (0/0, p=1.00) | +1.319 (+0.715, +2.032) | +1.313 (+0.944, +1.723) | 6/6 | — | **yes** |
| 6 (1011) | ag-b2d10138f1 | 2 (6111, 1183) | 2.7x / 20.1x | 54% | 0 (0/0, p=1.00) | +1.340 (+0.758, +2.032) | +1.241 (+0.785, +1.777) | 6/6 | — | **yes** |

## S1C: no change (regression check against version 3)

Successful seeds: **6/6** → market **passes**.

Criterion counts: C1 6/6, C2 6/6, C3 6/6, C4 6/6, C5 6/6, E 6/6.

Regimes found: 6 agents chose 1. Group growth per run +4.517 (95% CI +4.485, +4.551); 6 deaths in 720 runs.

Reference (true-parameter cost-aware decision): growth +4.446 per run (95% CI +4.009, +4.894), 1 deaths in 120 runs.

| Seed | Agent | Regimes (mean lengths) | Median / largest size | Trades | Deaths (agent-only / ref-only, p) | Growth per run (95% CI) | Gain over own single-regime model (95% CI) | Rounds beating random | Failed | Success |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (1012) | ag-df6b57f99d | 1 (1000000000000) | 98.6x / 118.8x | 36% | 1 (0/0, p=1.00) | +4.494 (+4.046, +4.951) | — | 6/6 | — | **yes** |
| 2 (1013) | ag-be1f8af6a5 | 1 (1000000000000) | 99.2x / 129.6x | 37% | 1 (0/0, p=1.00) | +4.581 (+4.122, +5.054) | — | 6/6 | — | **yes** |
| 3 (1014) | ag-0b57fb3b85 | 1 (1000000000000) | 96.4x / 122.7x | 36% | 1 (0/0, p=1.00) | +4.523 (+4.074, +4.984) | — | 6/6 | — | **yes** |
| 4 (1015) | ag-2ff6ec2666 | 1 (1000000000000) | 99.3x / 121.9x | 37% | 1 (0/0, p=1.00) | +4.555 (+4.101, +5.021) | — | 6/6 | — | **yes** |
| 5 (1016) | ag-91ca0d95c3 | 1 (1000000000000) | 96.7x / 114.9x | 37% | 1 (0/0, p=1.00) | +4.460 (+4.024, +4.907) | — | 6/6 | — | **yes** |
| 6 (1017) | ag-daf508c7d4 | 1 (1000000000000) | 98.9x / 119.0x | 36% | 1 (0/0, p=1.00) | +4.487 (+4.041, +4.945) | — | 6/6 | — | **yes** |

## S6: no change, no costs (regression check against version 2)

Successful seeds: **5/6** → market **passes**.

Criterion counts: W1 6/6, W2 6/6, W3 6/6, W5 5/6, W6 6/6, W4 6/6, W7 6/6, E 6/6.

Regimes found: 6 agents chose 1. Group growth per run +0.178 (95% CI +0.169, +0.187); 1 deaths in 720 runs.

Reference (exact optimal constant size): growth +0.182 per run (95% CI +0.047, +0.314), 0 deaths in 120 runs.

| Seed | Agent | Regimes | Median size | Ruinous | Deaths | Growth per run (95% CI) | Rounds beating random | Q2 | Success |
|---|---|---|---|---|---|---|---|---|---|
| 1 (1084) | ag-960e97b21d | 1 | +2.80x | 0.00% | 0 | +0.1857 (+0.0576, +0.3103) | 6/6 | +1.00 | **yes** |
| 2 (1085) | ag-a2e363bec2 | 1 | +3.29x | 0.00% | 0 | +0.1650 (+0.0100, +0.3157) | 6/6 | +0.98 | **yes** |
| 3 (1086) | ag-ea0b2c95f6 | 1 | +2.79x | 0.00% | 0 | +0.1858 (+0.0580, +0.3101) | 6/6 | +1.00 | **yes** |
| 4 (1087) | ag-d68b79bedb | 1 | +3.38x | 0.00% | 1 | +0.1595 (-0.0002, +0.3148) | 6/6 | +0.97 | no |
| 5 (1088) | ag-2bb50ec5a4 | 1 | +2.78x | 0.00% | 0 | +0.1861 (+0.0590, +0.3097) | 5/6 | +1.00 | **yes** |
| 6 (1089) | ag-5327faaf49 | 1 | +2.64x | 0.00% | 0 | +0.1883 (+0.0686, +0.3047) | 5/6 | +0.99 | **yes** |

## Limits

- Six seeds per market; intervals show the range consistent with the data.
- Synthetic markets with one instrument and known structure; real historical data is the next test.
- The world model is linear in the last three returns and the visible side features within each regime; version 3's non-linear network is not used by version 4.
- Full growth-optimal sizing of a weak signal that can silently reverse dies often even with the true rules (S2X); whether a live account should bet a fraction of it is the owner's risk decision.
- References are given the true market rules and are references only, never agents.
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
- Experiment M3 registry: content hash before / now: e18b3c544c6d362d… / e18b3c544c6d362d… (identical)
- Experiment M3 registry: seal verification: all sealed records intact
- Experiment M4 registry: content hash before / now: 5e8791e0db6b006f… / 5e8791e0db6b006f… (identical)
- Experiment M4 registry: seal verification: all sealed records intact
- Experiment M5 registry: seal verification: all sealed records intact
