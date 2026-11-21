# Experiment M4: model-based growth-optimal agent v4, markets that change (phase3-m4)

Pre-registered plan: `registry-m4/experiments/phase3-m4/manifest.json` (planned 2026-11-21T17:13:43Z); configuration and code checksums verified identical before training, evaluation and this analysis.

Each agent learns from its own fresh 200,000-bar market a regime-switching world model: how many hidden regimes the market has (one = it does not change), how each behaves (pattern, trend response, volatility, crash frequency) and how long each lasts. While trading it updates its belief about the current regime with every bar it sees and decides as version 3 (growth after costs). Nothing about the market is given to it. Final checkpoints only; six fresh rounds of 20 runs (120 runs per agent). The true-model filter knows the market's rules and infers the regime like the agent; the oracle is told the regime (an upper bound).

**M4 passes (all five markets): NO**

## S1X: bounce-back pattern that flips to momentum and back at unannounced times

Successful seeds: **6/6** → market **passes**.

Criterion counts: A1 6/6, A2 6/6, A3 6/6, A4 6/6, A5 6/6, E 6/6.

Regimes found: 6 agents chose 2. Group growth per run +4.502 (95% CI +4.452, +4.559); 7 deaths in 720 runs.

Reference (true-filter): growth +4.422 per run (95% CI +3.868, +4.982), 1 deaths in 120 runs.

Oracle (told the current regime; upper bound): growth +5.570 per run, 0 deaths in 120 runs. Own single-regime model (ablation): growth -0.269 per run, 16 deaths in 720 runs.

| Seed | Agent | Regimes (mean lengths) | Median / largest size | Trades | Deaths (agent-only / ref-only, p) | Growth per run (95% CI) | Gain over own single-regime model (95% CI) | Rounds beating random | Failed | Success |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (1000) | ag-c67f1a8bf6 | 2 (3235, 2998) | 93.2x / 99.7x | 49% | 1 (0/0, p=1.00) | +4.441 (+3.884, +4.998) | +5.157 (+4.640, +5.699) | 6/6 | — | **yes** |
| 2 (1001) | ag-c0ef1f141d | 2 (4147, 3177) | 95.7x / 102.1x | 49% | 1 (0/0, p=1.00) | +4.454 (+3.891, +5.018) | +4.600 (+4.003, +5.209) | 6/6 | — | **yes** |
| 3 (1002) | ag-c23e671e16 | 2 (1884, 3139) | 92.8x / 100.1x | 47% | 1 (0/0, p=1.00) | +4.475 (+3.947, +5.014) | +4.602 (+4.005, +5.204) | 6/6 | — | **yes** |
| 4 (1003) | ag-ee463b1e2c | 2 (3685, 2630) | 98.7x / 107.3x | 46% | 1 (0/0, p=1.00) | +4.617 (+4.052, +5.186) | +4.794 (+4.184, +5.402) | 6/6 | — | **yes** |
| 5 (1004) | ag-dbc9dca543 | 2 (2937, 2717) | 92.2x / 99.7x | 45% | 1 (0/0, p=1.00) | +4.576 (+4.048, +5.115) | +4.932 (+4.411, +5.471) | 6/6 | — | **yes** |
| 6 (1005) | ag-ed43a7b945 | 2 (2585, 4042) | 93.2x / 99.8x | 49% | 2 (1/0, p=0.50) | +4.452 (+3.897, +5.006) | +4.544 (+3.941, +5.148) | 6/6 | — | **yes** |

## S2X: announced trends whose meaning flips at unannounced times

Successful seeds: **6/6** → market **passes**.

Criterion counts: A1 6/6, A2 6/6, A3 6/6, A4 6/6, A5 6/6, E 6/6.

Regimes found: 6 agents chose 2. Group growth per run +3.566 (95% CI +3.484, +3.639); 102 deaths in 720 runs.

Reference (true-filter): growth +3.705 per run (95% CI +3.056, +4.347), 16 deaths in 120 runs.

Oracle (told the current regime; upper bound): growth +5.313 per run, 6 deaths in 120 runs. Own single-regime model (ablation): growth +0.010 per run, 6 deaths in 720 runs.

| Seed | Agent | Regimes (mean lengths) | Median / largest size | Trades | Deaths (agent-only / ref-only, p) | Growth per run (95% CI) | Gain over own single-regime model (95% CI) | Rounds beating random | Failed | Success |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (1000) | ag-790b6c0736 | 2 (3056, 3171) | 100.1x / 100.7x | 54% | 17 (1/0, p=0.50) | +3.625 (+2.965, +4.271) | +3.625 (+2.965, +4.271) | 6/6 | — | **yes** |
| 2 (1001) | ag-a7b98c73f3 | 2 (3904, 3637) | 97.7x / 98.8x | 54% | 16 (0/0, p=1.00) | +3.687 (+3.035, +4.325) | +3.687 (+3.035, +4.325) | 6/6 | — | **yes** |
| 3 (1002) | ag-768b64af6b | 2 (4358, 2901) | 99.6x / 101.1x | 54% | 18 (2/0, p=0.25) | +3.408 (+2.752, +4.057) | +3.434 (+2.773, +4.087) | 6/6 | — | **yes** |
| 4 (1003) | ag-7ab5aa9c99 | 2 (2020, 2452) | 97.2x / 98.3x | 54% | 16 (1/1, p=0.75) | +3.636 (+2.986, +4.275) | +3.614 (+2.959, +4.259) | 6/6 | — | **yes** |
| 5 (1004) | ag-b34a4f4fed | 2 (2984, 3674) | 99.9x / 100.7x | 54% | 18 (2/0, p=0.25) | +3.524 (+2.858, +4.176) | +3.462 (+2.790, +4.123) | 6/6 | — | **yes** |
| 6 (1005) | ag-979ad80e78 | 2 (3319, 3654) | 100.3x / 101.3x | 54% | 17 (1/0, p=0.50) | +3.517 (+2.859, +4.172) | +3.517 (+2.859, +4.172) | 6/6 | — | **yes** |

## S6X: crash frequency switching between calm and storm spells

Successful seeds: **3/6** → market **does not pass**.

Criterion counts: A1 6/6, A2 5/6, A3 3/6, A4 6/6, A5 3/6, E 6/6.

Regimes found: 3 agents chose 1, 3 agents chose 3. Group growth per run +0.997 (95% CI +0.643, +1.360); 3 deaths in 720 runs.

Reference (true-filter): growth +1.970 per run (95% CI +1.257, +2.778), 0 deaths in 120 runs.

Oracle (told the current regime; upper bound): growth +2.877 per run, 2 deaths in 120 runs. Own single-regime model (ablation): growth +0.497 per run, 0 deaths in 720 runs.

| Seed | Agent | Regimes (mean lengths) | Median / largest size | Trades | Deaths (agent-only / ref-only, p) | Growth per run (95% CI) | Gain over own single-regime model (95% CI) | Rounds beating random | Failed | Success |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (1000) | ag-85d9b12339 | 3 (187, 187, 1) | 3.3x / 27.6x | 43% | 1 (1/0, p=0.50) | +1.434 (+0.923, +1.994) | +1.030 (+0.668, +1.422) | 6/6 | — | **yes** |
| 2 (1001) | ag-12fba08f38 | 3 (260, 1, 211) | 3.5x / 8.4x | 47% | 0 (0/0, p=1.00) | +1.340 (+0.880, +1.848) | +1.010 (+0.666, +1.379) | 6/6 | — | **yes** |
| 3 (1002) | ag-76dd81e54b | 1 (1000000000000) | 3.4x / 3.4x | 44% | 0 (0/0, p=1.00) | +0.634 (+0.276, +1.013) | +0.000 (+0.000, +0.000) | 5/6 | A3, A5 | no |
| 4 (1003) | ag-011aa9d2b2 | 1 (1000000000000) | 1.9x / 1.9x | 44% | 0 (0/0, p=1.00) | +0.446 (+0.237, +0.668) | +0.000 (+0.000, +0.000) | 4/6 | A2, A3, A5 | no |
| 5 (1004) | ag-c16ea7cce9 | 1 (1000000000000) | 2.3x / 2.3x | 44% | 0 (0/0, p=1.00) | +0.519 (+0.262, +0.791) | +0.000 (+0.000, +0.000) | 5/6 | A3, A5 | no |
| 6 (1005) | ag-ba87dffcaa | 3 (223, 1, 213) | 3.7x / 13.9x | 42% | 2 (2/0, p=0.25) | +1.611 (+0.971, +2.314) | +0.964 (+0.609, +1.344) | 6/6 | — | **yes** |

## S1C: no change (regression check against version 3)

Successful seeds: **6/6** → market **passes**.

Criterion counts: C1 6/6, C2 6/6, C3 6/6, C4 6/6, C5 6/6, E 6/6.

Regimes found: 6 agents chose 1. Group growth per run +3.626 (95% CI +3.612, +3.639); 22 deaths in 720 runs.

Reference (true-parameter cost-aware decision): growth +3.594 per run (95% CI +3.113, +4.077), 3 deaths in 120 runs.

| Seed | Agent | Regimes (mean lengths) | Median / largest size | Trades | Deaths (agent-only / ref-only, p) | Growth per run (95% CI) | Gain over own single-regime model (95% CI) | Rounds beating random | Failed | Success |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (1006) | ag-af0854d4a0 | 1 (1000000000000) | 98.3x / 121.3x | 36% | 3 (0/0, p=1.00) | +3.622 (+3.136, +4.115) | — | 6/6 | — | **yes** |
| 2 (1007) | ag-aaa0a1c5c2 | 1 (1000000000000) | 97.3x / 118.9x | 36% | 3 (0/0, p=1.00) | +3.596 (+3.117, +4.083) | — | 6/6 | — | **yes** |
| 3 (1008) | ag-c2b79332ad | 1 (1000000000000) | 97.8x / 121.9x | 35% | 3 (0/0, p=1.00) | +3.636 (+3.149, +4.126) | — | 6/6 | — | **yes** |
| 4 (1009) | ag-1e0c7e5f8f | 1 (1000000000000) | 102.2x / 128.1x | 36% | 5 (2/0, p=0.25) | +3.640 (+3.120, +4.161) | — | 6/6 | — | **yes** |
| 5 (1010) | ag-59459badb9 | 1 (1000000000000) | 99.9x / 128.2x | 35% | 5 (2/0, p=0.25) | +3.648 (+3.131, +4.163) | — | 6/6 | — | **yes** |
| 6 (1011) | ag-5fe62b6f50 | 1 (1000000000000) | 98.6x / 121.3x | 36% | 3 (0/0, p=1.00) | +3.614 (+3.128, +4.106) | — | 6/6 | — | **yes** |

## S6: no change, no costs (regression check against version 2)

Successful seeds: **5/6** → market **passes**.

Criterion counts: W1 6/6, W2 6/6, W3 6/6, W5 5/6, W6 5/6, W4 5/6, W7 5/6, E 6/6.

Regimes found: 5 agents chose 1, 1 agents chose 3. Group growth per run +0.261 (95% CI +0.182, +0.304); 2 deaths in 720 runs.

Reference (exact optimal constant size): growth +0.307 per run (95% CI +0.185, +0.421), 0 deaths in 120 runs.

| Seed | Agent | Regimes | Median size | Ruinous | Deaths | Growth per run (95% CI) | Rounds beating random | Q2 | Success |
|---|---|---|---|---|---|---|---|---|---|
| 1 (1078) | ag-8a9a976cdd | 1 | +2.92x | 0.00% | 0 | +0.3072 (+0.1858, +0.4204) | 6/6 | +1.00 | **yes** |
| 2 (1079) | ag-4f42eea986 | 3 | +3.44x | 0.00% | 2 | +0.0679 (-0.1319, +0.2607) | 5/6 | +0.32 | no |
| 3 (1080) | ag-4bfdc267fd | 1 | +3.50x | 0.00% | 0 | +0.3053 (+0.1548, +0.4453) | 6/6 | +0.95 | **yes** |
| 4 (1081) | ag-bd3f6939d3 | 1 | +2.79x | 0.00% | 0 | +0.3041 (+0.1891, +0.4115) | 6/6 | +1.00 | **yes** |
| 5 (1082) | ag-d9f9c98445 | 1 | +2.44x | 0.00% | 0 | +0.2901 (+0.1912, +0.3823) | 6/6 | +0.97 | **yes** |
| 6 (1083) | ag-b09839e4d7 | 1 | +2.49x | 0.00% | 0 | +0.2923 (+0.1913, +0.3863) | 6/6 | +0.97 | **yes** |

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
- Experiment M4 registry: seal verification: all sealed records intact
