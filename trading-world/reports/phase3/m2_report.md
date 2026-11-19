# Experiment M2: model-based growth-optimal agent v2 (phase3-m2)

Pre-registered plan: `registry-m2/experiments/phase3-m2/manifest.json` (planned 2026-11-19T12:24:22Z); configuration and code checksums verified identical before training, evaluation and this analysis.

Each agent learns from its own fresh 200,000-bar market: it fits an unconditional return mixture (exactly, by EM) and a conditional mixture-density network, keeps whichever predicts held-out experience better, corrects its edge for estimation error (James-Stein), and sizes every decision for maximum expected log growth. Nothing about the market is given to it. Final checkpoints only; six fresh rounds of 20 runs (120 runs per agent). Benchmarks hold the exact growth-optimal size on the same runs.

**M2 passes (all three markets): YES**

**Protocol amendment 0001** (2026-11-19T12:41:53Z; recorded before any result was analysed): The run stopped with 'unknown evaluation round m2-s4-curve-199k'. Each agent's 200,000-bar market yields 199,967 learning examples (after the 30-bar observation window and the one-bar-ahead target), so the last checkpoint's step is 199,967, and the runner named its learning-curve round from that step ('199k') instead of the planned '200k' round. The runner now maps each checkpoint step to the nearest planned data point; the report lists every amendment. Effect: round labels and the report only. Agents, data, training, test rounds, criteria, analysis rules and the evaluated checkpoints are unchanged; the last learning-curve point is the final checkpoint (199,967 examples), which the pre-registered analysis already treats as the 200,000-bar point. State when made: agents trained 18, evaluations analysed or inspected 0, evaluations recorded 6.

## S6: drift plus rare crashes (optimal 2.93x)

Successful seeds: **6/6** → market **passes**; interpretation: **safe growth**.

Criterion counts: W1 6/6, W2 6/6, W3 6/6, W5 6/6, W6 6/6, W4 6/6, W7 6/6, E 6/6.

Models chosen by the agents: 6 unconditional, 0 conditional. Benchmark (exact optimal 2.93x): growth +0.2555 per run (95% CI +0.1321, +0.3784), 0 deaths in 120 runs.

Group: Q2 +1.00 (+0.99, +1.00); ruinous 0.00%; median size 3.04x (2.92, 3.15); 0 deaths in 720 runs.

| Comparison (Holm over 6) | MK2-S6 minus reference (95% CI) | Reference mean | p raw / Holm |
|---|---|---|---|
| MK2-S6 vs M1 MK-S6: q2 | +0.280 (+0.168, +0.377) | +0.715 | 0.002 / 0.013 |
| MK2-S6 vs M1 MK-S6: ruinous | +0.000 (+0.000, +0.000) | +0.000 | 1.000 / 1.000 |
| MK2-S6 vs M1 MK-S6: size_distance | -0.186 (-0.215, -0.155) | +0.236 | 0.002 / 0.013 |
| MK2-S6 vs R-FL R1 control: q2 | +2.669 (+1.981, +3.327) | -1.673 | 0.002 / 0.013 |
| MK2-S6 vs R-FL R1 control: ruinous | -0.075 (-0.101, -0.047) | +0.075 | 0.002 / 0.013 |
| MK2-S6 vs R-FL R1 control: size_distance | -1.165 (-1.541, -0.794) | +1.215 | 0.002 / 0.013 |

| Seed | Agent | Model | Edge t / shrink | Median size | Largest | Ruinous | Deaths | Growth per run (95% CI) | t | Rounds beating random | Q2 | Tilt | Learning curve: median size (Q2) at 20k → 60k → 200k bars | Success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (1066) | ag-a2ee93fdf0 | unconditional K=3 | 12.9 / 0.99 | +3.25x | 3.25x | 0.00% | 0 | +0.2506 (+0.1109, +0.3896) | +3.5 | 6/6 | +0.98 | 1.00 | 3.79x (+0.89) → 2.79x (+1.00) → 3.25x (+0.98) | **yes** |
| 2 (1067) | ag-042b699ece | unconditional K=3 | 11.6 / 0.99 | +2.97x | 2.97x | 0.00% | 0 | +0.2553 (+0.1302, +0.3799) | +4.0 | 6/6 | +1.00 | 1.00 | 3.32x (+0.98) → 2.95x (+1.00) → 2.97x (+1.00) | **yes** |
| 3 (1068) | ag-8ac2fe3c88 | unconditional K=6 | 10.8 / 0.99 | +2.79x | 2.79x | 0.00% | 0 | +0.2552 (+0.1387, +0.3714) | +4.3 | 6/6 | +1.00 | 1.00 | 2.66x (+0.99) → 2.71x (+0.99) → 2.79x (+1.00) | **yes** |
| 4 (1069) | ag-cc04b3e799 | unconditional K=2 | 12.0 / 0.99 | +3.08x | 3.08x | 0.00% | 0 | +0.2542 (+0.1235, +0.3844) | +3.8 | 6/6 | +1.00 | 1.00 | 2.59x (+0.98) → 3.01x (+1.00) → 3.08x (+1.00) | **yes** |
| 5 (1070) | ag-e3818a2b71 | unconditional K=2 | 11.6 / 0.99 | +2.97x | 2.97x | 0.00% | 0 | +0.2553 (+0.1299, +0.3802) | +4.0 | 6/6 | +1.00 | 1.00 | 3.43x (+0.96) → 3.20x (+0.99) → 2.97x (+1.00) | **yes** |
| 6 (1071) | ag-b703757170 | unconditional K=2 | 12.4 / 0.99 | +3.15x | 3.15x | 0.00% | 0 | +0.2530 (+0.1185, +0.3869) | +3.7 | 6/6 | +0.99 | 1.00 | 1.87x (+0.85) → 3.17x (+0.99) → 3.15x (+0.99) | **yes** |

## S7: drift, no crashes (optimal 5x)

Successful seeds: **4/6** → market **passes**.

Criterion counts: W1 6/6, W2 6/6, W3 6/6, W5 6/6, W6 6/6, W7 4/6, E 6/6.

Models chosen by the agents: 6 unconditional, 0 conditional. Benchmark (exact optimal 4.99x): growth +1.5659 per run (95% CI +1.2803, +1.8498), 1 deaths in 120 runs.

| Seed | Agent | Model | Edge t / shrink | Median size | Largest | Ruinous | Deaths | Growth per run (95% CI) | t | Rounds beating random | Q2 | Tilt | Learning curve: median size (Q2) at 20k → 60k → 200k bars | Success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (1006) | ag-d4f24a9a38 | unconditional K=3 | 23.3 / 1.00 | +5.19x | 5.19x | 0.00% | 2 | +1.5701 (+1.2706, +1.8657) | +10.3 | 6/6 | +1.00 | 1.00 | 4.87x (+1.00) → 4.88x (+1.00) → 5.19x (+1.00) | no |
| 2 (1007) | ag-ec467eb6a8 | unconditional K=1 | 21.3 / 1.00 | +4.74x | 4.74x | 0.00% | 0 | +1.5467 (+1.2752, +1.8165) | +11.2 | 6/6 | +1.00 | 1.00 | 4.68x (+1.00) → 4.99x (+1.00) → 4.74x (+1.00) | **yes** |
| 3 (1008) | ag-cee00b3985 | unconditional K=2 | 23.1 / 1.00 | +5.16x | 5.16x | 0.00% | 2 | +1.5689 (+1.2705, +1.8635) | +10.3 | 6/6 | +1.00 | 1.00 | 4.71x (+1.00) → 4.77x (+1.00) → 5.16x (+1.00) | **yes** |
| 4 (1009) | ag-b1fafd1b07 | unconditional K=2 | 21.1 / 1.00 | +4.71x | 4.71x | 0.00% | 0 | +1.5434 (+1.2740, +1.8112) | +11.2 | 6/6 | +1.00 | 1.00 | 6.04x (+0.96) → 5.43x (+0.99) → 4.71x (+1.00) | **yes** |
| 5 (1010) | ag-51581b02c7 | unconditional K=3 | 21.8 / 1.00 | +4.85x | 4.86x | 0.00% | 0 | +1.5560 (+1.2782, +1.8322) | +11.0 | 6/6 | +1.00 | 1.00 | 4.89x (+1.00) → 4.63x (+0.99) → 4.85x (+1.00) | no |
| 6 (1011) | ag-18501b2212 | unconditional K=6 | 23.4 / 1.00 | +5.21x | 5.21x | 0.00% | 2 | +1.5717 (+1.2709, +1.8687) | +10.2 | 6/6 | +1.00 | 1.00 | 4.76x (+1.00) → 5.29x (+1.00) → 5.21x (+1.00) | **yes** |

## S4: no edge (right behaviour: claim none)

Successful seeds: **6/6** → market **passes**.

Criterion counts: no_edge 6/6, no_deaths 6/6, E 6/6.

Models chosen by the agents: 6 unconditional, 0 conditional. Benchmark (exact optimal 0.00x): growth +0.0000 per run (95% CI +0.0000, +0.0000), 0 deaths in 120 runs.

| Seed | Agent | Model | Edge t / shrink | Median size | Largest | Ruinous | Deaths | Growth per run (95% CI) | t | Rounds beating random | Q2 | Tilt | Learning curve: median size (Q2) at 20k → 60k → 200k bars | Success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (1000) | ag-6c5707a489 | unconditional K=2 | 0.5 / 0.00 | +0.00x | 0.00x | 0.00% | 0 | +0.0000 (+0.0000, +0.0000) | +0.0 | 0/6 | — | 1.00 | 0.00x (—) → 0.00x (—) → 0.00x (—) | **yes** |
| 2 (1001) | ag-8e6a147f71 | unconditional K=2 | -0.9 / 0.00 | +0.00x | 0.00x | 0.00% | 0 | +0.0000 (+0.0000, +0.0000) | +0.0 | 0/6 | — | 1.00 | 0.00x (—) → 0.00x (—) → 0.00x (—) | **yes** |
| 3 (1002) | ag-1fd2344864 | unconditional K=4 | 1.2 / 0.35 | +0.98x | 0.98x | 0.00% | 0 | +0.0009 (-0.0043, +0.0061) | +0.3 | 0/6 | — | 1.00 | 0.00x (—) → 0.00x (—) → 0.98x (—) | **yes** |
| 4 (1003) | ag-7700f9c047 | unconditional K=2 | 1.0 / 0.00 | +0.00x | 0.00x | 0.00% | 0 | +0.0000 (+0.0000, +0.0000) | +0.0 | 0/6 | — | 1.00 | 0.00x (—) → 0.00x (—) → 0.00x (—) | **yes** |
| 5 (1004) | ag-d3c078f86c | unconditional K=2 | 0.7 / 0.00 | +0.00x | 0.00x | 0.00% | 0 | +0.0000 (+0.0000, +0.0000) | +0.0 | 0/6 | — | 1.00 | 0.07x (—) → 0.00x (—) → 0.00x (—) | **yes** |
| 6 (1005) | ag-af66c14aa6 | unconditional K=3 | 1.6 / 0.59 | +2.06x | 2.06x | 0.00% | 0 | +0.0007 (-0.0102, +0.0118) | +0.1 | 1/6 | — | 1.00 | 10.11x (—) → 0.00x (—) → 2.06x (—) | **yes** |

## Limits

- Six seeds per market; intervals show the range consistent with the data.
- S4, S6 and S7 have i.i.d. returns and no trading costs; predictable markets with costs are the next test.
- Benchmarks are given the true market parameters and are references only, never agents.
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
- Experiment M2 registry: seal verification: all sealed records intact
