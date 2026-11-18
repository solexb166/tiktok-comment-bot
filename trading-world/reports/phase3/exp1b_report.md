# Phase 3 Experiment 1b: C1 and C2 with exploration matched in position-size terms (phase3-exp1b)

Pre-registered plan: `registry-phase3b/experiments/phase3-exp1b/manifest.json` (planned 2026-11-18T07:28:38Z); configuration and code checksums verified identical before training, evaluation and this analysis.

Six paired seeds per group (the same fresh training seed in every group for a seed index), 3,000,000 steps each, final checkpoints only, two fresh evaluation rounds of 20 test runs. Growth-optimal size G* = 2.93x; ruinous = 9.5x or more.

| Group | Configuration | Config SHA-256 |
|---|---|---|
| C1m | size = e^z, z in [-25, 25]; starting size-noise log std -0.2797 (sigma 0.7560): x2.13 per sd at every size, as the control | `d26a79524c62…` |
| C2-tail | size = ln(1 + e^z), z in [-25, 10,000]; starting size-noise log std 1.4489 (sigma 4.2583): chance of exploring to >= 9.5x from the optimum = 6.0%, as the control | `a8551e3f5796…` |
| C2-up | size = ln(1 + e^z), z in [-25, 10,000]; starting size-noise log std 1.2131 (sigma 3.3640): +1 sd at the optimum = x2.13, as the control | `062489460079…` |

## Pre-registered verdicts

A seed is a successful growth agent only if W1–W7 and E all hold. A group supports safe growth if at least 4 of 6 seeds do. Zero ruinous decisions alone is not counted as successful learning.

| Group | W1 ruin<1% | W2 deaths≤1 | W3 size 1.47–5.86x | W4 Q2≥0.5 | W5 growth | W6 tilt<1.5 | W7 progress | E | **Successful seeds** | **Interpretation** |
|---|---|---|---|---|---|---|---|---|---|---|
| C1m | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 1/6 | 0/6 | 5/6 | **0/6** | **earlier safety came from reduced exploration** |
| C2-up | 0/6 | 0/6 | 3/6 | 0/6 | 0/6 | 2/6 | 0/6 | 5/6 | **0/6** | **earlier safety came from reduced exploration** |
| C2-tail | 0/6 | 0/6 | 4/6 | 0/6 | 0/6 | 2/6 | 0/6 | 5/6 | **0/6** | **earlier safety came from reduced exploration** |

Group means across seeds (95% bootstrap interval):

| Group | Q2 | Ruinous share | Median size | Seeds with ruinous ≥ 1% |
|---|---|---|---|---|
| C1m | -1.33 (-1.81, -0.79) | 5.56% (3.52, 7.65) | 0.66x (0.48, 0.88) | 6/6 |
| C2-up | -5.49 (-7.39, -3.68) | 20.82% (13.41, 28.55) | 3.08x (1.75, 4.45) | 6/6 |
| C2-tail | -7.18 (-9.02, -5.50) | 27.56% (20.28, 35.46) | 4.27x (2.74, 5.90) | 6/6 |

## Comparison with Experiment 1 (planned; Holm-corrected over 18 comparisons)

Difference = new group mean minus Experiment 1 group mean, per seed; distance = |ln(median size / G*)| (0 = optimal).

| Comparison | Difference (95% CI) | p raw / Holm |
|---|---|---|
| C1m vs Exp1 C1: q2 | -1.791 (-2.274, -1.266) | 0.002 / 0.039 |
| C1m vs Exp1 C1: ruinous | +0.056 (+0.035, +0.076) | 0.002 / 0.039 |
| C1m vs Exp1 C1: size_distance | +0.409 (+0.096, +0.709) | 0.041 / 0.329 |
| C1m vs Exp1 control: q2 | +1.518 (-0.326, +3.448) | 0.221 / 0.623 |
| C1m vs Exp1 control: ruinous | -0.056 (-0.126, +0.011) | 0.180 / 0.623 |
| C1m vs Exp1 control: size_distance | +0.517 (-0.012, +1.065) | 0.110 / 0.623 |
| C2-up vs Exp1 C2: q2 | -5.812 (-7.710, -4.011) | 0.002 / 0.039 |
| C2-up vs Exp1 C2: ruinous | +0.208 (+0.134, +0.285) | 0.002 / 0.039 |
| C2-up vs Exp1 C2: size_distance | -0.947 (-1.287, -0.599) | 0.002 / 0.039 |
| C2-up vs Exp1 control: q2 | -2.646 (-5.294, -0.019) | 0.104 / 0.623 |
| C2-up vs Exp1 control: ruinous | +0.097 (-0.003, +0.199) | 0.115 / 0.623 |
| C2-up vs Exp1 control: size_distance | -0.464 (-1.011, +0.107) | 0.158 / 0.623 |
| C2-tail vs Exp1 C2: q2 | -7.500 (-9.337, -5.812) | 0.002 / 0.039 |
| C2-tail vs Exp1 C2: ruinous | +0.276 (+0.203, +0.355) | 0.002 / 0.039 |
| C2-tail vs Exp1 C2: size_distance | -1.139 (-1.405, -0.844) | 0.002 / 0.039 |
| C2-tail vs Exp1 control: q2 | -4.335 (-6.939, -1.836) | 0.022 / 0.195 |
| C2-tail vs Exp1 control: ruinous | +0.164 (+0.066, +0.268) | 0.019 / 0.195 |
| C2-tail vs Exp1 control: size_distance | -0.657 (-1.171, -0.118) | 0.078 / 0.545 |

Paired by seed index between the new groups (descriptive):

| Difference | Mean | sign-flip p |
|---|---|---|
| C1m - C2-up: q2 | +4.165 | 0.031 |
| C1m - C2-up: ruinous | -0.153 | 0.031 |
| C1m - C2-up: size_distance | +0.981 | 0.031 |
| C1m - C2-tail: q2 | +5.854 | 0.031 |
| C1m - C2-tail: ruinous | -0.220 | 0.031 |
| C1m - C2-tail: size_distance | +1.173 | 0.031 |
| C2-up - C2-tail: q2 | +1.689 | 0.156 |
| C2-up - C2-tail: ruinous | -0.067 | 0.156 |
| C2-up - C2-tail: size_distance | +0.192 | 0.469 |

## Every agent (final checkpoint, 40 test runs)

| Group | Seed | Agent | Q2 | Ruinous | Test deaths | Median size | Distance | 95th pct | Largest | In market | Long | Growth per run (95% CI) | Beats random A/B | Tilt | Train size first→last | Train ruin last 20% | Train deaths | Safeguard at bound | W1–W7,E | Success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1m | 1 (1048) | ag-bd6d29722b | -0.73 | 3.39% | 7 | 0.35x | 2.12 | 7.00x | 123.94x | 100% | 60% | -0.1915 (-0.6655, +0.2768) | n/n | 51.58 | 0.67x→1.91x | 3.87% | 172 | 0.000% | ✗✗✗✗✗✗✗✓ | no |
| C1m | 2 (1049) | ag-7367299748 | -1.82 | 8.22% | 5 | 0.52x | 1.74 | 14.50x | 202.13x | 100% | 72% | +0.4302 (-0.1122, +0.9425) | y/y | 4.73 | 0.92x→2.85x | 7.31% | 447 | 0.000% | ✗✗✗✗✗✗✗✓ | no |
| C1m | 3 (1050) | ag-784c291f84 | -2.17 | 9.16% | 8 | 1.14x | 0.95 | 14.92x | 124.14x | 100% | 83% | -0.0067 (-0.5394, +0.5174) | y/y | 1.59 | 0.85x→3.54x | 8.85% | 431 | 0.000% | ✗✗✗✗✗✗✗✓ | no |
| C1m | 4 (1051) | ag-3e932cb071 | -1.67 | 6.79% | 7 | 0.48x | 1.80 | 12.79x | 434.24x | 100% | 58% | -0.0956 (-0.5938, +0.3442) | n/y | 2.72 | 0.72x→3.02x | 6.69% | 413 | 0.000% | ✗✗✗✗✗✗✗✓ | no |
| C1m | 5 (1052) | ag-5b660bcc56 | -0.32 | 2.06% | 4 | 0.79x | 1.32 | 6.10x | 73.90x | 100% | 77% | -0.1407 (-0.5012, +0.1844) | n/n | 3.72 | 0.66x→1.73x | 2.29% | 113 | 0.000% | ✗✗✗✗✗✗✗✓ | no |
| C1m | 6 (1053) | ag-55f8ba1b4c | -1.25 | 3.71% | 5 | 0.69x | 1.45 | 8.10x | 105.95x | 100% | 80% | -0.2264 (-0.6429, +0.1588) | n/n | 1.12 | 0.58x→2.14x | 3.91% | 158 | 0.000% | ✗✗✗✗✗✓✗✗ | no |
| C2-tail | 1 (1048) | ag-b097949266 | -5.40 | 18.90% | 19 | 3.05x | 0.04 | 16.80x | 24.80x | 100% | 63% | -0.8383 (-1.5195, -0.1518) | n/n | 4.24 | 0.96x→5.31x | 20.16% | 956 | 0.000% | ✗✗✓✗✗✗✗✓ | no |
| C2-tail | 2 (1049) | ag-78bb1bf792 | -10.59 | 41.79% | 27 | 7.41x | 0.93 | 21.69x | 28.10x | 99% | 82% | -1.1158 (-1.9496, -0.2299) | y/y | 2.48 | 1.52x→8.86x | 40.24% | 1756 | 0.000% | ✗✗✗✗✗✗✗✓ | no |
| C2-tail | 3 (1050) | ag-8e94fe2ac9 | -4.77 | 17.92% | 12 | 2.38x | 0.21 | 16.09x | 21.64x | 100% | 69% | -0.1772 (-0.7928, +0.4214) | n/n | 1.04 | 1.07x→4.02x | 14.92% | 582 | 0.000% | ✗✗✓✗✗✓✗✓ | no |
| C2-tail | 4 (1051) | ag-674412c1fa | -7.16 | 27.43% | 20 | 2.51x | 0.15 | 20.73x | 27.81x | 96% | 77% | -0.5594 (-1.3181, +0.2204) | n/n | 7.71 | 1.12x→6.10x | 27.42% | 1326 | 0.000% | ✗✗✓✗✗✗✗✗ | no |
| C2-tail | 5 (1052) | ag-c3c151e5bc | -9.67 | 38.70% | 23 | 6.86x | 0.85 | 19.98x | 24.52x | 100% | 84% | -0.8851 (-1.6566, -0.0803) | y/n | 1.64 | 1.26x→7.70x | 35.19% | 1523 | 0.000% | ✗✗✗✗✗✗✗✓ | no |
| C2-tail | 6 (1053) | ag-8b5ef56d3a | -5.49 | 20.60% | 20 | 3.39x | 0.15 | 14.62x | 19.00x | 100% | 69% | -1.1850 (-1.7884, -0.5520) | n/n | 1.06 | 1.26x→4.60x | 16.63% | 677 | 0.000% | ✗✗✓✗✗✓✗✓ | no |
| C2-up | 1 (1048) | ag-8ffe5cf11b | -2.73 | 9.96% | 9 | 1.05x | 1.02 | 12.28x | 17.72x | 100% | 57% | -0.1799 (-0.7117, +0.3250) | n/n | 5.48 | 0.82x→2.50x | 7.68% | 263 | 0.000% | ✗✗✗✗✗✗✗✓ | no |
| C2-up | 2 (1049) | ag-95cf025f79 | -9.53 | 37.21% | 27 | 5.99x | 0.72 | 21.22x | 26.36x | 100% | 76% | -1.1790 (-1.9802, -0.3216) | n/y | 3.26 | 0.99x→7.60x | 35.47% | 1451 | 0.000% | ✗✗✗✗✗✗✗✗ | no |
| C2-up | 3 (1050) | ag-5d050157ac | -6.65 | 26.37% | 17 | 3.92x | 0.29 | 16.33x | 21.14x | 100% | 84% | -0.4084 (-1.1607, +0.3496) | y/y | 1.25 | 1.06x→5.34x | 23.66% | 830 | 0.000% | ✗✗✓✗✗✓✗✓ | no |
| C2-up | 4 (1051) | ag-e0b54f65df | -2.59 | 9.18% | 5 | 0.89x | 1.20 | 11.71x | 17.19x | 100% | 54% | +0.0156 (-0.4094, +0.4019) | n/y | 2.58 | 0.86x→2.34x | 7.25% | 285 | 0.000% | ✗✗✗✗✗✗✗✓ | no |
| C2-up | 5 (1052) | ag-45be0cdce4 | -6.22 | 22.06% | 17 | 3.74x | 0.24 | 14.05x | 17.73x | 100% | 79% | -0.7073 (-1.3651, -0.0329) | y/n | 2.64 | 0.81x→4.78x | 18.84% | 729 | 0.000% | ✗✗✓✗✗✗✗✓ | no |
| C2-up | 6 (1053) | ag-7b1403c797 | -5.23 | 20.12% | 20 | 2.90x | 0.01 | 13.79x | 18.72x | 100% | 75% | -1.1442 (-1.7708, -0.4927) | n/n | 1.37 | 1.18x→4.44x | 17.60% | 648 | 0.000% | ✗✗✓✗✗✓✗✓ | no |

## Exploration actually applied during training

From the exploration monitor (every training update): spread of ln(sampled size / mean size) as a multiplier per standard deviation; share of sampled sizes at 9.5x or more; median mean size. First / middle / last update, averaged over the group's seeds; maximum share of sampled size outputs clipped by the numerical range.

| Group | Size x per sd (first / mid / last) | Position x per sd (first / last) | Sampled ≥ 9.5x (first / last) | Median mean size (first / last) | Spread by mean size band, last update (<1x / 1–3x / 3–9.5x) | Max clipped share |
|---|---|---|---|---|---|---|
| C1m | x2.13 / x1.45 / x1.12 | x5.45 / x1.62 | 0.13% / 7.18% | 1.00x / 1.42x | x1.12 / x1.12 / x1.12 | 0.0000% |
| C2-tail | x18.38 / x7.67 / x5.51 | x26.46 / x6.04 | 1.21% / 43.75% | 0.69x / 8.13x | x27.51 / x8.75 / x3.32 | 0.2197% |
| C2-up | x10.26 / x4.61 / x2.85 | x16.05 / x3.16 | 0.23% / 29.72% | 0.69x / 5.98x | x7.32 / x3.40 / x1.75 | 0.0366% |

Designed starting exploration (pre-registered, by starting size): multiplier at +1 sd / -1 sd and chance of reaching 9.5x or more.

| Start | C1m | C2-tail | C2-up | Exp1 C1 | Exp1 C2 | control |
|---|---|---|---|---|---|---|
| 0.10x | x2.13 / ÷2.13 / 0.00% | x21.39 / ÷67.11 / 0.29% | x14.01 / ÷27.48 / 0.02% | x1.09 / ÷1.09 / 0.00% | x1.08 / ÷1.08 / 0.00% | x2.13 / ÷2.13 / 0.00% |
| 0.50x | x2.13 / ÷2.13 / 0.01% | x7.71 / ÷54.61 / 1.00% | x5.98 / ÷22.49 / 0.16% | x1.09 / ÷1.09 / 0.00% | x1.07 / ÷1.07 / 0.00% | x2.13 / ÷2.13 / 0.01% |
| 0.69x | x2.13 / ÷2.13 / 0.03% | x6.17 / ÷49.23 / 1.30% | x4.91 / ÷20.34 / 0.24% | x1.09 / ÷1.09 / 0.00% | x1.06 / ÷1.06 / 0.00% | x2.13 / ÷2.13 / 0.03% |
| 1.00x | x2.13 / ÷2.13 / 0.14% | x4.82 / ÷41.54 / 1.80% | x3.93 / ÷17.29 / 0.38% | x1.09 / ÷1.09 / 0.00% | x1.05 / ÷1.05 / 0.00% | x2.13 / ÷2.13 / 0.14% |
| 1.42x | x2.13 / ÷2.13 / 0.59% | x3.81 / ÷32.63 / 2.51% | x3.19 / ÷13.76 / 0.65% | x1.09 / ÷1.09 / 0.00% | x1.04 / ÷1.05 / 0.00% | x2.13 / ÷2.13 / 0.59% |
| 2.93x | x2.13 / ÷2.13 / 6.00% | x2.44 / ÷13.06 / 6.00% | x2.13 / ÷6.11 / 2.47% | x1.09 / ÷1.09 / 0.00% | x1.03 / ÷1.03 / 0.00% | x2.13 / ÷2.13 / 6.00% |
| 5.00x | x2.13 / ÷2.13 / 19.87% | x1.85 / ÷4.43 / 14.53% | x1.67 / ÷2.76 / 9.02% | x1.09 / ÷1.09 / 0.00% | x1.02 / ÷1.02 / 0.00% | x2.13 / ÷2.13 / 19.87% |
| 9.50x | x2.13 / ÷2.13 / 50.04% | x1.45 / ÷1.81 / 50.04% | x1.35 / ÷1.55 / 50.04% | x1.09 / ÷1.09 / 50.04% | x1.01 / ÷1.01 / 50.04% | x2.13 / ÷2.13 / 50.04% |
| 20.00x | x2.13 / ÷2.13 / 83.77% | x1.21 / ÷1.27 / 99.31% | x1.17 / ÷1.20 / 99.91% | x1.09 / ÷1.09 / 100.00% | x1.00 / ÷1.00 / 100.00% | x2.13 / ÷2.13 / 83.77% |

## Learning curves (development round; descriptive only)

Group mean of Q2 / ruinous share / median size by training steps.

| Group | 500k | 1000k | 1500k | 2000k | 2500k | 3000k |
|---|---|---|---|---|---|---|
| C1m | +0.20 / 0.1% / 0.38x | -0.19 / 1.5% / 0.55x | -0.85 / 4.1% / 0.61x | -1.08 / 5.0% / 0.63x | -1.17 / 5.4% / 0.64x | -1.24 / 5.6% / 0.65x |
| C2-tail | +0.29 / 0.0% / 0.50x | -1.21 / 2.6% / 1.63x | -3.87 / 14.2% / 2.56x | -5.53 / 21.4% / 3.51x | -6.54 / 25.7% / 4.18x | -6.99 / 27.8% / 4.37x |
| C2-up | +0.29 / 0.0% / 0.47x | -0.23 / 0.4% / 1.17x | -1.95 / 5.6% / 1.83x | -3.25 / 11.1% / 2.24x | -4.58 / 17.2% / 2.78x | -5.39 / 20.9% / 3.10x |

## Limits

- Six seeds per group resolve only large effects; intervals show the range consistent with the data.
- Experiment 1 references used different evaluation data; Q2 depends only on decisions, not on realised luck.
- Verdicts are the pre-registered rules applied mechanically; nothing was changed after training began.
- Every agent, version (including interrupted attempts), checkpoint and evaluation is preserved.

## Protection of earlier work

- original registry: content hash before / now: 1ba10554e4724f90… / 1ba10554e4724f90… (identical)
- original registry: seal verification: all sealed records intact
- Phase 3 Experiment 1 registry: content hash before / now: 90bcdabfa9d5e246… / 90bcdabfa9d5e246… (identical)
- Phase 3 Experiment 1 registry: seal verification: all sealed records intact
- Experiment 1b registry: seal verification: all sealed records intact
