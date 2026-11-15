# ADR 0001: Phase 1 defaults (approved 2026-11-15)

Approved with the architecture (r4), using the recommended options:

1. **Action space X1:** `[d_1..d_n, u]` in [-1, 1]; `G = 10^(4u)` (1e-4x .. 1e4x, u = 0 -> 1x);
   `w_i = G d_i / max(1, sum|d|)`. No behavioural cap. Benchmark caps rescale the log range.
2. **Environment A margin:** A-reference schedule (1:2000 leverage-based for FX and gold;
   fixed 0.25% for BTC and NAS100, placeholders). Unlimited-margin is a labelled variant
   (`configs/environments/A_unlimited_margin.yaml`).
3. **Insufficient margin:** atomic rejection of all increases in a decision; reductions always execute.
