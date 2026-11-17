# Phase 2 gate, first attempt: protective copy

Copies of the Phase 2 gate agents (final PPO checkpoints and their evaluation results)
as produced by `scripts/phase2_synthetic.py`. `runs/` is not tracked by git and lives on
an ephemeral cloud machine, so these copies are what preserves the agents' learning.

Files are write-once: never edit or overwrite them. `SHA256SUMS` lists their checksums.
These will be imported as version 1 of each agent when the agent registry is built.
