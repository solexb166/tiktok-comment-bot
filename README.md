# Phase 3 Experiment 1b: artifact storage

Content-addressed blobs (checkpoints, optimiser and RNG state, training histories, decision
samples) for the agents of experiment `phase3-exp1b`. File names are SHA-256 checksums of their
contents; the records that reference them live on branch `claude/phase3-exp1b` under
`trading-world/registry-phase3b/`. Blobs are write-once and are never modified or deleted.
