# Phase 3 Experiment 1 registry

Separate registry for experiment `phase3-exp1` (proposal 0006). Its agents, versions, checkpoints, evaluations, rounds and seals are independent of the original registry (`registry/`), which this experiment never writes to. `metrics/Q2.0.json` is a byte-identical copy of the frozen Q2.0 record. `data/inherits.json` makes the original data ledger count as used, so all data here is fresh.

Blobs: branch `claude/phase3-exp1-artifacts`, checked out at `../../../phase3-exp1-artifacts/blobs` relative to this directory (or set `TRADEWORLD_PHASE3_BLOBS`).
