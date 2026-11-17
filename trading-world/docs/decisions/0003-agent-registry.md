# 0003: Agent registry, continuous learning and immutable evaluation

Status: implemented 2026-11-17 (owner request "Build the Agent Registry, Continuous Learning
and Evolution System"). The simulator, trading mechanics, reward functions, scoring rules and
evaluation criteria are unchanged by this work.

## Principle

An agent's previous learning is never thrown away because it failed an evaluation. Failed agents
continue learning from their own checkpoints; each attempt is a new version; every result is kept.

## Components (code)

| Part | Where |
|---|---|
| Storage: content-addressed SHA-256 blobs, write-once JSON, append-only event logs | `src/tradeworld/registry/store.py` |
| Identity, lifecycle, immutable versions, write-once evaluations, frozen-rule guard | `src/tradeworld/registry/registry.py` |
| Checkpoints: weights, optimizer, learned exploration, hyperparameters, step, RNG (PyTorch, NumPy, Python, each environment, each account) | `src/tradeworld/learning/checkpoint.py` |
| Training history: every episode and every learning update; periodic checkpoints with an in-flight log | `src/tradeworld/learning/history.py` |
| Decision and consequence recording | `src/tradeworld/learning/recorder.py` |
| Immutable evaluation, criteria pinning, fresh rounds | `src/tradeworld/learning/evaluate.py` |
| Test-data ledger, sealed final data, real-data splits | `src/tradeworld/registry/datasets.py`, `configs/data/splits.yaml` |
| Failure diagnosis | `src/tradeworld/learning/diagnosis.py` |
| Continued training, crash recovery | `src/tradeworld/learning/continual.py` |
| Comparison | `src/tradeworld/learning/compare.py` |
| Command line | `scripts/agents.py` |
| Dashboard export | `scripts/build_dashboard_data.py` |

## Rules the registry enforces

1. **Permanent identity.** `identity.json` is written once. It fixes the practice market, scoring
   rule, environment settings, action space, simulator profile and episode length.
2. **Lifecycle** is derived from an append-only event log: created, training, awaiting
   evaluation, passed, failed, continued training, improved (passed after an earlier failure),
   retired. Passing on attempt 5 is recorded differently from passing on attempt 1
   (`passed_on_version`, `passed_on_evaluation`, `failed_evaluations`).
3. **Versions are write-once and numbered without gaps.** Each records its parent version, the
   exact parent checkpoint (checksum and step), steps added, training data (seed and data
   fingerprint), every setting change (old and new value), the code version, its checkpoints and
   its training history. An attempt interrupted by a container restart is recorded as its own
   `interrupted` version (with its checkpoints and history up to the last checkpoint);
   recovery continues from it as the next version.
4. **Frozen experimental rules.** A new version may change only: training steps, training data,
   learning rate, exploration noise (`log_std`), entropy coefficient, clip range, epochs, batch
   size, discount (gamma), GAE lambda, gradient clipping, value-loss weight, number of parallel
   environments, checkpoint interval, and which earlier checkpoint to resume from. Anything else
   (market, reward, environment, action space, simulator, criteria) raises `FrozenRuleError` and
   must be proposed separately for approval.
5. **Evaluations are write-once.** Each is linked to the exact version and checkpoint, the
   criteria version, the test data (seed and fingerprint), episode starts and reset seeds, the
   date and every episode's result. A second evaluation of the same version in the same round is
   refused. A failed evaluation stays failed.
6. **Criteria pinning.** The first evaluation pins the SHA-256 of `learning/gate.py` and
   `docs/decisions/0002-phase2-gate-criteria.md`. If either changes, evaluation refuses to run.
7. **Fresh data every round.** A round draws one never-used test seed per market from the
   ledger; every version evaluated in that round sees identical data (fair comparison). A seed
   used for anything (training, a round) is never used for anything else.
8. **Sealed final data.** Seeds 900,000+ (synthetic) and the real-data final holdout
   (2025-07-01 onward) are sealed: they cannot be used for training, validation or tuning, and
   every access needs a reason and an approver and is logged.

## Storage

- **Permanent tier (git, for now):** identities, versions, evaluations, rounds, ledger,
  every checkpoint, training history, diagnosis, per-episode results, and two sample episodes
  (worst and median) of each evaluation's decisions.
- **Regenerable cache (not tracked):** full decision logs (~2 MB per evaluation). They are
  reproduced exactly by replaying the evaluated checkpoint on the recorded test data, and a
  regenerated log is accepted only if it matches the fingerprint stored in the evaluation.
- Records refer to blobs only by checksum, so the git store can be replaced by dedicated
  artifact storage by implementing `BlobStore` and copying blobs; no record changes.

Growth estimate: a checkpoint is ~0.44 MB. With the default checkpoint every 250,000 steps, a
1M-step version adds ~2.2 MB and a 3M-step version ~5.8 MB. Moving to artifact storage is
recommended before large-scale continued training.

## Known limits

- Resuming restores weights, optimizer, exploration and every random stream exactly, but
  episodes that were in progress at the checkpoint restart (new episodes from the restored
  streams). Two resumes from one checkpoint are bit-identical; a resumed run is not
  bit-identical to an uninterrupted one.
- The original gate agents (imported as v1) have no RNG state and no training history: the
  first gate did not record them. Their decision logs were reconstructed by replaying their
  checkpoints on the original test data and attached only where the replay reproduced the
  stored results exactly.
- Diagnosis judges decision quality against the known expected return of synthetic markets.
  On real market data the expected return is unknown, so only behavioural and consequence
  measures apply.
