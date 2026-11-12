# SL/RL data-pipeline notes

The data-synced SL/RL pair was retired on the `sync-training` branch in favour of the
step-matched protocol in [`ComparisonProtocol.md`](ComparisonProtocol.md). This note keeps the
measurements that are still load-bearing for the remaining file-backed datasets. The old pair
is preserved in git history (commit `70a663e`).

## The slot draw is not the documented draw

`CVRPSolutionDataset` exposes every `(instance, route, direction)` trajectory as a flat pool of
`sum(2R)` slots and samples `size` of them uniformly. Because an instance owns only ~21 of the
21M training slots, a 100,000-slot epoch touches ~95,300 distinct instances at ~1.05 rows each
(max 4), not the documented ~4,762 instances x 21 starts. An instance's other optimal starts
are never seen alongside each other.

The step-matched SL arm therefore uses `SyncedCVRPEnv` + `CVRPSyncedSolutionDataset`, which
select instances and expand each into `k` of its `2R` starts. `k=16` is the largest uniform
count: the training range's minimum `2R` is exactly 16 (measured distribution: 16 at 0.01%,
18 at 2.57%, 20 at 46.1%, 22 at 48.2%, 24 at 3.1%, 26 at 0.01%; mean 21.04). POMO's shared
baseline needs `k > 1`, so the supervised side comes up to `k=16` rather than RL dropping to 1.

Rows are instance-major, so the DataLoader must not shuffle; randomisation comes from the
per-epoch instance permutation (`seed + epoch`). Row-level shuffling would not change the
cross-entropy value -- it is a mean over rows -- so it would only show up as a silent change in
effective sample size.

## The demand-normalisation contract

RL training calls `POMO.shared_step`, which passes the batch straight to `env.reset` and never
goes through `InstanceInputMixin.make_input_td`. The generated dataset therefore normalises
demand itself: `CVRPGenerator._generate` divides by capacity, and the generated step arm must
not divide again. For file-backed batches, `make_input_td` divides demand by the per-record
capacity for the supervised/validation path; `CVRPSyncedSolutionDataset` reuses the verified
`CVRPSolutionDataset.collate_fn`, which emits raw demand and the file-unit capacity. Do not
double-normalise either path; the action mask assumes `vehicle_capacity=1.0`.

## Worker memory

rl4co forces the `spawn` start method, so every DataLoader worker is a fresh interpreter that
re-imports torch and rl4co before touching data. Measured floor: ~790 MB per worker, against a
~10 MB dataset. The step-matched scripts use four workers; two concurrent trainings at eight
workers each exceed a 16 GB cgroup, and the kernel kill surfaces as a truncated pickle rather
than an OOM message. Raise `--mem-per-cpu` or lower the worker count first; the worker count
does not change the draw because the file-backed datasets seed each item independently.

## Current pieces

- `notebooks/POMO_sl_step.py`, `notebooks/POMO_rl_step.py`,
  `notebooks/POMO_sl_lehd_step.py`, `notebooks/POMO_rl_lehd_step.py` -- the step-matched arms.
- `src/utils/comparison.py` -- the shared constants and `check_step_math`.
- `src/envs/synced_dataset.py` -- the supervised 16-start draw (`sample_epoch_slots`,
  `CVRPSyncedSolutionDataset`).
- `src/envs/cvrp_sl_synced.py` -- `SyncedCVRPEnv`.
- `src/envs/cvrp_rl_step.py` -- generated RL with 16 starts.
- `src/envs/lehd_step_dataset.py`, `src/envs/cvrp_lehd_step.py` -- `k` partials per instance.
- `scripts/verify_comparison.py` -- the protocol verifier.

The retired pieces are `CVRPSyncedDataset`, `SyncedRLCVRPEnv`,
`notebooks/POMO_sl_synced.py`, `notebooks/POMO_rl_synced.py`, `scripts/submit_sync.sh`, and
`scripts/verify_synced_data.py`.
