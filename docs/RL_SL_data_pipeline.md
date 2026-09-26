# The SL and RL data pipelines, and how the synced pair aligns them

Notes on the configuration and the data draw of the four CVRP training scripts, written for
anyone reading or changing `notebooks/POMO_sl.py`, `notebooks/POMO_og.py`,
`notebooks/POMO_sl_synced.py` or `notebooks/POMO_rl_synced.py`, and the `src/envs/` modules
behind them. The measurements were taken against the installed rl4co (0.6.0, torch 2.14.0) at
revision `9979468`.

The RL-trained POMO outperformed the supervised one, but the two runs differed in several ways
at once, so the gap could not be attributed to data or to training. The synced pair exists to
remove the data-side differences and leave only the objective. This document records both the
configuration and the reasoning, since the numbers that matter here — how many starts an
instance contributes, how many instances an epoch touches — are not visible in either script.

## The configs

| | `POMO_sl.py` | `POMO_og.py` | `POMO_sl_synced.py` | `POMO_rl_synced.py` |
| --- | --- | --- | --- | --- |
| `DATA_PATH` | `vrp100_hgs_train_100w-001.txt` | same | same | same |
| `VAL_DATA_PATH` | `None` | `None` | `None` | `None` |
| `NUM_LOC` / `CAPACITY` | 100 / 50.0 | 100 / 50.0 | 100 / 50.0 | 100 / 50.0 |
| `LR` / `WEIGHT_DECAY` | 1e-4 / 1e-6 | 1e-4 / 1e-6 | 1e-4 / 1e-6 | 1e-4 / 1e-6 |
| `TRAIN_*` | `train_data_size=100_000` | `train_data_size=100_000` | `TRAIN_INSTANCES=6_250` | `TRAIN_INSTANCES=6_250` |
| starts per instance | ~1.05 (emergent) | 100 | `STARTS_PER_INSTANCE=16` | `STARTS_PER_INSTANCE=16` |
| instances per step | ~251 (emergent) | `BATCH_SIZE=64` | `INSTANCES_PER_BATCH=64` | `INSTANCES_PER_BATCH=64` |
| DataLoader batch | `BATCH_SIZE=256` rows | 64 instances | `BATCH_SIZE=1024` rows | 64 instances |
| `VAL_BATCH_SIZE` | 256 | 256 | 256 | 256 |
| `VAL_DATA_SIZE` | 10,000 | 10,000 | 10,000 | 10,000 |
| `MAX_EPOCHS` | 200 | 200 | 200 | 200 |
| `PRECISION` | `bf16-mixed` | `bf16-mixed` | `bf16-mixed` | `bf16-mixed` |
| `shuffle_train_dataloader` | `True` | omitted (`False`) | **`False`** | omitted (`False`) |
| `dataloader_num_workers` | 8 | omitted (0) | 8 | 8 |
| checkpoint monitor | `val/reward` | `val/reward` | `val/reward` | `val/reward` |
| `RUN_NAME` | `sl-vrp100-…-B-256` | `rl-vrp100-…-B-64` | `sl-synced-vrp100-…-B-64-K-16` | `rl-synced-vrp100-…-B-64-K-16` |

`TRAIN_*` is the one field whose *unit* changes between the columns, which is the single most
confusing thing about this table. In `POMO_sl.py` `train_data_size` counts **trajectories**; in
`POMO_og.py` it counts **instances**; in both synced scripts it counts **instances**, and the
dataset expands each to `STARTS_PER_INSTANCE` rows. The synced pair still lands on 100,000 rows
per epoch, so the epoch's cost is unchanged — only its composition is.

Everything below the draw — `VAL_*`, `MAX_EPOCHS`, `PRECISION`, the `bf16-mixed` rationale, the
checkpoint criterion — is deliberately identical across all four, because validation is shared
code (`POMOInferenceMixin`) and leaving it alone is what keeps `val/reward`,
`val/reward_greedy` and `val/gap_ref` plottable across every run.

## What actually differed

| | `POMO_sl.py` | `POMO_og.py` |
| --- | --- | --- |
| Training instances | `CVRPSolutionDataset` over the HGS file | rl4co `CVRPGenerator`, fresh each epoch |
| Rows/epoch | 100,000 trajectories | 100,000 instances |
| Distinct instances/epoch | ~95,300 (9.5% of the split) | 100,000, all fresh |
| Starts per instance/epoch | ~1.05 | 100 — every customer |
| Train-time augmentation | none | none |

The augmentation belief is worth correcting explicitly, since it is a natural thing to assume:
**both pipelines train unaugmented.** `dihedral8` is applied only at validation, on both sides,
through the shared `validation_augmentation`. `POMO.shared_step` sets `n_aug = 0` for
`phase == "train"`, and `POMOSL` never reaches for an augmentation at all.

The real confounds were therefore the instance *source* — generator samples against the HGS
file — and the start *set*. `n_start` comes from `env.get_num_starts(td)`, which for CVRP is
`action_mask.shape[-1] - 1 = N`; the nodes come from `arange(N) % num_loc + 1`, so RL began
from all 100 customers while only ~21 of those starts traverse a depot edge the reference
solution contains.

The synced pair fixes both: instances and start nodes both come from the file and its labels.

## The shared draw

`src/envs/synced_dataset.py` holds the single sampling implementation. It is the only place an
epoch is decided, and both datasets call it, so the two cannot drift apart.

`sample_epoch_slots(num_routes, instance_range, seed, instances, k)` draws `instances` instance
indices with `default_rng(seed + epoch)` — the same `seed + epoch` scheme
`CVRPEnv.dataset` already uses to refresh each epoch — and then, per instance, `k` distinct
slots in `[0, 2R)` from `default_rng([seed, instance])`. Seeding by instance rather than from a
running stream is what makes an item reproducible regardless of which DataLoader worker reads
it or in what order.

The **slot index is the shared unit of currency**. `divmod(slot, 2)` gives `(route_index,
direction)`, the same convention `CVRPSolutionDataset.__getitem__` already uses, and:

* the supervised dataset turns a slot into a full action sequence via
  `CVRPSolutionDataset.build_trajectory`;
* the RL dataset turns the same slot into a start node via `feasible_starts`.

`feasible_starts` returns the `2R` starts in **interleaved** order — index `2k` is route `k`'s
first customer (served forwards), `2k+1` its last (backwards) — so that a slot index alone
determines both the start node and the trajectory that follows from it. Concatenating the
firsts and then the lasts would look right and silently pair every route with the wrong start.

**The order of the returned instances is load-bearing.** It is a random permutation, and both
pipelines batch sequentially over it, so batch `i` covers the same instances on either side.
Sorting it, or shuffling rows afterwards, breaks the pairing without changing any shape.

## Why K = 16

`2R` varies per instance, and POMO's multi-start is a *scalar*:
`POMO.shared_step` computes `n_start = env.get_num_starts(td)`, then `batchify(td, n_start)` and
`unbatchify(reward, (n_aug, n_start))`. Every row in a batch must therefore offer the same
number of starts. Measured over the training file:

```
2R = 16 (0.01%) · 18 (2.57%) · 20 (46.1%) · 22 (48.2%) · 24 (3.1%) · 26 (0.01%)   mean 21.04
```

`K = 16 = 2·R_min` is the largest count every instance can supply, so nothing is dropped,
nothing is padded, and no instance is over-sampled. `sample_epoch_slots` raises rather than
quietly clamping if `k` exceeds the smallest `2R` in the range, because a per-instance start
count is not something the decoder can express.

**K cannot be reduced to 1**, which is why the supervised side had to move up rather than the RL
side moving down. Two independent reasons:

1. The shared baseline is `reward.mean(dim=1)` over the start axis
   (`rl4co/models/rl/reinforce/baselines.py:60`), and `POMO.shared_step` asserts `n_start > 1`
   during training. At one start the baseline *is* the reward and the advantage is identically
   zero.
2. `num_starts=1` does **not** force a first action.
   `DecodingStrategy.__init__` sets `multistart = num_starts > 1`
   (`rl4co/utils/decoding.py:249-250`), so it falls to the `else` branch, `self.num_starts`
   collapses to 0, and the rollout starts by sampling from the depot. Forcing step 0 is
   possible only through multistart or through teacher forcing — that is, only through SL.

## Batching, and why shuffling is off

One step is **64 instances × 16 starts = 1024 `(instance, start)` pairs on both sides**: the
supervised pipeline feeds 1024 rows straight through teacher forcing, the RL pipeline expands
64 instances by 16 through `batchify` into 1024 rollouts. Both run
`ceil(6250 / 64) = 98` steps per epoch, and the trailing partial batch is the same 42 instances
on both sides.

Matching the *grouping* and not merely the epoch totals is what makes the per-step data
identical, and it is why `POMO_sl_synced.py` passes `shuffle_train_dataloader=False` where
`POMO_sl.py` passes `True`. `CVRPSyncedSolutionDataset` indexes rows instance-major
(`position * K + slot`), so an instance's 16 rows are contiguous and sequential batching groups
them correctly; randomisation comes from the draw, which reorders instances every epoch. No
custom `BatchSampler` and no `train_dataloader` override are needed.

Row-level shuffling here would be a silent bug. Supervised cross-entropy is a plain mean over
rows, so its *value* is insensitive to whether an instance's 16 starts share a batch or are
scattered across 16 of them — the loss curve would look entirely normal. Only the effective
sample size, and so the gradient variance, would change. A step would cover ~1024 near-distinct
instances instead of 64, and the two pipelines would no longer be training on the same thing.

## Coverage, and what the synced runs give up

Each instance-visit now costs `K` rows rather than ~1, so at a fixed row budget the epoch
covers far less of the file. Over `E` epochs the expected visits per instance is
`E · TRAIN_INSTANCES / 999_000`, and the share of the split seen at least once is
`1 - exp(-visits)`:

| total rows | epochs | SL wall clock | instances seen | visits/instance |
| --- | --- | --- | --- | --- |
| 20M (100k × 200) | 200 | ~18 h | 71% | ~1.75 |
| 40M (100k × 400) | 400 | ~37 h | 92% | ~2.5 |
| 80M (100k × 800) | 800 | ~73 h | 99% | ~5.0 |

Against the old supervised run — which covered the entire split at ~20 visits each, as
simulated below — the synced supervised curve should be expected to read *worse*. That is the
breadth-for-depth trade, not a regression, and it applies to both synced runs equally, which is
what keeps their comparison meaningful. **Epoch count is the lever, not K**; `TRAIN_INSTANCES`,
`STARTS_PER_INSTANCE` and `MAX_EPOCHS` are top-level constants, so moving along this table is a
one-line change in each script (and must be made in both, or the draws stop matching).

The other cost is memory: the supervised step grows from 256 to 1024 rows, roughly 4× the
activation memory. Extrapolating the measurement in `POMO_sl.py:60-62` (2.65 GiB at batch 128)
that is ~21 GiB — fine on a 40 GB card, and an OOM on the 7.62 GiB card those numbers came
from. The same symmetry holds at 16 instances × 16 starts = 256 rows per step and 391 steps per
epoch if that bites; the epoch totals and the coverage table above are unaffected.

## The demand-normalisation contract

`POMO.shared_step` calls `self.env.reset(batch)` directly and never routes through
`InstanceInputMixin.make_input_td`, which is where the supervised path divides demand by
capacity. `CVRPSyncedDataset.collate_fn` therefore normalises **in the collate function**, and
emits `capacity` in file units with rl4co's `vehicle_capacity` left at 1.0 — the same contract
`CVRPGenerator._generate` produces with `demand / self.capacity`.

Getting this wrong scales the action mask by `capacity`, which does not raise: it lets routes
exceed capacity by a factor of 50. `scripts/verify_synced_data.py` pins it.

## What is still different between the synced pair

Only the objective, which is the point — but two consequences of *how* it differs are worth
knowing, because neither is a data difference and neither can be removed:

* **Decode mode.** Supervised training passes `actions` to the policy, which switches the decode
  type to `"evaluate"`: the reference action is replayed at every step regardless of what the
  model predicts. RL uses multistart sampling, forcing only step 0 and sampling the rest.
* **Step 0 is not scored in RL.** In multistart the forced first action's log-probability is
  pinned to `0`, so it contributes nothing to the loss; under teacher forcing step 0 is scored
  like any other. Inherent to POMO's design — the start is a given, not a decision.

Batch size is no longer on this list: RL's `num_starts` remains `None`, so
`POMO.shared_step` takes `K` from `env.get_num_starts` during training and `N` during
validation, where the batch is built by `make_input_td` and carries no start nodes for either
override to find.

## A finding: the old supervised draw did not do what it documented

`CVRPSolutionDataset` is designed so each instance contributes all `2R` of its starts. Its
docstring says *"an instance with `R = 11` contributes 22 addressable training examples instead
of one"*, and `POMO_sl.py:49-52` encodes the intended arithmetic: *"each instance contributes 2R
of them (about 21 for N=100), so divide by ~21 to get distinct instances."*

The implementation does not do that. `size` is applied as `rng.choice(pool_size, size,
replace=False)` over the 21,014,906-slot pool, and because an instance owns only ~21 of those
slots the draw is effectively independent of ownership. Drawing 100,000 slots:

```
distinct instances touched : 95,318
trajectories per touched instance: 1 (95.2%) · 2 (4.6%) · 3 (0.1%)
```

So an epoch gave ~95,300 instances at ~1.05 starts each, and the arithmetic that would justify
"divide by ~21" — 4,762 instances × 21 — would have required an instance-grouped sampler
covering 0.48% of the split per epoch instead of 9.5%. `CVRPSyncedSolutionDataset` is the first
implementation of the draw those comments always described.

Over a full run the difference narrows: simulating the real 200-epoch schedule gives 20.02 mean
visits per instance, 0 instances unvisited, and 12.92 of each instance's ~21 feasible starts
covered. So the old supervised run was not starved of data overall — but per epoch it saw each
instance from one start, never from its others.

## Running the pair

```bash
sbatch scripts/submit_sync.sh
```

`scripts/submit_sync.sh` requests two GPUs in one job and runs both trainings at once, pinned
with `CUDA_VISIBLE_DEVICES=0` and `1`. Since the two are meant to be read against each other,
running them concurrently keeps their epoch counts aligned rather than leaving one a day ahead,
and it halves the wall clock. Per-training logs go to `logs/sync/{sl,rl}-synced.log`, since the
job's own stdout would interleave them. The script waits on both and exits non-zero if either
fails, so one training's failure cannot be masked by the other succeeding.

It uses `.venv/bin/python` rather than `uv run`: two concurrent `uv run` invocations contend on
the environment lock, and the `uv sync` above has already built everything they need.

**Host memory is the first thing that breaks, and it does not look like a memory problem.**
rl4co forces the `spawn` start method (`torchrl` sets it globally on import, which is why both
entry scripts guard their work behind `if __name__ == "__main__"`), so every DataLoader worker
is a fresh interpreter that re-imports torch and rl4co before reading anything. Measured, that
floor is **~790 MB per worker** — about 80× the 10 MB dataset it then holds:

```
8 workers x 790 MB x 2 trainings   ~= 12.6 GB
2 main processes + CUDA host ctx   ~=  3-4  GB
                                      --------
                                      ~16-17 GB   vs  --mem-per-cpu=1G x -c 16 = 16 GB
```

That sits on the cgroup cliff, and the failure is badly mislabelled: a parent killed mid-spawn
leaves its child reporting `_pickle.UnpicklingError: pickle data was truncated`, with dozens of
leaked semaphores logged at shutdown. It reads like a pickling bug and is not one — both
datasets pickle to 10.85 MB and round-trip cleanly, and *which* training dies is timing-
dependent, so pinning it on the RL run is a red herring. Confirm with

```bash
sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS,ReqMem,Elapsed
```

where `OUT_OF_MEMORY`, or a `MaxRSS` sitting at `ReqMem`, means this. Hence `--mem-per-cpu=4G`
and `DATALOADER_WORKERS = 4` in the shipped scripts. The worker count is the cheaper lever
(8→4 saves ~6.3 GB across the pair) and is free in data terms, since the draw is seeded from the
dataset rather than the worker count. The many-worker cost is worth remembering generally: with
`spawn`, worker count multiplies a ~790 MB import, not a 10 MB payload.

**GPU memory** is the second thing. The supervised run batches 1024 rows per step, ~4× what
`POMO_sl.py` was tuned around. On an A40's 48 GiB that is comfortable, but a CUDA OOM here means
lowering `INSTANCES_PER_BATCH` from 64 to 16 in *both* entry scripts — which keeps the
supervised batch at its original 256 rows and shrinks only the RL one — and raising `MAX_EPOCHS`
to compensate for the smaller per-epoch draw.

## Verifying it

```bash
.venv/bin/python scripts/verify_synced_data.py   # 46 checks, most importantly draw equality
.venv/bin/python scripts/verify_sl_data.py       # unchanged; the old pipeline is untouched
.venv/bin/python scripts/verify_val_parity.py    # unchanged; the other SL/RL pair still agrees
```

`scripts/verify_synced_data.py` checks, in order: that the two entry scripts agree on every
constant that shapes the draw; that the two datasets select the same instances and slots and
emit the same start node for every `(instance, start)` pair, across several epochs; that every
emitted start is a reference route endpoint and never the depot; that `select_start_nodes`
returns slot-major order and the forced first action of each rollout really is its instance's
start — the check that catches a `k*B + b` versus `b*K + k` mistake, which would otherwise pair
every instance with another instance's starts while every shape stayed correct; that demand is
normalised as the environment expects; and that supervision still reproduces the reference cost
to the file's ~6e-5 relative tolerance while validation stays exact between the two synced
models.

Two of those checks are worth calling out for how they are built, because the obvious version
of each does not work. The batching check compares batch **contents** — the coordinate blocks
in a supervised step against the instances in a reinforcement-learning one — rather than
mapping row positions back through the dataset. A shuffled batch holds a permutation of the
rows, so the positional version describes the order the batches were *meant* to be in and lets
a shuffling bug pass; reading coordinates catches it. And the step size used there is
deliberately smaller than the draw, since with a single batch every batch trivially contains
every instance and the check cannot fail at all. Both were confirmed to fail when
`shuffle_train_dataloader` is turned back on, which is the only thing that makes them worth
having.

`pytest` is not installed, so these scripts are the test suite.
