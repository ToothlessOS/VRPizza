# The step-matched comparison protocol

The four CVRP training objectives -- POMO REINFORCE on generated instances, teacher forcing on
HGS tours, LEHD partial reconstruction, and REINFORCE on LEHD partials -- are compared **per
optimizer step**, not per dataset. Every arm decodes the same number of sequences per update,
uses the same optimizer recipe, runs the same number of updates, and is scored on the same
validation and test sets. Each arm nevertheless draws from its natural data source, so RL keeps
its label-free, unlimited-sample advantage and SL/LEHD keep their labeled-data regime.

## The step budget

| Constant | Value |
| --- | --- |
| decoded sequences per optimizer step | **2,000** |
| instances per step | **125** |
| examples per instance | **16** |
| optimizer steps per epoch | **100** |
| epochs | **200** |
| total optimizer steps | **20,000** |
| decoded sequences per epoch | **200,000** |
| total decoded sequences | **40M** |

One step is 125 instances x 16 examples. The example type is method-specific: a POMO start
rollout, a teacher-forced full trajectory, a LEHD partial, or a sampled partial reconstruction.

| Arm | Data source | Per step | Kernel |
| --- | --- | --- | --- |
| RL full-tour | rl4co generator, fresh | 125 generated instances x 16 starts = 2,000 rollouts | `POMO_rl_step.py` |
| SL full-tour | HGS file | 125 labeled instances x 16 starts = 2,000 rows | `POMO_sl_step.py` |
| LEHD-SL | HGS file | 125 labeled instances x 16 partials = 2,000 rows | `POMO_sl_lehd_step.py` |
| LEHD-RL teacher | HGS file | 125 x 16 partials x K=1 = 2,000 rollouts | `POMO_rl_lehd_step.py --baseline teacher` |
| LEHD-RL shared | HGS file | 125 x 4 partials x K=4 = 2,000 rollouts | `POMO_rl_lehd_step.py --baseline shared` |

The shared arm's 2,000 rollouts are **500 independent partials x 4 correlated samples**, not
2,000 independent samples; each partial's advantages sum to zero under the shared baseline.
The natural shared arm (16 partials x K=4 = 8,000 rollouts) is a 4x-cost secondary experiment.

## The shared optimizer recipe

Adam, LR `1e-4`, weight decay `1e-6`, constant LR (no scheduler), gradient clip `1.0`,
`bf16-mixed`, no train-time augmentation. All arms use `RL4COTrainer` so
`reload_dataloaders_every_n_epochs=1` is in force; the dataset is rebuilt in
`on_train_epoch_end` with `seed + epoch`.

## Micro-batch vs effective step

The effective step is fixed at 2,000 decoded sequences. The micro-batch is a memory knob; use
the largest value that fits and gradient accumulation for the rest:

| Arm | Recommended micro x accumulation | Effective |
| --- | --- | --- |
| SL / LEHD-SL / LEHD-RL teacher | 500 x 4 (or 1,000 x 2) | 2,000 rows |
| RL full-tour | 125 instances x 1 (fallback 25 x 5) | 2,000 rollouts |
| LEHD-RL shared | 500 partials x 1 (or 250 x 2) | 2,000 rollouts |

2,000 rows in one full-tour forward is ~41 GiB at N=100 and is deliberately avoided. LEHD's
`w` is constant per optimizer-step chunk, so a chunk near `w=100` costs the same as a full-tour
row; do not size LEHD from its ~52 mean. `check_step_math()` in
`src/utils/comparison.py` asserts `rows_per_epoch == 100 * micro * accumulation` and rejects
non-divisors.

## Syncing one epoch

An epoch is 100 optimizer steps and 200,000 decoded sequences for every arm. The per-epoch
dataset sizes are:

| Arm | Rows/instances per epoch |
| --- | --- |
| RL | 12,500 generated instances |
| SL | 200,000 trajectories (12,500 x 16) |
| LEHD-SL / teacher | 200,000 partials (12,500 x 16) |
| LEHD-RL shared | 50,000 partials (12,500 x 4) |

Validation and checkpointing therefore land on the same optimizer step for every arm. The
epoch *content* remains method-specific; only the update count and cadence are synced.

## Validation and evaluation

The held-out range is the tail 1,000 instances of the HGS file. Training monitors a fixed
2,000-slot subset of the 21,044-slot tail pool (`VAL_DATA_SIZE=2000`, `VAL_INSTANCES=1000`);
the scientific number is the full LKH test evaluation via `notebooks/eval_pomo.py`.

A full 10,000-row validation is 10,000 x 8 augmentations x 100 starts = 8M rollouts, ~40x the
40M training sequences if run every epoch. The subset keeps the monitor affordable while
remaining identical across arms. Evaluate all five test files with `num_augment=8`,
`num_starts=N`, plus the `(N,1)` and `(1,8)` ablations. Use explicit `--checkpoint
LABEL=PATH` labels to avoid the duplicate-label discovery collision, and `--select best` to
pin the same checkpoint-selection rule for every arm.

## Seeding

Every entry point calls `seed_everything(SEED, workers=True)` before env/model construction and
passes `seed=SEED` to the env. `CVRPEnv` forwards the seed to rl4co's env base, which otherwise
draws a random seed and unseeds model init. Generated RL uses a dedicated `seed + epoch`
generator for both instances and start nodes, and resume re-anchors `env._epoch`; file-backed
draws use per-item NumPy RNGs and `shuffle=False`, so they are worker-count independent.

## What is deliberately not equalized

- **Token-steps / FLOPs.** A full trajectory is ~111 actions; a LEHD partial averages ~52. RL
  also amortizes the encoder across 16 starts. Equal decoded sequences is not equal compute.
- **Data passes.** RL's 125 instances per step are fresh; SL/LEHD's 125 are drawn from the
  labeled file. At 40M sequences, the 16-example SL/LEHD draw visits ~92% of the 999k split
  with ~2.5 visits per instance; RL sees 2.5M unique generated instances.
- **Distinct-instance composition.** The canonical one-partial-per-instance LEHD draw covers
  2,000 distinct instances per step; the headline arm uses 16 partials per instance to match
  the 125-instance geometry. The canonical draw is kept only as a labelled reference.

Those differences are the methods' characteristics. Report them alongside the step curves;
do not compare a 20,000-step arm's best checkpoint to a 78,000-step artifact's best without
saying so. Extend `MAX_EPOCHS` for all arms if the old operating point is the target.
