# The step-matched comparison protocol

The five CVRP training objectives -- POMO REINFORCE on generated instances, teacher forcing on
HGS tours, LEHD partial reconstruction, REINFORCE on LEHD partials, and local-reconstruction
self-improvement (SIL-RC) -- are compared **per optimizer step**, not per dataset. Every arm
decodes the same number of sequences per update, uses the same optimizer recipe, runs the same
number of updates, and is scored on the same validation and test sets. Each arm nevertheless
draws from its natural data source, so RL keeps its label-free, unlimited-sample advantage and
SL/LEHD keep their labeled-data regime. SIL-RC is the second label-free arm; it spends its first
50 epochs on the paper's RL warm-up and its remaining 150 on teacher forcing against labels it
improves itself.

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
| SIL-RC | generator, 12,500 fixed per cycle | 125 instances x 16 partials = 2,000 rows | `POMO_sil_rc_step.py` |
| LEHD-backbone SIL-RC | generator, 12,500 fixed per cycle | 125 instances x 16 partials = 2,000 rows | `LEHD_sil_rc_step.py` |
| SIL-RC chunked | generator, 12,500 per outer cycle split into S chunks | 125 instances x 16 partials = 2,000 rows | `POMO_sil_rc_step.py --subcycles S` |
| LEHD-backbone SIL-RC chunked | same | 125 instances x 16 partials = 2,000 rows | `LEHD_sil_rc_step.py --subcycles S` |

The shared arm's 2,000 rollouts are **500 independent partials x 4 correlated samples**, not
2,000 independent samples; each partial's advantages sum to zero under the shared baseline.
The natural shared arm (16 partials x K=4 = 8,000 rollouts) is a 4x-cost secondary experiment.

SIL-RC has a two-phase budget: the first 50 epochs (5,000 steps) are POMO REINFORCE on
generated CVRP20 with 125 instances x 16 starts per step, and the remaining 150 epochs
(15,000 steps) are the supervised phase. The supervised phase is five 30-epoch label cycles;
each cycle regenerates 12,500 CVRP100 instances, improves their labels with 100 RRC rounds per
instance, then trains on 12,500 x 16 = 200,000 partial windows per epoch. Label repair is extra
compute outside the decoded-sequence count, comparable to the HGS solver's label acquisition for
the SL arms.

`--subcycles S` splits each 30-epoch cycle into S chunk sub-cycles. Each chunk is repaired with
the best policy and then trained for 30 chunk-epochs, so the total repair work, rows and optimizer
steps are unchanged (15,000 SIL steps); the steps per chunk-epoch fall to 100/S and the Lightning
epoch count rises to 150S. The chunk order is reshuffled per outer cycle. Validation remains per
Lightning epoch, so the validation count per outer cycle rises by S; S must divide the 12,500
labels and keep 100/S steps per chunk-epoch.

LEHD-backbone SIL-RC is the same budget with the reference LEHD light encoder / heavy decoder.
The effective step stays 2,000; the micro-batch falls to 125 x 16 (the six-layer decoder
re-encodes at every step, so the split, not the effective size, is the memory knob), the repair
batch is 125, and validation defaults to one start (the LEHD paper's greedy protocol) instead of
`N`. The teacher-forced loss is the reference's exact 2N CE; the rollout/RRC path uses the
repo's N+1 action space. See [LEHD_SIL_RC.md](LEHD_SIL_RC.md).

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
| SIL-RC SIL phase | 500 partials x 4 | 2,000 partials |
| SIL-RC warm-up | 125 instances x 1 | 2,000 rollouts |
| LEHD-backbone SIL-RC SIL phase | 125 partials x 16 | 2,000 partials |
| LEHD-backbone SIL-RC warm-up | 125 instances x 1 | 2,000 rollouts |

2,000 rows in one full-tour forward is ~41 GiB at N=100 and is deliberately avoided. LEHD's
`w` is constant per optimizer-step chunk, so a chunk near `w=100` costs the same as a full-tour
row; do not size LEHD from its ~52 mean. `check_step_math()` in
`src/utils/comparison.py` asserts `rows_per_epoch == steps_per_epoch * micro * accumulation`;
the whole-store arms pass 100, a chunked SIL sub-cycle passes 100/S, and non-divisors are
rejected.

## Syncing one epoch

An epoch is 100 optimizer steps and 200,000 decoded sequences for every arm. The per-epoch
dataset sizes are:

| Arm | Rows/instances per epoch |
| --- | --- |
| RL | 12,500 generated instances |
| SL | 200,000 trajectories (12,500 x 16) |
| LEHD-SL / teacher | 200,000 partials (12,500 x 16) |
| LEHD-RL shared | 50,000 partials (12,500 x 4) |
| SIL-RC SIL phase | 200,000 partials (12,500 x 16); labels regenerated per 30-epoch cycle |
| SIL-RC warm-up | 12,500 generated CVRP20 instances (125 x 16 starts) |
| LEHD-backbone SIL-RC SIL phase | 200,000 partials (12,500 x 16); labels regenerated per 30-epoch cycle |
| LEHD-backbone SIL-RC warm-up | 12,500 generated CVRP20 instances (125 x 16 starts) |

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

The LEHD-backbone arm defaults to `num_starts=1` (8 augmentations x 1 start) because POMO's
`N`-start rollout is `N` times the sequences and the heavy decoder re-encodes at every step.
`val/reward_greedy` is the same unaugmented greedy rollout in both arms; the paper-comparable
number is the greedy + RRC evaluation. `--num-starts N` with a small batch restores the POMO
protocol if needed.

## Seeding

Every entry point calls `seed_everything(SEED, workers=True)` before env/model construction and
passes `seed=SEED` to the env. `CVRPEnv` forwards the seed to rl4co's env base, which otherwise
draws a random seed and unseeds model init. Generated RL uses a dedicated `seed + epoch`
generator for both instances and start nodes, and resume re-anchors `env._epoch`; file-backed
draws use per-item NumPy RNGs and `shuffle=False`, so they are worker-count independent.

SIL-RC keys its generated instance sets by `[LABEL_SEED, block]` and
`LABEL_SEED + outer`, its chunk order by `[cycle_seed(outer), outer]`, its repair windows by
`[seed, iteration, global_label_index]` with `seed = LABEL_SEED + repair_step`, and its training
windows by `[seed + epoch, label_index, window_index]`. The label sidecar is loaded on resume
rather than redrawn; it records the backbone, policy signature and chunk schedule, so a
POMO-initialised or differently scheduled sidecar is refused (legacy sidecars count as POMO / S=1).
The full contract is in [SIL_RC.md](SIL_RC.md#seeding-and-reproducibility) and
[LEHD_SIL_RC.md](LEHD_SIL_RC.md#seeding-and-reproducibility).

## What is deliberately not equalized

- **Token-steps / FLOPs.** A full trajectory is ~111 actions; a LEHD partial averages ~52. RL
  also amortizes the encoder across 16 starts. Equal decoded sequences is not equal compute.
- **Data passes.** RL's 125 instances per step are fresh; SL/LEHD's 125 are drawn from the
  labeled file. At 40M sequences, the 16-example SL/LEHD draw visits ~92% of the 999k split
  with ~2.5 visits per instance; RL sees 2.5M unique generated instances.
- **Distinct-instance composition.** The canonical one-partial-per-instance LEHD draw covers
  2,000 distinct instances per step; the headline arm uses 16 partials per instance to match
  the 125-instance geometry. The canonical draw is kept only as a labelled reference.
- **SIL-RC method overhead.** Five label cycles x 12,500 instances x 100 RRC rounds, plus the
  5,000-step CVRP20 warm-up. Its training steps decode the same 2,000 partials as the LEHD
  arms, but the pseudo-label generation is compute the other arms do not pay per cycle.
- **Chunked SIL-RC cadence.** `--subcycles S` keeps the total repairs, rows and optimizer
  steps, but validates S times more often per outer cycle and rewrites the label sidecar after
  each chunk. It refreshes one chunk's labels just before training it instead of repairing the
  whole store at the cycle boundary.
- **LEHD-backbone SIL-RC compute.** The heavy decoder re-runs six self-attention layers at
  every decode step, so the same 2,000 sequences cost substantially more wall-clock than the
  POMO arm, and its validation monitors one start rather than `N`. The decoded-sequence and
  optimizer-step counts are still the comparable unit; FLOPs and validation rollouts are not.

Those differences are the methods' characteristics. Report them alongside the step curves;
do not compare a 20,000-step arm's best checkpoint to a 78,000-step artifact's best without
saying so. Extend `MAX_EPOCHS` for all arms if the old operating point is the target.
