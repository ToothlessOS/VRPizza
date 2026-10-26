# LEHD-style supervised POMO and Random Re-Construct

How to train POMO on LEHD's partial-solution objective and how to evaluate it with
inference-time Random Re-Construct (RRC). Written for anyone running or changing
`notebooks/POMO_sl_lehd.py`, `notebooks/eval_rrc.py`, `src/lehd/`, `src/rrc/` and the
`LEHD` environments in `src/envs/`.

The reference implementation is
`/home/toothlessos/Projects/nrp/CIAM-Group NCO_code main single_objective-LEHD/CVRP/`
(`VRPEnv.py:sampling_subpaths`, `VRPTester.py:_test_one_batch`). The paper is *Neural
Combinatorial Optimization with Heavy Decoder* (NeurIPS 2023).

## Why this exists

`POMO_sl` trains on whole HGS trajectories and underperforms POMO-RL. LEHD's supervised
scheme trains on **partial solutions**: a contiguous window of `w ~ Unif{4..N}` customers,
ending at a route end, cut out of an optimal tour. The prefix and suffix are discarded, and
the model reconstructs the window from its first customer and the remaining capacity. The
window can start mid-route and span several routes, so the model sees many local
reconstruction problems per tour instead of one full trajectory.

The experiment keeps POMO's encoder, decoder, multistart and dihedral inference, and replaces
only the supervised training distribution and adds RRC. That makes the table separate three
effects:

1. **Training** -- `lehd` multistart+8x vs `rl`/`sl` multistart+8x, identical inference.
2. **RRC** -- `lehd` RRC vs the same `lehd` multistart+8x incumbent.
3. **Combined** -- `lehd` RRC vs the baseline training methods.

## The recipe, and what is replicated exactly

| item | LEHD reference | here |
| --- | --- | --- |
| labels | precomputed HGS tours in the tagged file | same file, `vrp100_hgs_train_100w-001.txt` |
| window | contiguous circular window of `w` customers ending at a route end | `src.lehd.subpath.sample_window` |
| `w` | `Unif{4..N}`, **once per batch** | once per **effective** batch; microbatch/accumulation preserves it |
| start | first window customer, forced and unscored | same, `PartialResetMixin.reset_partial` |
| reduced problem | depot + `w` window customers, reindexed by ascending id | same |
| capacity at start | `C -` demand served earlier in the start's route | `used_capacity0 = prefix_demand + demand[start]`, normalised by `C` |
| loss | per-step cross-entropy | masked token CE, per-row mean (see below) |
| schedule | Adam 1e-4, batch 1024, 40 x 1M, MultiStepLR 0.9/epoch | same, one optimizer step per effective batch |
| RRC | flip, sample a route-end window, greedily reconstruct, splice, accept iff shorter | `src.rrc.runner.run_rrc` |
| RRC budget | 50/100/300/500/1000 iterations | nested: one run to the largest budget |

### The one structural deviation: POMO's N+1 actions

LEHD scores a `(customer, via_depot)` pair in one step. POMO's CVRP action space is
`0 = depot`, `1..N = customers`, so a route boundary is a depot action followed by the next
customer. The reconstruction target is therefore `w-1` customer actions plus one depot action
per interior route boundary. Nothing about the window, start, capacity or objective changes;
a route boundary simply costs an extra scored step. This is what lets the LEHD checkpoint go
through the existing POMO evaluator unchanged.

### The second deviation: per-batch optimizer updates

The reference steps the optimizer at every construction step, with the encoder recomputed at
each step. Here the masked CE is averaged over the partial solution and one step is taken per
effective batch. The objective is the same; the update count is much lower. If the first run
underperforms, this is the first thing to revisit, not the data.

The loss uses a **per-row** mean, not the global token mean. Rows have different numbers of
route boundaries, so a global token mean would weight a row with more depot actions more
heavily. LEHD's batches give every window the same `w-1` decisions; the per-row mean restores
that balance. `train/token_ce` logs the global mean for comparison with `POMOSL`.

### Other deviations, deliberately recorded

- **Reduced instance.** The model is trained on depot + the `w` window nodes, as in LEHD, not
  on the full instance with a mask. Validation still uses full instances and the existing
  `POMOInferenceMixin`, so `val/reward`, `val/reward_greedy` and `val/gap_ref` stay comparable
  with the `sl-` and `rl-` runs.
- **POMO backbone.** Six encoder layers and a pointer decoder, not LEHD's light encoder/heavy
  decoder. The capacity conditioning is already in rl4co's `VRPContext`; LEHD's remaining
  decoder conditions match it.
- **bf16-mixed** (the repo's setting) rather than the reference's full fp32.
- **No gradient clipping**, matching the reference; `RL4COTrainer` defaults to 1.0 and the
  training script passes `None`.
- **No training-time dihedral.** LEHD's flips/rotations are label-preserving order transforms;
  the 8x dihedral remains an inference option. A training-time random dihedral is a separate
  switch, off by default.
- **Small-`w` normalisation.** POMO normalises over the instance's nodes; at `w=4` that is five
  nodes. Watch the loss/gradient norms if the small-`w` batches prove unstable.

## Geometry and the data draw

All geometry lives in `src/lehd/subpath.py`, shared by training and RRC:

1. Apply LEHD's label-preserving flips: with p=0.5 reverse the whole solution (route order and
   every route), then reverse each route independently with p=0.5. Flags are recomputed from
   the transformed route list, which is equivalent to the reference's rolled-flag version.
2. Choose a route uniformly and rotate the route list so it is last. The window is the last
   `w` customers, so it ends at that route's end. This is distributionally equivalent to
   LEHD's shift-plus-doubled-buffer window and makes the representation linear.
3. Reindex the window by ascending original id; local id 1..w. Build the teacher actions: for
   each window position after the start, insert `0` if the position starts a route, then the
   local id. The final depot return is implicit.
4. The capacity prefix is the demand from the start's route start up to (not including) the
   start. If the start itself has flag 1, the prefix is empty. `used_capacity0` then adds the
   start's own demand, because the start is pre-served.

`LEHDPartialDataset` emits one partial per instance per epoch. The per-batch `w` is keyed by
`index // effective_batch_size`; per-instance flips/route-end draws are seeded by
`[seed, epoch, instance]`, so worker count and read order cannot change an item. The training
script passes the effective batch (1024) to the environment and uses a microbatch of 128 with
accumulation 8; `collate_partials` refuses a microbatch whose rows disagree on `w`.

`LEHDCVRPEnv.dataset` serves this dataset for training and falls through to the parent's
`CVRPSolutionDataset` for validation/testing.

## Validation metrics

`LEHDPOMOSL` inherits `POMOInferenceMixin._inference_step`, so the numbers mean exactly what
they mean for the existing `sl-`/`rl-` runs:

- `val/reward` is the **best of 8 x N** greedy rollouts: 8 dihedral augmentations, one
  forced start per customer, then the maximum over both axes. The base decode type is greedy
  (`val_decode_type="greedy"`), not sampling.
- `val/reward_greedy` is a separate, **unaugmented single** greedy rollout. It is a diagnostic,
  not the checkpoint criterion.
- `val/gap_ref` is computed from the augmented multistart reward against the reference cost.
- Training is **unaugmented**: the LEHD windows are order-flipped (whole solution and routes),
  not dihedral-transformed. The 8x applies only at validation and, as an eval flag, at
  inference.
- The checkpoint monitor is `val/reward`, i.e. the 8xN number, the same criterion every other
  run uses.

## Training

```bash
.venv/bin/python notebooks/POMO_sl_lehd.py            # the full run
.venv/bin/python notebooks/POMO_sl_lehd.py --smoke    # 4096 partials, one epoch, no checkpoint
```

Constants are at the top of the script: `999_000` partials/epoch, 40 epochs, LR 1e-4, weight
decay 0, `MultiStepLR` gamma 0.9 per epoch, effective batch 1024 (microbatch 128 x 8),
`val_data_size=10_000`, `num_augment=8`, bf16, checkpoint on `val/reward`.

The checkpoints are named `lehd-vrp100-...`. `discover_checkpoints` labels a run by the text
before its first `-`, so a second `lehd-` directory (a pilot, an ablation) would collide with
the main run. Either remove/move the pilot before evaluating, or use a distinct prefix
registered in `MODEL_CLASSES`. The smoke run writes no checkpoint for this reason.

Expected cost: a partial averages `w ~ 52` customers and ~50 decode steps, roughly half a full
trajectory row. 40M partials is about the compute of the existing 20M-row SL run (~18 h on the
card that measurement came from). I/O of ~1M random record reads per epoch is the main unknown;
run the smoke and a short pilot first.

## RRC evaluation

```bash
.venv/bin/python notebooks/eval_rrc.py --limit 8 --rrc-budgets 1 2 5          # smoke
.venv/bin/python notebooks/eval_rrc.py --datasets vrp100_test_lkh             # main run
.venv/bin/python notebooks/eval_rrc.py --num-augment 1 --num-starts 1         # no multipliers
.venv/bin/python notebooks/eval_rrc.py --rrc-init greedy                      # LEHD protocol
.venv/bin/python notebooks/eval_rrc.py --report-from logs/eval_rrc/results.npz
```

Each run records the plain incumbent as `{model}_ms{A}x{S}` (or `{model}_greedy`) and the
budget rows as the same label with `_rrc{budget}`. Both come from the same batch and the same
random draws, so the printed paired difference measures RRC, not a different initial solution.
The label includes the multipliers because they are inference settings, not weights:

- `--num-starts {N,1}` and `--num-augment {1,8}` control the matched plain baseline and the
  RRC incumbent. `--num-starts 1` is a single rollout with the start chosen by the model (rl4co
  disables its multistart path below 2); `N` is POMO's forced-start multistart.
- `--rrc-num-augment {1,8}` controls each reconstruction. Default 1 is faithful to LEHD; 8
  chooses the best of eight candidates per window.
- Multistart cannot be used *inside* a reconstruction: the window's start is fixed by the
  splice, because the prefix must remain connected. Multistart belongs to the initial
  complete solution, which is where POMO's inference multiplier lives.

The budget curve is nested: one run goes to `max(budgets)` and snapshots the incumbent at each
configured budget. The flips, route-end choices and `w` are seeded by
`(seed, iteration, instance position)`, so the same draws are used across budgets and
checkpoints. Acceptance is on the exact full-tour cost, not the partial problem's reward (a
mid-route start prepends a constant depot-to-start edge to that reward). Each accepted solution
is a permutation of the window and capacity-feasible, and the label-preserving transform is
asserted not to change the incumbent's cost.

Compute is reported as accepted repairs, decoder rows and wall clock. The initial POMO rollout
is not included in the RRC decoder rows; compare RRC at matched iterations and, separately, at
matched model evaluations.

## Verification

```bash
.venv/bin/python scripts/verify_lehd_data.py   # geometry, draw, workers, teacher forcing
.venv/bin/python scripts/verify_rrc.py         # rollout/replay, monotonicity, extremes, parity
.venv/bin/python scripts/verify_sl_data.py     # unchanged pipeline still passes
.venv/bin/python scripts/verify_synced_data.py
.venv/bin/python scripts/verify_val_parity.py
```

`verify_lehd_data.py` pins the window's route-end property, the action replay, the capacity
prefix, local-id round-trips, the one-`w`-per-effective-chunk draw (including across two
DataLoader workers), and the finite/teacher-forced loss. `verify_rrc.py` pins the multistart
action-to-reward replay, monotonicity and nested budgets, greedy and 8x modes, `w=4`/`w=N`, the
`num_starts` option, and the POMOSL/LEHDPOMOSL state-dict equality.

## Experiment matrix

1. Smoke, then a pilot (e.g. 10 epochs x 100k partials), then the full `40 x 999k` run.
2. Plain POMO on `vrp100_test_lkh` (10k), `vrp200/500/1000` (128 each), at `(N,8)`, `(N,1)`,
   `(1,8)`, for `lehd`, `rl`, `sl`; paired comparisons.
3. RRC for `lehd` at 50/100/300/500/1000 from the matched `(N,8)` and `(N,1)` incumbents;
   repeat for `rl` and `sl`.
4. Optional: greedy init + RRC (LEHD's protocol), 8x reconstruction, RRC from all starts.

## Files

| file | role |
| --- | --- |
| `src/lehd/subpath.py` | flips, window/prefix/capacity, replay, splice, cost |
| `src/envs/lehd_dataset.py` | lazy partial dataset and collate |
| `src/envs/cvrp_sl_lehd.py` | `LEHDCVRPEnv`, `LEHDEvalCVRPEnv`, `reset_partial` |
| `src/models/pomo_sl_lehd.py` | `LEHDPOMOSL` train branch and per-row loss |
| `src/rrc/config.py`, `src/rrc/runner.py` | RRC protocol and batched loop |
| `notebooks/POMO_sl_lehd.py` | training entry point |
| `notebooks/eval_rrc.py` | RRC evaluation, report and npz |
| `scripts/verify_lehd_data.py`, `scripts/verify_rrc.py` | PASS/FAIL suites |
| `scripts/submit_lehd.sh`, `scripts/submit_rrc.sh` | sbatch templates |
