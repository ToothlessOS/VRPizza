# LEHD-style RL: REINFORCE on POMO's partial solutions

How to train POMO on LEHD's partial-solution distribution with a policy-gradient objective, and
why the reward is what it is. Written for anyone running or changing
`notebooks/POMO_rl_lehd.py`, `src/envs/cvrp_rl_lehd.py` and `src/models/pomo_rl_lehd.py`.

The reference for the partial geometry is the LEHD supervised path in this repo
(`docs/LEHD_SL.md`, `src/lehd/subpath.py`, `src/envs/lehd_dataset.py`). This experiment keeps
that data distribution and the POMO backbone, and replaces teacher forcing with REINFORCE.

## Why this exists

LEHD's supervised scheme trains on contiguous windows of `w ~ Unif{4..N}` customers cut from
HGS tours, ending at a route end. The POMO model trained that way is the best in the repo once
inference-time Random Re-Construct (RRC) is applied. POMO's own RL recipe — REINFORCE on full,
freshly generated CVRP100 instances with the shared baseline over starts — is the best method
without RRC.

The natural experiment is to keep REINFORCE but train on the LEHD partial distribution: the
same windows, starts and residual capacities, with the policy choosing the reconstruction
instead of replaying the HGS order. The question this document answers is how to score those
reconstructions so that windows of different lengths train one policy sensibly.

## The partial problem and its exact reward

A LEHD item is a reduced problem:

* depot (the original depot),
* `w` window customers, sorted by original id and reindexed `1..w`,
* the forced start `s` (the first window customer in solution order), already served,
* residual capacity `q = C - prefix_demand`, where `prefix_demand` is the demand served
  earlier in the start's route (zero if the start begins a route).

`sample_window` rotates the route list so the chosen route is last, and takes the last `w`
customers. The window therefore ends at a route end and is followed by the depot: the suffix
after the window is empty. Define the **reduced closed-tour cost** of a reconstruction `τ` as

```
C_red(τ) = d(depot, s)
         + the edges of τ (depot returns at interior route boundaries included)
         + d(last(τ), depot)
```

Because the window is a suffix of the tour and the prefix is fixed, splicing `τ` back into the
tour is feasible for every `τ` the environment can emit, and

```
C_full(prefix, τ) = C_red(τ) + K(prefix, s)
K(prefix, s)     = F + d(prefix_last, s) - d(depot, s)
```

where `F` collects the costs of the fixed prefix and `prefix_last` is the customer immediately
before the start (the depot when the prefix is empty). `K` depends only on the window, the
start and the prefix — **not on `τ`**. Minimising `C_red` is therefore exactly minimising the
repaired full-tour cost, for every window, with no reward shaping, discount, or delta reward.
The `depot -> s` edge is artificial (the true edge is `prefix_last -> s`), but it is part of the
constant `K`, so it cannot change the ranking of two reconstructions of the same window.

### The fixed-start invariant

`K` is constant only because the start is fixed. If the reduced instance were run with POMO's
`num_starts = w` multistart — every window customer as a start — then:

* only the sampled start has a defined prefix and residual capacity; the other starts have
  neither, so the state is not the LEHD partial anymore, and
* the `depot -> s_k` edge differs per start, and the shared baseline over starts does not
  remove it, so the advantage carries an action-independent but start-dependent bias.

RRC also always reconstructs from the spliced start, because the prefix must stay connected.
Training one rollout from the same pre-served start matches the geometry, the inference
distribution and the reward algebra. That is the training unit.

### The scale problem across `w`

Tour length for uniform points grows roughly as `sqrt(w)`. Over `w ∈ [4, 100]` the reward
magnitude and its within-window spread vary by roughly 5–10×. A raw advantage would give a
window-size-dependent effective step size at a fixed learning rate, and the score-function sum
has `Θ(w)` terms. This is a normalisation problem, not a reward-definition problem: the reward
above is already exact.

## The two baseline arms

Both arms compute `R = -C_red` from the trajectory the policy actually sampled. They differ in
the baseline subtracted from it:

| | `teacher` arm | `shared` arm |
| --- | --- | --- |
| baseline `b` | `-C_teacher` (HGS window's reduced cost) | `-mean_{j=1..K} C_j` (the policy's own K samples) |
| advantage `A` | `(C_teacher - C_red) / C_teacher` | `(mean_j C_j - C_red) / mean_j C_j` |
| measures | regret against the expert's reconstruction | relative quality among K reconstructions of the same window |
| labels / ground truth in the signal | yes, one detached scalar | **none** |
| rollouts | one (`K = 1`); K > 1 also valid | `K >= 2`; `K = 1` gives zero advantage |
| unbiased | exactly: `b` does not depend on the sampled action | naive mean scales the gradient by `(K-1)/K`; direction is exact |
| cost | ~`w` decoder steps per partial | `K × w` |

Both are true REINFORCE. The policy is reinforced from the cost of its own trajectory. The
teacher arm uses the label only as a control variate: if the policy's reconstruction is shorter
than the teacher's, `A > 0` and it is reinforced, so the teacher can be beaten; the teacher path
is never replayed or scored. The shared arm does not read the teacher actions or the reference
cost at all.

The loss is the same for both arms once `A` is formed:

```
loss = mean[ -(A * sum_logp) / w ]
```

Dividing by `w` mirrors `LEHDPOMOSL`'s per-row mean over ~`w` decisions and keeps the gradient
magnitude stable across effective batches with different `w`.

Sign check: a sampled reconstruction worse than the baseline has `C_red > baseline`, hence
`A < 0`, and its action log-probabilities are pushed down; better than the baseline gives
`A > 0` and is reinforced.

Post-decode steps are forced depot actions: with the depot already visited, the mask allows only
the depot once every customer is served, so the log-probability is `0` and the edge cost is
`0`. Variable trajectory length therefore does not bias the estimator.

## Data use, and the label boundary

The LEHD item is derived from an HGS tour at data-construction time: the tour and its flags
define the window, the start and the residual capacity. Both arms inherit that geometry.
They differ in what the objective reads:

| batch field | teacher arm | shared arm | role |
| --- | --- | --- | --- |
| `depot`, `locs`, `demand`, `capacity` | yes | yes | reduced problem geometry |
| `start_node` | yes | yes | pre-served start |
| `used_capacity0` | yes | yes | residual capacity |
| `num_loc` (= `w`) | yes | yes | horizon normaliser |
| `actions` (teacher order) | yes — one scalar `C_teacher` | **no** | never a target |
| `valid` | no | no | only the SL loss uses it |
| `cost` (full reference) | no | no | validation / evaluation only |

The shared arm is therefore label- and ground-truth-free *in its training signal*: its advantage
compares only the policy's own samples. The data distribution is still LEHD's — the windows are
cut from HGS tours — because that is the experiment. Generating the windows from the policy's
own solutions is the separate self-improvement-learning experiment.

`scripts/verify_lehd_rl.py` enforces the boundary: replacing `actions` with a random permutation
and `cost` with random values leaves the shared arm's loss unchanged, while the teacher arm's
loss changes.

## What is shared with LEHD SL, and what is new

Shared, unchanged:

* the tagged-file reader, `sample_window`, the route-end window, ascending-id reindexing, the
  local `0`/`1..w` action space, `used_capacity0`;
* the one-`w`-per-effective-batch draw and per-instance seeding in `LEHDPartialDataset`;
* `PartialResetMixin.reset_partial`, the post-start state used by both SL and RRC;
* the POMO encoder/decoder, so checkpoints remain interchangeable with `POMOSL`,
  `LEHDPOMOSL` and `POMORL`;
* validation (`POMOInferenceMixin`) and RRC (`src/rrc`).

New:

* `LEHDRLCVRPEnv.reset_partial` keeps `start_node` on the state, and `_get_reward` prepends it
  to the action sequence, so the reward is `-C_red` rather than the base env's
  `depot -> first-remaining -> ... -> depot` (which omits the start edge and is not the window
  cost);
* `reduced_window_cost`, shared by the reward and the teacher baseline, computes
  `depot -> s -> ... -> last -> depot` from local ids;
* `LEHDPOMORL`'s training branch: one sampled rollout (`teacher`) or K sampled rollouts
  (`shared`), the two baselines, relative advantage, horizon normalisation.

## Training

```bash
.venv/bin/python notebooks/POMO_rl_lehd.py            # teacher arm, K=1
# edit ADVANTAGE_BASELINE = "shared" and re-run for the label-free arm (K=4)
.venv/bin/python notebooks/POMO_rl_lehd.py --smoke    # 4096 partials, one epoch, no checkpoint
```

Constants at the top of the script: `999_000` partials per epoch, 40 epochs, Adam 1e-4, weight
decay 0, `MultiStepLR` gamma 0.9 per epoch, effective batch 1024 (microbatch 128 × 8), bf16,
gradient clipping 1.0, checkpoint on `val/reward`. `ADVANTAGE_BASELINE` selects the arm;
`NUM_SAMPLES` is 1 for `teacher` and 4 for `shared`.

Compute: one sampled partial costs ~`w` decoder steps, the same order as the SL teacher-forced
pass, so the teacher arm's 40M partials are compute-comparable to the LEHD SL run (~18 h on the
reference card). The shared arm costs `K×`; use `K = 2` for a pilot if `K = 4` is too slow.

Run names start with `lehd_rl-` (teacher) or `lehd_rl_shared-` (shared). The prefix before the
first `-` is the checkpoint label `discover_checkpoints` uses; both labels are registered in
`notebooks/eval_pomo.py`'s `MODEL_CLASSES`.

## Validation and RRC

`LEHDPOMORL` inherits `POMOInferenceMixin`, so `val/reward`, `val/reward_greedy` and
`val/gap_ref` have exactly the same meaning as in the `rl-`, `sl-` and `lehd-` runs: best of
8 dihedral × `N` starts for `val/reward`; an unaugmented single greedy rollout for
`val/reward_greedy`; the reference gap on the headline rollout. Training is unaugmented, as in
LEHD SL.

RRC is unchanged: reconstructions always start from the spliced start, run one greedy rollout,
and accept only an exact full-tour improvement. Evaluate both arms with

```bash
.venv/bin/python notebooks/eval_rrc.py --checkpoints lehd_rl lehd_rl_shared \
    --datasets vrp100_test_lkh --rrc-budgets 50 100 300 500
```

The new reward does not change RRC: RRC never compares partial rewards (it recomputes exact
full-tour costs). The training reward is aligned with RRC's objective up to the per-window
constant `K(prefix, s)`.

## Metrics to watch

* `train/reward`: mean `-C_red` of the sampled reconstructions.
* `train/baseline_cost`: `C_teacher` or `mean_j C_j`; it should track `sqrt(w)` as the
  per-effective-batch `w` draw changes.
* `train/regret` (teacher only): `(C_red - C_teacher)/C_teacher`; positive means worse than the
  expert. It should fall.
* `train/advantage_abs`: mean `|A|`; should shrink as the policy improves.
* `val/reward_greedy`: the honest early signal; `val/reward` is the checkpoint criterion.

## Verification

```bash
.venv/bin/python scripts/verify_lehd_rl.py
.venv/bin/python scripts/verify_lehd_data.py
.venv/bin/python scripts/verify_rrc.py
.venv/bin/python scripts/verify_synced_data.py
.venv/bin/python scripts/verify_val_parity.py
```

`verify_lehd_rl.py` pins the reward algebra (the helper, the constant-invariant across two
different reconstructions, the env reward, the start state), both arms' rollout shapes and loss
math, the shared arm's label-freeness, and state-dict/validation parity with `POMORL` and
`LEHDPOMOSL`.

## Experiment matrix

1. Smoke, then a short pilot for each arm; check `val/reward_greedy` and `train/regret` /
   `train/advantage_abs`.
2. Teacher arm, K=1, 40 × 999k partials.
3. Shared arm, K=4 (or K=2 on a pilot), same schedule.
4. RRC at 50/100/300/500/1000 from matched `(N,8)` and `(N,1)` incumbents, paired against
   `lehd`, `rl`, `sl`, and both `lehd_rl*` arms.
5. Optional: add a teacher-CE auxiliary term if pure RL lags; self-generated windows (SIL) to
   remove the optimal-tour distribution shift.

## Files

| file | role |
| --- | --- |
| `src/envs/cvrp_rl_lehd.py` | `LEHDRLCVRPEnv`, `reduced_window_cost` |
| `src/models/pomo_rl_lehd.py` | `LEHDPOMORL`: sampled rollouts, teacher/shared baselines, loss |
| `notebooks/POMO_rl_lehd.py` | training entry point |
| `src/lehd/subpath.py`, `src/envs/lehd_dataset.py` | shared LEHD geometry and dataset |
| `src/envs/cvrp_sl_lehd.py` | shared `reset_partial` state |
| `scripts/verify_lehd_rl.py` | PASS/FAIL suite |
| `scripts/submit_lehd_rl.sh` | sbatch template |
| `notebooks/eval_pomo.py`, `notebooks/eval_rrc.py` | checkpoint evaluation |
