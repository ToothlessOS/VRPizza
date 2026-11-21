# LEHD-backbone SIL-RC

The local-reconstruction self-improvement pipeline from [SIL_RC.md](SIL_RC.md),
with the reference LEHD *light encoder / heavy decoder* in place of POMO's
`AttentionModelPolicy`. The data, schedules, label cycles, reconstruction pass
and evaluation data are unchanged; only the network is different.

Written for anyone running or changing `notebooks/LEHD_sil_rc.py`,
`notebooks/LEHD_sil_rc_step.py`, `src/models/components/lehd_policy.py` and
`src/models/sil_rc_lehd.py`.

## Why this exists

The repo trains CVRP100 five ways: POMO REINFORCE (`rl-`), HGS teacher forcing
(`sl-`), LEHD-style partial reconstruction on POMO (`lehd-`), SBS
self-improvement (`sil-`), and local-reconstruction self-improvement
(`sil_rc-`). All of them use rl4co's POMO policy. This arm swaps in the
reference LEHD network so the label-free SIL-RC result can be compared with the
LEHD paper's supervised result on the same test files.

The pipeline is identical to [SIL_RC.md](SIL_RC.md): an RL warm-up on generated
CVRP20, then repeated label cycles; each cycle improves every label with RRC and
trains the policy on random sub-paths of the improved labels. The reconstruction
uses the same `src/rrc` loop and the same `src/sil_rc` label store.

## The reference architecture

The port is `CIAM-Group NCO_code main single_objective-LEHD/CVRP/VRPModel.py`.
It is a **light encoder / heavy decoder**:

| component | reference value |
| --- | --- |
| encoder input | `(x, y, demand / capacity)`, depot at index 0, depot demand 0 |
| encoder embedding | `Linear(3, 128, bias=True)` |
| encoder layers | **1** (hard-coded) |
| decoder layers | **6** |
| attention | full MHA, 8 heads x `qkv_dim=16` (=128), `Linear(128,128)` combine, residual |
| FFN | `Linear(128,512)` -> ReLU -> `Linear(512,128)`, residual |
| normalization | none |
| positional encoding | none (no CPE/APE, no graph/edge features) |
| decoder context | `[encoded depot + remaining_capacity]`, `[encoded last-selected + remaining_capacity]` |
| decoder sequence | context-first + **unselected customers in ascending id** + context-last |
| output head | `Linear(128,2)` per position: channel 0 = direct, channel 1 = via-depot |
| action space | **2N** `(customer, flag)`; the depot is never an action |
| parameters | 1,415,682 |

The reference checkpoint at
`.../CVRP/result/20230817_235537_train/checkpoint-40.pt` loads into the port with
`strict=True`: the module names and shapes match exactly.

## The action-space adapter

The repo's environment, sub-path labels, RRC parser and loss all use POMO's
`0 = depot, 1..N = customers` action space; the reference uses `2N`
`(customer, via_depot)`. The port keeps the repo's space and exposes the
reference's distribution as `N+1` logits at every step:

```
logits[0] = logsumexp_u z_via(u)
logits[c] = z_direct(c)          for an unselected customer
logits[c] = -inf                 for a visited customer
```

A softmax over those logits is exactly the reference's next-action marginal
(`logits[0]` is the total via mass, not the joint 2N normaliser). This is what
lets the unchanged `ConstructivePolicy` machinery drive sampling, greedy
rollout, multistart, teacher forcing and `get_log_likelihood`.

A via-depot action is one reference token but two POMO tokens (depot, then
customer). The **training loss** therefore does not use the coarse N+1
cross-entropy:

* a customer token that does not follow a depot is scored as `direct` at its own
  state;
* a depot token is scored as `via` for the **next** customer at the pre-depot
  state, and the structural customer token that follows is not scored again.

Each reference action is one of the window's `w-1` customers, so the per-row
denominator is LEHD's, not the POMO token count. `src/models/sil_rc_lehd.py`
implements this from the decoder's raw-logit captures;
`scripts/verify_lehd_sil_rc.py` checks the resulting CE against the reference
model's own per-step loss to ~1e-7.

The **rollout** path (labels, RRC, validation) uses the N+1 marginal. The
`LEHDPolicy.reference_rollout` helper is the faithful 2N argmax greedy for
paper-parity checks.

## Entry points

Both copy the POMO notebooks; only the module, run labels and memory defaults
change.

### Strict port (`notebooks/LEHD_sil_rc.py`)

Same schedule as `notebooks/POMO_sil_rc.py`: 100 CVRP20 warm-up epochs x 50,000
instances, then 5 cycles x 20 epochs over a fixed 1M generated CVRP100 label set,
100 RRC rounds/cycle, Adam 1e-4, no weight decay, gradient clip 10, LR 0.97 per
SIL epoch.

The heavy decoder changes two memory knobs, not the effective sizes:

| knob | POMO arm | LEHD arm |
| --- | --- | --- |
| SIL micro-batch x accumulation | 1024 x 1 | 128 x 8 (still 1,024) |
| repair batch | 1024 | 256 |
| validation batch | 256 | 64 |
| validation starts | N | 1 (paper's greedy protocol) |
| validation augmentations | 8 | 8 |
| decoder gradient checkpointing | n/a | on |

`--sil-micro-batch`, `--sil-accum`, `--val-starts`, `--val-augment`,
`--val-batch-size`, `--precision`, `--no-exact-ce` and
`--no-gradient-checkpointing` expose the knobs.

`--policy-checkpoint` warm-starts from either a repo Lightning checkpoint or the
raw reference `model_state_dict`; the reference's supervised weights are a cheap
alternative to the RL warm-up. A POMO checkpoint is rejected by key inspection,
not silently loaded into nothing.

`--subcycles S` / `--subcycle-epochs` enable the chunked schedule: each outer
cycle repairs one chunk of the fixed store, trains on it, then moves to the next.
`S=1` is the reference; the default per-chunk epochs preserve the total repair
work and training rows per outer cycle. Validation runs every epoch, so an
S-chunked run validates S times more often; `--val-starts` / `--val-augment` are
the mitigation. Example: `--labels 1000000 --subcycles 10` (100K chunks).

### Step-matched port (`notebooks/LEHD_sil_rc_step.py`)

Same 200-epoch / 20,000-step protocol as `notebooks/POMO_sil_rc_step.py`: 50
warm-up epochs (125 CVRP20 instances x 16 starts = 2,000 rollouts/step), then
five 30-epoch label cycles (12,500 fresh instances/cycle, 16 partials each =
200,000 rows/epoch, 2,000 partials/step). The LEHD split is 125 x 16 (still
2,000); the repair batch is 125; validation is 2,000 slots with one start.

The run label is `lehd_sil_rc_step` (the strict arm is `lehd_sil_rc`), so both
can be listed in one evaluation.

The step arm accepts the same `--subcycles` / `--subcycle-epochs`; S must divide
the 12,500 labels and keep `100/S` steps per chunk epoch, so the total stays
20,000 optimizer steps. The run name gains `-s{S}` / `-e{E_c}`.

## Seeding and reproducibility

Both notebooks call `seed_everything(SEED, workers=True)` **before** the policy
is constructed, so the architecture's initialisation is pinned. The data draw
contract is unchanged from [SIL_RC.md](SIL_RC.md): block-seeded instance
generation, global label indices for repair windows, per-item NumPy generators,
one `w` per effective batch, atomic sidecars.

Two additional guards matter here:

* the label sidecar records `backbone` and `policy` (e.g.
  `lehd-e1-d6-h8-q16-ff512`); a POMO-initialised sidecar is refused by a LEHD
  run, and a legacy sidecar without the field is treated as POMO so an
  in-progress POMO run stays resumable;
* the sidecar also records `subcycles_per_cycle`, `repair_step` and `subcycle`;
  a sidecar from a different chunk schedule is ignored/regenerated. The
  checkpoint records the full label schedule (including `subcycle_epochs`) and
  rejects a cross-schedule resume;
* checkpoints record the same `backbone` and label schedule; resuming a POMO
  checkpoint into the LEHD module raises before `load_state_dict` can fail
  obscurely.

`scripts/verify_lehd_sil_rc.py` checks that re-seeding reproduces every policy
parameter and that the same label seed reproduces the instance set. As elsewhere
in this repo, bf16/cuDNN training is not promised bitwise.

## Memory and compute

The reference steps the optimizer once per decoded token; this port keeps the
repo's one-step-per-effective-batch convention, so the whole teacher-forced
window is differentiated at once. The decoder re-runs six self-attention layers
at every step, so two things are essential:

* **gradient checkpointing** on the decoder stack (default on; each decode
  step's input is stored and the layers recomputed in backward);
* a **small micro-batch** with accumulation preserving the effective batch. The
  effective batch keys the `w` draw, so it must not change: 1,024 strict,
  2,000 step-matched.

Validation defaults to one start because POMO's `8 x N` rollout would be `N`
times the sequences and is intractable with a heavy decoder at every step. The
augmented single-start metric is the LEHD paper's protocol; `--val-starts N`
with a small `--batch-size` still allows the POMO-style comparison.

The strict arm's 1M-label x 100-round repair pass is the dominant cost: millions
of window reconstructions at six layers per step, roughly 80x the step arm's
per-cycle repair. Run the step arm first; use `--labels 100000` (or smaller) if
the strict schedule is needed.

The chunked schedule (`--subcycles S`) repairs one chunk and trains on it before
the next. With the default per-chunk epochs the total repairs and rows are
unchanged, but the first chunk's training starts after one chunk of RRC work
(the one-off full-store greedy/RI bootstrap still happens), and the ~1.2 GB
sidecar is rewritten after each chunk. Validation runs every epoch, so S chunks
mean S times more validation per outer cycle; reduce `--val-starts` /
`--val-augment`.

## Deliberate deviations

1. **N+1 action space via marginalisation.** The next-action marginal is exact;
   a via-depot action is decoded in two POMO steps. The exact training CE
   recovers the reference's via action, but the rollout/RRC tokenisation is the
   repo's.
2. **Capacity masking.** The repo's environment masks capacity-infeasible
   customers; the reference lets the argmax choose them and flips the flag to 1.
   The model must learn the depot action instead.
3. **One optimizer step per effective batch**, not per decoded token.
4. **Encoder cached per episode** (the reference recomputes it in training); the
   encoder input is static, so the forward values are identical.
5. **Normalised capacity.** The repo normalises demand by the batch capacity and
   the vehicle capacity is 1.0; the reference broadcasts a raw capacity scalar.
   The values are equivalent for the homogeneous capacity here.
6. **No probability floor in training.** The reference adds `1e-7` to
   probabilities `<= 1e-5`; the port uses log-softmax. The faithful
   `reference_probs` helper applies the floor for parity checks.
7. **Validation protocol.** The LEHD arm defaults to 8 augmentations x 1 start;
   the POMO arms keep 8 x N. `val/reward_greedy` is the same unaugmented greedy
   rollout on both sides, and `--num-starts` can request the POMO protocol.
8. **Detached customer gathers** are the reference's quirk; the port exposes
   `detach_encoder_nodes` (default off for SIL-RC, which trains the encoder) and
   documents that inference/checkpoint parity is identical either way.
9. **The first/last context after a depot action** uses the depot embedding; the
   reference has no mid-episode depot state (a via action jumps straight to a
   customer).
10. **Precision.** The repo's `bf16-mixed` recipe is kept; the reference ran
    fp32. `--precision 32-true` is available if the un-normalised decoder proves
    unstable in bf16.
11. **`--mcts` is not ported.** It reaches into POMO's `PrecomputedCache`; the
    eval driver refuses it for the LEHD label.
12. **Chunked sub-cycles.** `--subcycles S` interleaves a chunk's repair and
    training; the reference repairs the whole store first. With the default
    per-chunk epochs the total repairs and rows are unchanged, but each chunk's
    teacher is the best snapshot just before it and validation runs S times more
    often. This is a target-refresh experiment, not on-policy RL.

## Running it

```bash
# Pipeline checks (tiny, CPU-capable, no SwanLab)
.venv/bin/python notebooks/LEHD_sil_rc.py --smoke
.venv/bin/python notebooks/LEHD_sil_rc_step.py --smoke

# Unit/invariant checks, including reference-checkpoint parity
.venv/bin/python scripts/verify_lehd_sil_rc.py

# Full arms, one GPU each
sbatch scripts/submit_lehd_sil_rc.sh

# Both strict backbones side by side (POMO on GPU 0, LEHD on GPU 1)
sbatch scripts/submit_strict_sil_rc.sh

# A cheaper first strict run
.venv/bin/python notebooks/LEHD_sil_rc.py --labels 100000

# Chunked strict LEHD: 10 x 100K chunks, same total repair/rows
.venv/bin/python notebooks/LEHD_sil_rc.py --labels 1000000 --subcycles 10

# Warm-start SIL from the reference supervised checkpoint
.venv/bin/python notebooks/LEHD_sil_rc.py \
    --policy-checkpoint "/path/CIAM-.../CVRP/result/20230817_235537_train/checkpoint-40.pt"
```

Evaluation:

```bash
# Repo protocol; the heavy decoder needs a small batch and one start
.venv/bin/python notebooks/eval_pomo.py \
    --checkpoint lehd_sil_rc_step=<ckpt> --num-starts 1 --num-augment 1 \
    --batch-size 8

# The LEHD paper's reconstruction protocol (greedy, RRC budget 10)
.venv/bin/python notebooks/eval_rrc.py \
    --checkpoint lehd_sil_rc_step=<ckpt> --rrc-init greedy --num-starts 1 \
    --num-augment 1 --rrc-num-augment 1 --rrc-budgets 10
```

The evaluation datasets are the LEHD repo's files already used by every other
arm, so the gap is directly comparable to the paper's supervised number.

## Verification

`scripts/verify_lehd_sil_rc.py` checks:

1. parameter count 1,415,682 and state-dict equality with the reference
   checkpoint (skipped if the external checkout is absent);
2. first-step, post-customer and last-customer 2N probabilities against the
   reference decoder (max ~1e-6), and the full greedy rollout against the
   reference tester's node/flag/cost loop;
3. the 2N -> N+1 marginal identity;
4. the rl4co policy contract: greedy, multistart row order, teacher-forced
   `evaluate`, captured token distribution, the done-row depot fallback, and the
   faithful `reference_rollout`;
5. the exact CE against the reference model's per-step loss (~1e-7);
6. RRC monotonicity and coverage with the LEHD policy;
7. the sidecar/checkpoint backbone and chunk-schedule guards, a tiny two-phase
   fit, and a tiny S=2 chunked exact-CE fit;
8. seed reproducibility and the evaluation registry.

Regressions: `scripts/verify_sil_rc.py` (the POMO arm must be unchanged),
`verify_rrc.py`, `verify_lehd_data.py`, `verify_comparison.py`, `verify_sil.py`.

## Files

New:

* `src/models/components/lehd_policy.py` — encoder, decoder, policy, the N+1
  adapter and the faithful 2N helpers.
* `src/models/sil_rc_lehd.py` — `SILRCLEHDLitModule` and the exact CE.
* `notebooks/LEHD_sil_rc.py`, `notebooks/LEHD_sil_rc_step.py`.
* `scripts/verify_lehd_sil_rc.py`, `scripts/submit_lehd_sil_rc.sh`, and
  `scripts/submit_strict_sil_rc.sh` (strict POMO + strict LEHD side by side).

Modified:

* `src/models/pomo_sil_rc.py` — `BACKBONE`, `policy_signature`, overridable
  `_sil_step`, backbone guards in the sidecar and checkpoint.
* `src/models/components/pomo_inference.py` — optional `inference_num_starts`.
* `src/sil_rc/labels.py` — legacy sidecars default to the POMO backbone.
* `src/models/__init__.py`, `notebooks/eval_pomo.py` — exports and registry.
* `notebooks/POMO_sil_rc.py` — `seed_everything` (the docs already claimed it).
