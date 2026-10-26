# Running the POMO evaluation

How to use `notebooks/eval_pomo.py`, what its numbers mean, and where the traps are. Written for
anyone running or changing it; `docs/CVRPEnv.md` covers the action space the rollout drives, and
`prompts/MCTS.md` is the specification the searched mode implements.

The script scores trained checkpoints on the CVRP test sets. It has two modes:

| mode | what it runs | writes to |
| --- | --- | --- |
| plain (default) | POMO's multi-start, dihedral-augmented rollout | `logs/eval_pomo/results.npz` |
| searched (`--mcts`) | the POMO-guided search in `src/mcts`, plus the rollout it was seeded from | `logs/eval_pomo_cvrp/results.npz` |

Everything is driven by `argparse`; there is no Hydra config for this path, and no DataLoader.
Both are deliberate and explained under [Why this looks the way it does](#why-this-looks-the-way-it-does).

## Before you run

**Checkpoints.** `discover_checkpoints` globs `checkpoints/*/last.ckpt` and takes the model class
from the *directory name prefix*: `checkpoints/rl-vrp100-…/last.ckpt` is loaded as `POMORL`,
`sl-…` as `POMOSL`, and `lehd-…` as `LEHDPOMOSL` (the LEHD objective shares POMO's architecture,
so the weights are interchangeable). A directory whose prefix is not registered is refused
rather than guessed at, and `epoch_*.ckpt` files are never picked up — only `last.ckpt`. The
prefix is the text before the first `-`, so two `lehd-` directories collide; use a distinct
prefix for an ablation run, or evaluate the main run only.

Loading goes through `torch.load(…, weights_only=False)` and then `load_state_dict`, deliberately
*not* `LightningModule.load_from_checkpoint`: the RL checkpoint pickles its training environment
into `hyper_parameters`, and that object holds an absolute data path from the cluster it was
trained on. Resurrecting it would evaluate against data that is not here.

**Data.** `data/CVRP/testing dataset/`, all five files: `vrp100/200/500/1000_test_lkh.txt` and
`vrplib_192instances.txt`. The first four are generated directly in `[0, 1]`; `vrplib` is in its
published integer coordinates and is rescaled on the way in, which is not cosmetic — measured over
eight instances the mean gap is 69.4% unscaled against 4.4% rescaled.

**A GPU.** The defaults assume one; `--device cpu` works but a full run is not practical on it.

## Running the plain evaluation

```bash
.venv/bin/python notebooks/eval_pomo.py --limit 8                     # smoke test, seconds
.venv/bin/python notebooks/eval_pomo.py                               # full run, ~65 min measured
.venv/bin/python notebooks/eval_pomo.py --datasets vrp200_test_lkh    # one dataset
.venv/bin/python notebooks/eval_pomo.py --checkpoints rl              # one checkpoint
```

| flag | default | notes |
| --- | --- | --- |
| `--datasets` | all five | file stems, without `.txt` |
| `--checkpoints` | every one found | labels, i.e. directory prefixes |
| `--limit` | none | instances per dataset, taken from the front |
| `--batch-size` | per-`N` table | overrides `BATCH_BY_N`; OOM retries halve it |
| `--num-augment` | `8` | leave it; `dihedral8` requires exactly 8 |
| `--num-starts` | `N` | POMO starts; `1` disables multistart (a single model-chosen start) |
| `--device` | `cuda` if available | |
| `--out` | `logs/eval_pomo/results.npz` | |
| `--no-save` | off | print only |
| `--report-from` | none | re-print a saved `.npz`, no inference |

`--limit 8` is the cheapest useful check: it still runs the full `8 x N` rollout, so it exercises
every code path, and it takes seconds.

`--num-starts` and `--num-augment` are inference-only, so one checkpoint supports the whole
ablation: `(N,8)` is POMO's standard best-of-`8N`, `(N,1)` removes the dihedral transforms, and
`(1,8)` removes multistart (rl4co disables its forced-start path below two starts, leaving one
model-chosen start). The same two settings drive the matched baseline in
`notebooks/eval_rrc.py`, which is where RRC is evaluated; the checkpoint's training weights do
not change.

## Reading the output

```
=== RL  rl-vrp100-lr-0.0001-wd-1e-06-B-256 ===
dataset                       n       ref        tour length            gap %     SEM  gap of means      batch
--------------------------------------------------------------------------------------------------------------
vrp100_test_lkh           10000    15.647    15.899 ± 1.818    1.617 ± 0.738    0.007        1.612%        256
```

* **`tour length`** is in the file's own units — rescaled back for `vrplib`. It is the *best* of
  `8 x N` rollouts, so it is an optimistic bound and is **not** comparable to published
  single-rollout POMO numbers.
* **`gap %`** is against the file's own `cost`: LKH solutions for the LKH sets, best-known
  solutions for vrplib. Neither is proven optimal, so a small negative gap is legitimate rather
  than a bug.
* **`gap of means`** is `tour.mean() / ref.mean() - 1`, recomputed from the two means rather than
  averaged over the per-instance gaps. It is not the same number as `gap %`, and the table shows
  both because either can be the one someone wants.
* **`batch`** is recorded because per-instance results are **not batch-invariant** — see below.

Two extra blocks appear automatically: a **paired comparison** when two labels are present, which
is the comparison to read (its SEM was ~60× smaller than the unpaired one on 256 instances), and
a **per-bucket breakdown** for `vrplib`, whose sizes span three orders of magnitude and whose raw
tour lengths are therefore meaningless pooled.

## Running the searched evaluation

```bash
.venv/bin/python notebooks/eval_pomo.py --limit 32 --mcts --mcts-sims-per-root 32
.venv/bin/python notebooks/eval_pomo.py --limit 8 --mcts --mcts-sims-per-root 9 --no-save
.venv/bin/python notebooks/eval_pomo.py --report-from logs/eval_pomo_cvrp/results.npz
```

This produces **four configurations** — SL, SL+MCTS, RL, RL+MCTS — as four result labels:
`rl`, `rl_mcts`, `sl`, `sl_mcts`. Both labels of a pair come from a single pass over the same
batch, so the paired comparison between them is valid.

| flag | default | notes |
| --- | --- | --- |
| `--mcts` | off | score every checkpoint with the search as well |
| `--mcts-sims-per-root` | `32` | **the budget knob**; see the threshold below |
| `--mcts-k0` | `8` | initial customer frontier |
| `--mcts-c-pw`, `--mcts-alpha-pw` | `2.0`, `0.5` | progressive widening |
| `--mcts-endgame` | `8` | expand everything when this few customers remain |
| `--mcts-expansion` | `one_child` | or `progressive`, `fixed_k` |
| `--mcts-c-puct` | `1.0` | exploration weight; **untuned**, see below |
| `--mcts-prior-temp` | `1.0` | prior temperature |
| `--mcts-rollout` | `greedy` | or `sample`, `best_of_n_samples` |
| `--mcts-samples` | `1` | draws per leaf for `best_of_n_samples` |
| `--mcts-reference` | `baseline_best` | or `baseline_mean`, what the value is normalised against |
| `--mcts-validate` | off | run the solution validator on every rollout (slow) |
| `--mcts-batch` | `batch_size_for(N) // 8` | instances per searched batch |

### The budget threshold is not a detail

**Below `k0 + 1 = 9` simulations per root, PUCT selection never runs at all.** The widening rule
holds the frontier at `k0` until a node has been visited 21 times, and the depot child is created
at initialisation without consuming a customer slot, so at a root the first `k0` simulations each
materialise one customer child and return. Selection happens on the ninth. Below that the run
measures depth-1 expansion, not search, and §1's fixed-top-k versus progressive-widening ablation
compares two identical algorithms. The script warns when you ask for less.

Measured, on 8 instances at `N=100`: at `--mcts-sims-per-root 9` the statistics read
`expansions 57600` and `PUCT levels 6400` for 6400 roots — exactly one selection per root, on the
ninth simulation, and none before. That is the threshold as an observation rather than as an
argument.

Widening beyond `k0` first happens at 21, and the endgame override needs a node deep enough that
at most `--mcts-endgame` customers remain — roughly `N - 8` visits deep. `--mcts-sims-per-root 32`
clears the first two.

### Reading a searched run

The **search statistics** block is the part that says what the run bought and what it cost:

```
=== search statistics (per configuration, all datasets) ===
  rl_mcts:
    simulations                12  over 3200 roots
    expansions              38400  (0 terminal, PUCT levels 12800)
    decoder calls            1450
    decoder rows          4640000  (1160000/instance, 3200 rows/call)
    rollouts                41600  (11 improved, 0 invalid)
    wall clock               39.1s  (baseline 2.3s, search 36.8s)
```

`PUCT levels 0` means the budget was too small. `improved` counts rollouts that beat the
incumbent. `decoder rows / instance` is the work, and comparing it against the baseline's single
pass is the only honest cost measure — a simulation at a root is a full completion, one at depth
20 is a shorter one.

**The baseline row is not evidence of improvement.** The search seeds its incumbent from the
baseline, so `gap(mcts) <= gap(pomo)` holds by construction and "MCTS beats POMO" is a property
of the code rather than a result. What the search adds is bounded above by how often a greedy
completion from a different root beats the best of `8N` — which for these checkpoints is a small
fraction of instances. The honest control is POMO at matched compute, which is why the statistics
report decoder rows and wall clock.

### Cost

At `N=100` the search costs roughly **10 s per instance at 12 simulations per root**, and scales
roughly proportionally with the budget. Measured against the baseline it is seeded from, on four
instances: 36.8 s of search against 2.3 s of baseline, a factor of about 16 — slightly more than
the 12 simulations, because the first round also pays for the priors of every root. Both numbers
are batch-sensitive, so the factor is the stable quantity and the absolute times are not:

| setup | per instance | full 10,000-instance `vrp100` |
| --- | --- | --- |
| plain | ~0.2 s | ~65 min for both checkpoints |
| `--mcts-sims-per-root 12` | ~10 s | ~27 h per checkpoint |
| `--mcts-sims-per-root 32` | ~25 s | ~70 h per checkpoint |

Use `--limit`. The searched mode defaults to a much smaller instance batch than the plain one —
`batch_size_for(N) // 8`, so 32 at `N=100` and 1 at `N=1000` — because the constraint is the
*tree*, which holds one node per root per simulation, not the decoder batch, which is already the
same size as the plain one.

## Output files

Both modes write a single compressed `.npz` of per-instance arrays, keyed
`"${label}|${dataset}|${field}"` with fields `tour`, `gap`, `ref`, `num_loc`, plus
`"batch|${dataset}"`. A searched run adds four-part `"stats|${label}|${dataset}|${field}"` keys —
four parts on purpose, because `report` picks result keys by counting separators and a three-part
`stats|…` key would register a phantom label.

Everything is indexed by the instance's **original position** in the file, with `nan` where an
instance was not scored. That is what makes the labels pairable, and it is why the arrays are
worth keeping even though the tables can be re-rendered.

`run.log` and `report.txt` in `logs/eval_pomo/` are not written by the script — they are `tee`
captures of a run and of a `--report-from`.

## Traps

**Per-instance results are not batch-invariant.** The encoder's reduction order shifts with the
batch shape and flips near-ties in the argmax. Aggregates are stable to ~0.01%; individual
instances are not. This is why the batch size is recorded, why the SL/RL comparison is made
paired, and why a searched run uses a smaller batch for *both* of its labels — scoring them at
different shapes would break the pairing.

**`vrp500` and `vrp1000` are not independent evidence.** `vrp500.i0`'s customers are exactly
`vrp1000.i0[0:500]`, `vrp500.i1`'s are `vrp1000.i0[500:1000]`, and `vrp500.i2`'s are
`vrp1000.i1[0:500]`. The same points are re-partitioned with different depots.

**`vrplib` changes three things at once** — reference quality, coordinate scale, and a capacity
regime it never trained on (135 of its 192 instances have a demand/capacity ratio above the
training maximum of 0.18). A poor gap there is not by itself evidence about size generalization.

**`num_loc` must equal the batch's `N`.** rl4co picks multi-start nodes as
`arange(num_starts) % generator.num_loc + 1`, so a smaller value silently confines every start to
the first `num_loc` customers. The driver builds one environment per size group to honour this.

**A demand above capacity is not an error.** The forced start bypasses the mask, so such an
instance parks at the depot with `done` never firing and is scored as an incomplete tour, silently
— `EvalCVRPEnv` defaults `check_solution=False`. `build_batch`'s demand-versus-capacity guard is
what prevents this, and it lives in the driver, not the environment. It is load-bearing.

**`--mcts-c-puct` is untuned.** The value spread is ~0.02–0.06 while the root exploration bonus is
~0.05–0.4, so at the default `1.0` the search is close to prior-ordered breadth expansion and `Q`
barely discriminates. The first ablation to run is a sweep of that flag, not a larger budget.

## Why this looks the way it does

**No Hydra.** `src/train.py` and `src/eval.py` are the Lightning-Hydra template and still default
to MNIST; there are no CVRP configs. This script is standalone `argparse` over `src/envs` and
`src/models`.

**No DataLoader.** Per-instance results are indexed by original position so the checkpoints can be
compared pairwise, and a DataLoader's shuffling and worker splitting would destroy that alignment.

**Everything inside `main()` behind a `__name__` guard.** Importing rl4co pulls in torchrl, which
sets the global multiprocessing start method to `spawn`; under spawn each worker re-imports the
module, so any unguarded module-level setup would run once per worker and spawn workers
recursively.

**`torch.set_float32_matmul_precision("medium")`.** `RL4COTrainer` sets this and Lightning alone
does not. Leaving it at `"highest"` would compute the same rollout with different matmul kernels
than training used and produce slightly different numbers for no reason.

**A fresh batch per model per chunk.** `env.reset` widens `locs` in place, so a batch that has
been reset cannot be reset again.

## Related scripts

```bash
.venv/bin/python notebooks/eval_rrc.py           # LEHD Random Re-Construct + matched baseline
.venv/bin/python scripts/verify_mcts.py          # the MCTS correctness suite, ~40 checks
.venv/bin/python scripts/verify_val_parity.py    # SL and RL validate identically
.venv/bin/python scripts/verify_sl_data.py       # the supervised data pipeline
.venv/bin/python scripts/verify_lehd_data.py     # LEHD partial-solution geometry and draw
.venv/bin/python scripts/verify_rrc.py           # RRC and the POMO/LEHD model parity
```

All of these are plain harnesses that print PASS/FAIL and exit non-zero, because pytest is not
installed in this virtualenv and the repository's convention is a script rather than a test
dependency. `verify_mcts.py` needs no checkpoint — it builds a randomly initialised `POMOSL` at
`N=10`, so it runs in seconds and exercises the real forward pass.
