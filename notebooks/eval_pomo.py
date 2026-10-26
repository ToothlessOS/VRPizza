"""Score the trained POMO checkpoints on the CVRP test sets.

Usage, flags and how to read the output are documented in ``docs/EvalPomo.md``; this docstring
covers what the protocol means rather than how to drive it.

Both checkpoints were trained at ``N=100``. Scored on ``vrp200``, ``vrp500`` and ``vrp1000``
they are therefore measuring zero-shot generalization rather than the objective they were
selected on, which is the whole point of running them: the network carries no ``N``-dependent
parameter, so a size it never saw is a legitimate test rather than an error.

Scoring goes through :meth:`~src.models.components.pomo_inference.POMOInferenceMixin.
_inference_step` -- the same multi-start, dihedral-augmented rollout that produced the
``val/reward`` and ``val/gap_ref`` the two runs were checkpointed on -- so these numbers mean
the same thing as the curves in wandb. Where they differ, they differ because the data does:
the logged values come from a held-out split of the HGS training file, these come from the
LKH and VRPLIB test files.

Three properties of the protocol are worth knowing before reading the table, and are printed
alongside it:

* The reported tour is the best of ``8 x N`` rollouts (8 dihedral augmentations, one start per
  customer), so it is an optimistic bound and is **not** comparable to published single-rollout
  POMO numbers.
* Per-instance results are not batch-invariant. The encoder's reduction order shifts under a
  different batch shape and flips near-ties in the argmax, so the batch size used is recorded
  and the SL/RL comparison is made *paired* -- same instances, same batches, same order.
* ``vrplib_192instances.txt`` is a transfer test on two counts at once, and its coordinates
  must be rescaled to be meaningful at all. See :func:`load_instances` and
  :meth:`~src.envs.instances.InstanceSet.model_records`.

With ``--mcts`` the same checkpoints are instead scored by the POMO-guided search in
:mod:`src.mcts`, which produces two result labels per checkpoint -- the search's answer and the
ordinary rollout it was seeded from -- and writes to :data:`MCTS_OUT` rather than
:data:`DEFAULT_OUT`. Both labels come from one pass over the same batch, so they are pairable,
and the baseline is guaranteed identical: the search seeds its incumbent from it, which means
``gap(MCTS) <= gap(POMO)`` holds by construction and is not evidence that the search helped.
The comparison that does say something is against a matched compute budget, which is what the
search-statistics section reports.

A searched run is far more expensive per instance than a plain one -- roughly ``sims_per_root``
times the forward passes -- so it defaults to a much smaller instance batch and is meant to be
run over a subset:

    .venv/bin/python notebooks/eval_pomo.py --limit 8      # smoke test, seconds
    .venv/bin/python notebooks/eval_pomo.py                # full run, ~65 min measured
    .venv/bin/python notebooks/eval_pomo.py --limit 32 --mcts --mcts-sims-per-root 32
    .venv/bin/python notebooks/eval_pomo.py --report-from logs/eval_pomo_cvrp/results.npz

or, equivalently, ``uv run notebooks/eval_pomo.py``. Paths are resolved against the repository
root, so the working directory does not matter.

Everything below is wrapped in ``main()`` behind a ``__name__`` guard on purpose. Importing
rl4co pulls in torchrl, which sets the global multiprocessing start method to ``spawn``; under
spawn each worker re-imports this module, so any unguarded module-level setup would run once
per worker and spawn workers recursively. This script uses no DataLoader, for a second reason:
per-instance results are indexed by original instance position so the two checkpoints can be
compared pairwise, and a DataLoader's shuffling and worker splitting would destroy that
alignment.
"""

from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from src.envs import EvalCVRPEnv, InstanceSet, load_instances, project_root
from src.mcts import POMOMCTSConfig, SearchStatistics, solve_batch_with_pomo_mcts
from src.models import LEHDPOMOSL, POMORL, POMOSL

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
TEST_DIR = "data/CVRP/testing dataset"
DEFAULT_OUT = "logs/eval_pomo/results.npz"

#: Where a searched run writes. Kept separate from :data:`DEFAULT_OUT` so a plain evaluation's
#: arrays -- a 65-minute run -- are never overwritten by an experiment.
MCTS_OUT = "logs/eval_pomo_cvrp/results.npz"

#: Checkpoint label (the prefix of the run directory name) to the model that loads it. The two
#: runs were built to be compared, so they are scored side by side; a run directory whose
#: prefix is not listed here is refused rather than guessed at.
#:
#: This namespace is for *checkpoints* only. The search adds a method dimension, and mixing the
#: two would make `discover_checkpoints` hunt for a directory called `rl_mcts`; see
#: :func:`method_label`.
MODEL_CLASSES = {"rl": POMORL, "sl": POMOSL, "lehd": LEHDPOMOSL}

#: Suffix a result label carries when the search produced it.
MCTS_SUFFIX = "_mcts"

#: Dihedral augmentations. ``dihedral8`` requires exactly 8, and both checkpoints were
#: validated with this value, so changing it makes the numbers incomparable to wandb.
NUM_AUGMENT = 8

#: Datasets whose coordinates are not already in the model's training range. Keyed by a
#: substring of the file stem. The LKH files are generated directly in ``[0, 1]``; the
#: ``vrplib`` instances are in their published integer coordinates and must be rescaled, which
#: is not cosmetic -- measured over eight instances, the mean gap is 69.4% unscaled against
#: 4.4% rescaled.
NORMALIZE_WHEN = ("vrplib",)

#: Batch size per customer count, measured on the 8 GiB card this was developed on with
#: ``num_augment=8`` and ``num_starts=N``. Peak memory scales with ``B * num_augment * N``, so
#: the batch has to fall as ``N`` grows; per-instance wall clock is step-bound and roughly
#: independent of ``B`` once ``N >= 500``, so this is a memory bound rather than a speed knob.
BATCH_BY_N = {100: 256, 200: 16, 500: 8, 1000: 4}

#: Customer-count buckets for the mixed-size ``vrplib`` file. Averaging raw tour lengths
#: across instances whose coordinates span three orders of magnitude would be meaningless, so
#: that dataset is reported per bucket and only its gap is pooled.
VRPLIB_BUCKETS = ((0, 50), (50, 100), (100, 200), (200, 500), (500, 10_000))


def method_label(checkpoint_label: str, searched: bool) -> str:
    """Result label for one checkpoint under one method.

    ``("rl", False) -> "rl"`` and ``("rl", True) -> "rl_mcts"``. Keeping the checkpoint and
    method namespaces apart matters in both directions: ``discover_checkpoints`` validates run
    directory prefixes against :data:`MODEL_CLASSES`, so folding ``rl_mcts`` into that table
    would send it looking for a checkpoint that does not exist.

    Args:
        checkpoint_label: ``"rl"`` or ``"sl"``.
        searched: Whether the search produced the numbers.

    Returns:
        The label to store results under.
    """
    return f"{checkpoint_label}{MCTS_SUFFIX}" if searched else checkpoint_label


def mcts_sim_threshold(cfg: POMOMCTSConfig) -> int:
    """Simulations per root below which the search never selects, only expands.

    Worth knowing before choosing a budget, because below this number the three mechanisms the
    specification exists to provide -- PUCT, progressive widening and the endgame override -- are
    all unreachable, and the "search" is just a wider greedy rollout. Section 7.1 holds the
    frontier at ``k0`` until a node has been visited 21 times, and section 7.3 creates the depot
    child at initialisation without letting it consume a customer slot, so at a root the first
    ``k0`` simulations each materialise one customer child and return; selection happens on
    simulation ``k0 + 1``. Widening needs 21, and the endgame override needs a node deep enough
    that at most ``endgame_customer_threshold`` customers remain.

    Args:
        cfg: Search configuration.

    Returns:
        ``k0 + 1``, the first simulation at which PUCT selection can run.
    """
    return cfg.k0 + 1


def mcts_batch_for(num_loc: int) -> int:
    """Instance batch size for a searched run.

    Small on purpose, and much smaller than :func:`batch_size_for`. The round's decode batch is
    ``instances * num_augment * N`` rows, which for the tabulated sizes is deliberately the same
    size as a full POMO batch -- so the batch is not what limits a searched run. The *tree* is:
    every root keeps one node per simulation, so ``instances * 8N * (1 + sims_per_root)`` nodes
    stay resident, and at ``N=1000`` that is several kilobytes each. Dividing by the
    augmentation count keeps a searched round's row count comparable to the unsearched batch it
    is being compared against, and one instance at the largest sizes.

    Args:
        num_loc: Number of customers ``N``.

    Returns:
        Instances per searched batch, at least one.
    """
    return max(1, batch_size_for(num_loc) // NUM_AUGMENT)


def batch_size_for(num_loc: int) -> int:
    """Pick a batch size for a customer count.

    Args:
        num_loc: Number of customers ``N``.

    Returns:
        The tabulated batch size when ``N`` is one of the benchmark sizes, otherwise one
        scaled from the largest entry by ``B ~ N ** -1.9``, clamped to ``[1, 256]``.
    """
    if num_loc in BATCH_BY_N:
        return BATCH_BY_N[num_loc]
    scaled = BATCH_BY_N[1000] * (1000 / max(num_loc, 1)) ** 1.9
    return int(max(1, min(256, round(scaled))))


def discover_checkpoints() -> dict[str, Path]:
    """Find the ``last.ckpt`` of every training run under ``checkpoints/``.

    Returns:
        A dict mapping the run's label -- the prefix of its directory name, ``rl``, ``sl`` or
        ``lehd`` -- to the checkpoint path.

    Raises:
        ValueError: If a run directory's prefix is not in :data:`MODEL_CLASSES`, since the
            label decides which Lightning module the weights are loaded into.
    """
    found: dict[str, Path] = {}
    for checkpoint in sorted((project_root() / "checkpoints").glob("*/last.ckpt")):
        label = checkpoint.parent.name.split("-", 1)[0]
        if label not in MODEL_CLASSES:
            raise ValueError(
                f"{checkpoint.parent.name} does not start with one of "
                f"{sorted(MODEL_CLASSES)}; add its label to MODEL_CLASSES"
            )
        found[label] = checkpoint
    if not found:
        raise FileNotFoundError("no checkpoints/*/last.ckpt found")
    return found


def natural_key(name: str) -> tuple:
    """Sort key that orders ``vrp100`` before ``vrp1000``.

    Plain string ordering puts ``vrp1000`` first, because ``"0"`` sorts before ``"_"``, which
    also means the table leads with its slowest row. Each element is tagged with its type so a
    numeric segment is never compared against a textual one.

    Args:
        name: A dataset's file stem.

    Returns:
        A sortable tuple.
    """
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part) for part in re.split(r"(\d+)", name)
    )


def discover_datasets(only: list[str] | None = None) -> list[tuple[str, Path]]:
    """List the test files, smallest benchmark first.

    Args:
        only: Optional list of file stems to restrict to.

    Returns:
        ``(stem, path)`` pairs.

    Raises:
        FileNotFoundError: If the test directory holds nothing, or ``only`` names a stem that
            is not there.
    """
    directory = project_root() / TEST_DIR
    files = {p.stem: p for p in sorted(directory.glob("*.txt"), key=lambda p: natural_key(p.stem))}

    if only:
        missing = [name for name in only if name not in files]
        if missing:
            raise FileNotFoundError(f"no such test file(s) in {directory}: {missing}")
        return [(name, files[name]) for name in only]

    if not files:
        raise FileNotFoundError(f"no .txt test files in {directory}")
    return list(files.items())


def should_normalize(name: str) -> bool:
    """Whether a dataset's coordinates need rescaling into the model's training range.

    Args:
        name: The dataset's file stem.

    Returns:
        ``True`` for the datasets matched by :data:`NORMALIZE_WHEN`.
    """
    return any(token in name for token in NORMALIZE_WHEN)


def build_batch(records: list[dict[str, np.ndarray]]) -> TensorDict:
    """Stack per-instance records into a TensorDict the environment can reset.

    Args:
        records: Records as returned by
            :meth:`~src.envs.instances.InstanceSet.model_records`.

    Returns:
        A TensorDict with ``depot`` ``[B, 2]``, ``locs`` ``[B, N, 2]``, ``demand`` ``[B, N]``,
        ``capacity`` ``[B]`` and ``cost`` ``[B]``, all float32.

    Raises:
        ValueError: If the records are not all the same size, or if any demand exceeds its
            capacity. The second is the guard against the environment's ``vehicle_capacity``
            being set to the file's capacity: demands are normalised by the batch's own
            capacity on the way in, so the two must be compared in the same units, and getting
            it wrong makes every action look feasible without anything failing.
    """
    sizes = {record["locs"].shape[0] for record in records}
    if len(sizes) != 1:
        raise ValueError(f"a batch must be uniform in num_loc, got {sorted(sizes)}")

    def stack(key: str) -> torch.Tensor:
        # float32 explicitly: `torch.as_tensor` follows numpy's float64 and the first matmul
        # would then fail with "mat1 and mat2 must have the same dtype".
        return torch.as_tensor(np.stack([r[key] for r in records]), dtype=torch.float32)

    batch = TensorDict(
        {
            "depot": stack("depot"),
            "locs": stack("locs"),
            "demand": stack("demand"),
            "capacity": stack("capacity"),
            "cost": stack("cost"),
        },
        batch_size=[len(records)],
    )

    infeasible = batch["demand"] > batch["capacity"][:, None] * (1.0 + 1e-6)
    if bool(infeasible.any()):
        raise ValueError(
            f"{int(infeasible.sum())} demand(s) exceed capacity; the environment's "
            f"vehicle_capacity and the batch's normalisation disagree"
        )
    return batch


@torch.no_grad()
def run_batch(
    model, batch: TensorDict, scale: np.ndarray, num_starts: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Score one batch with the checkpoint's rollout.

    Args:
        model: A fitted :class:`~src.models.POMORL`, :class:`~src.models.POMOSL` or
            :class:`~src.models.LEHDPOMOSL`, already pointed at an environment of the batch's
            size.
        batch: A batch from :func:`build_batch`, in model units.
        scale: ``[B]`` factors converting a length in model units back to the file's units.
        num_starts: POMO start nodes, or ``None`` for the environment's ``N``. ``1`` is a
            single rollout and is the "no multistart" arm of the inference ablation.

    Returns:
        ``(tour, gap)`` as float64 arrays of length ``B``. ``tour`` is in the file's units;
        ``gap`` is a percentage.

    Raises:
        AssertionError: If the environment's ``gap_ref`` disagrees with the gap recomputed from
            the reward and the batch's cost. Both are derived from the same two quantities
            inside the rollout, so a mismatch means they are in different units -- exactly the
            failure a rescaled dataset invites, and one that otherwise yields a plausible but
            wrong number.
    """
    out = model._inference_step(batch, "test", num_starts=num_starts)

    reward = out["reward"].double().cpu().numpy()  # negative tour length, model units
    gap = out["gap_ref"].double().cpu().numpy() * 100.0
    reference = batch["cost"].double().cpu().numpy()

    recomputed = (-reward - reference) / reference
    if not np.allclose(gap / 100.0, recomputed, rtol=1e-4, atol=1e-6):
        raise AssertionError(
            "gap_ref from the rollout does not match (tour - ref) / ref recomputed here; "
            "the reward and the batch cost are in different units"
        )
    return -reward * scale, gap


#: Relative tolerance for recomputing a returned tour's length from the original coordinates.
#: Loose on purpose. The action sequence the search returns is a permutation of customers, and
#: permutations are invariant under the dihedral augmentation, so replaying it against the
#: un-augmented coordinates must give the same length -- but only up to the float noise that
#: reordering a sum of Euclidean distances introduces, which the driver's own notes already
#: describe as enough to flip a near-tie.
INCUMBENT_RTOL = 1e-3


@torch.no_grad()
def check_incumbent(env, batch: TensorDict, result) -> None:
    """Validate every solution the search returned, against the un-augmented instance.

    The specification asks for this separately from validating rollouts, and it is worth the
    separate call: a rollout is scored inside an augmented frame, so the one thing the search
    never does is replay a returned action sequence against the coordinates the reference cost
    was measured on. That is the step where a unit error or an index mix-up would survive
    everything else, since a dihedral transform is an isometry and therefore hides it.

    Args:
        env: The environment. Its ``check_solution`` flag is overridden per call, so it does not
            need to be built with validation on -- which matters, because that flag also applies
            to the search's per-rollout scoring and would make it far slower.
        batch: The instances as handed to the search.
        result: The search's answer.

    Raises:
        AssertionError: If a returned solution is infeasible, or its length does not reproduce.
    """
    td = env.reset(env.make_input_td(batch))
    for row, actions in enumerate(result.actions):
        if not actions:
            raise AssertionError(f"the search returned no solution for instance {row}")
        rewards = env.get_reward(
            td[row].unsqueeze(0),
            torch.tensor([actions], dtype=torch.long, device=td.device),
            check_solution=True,
        )
        replayed = -float(rewards.item())
        reported = float(result.cost[row])
        if not np.isclose(replayed, reported, rtol=INCUMBENT_RTOL, atol=1e-4):
            raise AssertionError(
                f"instance {row}: the search reported {reported:.6f} but replaying its own "
                f"action sequence against the un-augmented instance gives {replayed:.6f}; the "
                "solution and the cost are not in the same frame"
            )


@torch.no_grad()
def run_search_batch(
    model, batch: TensorDict, scale: np.ndarray, env, cfg: POMOMCTSConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SearchStatistics]:
    """Score one batch with the guided search, returning both the searched and baseline rows.

    Both come from one pass. The search needs the ordinary POMO rollout anyway -- as the
    reference cost for its value normalisation and as the incumbent's seed -- so computing it
    again through ``_inference_step`` would double the baseline's cost for a number that is
    identical to this one to the last bit. ``scripts/verify_mcts.py`` checks that identity
    exactly, so the two are interchangeable and the cheaper one is used.

    Args:
        model: A fitted checkpoint, already pointed at an environment of the batch's size.
        batch: A batch from :func:`build_batch`, in model units.
        scale: ``[B]`` factors converting a length in model units back to the file's units.
        env: The environment the model is pointed at. Passed explicitly because the search
            re-binds the model's environment as evaluation moves between customer counts.
        cfg: Search configuration.

    Returns:
        ``(baseline_tour, baseline_gap, tour, gap, stats)``, each an array of length ``B``
        except the statistics. Tours are in the file's units; gaps are percentages.
    """
    result = solve_batch_with_pomo_mcts(batch, model, env, cfg)
    check_incumbent(env, batch, result)
    reference = batch["cost"].double().cpu().numpy()

    # Note the sign: `run_batch` negates because the environment reports a *reward*, which is the
    # negated tour length. The search reports costs, so negating here would flip the table.
    baseline_tour = result.baseline_cost.double().cpu().numpy() * scale
    tour = result.cost.double().cpu().numpy() * scale
    return (
        baseline_tour,
        (baseline_tour - reference) / reference * 100.0,
        tour,
        (tour - reference) / reference * 100.0,
        result.stats,
    )


def load_model(
    label: str, checkpoint: Path, env: EvalCVRPEnv, device: str, num_augment: int = NUM_AUGMENT
):
    """Build a checkpoint's model and load its weights.

    Args:
        label: The run's label, indexing :data:`MODEL_CLASSES`.
        checkpoint: Path to the ``.ckpt``.
        env: The environment to attach. Only its ``name`` matters at construction -- the
            network is built from the environment's name plus rl4co's POMO defaults, which
            both training scripts used -- but it is attached so the mixin finds it and is
            replaced per customer count as evaluation proceeds.
        device: Torch device.
        num_augment: Dihedral augmentations to validate with. Must match what the model counts
            or :meth:`~src.models.components.pomo_inference.POMOInferenceMixin._inference_step`
            raises.

    Returns:
        The fitted model, in eval mode.
    """
    model = MODEL_CLASSES[label](env, num_augment=num_augment, eval_greedy=False)

    # `weights_only=False` is required, not merely permissive: the RL checkpoint pickles its
    # training environment into `hyper_parameters`, and that object is a `RLCVRPEnv` holding
    # an absolute data path from the cluster it was trained on. Loading the state dict into a
    # model built here -- rather than going through `LightningModule.load_from_checkpoint`,
    # which would resurrect that stale environment -- is what keeps evaluation on this
    # machine's data.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval()


def evaluate(
    inst: InstanceSet,
    models: dict[str, torch.nn.Module],
    normalize: bool,
    device: str,
    batch_override: int | None,
    search_cfg: POMOMCTSConfig | None = None,
    search_batch: int | None = None,
    num_starts: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[int, int], dict[str, SearchStatistics]]:
    """Score every model over every instance of one dataset.

    Instances are grouped by customer count and processed in ascending size, so the cheap
    groups report first and a crash late in a large-``N`` run still leaves the earlier work
    usable in the log.

    With ``search_cfg`` set, each checkpoint is scored by the guided search instead, which
    produces two result labels per checkpoint: the search's own answer and the ordinary POMO
    rollout it was seeded from. Both come from the same forward passes and the same batch, which
    is what keeps them pairable -- and it is also why the searched run's batch size is reduced
    for *both* labels rather than only the searched one. Scoring them at different batch shapes
    would break the pairing, and the driver's own docstring records that per-instance results
    are not batch-invariant.

    Args:
        inst: The dataset.
        models: Label to fitted model. Their ``env`` is replaced per group.
        normalize: Whether to rescale coordinates into the model's training range.
        device: Torch device.
        batch_override: Fixed batch size, or ``None`` to pick one per customer count.
        search_cfg: Search configuration, or ``None`` for the plain POMO rollout.
        search_batch: Instances per searched batch, overriding :func:`mcts_batch_for`. Ignored
            when ``search_cfg`` is ``None``.
        num_starts: POMO start nodes for the plain rollout, or ``None`` for ``N``. Ignored by
            the searched path, whose baseline is seeded from the search.

    Returns:
        ``(tours, gaps, batch_sizes, stats)``. ``tours`` and ``gaps`` map label to ``[M]``
        float64 arrays indexed by the instance's *original* position, which is what makes the
        labels pairable; entries stay ``nan`` for any instance that was never scored.
        ``batch_sizes`` maps customer count to the batch size actually used, and ``stats`` maps
        label to its cost accounting.
    """
    total = len(inst)
    labels = [
        method_label(label, searched)
        for label in models
        for searched in ((False, True) if search_cfg is not None else (False,))
    ]
    tours = {label: np.full(total, np.nan) for label in labels}
    gaps = {label: np.full(total, np.nan) for label in labels}
    stats = {label: SearchStatistics() for label in labels}
    batch_sizes: dict[int, int] = {}

    for num_loc, positions in inst.groups_by_n().items():
        # One environment per size. `num_loc` must equal N: rl4co picks multi-start nodes as
        # `arange(num_starts) % generator.num_loc + 1`, so a smaller value would confine every
        # start to the first `num_loc` customers and quietly weaken the rollout.
        env = EvalCVRPEnv(num_loc=int(num_loc), capacity=float(inst.capacity[positions].max()))
        for model in models.values():
            model.env = env

        size = batch_override or batch_size_for(int(num_loc))
        if search_cfg is not None:
            size = min(size, search_batch or mcts_batch_for(int(num_loc)))
        started = time.time()
        done = 0

        while done < positions.size:
            chunk = positions[done : done + size]
            records, scale = inst.model_records(chunk, normalize=normalize)
            try:
                for label, model in models.items():
                    # A fresh batch per model rather than a shared one: `env.reset` mutates the
                    # TensorDict it is handed, and the two rollouts must see identical inputs.
                    batch = build_batch(records).to(device)
                    if search_cfg is None:
                        tour, gap = run_batch(model, batch, scale, num_starts=num_starts)
                        tours[label][chunk] = tour
                        gaps[label][chunk] = gap
                        continue

                    plain, plain_gap, tour, gap, run_stats = run_search_batch(
                        model, batch, scale, env, search_cfg
                    )
                    searched = method_label(label, True)
                    tours[label][chunk], gaps[label][chunk] = plain, plain_gap
                    tours[searched][chunk], gaps[searched][chunk] = tour, gap
                    stats[searched].merge(run_stats)
                    # The plain row's only cost is the baseline pass, which the search paid for
                    # as part of its own run; charging it here keeps the two rows' totals from
                    # double-counting the same forward passes.
                    stats[label].baseline_seconds += run_stats.baseline_seconds
                    stats[label].instances += run_stats.instances
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size == 1:
                    raise
                size = max(1, size // 2)
                print(f"    out of memory at N={num_loc}; retrying with batch size {size}")
                continue

            done += int(chunk.size)
            if done % (size * 8) < size:  # roughly every eight batches
                rate = done / max(time.time() - started, 1e-9)
                # Flushed: a full run is over an hour, and without this a piped or tee'd
                # stdout shows nothing until the process exits.
                print(
                    f"    N={num_loc:<5d} {done:>5d}/{positions.size:<5d} ({rate:5.1f} inst/s)",
                    flush=True,
                )

        batch_sizes[int(num_loc)] = size

    return tours, gaps, batch_sizes, stats


@dataclass(frozen=True)
class Summary:
    """Aggregate statistics for one dataset, over instances.

    Attributes:
        n: Number of instances scored.
        ref_mean: Mean reference objective, in the file's units.
        tour_mean: Mean tour length, in the file's units.
        tour_std: Sample standard deviation of the tour lengths, over instances.
        gap_mean: Mean of the per-instance optimality gaps, in percent.
        gap_std: Sample standard deviation of those gaps, in percent.
        gap_sem: Standard error of ``gap_mean``, in percent.
        gap_of_means: ``mean(tour) / mean(ref) - 1``, in percent. The other convention for
            reporting a gap; it differs from ``gap_mean`` because the ratio is nonlinear.
    """

    n: int
    ref_mean: float
    tour_mean: float
    tour_std: float
    gap_mean: float
    gap_std: float
    gap_sem: float
    gap_of_means: float


def summarize(tour: np.ndarray, gap: np.ndarray, reference: np.ndarray) -> Summary:
    """Reduce per-instance arrays to the reported statistics.

    ``ddof=1``: these are a sample of instances, not a population, and at 128 instances the
    difference from the population standard deviation is visible in the third digit.

    Args:
        tour: ``[n]`` tour lengths in the file's units.
        gap: ``[n]`` optimality gaps in percent.
        reference: ``[n]`` reference objectives.

    Returns:
        The :class:`Summary`.
    """
    n = int(gap.size)
    gap_std = float(gap.std(ddof=1)) if n > 1 else 0.0
    return Summary(
        n=n,
        ref_mean=float(reference.mean()),
        tour_mean=float(tour.mean()),
        tour_std=float(tour.std(ddof=1)) if n > 1 else 0.0,
        gap_mean=float(gap.mean()),
        gap_std=gap_std,
        gap_sem=gap_std / np.sqrt(n) if n > 0 else 0.0,
        gap_of_means=float((tour.mean() / reference.mean() - 1.0) * 100.0),
    )


def format_table(
    rows: list[tuple[str, Summary, dict[int, int]]], mixed: frozenset[str] = frozenset()
) -> str:
    """Render the summaries as a fixed-width table.

    Args:
        rows: ``(label, summary, batch_sizes)`` per dataset.
        mixed: Datasets whose instances come in different coordinate scales. Their pooled
            ``ref``, ``tour length`` and ``gap of means`` columns are not meaningful, because
            all three average lengths measured in unrelated units -- ``gap of means`` is
            ``mean(tour) / mean(ref)``, so it inherits the same defect. The ``gap %`` column
            stays valid, since each instance's gap is formed in its own frame before pooling.

    Returns:
        The table, without a trailing newline.
    """
    header = (
        f"{'dataset':<24} {'n':>6} {'ref':>9} {'tour length':>18} "
        f"{'gap %':>16} {'SEM':>7} {'gap of means':>13} {'batch':>10}"
    )
    lines = [header, "-" * len(header)]
    for label, s, sizes in rows:
        distinct = sorted({size for size in sizes.values() if size})
        # A mixed-size dataset groups by N, so it can legitimately use a different batch for
        # every group; "varies" is more useful there than 149 comma-separated numbers.
        batch = (
            str(distinct[0]) if len(distinct) == 1 else ("varies" if len(distinct) > 1 else "n/a")
        )
        if label in mixed:
            lengths = f"{'n/a (see buckets)':>18} "
            reference = f"{'n/a':>9}"
            of_means = f"{'n/a':>13}"
        else:
            lengths = f"{s.tour_mean:>9.3f} ± {s.tour_std:<6.3f} "
            reference = f"{s.ref_mean:>9.3f}"
            of_means = f"{s.gap_of_means:>12.3f}%"
        lines.append(
            f"{label:<24} {s.n:>6d} {reference} {lengths}"
            f"{s.gap_mean:>7.3f} ± {s.gap_std:<6.3f} "
            f"{s.gap_sem:>7.3f} {of_means} {batch:>10}"
        )
    return "\n".join(lines)


def paired_pairs(labels: list[str]) -> list[tuple[str, str]]:
    """Which label pairs the paired comparison should print.

    A searched run adds labels rather than replacing them, so the comparison that matters is no
    longer "the two checkpoints" but each checkpoint against itself under the two methods. The
    rule is therefore structural: every ``_mcts`` label pairs with its base, and only when no
    such pair exists does the legacy behaviour apply -- comparing exactly two labels, RL first.
    A four-label run therefore loses the rl-vs-sl comparison, which is deliberate: with four
    labels that line is no longer the interesting one, and printing it would bury the two
    method comparisons under it.

    Args:
        labels: Every result label present.

    Returns:
        ``(baseline_label, searched_label)`` pairs, searched second so a negative
        ``mean(a - b)`` reads as "the second is better", matching :func:`format_paired`.
    """
    present = set(labels)
    pairs = [
        (label[: -len(MCTS_SUFFIX)], label)
        for label in sorted(labels)
        if label.endswith(MCTS_SUFFIX) and label[: -len(MCTS_SUFFIX)] in present
    ]
    if not pairs and len(labels) == 2:
        pairs.append(paired_order(list(labels)))
    return pairs


def paired_order(labels: list[str]) -> tuple[str, str]:
    """Order two checkpoint labels for the paired comparison.

    RL is put first when present so that a negative difference reads as "RL is ahead", which
    is how the preliminary results came out and therefore the least surprising convention.

    Args:
        labels: Two labels.

    Returns:
        The two labels, RL first when it is among them.
    """
    ordered = [label for label in ("rl", "sl") if label in labels]
    ordered += [label for label in labels if label not in ordered]
    return ordered[0], ordered[1]


def format_paired(name: str, gaps: dict[str, np.ndarray], better: tuple[str, str]) -> str:
    """Render the paired comparison of two checkpoints on one dataset.

    Pairing is what makes this comparison sharp. The two models are scored on the same
    instances in the same batches, so the instance's own difficulty cancels in the difference;
    measured on 256 instances, the paired standard error was ~60x smaller than the unpaired
    one. At 128 instances an unpaired comparison would be dominated by which instances landed
    in the sample rather than by the models.

    Args:
        name: Dataset label.
        gaps: Label to ``[n]`` per-instance gaps, aligned by position.
        better: The two labels to compare, the first being the one a negative difference
            favours.

    Returns:
        One formatted line.
    """
    a, b = better
    diff = gaps[a] - gaps[b]
    n = int(diff.size)
    sem = float(diff.std(ddof=1)) / np.sqrt(n) if n > 1 else 0.0
    wins = float((diff < 0).mean()) * 100.0
    return (
        f"{name:<24} {n:>6d}  mean({a} - {b}) = {diff.mean():+7.3f} pp ± {sem:.3f}  "
        f"{a} better on {wins:5.1f}% of instances"
    )


def summarize_buckets(
    num_loc: np.ndarray, gap: np.ndarray, tour: np.ndarray, reference: np.ndarray
) -> str:
    """Break a mixed-size dataset down by customer count.

    Pooling tour lengths across ``vrplib`` instances would average quantities measured in
    unrelated coordinate scales -- one instance's coordinates run to 98, another's to 1000 --
    so only the gap is pooled; the tour means are reported within size buckets.

    Args:
        num_loc: ``[M]`` per-instance customer counts.
        gap: ``[M]`` gaps in percent.
        tour: ``[M]`` tour lengths in the file's units.
        reference: ``[M]`` reference objectives.

    Returns:
        A formatted block, one line per non-empty bucket.
    """
    lines = [f"    {'N range':<14} {'n':>5} {'gap %':>16} {'tour (mean)':>13} {'ref (mean)':>12}"]
    for low, high in VRPLIB_BUCKETS:
        mask = (num_loc >= low) & (num_loc < high) & ~np.isnan(gap)
        if not mask.any():
            continue
        label = f"{low + 1}-{high}" if low else f"<={high}"
        lines.append(
            f"    {label:<14} {int(mask.sum()):>5d} "
            f"{gap[mask].mean():>7.3f} ± {gap[mask].std(ddof=1) if mask.sum() > 1 else 0.0:<6.3f} "
            f"{tour[mask].mean():>13.3f} {reference[mask].mean():>12.3f}"
        )
    return "\n".join(lines)


CAVEATS = """\
Reading the table:
  * The tour is the best of 8 augmentations x N starts, so it is an optimistic bound and is not
    comparable to published single-rollout POMO numbers.
  * Per-instance results are not batch-invariant -- a different batch shape changes the
    encoder's reduction order and flips near-ties in the argmax. The batch sizes used are in
    the last column; the aggregate is stable to ~0.01%, individual instances are not.
  * vrp500 and vrp1000 are not independent evidence: vrp500.i0's customers are exactly
    vrp1000.i0[0:500], vrp500.i1's are vrp1000.i0[500:1000] and vrp500.i2's are
    vrp1000.i1[0:500]. The same points are re-partitioned with different depots.
  * The reference is the file's own `cost`: LKH solutions for the LKH sets, best-known
    solutions for vrplib. Neither is proven optimal, so small negative gaps are legitimate.
  * vrplib is scored against BKS *and* with rescaled coordinates *and* under a capacity regime
    it never trained on -- 135 of its 192 instances have a demand/capacity ratio above the
    training maximum of 0.18. A poor vrplib gap is not by itself evidence about size
    generalization, because three things changed at once.

Reading a searched run:
  * Its baseline row is not evidence of improvement. The search seeds its incumbent from that
    baseline, so it can never come out worse, and "MCTS >= POMO" is a property of the
    construction rather than a result. The rows to read are the paired comparison and the
    search statistics, which report what the extra compute bought and what it cost.
  * A searched run uses a smaller instance batch than a plain one, for the tree rather than for
    the decoder, so its baseline column will not reproduce a plain run's to the last digit. The
    paired comparison is still valid because both labels share the batch; a cross-run comparison
    of the baseline columns is not.
  * If the search statistics show no PUCT selections, the budget was below what the widening
    rule needs and the run measured depth-1 expansion rather than search."""


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and print the tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--datasets", nargs="*", default=None, help="file stems to evaluate (default: all)"
    )
    parser.add_argument("--checkpoints", nargs="*", default=None, help="labels, e.g. rl sl")
    parser.add_argument("--limit", type=int, default=None, help="instances per dataset")
    parser.add_argument("--batch-size", type=int, default=None, help="override the per-N batch")
    parser.add_argument("--num-augment", type=int, default=NUM_AUGMENT)
    parser.add_argument(
        "--num-starts",
        type=int,
        default=None,
        help="POMO start nodes for the plain rollout (default: N; 1 disables multistart)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out", default=None, help="where to save the per-instance arrays "
        f"(default: {DEFAULT_OUT}, or {MCTS_OUT} with --mcts)"
    )
    parser.add_argument("--no-save", action="store_true", help="print only")
    parser.add_argument("--report-from", default=None, help="re-print a saved .npz, no inference")

    search = parser.add_argument_group("search", "POMO-guided MCTS instead of the plain rollout")
    search.add_argument(
        "--mcts", action="store_true", help="score every checkpoint with the guided search as well"
    )
    search.add_argument("--mcts-sims-per-root", type=int, default=32)
    search.add_argument("--mcts-k0", type=int, default=8)
    search.add_argument("--mcts-c-pw", type=float, default=2.0)
    search.add_argument("--mcts-alpha-pw", type=float, default=0.5)
    search.add_argument("--mcts-endgame", type=int, default=8)
    search.add_argument("--mcts-expansion", default="one_child")
    search.add_argument("--mcts-c-puct", type=float, default=1.0)
    search.add_argument("--mcts-prior-temp", type=float, default=1.0)
    search.add_argument("--mcts-rollout", default="greedy")
    search.add_argument("--mcts-samples", type=int, default=1)
    search.add_argument("--mcts-reference", default="baseline_best")
    search.add_argument(
        "--mcts-validate",
        action="store_true",
        help="run the environment's solution validator on every rollout (slow, spec default)",
    )
    search.add_argument(
        "--mcts-batch", type=int, default=None, help="override instances per searched batch"
    )
    args = parser.parse_args(argv)

    # RL4COTrainer sets this; Lightning alone does not. Leaving it at the default "highest"
    # would compute the same rollout with different matmul kernels than training used and
    # produce slightly different numbers for no reason.
    torch.set_float32_matmul_precision("medium")

    if args.report_from:
        return report(np.load(args.report_from))

    checkpoints = discover_checkpoints()
    if args.checkpoints:
        unknown = [c for c in args.checkpoints if c not in checkpoints]
        if unknown:
            raise SystemExit(f"no checkpoint for {unknown}; found {sorted(checkpoints)}")
        checkpoints = {k: v for k, v in checkpoints.items() if k in args.checkpoints}

    datasets = discover_datasets(args.datasets)
    print(f"checkpoints: {', '.join(f'{k} ({v.parent.name})' for k, v in checkpoints.items())}")
    print(f"datasets:    {', '.join(name for name, _ in datasets)}")
    starts_label = "N" if args.num_starts is None else str(args.num_starts)
    print(
        f"device:      {args.device}   augmentations: {args.num_augment}   "
        f"starts: {starts_label}"
    )

    search_cfg = None
    if args.mcts:
        search_cfg = POMOMCTSConfig(
            sims_per_root=args.mcts_sims_per_root,
            k0=args.mcts_k0,
            c_pw=args.mcts_c_pw,
            alpha_pw=args.mcts_alpha_pw,
            endgame_customer_threshold=args.mcts_endgame,
            expansion=args.mcts_expansion,
            c_puct=args.mcts_c_puct,
            prior_temperature=args.mcts_prior_temp,
            rollout_mode=args.mcts_rollout,
            rollout_samples=args.mcts_samples,
            reference=args.mcts_reference,
            validate_every_solution=args.mcts_validate,
        )
        print(
            f"search:      MCTS applied to every checkpoint "
            f"(sims/root={search_cfg.sims_per_root}, k0={search_cfg.k0}, "
            f"c_pw={search_cfg.c_pw}, alpha={search_cfg.alpha_pw}, "
            f"c_puct={search_cfg.c_puct}, reference={search_cfg.reference},\n"
            f"             rollout={search_cfg.rollout_mode}, "
            f"batch={args.mcts_batch or 'auto (batch_size_for(N) // 8)'}"
        )
        threshold = mcts_sim_threshold(search_cfg)
        if search_cfg.sims_per_root < threshold:
            # Below this the frontier never saturates, so every simulation materialises a child
            # and PUCT selection never runs. The result is a wider greedy rollout, not a search,
            # and any ablation run at this budget compares two identical algorithms.
            print(
                f"  warning: sims_per_root={search_cfg.sims_per_root} is below the {threshold} "
                "this configuration needs before PUCT selection runs at all; every simulation "
                "will create one child and return, so this measures depth-1 expansion only"
            )
    print()

    # A searched run produces two labels per checkpoint, and both are scored on the same batch
    # so the pair stays comparable; see `evaluate`.
    result_labels = [
        method_label(label, searched)
        for label in checkpoints
        for searched in ((False, True) if search_cfg is not None else (False,))
    ]

    results: dict[str, np.ndarray] = {}
    summaries: dict[str, list[tuple[str, Summary, dict[int, int]]]] = {
        label: [] for label in result_labels
    }
    gaps_by_dataset: dict[str, dict[str, np.ndarray]] = {}
    stats_by_dataset: dict[str, dict[str, SearchStatistics]] = {}
    sets: dict[str, InstanceSet] = {}

    for name, path in datasets:
        inst = load_instances(path, limit=args.limit)
        normalize = should_normalize(name)
        sets[name] = inst
        sizes = sorted(set(inst.num_loc.tolist()))
        print(
            f"{name}: {len(inst)} instances, {len(sizes)} distinct size(s) "
            f"{sizes[0]}..{sizes[-1]}{' [rescaled to [0,1]]' if normalize else ''}",
            flush=True,
        )

        # One model per checkpoint, built once and re-pointed at a new environment whenever the
        # customer count changes -- `checkpoints/` holds two 15 MB files, so rebuilding them
        # per size group would be the only real cost on the 149-group vrplib file.
        first = inst.groups_by_n()[sizes[0]]
        models = {
            label: load_model(
                label,
                checkpoint,
                EvalCVRPEnv(num_loc=int(sizes[0]), capacity=float(inst.capacity[first].max())),
                args.device,
                args.num_augment,
            )
            for label, checkpoint in checkpoints.items()
        }

        started = time.time()
        tours, gaps, batch_sizes, run_stats = evaluate(
            inst,
            models,
            normalize,
            args.device,
            args.batch_size,
            search_cfg,
            args.mcts_batch,
            args.num_starts,
        )
        print(f"  done in {time.time() - started:.1f}s")

        for label in result_labels:
            summaries[label].append(
                (name, summarize(tours[label], gaps[label], inst.cost), batch_sizes)
            )
            results[f"{label}|{name}|tour"] = tours[label]
            results[f"{label}|{name}|gap"] = gaps[label]
            results[f"{label}|{name}|ref"] = inst.cost
            results[f"{label}|{name}|num_loc"] = inst.num_loc
            for field, value in run_stats[label].as_dict().items():
                # Four-part key on purpose: `report` picks result keys by counting separators,
                # so a three-part `stats|label|name` would register a phantom "stats" label.
                results[f"stats|{label}|{name}|{field}"] = np.asarray(value)
        # Saved once per dataset rather than per checkpoint: every label shares the batching by
        # construction, and a per-label batch size would not be representable.
        results[f"batch|{name}"] = np.asarray(sorted(batch_sizes.items()), dtype=np.int64).reshape(
            -1, 2
        )
        gaps_by_dataset[name] = gaps
        stats_by_dataset[name] = run_stats

        del models
        torch.cuda.empty_cache()

    for label, rows in summaries.items():
        base = label[: -len(MCTS_SUFFIX)] if label.endswith(MCTS_SUFFIX) else label
        suffix = " (guided search)" if label.endswith(MCTS_SUFFIX) else ""
        print(f"\n=== {label.upper()}  {checkpoints[base].parent.name}{suffix} ===")
        print(format_table(rows, frozenset(name for name, _, _ in rows if should_normalize(name))))

    if search_cfg is not None:
        print(f"\n=== search statistics (per configuration, all datasets) ===")
        for label in sorted(result_labels):
            if not label.endswith(MCTS_SUFFIX):
                continue
            total = SearchStatistics()
            for name, _ in datasets:
                total.merge(stats_by_dataset[name][label])
            print(f"  {label}:")
            for line in total.format_lines(prefix="    "):
                print(line)

    for a, b in paired_pairs(result_labels):
        print(f"\n=== paired comparison ({a} vs {b}, same instances and batches) ===")
        for name, _ in datasets:
            if f"{a}|{name}|gap" in results:
                print(format_paired(name, gaps_by_dataset[name], (a, b)))

    for name, _ in datasets:
        if should_normalize(name):
            inst = sets[name]
            ref = inst.cost
            print(f"\n=== {name} by instance size ===")
            for label in result_labels:
                gap = results[f"{label}|{name}|gap"]
                tour = results[f"{label}|{name}|tour"]
                print(f"  {label}:")
                print(summarize_buckets(inst.num_loc, gap, tour, ref))

    print(f"\n{CAVEATS}")

    if not args.no_save:
        out = Path(args.out or (MCTS_OUT if args.mcts else DEFAULT_OUT))
        if not out.is_absolute():
            out = project_root() / out
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **results)
        print(f"\nper-instance arrays saved to {out}")
    return 0


def report(results: np.lib.npyio.NpzFile | dict) -> int:
    """Print the tables from previously saved per-instance arrays.

    The ``vrp1000`` group takes ~24 minutes per checkpoint, so being able to re-render the
    tables while adjusting how they are presented is worth more than it costs.

    Args:
        results: The loaded ``.npz``.

    Returns:
        ``0``.
    """
    # Normalised to a plain dict so this works for both an `NpzFile` and a mapping, and so the
    # optional `batch|` entries can be probed without relying on `NpzFile.get`.
    keys = list(results.files) if hasattr(results, "files") else list(results)
    data = {key: results[key] for key in keys}

    # `${label}|${dataset}|${field}` for the results and `batch|${dataset}` for the batch
    # sizes, so the dataset names come from the three-part keys only.
    labels = sorted({k.split("|")[0] for k in keys if k.count("|") == 2})
    # Same ordering as a live run, so a re-rendered table reads identically.
    names = sorted({k.split("|")[1] for k in keys if k.count("|") == 2}, key=natural_key)

    summaries: dict[str, list[tuple[str, Summary, dict[int, int]]]] = {
        label: [] for label in labels
    }
    for label in labels:
        for name in names:
            key = f"{label}|{name}"
            if f"{key}|gap" not in data:
                continue
            tour, gap, reference = data[f"{key}|tour"], data[f"{key}|gap"], data[f"{key}|ref"]
            saved = data.get(f"batch|{name}")
            batch_sizes = {int(n): int(size) for n, size in saved} if saved is not None else {}
            summaries[label].append((name, summarize(tour, gap, reference), batch_sizes))

    for label in labels:
        print(f"\n=== {label.upper()} ===")
        print(
            format_table(
                summaries[label],
                frozenset(name for name, _, _ in summaries[label] if should_normalize(name)),
            )
        )

    searched = [label for label in labels if label.endswith(MCTS_SUFFIX)]
    if searched:
        print("\n=== search statistics (per configuration, all datasets) ===")
        for label in sorted(searched):
            fields: dict[str, list[float]] = {}
            for name in names:
                for key in keys:
                    if key.startswith(f"stats|{label}|{name}|"):
                        fields.setdefault(key.rsplit("|", 1)[1], []).append(float(data[key]))
            if not fields:
                continue
            print(f"  {label}:")
            print(f"    simulations        {sum(fields.get('simulations', [])):>10.0f}")
            print(
                f"    expansions         {sum(fields.get('expansions', [])):>10.0f}"
                f"  ({sum(fields.get('terminal_evaluations', [])):.0f} terminal, "
                f"PUCT levels {sum(fields.get('puct_selections', [])):.0f})"
            )
            print(f"    decoder calls      {sum(fields.get('decoder_calls', [])):>10.0f}")
            rows = sum(fields.get("decoder_rows", []))
            instances = sum(fields.get("instances", [])) or 1.0
            print(f"    decoder rows       {rows:>10.0f}  ({rows / instances:.0f}/instance)")
            print(
                f"    rollouts           {sum(fields.get('rollouts', [])):>10.0f}"
                f"  ({sum(fields.get('improvements', [])):.0f} improved, "
                f"{sum(fields.get('invalid_rollouts', [])):.0f} invalid)"
            )
            print(
                f"    wall clock         {sum(fields.get('total_seconds', [])):>10.1f}s"
                f"  (baseline {sum(fields.get('baseline_seconds', [])):.1f}s, "
                f"search {sum(fields.get('search_seconds', [])):.1f}s)"
            )

    for a, b in paired_pairs(labels):
        print(f"\n=== paired comparison ({a} vs {b}) ===")
        for name in names:
            if f"{a}|{name}|gap" in data:
                print(
                    format_paired(
                        name,
                        {a: data[f"{a}|{name}|gap"], b: data[f"{b}|{name}|gap"]},
                        (a, b),
                    )
                )

    # The size breakdown is what makes a mixed-scale row readable, since its pooled tour and
    # reference columns are suppressed above. Reconstructed from the saved per-instance sizes.
    for name in names:
        if not should_normalize(name):
            continue
        print(f"\n=== {name} by instance size ===")
        for label in labels:
            if f"{label}|{name}|gap" not in data:
                continue
            print(f"  {label}:")
            print(
                summarize_buckets(
                    data[f"{label}|{name}|num_loc"],
                    data[f"{label}|{name}|gap"],
                    data[f"{label}|{name}|tour"],
                    data[f"{label}|{name}|ref"],
                )
            )

    print(f"\n{CAVEATS}")
    return 0


if __name__ == "__main__":
    main()
