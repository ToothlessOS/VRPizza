#!/usr/bin/env python
"""Verify that the synced SL and RL pipelines train on the same data, without pytest.

Run from the project root::

    .venv/bin/python scripts/verify_synced_data.py

``notebooks/POMO_sl_synced.py`` and ``notebooks/POMO_rl_synced.py`` are only comparable if
they see the same ``(instance, start)`` pairs, in the same per-batch grouping, every epoch.
That is a stronger claim than "same file, similar sizes", and none of it is visible in a loss
curve: the supervised cross-entropy is a mean over rows, so it reads the same whether an
instance's 16 starts share a batch or are scattered across 16 of them. A desynced epoch
counter or a re-enabled row shuffle would therefore train on different data and look fine.

The checks below are ordered so the failure that would invalidate everything comes first. The
comparisons are exact rather than approximate: both pipelines decode greedily at validation
and the draws are deterministic, so an ``allclose`` tolerance here would hide exactly the
silent divergence this script exists to catch.
"""

from __future__ import annotations

import importlib.util
import sys

from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envs import CVRPSolutionDataset, SyncedCVRPEnv, SyncedRLCVRPEnv, feasible_starts  # noqa: E402
from src.envs.parser import InstanceLayout  # noqa: E402
from src.models import POMORL, POMOSL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = "data/CVRP/vrp100_hgs_train_100w-001.txt"
NUM_LOC = 100
CAPACITY = 50.0
#: The HGS training files score slightly different coordinates than they store, so their cost
#: column carries ~6e-5 relative noise; see scripts/verify_sl_data.py.
COST_TOLERANCE = 2e-3

# Deliberately small: the point is the draw, not a training run. 8 instances x 4 starts keeps
# the batches tiny while still exercising a partial model call.
INSTANCES = 8
K = 4

#: Instances per step, used only by the dataloader-grouping check. It must divide the draw
#: into *several* batches: if a step held the whole draw then every batch would contain every
#: instance, and the check below would pass even with row-level shuffling switched on.
STEP_INSTANCES = 2
VAL_INSTANCES = 20
VAL_SIZE = 64

#: The constants the two entry scripts must agree on. Anything that shapes the draw or the
#: validation subset has to match exactly, or the two runs are not the same experiment.
SHARED_CONSTANTS = (
    "DATA_PATH",
    "VAL_DATA_PATH",
    "NUM_LOC",
    "CAPACITY",
    "TRAIN_INSTANCES",
    "STARTS_PER_INSTANCE",
    "INSTANCES_PER_BATCH",
    "VAL_BATCH_SIZE",
    "VAL_DATA_SIZE",
    "MAX_EPOCHS",
)

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record and report the outcome of a single assertion."""
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")
    if not condition:
        failures.append(name)


def load_entry_point(filename: str):
    """Import one of the training scripts without running it.

    The scripts guard their work behind ``if __name__ == "__main__"``, so importing only
    evaluates the constants block -- which is what this wants to compare.
    """
    path = ROOT / "notebooks" / filename
    spec = importlib.util.spec_from_file_location(f"_verify_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_env(cls, **kwargs):
    """Build a synced environment over the shared file with this script's small held-out set."""
    return cls(
        data_path=DATA_PATH,
        num_loc=NUM_LOC,
        capacity=CAPACITY,
        val_instances=VAL_INSTANCES,
        starts_per_instance=K,
        **kwargs,
    )


def raw_record(dataset, instance: int) -> dict:
    """Read instance ``instance`` straight from the file, bypassing the dataset's own indexing."""
    return dataset._read_record(instance)


def route_starts(tour: np.ndarray, flags: np.ndarray) -> list[int]:
    """Recut the reference tour into routes and list their endpoints, independently.

    Deliberately does not call :func:`~src.envs.feasible_starts`, so that a bug in that
    function cannot make its own output look correct.
    """
    positions = np.flatnonzero(flags == 1).tolist()
    starts: list[int] = []
    for index, begin in enumerate(positions):
        end = positions[index + 1] if index + 1 < len(positions) else len(tour)
        starts.append(int(tour[begin]))
        starts.append(int(tour[end - 1]))
    return starts


def verify_scripts_agree() -> None:
    """Check the two entry scripts would drive the same draw."""
    print("\n[1/6] the two entry scripts agree on the shared constants")
    sl_script = load_entry_point("POMO_sl_synced.py")
    rl_script = load_entry_point("POMO_rl_synced.py")

    for name in SHARED_CONSTANTS:
        left, right = getattr(sl_script, name), getattr(rl_script, name)
        check(f"{name} matches", left == right, f"SL {left!r} vs RL {right!r}")

    expected_rows = sl_script.INSTANCES_PER_BATCH * sl_script.STARTS_PER_INSTANCE
    check(
        "the supervised batch is one step's worth of rows",
        sl_script.BATCH_SIZE == expected_rows,
        f"BATCH_SIZE {sl_script.BATCH_SIZE} vs {expected_rows}",
    )
    # A source-level check, because the grouping is only correct if the *shipped* script turns
    # shuffling off. The behavioural check further down constructs its own model and so cannot
    # see this; the two are complementary and neither is sufficient alone. Whitespace is
    # normalised so reformatting the call cannot make this fail spuriously.
    source = (ROOT / "notebooks" / "POMO_sl_synced.py").read_text()
    compact = "".join(source.split())
    check(
        "the supervised dataloader does not shuffle rows",
        "shuffle_train_dataloader=False" in compact,
        "row-level shuffling would scatter each instance's starts across batches",
    )


def verify_draws_match() -> None:
    """Check the two datasets draw the same instances, the same slots, in the same order."""
    print("\n[2/6] the supervised and reinforcement-learning draws are the same")
    sl_env = build_env(SyncedCVRPEnv)
    rl_env = build_env(SyncedRLCVRPEnv)

    first_draw: np.ndarray | None = None
    for epoch in (1, 2):
        sl = sl_env.dataset(INSTANCES, phase="train")
        rl = rl_env.dataset(INSTANCES, phase="train")
        if first_draw is None:
            first_draw = sl.instances.copy()
        check(
            f"epoch {epoch}: same instances in the same order",
            np.array_equal(sl.instances, rl.instances),
            f"{sl.instances.size} instances",
        )
        check(
            f"epoch {epoch}: same slots per instance",
            np.array_equal(sl.slots, rl.slots),
            f"{sl.slots.shape[0]} x {sl.slots.shape[1]} slots",
        )

        # Rows are instance-major, so a run of INSTANCES*K supervised rows covers exactly the
        # instances one reinforcement-learning batch of INSTANCES covers.
        sl_rows = [int(sl.instances[divmod(i, K)[0]]) for i in range(INSTANCES * K)]
        rl_rows = [int(rl.instances[i]) for i in range(INSTANCES)]
        check(
            f"epoch {epoch}: one batch covers the same instances on both sides",
            all(sl_rows[position * K + slot] == rl_rows[position] for position in range(INSTANCES)
                for slot in range(K)),
            f"batch of {INSTANCES} instances = {INSTANCES * K} supervised rows",
        )

        # The per-item starts, which is the claim that actually matters.
        sl_starts = [int(sl[i]["start_node"]) for i in range(INSTANCES * K)]
        rl_starts = rl.collate_fn([rl[i] for i in range(INSTANCES)])["start_nodes"]
        check(
            f"epoch {epoch}: the same start node for every (instance, start) pair",
            all(
                sl_starts[position * K + slot] == int(rl_starts[position][slot])
                for position in range(INSTANCES)
                for slot in range(K)
            ),
        )

    # Both environments are now on their third epoch; the draw must have moved on from the
    # first, and the two must still agree with each other.
    sl_third = sl_env.dataset(INSTANCES, phase="train")
    rl_third = rl_env.dataset(INSTANCES, phase="train")
    check(
        "successive epochs draw different data",
        not np.array_equal(sl_third.instances, first_draw),
        "seed + epoch must move the draw along",
    )
    check(
        "the two draws stay in step after several epochs",
        np.array_equal(sl_third.instances, rl_third.instances),
        "independent epoch counters could desync silently",
    )

    # The index arithmetic above is necessary but not sufficient: the grouping is produced by
    # the shipped DataLoaders, so it is those that have to be checked. This is the check that
    # fails if `shuffle_train_dataloader` is ever turned back on for the supervised run.
    rows_per_step = STEP_INSTANCES * K
    runs = {}
    for name, model in (
        ("SL", POMOSL(sl_env, batch_size=rows_per_step, val_batch_size=VAL_SIZE,
                      train_data_size=INSTANCES, val_data_size=VAL_SIZE, num_augment=1,
                      shuffle_train_dataloader=False, dataloader_num_workers=0)),
        ("RL", POMORL(rl_env, batch_size=STEP_INSTANCES, val_batch_size=VAL_SIZE,
                      train_data_size=INSTANCES, val_data_size=VAL_SIZE, num_augment=1,
                      dataloader_num_workers=0)),
    ):
        model.setup("fit")
        runs[name] = list(model.train_dataloader())

    sl_batches, rl_batches = runs["SL"], runs["RL"]
    check(
        "the shipped dataloaders batch the same way",
        len(sl_batches) > 1
        and len(sl_batches) == len(rl_batches)
        and tuple(sl_batches[0].batch_size) == (rows_per_step,)
        and tuple(rl_batches[0].batch_size) == (STEP_INSTANCES,),
        f"{len(sl_batches)} x {tuple(sl_batches[0].batch_size)} rows vs "
        f"{len(rl_batches)} x {tuple(rl_batches[0].batch_size)} instances",
    )

    # The comparison is made on batch *contents* rather than on index arithmetic, and that is
    # the whole point. A shuffled batch holds a permutation of the rows, so mapping a row
    # position back through `dataset.instances` describes the order the batches were meant to
    # be in, not the one they are in -- a shuffling bug would sail straight through it. Reading
    # the coordinates instead catches both the wrong grouping and the wrong order.
    distinct_ok = True
    grouping_ok = True
    starts_ok = True
    per_step: list[int] = []
    for sl_batch, rl_batch in zip(sl_batches, rl_batches):
        locs = sl_batch["locs"]
        unique = {locs[row].numpy().tobytes() for row in range(locs.shape[0])}
        per_step.append(len(unique))
        distinct_ok &= len(unique) == STEP_INSTANCES

        for position in range(STEP_INSTANCES):
            block = slice(position * K, (position + 1) * K)
            grouping_ok &= torch.equal(
                locs[block], rl_batch["locs"][position].unsqueeze(0).expand(K, -1, -1)
            )
            starts_ok &= torch.equal(
                sl_batch["start_node"][block], rl_batch["start_nodes"][position]
            )

    check(
        "a supervised batch holds exactly its step's instances, K rows each",
        distinct_ok,
        f"distinct instances per step {per_step}, expected {STEP_INSTANCES}",
    )
    check(
        "one supervised batch covers exactly the reinforcement-learning batch's instances",
        grouping_ok,
        f"{len(sl_batches)} steps of {STEP_INSTANCES} instances x {K} starts",
    )
    check(
        "the supervised batch's start nodes match the reinforcement-learning batch's",
        starts_ok,
    )


def verify_starts_are_feasible() -> None:
    """Check every emitted start is one the reference solution makes legal."""
    print("\n[3/6] every start node is label-feasible")
    env = build_env(SyncedRLCVRPEnv)
    dataset = env.dataset(INSTANCES, phase="train")

    inside = True
    distinct = True
    never_depot = True
    matches_endpoints = True
    for index in range(len(dataset)):
        item = dataset[index]
        record = raw_record(dataset, int(dataset.instances[index]))
        legal = set(route_starts(record["tour"], record["flags"]))
        emitted = [int(start) for start in item["start_nodes"]]

        inside &= all(start in legal for start in emitted)
        # `feasible_starts` is the same set in slot order, which is what the datasets index.
        matches_endpoints &= set(emitted) <= set(
            int(s) for s in feasible_starts(record["tour"], record["flags"])
        )
        distinct &= len(set(emitted)) == len(emitted)
        never_depot &= all(1 <= start <= NUM_LOC for start in emitted)

    check("starts come from the reference route endpoints", inside)
    check("starts agree with feasible_starts", matches_endpoints)
    check("the K starts within a row are distinct", distinct, f"K={K}")
    check("no start is the depot or out of range", never_depot, f"1..{NUM_LOC}")


def verify_start_ordering() -> None:
    """Check the start nodes reach the decoder in the order the reward is unbatchified in.

    ``batchify`` expands a batch of B by k as ``expand(k, B, ...).view(k * B, ...)`` and
    ``unbatchify`` inverts it, so the flat index must be ``slot * B + row``. Any other
    ordering silently pairs each instance with another instance's starts, which would train
    on the wrong data while every shape still looks right.
    """
    print("\n[4/6] start nodes keep the slot-major order the decoder expects")
    env = build_env(SyncedRLCVRPEnv)
    dataset = env.dataset(INSTANCES, phase="train")
    batch = dataset.collate_fn([dataset[i] for i in range(INSTANCES)])

    td = env.reset(batch)
    check("start_nodes survives the reset", "start_nodes" in td.keys(), "rl4co rebuilds the td")
    check("get_num_starts reports K", env.get_num_starts(td) == K, f"{env.get_num_starts(td)}")

    selected = env.select_start_nodes(td, num_starts=K)
    expected = batch["start_nodes"].transpose(0, 1).reshape(-1)
    check(
        "select_start_nodes is slot-major",
        torch.equal(selected, expected),
        f"shape {tuple(selected.shape)}",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("medium")  # what RL4COTrainer sets
    model = POMORL(env, batch_size=INSTANCES, val_batch_size=INSTANCES).to(device).eval()

    with torch.no_grad():
        out = model.policy(td.to(device), env, phase="train", num_starts=K)
    check(
        "the forced first action of each rollout is its instance's start",
        torch.equal(out["actions"][:, 0].cpu(), expected),
    )
    reward = out["reward"].reshape(K, INSTANCES)
    spread = float((reward.max(0).values - reward.min(0).values).mean())
    check(
        "rollouts from different starts score differently",
        spread > 0.0,
        f"mean best-worst spread {spread:.4f}",
    )


def verify_demand_contract() -> None:
    """Check the reinforcement-learning batch normalises demand the way the environment needs.

    ``POMO.shared_step`` hands the batch straight to ``env.reset`` rather than routing it
    through ``make_input_td``, so the dataset has to have normalised demand already. Miss it
    and the action mask is scaled by ``capacity``, which is a silent capacity violation
    rather than a crash.
    """
    print("\n[5/6] the reinforcement-learning batch matches the environment's demand contract")
    env = build_env(SyncedRLCVRPEnv)
    dataset = env.dataset(INSTANCES, phase="train")
    items = [dataset[i] for i in range(INSTANCES)]
    batch = dataset.collate_fn(items)

    raw = np.stack([item["demand"] for item in items])
    capacity = np.stack([item["capacity"] for item in items]).reshape(INSTANCES, 1)
    check("demand is normalised by capacity", np.allclose(batch["demand"].numpy(), raw / capacity))
    check(
        "no reference trajectory or mask leaks into the training batch",
        "actions" not in batch.keys() and "valid" not in batch.keys(),
        f"keys {sorted(batch.keys())}",
    )

    td = env.reset(batch)
    check(
        "vehicle capacity stays at rl4co's 1.0",
        float(td["vehicle_capacity"].max()) == 1.0,
        f"max {float(td['vehicle_capacity'].max())}",
    )


def verify_supervision_and_validation() -> None:
    """Check the supervised target survives, and that validation is untouched."""
    print("\n[6/6] supervision is intact and validation still matches the other runs")
    sl_env = build_env(SyncedCVRPEnv)
    rl_env = build_env(SyncedRLCVRPEnv)
    dataset = sl_env.dataset(INSTANCES, phase="train")

    worst = 0.0
    for index in range(INSTANCES * K):
        item = dataset[index]
        reference = raw_record(dataset, int(dataset.instances[divmod(index, K)[0]]))
        walk = np.concatenate([[0], item["actions"]])
        coords = np.vstack([reference["depot"][None, :], reference["locs"]])
        length = float(np.linalg.norm(np.diff(coords[walk], axis=0), axis=1).sum())
        relative = abs(length - float(reference["cost"])) / float(reference["cost"])
        worst = max(worst, relative)
    check(
        "every supervised trajectory reproduces the reference cost",
        worst < COST_TOLERANCE,
        f"worst relative error {worst:.2e} < {COST_TOLERANCE}",
    )
    # `valid` is built by the collate function rather than the item, exactly as in
    # CVRPSolutionDataset, so it is the batch that has to carry it.
    collated = dataset.collate_fn([dataset[i] for i in range(K)])
    check(
        "the supervised dataset exposes trajectory actions and their mask",
        "actions" in dataset[0] and "valid" in collated.keys(),
        f"item keys {sorted(dataset[0].keys())}",
    )
    check(
        "the padded steps are excluded from the loss",
        not bool(collated["valid"][0, -1]) or int(collated["actions"][0, -1]) == 0,
        "padding is depot-filled and masked",
    )

    # Validation must fall through to POMO's N starts on both sides, or the curves stop being
    # comparable with the existing runs. The val batch is built by `make_input_td`, which
    # carries no start nodes, so both overrides have to notice and step aside.
    val_dataset = CVRPSolutionDataset(
        DATA_PATH, instance_range=sl_env.val_range, size=VAL_SIZE, seed=0
    )
    val_batch = val_dataset.collate_fn([val_dataset[i] for i in range(VAL_SIZE)])
    rl_td = rl_env.reset(rl_env.make_input_td(val_batch))
    check(
        "validation keeps POMO's N starts",
        rl_env.get_num_starts(rl_td) == NUM_LOC,
        f"{rl_env.get_num_starts(rl_td)} vs {NUM_LOC}",
    )
    check(
        "validation keeps rl4co's start-node selection",
        torch.equal(
            rl_env.select_start_nodes(rl_td, NUM_LOC),
            torch.arange(NUM_LOC).repeat_interleave(VAL_SIZE) % NUM_LOC + 1,
        ),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("medium")
    sl = POMOSL(sl_env, batch_size=VAL_SIZE, val_batch_size=VAL_SIZE).to(device).eval()
    rl = POMORL(rl_env, batch_size=VAL_SIZE, val_batch_size=VAL_SIZE).to(device).eval()
    # Copying weights rather than loading a checkpoint: the two classes save different
    # hyperparameters by design, and what matters is that the architectures accept the same
    # parameters unchanged.
    rl.load_state_dict(sl.state_dict(), strict=True)

    val_batch = val_batch.to(device)
    with torch.no_grad():
        sl_out = sl._inference_step(val_batch, "val")
        rl_out = rl._inference_step(val_batch, "val")
    for key in ("reward", "reward_greedy", "gap_ref"):
        left, right = sl_out[key], rl_out[key]
        check(
            f"validation {key} is identical across the synced pipelines",
            torch.equal(left, right),
            f"{tuple(left.shape)}, mean {float(left.mean()):.4f}",
        )


def main() -> int:
    """Run every verification group and report a summary."""
    verify_scripts_agree()
    verify_draws_match()
    verify_starts_are_feasible()
    verify_start_ordering()
    verify_demand_contract()
    verify_supervision_and_validation()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
