#!/usr/bin/env python
"""Verify the step-matched comparison protocol without pytest.

Run from the project root::

    .venv/bin/python scripts/verify_comparison.py

The comparison is only meaningful if every arm decodes 2,000 sequences per optimizer step,
runs 100 steps per epoch, and shares the validation draw. This script checks the arithmetic,
the generated-RL draw, the supervised 16-start grouping, the LEHD 16-partial draw, the shared
arm's 500-partial ``w`` key, and the validation subset. It does not train.
"""

from __future__ import annotations

import sys

from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envs import (  # noqa: E402
    GeneratedStepRLCVRPEnv,
    LEHDStepCVRPEnv,
    LEHDStepPartialDataset,
    LEHDStepRLCVRPEnv,
    SyncedCVRPEnv,
    sample_epoch_slots,
)
from src.envs.dataset import build_index  # noqa: E402
from src.envs.parser import resolve_data_path  # noqa: E402
from src.utils import comparison as C  # noqa: E402

DATA = C.DATA_PATH
TRAIN_RANGE = (0, 999_000)
VAL_RANGE = (999_000, 1_000_000)
SEED = 1

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record and report one assertion."""
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")
    if not condition:
        failures.append(name)


def loader(dataset, batch_size):
    """Return one batch from the dataset's own collate function."""
    return next(
        iter(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=dataset.collate_fn,
            )
        )
    )


def main() -> int:
    """Run all checks and return a process exit code."""
    print("== constants ==")
    check("samples/step", C.SAMPLES_PER_STEP == 2_000, str(C.SAMPLES_PER_STEP))
    check("instances/step", C.INSTANCES_PER_STEP == 125, str(C.INSTANCES_PER_STEP))
    check("steps/epoch", C.STEPS_PER_EPOCH == 100, str(C.STEPS_PER_EPOCH))
    check("total steps", C.TOTAL_STEPS == 20_000, str(C.TOTAL_STEPS))
    check("rows/epoch", C.TRAIN_ROWS_PER_EPOCH == 200_000, str(C.TRAIN_ROWS_PER_EPOCH))
    check("shared rows/epoch", C.SHARED_ROWS_PER_EPOCH == 50_000, str(C.SHARED_ROWS_PER_EPOCH))

    print("\n== step arithmetic ==")
    try:
        C.check_step_math(
            label="sl",
            rows_per_epoch=C.TRAIN_ROWS_PER_EPOCH,
            micro_batch=C.FULL_TOUR_MICRO_BATCH,
            accumulate=C.FULL_TOUR_ACCUM,
            samples_per_step=C.FULL_TOUR_MICRO_BATCH * C.FULL_TOUR_ACCUM,
        )
        check("SL 500 x 4 -> 100 steps", True)
    except ValueError as exc:
        check("SL 500 x 4 -> 100 steps", False, str(exc))

    try:
        C.check_step_math(
            label="shared",
            rows_per_epoch=C.SHARED_ROWS_PER_EPOCH,
            micro_batch=C.SHARED_MICRO_BATCH,
            accumulate=C.SHARED_ACCUM,
            samples_per_step=C.SHARED_MICRO_BATCH * C.SHARED_ACCUM * C.SHARED_K,
        )
        check("shared 500 x 1 x K=4 -> 100 steps", True)
    except ValueError as exc:
        check("shared 500 x 1 x K=4 -> 100 steps", False, str(exc))

    rejected = False
    try:
        C.check_step_math(
            label="bad",
            rows_per_epoch=12_500,
            micro_batch=50,
            accumulate=3,
            samples_per_step=2_000,
        )
    except ValueError:
        rejected = True
    check("non-divisor is rejected (RL 50 x 3)", rejected)

    _, num_routes, _ = build_index(resolve_data_path(DATA))

    print("\n== supervised 16-start draw ==")
    sl_env = SyncedCVRPEnv(
        DATA,
        num_loc=C.NUM_LOC,
        capacity=C.CAPACITY,
        val_instances=C.VAL_INSTANCES,
        seed=C.SEED,
        starts_per_instance=C.EXAMPLES_PER_INSTANCE,
    )
    sl_ds = sl_env.dataset(C.INSTANCES_PER_STEP, phase="train")
    check("SL rows/step", len(sl_ds) == 2_000, str(len(sl_ds)))
    chosen, _ = sample_epoch_slots(
        num_routes, TRAIN_RANGE, SEED, C.INSTANCES_PER_STEP, C.EXAMPLES_PER_INSTANCE
    )
    check("SL instance order is the shared draw", np.array_equal(sl_ds.instances, chosen))
    batch = loader(sl_ds, 2_000)
    check("SL batch covers 125 instances", len({int(sl_ds.instances[i // 16]) for i in range(2_000)}) == 125)
    check("SL batch shape", tuple(batch["actions"].shape)[0] == 2_000)

    print("\n== generated RL draw ==")
    def generated_env(seed):
        return GeneratedStepRLCVRPEnv(
            DATA,
            num_loc=C.NUM_LOC,
            capacity=C.CAPACITY,
            val_instances=C.VAL_INSTANCES,
            seed=seed,
            starts_per_instance=C.EXAMPLES_PER_INSTANCE,
        )

    a = generated_env(C.SEED).dataset(125, phase="train")
    b = generated_env(C.SEED).dataset(125, phase="train")
    c = generated_env(C.SEED + 1).dataset(125, phase="train")
    ba, bb, bc = loader(a, 125), loader(b, 125), loader(c, 125)
    check("generated draw is seeded", torch.equal(ba["locs"], bb["locs"]) and torch.equal(ba["start_nodes"], bb["start_nodes"]))
    check("generated draw changes with seed", not torch.equal(ba["locs"], bc["locs"]))
    check("start_nodes shape", tuple(ba["start_nodes"].shape) == (125, 16))
    check("start nodes are distinct per instance", all(len(set(row)) == 16 for row in ba["start_nodes"][:8].tolist()))

    gen_env = generated_env(C.SEED)
    gen_batch = loader(gen_env.dataset(125, phase="train"), 125)
    td = gen_env.reset(gen_batch)
    check("training starts = 16", gen_env.get_num_starts(td) == 16)
    check("select_start_nodes is start-major", tuple(gen_env.select_start_nodes(td, 16).shape) == (16 * 125,))

    val_batch = loader(gen_env.dataset(64, phase="val"), 32)
    val_td = gen_env.reset(gen_env.make_input_td(val_batch))
    check("validation starts = N=100", gen_env.get_num_starts(val_td) == 100)

    print("\n== LEHD 16-partial draw ==")
    def lehd_dataset(partials, effective):
        return LEHDStepPartialDataset(
            DATA,
            instance_range=TRAIN_RANGE,
            instances=C.INSTANCES_PER_STEP,
            partials_per_instance=partials,
            effective_batch_size=effective,
            seed=SEED,
        )

    teacher = lehd_dataset(16, 2_000)
    shared = lehd_dataset(4, 500)
    check("teacher rows/step", len(teacher) == 2_000, str(len(teacher)))
    check("shared rows/step", len(shared) == 500, str(len(shared)))
    check("teacher instances match SL draw", np.array_equal(teacher.instances, chosen))
    check("shared instances match SL draw", np.array_equal(shared.instances, chosen))

    first = teacher[0]
    windows = [teacher[j] for j in range(16)]
    check("16 partials share w", len({int(item["num_loc"]) for item in windows}) == 1)
    distinct = {tuple(int(x) for x in item["actions"]) for item in windows}
    check("16 partials are not duplicates", len(distinct) >= 14, f"{len(distinct)} distinct")
    check("first partial has a valid window", int(first["num_loc"]) >= 4)

    le_env = LEHDStepCVRPEnv(
        DATA,
        num_loc=C.NUM_LOC,
        capacity=C.CAPACITY,
        val_instances=C.VAL_INSTANCES,
        seed=C.SEED,
        instances=C.INSTANCES_PER_STEP,
        partials_per_instance=16,
        effective_batch_size=2_000,
    )
    check("LEHD step env effective batch", le_env.step_effective_batch_size == 2_000)
    shared_env = LEHDStepRLCVRPEnv(
        DATA,
        num_loc=C.NUM_LOC,
        capacity=C.CAPACITY,
        val_instances=C.VAL_INSTANCES,
        seed=C.SEED,
        instances=C.INSTANCES_PER_STEP,
        partials_per_instance=C.SHARED_PARTIALS_PER_INSTANCE,
        effective_batch_size=C.SHARED_PARTIALS_PER_STEP,
    )
    check("shared arm effective batch", shared_env.step_effective_batch_size == 500)
    check("shared arm rows/epoch", len(shared_env.dataset(125, phase="train")) == 500)
    check("LEHD val rows", len(le_env.dataset(64, phase="val")) == 64)

    print("\n== validation identity ==")
    sl_val = sl_env.dataset(C.VAL_DATA_SIZE, phase="val")
    gen_val = gen_env.dataset(C.VAL_DATA_SIZE, phase="val")
    le_val = le_env.dataset(C.VAL_DATA_SIZE, phase="val")
    check("SL/RL val pool identical", np.array_equal(sl_val._pool, gen_val._pool))
    check("SL/LEHD val pool identical", np.array_equal(sl_val._pool, le_val._pool))
    check("val uses the tail 1,000 instances", sl_val.instance_range == VAL_RANGE)

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
