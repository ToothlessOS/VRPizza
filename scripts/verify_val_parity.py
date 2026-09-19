#!/usr/bin/env python
"""Verify that the RL and SL POMO models validate identically, without pytest.

Run from the project root::

    .venv/bin/python scripts/verify_val_parity.py

``notebooks/POMO_og.py`` and ``notebooks/POMO_sl.py`` train on different objectives, and the only
reason their curves can be plotted against each other is that the validation number means the
same thing on both sides. That rests on two claims, checked here in order: that the RL
environment really does hand its validation phase the same held-out subset the supervised
environment builds, and that the two models, given identical weights and the same batch, report
identical ``reward``, ``reward_greedy`` and ``gap_ref``.

The comparison is exact rather than approximate on purpose. Both models decode greedily, so
there is no sampling noise to absorb, and an ``allclose`` tolerance here would hide precisely
the kind of silent divergence -- a different subset, a different number of augmentations, a
different decode type -- that this script exists to catch.
"""

from __future__ import annotations

import sys

from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl4co.data.dataset import TensorDictDataset  # noqa: E402
from rl4co.models.rl.reinforce.reinforce import REINFORCE  # noqa: E402

from src.envs import CVRPEnv, RLCVRPEnv  # noqa: E402
from src.models import POMORL, POMOSL  # noqa: E402

DATA_PATH = "data/CVRP/vrp100_hgs_train_100w-001.txt"
NUM_LOC = 100
CAPACITY = 50.0

# Deliberately small: the point is to compare the two rollouts, not to reproduce a training run.
# Holding out 20 instances still exercises the real subset draw, and 8 trajectories still run
# the full 8 rotations x 100 starts per instance.
VAL_INSTANCES = 20
VAL_SIZE = 64
BATCH = 8

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record and report the outcome of a single assertion."""
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")
    if not condition:
        failures.append(name)


def build_env(cls):
    """Build an environment over the shared data file with this script's small held-out set."""
    return cls(
        data_path=DATA_PATH,
        num_loc=NUM_LOC,
        capacity=CAPACITY,
        val_instances=VAL_INSTANCES,
    )


def verify_env_split() -> tuple[RLCVRPEnv, torch.Tensor]:
    """Check that the RL environment generates training data and holds out the SL val set."""
    print("\n[1/2] the RL environment splits training and validation")
    env = build_env(RLCVRPEnv)

    train = env.dataset(1_000, phase="train")
    check(
        "training batches come from the generator",
        isinstance(train, TensorDictDataset) and "actions" not in train[0],
        f"{type(train).__name__}, keys {sorted(train[0].keys())}",
    )

    # The claim the whole comparison rests on: the same file, the same held-out range, the same
    # size and the same seed, so the same draw.
    rl_val = env.dataset(VAL_SIZE, phase="val")
    sl_val = build_env(CVRPEnv).dataset(VAL_SIZE, phase="val")
    check(
        "validation trajectory pool matches the supervised range",
        rl_val.pool_size == sl_val.pool_size,
        f"{rl_val.pool_size} vs {sl_val.pool_size}",
    )
    check(
        "validation subset is the same draw",
        rl_val._pool is not None and (rl_val._pool == sl_val._pool).all(),
        f"{rl_val._length} of {rl_val.pool_size} trajectories, seed 0",
    )
    check(
        "environment reports the pool it actually draws from",
        env.num_trajectories("val") == rl_val.pool_size,
        f"num_trajectories {env.num_trajectories('val')} vs pool {rl_val.pool_size}",
    )

    batch = rl_val.collate_fn([rl_val[i] for i in range(BATCH)])
    return env, batch


def verify_model_parity(env: RLCVRPEnv, batch) -> None:
    """Check that both models, given identical weights, produce identical validation metrics."""
    print("\n[2/2] the two models report the same validation metrics")

    # The MRO hazard: POMORL resolves `calculate_loss` through POMOInferenceMixin, so if that
    # mixin ever gained one it would silently become the supervised teacher-forcing loss.
    check(
        "POMORL trains with REINFORCE's loss",
        POMORL.calculate_loss is REINFORCE.calculate_loss,
        POMORL.calculate_loss.__qualname__,
    )
    check(
        "POMOSL keeps its own loss",
        POMOSL.calculate_loss is not REINFORCE.calculate_loss,
        POMOSL.calculate_loss.__qualname__,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("medium")  # what RL4COTrainer sets

    sl = POMOSL(env, batch_size=BATCH, val_batch_size=BATCH).to(device).eval()
    rl = POMORL(env, batch_size=BATCH, val_batch_size=BATCH).to(device).eval()

    # Not `load_from_checkpoint`: that reconstructs the class from saved hyperparameters, which
    # differ between the two by design. Copying the state dict checks the thing that matters --
    # that the parameters are the same and the architectures accept them unchanged.
    rl.load_state_dict(sl.state_dict(), strict=True)
    check("state dicts are interchangeable", True, f"{len(sl.state_dict())} tensors")

    batch = batch.to(device)
    with torch.no_grad():
        sl_out = sl._inference_step(batch, "val")
        rl_out = rl._inference_step(batch, "val")

    for key in ("reward", "reward_greedy", "gap_ref"):
        a, b = sl_out[key], rl_out[key]
        check(
            f"{key} is identical",
            torch.equal(a, b),
            f"{a.tolist()} vs {b.tolist()}",
        )


def main() -> int:
    """Run every verification group and report a summary."""
    env, batch = verify_env_split()
    verify_model_parity(env, batch)
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
