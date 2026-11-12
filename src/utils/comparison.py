"""The shared step budget for the four step-matched POMO training arms.

The comparison is defined per optimizer step, not per dataset: every arm decodes 2,000
sequences per update (125 instances x 16 examples), runs 100 updates per epoch, and runs 200
epochs. Each arm draws from its natural data source -- RL from the generator, SL/LEHD from the
HGS file -- so the step axis is comparable while the label-free advantage of RL is preserved.
See ``docs/ComparisonProtocol.md`` for the full rationale and the compute/data caveats.
"""

from __future__ import annotations

from pathlib import Path

DATA_PATH = "data/CVRP/vrp100_hgs_train_100w-001.txt"
NUM_LOC = 100
CAPACITY = 50.0
SEED = 0

# -- the comparability unit ---------------------------------------------------------------
SAMPLES_PER_STEP = 2_000
INSTANCES_PER_STEP = 125
EXAMPLES_PER_INSTANCE = 16
STEPS_PER_EPOCH = 100
MAX_EPOCHS = 200
TOTAL_STEPS = STEPS_PER_EPOCH * MAX_EPOCHS
SAMPLES_PER_EPOCH = SAMPLES_PER_STEP * STEPS_PER_EPOCH

#: Instances drawn from the labeled file per epoch by the SL and LEHD step arms.
TRAIN_INSTANCES_PER_EPOCH = INSTANCES_PER_STEP * STEPS_PER_EPOCH  # 12,500
#: Decoded sequences per epoch, and the row count of the SL/LEHD-teacher datasets.
TRAIN_ROWS_PER_EPOCH = TRAIN_INSTANCES_PER_EPOCH * EXAMPLES_PER_INSTANCE  # 200,000

# -- the shared optimizer recipe ----------------------------------------------------------
LR = 1e-4
WEIGHT_DECAY = 1e-6
GRADIENT_CLIP_VAL = 1.0
PRECISION = "bf16-mixed"
DATALOADER_WORKERS = 4

# -- the shared validation protocol -------------------------------------------------------
# The held-out range is always the tail 1,000 instances. Training monitors a fixed 2,000-slot
# subset of the 21,044-slot tail pool; the scientific number is the full LKH test evaluation.
VAL_INSTANCES = 1_000
VAL_DATA_SIZE = 2_000
VAL_BATCH_SIZE = 256
NUM_AUGMENT = 8

# -- micro-batch splits (memory knobs; the effective step is SAMPLES_PER_STEP) -------------
# 500 x 4 = 2,000 rows. 2,000 x 1 is a ~41 GiB peak on N=100 and is deliberately avoided.
FULL_TOUR_MICRO_BATCH = 500
FULL_TOUR_ACCUM = 4
# 125 instances x 16 starts = 2,000 rollouts.
RL_MICRO_INSTANCES = 125
RL_ACCUM = 1
# Shared arm: 500 partials x K=4 = 2,000 rollouts; the w draw is keyed on 500 partials.
SHARED_K = 4
SHARED_PARTIALS_PER_INSTANCE = 4
SHARED_PARTIALS_PER_STEP = SAMPLES_PER_STEP // SHARED_K  # 500
SHARED_MICRO_BATCH = 500
SHARED_ACCUM = 1
SHARED_ROWS_PER_EPOCH = TRAIN_INSTANCES_PER_EPOCH * SHARED_PARTIALS_PER_INSTANCE  # 50,000


def check_step_math(
    *,
    label: str,
    rows_per_epoch: int,
    micro_batch: int,
    accumulate: int,
    samples_per_step: int,
) -> int:
    """Assert that a micro/accumulation choice produces exactly one step-matched epoch.

    The dataset must be an exact multiple of the effective update size; otherwise Lightning
    emits a short final accumulation group and the step count silently changes. The decoded
    sequences per optimizer step must also equal :data:`SAMPLES_PER_STEP`.

    Args:
        label: Arm name for the error message.
        rows_per_epoch: Dataset rows (or instances) exposed per epoch.
        micro_batch: DataLoader batch size.
        accumulate: ``accumulate_grad_batches``.
        samples_per_step: Decoded sequences contributed by one optimizer step.

    Returns:
        The number of optimizer steps per epoch (always :data:`STEPS_PER_EPOCH`).
    """
    if micro_batch <= 0 or accumulate <= 0:
        raise ValueError(f"{label}: micro_batch and accumulate must be positive")
    per_step = micro_batch * accumulate
    if rows_per_epoch % per_step != 0:
        raise ValueError(
            f"{label}: {rows_per_epoch} rows/epoch is not divisible by micro_batch * "
            f"accumulate = {micro_batch} * {accumulate} = {per_step}; Lightning would emit a "
            "short final accumulation group and change the step count"
        )
    steps = rows_per_epoch // per_step
    if steps != STEPS_PER_EPOCH:
        raise ValueError(
            f"{label}: {rows_per_epoch} rows/epoch / {per_step} = {steps} steps, expected "
            f"{STEPS_PER_EPOCH}"
        )
    if samples_per_step != SAMPLES_PER_STEP:
        raise ValueError(
            f"{label}: {samples_per_step} decoded sequences/step, expected {SAMPLES_PER_STEP}"
        )
    return steps


def run_name(
    method: str,
    *,
    seed: int = SEED,
    examples: int = EXAMPLES_PER_INSTANCE,
    data: str = "labeled",
) -> str:
    """Return a run name that encodes the full step configuration.

    The text before the first ``-`` is the eval label: ``rl``, ``sl``, ``lehd``, ``lehd_rl`` or
    ``lehd_rl_shared``. ``data`` distinguishes generated from labeled training data.
    """
    return (
        f"{method}-step-vrp{NUM_LOC}-S{SAMPLES_PER_STEP}-e{STEPS_PER_EPOCH}"
        f"-E{MAX_EPOCHS}-x{examples}-lr{LR}-wd{WEIGHT_DECAY}-seed{seed}-{data}"
    )


def checkpoint_dir(name: str) -> Path:
    """Return the checkpoint directory for a run name under the step-matched root."""
    from src.envs.parser import project_root

    return Path(project_root()) / "checkpoints_step" / name


def resolve_resume(
    *,
    resume: bool,
    ckpt_path: str | None,
    checkpoint_dir: Path,
) -> str | None:
    """Resolve a resume request to an existing checkpoint path, or ``None``.

    ``last.ckpt`` is the epoch-end save, so it continues after the last completed epoch.
    Passing the best checkpoint would repeat every epoch after the best one.
    """
    if resume and ckpt_path is not None:
        raise ValueError("pass either --resume or --ckpt-path, not both")
    path = ckpt_path
    if path is None and resume:
        path = str(checkpoint_dir / "last.ckpt")
    if path is None:
        return None
    path = str(Path(path).expanduser())
    if not Path(path).is_file():
        raise FileNotFoundError(f"no checkpoint at {path}")
    return path


def reanchor_epoch(env, ckpt_path: str) -> None:
    """Set ``env._epoch`` from a checkpoint so the next draw continues the seed sequence.

    ``setup()`` builds the first epoch's dataset before Lightning restores ``current_epoch``,
    and the env's epoch counter is not in the checkpoint. Re-anchoring it is what lets a resumed
    file-backed or generated run continue instead of replaying the permutations of epochs 0..N.
    """
    import torch

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    env._epoch = int(checkpoint["epoch"]) + 1
    del checkpoint
