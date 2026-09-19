"""Supervised-learning POMO for CVRP, trained on the HGS reference solutions.

Unlike :mod:`notebooks.POMO_og`, which trains with REINFORCE on randomly generated
instances, this trains by teacher forcing on the optimal tours stored in ``data/CVRP``.
The environment is backed by that dataset, and each instance contributes ``2R``
trajectories -- one per optimal starting point -- rather than a single tour.

Run it with the project's virtualenv::

    .venv/bin/python notebooks/POMO_sl.py

or, equivalently, ``uv run notebooks/POMO_sl.py``. Data paths are resolved against the
repository root, so the working directory does not matter.

Everything below is wrapped in ``main()`` behind a ``__name__`` guard on purpose.
Importing rl4co pulls in torchrl, which sets the global multiprocessing start method to
``spawn``; under spawn each DataLoader worker re-imports this module, so any unguarded
module-level setup would run once per worker and spawn workers recursively.
"""

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning.pytorch.loggers import WandbLogger

from rl4co.utils.trainer import RL4COTrainer

from src.envs import CVRPEnv, project_root
from src.models import POMOSL

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
# The N=50 file is a drop-in alternative:
#   data/CVRP/vrp50_train_hgs_n1000000_C40-002.txt with num_loc=50, capacity=40.
# It has no companion test file, so validation is carved from the tail of the same file.
DATA_PATH = "data/CVRP/vrp100_hgs_train_100w-001.txt"
# Point this at a held-out file to evaluate transfer, e.g.
# "data/CVRP/testing dataset/vrp100_test_lkh.txt".
VAL_DATA_PATH = None

NUM_LOC = 100
CAPACITY = 50.0

# POMO defaults. Both data sizes count *trajectories*, not instances: each instance
# contributes 2R of them (about 21 for N=100), so divide by ~21 to get distinct
# instances. The N=100 training pool is roughly 21M trajectories, so 100k per epoch
# gives a ~5.5 minute epoch and covers the data over 200 epochs.
LR = 1e-4
WEIGHT_DECAY = 1e-6
BATCH_SIZE = 256
TRAIN_DATA_SIZE = 100_000
VAL_DATA_SIZE = 10_000
MAX_EPOCHS = 200

# Measured on the RTX 4060 (7.62 GiB): batch 64 peaks at 1.33 GiB / ~139 trajectories per
# second, batch 128 at 2.65 GiB / ~336. The larger batch is 2.4x faster per trajectory
# because the ~110-step decode loop dominates, so raise the batch before adding workers.
DATALOADER_WORKERS = 8

# Identifies this run in both wandb and the checkpoint directory, so the two can always be
# matched up. It names the settings most likely to be varied; widen it if you start sweeping
# something it does not mention, rather than letting two configurations share a directory.
RUN_NAME = f"sl-vrp{NUM_LOC}-lr-{LR}-wd-{WEIGHT_DECAY}-B-{BATCH_SIZE}"


def main() -> None:
    """Build the environment, model and trainer, then fit."""
    env = CVRPEnv(
        data_path=DATA_PATH,
        val_data_path=VAL_DATA_PATH,
        num_loc=NUM_LOC,
        capacity=CAPACITY,
        val_instances=VAL_DATA_SIZE // 10,
    )

    model = POMOSL(
        env,
        batch_size=BATCH_SIZE,
        val_batch_size=BATCH_SIZE,
        train_data_size=TRAIN_DATA_SIZE,
        val_data_size=VAL_DATA_SIZE,
        optimizer_kwargs={"lr": LR, "weight_decay": WEIGHT_DECAY},
        # The training subset is re-drawn each epoch, so shuffling is about decorrelating
        # within a batch rather than changing coverage.
        shuffle_train_dataloader=True,
        dataloader_num_workers=DATALOADER_WORKERS,
    )

    print(
        f"train pool: {env.num_trajectories('train'):,} trajectories "
        f"({TRAIN_DATA_SIZE:,} per epoch, {MAX_EPOCHS} epochs)"
    )
    print(f"val pool:   {env.num_trajectories('val'):,} trajectories")

    # Checkpoint on the multi-start, augmented reward so it stays comparable with POMO's
    # reported numbers; `val/reward_greedy` is the unaugmented rollout and is the better
    # signal for judging progress early on.
    #
    # Each run owns a directory named after itself, so configurations that differ in any of
    # the hyperparameters above cannot overwrite each other's `last.ckpt` and best-epoch file.
    # Re-running one identical configuration does overwrite, which is what a rerun usually
    # means; add a suffix to RUN_NAME if both should be kept.
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(project_root()) / "checkpoints" / RUN_NAME,
        filename="epoch_{epoch:03d}",
        # Without this Lightning prepends the monitored metric to the name it builds from
        # `{epoch}`, and the file lands as `epoch_epoch=000.ckpt`.
        auto_insert_metric_name=False,
        save_top_k=1,
        save_last=True,
        monitor="val/reward",
        mode="max",
    )

    rich_model_summary = RichModelSummary(max_depth=3)
    callbacks = [checkpoint_callback, rich_model_summary]

    logger = WandbLogger(
        project="cvrp-train-pomo",
        name=RUN_NAME,
    )

    trainer = RL4COTrainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        # bf16 rather than the 16-mixed default: `get_log_likelihood` asserts that
        # log-probs stay above -1000, and fp16 log-softmax underflow trips it.
        precision="bf16-mixed",
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
