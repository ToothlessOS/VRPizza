"""POMO trained LEHD-style on partial solutions sampled from the HGS reference tours.

The existing ``POMO_sl`` trains on whole reference trajectories; LEHD's supervised scheme
instead samples a contiguous window of ``w ~ Unif{4..N}`` customers ending at a route end,
discards the rest of the tour, and trains the model to reconstruct the window from its first
customer and the remaining capacity. See ``docs/LEHD_SL.md`` and
:mod:`src.lehd.subpath` for the geometry and the reference implementation.

This script keeps the POMO backbone and validation protocol, so ``val/reward`` is directly
comparable with the ``sl-`` and ``rl-`` runs. Two batching details matter:

* **The optimizer batch is 1024 (LEHD's), but the DataLoader batch is 128.** Gradient
  accumulation bridges the two. The partial dataset keys its one-``w``-per-batch draw by
  ``index // effective_batch_size``, so the draw stays constant across the eight
  microbatches, and ``collate_partials`` refuses a microbatch whose rows disagree on ``w``.
* **Row shuffling is off.** The dataset's rows are already a seeded permutation of instances;
  shuffling at row level would not change the loss value but would break the effective-batch
  grouping (and therefore the ``w`` draw).

Run it with the project's virtualenv::

    .venv/bin/python notebooks/POMO_sl_lehd.py
    .venv/bin/python notebooks/POMO_sl_lehd.py --smoke

Everything below is wrapped in ``main()`` behind a ``__name__`` guard because importing rl4co
sets the global multiprocessing start method to ``spawn``; under spawn each DataLoader worker
re-imports this module.
"""

import argparse

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pathlib import Path

import torch

from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning.pytorch.loggers import WandbLogger

from rl4co.utils.trainer import RL4COTrainer

from src.envs import LEHDCVRPEnv, project_root
from src.models import LEHDPOMOSL

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
DATA_PATH = "data/CVRP/vrp100_hgs_train_100w-001.txt"
VAL_DATA_PATH = None

NUM_LOC = 100
CAPACITY = 50.0

# LEHD's batch size, split into microbatches for the card. The effective batch is what the
# partial dataset uses to key its one-w-per-batch draw.
EFFECTIVE_BATCH_SIZE = 1024
MICRO_BATCH_SIZE = 128
ACCUMULATE_GRAD_BATCHES = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE

# One partial per training instance per epoch. The file has 1M lines and 1k are held out.
TRAIN_PARTIALS = 999_000
MAX_EPOCHS = 40

# LEHD: Adam 1e-4, no weight decay, MultiStepLR gamma 0.9 stepped once per epoch.
LR = 1e-4
WEIGHT_DECAY = 0.0
LR_GAMMA = 0.9

VAL_BATCH_SIZE = 256
VAL_DATA_SIZE = 10_000
NUM_AUGMENT = 8

# Every worker is a fresh interpreter under spawn and re-imports torch and rl4co before it
# reads a record (~790 MB measured in docs/RL_SL_data_pipeline.md). Eight workers is ~6.3 GB;
# request job memory accordingly.
DATALOADER_WORKERS = 8

# LEHD does not clip gradients; the POMO trainer defaults to 1.0. None keeps the reference
# recipe, and is recorded as part of the comparison.
GRADIENT_CLIP_VAL = None

RUN_NAME = (
    f"lehd-vrp100-lr-{LR}-wd-{WEIGHT_DECAY}"
    f"-B-{EFFECTIVE_BATCH_SIZE}-w-4-{NUM_LOC}"
)


def parse_args() -> argparse.Namespace:
    """Parse the optional smoke-test overrides."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run (few thousand partials, one epoch, no wandb) for pipeline checks",
    )
    return parser.parse_args()


def main() -> None:
    """Build the environment, model and trainer, then fit."""
    args = parse_args()

    train_partials = TRAIN_PARTIALS
    max_epochs = MAX_EPOCHS
    val_data_size = VAL_DATA_SIZE
    val_batch_size = VAL_BATCH_SIZE
    dataloader_workers = DATALOADER_WORKERS
    num_augment = NUM_AUGMENT
    run_name = RUN_NAME

    if args.smoke:
        train_partials = 4_096
        max_epochs = 1
        val_data_size = 512
        val_batch_size = 64
        dataloader_workers = 0
        num_augment = 1
        run_name = f"{RUN_NAME}-smoke"

    if EFFECTIVE_BATCH_SIZE % MICRO_BATCH_SIZE != 0:
        raise ValueError(
            f"effective batch {EFFECTIVE_BATCH_SIZE} is not a multiple of the microbatch "
            f"{MICRO_BATCH_SIZE}; a microbatch would straddle two w draws"
        )

    env = LEHDCVRPEnv(
        data_path=DATA_PATH,
        val_data_path=VAL_DATA_PATH,
        num_loc=NUM_LOC,
        capacity=CAPACITY,
        val_instances=val_data_size // 10,
        effective_batch_size=EFFECTIVE_BATCH_SIZE,
    )

    model = LEHDPOMOSL(
        env,
        # `batch_size` counts partial examples in a microbatch; `train_data_size` counts
        # partials per epoch and is passed to the environment's dataset.
        batch_size=MICRO_BATCH_SIZE,
        val_batch_size=val_batch_size,
        train_data_size=train_partials,
        val_data_size=val_data_size,
        optimizer_kwargs={"lr": LR, "weight_decay": WEIGHT_DECAY},
        lr_scheduler="MultiStepLR",
        lr_scheduler_kwargs={"milestones": list(range(1, max_epochs)), "gamma": LR_GAMMA},
        lr_scheduler_interval="epoch",
        num_augment=num_augment,
        # Load-bearing: the dataset's one-w-per-effective-batch draw assumes sequential row
        # batching. Randomisation comes from the epoch draw, not from row shuffling.
        shuffle_train_dataloader=False,
        dataloader_num_workers=dataloader_workers,
    )

    chunks = -(-train_partials // EFFECTIVE_BATCH_SIZE)
    print(
        f"train: {train_partials:,} partials per epoch, {max_epochs} epochs, "
        f"{chunks:,} w draws"
    )
    print(
        f"       microbatch {MICRO_BATCH_SIZE} x {ACCUMULATE_GRAD_BATCHES} accumulation "
        f"= effective batch {EFFECTIVE_BATCH_SIZE}"
    )
    print(
        f"val:   {env.num_trajectories('val'):,} held-out trajectories, "
        f"num_augment={num_augment}"
    )

    callbacks = [RichModelSummary(max_depth=3)]
    if not args.smoke:
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(project_root()) / "checkpoints" / run_name,
            filename="epoch_{epoch:03d}",
            auto_insert_metric_name=False,
            save_top_k=1,
            save_last=True,
            monitor="val/reward",
            mode="max",
        )
        callbacks.append(checkpoint_callback)
    # The smoke run deliberately writes no checkpoint: a `lehd-...-smoke` directory would be
    # discovered as the `lehd` label and could shadow the real run's `last.ckpt`.

    logger = False if args.smoke else WandbLogger(project="cvrp-train-pomo", name=run_name)

    trainer = RL4COTrainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        precision="bf16-mixed",
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        # The smoke run writes nothing: no explicit checkpoint callback, and Lightning's
        # default callback would otherwise drop a `epoch=...ckpt` beside the run directories.
        enable_checkpointing=not args.smoke,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
