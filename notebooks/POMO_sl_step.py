"""Step-matched POMO-SL: teacher forcing on HGS tours, 16 starts per instance.

This is the supervised arm of the per-optimizer-step comparison. One update decodes 2,000
sequences (125 instances x 16 label-feasible starts); 100 updates make an epoch; 200 epochs is
20,000 updates. The data source is the HGS file, as it must be for supervised learning.

The micro-batch is 500 rows x 4 accumulation: 2,000 rows in one forward pass is a ~41 GiB peak
on N=100, so the effective step is kept at 2,000 while the GPU sees a 500-row forward.

Run it with the project's virtualenv::

    .venv/bin/python notebooks/POMO_sl_step.py
"""

import argparse

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from swanlab.integration.pytorch_lightning import SwanLabLogger

from rl4co.utils.trainer import RL4COTrainer

from src.envs import SyncedCVRPEnv
from src.models import POMOSL
from src.utils import comparison as C

RUN_NAME = C.run_name("sl", seed=C.SEED, data="hgs")


def parse_args() -> argparse.Namespace:
    """Parse the optional resume and budget overrides."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resume", action="store_true", help="resume from last.ckpt")
    parser.add_argument("--ckpt-path", default=None, help="explicit checkpoint to resume from")
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="override MAX_EPOCHS (total budget, not additional epochs)",
    )
    return parser.parse_args()


def main() -> None:
    """Build the environment, model and trainer, then fit."""
    args = parse_args()
    max_epochs = args.max_epochs if args.max_epochs is not None else C.MAX_EPOCHS
    max_steps = (
        args.max_epochs * C.STEPS_PER_EPOCH if args.max_epochs is not None else C.TOTAL_STEPS
    )
    checkpoint_dir = C.checkpoint_dir(RUN_NAME)
    resume_path = C.resolve_resume(
        resume=args.resume, ckpt_path=args.ckpt_path, checkpoint_dir=checkpoint_dir
    )

    C.check_step_math(
        label="sl",
        rows_per_epoch=C.TRAIN_ROWS_PER_EPOCH,
        micro_batch=C.FULL_TOUR_MICRO_BATCH,
        accumulate=C.FULL_TOUR_ACCUM,
        samples_per_step=C.FULL_TOUR_MICRO_BATCH * C.FULL_TOUR_ACCUM,
    )
    seed_everything(C.SEED, workers=True)

    env = SyncedCVRPEnv(
        data_path=C.DATA_PATH,
        val_data_path=None,
        num_loc=C.NUM_LOC,
        capacity=C.CAPACITY,
        val_instances=C.VAL_INSTANCES,
        seed=C.SEED,
        starts_per_instance=C.EXAMPLES_PER_INSTANCE,
    )
    if resume_path is not None:
        C.reanchor_epoch(env, resume_path)

    model = POMOSL(
        env,
        batch_size=C.FULL_TOUR_MICRO_BATCH,
        val_batch_size=C.VAL_BATCH_SIZE,
        train_data_size=C.TRAIN_INSTANCES_PER_EPOCH,
        val_data_size=C.VAL_DATA_SIZE,
        optimizer_kwargs={"lr": C.LR, "weight_decay": C.WEIGHT_DECAY},
        num_augment=C.NUM_AUGMENT,
        # Load-bearing: rows are instance-major, so the DataLoader must batch sequentially.
        shuffle_train_dataloader=False,
        dataloader_num_workers=C.DATALOADER_WORKERS,
    )

    print(f"Samples/step: {C.SAMPLES_PER_STEP:,}")
    print(
        f"Steps/epoch: {C.STEPS_PER_EPOCH} | epochs: {max_epochs} | "
        f"total steps: {max_steps:,}"
    )
    print(
        f"Rows/epoch: {C.TRAIN_ROWS_PER_EPOCH:,} "
        f"({C.TRAIN_INSTANCES_PER_EPOCH:,} instances x {C.EXAMPLES_PER_INSTANCE} starts)"
    )
    print(
        f"Micro-batch: {C.FULL_TOUR_MICRO_BATCH} rows x {C.FULL_TOUR_ACCUM} accumulation "
        f"= {C.SAMPLES_PER_STEP:,} rows/step"
    )
    print(f"Validation monitor: {C.VAL_DATA_SIZE:,} slots, {C.VAL_BATCH_SIZE} per batch")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch_{epoch:03d}",
        auto_insert_metric_name=False,
        save_top_k=1,
        save_last=True,
        monitor="val/reward",
        mode="max",
    )
    callbacks = [checkpoint_callback, RichModelSummary(max_depth=3)]

    logger = SwanLabLogger(project="cvrp-train-pomo", experiment_name=RUN_NAME)
    trainer = RL4COTrainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        precision=C.PRECISION,
        accumulate_grad_batches=C.FULL_TOUR_ACCUM,
        gradient_clip_val=C.GRADIENT_CLIP_VAL,
    )
    if resume_path is None:
        trainer.fit(model)
    else:
        # `RL4COTrainer.fit` does not forward Lightning's `weights_only`; our checkpoints
        # pickle the training env into `hyper_parameters`, so use the base implementation.
        Trainer.fit(trainer, model, ckpt_path=resume_path, weights_only=False)


if __name__ == "__main__":
    main()
