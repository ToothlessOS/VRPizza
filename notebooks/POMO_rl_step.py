"""Step-matched POMO-RL: REINFORCE on generated instances, 16 starts per instance.

This is the label-free arm of the per-optimizer-step comparison. It keeps RL's natural data
source -- freshly generated instances -- but caps the multi-start at 16 so one update decodes
2,000 rollouts (125 instances x 16 starts), matching the supervised arms' decoded-sequence
budget. Validation stays on the HGS tail with POMO's N=100 starts.

The generated instances and the 16 start nodes are drawn from a ``seed + epoch`` RNG, so the
data is reproducible and a resumed run continues the same sequence.

Run it with the project's virtualenv::

    .venv/bin/python notebooks/POMO_rl_step.py
"""

import argparse

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from swanlab.integration.pytorch_lightning import SwanLabLogger

from rl4co.utils.trainer import RL4COTrainer

from src.envs import GeneratedStepRLCVRPEnv
from src.models import POMORL
from src.utils import comparison as C

RUN_NAME = C.run_name("rl", seed=C.SEED, data="generated")


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
        label="rl",
        rows_per_epoch=C.TRAIN_INSTANCES_PER_EPOCH,
        micro_batch=C.RL_MICRO_INSTANCES,
        accumulate=C.RL_ACCUM,
        samples_per_step=C.RL_MICRO_INSTANCES * C.EXAMPLES_PER_INSTANCE * C.RL_ACCUM,
    )
    seed_everything(C.SEED, workers=True)

    env = GeneratedStepRLCVRPEnv(
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

    model = POMORL(
        env,
        batch_size=C.RL_MICRO_INSTANCES,
        val_batch_size=C.VAL_BATCH_SIZE,
        train_data_size=C.TRAIN_INSTANCES_PER_EPOCH,
        val_data_size=C.VAL_DATA_SIZE,
        optimizer_kwargs={"lr": C.LR, "weight_decay": C.WEIGHT_DECAY},
        num_augment=C.NUM_AUGMENT,
        shuffle_train_dataloader=False,
        dataloader_num_workers=C.DATALOADER_WORKERS,
    )

    print(f"Samples/step: {C.SAMPLES_PER_STEP:,}")
    print(
        f"Steps/epoch: {C.STEPS_PER_EPOCH} | epochs: {max_epochs} | "
        f"total steps: {max_steps:,}"
    )
    print(
        f"Instances/epoch: {C.TRAIN_INSTANCES_PER_EPOCH:,} generated "
        f"({C.RL_MICRO_INSTANCES} per step x {C.EXAMPLES_PER_INSTANCE} starts)"
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
        accumulate_grad_batches=C.RL_ACCUM,
        gradient_clip_val=C.GRADIENT_CLIP_VAL,
    )
    if resume_path is None:
        trainer.fit(model)
    else:
        Trainer.fit(trainer, model, ckpt_path=resume_path, weights_only=False)


if __name__ == "__main__":
    main()
