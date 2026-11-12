"""Step-matched LEHD-RL: REINFORCE on partial reconstructions.

Two arms are available:

* ``teacher``: K=1 rollout per partial, baseline = the HGS window's reduced cost. One update is
  2,000 partials = 2,000 rollouts (125 instances x 16 partials).
* ``shared``: K=4 rollouts per partial, baseline = their mean. To keep the update at 2,000
  rollouts, each instance contributes 4 partials, not 16: 125 x 4 partials x 4 samples = 2,000.
  The 2,000 rollouts are 500 independent partials x 4 correlated samples, not 2,000 independent
  samples.

The ``w`` draw is keyed on the optimizer-step partial count, so the teacher arm uses 2,000 and
the shared arm 500. Both are 100 updates per epoch.

Run it with the project's virtualenv::

    .venv/bin/python notebooks/POMO_rl_lehd_step.py                     # teacher, K=1
    .venv/bin/python notebooks/POMO_rl_lehd_step.py --baseline shared   # label-free, K=4
    .venv/bin/python notebooks/POMO_rl_lehd_step.py --smoke             # pipeline check
"""

import argparse

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pathlib import Path

import torch

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from swanlab.integration.pytorch_lightning import SwanLabLogger

from rl4co.utils.trainer import RL4COTrainer

from src.envs import LEHDStepRLCVRPEnv
from src.models import LEHDPOMORL
from src.utils import comparison as C


def parse_args() -> argparse.Namespace:
    """Parse the arm and smoke-test overrides."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--baseline",
        choices=("teacher", "shared"),
        default="teacher",
        help="advantage baseline; 'shared' runs the label-free signal",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="rollouts per partial (default: 1 for teacher, 4 for shared)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run (16 steps, no SwanLab/checkpoint) for pipeline checks",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from <checkpoints_step>/<run_name>/last.ckpt",
    )
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
    baseline = args.baseline
    num_samples = args.num_samples
    if num_samples is None:
        num_samples = 1 if baseline == "teacher" else C.SHARED_K
    if baseline == "shared" and num_samples < 2:
        raise ValueError("--baseline shared needs --num-samples >= 2")

    if baseline == "teacher":
        partials_per_instance = C.EXAMPLES_PER_INSTANCE
        effective_batch_size = C.SAMPLES_PER_STEP
        micro_batch = C.FULL_TOUR_MICRO_BATCH
        accumulate = C.FULL_TOUR_ACCUM
    else:
        partials_per_instance = C.SHARED_PARTIALS_PER_INSTANCE
        effective_batch_size = C.SHARED_PARTIALS_PER_STEP
        micro_batch = C.SHARED_MICRO_BATCH
        accumulate = C.SHARED_ACCUM

    train_instances = C.TRAIN_INSTANCES_PER_EPOCH
    rows_per_epoch = train_instances * partials_per_instance
    samples_per_step = micro_batch * accumulate * num_samples
    max_epochs = C.MAX_EPOCHS
    val_data_size = C.VAL_DATA_SIZE
    val_batch_size = C.VAL_BATCH_SIZE
    num_augment = C.NUM_AUGMENT
    workers = C.DATALOADER_WORKERS
    max_steps = C.TOTAL_STEPS

    if args.smoke:
        train_instances = 64
        partials_per_instance = 4
        effective_batch_size = 64
        micro_batch = 16
        accumulate = 1
        rows_per_epoch = train_instances * partials_per_instance
        samples_per_step = micro_batch * accumulate * num_samples
        max_epochs = 1
        max_steps = -1
        val_data_size = 64
        val_batch_size = 64
        num_augment = 1
        workers = 0
    else:
        C.check_step_math(
            label="lehd_rl",
            rows_per_epoch=rows_per_epoch,
            micro_batch=micro_batch,
            accumulate=accumulate,
            samples_per_step=samples_per_step,
        )

    if args.max_epochs is not None:
        max_epochs = args.max_epochs

    label = "lehd_rl" if baseline == "teacher" else "lehd_rl_shared"
    run_name = C.run_name(
        label,
        seed=C.SEED,
        examples=partials_per_instance,
        data="hgs",
    )
    if args.smoke:
        run_name = f"{run_name}-smoke"

    checkpoint_dir = C.checkpoint_dir(run_name)
    resume_path = args.ckpt_path
    if resume_path is None and args.resume:
        if args.smoke:
            raise ValueError("--smoke runs write no checkpoints")
        resume_path = str(checkpoint_dir / "last.ckpt")
    if resume_path is not None:
        resume_path = str(Path(resume_path).expanduser())
        if not Path(resume_path).is_file():
            raise FileNotFoundError(f"no checkpoint at {resume_path}")

    seed_everything(C.SEED, workers=True)

    env = LEHDStepRLCVRPEnv(
        data_path=C.DATA_PATH,
        val_data_path=None,
        num_loc=C.NUM_LOC,
        capacity=C.CAPACITY,
        val_instances=C.VAL_INSTANCES,
        seed=C.SEED,
        instances=train_instances,
        partials_per_instance=partials_per_instance,
        effective_batch_size=effective_batch_size,
    )

    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        env._epoch = int(checkpoint["epoch"]) + 1
        del checkpoint

    model = LEHDPOMORL(
        env,
        batch_size=micro_batch,
        val_batch_size=val_batch_size,
        train_data_size=train_instances,
        val_data_size=val_data_size,
        optimizer_kwargs={"lr": C.LR, "weight_decay": C.WEIGHT_DECAY},
        num_augment=num_augment,
        advantage_baseline=baseline,
        num_samples=num_samples,
        shuffle_train_dataloader=False,
        dataloader_num_workers=workers,
    )

    print(f"Arm: {baseline} | K={num_samples}")
    print(
        f"Samples/step: {samples_per_step:,} | steps/epoch: {C.STEPS_PER_EPOCH} | "
        f"epochs: {max_epochs} | total steps: {max_steps if max_steps > 0 else 'smoke'}"
    )
    print(
        f"Rows/epoch: {rows_per_epoch:,} "
        f"({train_instances:,} instances x {partials_per_instance} partials)"
    )
    print(
        f"Micro-batch: {micro_batch} partials x {accumulate} accumulation x K={num_samples} "
        f"= {samples_per_step:,} rollouts/step; w per {effective_batch_size} partials"
    )

    callbacks = [RichModelSummary(max_depth=3)]
    if not args.smoke:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="epoch_{epoch:03d}",
                auto_insert_metric_name=False,
                save_top_k=1,
                save_last=True,
                monitor="val/reward",
                mode="max",
            )
        )

    logger = (
        False
        if args.smoke
        else SwanLabLogger(project="cvrp-train-pomo", experiment_name=run_name)
    )
    trainer = RL4COTrainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        precision=C.PRECISION if torch.cuda.is_available() else "32-true",
        accumulate_grad_batches=accumulate,
        gradient_clip_val=C.GRADIENT_CLIP_VAL,
        enable_checkpointing=not args.smoke,
    )
    if resume_path is None:
        trainer.fit(model)
    else:
        Trainer.fit(trainer, model, ckpt_path=resume_path, weights_only=False)


if __name__ == "__main__":
    main()
