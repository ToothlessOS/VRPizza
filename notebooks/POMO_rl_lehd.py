"""POMO trained by REINFORCE on LEHD partial solutions.

The dataset and state are LEHD's (``docs/LEHD_SL.md``); the objective is REINFORCE on the
sampled reconstruction, as described in ``docs/LEHD_RL.md``. Two arms are available:

* ``teacher``: one rollout per partial, baseline = the HGS window's reduced cost;
* ``shared``: K >= 2 rollouts per partial, baseline = their mean cost. This arm never reads
  the teacher actions or the reference cost.

Run it with the project's virtualenv::

    .venv/bin/python notebooks/POMO_rl_lehd.py                     # teacher, K=1
    .venv/bin/python notebooks/POMO_rl_lehd.py --baseline shared    # label-free, K=4
    .venv/bin/python notebooks/POMO_rl_lehd.py --smoke              # tiny pipeline check
    .venv/bin/python notebooks/POMO_rl_lehd.py --pilot              # 100k partials, 3 epochs
    .venv/bin/python notebooks/POMO_rl_lehd.py --resume             # continue last.ckpt

``--resume`` (or an explicit ``--ckpt-path``) continues a run through
``trainer.fit(ckpt_path=...)``, restoring weights, optimizer, scheduler, epoch and the
checkpoint callback's best score. ``--max-epochs`` overrides the total budget, so extending a
finished run means passing a larger value. Pass the same CLI options as the original run so
``run_name`` -- and therefore the checkpoint directory -- matches.

Everything below is wrapped in ``main()`` behind a ``__name__`` guard because importing rl4co
pulls in torchrl, which sets the global multiprocessing start method to ``spawn``; under spawn
each DataLoader worker re-imports this module.

Two batching details are inherited from the LEHD SL path and are load-bearing:

* the optimizer batch is 1024 (LEHD's), the DataLoader batch 128, bridged by gradient
  accumulation, so the dataset's one-``w``-per-effective-batch draw stays constant across the
  eight microbatches;
* row shuffling is off, because that draw is keyed by ``index // effective_batch_size``.
"""

import argparse

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pathlib import Path

import torch

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from swanlab.integration.pytorch_lightning import SwanLabLogger

from rl4co.utils.trainer import RL4COTrainer

from src.envs import LEHDRLCVRPEnv, project_root
from src.models import LEHDPOMORL

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
DATA_PATH = "data/CVRP/vrp100_hgs_train_100w-001.txt"
VAL_DATA_PATH = None

NUM_LOC = 100
CAPACITY = 50.0

# LEHD's batch size, split into microbatches for the card. The effective batch keys the
# one-w-per-batch draw.
EFFECTIVE_BATCH_SIZE = 1024
MICRO_BATCH_SIZE = 128
ACCUMULATE_GRAD_BATCHES = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE

# One partial per training instance per epoch, as in LEHD SL.
TRAIN_PARTIALS = 999_000
MAX_EPOCHS = 40

# LEHD's optimizer/schedule, so the comparison with the SL run isolates the objective.
LR = 1e-4
WEIGHT_DECAY = 0.0
LR_GAMMA = 0.9

VAL_BATCH_SIZE = 256
VAL_DATA_SIZE = 10_000
NUM_AUGMENT = 8

# Every worker is a fresh interpreter under spawn and re-imports torch and rl4co (~790 MB);
# lower this first if the job is OOM-killed, not the model batch.
DATALOADER_WORKERS = 8

# RL's default clipping; LEHD SL passes None. Recorded here as part of the recipe.
GRADIENT_CLIP_VAL = 1.0

# The baseline arm. `teacher` uses the HGS window cost as a control variate (one rollout);
# `shared` compares K sampled reconstructions of the same window and never reads the labels.
ADVANTAGE_BASELINE = "teacher"


def parse_args() -> argparse.Namespace:
    """Parse the arm, sample count and smoke overrides."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--baseline",
        choices=("teacher", "shared"),
        default=ADVANTAGE_BASELINE,
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
        help="tiny run (4096 partials, one epoch, no SwanLab logging) for pipeline checks",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="short run (100k partials, three epochs, no SwanLab/checkpoint) to watch the signal",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from <checkpoints>/<run_name>/last.ckpt; pass the same CLI options as "
        "the original run so run_name (and the checkpoint directory) match",
    )
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="explicit checkpoint to resume from; overrides --resume",
    )
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
        num_samples = 1 if baseline == "teacher" else 4

    train_partials = TRAIN_PARTIALS
    max_epochs = MAX_EPOCHS
    val_data_size = VAL_DATA_SIZE
    val_batch_size = VAL_BATCH_SIZE
    dataloader_workers = DATALOADER_WORKERS
    num_augment = NUM_AUGMENT
    # The checkpoint label is the text before the first "-"; keep the two arms in separate
    # directories so evaluation can discover them side by side.
    label = "lehd_rl" if baseline == "teacher" else "lehd_rl_shared"
    run_name = (
        f"{label}-vrp{NUM_LOC}-lr-{LR}-wd-{WEIGHT_DECAY}"
        f"-B-{EFFECTIVE_BATCH_SIZE}-{baseline}-K-{num_samples}-w-4-{NUM_LOC}"
    )

    if args.smoke:
        train_partials = 4_096
        max_epochs = 1
        val_data_size = 512
        val_batch_size = 64
        dataloader_workers = 0
        num_augment = 1
        # The arm is whatever was selected, so the smoke exercises the same loss path a real
        # run would; only the data and validation sizes are reduced.
        run_name = f"{run_name}-smoke"
    elif args.pilot:
        train_partials = 100_000
        max_epochs = 3
        val_data_size = 1_024
        val_batch_size = 128
        dataloader_workers = 4
        num_augment = 1
        run_name = f"{run_name}-pilot"

    if args.max_epochs is not None:
        max_epochs = args.max_epochs

    if args.resume and args.ckpt_path is not None:
        raise ValueError("pass either --resume or --ckpt-path, not both")

    # All resumed state (weights, optimizer, scheduler, epoch, callback best score) comes from
    # the `ckpt_path` handed to `trainer.fit`. `last.ckpt` is the epoch-end save, so it
    # continues after the last completed epoch; the best checkpoint would repeat every epoch
    # after the best one.
    checkpoint_dir = Path(project_root()) / "checkpoints" / run_name
    resume_path = args.ckpt_path
    if resume_path is None and args.resume:
        if args.smoke or args.pilot:
            raise ValueError(
                "--smoke/--pilot runs write no checkpoints; pass --ckpt-path explicitly if "
                "you mean to resume from one"
            )
        resume_path = str(checkpoint_dir / "last.ckpt")
    if resume_path is not None:
        resume_path = str(Path(resume_path).expanduser())
        if not Path(resume_path).is_file():
            raise FileNotFoundError(f"no checkpoint at {resume_path}")

    if baseline == "shared" and num_samples < 2:
        raise ValueError(
            "--baseline shared needs --num-samples >= 2; with one sample the shared "
            "advantage is identically zero"
        )
    if EFFECTIVE_BATCH_SIZE % MICRO_BATCH_SIZE != 0:
        raise ValueError(
            f"effective batch {EFFECTIVE_BATCH_SIZE} is not a multiple of the microbatch "
            f"{MICRO_BATCH_SIZE}; a microbatch would straddle two w draws"
        )

    env = LEHDRLCVRPEnv(
        data_path=DATA_PATH,
        val_data_path=VAL_DATA_PATH,
        num_loc=NUM_LOC,
        capacity=CAPACITY,
        val_instances=val_data_size // 10,
        effective_batch_size=EFFECTIVE_BATCH_SIZE,
    )

    if resume_path is not None:
        # `setup()` rebuilds the first epoch's partial dataset before Lightning restores
        # `current_epoch`, and LEHDCVRPEnv keys that draw on this counter, which is not in the
        # checkpoint either. Re-anchor it so the resumed run continues the `seed + epoch`
        # sequence instead of replaying the permutations of epochs 0..N.
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        env._epoch = int(checkpoint["epoch"]) + 1
        del checkpoint  # Lightning loads the file again; don't hold two copies

    model = LEHDPOMORL(
        env,
        # `batch_size` counts partial examples in a microbatch; `train_data_size` counts
        # partials per epoch and is passed to the environment's dataset.
        batch_size=MICRO_BATCH_SIZE,
        val_batch_size=val_batch_size,
        train_data_size=train_partials,
        val_data_size=val_data_size,
        optimizer_kwargs={"lr": LR, "weight_decay": WEIGHT_DECAY},
        lr_scheduler="MultiStepLR",
        lr_scheduler_kwargs={
            "milestones": list(range(1, max_epochs)),
            "gamma": LR_GAMMA,
        },
        lr_scheduler_interval="epoch",
        num_augment=num_augment,
        advantage_baseline=baseline,
        num_samples=num_samples,
        # Load-bearing: the dataset's one-w-per-effective-batch draw assumes sequential row
        # batching. Randomisation comes from the epoch draw, not row shuffling.
        shuffle_train_dataloader=False,
        dataloader_num_workers=dataloader_workers,
    )

    chunks = -(-train_partials // EFFECTIVE_BATCH_SIZE)
    print(
        f"train: {train_partials:,} partials per epoch, {max_epochs} epochs, "
        f"{chunks:,} w draws, baseline={baseline}, K={num_samples}"
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
    quiet = args.smoke or args.pilot
    if not quiet:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch_{epoch:03d}",
            auto_insert_metric_name=False,
            save_top_k=1,
            save_last=True,
            monitor="val/reward",
            mode="max",
        )
        callbacks.append(checkpoint_callback)
    # Smoke and pilot runs write no checkpoint, so they cannot shadow a real run under the
    # same label.

    logger = (
        False
        if quiet
        else SwanLabLogger(project="cvrp-train-pomo", experiment_name=run_name)
    )
    cuda = torch.cuda.is_available()
    trainer = RL4COTrainer(
        max_epochs=max_epochs,
        accelerator="gpu" if cuda else "cpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        precision="bf16-mixed" if cuda else "32-true",
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        enable_checkpointing=not quiet,
    )
    if resume_path is None:
        trainer.fit(model)
    else:
        # `RL4COTrainer.fit` predates Lightning's `weights_only` argument and does not forward
        # it. Its only addition is clearing the gradient clip for manual-optimization models,
        # which does not apply here, so call the base implementation for the resume.
        # `weights_only=False`: our own checkpoints embed the training env in
        # `hyper_parameters`, which torch 2.6's restricted unpickler refuses (same trust
        # boundary as `--warmup-checkpoint` in the SIL driver).
        Trainer.fit(trainer, model, ckpt_path=resume_path, weights_only=False)


if __name__ == "__main__":
    main()
