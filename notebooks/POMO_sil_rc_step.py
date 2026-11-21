"""Step-matched local-reconstruction SIL: 2,000 decoded sequences per optimizer step.

The arm fits the comparison protocol's budget: 200 epochs x 100 steps, with the
first 50 epochs / 5,000 steps an RL warm-up on generated CVRP20 and the remaining
150 epochs / 15,000 steps the SIL phase. The SIL phase is five 30-epoch label
cycles; each cycle generates 12,500 fresh CVRP100 instances, bootstraps greedy
labels, improves them with 100 RRC rounds per instance, then trains on
12,500 instances x 16 partial windows = 200,000 rows per epoch. One optimizer step
is 500 partials x 4 accumulation = 2,000 partials.

The warm-up uses 125 generated CVRP20 instances x 16 starts = 2,000 rollouts per
step, so the same decoded-sequence unit holds in both phases. The accumulation
factor is epoch-keyed because the warm-up uses 1 and the SIL phase 4.

The label store is fresh every cycle (62,500 unique instances over five cycles)
and cached at ``<checkpoint_dir>/sil_rc_labels.npz``; a resume in the same cycle
reuses it, a resume at a cycle boundary regenerates the new cycle's set from its
seeded draw.

Run it with the project's virtualenv::

    .venv/bin/python notebooks/POMO_sil_rc_step.py --smoke
    .venv/bin/python notebooks/POMO_sil_rc_step.py
    .venv/bin/python notebooks/POMO_sil_rc_step.py --resume
"""

import argparse

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from swanlab.integration.pytorch_lightning import SwanLabLogger

from rl4co.utils.trainer import RL4COTrainer

from src.envs import SILRCStepCVRPEnv
from src.models import PhaseAccumulationCallback, SILRCPOMOLitModule
from src.utils import comparison as C

RUN_NAME = C.run_name(
    "sil_rc", seed=C.SEED, examples=C.SIL_RC_PARTIALS_PER_INSTANCE, data="generated"
)

WARMUP_NUM_LOC = 20
WARMUP_CAPACITY = 30.0


def parse_args() -> argparse.Namespace:
    """Parse the optional resume, budget and smoke overrides."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run (few labels/epochs, no SwanLab logging) for pipeline checks",
    )
    parser.add_argument("--resume", action="store_true", help="resume from last.ckpt")
    parser.add_argument("--ckpt-path", default=None, help="explicit checkpoint to resume from")
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="override MAX_EPOCHS (total budget, not additional epochs)",
    )
    parser.add_argument(
        "--subcycles",
        type=int,
        default=None,
        help="chunks per outer SIL cycle (default 1); must divide the 12,500 "
        "labels and keep 100 steps per whole-store epoch",
    )
    parser.add_argument(
        "--subcycle-epochs",
        type=int,
        default=None,
        help="Lightning epochs per chunk (default 30); use 30 // --subcycles to "
        "keep the old total epoch count",
    )
    parser.add_argument(
        "--run-suffix",
        default="",
        help="appended to the run name/checkpoint directory, e.g. for an ablation",
    )
    return parser.parse_args()


def main() -> None:
    """Build the environment, model and trainer, then fit."""
    args = parse_args()

    run_name = RUN_NAME
    warmup_epochs = C.SIL_RC_WARMUP_EPOCHS
    cycle_epochs = C.SIL_RC_CYCLE_EPOCHS
    labels = C.SIL_RC_INSTANCES_PER_CYCLE
    partials = C.SIL_RC_PARTIALS_PER_INSTANCE
    recon_iterations = C.SIL_RC_RECON_ITERATIONS
    repair_batch = C.SIL_RC_MICRO_BATCH
    warmup_micro = C.SIL_RC_WARMUP_MICRO_INSTANCES
    warmup_data = C.SIL_RC_WARMUP_INSTANCES_PER_EPOCH
    warmup_starts = C.SIL_RC_WARMUP_STARTS
    sil_micro = C.SIL_RC_MICRO_BATCH
    sil_accum = C.SIL_RC_ACCUM
    val_instances = C.VAL_INSTANCES
    val_data_size = C.VAL_DATA_SIZE
    val_batch_size = C.VAL_BATCH_SIZE
    num_augment = C.NUM_AUGMENT
    precision = C.PRECISION

    subcycles = args.subcycles if args.subcycles is not None else 1
    subcycle_epochs = (
        args.subcycle_epochs if args.subcycle_epochs is not None else cycle_epochs
    )
    if subcycles <= 0 or subcycle_epochs <= 0:
        raise ValueError("--subcycles and --subcycle-epochs must be positive")
    if subcycles > labels or labels % subcycles != 0:
        raise ValueError(
            f"--subcycles {subcycles} must divide the {labels:,} labels"
        )
    chunk_size = labels // subcycles
    chunk_rows = chunk_size * partials
    per_step = sil_micro * sil_accum
    if chunk_rows % per_step != 0:
        raise ValueError(
            f"chunk rows {chunk_rows:,} must be divisible by "
            f"micro*accum={per_step:,}"
        )
    steps_per_sil_epoch = chunk_rows // per_step
    if steps_per_sil_epoch * subcycles != C.STEPS_PER_EPOCH:
        raise ValueError(
            f"a chunk has {steps_per_sil_epoch} steps and {subcycles} chunks "
            f"give {steps_per_sil_epoch * subcycles} steps per whole-store "
            f"epoch, expected {C.STEPS_PER_EPOCH}"
        )
    max_epochs = warmup_epochs + C.SIL_RC_CYCLES * subcycles * subcycle_epochs
    if args.max_epochs is not None:
        max_epochs = args.max_epochs
    max_steps = (
        warmup_epochs * C.STEPS_PER_EPOCH
        + (max_epochs - warmup_epochs) * steps_per_sil_epoch
    )
    if subcycles > 1:
        run_name = f"{run_name}-s{subcycles}"
    if subcycle_epochs != cycle_epochs:
        run_name = f"{run_name}-e{subcycle_epochs}"

    # The step protocol is only meaningful at the full sizes; smoke skips it.
    if not args.smoke:
        C.check_step_math(
            label="sil_rc-warmup",
            rows_per_epoch=warmup_data,
            micro_batch=warmup_micro,
            accumulate=1,
            samples_per_step=C.SIL_RC_WARMUP_SAMPLES_PER_STEP,
        )
        C.check_step_math(
            label="sil_rc-sil",
            rows_per_epoch=chunk_rows,
            micro_batch=sil_micro,
            accumulate=sil_accum,
            samples_per_step=C.SAMPLES_PER_STEP,
            steps_per_epoch=steps_per_sil_epoch,
        )

    if args.smoke:
        max_epochs = 2
        max_steps = -1
        warmup_epochs = 1
        cycle_epochs = 1
        labels = 4
        partials = 2
        subcycles = min(subcycles, labels)
        subcycle_epochs = 1
        chunk_size = labels // subcycles
        chunk_rows = chunk_size * partials
        recon_iterations = 1
        repair_batch = 2
        warmup_micro = 4
        warmup_data = 8
        warmup_starts = 4
        sil_micro = 2
        sil_accum = 2
        per_step = sil_micro * sil_accum
        steps_per_sil_epoch = chunk_rows // per_step
        val_instances = 10
        val_data_size = 4
        val_batch_size = 2
        num_augment = 1
        precision = "32-true"
        run_name = f"{run_name}-smoke"
    if args.run_suffix:
        run_name = f"{run_name}-{args.run_suffix}"

    accum_callback = PhaseAccumulationCallback(warmup_epochs, 1, sil_accum)
    checkpoint_dir = C.checkpoint_dir(run_name)
    resume_path = C.resolve_resume(
        resume=args.resume, ckpt_path=args.ckpt_path, checkpoint_dir=checkpoint_dir
    )
    seed_everything(C.SEED, workers=True)

    env = SILRCStepCVRPEnv(
        data_path=C.DATA_PATH,
        val_data_path=None,
        num_loc=C.NUM_LOC,
        capacity=C.CAPACITY,
        val_instances=val_instances,
        seed=C.SEED,
        starts_per_instance=warmup_starts,
        warmup_num_loc=WARMUP_NUM_LOC,
        warmup_capacity=WARMUP_CAPACITY,
    )
    if resume_path is not None:
        C.reanchor_epoch(env, resume_path)

    model = SILRCPOMOLitModule(
        env,
        warmup_epochs=warmup_epochs,
        warmup_batch_size=warmup_micro,
        warmup_data_size=warmup_data,
        cycle_epochs=cycle_epochs,
        labels_per_cycle=labels,
        subcycles_per_cycle=subcycles,
        subcycle_epochs=subcycle_epochs,
        fresh_labels_per_cycle=True,
        init="greedy",
        improve_first_cycle=True,
        recon_iterations=recon_iterations,
        repair_batch=repair_batch,
        w_min=4,
        w_max=None,
        partials_per_instance=partials,
        sil_batch_size=sil_micro,
        sil_effective_batch_size=sil_micro * sil_accum,
        lr_gamma=1.0,
        label_seed=C.SEED,
        label_cache_path=str(checkpoint_dir / "sil_rc_labels.npz"),
        generation_selection_metric="val/reward_greedy",
        batch_size=warmup_micro,
        train_data_size=warmup_data,
        val_batch_size=val_batch_size,
        val_data_size=val_data_size,
        optimizer_kwargs={"lr": C.LR, "weight_decay": C.WEIGHT_DECAY},
        num_augment=num_augment,
        shuffle_train_dataloader=False,
        dataloader_num_workers=0,
    )

    print(
        f"Warm-up: {warmup_epochs} epochs x {C.STEPS_PER_EPOCH} steps, "
        f"{warmup_micro} CVRP{WARMUP_NUM_LOC} instances x {warmup_starts} starts "
        f"= {C.SIL_RC_WARMUP_SAMPLES_PER_STEP:,} rollouts/step"
    )
    print(
        f"SIL:     {max_epochs - warmup_epochs} epochs, {C.SIL_RC_CYCLES} x "
        f"{subcycles} sub-cycles x {subcycle_epochs}-epoch chunks, "
        f"{labels:,} labels/outer ({chunk_size:,}/chunk), {partials} partials/"
        f"instance, {recon_iterations} RRC rounds/outer"
    )
    print(
        f"         {sil_micro} partials x {sil_accum} accumulation = "
        f"{sil_micro * sil_accum:,} partials/step; "
        f"{steps_per_sil_epoch} steps/chunk-epoch x {subcycles} chunks = "
        f"{steps_per_sil_epoch * subcycles} steps/whole-store epoch; "
        f"total {max_steps:,} steps; constant LR {C.LR}"
    )
    print(f"Validation monitor: {val_data_size:,} slots, {val_batch_size} per batch")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch_{epoch:03d}",
        auto_insert_metric_name=False,
        save_top_k=1,
        save_last=True,
        monitor="val/reward",
        mode="max",
    )
    callbacks = [accum_callback, RichModelSummary(max_depth=3)]
    if not args.smoke:
        callbacks.insert(0, checkpoint_callback)

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
        precision=precision,
        accumulate_grad_batches=1,
        gradient_clip_val=C.GRADIENT_CLIP_VAL,
        num_sanity_val_steps=0,
    )
    if resume_path is None:
        trainer.fit(model)
    else:
        Trainer.fit(trainer, model, ckpt_path=resume_path, weights_only=False)

    print(
        f"best {model.generation_selection_metric}: {model._best_metric}; "
        f"labels cycle: {model._labels_cycle}, step: {model._labels_step}"
    )


if __name__ == "__main__":
    main()
