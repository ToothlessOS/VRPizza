"""Step-matched LEHD-backbone local-reconstruction SIL: 2,000 sequences per step.

This is :mod:`notebooks.POMO_sil_rc_step` with the policy swapped for the
reference LEHD "light encoder / heavy decoder". The protocol is unchanged: 200
epochs x 100 optimizer steps, the first 50 epochs an RL warm-up on generated
CVRP20 and the remaining 150 a SIL phase of five 30-epoch label cycles. Each
cycle generates 12,500 fresh CVRP100 instances, bootstraps greedy labels,
improves them with 100 RRC rounds per instance, then trains on 12,500 x 16 =
200,000 partial windows per epoch.

The heavy decoder makes a 500-row micro-batch infeasible, so the default split
is ``125 x 16`` -- still exactly 2,000 partials per optimizer step. The decoder
stack is gradient-checkpointed, and validation defaults to one start (the
reference's greedy protocol) with eight augmentations.

Run it with the project's virtualenv::

    .venv/bin/python notebooks/LEHD_sil_rc_step.py --smoke
    .venv/bin/python notebooks/LEHD_sil_rc_step.py
    .venv/bin/python notebooks/LEHD_sil_rc_step.py --resume
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
from src.models import PhaseAccumulationCallback, SILRCLEHDLitModule
from src.utils import comparison as C

RUN_NAME = C.run_name(
    "lehd_sil_rc_step",
    seed=C.SEED,
    examples=C.SIL_RC_PARTIALS_PER_INSTANCE,
    data="generated",
)

WARMUP_NUM_LOC = 20
WARMUP_CAPACITY = 30.0

#: 125 x 16 = 2,000 partials per optimizer step; the effective batch is unchanged.
SIL_MICRO_BATCH = 125
SIL_ACCUM = 16
REPAIR_BATCH = 125
VAL_BATCH_SIZE = 64
VAL_STARTS = 1
VAL_AUGMENT = C.NUM_AUGMENT
EXACT_CE = True
GRADIENT_CHECKPOINTING = True


def parse_args() -> argparse.Namespace:
    """Parse the optional resume, budget, memory and smoke overrides."""
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
        "--sil-micro-batch",
        type=int,
        default=None,
        help=f"partials per SIL micro-batch (default {SIL_MICRO_BATCH})",
    )
    parser.add_argument(
        "--sil-accum",
        type=int,
        default=None,
        help=f"SIL gradient accumulation (default {SIL_ACCUM})",
    )
    parser.add_argument(
        "--val-starts",
        type=int,
        default=None,
        help=f"POMO starts for validation (default {VAL_STARTS})",
    )
    parser.add_argument(
        "--val-augment",
        type=int,
        default=None,
        help=f"dihedral augmentations for validation (default {VAL_AUGMENT})",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help=f"validation batch (default {VAL_BATCH_SIZE}); lower it if "
        "--val-starts makes the rollout too large",
    )
    parser.add_argument(
        "--no-exact-ce",
        action="store_true",
        help="use the repo's N+1 partial CE instead of the reference 2N CE",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="disable per-step decoder checkpointing",
    )
    parser.add_argument(
        "--precision",
        default=None,
        help="Lightning precision override (default bf16-mixed; use 32-true if "
        "the un-normalised decoder is unstable in bf16)",
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
    max_epochs = args.max_epochs if args.max_epochs is not None else C.MAX_EPOCHS
    max_steps = (
        args.max_epochs * C.STEPS_PER_EPOCH if args.max_epochs is not None else C.TOTAL_STEPS
    )

    run_name = RUN_NAME
    warmup_epochs = C.SIL_RC_WARMUP_EPOCHS
    cycle_epochs = C.SIL_RC_CYCLE_EPOCHS
    labels = C.SIL_RC_INSTANCES_PER_CYCLE
    partials = C.SIL_RC_PARTIALS_PER_INSTANCE
    recon_iterations = C.SIL_RC_RECON_ITERATIONS
    repair_batch = REPAIR_BATCH
    warmup_micro = C.SIL_RC_WARMUP_MICRO_INSTANCES
    warmup_data = C.SIL_RC_WARMUP_INSTANCES_PER_EPOCH
    warmup_starts = C.SIL_RC_WARMUP_STARTS
    sil_micro = args.sil_micro_batch if args.sil_micro_batch is not None else SIL_MICRO_BATCH
    sil_accum = args.sil_accum if args.sil_accum is not None else SIL_ACCUM
    val_starts = args.val_starts if args.val_starts is not None else VAL_STARTS
    val_augment = args.val_augment if args.val_augment is not None else VAL_AUGMENT
    exact_ce = not args.no_exact_ce and EXACT_CE
    gradient_checkpointing = not args.no_gradient_checkpointing and GRADIENT_CHECKPOINTING
    val_instances = C.VAL_INSTANCES
    val_data_size = C.VAL_DATA_SIZE
    val_batch_size = VAL_BATCH_SIZE
    precision = C.PRECISION

    # The step protocol is only meaningful at the full sizes; smoke skips it.
    if not args.smoke:
        C.check_step_math(
            label="lehd_sil_rc_step-warmup",
            rows_per_epoch=warmup_data,
            micro_batch=warmup_micro,
            accumulate=1,
            samples_per_step=C.SIL_RC_WARMUP_SAMPLES_PER_STEP,
        )
        C.check_step_math(
            label="lehd_sil_rc_step-sil",
            rows_per_epoch=C.SIL_RC_ROWS_PER_EPOCH,
            micro_batch=sil_micro,
            accumulate=sil_accum,
            samples_per_step=C.SAMPLES_PER_STEP,
        )

    if args.smoke:
        max_epochs = 2
        max_steps = -1
        warmup_epochs = 1
        cycle_epochs = 1
        labels = 4
        partials = 2
        recon_iterations = 1
        repair_batch = 2
        warmup_micro = 4
        warmup_data = 8
        warmup_starts = 4
        sil_micro = 2
        sil_accum = 2
        val_starts = 1
        val_instances = 10
        val_data_size = 4
        val_batch_size = 2
        val_augment = 1
        precision = "32-true"
        run_name = f"{run_name}-smoke"
    if args.precision is not None:
        precision = args.precision
    if args.val_batch_size is not None:
        val_batch_size = args.val_batch_size
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

    model = SILRCLEHDLitModule(
        env,
        warmup_epochs=warmup_epochs,
        warmup_batch_size=warmup_micro,
        warmup_data_size=warmup_data,
        cycle_epochs=cycle_epochs,
        labels_per_cycle=labels,
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
        num_augment=val_augment,
        shuffle_train_dataloader=False,
        dataloader_num_workers=0,
        exact_ce=exact_ce,
        inference_num_starts=val_starts,
        gradient_checkpointing=gradient_checkpointing,
    )

    print(
        f"Warm-up: {warmup_epochs} epochs x {C.STEPS_PER_EPOCH} steps, "
        f"{warmup_micro} CVRP{WARMUP_NUM_LOC} instances x {warmup_starts} starts "
        f"= {C.SIL_RC_WARMUP_SAMPLES_PER_STEP:,} rollouts/step"
    )
    print(
        f"SIL:     {max_epochs - warmup_epochs} epochs, {labels:,} labels/cycle x "
        f"{cycle_epochs} epochs x {C.SIL_RC_CYCLES} cycles, {recon_iterations} RRC "
        f"rounds/cycle, {partials} partials/instance"
    )
    print(
        f"         {sil_micro} partials x {sil_accum} accumulation = "
        f"{sil_micro * sil_accum:,} partials/step; constant LR {C.LR}; "
        f"exact_ce={exact_ce}, gradient_checkpointing={gradient_checkpointing}"
    )
    print(
        f"Validation: {val_starts} start(s) x {val_augment} augmentation(s), "
        f"{val_data_size:,} slots, {val_batch_size} per batch"
    )

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
        f"labels cycle: {model._labels_cycle}"
    )


if __name__ == "__main__":
    main()
