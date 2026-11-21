"""Strict port of local-reconstruction SIL: CVRP20 RL warm-up, then CVRP100 SIL.

The pipeline is the reference/SIT schedule:

* warm-up: POMO REINFORCE on generated CVRP20, 100 epochs x 50,000 instances,
  batch 64, Adam 1e-4 -- the paper's 15.42 h stage;
* SIL: 5 cycles x 20 epochs over a fixed 1,000,000 generated CVRP100 instances.
  Each cycle first improves every label by 100 RRC rounds with the best validation
  policy so far (the reference's ``repair_all_data``), then trains the policy by
  teacher-forced CE on random sub-paths of those labels. The SIL optimizer is
  Adam 1e-4 with no weight decay, gradient clip 10, and ``MultiStepLR(gamma=0.97)``
  stepped once per SIL epoch. The warm-up LR stays constant.

The 1,000,000-instance default is the reference's commented CVRP100 entry
(``100: [1000000, 1024, 1024, 5]``); use ``--labels`` for a smaller first run or
``--smoke`` for a pipeline check. The label store is cached at
``<checkpoint_dir>/sil_rc_labels.npz`` so a resume does not redraw it.

The optional chunked schedule (``--subcycles S``) splits each outer cycle into
S chunks: it repairs only the current chunk with the best validation policy,
trains on it for ``--subcycle-epochs`` epochs, then moves to the next chunk.
``S=1`` is the whole-store reference; the default per-chunk epoch count keeps
the total repair work and training rows per outer cycle unchanged. The POMO
validation is expensive, so a chunked run can pass ``--val-starts 1`` and/or
``--val-augment 1`` to reduce the per-epoch cost.

Initial labels follow the reference's two paths:

* ``--init greedy`` (default when a warm-up is present): one model greedy rollout
  per instance, then the first cycle's 100-round repair;
* ``--init random_insertion`` (the default with ``--from-scratch``): the
  deterministic polar-angle cheapest insertion, and the from-scratch reference
  skips the first repair (``RI_initial=True``'s ``budget=0``).

``--warmup-checkpoint`` starts directly in SIL from weights saved by another run
and sets the SIL budget to the reference's 100 epochs. ``--resume`` / ``--ckpt-path``
continue the same run, optimizer and label cycle.

Run it with the project's virtualenv::

    .venv/bin/python notebooks/POMO_sil_rc.py --smoke
    .venv/bin/python notebooks/POMO_sil_rc.py --init greedy
    .venv/bin/python notebooks/POMO_sil_rc.py --from-scratch --init random_insertion
    .venv/bin/python notebooks/POMO_sil_rc.py --warmup-checkpoint checkpoints/.../last.ckpt
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

from src.envs import SILRCCVRPEnv, project_root
from src.models import PhaseAccumulationCallback, SILRCPOMOLitModule

# --------------------------------------------------------------------------------------
# Configuration (reference CVRP100 comment + paper warm-up)
# --------------------------------------------------------------------------------------
DATA_PATH = "data/CVRP/vrp100_hgs_train_100w-001.txt"
VAL_DATA_PATH = None

NUM_LOC = 100
CAPACITY = 50.0
WARMUP_NUM_LOC = 20
WARMUP_CAPACITY = 30.0

WARMUP_EPOCHS = 100
WARMUP_BATCH_SIZE = 64
WARMUP_DATA_SIZE = 50_000

SIL_CYCLES = 5
SIL_EPOCHS_PER_CYCLE = 20
SIL_INSTANCES = 1_000_000
SUBCYCLES_PER_CYCLE = 1
RECON_ITERATIONS = 100
REPAIR_BATCH = 1024

SIL_MICRO_BATCH = 1024
SIL_ACCUM = 1
PARTIALS_PER_INSTANCE = 1

LR = 1e-4
WEIGHT_DECAY = 0.0
LR_GAMMA = 0.97
GRADIENT_CLIP_VAL = 10.0

VAL_BATCH_SIZE = 256
VAL_DATA_SIZE = 10_000
NUM_AUGMENT = 8
#: ``None`` keeps POMO's N starts; a chunked run can pass ``--val-starts 1``.
VAL_STARTS = None
LABEL_SEED = 0
SEED = 0


def parse_args() -> argparse.Namespace:
    """Parse the schedule overrides, label bootstrap and resume/warm-up options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run (few labels/epochs, no SwanLab logging) for pipeline checks",
    )
    parser.add_argument(
        "--init",
        choices=("greedy", "random_insertion"),
        default=None,
        help="initial label bootstrap; defaults to greedy with a warm-up, "
        "random_insertion with --from-scratch",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="skip the warm-up and use the reference's RI_initial=True path",
    )
    parser.add_argument(
        "--improve-first-cycle",
        action="store_true",
        help="repair the initial labels before the first cycle's training",
    )
    parser.add_argument(
        "--no-improve-first-cycle",
        action="store_true",
        help="skip the first repair (the from-scratch reference behavior)",
    )
    parser.add_argument(
        "--labels",
        type=int,
        default=None,
        help=f"instances in the fixed label set (default {SIL_INSTANCES:,})",
    )
    parser.add_argument(
        "--subcycles",
        type=int,
        default=None,
        help=f"chunks per outer SIL cycle (default {SUBCYCLES_PER_CYCLE}); "
        "repair one chunk, train on it, then move to the next",
    )
    parser.add_argument(
        "--subcycle-epochs",
        type=int,
        default=None,
        help=f"Lightning epochs per chunk (default {SIL_EPOCHS_PER_CYCLE}); use "
        f"{SIL_EPOCHS_PER_CYCLE} // --subcycles to keep the old total epoch count",
    )
    parser.add_argument(
        "--recon-iterations",
        type=int,
        default=None,
        help=f"RRC rounds per instance per cycle (default {RECON_ITERATIONS})",
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
        help="POMO starts for validation (default: the env's N starts); "
        "--val-starts 1 is useful for a chunked run",
    )
    parser.add_argument(
        "--val-augment",
        type=int,
        default=None,
        help=f"dihedral augmentations for validation (default {NUM_AUGMENT})",
    )
    parser.add_argument(
        "--warmup-checkpoint",
        default=None,
        help="Lightning checkpoint whose policy weights start the SIL phase; skips warm-up",
    )
    parser.add_argument("--resume", action="store_true", help="resume from last.ckpt")
    parser.add_argument("--ckpt-path", default=None, help="explicit checkpoint to resume from")
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="override the total epoch budget (not additional epochs)",
    )
    parser.add_argument(
        "--run-suffix",
        default="",
        help="appended to the run name/checkpoint directory, e.g. for a shared warm-up",
    )
    return parser.parse_args()


def main() -> None:
    """Build the environment, model and trainer, then fit."""
    args = parse_args()
    if args.resume and args.ckpt_path is not None:
        raise ValueError("pass either --resume or --ckpt-path, not both")
    if (args.resume or args.ckpt_path is not None) and args.warmup_checkpoint is not None:
        raise ValueError(
            "--warmup-checkpoint warm-starts from weights only; --resume/--ckpt-path "
            "continue the full training state. Pass only one"
        )
    if args.from_scratch and args.warmup_checkpoint is not None:
        raise ValueError("--from-scratch and --warmup-checkpoint are mutually exclusive")

    # -- schedule and bootstrap --------------------------------------------------
    warmup_epochs = 0 if (args.from_scratch or args.warmup_checkpoint) else WARMUP_EPOCHS
    init = args.init
    if init is None:
        init = "random_insertion" if args.from_scratch else "greedy"
    if args.improve_first_cycle and args.no_improve_first_cycle:
        raise ValueError("pass either --improve-first-cycle or --no-improve-first-cycle")
    if args.improve_first_cycle:
        improve_first_cycle = True
    elif args.no_improve_first_cycle:
        improve_first_cycle = False
    else:
        improve_first_cycle = init == "greedy"

    labels = args.labels if args.labels is not None else SIL_INSTANCES
    subcycles = (
        args.subcycles if args.subcycles is not None else SUBCYCLES_PER_CYCLE
    )
    subcycle_epochs = (
        args.subcycle_epochs
        if args.subcycle_epochs is not None
        else SIL_EPOCHS_PER_CYCLE
    )
    if subcycles <= 0:
        raise ValueError("--subcycles must be positive")
    if subcycles > labels:
        raise ValueError(f"--subcycles {subcycles} exceeds --labels {labels}")
    if subcycle_epochs <= 0:
        raise ValueError("--subcycle-epochs must be positive")
    chunk_size = (labels + subcycles - 1) // subcycles
    recon_iterations = (
        args.recon_iterations if args.recon_iterations is not None else RECON_ITERATIONS
    )
    sil_micro = args.sil_micro_batch if args.sil_micro_batch is not None else SIL_MICRO_BATCH
    sil_accum = args.sil_accum if args.sil_accum is not None else SIL_ACCUM
    sil_effective = sil_micro * sil_accum
    if sil_effective % sil_micro != 0:
        raise AssertionError("effective SIL batch must be a multiple of the micro-batch")
    val_starts = args.val_starts if args.val_starts is not None else VAL_STARTS
    val_augment = args.val_augment if args.val_augment is not None else NUM_AUGMENT
    # The strict arm deliberately allows a short final accumulation group (1M labels
    # / 1024 is not integral, matching the reference's full-dataset epoch). Only the
    # step-matched arm, whose step count is the comparison unit, requires exactness.

    sil_epochs = SIL_CYCLES * subcycles * subcycle_epochs
    max_epochs = warmup_epochs + sil_epochs
    if args.max_epochs is not None:
        max_epochs = args.max_epochs

    run_name = (
        f"sil-rc-strict-vrp{NUM_LOC}-n{labels}-c{SIL_CYCLES}x{SIL_EPOCHS_PER_CYCLE}"
        f"-r{recon_iterations}-{init}-seed{SEED}"
    )
    if subcycles > 1:
        run_name = f"{run_name}-s{subcycles}"
    if subcycle_epochs != SIL_EPOCHS_PER_CYCLE:
        run_name = f"{run_name}-e{subcycle_epochs}"

    # -- smoke overrides ---------------------------------------------------------
    warmup_batch = WARMUP_BATCH_SIZE
    warmup_data = WARMUP_DATA_SIZE
    cycle_epochs = SIL_EPOCHS_PER_CYCLE
    repair_batch = REPAIR_BATCH
    val_batch_size = VAL_BATCH_SIZE
    val_data_size = VAL_DATA_SIZE
    num_augment = val_augment
    precision = "bf16-mixed"
    if args.smoke:
        warmup_epochs = (
            0 if (args.from_scratch or args.warmup_checkpoint is not None) else 1
        )
        labels = 6
        subcycles = min(subcycles, labels)
        subcycle_epochs = 1
        chunk_size = (labels + subcycles - 1) // subcycles
        recon_iterations = 1
        repair_batch = 3
        cycle_epochs = 1
        max_epochs = warmup_epochs + subcycles * cycle_epochs
        warmup_batch = 4
        warmup_data = 8
        sil_micro = 2
        sil_accum = 3
        sil_effective = sil_micro * sil_accum
        val_batch_size = 2
        val_data_size = 4
        val_augment = 1
        num_augment = 1
        precision = "32-true"
        run_name = f"{run_name}-smoke"
    if args.run_suffix:
        run_name = f"{run_name}-{args.run_suffix}"

    # The full run holds out 1,000 instances for a 10,000-slot validation draw; a
    # smoke that shrinks the draw must still hold out at least one instance, or the
    # validation dataloader is empty and no best-policy snapshot is taken.
    val_instances = max(1, val_data_size // 10)
    checkpoint_dir = Path(project_root()) / "checkpoints" / run_name
    label_cache_path = checkpoint_dir / "sil_rc_labels.npz"
    resume_path = args.ckpt_path
    if resume_path is None and args.resume:
        resume_path = str(checkpoint_dir / "last.ckpt")
    if resume_path is not None:
        resume_path = str(Path(resume_path).expanduser())
        if not Path(resume_path).is_file():
            raise FileNotFoundError(f"no checkpoint at {resume_path}")

    # The warm-up uses no accumulation; SIL uses `sil_accum`. Lightning 2.6 does not
    # take an epoch-keyed dict, so a callback switches the value at each phase.
    accum_callback = PhaseAccumulationCallback(warmup_epochs, 1, sil_accum)

    seed_everything(SEED, workers=True)

    env = SILRCCVRPEnv(
        data_path=DATA_PATH,
        val_data_path=VAL_DATA_PATH,
        num_loc=NUM_LOC,
        capacity=CAPACITY,
        val_instances=val_instances,
        seed=SEED,
        warmup_num_loc=WARMUP_NUM_LOC,
        warmup_capacity=WARMUP_CAPACITY,
    )

    model = SILRCPOMOLitModule(
        env,
        warmup_epochs=warmup_epochs,
        warmup_batch_size=warmup_batch,
        warmup_data_size=warmup_data,
        cycle_epochs=cycle_epochs,
        labels_per_cycle=labels,
        subcycles_per_cycle=subcycles,
        subcycle_epochs=subcycle_epochs,
        fresh_labels_per_cycle=False,
        init=init,
        improve_first_cycle=improve_first_cycle,
        recon_iterations=recon_iterations,
        repair_batch=repair_batch,
        w_min=4,
        w_max=None,
        partials_per_instance=PARTIALS_PER_INSTANCE,
        sil_batch_size=sil_micro,
        sil_effective_batch_size=sil_effective,
        lr_gamma=LR_GAMMA,
        label_seed=LABEL_SEED,
        label_cache_path=str(label_cache_path),
        generation_selection_metric="val/reward_greedy",
        batch_size=warmup_batch,
        train_data_size=warmup_data,
        val_batch_size=val_batch_size,
        val_data_size=val_data_size,
        optimizer_kwargs={"lr": LR, "weight_decay": WEIGHT_DECAY},
        num_augment=num_augment,
        shuffle_train_dataloader=False,
        dataloader_num_workers=0,
    )

    if val_starts is not None:
        model.inference_num_starts = val_starts

    if args.warmup_checkpoint is not None:
        checkpoint = torch.load(
            args.warmup_checkpoint, map_location="cpu", weights_only=False
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"warm-up weights: {args.warmup_checkpoint} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    print(
        f"warm-up: {warmup_epochs} epochs x {warmup_data:,} CVRP{WARMUP_NUM_LOC} instances, "
        f"batch {warmup_batch}"
    )
    print(
        f"SIL:     {max_epochs - warmup_epochs} epochs, {SIL_CYCLES} x "
        f"{subcycles} sub-cycles x {subcycle_epochs}-epoch chunks, "
        f"{labels:,} fixed CVRP{NUM_LOC} labels ({chunk_size:,}/chunk), "
        f"{recon_iterations} RRC rounds/outer"
    )
    print(
        f"         {sil_micro} partials x {sil_accum} accumulation = "
        f"{sil_effective:,} partials/step; init={init}, "
        f"improve_first_cycle={improve_first_cycle}, val_starts={val_starts}, "
        f"val_augment={val_augment}"
    )

    callbacks = [accum_callback, RichModelSummary(max_depth=3)]
    if not args.smoke:
        callbacks.insert(
            0,
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="epoch_{epoch:03d}",
                auto_insert_metric_name=False,
                save_top_k=1,
                save_last=True,
                monitor="val/reward",
                mode="max",
            ),
        )

    logger = (
        False
        if args.smoke
        else SwanLabLogger(project="cvrp-train-pomo", experiment_name=run_name)
    )
    trainer = RL4COTrainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        precision=precision,
        accumulate_grad_batches=1,
        gradient_clip_val=GRADIENT_CLIP_VAL,
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
