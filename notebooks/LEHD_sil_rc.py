"""Strict LEHD-backbone local-reconstruction SIL: CVRP20 warm-up, then CVRP100 SIL.

This is :mod:`notebooks.POMO_sil_rc` with the policy swapped for the reference
LEHD "light encoder / heavy decoder" (``src/models/components/lehd_policy.py``).
The data, schedules and label cycles are unchanged:

* warm-up: POMO REINFORCE on generated CVRP20, 100 epochs x 50,000 instances,
  batch 64, Adam 1e-4;
* SIL: 5 cycles x 20 epochs over a fixed 1,000,000 generated CVRP100 instances.
  Each cycle improves every label by 100 RRC rounds with the best validation
  policy so far, then trains on teacher-forced sub-paths. The LEHD module scores
  the reference's exact 2N action sequence (see ``src/models/sil_rc_lehd.py``).

The heavy decoder re-runs six self-attention layers at every decode step, so the
default micro-batch is smaller than the POMO arm's while the effective batch
stays 1,024 (``128 x 8``), and the decoder stack is gradient-checkpointed. The
validation rollout defaults to one start (the reference's greedy protocol) and
eight dihedral augmentations; ``--val-starts``/``--val-augment`` expose the POMO
protocol for an explicit comparison.

``--policy-checkpoint`` warm-starts from either a repo Lightning checkpoint or
the raw reference ``model_state_dict``; the weights are architecture-compatible,
so this is a cheap alternative to the RL warm-up.

The optional chunked schedule (``--subcycles S``) splits each outer cycle into
S chunks: repair one chunk with the best policy, train on it for
``--subcycle-epochs`` epochs, then move to the next. ``S=1`` is the reference;
the default per-chunk epochs keep the total repair work and training rows per
outer cycle unchanged. Validation runs every epoch, so a chunked run validates
S times more often; ``--val-starts``/``--val-augment`` can reduce that cost.

Run it with the project's virtualenv::

    .venv/bin/python notebooks/LEHD_sil_rc.py --smoke
    .venv/bin/python notebooks/LEHD_sil_rc.py
    .venv/bin/python notebooks/LEHD_sil_rc.py --labels 100000
    .venv/bin/python notebooks/LEHD_sil_rc.py --policy-checkpoint checkpoints/.../last.ckpt
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
from src.models import PhaseAccumulationCallback, SILRCLEHDLitModule

# --------------------------------------------------------------------------------------
# Configuration (reference CVRP100 comment + paper warm-up)
# --------------------------------------------------------------------------------------
DATA_PATH = "data/CVRP/vrp100_hgs_train_100w-001.txt"
VAL_DATA_PATH = None

NUM_LOC = 100
CAPACITY = 50.0
WARMUP_NUM_LOC = 20
WARMUP_CAPACITY = 30.0

# Changed based on observation that 20 epochs will do (based on validation gap)
WARMUP_EPOCHS = 20
WARMUP_BATCH_SIZE = 64
WARMUP_DATA_SIZE = 50_000

SIL_CYCLES = 10
SIL_EPOCHS_PER_CYCLE = 20
SIL_INSTANCES = 128_000
SUBCYCLES_PER_CYCLE = 5
RECON_ITERATIONS = 100
REPAIR_BATCH = 256

#: Effective SIL batch is 1,024, as in the reference; the heavy decoder makes a
#: 1,024-row micro-batch infeasible, so the split is a memory knob only.
SIL_MICRO_BATCH = 256
SIL_ACCUM = 4
PARTIALS_PER_INSTANCE = 1

LR = 1e-4
WEIGHT_DECAY = 0.0
LR_GAMMA = 0.97
GRADIENT_CLIP_VAL = 10.0

VAL_BATCH_SIZE = 64
VAL_DATA_SIZE = 10_000
NUM_AUGMENT = 8
#: The reference's greedy protocol: one rollout, best over augmentations.
VAL_STARTS = 1
EXACT_CE = True
GRADIENT_CHECKPOINTING = True

LABEL_SEED = 0
SEED = 0


def parse_args() -> argparse.Namespace:
    """Parse the schedule overrides, label bootstrap and resume/policy options."""
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
        help=f"POMO starts for validation (default {VAL_STARTS}); 1 is the "
        "reference's greedy protocol",
    )
    parser.add_argument(
        "--val-augment",
        type=int,
        default=None,
        help=f"dihedral augmentations for validation (default {NUM_AUGMENT})",
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
        "--policy-checkpoint",
        default=None,
        help="repo Lightning checkpoint or raw reference model_state_dict to warm-start",
    )
    parser.add_argument("--resume", action="store_true", help="resume from last.ckpt")
    parser.add_argument(
        "--ckpt-path", default=None, help="explicit checkpoint to resume from"
    )
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


def _load_policy_checkpoint(model, path: str) -> None:
    """Load a repo Lightning state dict or a raw reference ``model_state_dict``.

    The reference checkpoint's keys are ``encoder.*``/``decoder.*``; the
    Lightning model's are ``policy.encoder.*``/``policy.decoder.*``. Both are
    accepted, and a POMO checkpoint is rejected before it can silently load
    nothing.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))

    if any(key.startswith("policy.encoder.embedding.") for key in state):
        wrapped = state
    elif any(key.startswith("encoder.embedding.") for key in state):
        wrapped = {f"policy.{key}": value for key, value in state.items()}
    else:
        raise ValueError(
            f"{path} does not contain an LEHD policy (expected policy.encoder.embedding.* "
            "or encoder.embedding.*); a POMO checkpoint cannot warm-start this model"
        )

    missing, unexpected = model.load_state_dict(wrapped, strict=False)
    print(
        f"policy weights: {path} (missing={len(missing)}, unexpected={len(unexpected)})"
    )


def main() -> None:
    """Build the environment, model and trainer, then fit."""
    args = parse_args()
    if args.resume and args.ckpt_path is not None:
        raise ValueError("pass either --resume or --ckpt-path, not both")
    if (
        args.resume or args.ckpt_path is not None
    ) and args.policy_checkpoint is not None:
        raise ValueError(
            "--policy-checkpoint warm-starts from weights only; --resume/--ckpt-path "
            "continue the full training state. Pass only one"
        )
    if args.from_scratch and args.policy_checkpoint is not None:
        raise ValueError(
            "--from-scratch and --policy-checkpoint are mutually exclusive"
        )

    # -- schedule and bootstrap --------------------------------------------------
    warmup_epochs = (
        0 if (args.from_scratch or args.policy_checkpoint) else WARMUP_EPOCHS
    )
    init = args.init
    if init is None:
        init = "random_insertion" if args.from_scratch else "greedy"
    if args.improve_first_cycle and args.no_improve_first_cycle:
        raise ValueError(
            "pass either --improve-first-cycle or --no-improve-first-cycle"
        )
    if args.improve_first_cycle:
        improve_first_cycle = True
    elif args.no_improve_first_cycle:
        improve_first_cycle = False
    else:
        improve_first_cycle = init == "greedy"

    labels = args.labels if args.labels is not None else SIL_INSTANCES
    subcycles = args.subcycles if args.subcycles is not None else SUBCYCLES_PER_CYCLE
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
    sil_micro = (
        args.sil_micro_batch if args.sil_micro_batch is not None else SIL_MICRO_BATCH
    )
    sil_accum = args.sil_accum if args.sil_accum is not None else SIL_ACCUM
    sil_effective = sil_micro * sil_accum
    if sil_effective <= 0:
        raise ValueError("the effective SIL batch must be positive")
    val_starts = args.val_starts if args.val_starts is not None else VAL_STARTS
    val_augment = args.val_augment if args.val_augment is not None else NUM_AUGMENT
    exact_ce = not args.no_exact_ce and EXACT_CE
    gradient_checkpointing = (
        not args.no_gradient_checkpointing and GRADIENT_CHECKPOINTING
    )

    sil_epochs = SIL_CYCLES * subcycles * subcycle_epochs
    max_epochs = warmup_epochs + sil_epochs
    if args.max_epochs is not None:
        max_epochs = args.max_epochs

    run_name = (
        f"lehd_sil_rc-strict-vrp{NUM_LOC}-n{labels}-c{SIL_CYCLES}x{SIL_EPOCHS_PER_CYCLE}"
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
    precision = "bf16-mixed"
    if args.smoke:
        warmup_epochs = (
            0 if (args.from_scratch or args.policy_checkpoint is not None) else 1
        )
        labels = 6
        subcycles = min(subcycles, labels)
        subcycle_epochs = 1
        chunk_size = (labels + subcycles - 1) // subcycles
        recon_iterations = 1
        repair_batch = 2
        cycle_epochs = 1
        max_epochs = warmup_epochs + subcycles * cycle_epochs
        warmup_batch = 4
        warmup_data = 8
        sil_micro = 2
        sil_accum = 2
        sil_effective = sil_micro * sil_accum
        val_batch_size = 2
        val_data_size = 4
        val_augment = 1
        val_starts = 1
        precision = "32-true"
        run_name = f"{run_name}-smoke"
    if args.precision is not None:
        precision = args.precision
    if args.val_batch_size is not None:
        val_batch_size = args.val_batch_size
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

    model = SILRCLEHDLitModule(
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
        num_augment=val_augment,
        shuffle_train_dataloader=False,
        dataloader_num_workers=0,
        exact_ce=exact_ce,
        inference_num_starts=val_starts,
        gradient_checkpointing=gradient_checkpointing,
    )

    if args.policy_checkpoint is not None:
        _load_policy_checkpoint(model, args.policy_checkpoint)

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
        f"improve_first_cycle={improve_first_cycle}, exact_ce={exact_ce}, "
        f"gradient_checkpointing={gradient_checkpointing}"
    )
    print(
        f"validation: {val_starts} start(s) x {val_augment} augmentation(s), "
        f"{val_data_size:,} slots, batch {val_batch_size}"
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
