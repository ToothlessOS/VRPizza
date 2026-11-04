"""POMO trained by an RL warm-up followed by self-improvement learning (SIL) on CVRP.

The run has two phases in one ``trainer.fit``:

* epochs ``0 .. WARMUP_EPOCHS-1`` are ordinary POMO REINFORCE on instances from rl4co's
  ``CVRPGenerator`` -- the same loop as ``notebooks/POMO_og.py``;
* from ``WARMUP_EPOCHS`` on, each epoch generates pseudo-experts by searching freshly
  generated instances with the current (or best) policy, then trains by teacher-forced
  cross-entropy on their ``2R`` legal starts. The generator is unchanged, so SIL is trained on
  the same instance distribution as the warm-up.

Two samplers are compared: ``--search-type beam_search`` (deterministic beam search) and
``--search-type wor`` (stochastic beam search sampling without replacement across rounds).
The beam arm uses ``BEAM_WIDTH * NUM_ROUNDS`` as its width so both arms see the same number
of candidate trajectories per instance.

Validation is the shared POMO rollout on the same held-out HGS subset as the ``rl-``, ``sl-``
and ``lehd-`` runs, so the curves are directly comparable.

One consequence of an RL warm-up deserves a note: POMO forces the first customer at every
multi-start rollout and does not score that action, so the policy's depot-start first-action
distribution is effectively untrained when SIL begins. The search starts at the depot (as
Gumbeldore's does), and SIL's teacher forcing scores the first action, so the first SIL epochs
are also where that distribution is learned. A supervised warm-up checkpoint would remove the
effect, which is what ``--warmup-checkpoint`` is for.

Run it with the project's virtualenv::

    .venv/bin/python notebooks/POMO_sil.py --search-type wor
    .venv/bin/python notebooks/POMO_sil.py --search-type beam_search
    .venv/bin/python notebooks/POMO_sil.py --search-type wor --smoke

``--warmup-checkpoint PATH`` loads a completed run's ``state_dict`` and starts directly in
SIL (``WARMUP_EPOCHS`` is then treated as 0). Use it to give both sampling arms identical
warm-up weights instead of two independent RL warm-ups.

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

from src.envs import SILCVRPEnv, project_root
from src.models import POMOSIL
from src.sil import SILSearchConfig

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
# The HGS file is the source of the held-out validation subset only; both training phases use
# generated instances. Keep it identical to the other scripts so the validation draw matches.
DATA_PATH = "data/CVRP/vrp100_hgs_train_100w-001.txt"
VAL_DATA_PATH = None

NUM_LOC = 100
CAPACITY = 50.0

# Phase lengths. The SIL phase is MAX_EPOCHS - WARMUP_EPOCHS epochs.
MAX_EPOCHS = 200
WARMUP_EPOCHS = 50

LR = 1e-4
WEIGHT_DECAY = 1e-6

# Warm-up batch: POMO_og's 64 generated instances, 100 starts each. SIL batch: trajectories.
WARMUP_BATCH_SIZE = 64
TRAIN_DATA_SIZE = 100_000  # warm-up instances per epoch
SIL_BATCH_SIZE = 256
SIL_BATCHES_PER_EPOCH = 100

VAL_BATCH_SIZE = 256
VAL_DATA_SIZE = 10_000

# Sampler. WOR uses `BEAM_WIDTH` per round for `NUM_ROUNDS` rounds; plain beam uses the same
# total budget in a single round, so the arms differ only in how they sample.
#
# Generation cost measured on the workstation GPU (32 instances, N=100, chunk 32):
# WOR k=32 x 4 -> ~57 s, beam k=128 x 1 -> ~23 s, i.e. roughly 30 min / 12 min per 1000
# instances per SIL epoch. `NUM_INSTANCES_TO_GENERATE`, `BEAM_WIDTH` and `NUM_ROUNDS` are the
# first things to lower if the SIL phase is too slow; generation, not the CE training, is the
# bottleneck (~100 training steps take well under a minute).
SEARCH_TYPE = "wor"
BEAM_WIDTH = 32
NUM_ROUNDS = 4
NUM_INSTANCES_TO_GENERATE = 1_000
SEARCH_CHUNK_SIZE = 32  # decoder rows per chunk are at most chunk_size * beam_width
SEARCH_SEED = 0

# Gumbeldore's SIL loop: generate with the best model so far, and keep appending generated
# data while the model is not improving.
USE_BEST_MODEL_FOR_GENERATION = True
APPEND_IF_NOT_NEW_BEST = True
MAX_SIL_INSTANCES = 20_000

# `val/reward_greedy` is a single-start rollout like the search; `val/reward` is a max over
# 800 rollouts and remains the checkpoint monitor for comparability with the other runs.
GENERATION_SELECTION_METRIC = "val/reward_greedy"

RUN_STEM = f"vrp{NUM_LOC}-lr-{LR}-wd-{WEIGHT_DECAY}"


def parse_args() -> argparse.Namespace:
    """Parse the search arm, the optional smoke overrides and the warm-up checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--search-type",
        choices=("wor", "beam_search"),
        default=SEARCH_TYPE,
        help="pseudo-expert sampler: WOR-SBS or deterministic beam search",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run (few epochs and instances, no wandb) for pipeline checks",
    )
    parser.add_argument(
        "--warmup-checkpoint",
        default=None,
        help="Lightning checkpoint whose policy weights start the SIL phase; skips warm-up",
    )
    parser.add_argument(
        "--run-suffix",
        default="",
        help="appended to the run name and checkpoint directory, e.g. to separate ablations",
    )
    return parser.parse_args()


def main() -> None:
    """Build the environment, model and trainer, then fit."""
    args = parse_args()

    search_type = args.search_type
    max_epochs = MAX_EPOCHS
    warmup_epochs = WARMUP_EPOCHS if args.warmup_checkpoint is None else 0
    warmup_batch_size = WARMUP_BATCH_SIZE
    train_data_size = TRAIN_DATA_SIZE
    sil_batch_size = SIL_BATCH_SIZE
    sil_batches_per_epoch = SIL_BATCHES_PER_EPOCH
    val_batch_size = VAL_BATCH_SIZE
    val_data_size = VAL_DATA_SIZE
    num_instances = NUM_INSTANCES_TO_GENERATE
    base_beam_width = BEAM_WIDTH
    base_num_rounds = NUM_ROUNDS
    search_chunk_size = SEARCH_CHUNK_SIZE
    num_augment = 8
    precision = "bf16-mixed"
    # The run name keys on the CLI choice, not the module default, so the two arms cannot
    # write into the same checkpoint directory.
    run_name = f"sil-{search_type}-{RUN_STEM}"

    if args.warmup_checkpoint is not None and warmup_epochs != 0:
        raise AssertionError("a warm-up checkpoint implies starting directly in SIL")

    if args.smoke:
        max_epochs = 2
        warmup_epochs = 1 if args.warmup_checkpoint is None else 0
        warmup_batch_size = 16
        train_data_size = 64
        sil_batch_size = 8
        sil_batches_per_epoch = 1
        val_batch_size = 10
        val_data_size = 20
        num_instances = 4
        base_beam_width = 2
        base_num_rounds = 2
        search_chunk_size = 2
        num_augment = 1
        precision = "32-true"
        run_name = f"{run_name}-smoke"

    # The beam arm gets the same candidate budget (width x rounds) in a single deterministic
    # pass, so the two arms differ only in how they sample.
    beam_width = base_beam_width
    num_rounds = base_num_rounds
    if search_type == "beam_search":
        beam_width = base_beam_width * base_num_rounds
        num_rounds = 1

    run_name = (
        f"{run_name}-k{beam_width}x{num_rounds}-n{num_instances}"
        f"-S{sil_batch_size}x{sil_batches_per_epoch}-w{warmup_epochs}"
    )
    if args.run_suffix:
        run_name = f"{run_name}-{args.run_suffix}"

    env = SILCVRPEnv(
        data_path=DATA_PATH,
        val_data_path=VAL_DATA_PATH,
        num_loc=NUM_LOC,
        capacity=CAPACITY,
        # The held-out range is carved from the tail of the file; this is what fixes the
        # validation instances and mirrors the other training scripts.
        val_instances=val_data_size // 10,
    )

    model = POMOSIL(
        env,
        warmup_epochs=warmup_epochs,
        batch_size=warmup_batch_size,
        train_data_size=train_data_size,
        val_batch_size=val_batch_size,
        val_data_size=val_data_size,
        optimizer_kwargs={"lr": LR, "weight_decay": WEIGHT_DECAY},
        num_augment=num_augment,
        sil_batch_size=sil_batch_size,
        sil_batches_per_epoch=sil_batches_per_epoch,
        num_instances_to_generate=num_instances,
        search_config=SILSearchConfig(
            search_type=search_type,
            beam_width=beam_width,
            num_rounds=num_rounds,
            chunk_size=search_chunk_size,
            seed=SEARCH_SEED,
        ),
        use_best_model_for_generation=USE_BEST_MODEL_FOR_GENERATION,
        append_if_not_new_best=APPEND_IF_NOT_NEW_BEST,
        max_sil_instances=MAX_SIL_INSTANCES,
        generation_selection_metric=GENERATION_SELECTION_METRIC,
        generation_seed=SEARCH_SEED,
    )

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
        f"warm-up: {warmup_epochs} epochs x {train_data_size:,} generated instances, "
        f"batch {warmup_batch_size}"
    )
    print(
        f"SIL:     {max_epochs - warmup_epochs} epochs, {num_instances:,} generated instances "
        f"per epoch, {search_type} k={beam_width} x {num_rounds}, "
        f"{sil_batches_per_epoch} x {sil_batch_size} trajectories"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(project_root()) / "checkpoints" / run_name,
        filename="epoch_{epoch:03d}",
        auto_insert_metric_name=False,
        save_top_k=1,
        save_last=True,
        monitor="val/reward",
        mode="max",
    )
    callbacks = [checkpoint_callback, RichModelSummary(max_depth=3)]

    logger = (
        False
        if args.smoke
        else WandbLogger(project="cvrp-train-pomo", name=run_name)
    )

    trainer = RL4COTrainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        precision=precision,
    )

    trainer.fit(model)
    print(
        f"best {GENERATION_SELECTION_METRIC}: {model._best_metric}; "
        f"generation snapshot: {'yes' if model._best_policy_state else 'no'}"
    )


if __name__ == "__main__":
    main()
