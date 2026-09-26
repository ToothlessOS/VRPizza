"""Supervised POMO trained on the same per-epoch draw as the synced REINFORCE run.

:mod:`notebooks.POMO_sl` draws its epoch as ``size`` slots sampled uniformly from the whole
``sum(2R)`` pool. Because an instance owns only ~21 of the 21M slots, a 100,000-slot epoch
lands on ~95,300 instances at ~1.05 starts each, so an instance's other optimal starts are
never seen alongside each other -- which is the opposite of what
:class:`~src.envs.dataset.CVRPSolutionDataset` documents ("an instance with ``R = 11``
contributes 22 addressable training examples instead of one").

This script uses the draw those comments describe: ``TRAIN_INSTANCES`` instances, each
contributing ``STARTS_PER_INSTANCE`` of its ``2R`` feasible starts. That is the same draw
:mod:`notebooks.POMO_rl_synced` trains on, batch for batch, so the two runs differ only in
the objective. See :mod:`src.envs.cvrp_sl_synced` and ``docs/RL_SL_data_pipeline.md``.

Two consequences to expect. The same 100,000 rows now cover 6,250 instances at 16 starts
each rather than 95,300 at ~1, so this run sees far less of the *file* per epoch than
POMO_sl did; raising `MAX_EPOCHS` is the lever, since coverage goes as
``1 - exp(-(total_rows/K)/999000)``. And because each instance contributes 16 correlated
rows, an epoch's effective sample size is lower than its row count suggests. Both apply
equally to the synced RL run, which is the comparison this exists for.

Validation is deliberately unchanged, so ``val/reward`` stays directly comparable with the
existing ``sl-`` and ``rl-`` runs as well as the synced ``rl-synced-`` one.

Run it with the project's virtualenv::

    .venv/bin/python notebooks/POMO_sl_synced.py

or, equivalently, ``uv run notebooks/POMO_sl_synced.py``. Data paths are resolved against
the repository root, so the working directory does not matter.

Everything below is wrapped in ``main()`` behind a ``__name__`` guard on purpose. Importing
rl4co pulls in torchrl, which sets the global multiprocessing start method to ``spawn``;
under spawn each DataLoader worker re-imports this module, so any unguarded module-level
setup would run once per worker and spawn workers recursively.
"""

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning.pytorch.loggers import WandbLogger

from rl4co.utils.trainer import RL4COTrainer

from src.envs import SyncedCVRPEnv, project_root
from src.models import POMOSL

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
# Must match POMO_rl_synced.py: the file, the held-out size, and the three draw constants
# below are what make the two training sets the same draw rather than two similar ones.
DATA_PATH = "data/CVRP/vrp100_hgs_train_100w-001.txt"
VAL_DATA_PATH = None

NUM_LOC = 100
CAPACITY = 50.0

# The shared draw; see POMO_rl_synced.py for why K is 16 rather than something smaller.
# Rows per epoch are TRAIN_INSTANCES * STARTS_PER_INSTANCE = 100,000, the budget POMO_sl.py
# already works in -- so the epoch costs the same, it just spends it differently.
TRAIN_INSTANCES = 6_250
STARTS_PER_INSTANCE = 16

# Instances per step. The dataloader batches *rows*, so one step is these 64 instances' 16
# starts each -- the same 1024 (instance, start) pairs the RL run puts in one step.
INSTANCES_PER_BATCH = 64
BATCH_SIZE = INSTANCES_PER_BATCH * STARTS_PER_INSTANCE

LR = 1e-4
WEIGHT_DECAY = 1e-6
VAL_BATCH_SIZE = 256  # matches the other runs, so the epoch mean is weighted identically
VAL_DATA_SIZE = 10_000
MAX_EPOCHS = 200

DATALOADER_WORKERS = 8

# Identifies this run in both wandb and the checkpoint directory. The `-synced` infix keeps
# it apart from POMO_sl.py's `sl-` runs, and pairs it with `rl-synced-` for selection.
RUN_NAME = (
    f"sl-synced-vrp{NUM_LOC}-lr-{LR}-wd-{WEIGHT_DECAY}"
    f"-B-{INSTANCES_PER_BATCH}-K-{STARTS_PER_INSTANCE}"
)


def main() -> None:
    """Build the environment, model and trainer, then fit."""
    env = SyncedCVRPEnv(
        data_path=DATA_PATH,
        val_data_path=VAL_DATA_PATH,
        num_loc=NUM_LOC,
        capacity=CAPACITY,
        val_instances=VAL_DATA_SIZE // 10,
        starts_per_instance=STARTS_PER_INSTANCE,
    )

    model = POMOSL(
        env,
        # `batch_size` counts rows here -- trajectories -- while `train_data_size` counts
        # instances, since the environment interprets that as how many instances to draw and
        # expands each to STARTS_PER_INSTANCE rows.
        batch_size=BATCH_SIZE,
        val_batch_size=VAL_BATCH_SIZE,
        train_data_size=TRAIN_INSTANCES,
        val_data_size=VAL_DATA_SIZE,
        optimizer_kwargs={"lr": LR, "weight_decay": WEIGHT_DECAY},
        # Off, and this is load-bearing. The dataset indexes rows instance-major so that an
        # instance's 16 trajectories are contiguous; shuffling at row level would scatter
        # them and put ~1024 distinct instances in a step instead of 64, silently breaking
        # the pairing with the RL run. Row-level shuffling would not change the loss value
        # at all -- cross-entropy is a mean over rows -- so nothing would look wrong; only
        # the effective sample size, and so the gradient variance, would differ. The
        # randomisation this gives up comes from the draw, which reorders instances every
        # epoch.
        shuffle_train_dataloader=False,
        dataloader_num_workers=DATALOADER_WORKERS,
    )

    rows = TRAIN_INSTANCES * STARTS_PER_INSTANCE
    print(
        f"train: {TRAIN_INSTANCES:,} instances x {STARTS_PER_INSTANCE} label-feasible starts "
        f"= {rows:,} trajectories per epoch, {MAX_EPOCHS} epochs"
    )
    print(
        f"       {INSTANCES_PER_BATCH} instances per step "
        f"({BATCH_SIZE} trajectories), {-(-TRAIN_INSTANCES // INSTANCES_PER_BATCH)} steps per epoch"
    )
    print(f"val:   {env.num_trajectories('val'):,} held-out trajectories (the shared subset)")

    # Checkpoint on the augmented, multi-start reward, exactly as the other runs do, so all
    # these runs' "best" checkpoints are selected by the same criterion.
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(project_root()) / "checkpoints" / RUN_NAME,
        filename="epoch_{epoch:03d}",
        # Without this Lightning prepends the monitored metric to the name it builds from
        # `{epoch}`, and the file lands as `epoch_epoch=000.ckpt`.
        auto_insert_metric_name=False,
        save_top_k=1,
        save_last=True,
        monitor="val/reward",
        mode="max",
    )

    rich_model_summary = RichModelSummary(max_depth=3)
    callbacks = [checkpoint_callback, rich_model_summary]

    logger = WandbLogger(
        project="cvrp-train-pomo",
        name=RUN_NAME,
    )

    trainer = RL4COTrainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        # bf16 rather than the 16-mixed default: `get_log_likelihood` asserts that log-probs
        # stay above -1000, and fp16 log-softmax underflow trips it.
        precision="bf16-mixed",
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
