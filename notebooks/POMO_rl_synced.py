"""POMO trained by REINFORCE on the supervised file, with the multi-start bounded by its labels.

:mod:`notebooks.POMO_og` trains POMO as originally specified: instances sampled fresh from
rl4co's generator every epoch, and a multi-start that begins from all ``N = 100`` customers.
Both of those are confounds against :mod:`notebooks.POMO_sl` -- the instances come from a
different distribution than the HGS file, and only ~21 of the 100 starts begin with a depot
edge the reference solution contains.

This script removes them. Training draws from the same file, the same ``train_range`` and
the same per-epoch seed scheme as the supervised run, and each instance's multi-start uses
``STARTS_PER_INSTANCE`` of its ``2R`` label-feasible starts rather than every customer. The
two pipelines then see the identical ``(instance, start)`` pairs each epoch, in the same
per-batch grouping, and differ only in the objective -- teacher forcing against REINFORCE.
See :mod:`src.envs.cvrp_rl_synced` and ``docs/RL_SL_data_pipeline.md``.

Validation is deliberately unchanged: it is shared code
(:class:`~src.models.components.pomo_inference.POMOInferenceMixin`), so ``val/reward``,
``val/reward_greedy`` and ``val/gap_ref`` stay on the same scale as the existing ``sl-``,
``rl-`` and synced ``sl-`` runs.

Run it with the project's virtualenv::

    .venv/bin/python notebooks/POMO_rl_synced.py

or, equivalently, ``uv run notebooks/POMO_rl_synced.py``. Data paths are resolved against the
repository root, so the working directory does not matter.

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

from src.envs import SyncedRLCVRPEnv, project_root
from src.models import POMORL

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
# Must match POMO_sl_synced.py: the file, the held-out size, and the three draw constants
# below are what make the two training sets the same draw rather than two similar ones.
DATA_PATH = "data/CVRP/vrp100_hgs_train_100w-001.txt"
VAL_DATA_PATH = None

NUM_LOC = 100
CAPACITY = 50.0

# The shared draw. `TRAIN_INSTANCES * STARTS_PER_INSTANCE = 100_000` rows per epoch, the
# same budget the existing supervised run works in, so nothing about the amount of data
# changes -- only how it is distributed over instances. K=16 is the largest count every
# instance can supply, since the smallest 2R in the file is 16; POMO's shared baseline
# needs more than one start, so K cannot go to 1 and SL has to come up to K rather than
# RL dropping to SL's ~1.
TRAIN_INSTANCES = 6_250
STARTS_PER_INSTANCE = 16

# Instances per step. SL feeds the same 64 instances' 16 starts each as 1024 rows, so both
# pipelines cover the identical 64 instances per step; only the loss differs.
INSTANCES_PER_BATCH = 64

LR = 1e-4
WEIGHT_DECAY = 1e-6
VAL_BATCH_SIZE = 256  # matches the supervised runs, so the epoch mean is weighted identically
VAL_DATA_SIZE = 10_000
MAX_EPOCHS = 200

# Not 8, which is what the single-GPU supervised script uses. rl4co forces the `spawn` start
# method, so each worker is a fresh interpreter that re-imports torch and rl4co -- measured at
# ~790 MB before it holds any data, against a 10 MB dataset. Two concurrent trainings at 8
# workers each come to ~12.6 GB of workers alone, which does not fit the 16 GB a `-c 16`
# `--mem-per-cpu=1G` job gets, and the resulting kill surfaces as a truncated pickle in a
# worker rather than as anything resembling an out-of-memory error. See scripts/submit_sync.sh.
#
# Kept equal to `POMO_sl_synced.py`'s value for symmetry, though unlike the draw constants
# above it has no bearing on the data: the draw comes from the dataset's seed, not from the
# worker count, so changing this moves throughput and nothing else.
DATALOADER_WORKERS = 4

# Bounds the log-likelihood sum from below in bf16; see POMO_og.py. The supervised side runs
# bf16 for the same reason, so the two are also numerically alike.
PRECISION = "bf16-mixed"

# Names the settings most likely to be varied, so two configurations cannot share a
# checkpoint directory by accident. The `-synced` infix keeps it apart from POMO_og.py's
# `rl-` runs and the supervised side's `sl-synced-` ones, which are the runs it is meant to
# be read against.
RUN_NAME = (
    f"rl-synced-vrp{NUM_LOC}-lr-{LR}-wd-{WEIGHT_DECAY}"
    f"-B-{INSTANCES_PER_BATCH}-K-{STARTS_PER_INSTANCE}"
)


def main() -> None:
    """Build the environment, model and trainer, then fit."""
    env = SyncedRLCVRPEnv(
        data_path=DATA_PATH,
        val_data_path=VAL_DATA_PATH,
        num_loc=NUM_LOC,
        capacity=CAPACITY,
        # Carves the held-out range out of the tail of the file, exactly as the supervised
        # runs do, which is what keeps the validation subset the same one.
        val_instances=VAL_DATA_SIZE // 10,
        starts_per_instance=STARTS_PER_INSTANCE,
    )

    model = POMORL(
        env,
        # `batch_size` counts instances here, not rows: the multi-start expands each to
        # STARTS_PER_INSTANCE rollouts inside the decoder. `num_starts` is left unset so the
        # environment decides -- STARTS_PER_INSTANCE while training, and POMO's N while
        # validating, since the validation batch carries no start nodes.
        batch_size=INSTANCES_PER_BATCH,
        val_batch_size=VAL_BATCH_SIZE,
        train_data_size=TRAIN_INSTANCES,
        val_data_size=VAL_DATA_SIZE,
        optimizer_kwargs={"lr": LR, "weight_decay": WEIGHT_DECAY},
        dataloader_num_workers=DATALOADER_WORKERS,
    )

    print(
        f"train: {TRAIN_INSTANCES:,} instances x {STARTS_PER_INSTANCE} label-feasible starts "
        f"= {TRAIN_INSTANCES * STARTS_PER_INSTANCE:,} rollouts per epoch, {MAX_EPOCHS} epochs"
    )
    print(
        f"       {INSTANCES_PER_BATCH} instances per step "
        f"({INSTANCES_PER_BATCH * STARTS_PER_INSTANCE} rollouts), "
        f"{-(-TRAIN_INSTANCES // INSTANCES_PER_BATCH)} steps per epoch"
    )
    print(f"val:   {env.num_trajectories('val'):,} held-out trajectories (SL's own subset)")

    # Checkpoint on the augmented, multi-start reward, exactly as the other runs do, so the
    # three curves' "best" checkpoints are selected by the same criterion. As in POMO_og.py
    # the validation multi-start is POMO's full N, so this stays comparable.
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
        precision=PRECISION,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
