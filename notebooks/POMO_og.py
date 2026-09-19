"""POMO trained by REINFORCE, validated on the supervised model's held-out set.

Training is unchanged from POMO's original setup: REINFORCE with the shared baseline on 100,000
instances sampled fresh from rl4co's CVRP generator every epoch. What is new is the validation.
Instead of being scored on another 10,000 generated instances, the model is now scored on the
same fixed subset of ``data/CVRP`` that :mod:`notebooks.POMO_sl` validates on, and reports the
same three metrics -- ``val/reward``, ``val/reward_greedy`` and ``val/gap_ref`` -- so the two
training curves can be read against each other in wandb. Both come from
:class:`~src.models.components.pomo_inference.POMOInferenceMixin`, so they cannot drift apart.

Two knobs have to agree with ``POMO_sl.py`` for the subsets to be the same draw rather than
merely the same size: ``VAL_DATA_SIZE``, from which ``val_instances`` is derived, and the data
file itself. ``VAL_BATCH_SIZE`` does not change *which* instances are validated but does change
the epoch mean -- Lightning weights its epoch reduction by batch size -- so it is matched to the
supervised run too.

Run it with the project's virtualenv::

    .venv/bin/python notebooks/POMO_og.py

or, equivalently, ``uv run notebooks/POMO_og.py``. Data paths are resolved against the repository
root, so the working directory does not matter.

Everything below is wrapped in ``main()`` behind a ``__name__`` guard on purpose. Importing rl4co
pulls in torchrl, which sets the global multiprocessing start method to ``spawn``; under spawn
each DataLoader worker re-imports this module, so any unguarded module-level setup would run once
per worker and spawn workers recursively.
"""

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning.pytorch.loggers import WandbLogger

from rl4co.utils.trainer import RL4COTrainer

from src.envs import RLCVRPEnv, project_root
from src.models import POMORL

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
# Must match POMO_sl.py: the file, and VAL_DATA_SIZE (which sets how many instances are held
# out), are what make the two validation sets the same subset rather than two similar ones.
DATA_PATH = "data/CVRP/vrp100_hgs_train_100w-001.txt"
VAL_DATA_PATH = None

NUM_LOC = 100
CAPACITY = 50.0

# Training keeps POMO's original REINFORCE settings, so the RL curve stays the RL curve.
LR = 1e-4
WEIGHT_DECAY = 1e-6
BATCH_SIZE = 64
TRAIN_DATA_SIZE = 100_000
VAL_BATCH_SIZE = 256  # matches POMO_sl.py, so the epoch mean is weighted identically
VAL_DATA_SIZE = 10_000
MAX_EPOCHS = 200

# Bounds the log-likelihood sum from below in bf16. rl4co's `16-mixed` default underflows the
# log-softmax on long sequences and trips `get_log_likelihood`'s `logprobs > -1000` assert;
# POMO_sl.py runs bf16 for the same reason, so the two are also numerically alike here.
PRECISION = "bf16-mixed"

# Identifies this run in both wandb and the checkpoint directory, so the two can always be
# matched up. The `rl-` prefix is what keeps it apart from POMO_sl.py's `sl-` runs: the two
# scripts share a checkpoint root and a wandb project. It names the settings most likely to be
# varied; widen it if you start sweeping something it does not mention.
RUN_NAME = f"rl-vrp{NUM_LOC}-lr-{LR}-wd-{WEIGHT_DECAY}-B-{BATCH_SIZE}"


def main() -> None:
    """Build the environment, model and trainer, then fit."""
    env = RLCVRPEnv(
        data_path=DATA_PATH,
        val_data_path=VAL_DATA_PATH,
        num_loc=NUM_LOC,
        capacity=CAPACITY,
        # The held-out range is carved from the tail of the file, so this is what fixes which
        # instances the validation subset is drawn from. It mirrors POMO_sl.py exactly.
        val_instances=VAL_DATA_SIZE // 10,
    )

    model = POMORL(
        env,
        batch_size=BATCH_SIZE,
        val_batch_size=VAL_BATCH_SIZE,
        train_data_size=TRAIN_DATA_SIZE,
        val_data_size=VAL_DATA_SIZE,
        optimizer_kwargs={"lr": LR, "weight_decay": WEIGHT_DECAY},
    )

    # `num_trajectories("train")` would report the file's pool, which this run never touches:
    # training instances come from the generator.
    print(f"train: {TRAIN_DATA_SIZE:,} generated instances per epoch, {MAX_EPOCHS} epochs")
    print(f"val:   {env.num_trajectories('val'):,} held-out trajectories (POMO SL's subset)")

    # Checkpoint on the augmented, multi-start reward, exactly as POMO_sl.py does, so the two
    # runs' "best" checkpoints are selected by the same criterion. `val/reward_greedy` is the
    # better signal for judging progress early on.
    #
    # Each run owns a directory named after itself, so configurations that differ in any of
    # the hyperparameters above cannot overwrite each other's `last.ckpt` and best-epoch file.
    # Re-running one identical configuration does overwrite, which is what a rerun usually
    # means; add a suffix to RUN_NAME if both should be kept.
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

    # Named to pair with POMO_sl.py's `sl-vrp100-...` runs in the same project, so the two
    # curves can be selected together; the same name is the checkpoint directory.
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
