import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary

from rl4co.envs import CVRPEnv
from rl4co.models.zoo import POMO
from rl4co.utils.trainer import RL4COTrainer

import wandb
from lightning.pytorch.loggers import WandbLogger

# Params (POMO default)
lr = 1e-4
weight_decay = 1e-6
batch_size = 64
max_epochs = 200

# RL4CO env based on TorchRL
env = CVRPEnv(generator_params=dict(num_loc=100))
model = POMO(
    env,  # w/ default config - shared baseline; number_of_avail_action starting points; x8 augment
    train_data_size=100_000,
    val_data_size=10_000,
    optimizer_kwargs={"lr": lr, "weight_decay": weight_decay},
    batch_size=batch_size,
)


# Checkpointing callback: save models when validation reward improves
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",  # save to checkpoints/
    filename="epoch_{epoch:03d}",  # save as epoch_XXX.ckpt
    save_top_k=1,  # save only the best model
    save_last=True,  # save the last model
    monitor="val/reward",  # monitor validation reward
    mode="max",
)  # maximize validation reward

# Print model summary
rich_model_summary = RichModelSummary(max_depth=3)

# Callbacks list
callbacks = [checkpoint_callback, rich_model_summary]

# Logging
wandb.login()
logger = WandbLogger(
    project="cvrp-train-pomo",
    name=f"og-rl-lr-{lr}-weight-decay-{weight_decay}-B-{batch_size}",
)

trainer = RL4COTrainer(
    max_epochs=max_epochs,
    accelerator="gpu",
    devices=1,
    logger=logger,
    callbacks=callbacks,
)

trainer.fit(model)
