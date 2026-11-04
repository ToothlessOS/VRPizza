"""POMO trained with an RL warm-up followed by self-improvement learning (SIL).

The first ``warmup_epochs`` epochs are ordinary POMO REINFORCE on freshly generated CVRP
instances -- the same loop as ``notebooks/POMO_og.py``. From then on the objective becomes
teacher-forced cross-entropy on a pseudo-expert dataset produced by the policy itself: each
epoch, the current (or best) policy explores a fresh batch of generated instances with
WOR-SBS or plain beam search, the cheapest trajectory per instance is kept, and the network is
trained for ``sil_batches_per_epoch`` steps on the ``2R`` legal starts of those solutions.

Why one module and one ``trainer.fit``: the policy and optimizer carry over the phase switch,
the W&B run and checkpoint directory stay single, and the validation metrics remain the shared
POMO ones. The switch is keyed on ``current_epoch`` in both :meth:`shared_step` and
:meth:`train_dataloader`, which Lightning calls at the start of each epoch after it has set
``current_epoch``.

The phase switch is the only structural difference from the existing runs; everything that
touches the network (architecture, checkpoint layout, validation rollout) is shared, so a SIL
checkpoint loads into ``POMORL``/``POMOSL`` and is evaluated by the same driver.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch.utils.data import DataLoader

from rl4co.models.zoo import POMO

from src.envs.cvrp_sil import SILCVRPEnv
from src.models.pomo_rl import POMORL
from src.models.pomo_sl import teacher_forced_train_step
from src.sil.dataset import SILSolutionDataset
from src.sil.search import SILRecord, SILSearchConfig, generate_best_solutions

__all__ = ["POMOSIL"]


class POMOSIL(POMORL):
    """POMO with an RL warm-up and a SIL fine-tuning phase.

    Args:
        env: A :class:`~src.envs.SILCVRPEnv`.
        warmup_epochs: Number of POMO REINFORCE epochs before the first generation round.
            ``0`` starts directly in SIL, which is what a shared warm-up checkpoint implies.
        sil_batch_size: Batch size for SIL teacher-forced training.
        sil_batches_per_epoch: Optimizer steps per SIL epoch. Fixed explicitly because the
            pseudo-expert pool changes across epochs and the step count must not.
        num_instances_to_generate: Fresh instances per SIL round. Search cost scales with
            this, the beam width and the rounds.
        search_config: Sampler configuration. The experiment driver restricts ``search_type``
            to ``"wor"`` or ``"beam_search"``; the advantage updates are dormant.
        use_best_model_for_generation: Generate with the best-validation policy rather than
            the current one (Gumbeldore's default).
        append_if_not_new_best: Append a round's records to the pool when the previous epoch
            did not improve ``generation_selection_metric``; replace the pool otherwise.
        max_sil_instances: Cap on the accumulated record pool. Oldest records are dropped.
        generation_selection_metric: Validation key used to decide "new best" for generation.
            Defaults to ``val/reward_greedy``, a single-start rollout like the search, rather
            than the maximised-over-800-rollouts ``val/reward`` used for checkpointing.
        generation_seed: Base seed; generation rounds use ``generation_seed + epoch``.
        **kwargs: Forwarded to :class:`~src.models.POMORL`, including the Lightning-module
            arguments (``batch_size`` for the warm-up, ``train_data_size``, validation sizes,
            optimizer settings and so on).
    """

    def __init__(
        self,
        env: SILCVRPEnv,
        warmup_epochs: int = 50,
        sil_batch_size: int = 256,
        sil_batches_per_epoch: int = 100,
        num_instances_to_generate: int = 1_000,
        search_config: SILSearchConfig | None = None,
        use_best_model_for_generation: bool = True,
        append_if_not_new_best: bool = True,
        max_sil_instances: int = 20_000,
        generation_selection_metric: str = "val/reward_greedy",
        generation_seed: int = 0,
        **kwargs,
    ) -> None:
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be non-negative, got {warmup_epochs}")
        if sil_batch_size <= 0 or sil_batches_per_epoch <= 0:
            raise ValueError("sil_batch_size and sil_batches_per_epoch must be positive")
        if num_instances_to_generate <= 0:
            raise ValueError("num_instances_to_generate must be positive")

        # Stored before `super().__init__`, which calls `save_hyperparameters` and walks this
        # frame: the values then land in the checkpoint alongside the model hyperparameters.
        self.warmup_epochs = int(warmup_epochs)
        self.sil_batch_size = int(sil_batch_size)
        self.sil_batches_per_epoch = int(sil_batches_per_epoch)
        self.num_instances_to_generate = int(num_instances_to_generate)
        self.search_config = search_config or SILSearchConfig()
        self.use_best_model_for_generation = bool(use_best_model_for_generation)
        self.append_if_not_new_best = bool(append_if_not_new_best)
        self.max_sil_instances = int(max_sil_instances)
        self.generation_selection_metric = generation_selection_metric
        self.generation_seed = int(generation_seed)

        # `log_metrics` silently drops undeclared keys. The union covers the RL warm-up
        # (`reward`, `max_reward`) and the SIL phase (`ce_loss`, `ppl`, `accuracy`) plus the
        # generation diagnostics.
        kwargs.setdefault(
            "metrics",
            {
                "train": [
                    "loss",
                    "reward",
                    "max_reward",
                    "ce_loss",
                    "ppl",
                    "accuracy",
                    "gen_mean_cost",
                    "gen_best_cost",
                    "gen_instances",
                    "search_seconds",
                    "records",
                ],
                "val": ["reward", "reward_greedy", "gap_ref"],
                "test": ["reward", "reward_greedy", "gap_ref"],
            },
        )
        super().__init__(env, **kwargs)

        self._sil_records: list[SILRecord] = []
        self._best_policy_state: dict | None = None
        self._best_metric: float | None = None
        self._fresh_best = True
        self._generated_epoch: int | None = None
        self._generation_metrics: dict | None = None
        self._generation_metrics_epoch: int | None = None

    # -- phases --------------------------------------------------------------------

    def shared_step(self, batch, batch_idx, phase: str, dataloader_idx=None) -> dict:
        """REINFORCE during warm-up, teacher-forced CE during SIL, shared validation otherwise."""
        if phase != "train":
            return POMORL.shared_step(self, batch, batch_idx, phase, dataloader_idx)

        if self.current_epoch < self.warmup_epochs:
            return POMO.shared_step(self, batch, batch_idx, phase, dataloader_idx)

        out = teacher_forced_train_step(self, batch)
        if self._generation_metrics is not None and (
            self._generation_metrics_epoch != self.current_epoch
        ):
            out.update(self._generation_metrics)
            self._generation_metrics_epoch = self.current_epoch
        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def train_dataloader(self):
        """The generator dataset during warm-up, the pseudo-expert dataset afterwards."""
        if self.current_epoch < self.warmup_epochs:
            return super().train_dataloader()

        self._ensure_generated()
        dataset = SILSolutionDataset(
            self._sil_records,
            length=self.sil_batches_per_epoch * self.sil_batch_size,
            seed=self.generation_seed + self.current_epoch,
        )
        return DataLoader(
            dataset,
            batch_size=self.sil_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.collate_fn,
        )

    def on_train_epoch_end(self) -> None:
        """Refresh the RL dataset during warm-up and record whether this epoch was a new best.

        The refresh is skipped in SIL: the next ``train_dataloader`` generates and installs
        the dataset, so the base class refresh (and REINFORCE's baseline callback, a no-op for
        the shared baseline) is deliberately not run.

        Best-state tracking lives here rather than in ``on_validation_epoch_end`` because
        Lightning calls the validation hook before it populates ``trainer.callback_metrics``;
        by the epoch-end hook the validation metrics are available.
        """
        if self.current_epoch < self.warmup_epochs:
            super().on_train_epoch_end()
        self._update_best_from_metrics()

    # -- generation ----------------------------------------------------------------

    def _ensure_generated(self) -> None:
        """Generate this epoch's pseudo-experts once, with the best policy if configured."""
        if self._generated_epoch == self.current_epoch:
            return
        if self.trainer.world_size != 1:
            raise RuntimeError(
                "SIL generation runs on a single process; with more ranks each rank would "
                "build a different dataset and the models would diverge"
            )

        was_training = self.policy.training
        # Deep-copy the live weights before loading the best snapshot: `load_state_dict`
        # copies into the live tensors, so a shallow `state_dict()` copy would be corrupted.
        current_state = {
            key: value.detach().cpu().clone()
            for key, value in self.policy.state_dict().items()
        }
        try:
            if self.use_best_model_for_generation and self._best_policy_state is not None:
                self.policy.load_state_dict(self._best_policy_state)
            self.policy.eval()
            config = replace(
                self.search_config, seed=self.generation_seed + self.current_epoch
            )
            with torch.autocast(device_type=self.device.type, enabled=False):
                records, metrics = generate_best_solutions(
                    self.policy,
                    self.env,
                    self.num_instances_to_generate,
                    config,
                    device=self.device,
                )
        finally:
            self.policy.load_state_dict(current_state)
            self.policy.train(was_training)

        append = (
            self.append_if_not_new_best
            and not self._fresh_best
            and bool(self._sil_records)
        )
        if append:
            self._sil_records.extend(records)
        else:
            self._sil_records = records
        if len(self._sil_records) > self.max_sil_instances:
            self._sil_records = self._sil_records[-self.max_sil_instances :]

        self.env.set_sil_records(self._sil_records)
        metrics = dict(metrics)
        metrics["records"] = len(self._sil_records)
        self._generation_metrics = metrics
        self._generation_metrics_epoch = None
        self._generated_epoch = self.current_epoch

    # -- best-so-far bookkeeping ---------------------------------------------------

    def _update_best_from_metrics(self) -> None:
        """Snapshot the policy when ``generation_selection_metric`` improves."""
        if self.trainer is None or self.trainer.sanity_checking:
            return
        metric = self.trainer.callback_metrics.get(self.generation_selection_metric)
        if metric is None:
            return
        value = float(metric)
        if self._best_metric is None or value > self._best_metric:
            self._best_metric = value
            self._best_policy_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in self.policy.state_dict().items()
            }
            self._fresh_best = True
        else:
            self._fresh_best = False

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Persist the small best-so-far state; the record pool is regenerated on resume."""
        checkpoint["sil_state"] = {
            "best_policy_state": self._best_policy_state,
            "best_metric": self._best_metric,
            "fresh_best": self._fresh_best,
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Restore the best-so-far state and force a fresh generation round after a resume."""
        state = checkpoint.get("sil_state")
        if state is not None:
            self._best_policy_state = state.get("best_policy_state")
            self._best_metric = state.get("best_metric")
            self._fresh_best = state.get("fresh_best", True)
        self._sil_records = []
        self._generated_epoch = None
        self._generation_metrics = None
        self._generation_metrics_epoch = None
        self.env.clear_sil_records()
