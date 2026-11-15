"""POMO trained by RL warm-up followed by local-reconstruction SIL.

One ``trainer.fit`` has two phases:

* epochs ``0 .. warmup_epochs-1``: POMO REINFORCE on generated CVRP20 instances
  (the paper's warm-up), exactly as :class:`~src.models.POMORL`;
* from ``warmup_epochs`` on: teacher-forced CE on partial windows sampled from the
  current pseudo-label store, after a full local-reconstruction pass at every cycle
  boundary.

The label cycles mirror the reference's ``improve_iterations`` loop. At the first
SIL epoch the instance set is generated once (strict arm) or per cycle
(step-matched arm), each instance gets a random-insertion or model-greedy label,
and the labels are improved by ``recon_iterations`` RRC rounds with the best
validation policy so far. The cycle then trains ``cycle_epochs`` epochs over the
whole store, after which the next cycle repairs all labels again. The label store
is persisted to a sidecar file so a resume does not redraw or lose it.

Why a separate module from :class:`~src.models.POMOSIL`: the SIL phases share the
two-phase/optimizer/checkpoint structure, but the data source is a label store
rather than a per-epoch search, and the SIL loss is the LEHD partial CE rather
than full-trajectory CE. Inheritance from ``POMORL`` keeps REINFORCE's loss for
the warm-up; the partial loss is called through the explicit
:func:`~src.models.pomo_sl_lehd.partial_teacher_forced_train_step` helper, the
same way the existing supervised SIL routes its CE.
"""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch.callbacks import Callback
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from rl4co.utils.pylogger import get_pylogger

from src.models.pomo_rl import POMORL
from src.models.pomo_sl_lehd import partial_teacher_forced_train_step
from src.sil_rc.dataset import SILPartialDataset
from src.sil_rc.labels import LABEL_GEN_BLOCK, LabelStore, initial_labels
from src.sil_rc.reconstruction import improve_labels

__all__ = ["PhaseAccumulationCallback", "SILRCPOMOLitModule"]

log = get_pylogger(__name__)


class PhaseAccumulationCallback(Callback):
    """Set ``trainer.accumulate_grad_batches`` from the current phase.

    The warm-up decodes 2,000 rollouts per step with no accumulation, while the
    SIL phase needs ``micro x accumulation = 2,000`` partials per step. Lightning
    2.6 does not accept an epoch-keyed accumulation dict, so the value is set from
    the epoch at the start of training and at every epoch start.

    Args:
        warmup_epochs: First SIL epoch.
        warmup_accum: Accumulation factor during the warm-up (normally 1).
        sil_accum: Accumulation factor during the SIL phase.
    """

    def __init__(self, warmup_epochs: int, warmup_accum: int, sil_accum: int) -> None:
        super().__init__()
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_accum = int(warmup_accum)
        self.sil_accum = int(sil_accum)

    def _apply(self, trainer) -> None:
        trainer.accumulate_grad_batches = (
            self.warmup_accum
            if trainer.current_epoch < self.warmup_epochs
            else self.sil_accum
        )

    def on_train_start(self, trainer, _pl_module) -> None:
        """Apply the epoch-0 factor before the first batch."""
        self._apply(trainer)

    def on_train_epoch_start(self, trainer, _pl_module) -> None:
        """Apply the phase factor for this epoch."""
        self._apply(trainer)


class _WarmupThenDecay:
    """``LambdaLR`` multiplier: constant during warm-up, ``gamma`` per epoch after.

    A module-level picklable callable rather than a lambda, so the scheduler can be
    rebuilt from the checkpoint's hyperparameters. ``last_epoch`` is the index the
    multiplier receives, so the first SIL epoch is at the base LR and each later one
    is ``gamma`` times the previous.
    """

    def __init__(self, warmup_epochs: int, gamma: float) -> None:
        self.warmup_epochs = int(warmup_epochs)
        self.gamma = float(gamma)

    def __call__(self, epoch: int) -> float:
        if epoch <= self.warmup_epochs:
            return 1.0
        return self.gamma ** (epoch - self.warmup_epochs)


class SILRCPOMOLitModule(POMORL):
    """POMO with an RL warm-up and local-reconstruction SIL fine-tuning.

    Args:
        env: A :class:`~src.envs.SILRCCVRPEnv` or
            :class:`~src.envs.SILRCStepCVRPEnv`.
        warmup_epochs: POMO REINFORCE epochs on generated CVRP20. ``0`` starts
            directly in SIL, which is what a shared warm-up checkpoint implies.
        warmup_batch_size: Generated instances per warm-up step.
        warmup_data_size: Generated instances per warm-up epoch.
        cycle_epochs: SIL epochs per label cycle (20 strict, 30 step-matched).
        labels_per_cycle: Instances in the label store. ``1_000_000`` for the strict
            reference configuration, ``12_500`` for the step-matched arm.
        fresh_labels_per_cycle: Generate a new instance set at every cycle instead
            of improving the same one. The step-matched arm sets this because its
            label budget is per cycle; the strict arm keeps the reference's fixed
            1M set.
        init: ``"greedy"`` (the reference's warm-up path) or
            ``"random_insertion"`` (its from-scratch path).
        improve_first_cycle: Run the reconstruction pass before cycle 0's training.
            The reference skips it only for the from-scratch random-insertion path.
        recon_iterations: RRC rounds per instance per cycle (the reference's
            ``budget``).
        repair_batch: Instances per reconstruction batch (GPU memory knob).
        w_min, w_max: Partial-window range; ``w_max=None`` means ``N``.
        partials_per_instance: Training windows per label per epoch (1 strict,
            16 step-matched).
        sil_batch_size: Micro-batch of partials.
        sil_effective_batch_size: Rows per optimizer step (micro x accumulation).
            Keys the one-``w``-per-step draw and must include gradient accumulation.
        lr_gamma: SIL LR decay per epoch; ``1.0`` for the step-matched constant LR.
        label_seed: Base seed for instances, reconstruction and training windows.
        label_cache_path: Sidecar ``.npz`` path for the label store. ``None`` keeps
            it in memory only.
        generation_selection_metric: Validation key used to snapshot the best policy
            for label repair. ``val/reward_greedy`` is the single greedy rollout the
            reference uses for checkpoint selection.
        **kwargs: Forwarded to :class:`~src.models.POMORL`.
    """

    #: Backbone fingerprint written to label sidecars and checkpoints. A sidecar or
    #: checkpoint from another backbone is refused instead of silently reused: the
    #: greedy bootstrap labels and the policy weights are not interchangeable.
    BACKBONE = "pomo"

    def __init__(
        self,
        env,
        warmup_epochs: int = 100,
        warmup_batch_size: int = 64,
        warmup_data_size: int = 50_000,
        cycle_epochs: int = 20,
        labels_per_cycle: int = 1_000_000,
        fresh_labels_per_cycle: bool = False,
        init: str = "greedy",
        improve_first_cycle: bool = True,
        recon_iterations: int = 100,
        repair_batch: int = 1024,
        w_min: int = 4,
        w_max: int | None = None,
        partials_per_instance: int = 1,
        sil_batch_size: int = 1024,
        sil_effective_batch_size: int | None = None,
        lr_gamma: float = 0.97,
        label_seed: int = 0,
        label_cache_path: str | None = None,
        generation_selection_metric: str = "val/reward_greedy",
        **kwargs,
    ) -> None:
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be non-negative, got {warmup_epochs}")
        if cycle_epochs <= 0:
            raise ValueError(f"cycle_epochs must be positive, got {cycle_epochs}")
        if labels_per_cycle <= 0:
            raise ValueError(f"labels_per_cycle must be positive, got {labels_per_cycle}")
        if init not in ("random_insertion", "greedy"):
            raise ValueError(
                f"init must be 'random_insertion' or 'greedy', got {init!r}"
            )
        if recon_iterations <= 0 or repair_batch <= 0:
            raise ValueError("recon_iterations and repair_batch must be positive")
        if partials_per_instance <= 0:
            raise ValueError("partials_per_instance must be positive")
        if sil_effective_batch_size is None:
            sil_effective_batch_size = sil_batch_size
        if sil_effective_batch_size % sil_batch_size != 0:
            raise ValueError(
                f"sil_effective_batch_size {sil_effective_batch_size} must be a "
                f"multiple of sil_batch_size {sil_batch_size}"
            )

        # Stored before `super().__init__`, which calls `save_hyperparameters` and
        # walks this frame: the values land in the checkpoint with the rest.
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_batch_size = int(warmup_batch_size)
        self.warmup_data_size = int(warmup_data_size)
        self.cycle_epochs = int(cycle_epochs)
        self.labels_per_cycle = int(labels_per_cycle)
        self.fresh_labels_per_cycle = bool(fresh_labels_per_cycle)
        self.init = init
        self.improve_first_cycle = bool(improve_first_cycle)
        self.recon_iterations = int(recon_iterations)
        self.repair_batch = int(repair_batch)
        self.w_min = int(w_min)
        self.w_max = None if w_max is None else int(w_max)
        self.partials_per_instance = int(partials_per_instance)
        self.sil_batch_size = int(sil_batch_size)
        self.sil_effective_batch_size = int(sil_effective_batch_size)
        self.lr_gamma = float(lr_gamma)
        self.label_seed = int(label_seed)
        self.label_cache_path = None if label_cache_path is None else str(label_cache_path)
        self.generation_selection_metric = generation_selection_metric

        # `log_metrics` silently drops undeclared keys. The union covers the warm-up
        # (`reward`, `max_reward`), the SIL CE metrics and the label diagnostics.
        kwargs.setdefault(
            "metrics",
            {
                "train": [
                    "loss",
                    "reward",
                    "max_reward",
                    "ce_loss",
                    "token_ce",
                    "ppl",
                    "accuracy",
                    "label_mean_cost",
                    "label_best_cost",
                    "label_improved",
                    "recon_iterations",
                    "recon_rounds",
                    "recon_accepted",
                    "recon_seconds",
                    "labels",
                ],
                "val": ["reward", "reward_greedy", "gap_ref"],
                "test": ["reward", "reward_greedy", "gap_ref"],
            },
        )
        super().__init__(env, **kwargs)

        self._labels: LabelStore | None = None
        self._labels_cycle: int | None = None
        self._label_metrics: dict | None = None
        self._label_metrics_epoch: int | None = None
        self._best_policy_state: dict | None = None
        self._best_metric: float | None = None
        # The validation metric is accumulated here rather than read from
        # `trainer.callback_metrics`: Lightning 2.6 does not expose the epoch's
        # validation values in `on_train_epoch_end`, and the best snapshot must be
        # taken before the next cycle's label repair.
        self._val_metric_sum = 0.0
        self._val_metric_count = 0

    # -- phases ------------------------------------------------------------------

    @property
    def sil_start_epoch(self) -> int:
        """First SIL epoch."""
        return self.warmup_epochs

    def _is_sil(self) -> bool:
        """Whether the current epoch belongs to the SIL phase."""
        return self.current_epoch >= self.warmup_epochs

    def shared_step(self, batch, batch_idx: int, phase: str, dataloader_idx=None) -> dict:
        """REINFORCE during warm-up, partial CE during SIL, shared inference otherwise."""
        if phase != "train":
            return POMORL.shared_step(self, batch, batch_idx, phase, dataloader_idx)
        if not self._is_sil():
            return POMORL.shared_step(self, batch, batch_idx, phase, dataloader_idx)

        out = self._sil_step(batch)
        if self._label_metrics is not None and (
            self._label_metrics_epoch != self.current_epoch
        ):
            out.update(self._label_metrics)
            self._label_metrics_epoch = self.current_epoch
        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def _sil_step(self, batch) -> dict:
        """One SIL training step; overridden by the LEHD-backbone module.

        The POMO backbone uses the repo's N+1 partial cross-entropy. The LEHD
        module replaces it with the reference's exact 2N teacher-forced CE,
        which needs the decoder's raw-logit captures.
        """
        return partial_teacher_forced_train_step(self, batch)

    def policy_signature(self) -> str:
        """Architecture fingerprint for label/checkpoint metadata."""
        return "pomo-am-e6-h8-ff512"

    def train_dataloader(self):
        """Generated CVRP20 during warm-up, the label-window dataset during SIL."""
        # `setup()` and the sanity check may ask for a loader before training starts.
        # The first SIL initialisation can be hours long, so never run it there; the
        # entry points also set `num_sanity_val_steps=0`.
        sanity = self.trainer is not None and self.trainer.sanity_checking
        if not self._is_sil() or sanity:
            dataset = self.env.warmup_dataset(
                self.warmup_data_size, self.env.seed + self.current_epoch
            )
            return DataLoader(
                dataset,
                batch_size=self.warmup_batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=dataset.collate_fn,
            )

        self._ensure_labels()
        dataset = SILPartialDataset(
            self._labels,
            partials_per_instance=self.partials_per_instance,
            effective_batch_size=self.sil_effective_batch_size,
            w_min=self.w_min,
            w_max=self.w_max,
            seed=self.env.seed + self.current_epoch,
        )
        return DataLoader(
            dataset,
            batch_size=self.sil_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.collate_fn,
        )

    def on_train_epoch_end(self) -> None:
        """Do nothing: both phases' loaders are rebuilt by ``train_dataloader``.

        The base ``RL4COLitModule`` hook regenerates ``env.dataset`` for the next
        epoch. Here ``train_dataloader`` builds the warm-up batch (or the SIL partial
        dataset) from ``current_epoch`` itself, so that regeneration would build an
        unused warm-up dataset every epoch. The shared baseline's epoch callback is a
        no-op, so skipping it loses nothing. Best-state tracking lives in the
        validation hooks.
        """

    def validation_step(self, batch, batch_idx: int, dataloader_idx=None):
        """Run the shared inference and accumulate the generation-selection metric.

        ``trainer.callback_metrics`` is not populated at the epoch-end hook in
        Lightning 2.6, so the weighted mean is collected here and the snapshot is
        taken in :meth:`on_validation_epoch_end`.
        """
        out = self.shared_step(batch, batch_idx, "val", dataloader_idx)
        if self.trainer is None or self.trainer.sanity_checking:
            return out
        value = out.get(self.generation_selection_metric)
        if value is not None:
            size = int(batch.batch_size[0])
            self._val_metric_sum += float(value) * size
            self._val_metric_count += size
        return out

    def on_validation_epoch_end(self) -> None:
        """Snapshot the policy when the generation-selection metric improves."""
        if self.trainer is None or self.trainer.sanity_checking:
            return
        if self._val_metric_count == 0:
            return
        value = self._val_metric_sum / self._val_metric_count
        self._val_metric_sum = 0.0
        self._val_metric_count = 0
        if self._best_metric is None or value > self._best_metric:
            self._best_metric = value
            self._best_policy_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in self.policy.state_dict().items()
            }
        log.info(
            "validation %s = %.6f (best %.6f)",
            self.generation_selection_metric,
            value,
            self._best_metric,
        )

    # -- label cycles -------------------------------------------------------------

    def _cycle_index(self) -> int:
        """Zero-based SIL cycle containing the current epoch."""
        return (self.current_epoch - self.warmup_epochs) // self.cycle_epochs

    def _cycle_seed(self, cycle: int) -> int:
        """Base seed of ``cycle``'s instance set."""
        return self.label_seed + (cycle if self.fresh_labels_per_cycle else 0)

    def _expected_label_meta(self, cycle: int) -> dict:
        """Metadata a sidecar must match before it is reused.

        A fresh-per-cycle store is keyed by ``label_seed + cycle``; a fixed-instance
        store by the base seed. Including the appropriate seed here is what stops a
        stale sidecar from a different cycle being mistaken for this one's.
        """
        expected = {
            "num_loc": int(self.env.num_loc),
            "capacity": float(self.env.capacity),
            "count": self.labels_per_cycle,
            "block_size": LABEL_GEN_BLOCK,
            "init": self.init,
            "recon_iterations": int(self.recon_iterations),
            "label_seed": int(self._cycle_seed(cycle)),
            "backbone": self.BACKBONE,
            "policy": self.policy_signature(),
        }
        return expected

    def _ensure_labels(self) -> None:
        """Build, load or repair the label store for the current SIL cycle."""
        cycle = self._cycle_index()
        if self._labels is not None and self._labels_cycle == cycle:
            return

        # Try the sidecar once per process. A fixed-instance store from an earlier
        # cycle is usable and only needs another repair pass; a fresh-per-cycle
        # store is usable only when it is already this cycle's.
        if self._labels is None and self.label_cache_path is not None:
            path = Path(self.label_cache_path)
            if path.is_file():
                try:
                    loaded = LabelStore.load(
                        path, expected=self._expected_label_meta(cycle)
                    )
                    if len(loaded) != self.labels_per_cycle:
                        raise ValueError(
                            f"sidecar has {len(loaded)} labels, expected "
                            f"{self.labels_per_cycle}"
                        )
                    self._labels = loaded
                    self._labels_cycle = int(loaded.meta.get("cycle", -1))
                    log.info(
                        "loaded SIL label store from %s (cycle %s)",
                        path,
                        self._labels_cycle,
                    )
                except (ValueError, OSError) as error:
                    log.warning("ignoring SIL label sidecar %s: %s", path, error)
                    self._labels = None
                    self._labels_cycle = None

        usable = self._labels is not None and (
            not self.fresh_labels_per_cycle or self._labels_cycle == cycle
        )
        if not usable:
            self._labels = self.env.sample_instances(
                self.labels_per_cycle, self._cycle_seed(cycle)
            )
            self._labels_cycle = None
            self._label_metrics = self._initial_labels()
            self._index_label_metrics(self._label_metrics)

        if self._labels_cycle is None or self._labels_cycle < cycle:
            if self.improve_first_cycle or cycle > 0:
                self._label_metrics = self._repair_labels(cycle)
                self._index_label_metrics(self._label_metrics)

        self._labels_cycle = cycle
        self._labels.meta.update(
            {
                "cycle": int(cycle),
                "init": self.init,
                "recon_iterations": int(self.recon_iterations),
                "backbone": self.BACKBONE,
                "policy": self.policy_signature(),
            }
        )
        self._save_labels()

    def _initial_labels(self) -> dict:
        """Bootstrap labels with the best policy so far.

        The reference loads the best validation checkpoint at the start of every
        repair pass (``repair_one_epoch``) and leaves those weights in the model for
        the training block; the optimizer state is not reloaded. Both paths here do
        the same; random insertion simply does not call the model.
        """
        was_training = self.policy.training
        try:
            if self._best_policy_state is not None:
                self.policy.load_state_dict(self._best_policy_state)
            self.policy.eval()
            return initial_labels(
                self._labels,
                model=self,
                env=self.env,
                init=self.init,
                seed=self.label_seed,
                batch_size=self.repair_batch,
                device=self.device,
            )
        finally:
            self.policy.train(was_training)

    def _repair_labels(self, cycle: int) -> dict:
        """Run one full reconstruction pass with the best policy so far.

        As in the reference, the best-validation weights are loaded before the pass
        and deliberately left in the model afterwards; the optimizer state persists.
        """
        was_training = self.policy.training
        try:
            if self._best_policy_state is not None:
                self.policy.load_state_dict(self._best_policy_state)
            self.policy.eval()
            metrics = improve_labels(
                self._labels,
                model=self,
                env=self.env,
                repair_batch=self.repair_batch,
                iterations=self.recon_iterations,
                cycle=cycle,
                seed=self.label_seed,
                w_min=self.w_min,
                w_max=self.w_max,
                device=self.device,
            )
        finally:
            self.policy.train(was_training)
        log.info(
            "repaired %d labels for cycle %d: mean %.4f, best %.4f, improved %d",
            metrics["labels"],
            cycle,
            metrics["label_mean_cost"],
            metrics["label_best_cost"],
            metrics["label_improved"],
        )
        return metrics

    def _index_label_metrics(self, metrics: dict) -> None:
        """Attach a cycle's diagnostics to the next logged step."""
        self._label_metrics = dict(metrics)
        self._label_metrics_epoch = None

    def _save_labels(self) -> None:
        """Persist the store atomically after a complete initialisation/repair."""
        if self.label_cache_path is None or self._labels is None:
            return
        self._labels.save(self.label_cache_path)
        log.info("saved SIL label store to %s", self.label_cache_path)

    # -- best-policy bookkeeping ---------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Persist the small best-so-far state and the label-cycle pointer."""
        checkpoint["sil_rc_state"] = {
            "best_policy_state": self._best_policy_state,
            "best_metric": self._best_metric,
            "labels_cycle": self._labels_cycle,
            "label_cache_path": self.label_cache_path,
            "backbone": self.BACKBONE,
            "policy": self.policy_signature(),
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Restore the best-so-far state; the label store is reloaded from disk.

        Raises:
            ValueError: If the checkpoint was written by another backbone. The
                policy weights would fail ``load_state_dict`` too, but this names
                the actual problem and also covers the best-policy snapshot.
        """
        state = checkpoint.get("sil_rc_state")
        if state is not None:
            saved = state.get("backbone", "pomo")
            if saved != self.BACKBONE:
                raise ValueError(
                    f"checkpoint was written by the {saved!r} backbone but this module "
                    f"is {self.BACKBONE!r}; resume it with the same model class"
                )
            self._best_policy_state = state.get("best_policy_state")
            self._best_metric = state.get("best_metric")
            self._labels_cycle = state.get("labels_cycle")
        self._labels = None
        self._label_metrics = None
        self._label_metrics_epoch = None

    # -- optimizer ----------------------------------------------------------------

    def configure_optimizers(self, parameters=None):
        """Adam with a warm-up-constant / SIL-decaying LR schedule."""
        result = super().configure_optimizers(parameters)
        if isinstance(result, tuple):
            optimizer = result[0][0]
        else:
            optimizer = result
        scheduler = LambdaLR(
            optimizer, lr_lambda=_WarmupThenDecay(self.warmup_epochs, self.lr_gamma)
        )
        return [optimizer], {"scheduler": scheduler, "interval": "epoch"}
