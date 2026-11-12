"""LEHD environments for the step-matched comparison: ``k`` partials per instance.

Only the training dataset changes. Validation and testing fall through to
:class:`~src.envs.LEHDCVRPEnv`/``LEHDRLCVRPEnv``'s full-trajectory path, so the shared
``POMOInferenceMixin`` metrics stay on full instances with N=100 starts.

The optimizer-step row count keys LEHD's one-``w``-per-batch draw. It differs between arms:
2,000 for SL and the teacher-baseline RL arm (500 x 4), but 500 for the shared-baseline arm
(500 partials x K=4 rollouts). Passing 2,000 there would make a ``w`` chunk four optimizer
steps long and desync the draw.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.envs.cvrp import _resolve_size
from src.envs.cvrp_sl_lehd import LEHDCVRPEnv
from src.envs.cvrp_rl_lehd import LEHDRLCVRPEnv
from src.envs.dataset import CVRPSolutionDataset
from src.envs.lehd_step_dataset import LEHDStepPartialDataset

__all__ = ["LEHDStepCVRPEnv", "LEHDStepRLCVRPEnv"]


class LEHDStepDatasetMixin:
    """Serve :class:`LEHDStepPartialDataset` for training, the parent dataset otherwise.

    Args:
        instances: Instances drawn per epoch. The dataset length is
            ``instances * partials_per_instance``; the script passes
            ``TRAIN_INSTANCES_PER_EPOCH`` (12,500).
        partials_per_instance: Windows per instance (16 for SL/teacher, 4 for shared).
        effective_batch_size: Rows per optimizer step that share one ``w``.
        **kwargs: Forwarded to the LEHD environment.
    """

    def __init__(
        self,
        *args,
        instances: int = 12_500,
        partials_per_instance: int = 16,
        effective_batch_size: int = 2_000,
        **kwargs,
    ) -> None:
        self.step_instances = int(instances)
        self.step_partials_per_instance = int(partials_per_instance)
        self.step_effective_batch_size = int(effective_batch_size)
        super().__init__(*args, **kwargs)

    def dataset(
        self, batch_size: int | Sequence = (), phase: str = "train", filename: str | None = None
    ) -> LEHDStepPartialDataset | CVRPSolutionDataset:
        """Build the step-matched partial dataset for training, the parent's otherwise."""
        if phase == "train" and filename is None:
            self._epoch += 1
            size = _resolve_size(batch_size)
            if size is None:
                size = self.step_instances
            return LEHDStepPartialDataset(
                self.data_path,
                instance_range=self.train_range,
                instances=int(size),
                partials_per_instance=self.step_partials_per_instance,
                effective_batch_size=self.step_effective_batch_size,
                seed=self.seed + self._epoch,
            )
        return super().dataset(batch_size, phase=phase, filename=filename)


class LEHDStepCVRPEnv(LEHDStepDatasetMixin, LEHDCVRPEnv):
    """Supervised LEHD arm with ``k`` partials per instance per epoch."""


class LEHDStepRLCVRPEnv(LEHDStepDatasetMixin, LEHDRLCVRPEnv):
    """REINFORCE-on-partials arm with ``k`` partials per instance per epoch."""
