"""Supervised CVRP drawing from the shared, instance-grouped epoch used by the synced RL env.

The difference from :class:`~src.envs.CVRPEnv` is the shape of one epoch. ``CVRPEnv`` samples
``size`` *slots* uniformly from the whole ``sum(2R)`` pool, so an epoch touches ~95,300
instances at ~1.05 starts each and an instance's other optimal starts are never seen
together. This environment samples *instances* and expands each into all ``k`` of its drawn
starts, which is the draw :class:`~src.envs.dataset.CVRPSolutionDataset` documents but does
not implement -- and, more to the point here, the same draw
:class:`~src.envs.SyncedRLCVRPEnv` trains on.

Validation and testing are untouched: they fall through to
:class:`~src.envs.CVRPEnv`, so the two pipelines are still scored on the identical held-out
subset.

The one thing a caller has to get right is batching. Rows are indexed instance-major, so an
instance's ``k`` trajectories are contiguous, and the dataloader must **not** shuffle at row
level -- otherwise a step covers ~``batch_size`` distinct instances instead of
``batch_size / k``, and the grouping stops matching the RL side. Randomisation comes from the
draw itself, which reorders the instances every epoch. Pass
``shuffle_train_dataloader=False`` to the Lightning module.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.envs.cvrp import CVRPEnv, _resolve_size
from src.envs.dataset import CVRPSolutionDataset
from src.envs.synced_dataset import CVRPSyncedSolutionDataset

__all__ = ["SyncedCVRPEnv"]


class SyncedCVRPEnv(CVRPEnv):
    """A CVRP environment whose training epochs are instance-grouped and label-bounded.

    Args:
        data_path: Path to the reference-solution file to train and validate on.
        val_data_path: Optional separate file for validation and testing, forwarded to
            :class:`~src.envs.CVRPEnv`.
        num_loc: Number of customers. Inferred from the file when omitted.
        capacity: Vehicle capacity in the file's own demand units. Inferred when omitted.
        val_instances: Held-out instances carved from the tail of ``data_path``.
        seed: Base seed for the draw. Training re-draws with ``seed + epoch``, the same
            scheme :class:`~src.envs.CVRPEnv` and :class:`~src.envs.SyncedRLCVRPEnv` use.
        check_solution: Forwarded to :class:`~src.envs.CVRPEnv`.
        starts_per_instance: Trajectories contributed per instance. Must not exceed the
            smallest ``2R`` in the training range.
        **kwargs: Forwarded to :class:`~src.envs.CVRPEnv`.
    """

    def __init__(
        self,
        data_path: str,
        val_data_path: str | None = None,
        num_loc: int | None = None,
        capacity: float | None = None,
        val_instances: int = 10_000,
        seed: int = 0,
        check_solution: bool = False,
        starts_per_instance: int = 16,
        **kwargs,
    ) -> None:
        self.starts_per_instance = int(starts_per_instance)
        super().__init__(
            data_path=data_path,
            val_data_path=val_data_path,
            num_loc=num_loc,
            capacity=capacity,
            val_instances=val_instances,
            seed=seed,
            check_solution=check_solution,
            **kwargs,
        )

    def dataset(
        self, batch_size: int | Sequence = (), phase: str = "train", filename: str | None = None
    ) -> CVRPSyncedSolutionDataset | CVRPSolutionDataset:
        """Build the dataset for a phase.

        Args:
            batch_size: Number of *instances* to expose for training; rl4co passes the
                phase's data size here. The dataset it returns is ``instances × k`` rows
                long. Validation and testing interpret it as a trajectory count, exactly as
                :class:`~src.envs.CVRPEnv` does.
            phase: One of ``"train"``, ``"val"`` or ``"test"``.
            filename: Optional path overriding the configured file, used verbatim. Naming a
                file sends every phase, training included, to the slot-level dataset.

        Returns:
            A :class:`~src.envs.synced_dataset.CVRPSyncedSolutionDataset` for training, and a
            :class:`~src.envs.dataset.CVRPSolutionDataset` for validation and testing.
        """
        if phase == "train" and filename is None:
            self._epoch += 1
            return CVRPSyncedSolutionDataset(
                self.data_path,
                instance_range=self.train_range,
                instances=_resolve_size(batch_size),
                k=self.starts_per_instance,
                seed=self.seed + self._epoch,
            )
        return super().dataset(batch_size, phase=phase, filename=filename)

    def num_instances(self, phase: str = "train") -> int:
        """Return how many instances a phase's draw selects.

        :meth:`~src.envs.CVRPEnv.num_trajectories` reports the ``2R`` trajectory pool, which
        is no longer the unit of a training epoch here; multiply by
        ``starts_per_instance`` for the row count.

        Args:
            phase: One of ``"train"``, ``"val"`` or ``"test"``.

        Returns:
            The number of instances in the phase's range.
        """
        start, stop = self.train_range if phase == "train" else self.val_range
        return int(stop - start)
