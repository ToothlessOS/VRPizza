"""LEHD partials for the step-matched protocol: ``k`` windows per instance.

The canonical :class:`~src.envs.lehd_dataset.LEHDPartialDataset` draws one partial from each of
``N`` distinct instances. At the step-matched budget that would make one optimizer step cover
2,000 distinct instances, while the RL/SL arms cover 125 instances x 16 examples. This dataset
closes that gap: it draws ``instances`` instances in the shared supervised order and emits
``partials_per_instance`` windows for each, instance-major.

The per-partial RNG is keyed by ``[seed, instance, j]`` rather than the canonical
``[seed, instance]``. Without the ``j`` every one of an instance's ``k`` rows would replay the
same window. ``seed`` is already ``base_seed + epoch``, so windows change across epochs as
before. All ``k`` rows of an instance still share the optimizer-step chunk's ``w`` -- that is
intrinsic to LEHD's one-``w``-per-effective-batch rule, and the sampler varies the route end and
flips instead.

Rows are instance-major, so the DataLoader must not shuffle. ``w`` is keyed on the
*optimizer-step* row count (``micro_batch * accumulate``), not the DataLoader micro-batch;
:func:`collate_partials` verifies that every row in a micro-batch agrees on ``w``.
"""

from __future__ import annotations

import os

import numpy as np

from torch.utils.data import Dataset

from src.envs.dataset import build_index
from src.envs.lehd_dataset import collate_partials
from src.envs.parser import resolve_data_path
from src.envs.synced_dataset import sample_epoch_slots
from src.envs.tagged_reader import TaggedFileReader
from src.lehd.subpath import sample_window, split_routes

__all__ = ["LEHDStepPartialDataset"]


class LEHDStepPartialDataset(TaggedFileReader, Dataset):
    """``k`` LEHD partial solutions per instance, instance-major.

    Args:
        path: Path to a tagged CVRP solution file.
        instance_range: Half-open ``(start, stop)`` range of instance indices to draw from.
        instances: Number of instances to expose this epoch. The dataset length is
            ``instances * partials_per_instance``.
        partials_per_instance: Windows sampled per instance (16 for SL/teacher, 4 for the
            step-matched shared arm).
        effective_batch_size: Rows per optimizer step. Keys the one-``w``-per-step draw; pass
            ``micro_batch * accumulate`` (2,000 for SL/teacher, 500 for shared).
        w_min: Minimum window length, inclusive. LEHD uses 4.
        w_max: Maximum window length, inclusive. Defaults to the file's ``N``.
        seed: Base seed; callers pass ``seed + epoch``.
    """

    collate_fn = staticmethod(collate_partials)

    def __init__(
        self,
        path: str | os.PathLike,
        instance_range: tuple[int, int] | None = None,
        instances: int = 12_500,
        partials_per_instance: int = 16,
        effective_batch_size: int = 2_000,
        w_min: int = 4,
        w_max: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.path = resolve_data_path(path)
        self.offsets, self.num_routes, self.layout = build_index(self.path)
        if not self.layout.has_tour:
            raise ValueError(f"{self.path} carries no reference tour; it cannot be trained on")

        total = int(self.num_routes.size)
        start, stop = instance_range if instance_range is not None else (0, total)
        self.instance_range = (int(start), int(stop))
        if self.instance_range[0] >= self.instance_range[1]:
            raise ValueError(f"instance_range {self.instance_range} selects no instances")

        self.partials_per_instance = int(partials_per_instance)
        if self.partials_per_instance <= 0:
            raise ValueError(
                f"partials_per_instance must be positive, got {partials_per_instance}"
            )
        self.effective_batch_size = int(effective_batch_size)
        if self.effective_batch_size <= 0:
            raise ValueError(
                f"effective_batch_size must be positive, got {effective_batch_size}"
            )
        if int(instances) <= 0:
            raise ValueError(f"instances must be positive, got {instances}")

        # Same instance order as the supervised 16-start draw: `sample_epoch_slots`'s
        # `chosen` does not depend on `k`, so k=1 here yields the same permutation.
        self.instances, _ = sample_epoch_slots(
            self.num_routes, self.instance_range, int(seed), int(instances), 1
        )

        self.w_min = int(w_min)
        self.w_max = int(self.layout.num_loc if w_max is None else w_max)
        if not 2 <= self.w_min <= self.w_max <= self.layout.num_loc:
            raise ValueError(
                f"window range [{self.w_min}, {self.w_max}] is not inside "
                f"[2, {self.layout.num_loc}]"
            )

        rows = int(self.instances.size) * self.partials_per_instance
        chunks = -(-rows // self.effective_batch_size)
        self.w_per_chunk = np.asarray(
            [
                np.random.default_rng([seed, chunk]).integers(self.w_min, self.w_max + 1)
                for chunk in range(chunks)
            ],
            dtype=np.int64,
        )
        self.seed = int(seed)

    def __len__(self) -> int:
        """Number of partial solutions exposed this epoch."""
        return int(self.instances.size) * self.partials_per_instance

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.generic]:
        """Return one partial solution.

        Args:
            index: Position in ``[0, len(self))``. Rows are instance-major: the instance is
                ``index // partials_per_instance`` and the window index is the remainder.

        Returns:
            A dict with the reduced instance fields plus ``start_node``,
            ``used_capacity0``, the teacher ``actions`` and ``num_loc``, matching
            :class:`~src.envs.lehd_dataset.LEHDPartialDataset`.
        """
        position, window_index = divmod(int(index), self.partials_per_instance)
        instance = int(self.instances[position])
        chunk = int(index) // self.effective_batch_size
        w = int(self.w_per_chunk[chunk])

        record = self._read_record(instance)
        routes = split_routes(record["tour"], record["flags"])
        # Seeded by (seed, instance, window index) so the k rows differ and the draw is
        # independent of worker count and read order.
        rng = np.random.default_rng([self.seed, instance, window_index])
        window = sample_window(routes, record["locs"], record["demand"], w, rng)

        return {
            "depot": record["depot"],
            "locs": window.locs_w,
            "demand": window.demand_w,
            "capacity": record["capacity"],
            "cost": record["cost"],
            "start_node": np.int64(window.start_local),
            "used_capacity0": np.float32(window.used_capacity0),
            "actions": window.actions,
            "num_loc": np.int64(w),
        }
