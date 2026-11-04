"""A supervised dataset over SIL pseudo-expert solutions.

The generated records are the same ``(tour, flags)`` form as the reference files, so this
dataset reuses :meth:`src.envs.dataset.CVRPSolutionDataset.build_trajectory` and its
``collate_fn`` unchanged: each record contributes ``2R`` trajectories, one per legal start
(each route served first, forwards or backwards), all reproducing the record's cost.

Unlike the file-backed dataset, the epoch length is explicit. ``length`` slots are drawn in
``__init__`` (with replacement when the requested epoch is longer than the pool), so appending
records across non-improving SIL epochs cannot silently change how many optimizer steps an
epoch has. ``__getitem__`` is pure and holds only NumPy arrays, which keeps it safe with
DataLoader workers, although the default is zero workers because a spawn worker pickles the
whole record pool.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from torch.utils.data import Dataset

from src.envs.dataset import CVRPSolutionDataset
from src.sil.search import SILRecord

__all__ = ["SILSolutionDataset"]


class SILSolutionDataset(Dataset):
    """Trajectory-level dataset over a list of generated pseudo-expert records.

    Args:
        records: Generated pseudo-experts, as returned by
            :func:`~src.sil.search.generate_best_solutions`.
        length: Number of trajectories to expose this epoch. ``None`` exposes the whole pool
            once.
        seed: Seed for the slot draw.
    """

    collate_fn = staticmethod(CVRPSolutionDataset.collate_fn)

    def __init__(
        self,
        records: Sequence[SILRecord],
        length: int | None = None,
        seed: int = 0,
    ) -> None:
        if not records:
            raise ValueError("SILSolutionDataset needs at least one record")
        self.records = list(records)
        self._csum = np.concatenate(
            [[0], 2 * np.cumsum([record.num_routes for record in self.records], dtype=np.int64)]
        )
        pool_size = int(self._csum[-1])
        if pool_size <= 0:
            raise ValueError("SIL records carry no trajectories")

        rng = np.random.default_rng(seed)
        if length is None:
            self._slots = rng.permutation(pool_size).astype(np.int64)
        elif length <= pool_size:
            self._slots = rng.choice(pool_size, size=int(length), replace=False).astype(np.int64)
        else:
            self._slots = rng.integers(pool_size, size=int(length), dtype=np.int64)

    def __len__(self) -> int:
        """Number of trajectories exposed this epoch."""
        return int(self._slots.size)

    @property
    def pool_size(self) -> int:
        """Total number of trajectories in the record pool."""
        return int(self._csum[-1])

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.generic]:
        """Return the trajectory at ``index`` in the same layout as ``CVRPSolutionDataset``."""
        slot = int(self._slots[int(index)])
        record_idx = int(np.searchsorted(self._csum, slot, side="right")) - 1
        local = slot - int(self._csum[record_idx])
        route_index, direction = divmod(local, 2)

        record = self.records[record_idx]
        actions = CVRPSolutionDataset.build_trajectory(
            record.tour, record.flags, route_index, direction
        )
        return {
            "depot": record.depot,
            "locs": record.locs,
            "demand": record.demand,
            "capacity": np.float32(record.capacity),
            "cost": np.float32(record.cost),
            "actions": actions,
            "start_node": np.int64(actions[0]),
            "num_routes": np.int64(record.num_routes),
        }
