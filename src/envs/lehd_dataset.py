"""Lazy LEHD-style partial-solution dataset over a tagged CVRP solution file.

Each item is one partial solution sampled from a reference tour: a contiguous window of ``w``
customers ending at a route end, with the rest of the tour discarded. The first window
customer is the forced start; the target is the remaining ``w-1`` customer actions plus a
depot action at every interior route boundary. See :mod:`src.lehd.subpath` for the geometry
and the reference implementation it mirrors.

Two properties of the draw are load-bearing:

* **One ``w`` per effective batch.** LEHD samples ``w ~ Unif{4..N}`` once per batch, not once
  per instance. A ``Dataset.__getitem__`` cannot see its batch, so the draw is keyed by
  ``index // effective_batch_size`` and the DataLoader must not shuffle rows. The training
  script passes the effective batch size (microbatch x gradient accumulation); the dataset
  asserts nothing about it here, but :func:`collate_partials` refuses a batch whose rows
  disagree on ``w``, which is what catches a misconfigured accumulation factor.
* **Per-instance seeding.** The flips and route-end choice are drawn from
  ``default_rng([seed, epoch, instance])``, never a shared stream, so the item is identical
  regardless of worker count or read order.

The window is emitted as a **reduced problem**: depot + the ``w`` window customers, sorted by
ascending original id and reindexed ``1..w``. Validation and testing deliberately fall through
to the parent environment's full-trajectory dataset, so the POMO validation metrics stay
comparable with the existing runs.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from tensordict.tensordict import TensorDict
from torch.utils.data import Dataset

from src.envs.dataset import build_index
from src.envs.parser import resolve_data_path
from src.envs.synced_dataset import _TaggedFileReader
from src.lehd.subpath import sample_window, split_routes

__all__ = ["LEHDPartialDataset", "collate_partials"]


class LEHDPartialDataset(_TaggedFileReader, Dataset):
    """One LEHD partial solution per instance per epoch.

    Args:
        path: Path to a tagged CVRP solution file.
        instance_range: Half-open ``(start, stop)`` range of instance indices to draw from.
        partials: Number of partial solutions to expose this epoch. ``None`` (or a value at
            least as large as the range) exposes every instance once.
        batch_size: **Effective** optimizer batch size used to key the per-batch ``w`` draw.
            With gradient accumulation this is larger than the DataLoader batch, and the draw
            still stays constant across the accumulation's microbatches.
        w_min: Minimum window length, inclusive. LEHD uses 4.
        w_max: Maximum window length, inclusive. Defaults to the file's ``N``.
        seed: Base seed; callers pass ``seed + epoch``.
    """

    @staticmethod
    def collate_fn(batch: list[dict[str, np.ndarray | np.generic]]) -> TensorDict:
        """Collate partial solutions; see :func:`collate_partials`."""
        return collate_partials(batch)

    def __init__(
        self,
        path: str | os.PathLike,
        instance_range: tuple[int, int] | None = None,
        partials: int | None = None,
        batch_size: int = 1024,
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
        available = self.instance_range[1] - self.instance_range[0]
        if available <= 0:
            raise ValueError(f"instance_range {self.instance_range} selects no instances")

        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        count = available if partials is None else min(int(partials), available)
        if count <= 0:
            raise ValueError(f"partials must be positive, got {partials}")

        rng = np.random.default_rng(seed)
        if count == available:
            # A permutation of the whole range: every instance once, fresh order each epoch.
            chosen = np.arange(self.instance_range[0], self.instance_range[1], dtype=np.int64)
            rng.shuffle(chosen)
        else:
            chosen = rng.choice(available, size=count, replace=False).astype(np.int64)
            chosen += self.instance_range[0]
        self.instances = chosen

        self.w_min = int(w_min)
        self.w_max = int(self.layout.num_loc if w_max is None else w_max)
        if not 2 <= self.w_min <= self.w_max <= self.layout.num_loc:
            raise ValueError(
                f"window range [{self.w_min}, {self.w_max}] is not inside "
                f"[2, {self.layout.num_loc}]"
            )

        # One w per effective batch. Keying by chunk rather than by DataLoader batch is what
        # lets gradient accumulation keep the LEHD draw: all microbatches in a chunk share w.
        chunks = -(-count // self.batch_size)
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
        return int(self.instances.size)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.generic]:
        """Return one partial solution.

        Args:
            index: Position in ``[0, len(self))``. The effective-batch chunk is
                ``index // batch_size``.

        Returns:
            A dict with the reduced instance (``depot``, ``locs``, ``demand``, ``capacity``,
            ``cost``) plus ``start_node``, ``used_capacity0``, the teacher ``actions`` and
            ``num_loc``. ``used_capacity0`` is in raw demand units; the environment divides it
            by the instance capacity exactly as it does for ``demand``.
        """
        index = int(index)
        chunk = index // self.batch_size
        w = int(self.w_per_chunk[chunk])
        instance = int(self.instances[index])

        record = self._read_record(instance)
        routes = split_routes(record["tour"], record["flags"])
        # Seeded by instance, so the draw is independent of worker count and read order.
        rng = np.random.default_rng([self.seed, instance])
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


def collate_partials(batch: list[dict[str, np.ndarray | np.generic]]) -> TensorDict:
    """Collate partial solutions, padding the teacher actions with depot.

    Every row must have the same window length ``w`` -- one ``w`` per effective batch is part
    of the LEHD draw. Padding action ``0`` is feasible at a customer and is excluded from the
    loss by ``valid``, so unequal target lengths (different numbers of route boundaries) can
    share a batch.

    Args:
        batch: Items as returned by :meth:`LEHDPartialDataset.__getitem__`.

    Returns:
        A :class:`~tensordict.TensorDict` with ``depot [B,2]``, ``locs [B,w,2]``,
        ``demand [B,w]``, ``capacity [B]``, ``cost [B]``, ``start_node [B]``,
        ``used_capacity0 [B]``, ``actions [B,T]``, ``valid [B,T]`` and ``num_loc [B]``.

    Raises:
        ValueError: If the rows disagree on ``w``, which means the DataLoader batching and
            the dataset's effective-batch key have drifted apart.
    """
    size = len(batch)
    widths = {int(item["num_loc"]) for item in batch}
    if len(widths) != 1:
        raise ValueError(
            f"a batch must share one window length, got {sorted(widths)}; check that the "
            "DataLoader batch divides the effective batch and that row shuffling is off"
        )

    length = max(int(item["actions"].size) for item in batch)
    actions = np.zeros((size, length), dtype=np.int64)
    valid = np.zeros((size, length), dtype=bool)
    for row, item in enumerate(batch):
        span = int(item["actions"].size)
        actions[row, :span] = item["actions"]
        valid[row, :span] = True

    return TensorDict(
        {
            # float32 explicitly: sample_window promotes demand to float64 for exact geometry,
            # and the policy's linear layers are float32. numpy's dtype would leak through.
            "depot": torch.as_tensor(
                np.stack([item["depot"] for item in batch]), dtype=torch.float32
            ),
            "locs": torch.as_tensor(
                np.stack([item["locs"] for item in batch]), dtype=torch.float32
            ),
            "demand": torch.as_tensor(
                np.stack([item["demand"] for item in batch]), dtype=torch.float32
            ),
            "capacity": torch.as_tensor(
                np.stack([item["capacity"] for item in batch]), dtype=torch.float32
            ),
            "cost": torch.as_tensor(
                np.stack([item["cost"] for item in batch]), dtype=torch.float32
            ),
            "start_node": torch.as_tensor(
                np.stack([item["start_node"] for item in batch]), dtype=torch.int64
            ),
            "used_capacity0": torch.as_tensor(
                np.stack([item["used_capacity0"] for item in batch]), dtype=torch.float32
            ),
            "actions": torch.as_tensor(actions),
            "valid": torch.as_tensor(valid),
            "num_loc": torch.as_tensor(
                np.stack([item["num_loc"] for item in batch]), dtype=torch.int64
            ),
        },
        batch_size=[size],
    )
