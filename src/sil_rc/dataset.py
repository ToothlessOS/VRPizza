"""In-memory LEHD-style partial windows over SIL pseudo-labels.

This is the analogue of :class:`~src.envs.lehd_step_dataset.LEHDStepPartialDataset`
for labels that live in memory instead of a tagged file. One item is a contiguous
window of ``w`` customers ending at a route end, cut from the current improved
solution; the environment replays it from the forced start and the loss is the
teacher-forced cross-entropy, exactly as in the LEHD-SL arm.

Two properties are load-bearing and copied from the LEHD design:

* **One ``w`` per effective optimizer batch.** The reference SIL trainer draws one
  ``fix_length`` per training batch (``Trainer.train_one_batch``), not per instance.
  A ``Dataset.__getitem__`` cannot see its batch, so ``w`` is keyed by
  ``index // effective_batch_size`` and the DataLoader must not shuffle.
  :func:`~src.envs.lehd_dataset.collate_partials` rejects a micro-batch whose rows
  disagree on ``w``, which catches a misconfigured accumulation factor.
* **Per-item seeding.** The route-end choice and flips come from
  ``default_rng([seed, label_index, window_index])``, never a shared stream, so an
  item is identical regardless of worker count or read order. ``seed`` is
  ``base + epoch`` at the call site, so windows change across epochs while the
  instance permutation does too.

Rows are instance-major: the ``k`` windows of a label are consecutive. The
step-matched arm uses ``k=16`` (125 instances x 16 = 2,000 partials per effective
batch); the strict arm uses ``k=1``, matching the reference's one sub-path per
instance per epoch.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from torch.utils.data import Dataset

from src.envs.lehd_dataset import collate_partials
from src.lehd.subpath import sample_window
from src.sil_rc.labels import LabelStore

__all__ = ["SILPartialDataset"]


class SILPartialDataset(Dataset):
    """Sampled partial windows from a :class:`~src.sil_rc.labels.LabelStore`.

    Args:
        store: The current pseudo-label store.
        partials_per_instance: Windows emitted per label per epoch. ``1`` for the
            strict port, ``16`` for the step-matched protocol.
        effective_batch_size: Rows per optimizer step (micro x accumulation). Keys
            the one-``w``-per-step draw.
        w_min: Minimum window length, inclusive. The reference uses 4.
        w_max: Maximum window length, inclusive. Defaults to the store's ``N``.
        seed: Base seed; callers pass ``seed + epoch``.
        label_indices: Global store rows to expose, or ``None`` for the whole
            store. A chunked SIL sub-cycle passes its chunk.
    """

    collate_fn = staticmethod(collate_partials)

    def __init__(
        self,
        store: LabelStore,
        partials_per_instance: int = 1,
        effective_batch_size: int = 1024,
        w_min: int = 4,
        w_max: int | None = None,
        seed: int = 0,
        label_indices: Sequence[int] | None = None,
    ) -> None:
        self.store = store
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
        if len(store) <= 0:
            raise ValueError("SILPartialDataset needs at least one labelled instance")

        self.w_min = int(w_min)
        self.w_max = int(store.num_loc if w_max is None else w_max)
        if not 2 <= self.w_min <= self.w_max <= store.num_loc:
            raise ValueError(
                f"window range [{self.w_min}, {self.w_max}] is not inside "
                f"[2, {store.num_loc}]"
            )

        rng = np.random.default_rng(int(seed))
        if label_indices is None:
            self.label_indices = None
            self.order = rng.permutation(len(store)).astype(np.int64)
            rows = len(store) * self.partials_per_instance
        else:
            subset = np.asarray(label_indices, dtype=np.int64)
            if subset.ndim != 1 or subset.size == 0:
                raise ValueError("label_indices must be a non-empty 1-D sequence")
            if subset.min() < 0 or subset.max() >= len(store):
                raise ValueError(
                    f"label_indices must be in [0, {len(store)}), got "
                    f"[{subset.min()}, {subset.max()}]"
                )
            if np.unique(subset).size != subset.size:
                raise ValueError("label_indices must not contain duplicates")
            self.label_indices = subset
            self.order = subset[rng.permutation(subset.size)].astype(np.int64)
            rows = subset.size * self.partials_per_instance
        chunks = -(-rows // self.effective_batch_size)
        self.w_per_chunk = np.asarray(
            [
                np.random.default_rng([int(seed), chunk]).integers(
                    self.w_min, self.w_max + 1
                )
                for chunk in range(chunks)
            ],
            dtype=np.int64,
        )
        self.seed = int(seed)

    def __len__(self) -> int:
        """Number of partial solutions exposed this epoch."""
        return int(self.order.size) * self.partials_per_instance

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.generic]:
        """Return one partial solution as the ``collate_partials`` contract."""
        index = int(index)
        position, window_index = divmod(index, self.partials_per_instance)
        label_index = int(self.order[position])
        chunk = index // self.effective_batch_size
        w = int(self.w_per_chunk[chunk])

        routes = self.store.routes_for(label_index)
        rng = np.random.default_rng([self.seed, label_index, window_index])
        window = sample_window(
            routes,
            self.store.locs[label_index],
            self.store.demand[label_index],
            w,
            rng,
        )

        return {
            "depot": self.store.depot[label_index],
            "locs": window.locs_w,
            "demand": window.demand_w,
            "capacity": self.store.capacity[label_index],
            "cost": self.store.cost[label_index],
            "start_node": np.int64(window.start_local),
            "used_capacity0": np.float32(window.used_capacity0),
            "actions": window.actions,
            "num_loc": np.int64(w),
        }
