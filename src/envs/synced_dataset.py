"""The supervised 16-start draw used by the step-matched SL arm.

An epoch selects instances from the HGS file and expands each into ``k`` of its ``2R``
label-feasible starts. That is the draw :class:`~src.envs.dataset.CVRPSolutionDataset`
documents but does not implement: a flat random slot draw touches ~95,000 instances at ~1 start
each, which is the opposite of "an instance with ``R = 11`` contributes 22 addressable
examples". It is also the draw the step-matched SL arm needs: 125 instances x 16 starts =
2,000 decoded sequences per optimizer step.

The old file-backed RL view (``CVRPSyncedDataset``) and the shared-draw equality checks were
retired with the data-synced experiment; the generated RL arm now uses
:class:`~src.envs.cvrp_rl_step.GeneratedStepRLCVRPEnv` and only the supervised dataset remains
here. `_TaggedFileReader` moved to :mod:`src.envs.tagged_reader` because the LEHD datasets use
it independently of this module.

The **slot index** is the shared unit of currency for the supervised draw:
:func:`sample_epoch_slots` returns ``k`` indices into an instance's ``2R`` slots, and
``divmod(slot, 2)`` recovers the ``(route_index, direction)`` pair that
:meth:`~src.envs.dataset.CVRPSolutionDataset.build_trajectory` expects.
"""

from __future__ import annotations

import os

import numpy as np

from torch.utils.data import Dataset

from src.envs.dataset import CVRPSolutionDataset, build_index
from src.envs.parser import resolve_data_path
from src.envs.tagged_reader import TaggedFileReader

__all__ = ["CVRPSyncedSolutionDataset", "sample_epoch_slots"]


def sample_epoch_slots(
    num_routes: np.ndarray,
    instance_range: tuple[int, int],
    seed: int,
    instances: int | None,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one epoch of ``(instance, start)`` pairs, grouped by instance.

    Args:
        num_routes: Per-instance route counts, as returned by
            :func:`~src.envs.dataset.build_index`.
        instance_range: Half-open ``(start, stop)`` range of instance indices to draw from.
        seed: Seed for the draw. Callers pass ``base_seed + epoch``, mirroring
            :meth:`~src.envs.CVRPEnv.dataset`, so successive epochs cover fresh data.
        instances: Number of instances to expose. ``None``, or a value at least as large as
            the range, exposes all of them.
        k: Starts to draw per instance. Must not exceed the smallest ``2R`` in the range,
            since the count has to be uniform across a batch for POMO's multi-start.

    Returns:
        A tuple ``(chosen, slots)`` where ``chosen`` is ``[M]`` int64 instance indices and
        ``slots`` is ``[M, k]`` int64 slot indices into each instance's ``2R``.

        **The order of ``chosen`` is load-bearing.** It is a random permutation of the
        selected instances, and the supervised pipeline batches sequentially over it, so batch
        ``i`` covers the intended 125 instances. Sorting it would silently break the grouping.

    Raises:
        ValueError: If ``k`` exceeds the smallest ``2R`` in the range, which would make a
            uniform start count impossible.
    """
    start, stop = int(instance_range[0]), int(instance_range[1])
    available = stop - start
    if available <= 0:
        raise ValueError(f"instance_range {instance_range} selects no instances")

    smallest = 2 * int(num_routes[start:stop].min())
    if k > smallest:
        raise ValueError(
            f"k={k} exceeds the smallest 2R={smallest} in {instance_range}; POMO's "
            "multi-start needs the same start count for every instance in a batch"
        )

    rng = np.random.default_rng(seed)
    if instances is None or instances >= available:
        chosen = np.arange(start, stop, dtype=np.int64)
        rng.shuffle(chosen)
    else:
        # `choice` already returns a random permutation, so the drawn order is preserved.
        chosen = rng.choice(available, size=int(instances), replace=False).astype(np.int64) + start

    slots = np.empty((chosen.size, k), dtype=np.int64)
    for row, instance in enumerate(chosen):
        two_r = 2 * int(num_routes[instance])
        if k == two_r:
            slots[row] = np.arange(two_r)
        else:
            # Seeded by instance rather than drawn from a running stream, so an item is
            # reproducible regardless of which worker reads it or in what order.
            slots[row] = np.random.default_rng([seed, int(instance)]).choice(
                two_r, size=k, replace=False
            )
    return chosen, slots


class CVRPSyncedSolutionDataset(TaggedFileReader, Dataset):
    """Supervised examples for one epoch, grouped so every chosen start is present.

    Each item is one ``(instance, start)`` pair, carrying the reference trajectory that
    starts there -- the same contract as
    :class:`~src.envs.dataset.CVRPSolutionDataset`, so its ``collate_fn`` is reused rather
    than copied.

    Items are indexed **instance-major** (``position * k + slot_index``), which keeps an
    instance's ``k`` items contiguous. That is what lets an unshuffled ``DataLoader`` at
    ``batch_size = instances_per_step * k`` reproduce the intended per-step grouping.

    Args:
        path: Path to a tagged CVRP solution file.
        instance_range: Half-open ``(start, stop)`` range of instances to draw from.
            Defaults to the whole file.
        instances: Number of instances to expose this epoch. ``None`` exposes all of them.
        k: Starts per instance.
        seed: Seed for the draw; callers pass ``base_seed + epoch``.
    """

    #: ``build_trajectory`` and ``collate_fn`` are reused from the verified dataset rather
    #: than reimplemented. Both are static methods, so ``staticmethod`` is required to stop
    #: the descriptor protocol from binding them to ``self``.
    collate_fn = staticmethod(CVRPSolutionDataset.collate_fn)

    def __init__(
        self,
        path: str | os.PathLike,
        instance_range: tuple[int, int] | None = None,
        instances: int | None = None,
        k: int = 16,
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
        self.k = int(k)
        self.instances, self.slots = sample_epoch_slots(
            self.num_routes, self.instance_range, int(seed), instances, self.k
        )

    def __len__(self) -> int:
        """Number of trajectories exposed by this epoch."""
        return int(self.instances.size) * self.k

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        """Return the trajectory at ``index`` as flat numpy arrays.

        Args:
            index: Position in ``[0, len(self))``, interpreted instance-major.

        Returns:
            A dict with the instance fields (``depot``, ``locs``, ``demand``, ``capacity``,
            ``cost``) plus ``actions``, ``start_node`` and ``num_routes``.
        """
        position, slot_index = divmod(int(index), self.k)
        instance = int(self.instances[position])
        slot = int(self.slots[position, slot_index])

        record = self._read_record(instance)
        actions = CVRPSolutionDataset.build_trajectory(
            record["tour"], record["flags"], *divmod(slot, 2)
        )
        record["actions"] = actions
        record["start_node"] = np.int64(actions[0])
        record["num_routes"] = np.int64(self.num_routes[instance])
        record.pop("tour")
        record.pop("flags")
        return record
