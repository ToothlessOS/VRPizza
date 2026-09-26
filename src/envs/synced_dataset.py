"""A shared, instance-grouped data draw for comparing supervised and RL POMO.

:mod:`src.envs.dataset` exposes every ``(instance, route, direction)`` trajectory of the
reference files as a flat pool of ``sum(2R)`` slots and samples ``size`` of them uniformly.
That is not the draw the supervised pipeline documents: because an instance owns only ~21
of the 21M slots, a 100,000-slot epoch touches ~95,300 instances at ~1.05 starts each, so
the ``2R`` starts of an instance are never seen alongside each other and the arithmetic in
``POMO_sl.py`` ("divide by ~21 to get distinct instances") does not hold.

This module implements the draw those comments describe, once, for both objectives. An
epoch selects a set of instances and, for each, ``k`` of its ``2R`` feasible starts; the
supervised and reinforcement-learning datasets then differ only in how they present the
result -- a full teacher-forcing trajectory per ``(instance, start)``, or one row per
instance carrying its ``k`` start nodes for POMO's multi-start. Neither dataset samples on
its own, so the two cannot drift apart.

The **slot index** is the shared unit of currency: :func:`sample_epoch_slots` returns
``k`` indices into an instance's ``2R`` slots, and ``divmod(slot, 2)`` recovers the
``(route_index, direction)`` pair that
:meth:`~src.envs.dataset.CVRPSolutionDataset.build_trajectory` expects and that
:func:`feasible_starts` enumerates.

Deliberate duplication: :class:`_TaggedFileReader` repeats the per-worker file-descriptor
handling of ``CVRPSolutionDataset._worker_fd``/``_read_record``
(:mod:`src.envs.dataset`, lines 176-198) rather than extracting a shared base, so that
adding this pipeline cannot change the behaviour of the verified one. The two are kept
byte-compatible on purpose; if they ever diverge, that is a bug in this file.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from tensordict.tensordict import TensorDict
from torch.utils.data import Dataset, get_worker_info

from src.envs.dataset import CVRPSolutionDataset, build_index
from src.envs.parser import resolve_data_path

__all__ = [
    "CVRPSyncedDataset",
    "CVRPSyncedSolutionDataset",
    "feasible_starts",
    "sample_epoch_slots",
]


def feasible_starts(tour: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """Return the ``2R`` starts a reference solution makes legal, in slot order.

    A CVRP rollout must begin by traversing a depot edge the solution already contains,
    so the only legal starting points are each route's first customer (served forwards)
    and its last (served backwards). This is exactly the set
    :meth:`~src.envs.dataset.CVRPSolutionDataset.build_trajectory` walks as it enumerates
    ``(route_index, direction)`` pairs.

    The layout is *interleaved* to match ``divmod(slot, 2)``: index ``2k`` is route
    ``k``'s first customer and index ``2k+1`` its last, so a slot index alone determines
    both the start node and the trajectory that follows from it.

    Args:
        tour: Permutation of ``1..N`` in visit order.
        flags: Binary route-start flags indexed by tour position.

    Returns:
        An int64 array of length ``2R``. Note a single-customer route contributes the same
        customer twice -- once forwards, once backwards -- which mirrors the duplicate
        trajectory ``build_trajectory`` would produce for it.

    Raises:
        ValueError: If the flags do not begin with a route start, which would leave the
            first route undefined.
    """
    cuts = np.flatnonzero(flags == 1)
    if cuts.size == 0 or cuts[0] != 0:
        raise ValueError("Route flags must begin with a 1 marking the first route")

    routes = np.split(tour, cuts[1:])
    starts = np.empty(2 * len(routes), dtype=np.int64)
    starts[0::2] = [route[0] for route in routes]
    starts[1::2] = [route[-1] for route in routes]
    return starts


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
        selected instances, and both pipelines batch sequentially over it, so that batch
        ``i`` covers the same 64 instances on either side. Sorting it would silently break
        the pairing.

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


class _TaggedFileReader:
    """Fork-safe access to the records of a tagged CVRP file.

    Deliberately duplicates ``CVRPSolutionDataset._worker_fd``/``_read_record``
    (:mod:`src.envs.dataset`, lines 176-198) so that the synced pipeline can be added
    without editing the verified module. See this module's docstring.

    Subclasses must set ``path``, ``offsets`` and ``layout`` before reading.
    """

    def __init__(self) -> None:
        self._fd: int | None = None
        self._fd_owner: int = -2  # sentinel distinct from the main process (-1)

    def _worker_fd(self) -> int:
        """Return a file descriptor private to the current worker."""
        owner = -1 if (worker := get_worker_info()) is None else worker.id
        if self._fd is None or self._fd_owner != owner:
            self._fd = os.open(self.path, os.O_RDONLY)
            self._fd_owner = owner
        return self._fd

    def _read_record(self, index: int) -> dict[str, np.ndarray]:
        """Read and parse instance ``index``."""
        fd = self._worker_fd()
        begin = int(self.offsets[index])
        raw = os.pread(fd, int(self.offsets[index + 1]) - begin, begin)
        return self.layout.parse(raw.decode())


class CVRPSyncedSolutionDataset(_TaggedFileReader, Dataset):
    """Supervised examples for one epoch, grouped so every start of an instance is present.

    Each item is one ``(instance, start)`` pair, carrying the reference trajectory that
    starts there -- the same contract as
    :class:`~src.envs.dataset.CVRPSolutionDataset`, so its ``collate_fn`` is reused rather
    than copied.

    Items are indexed **instance-major** (``position * k + slot_index``), which keeps an
    instance's ``k`` items contiguous. That is what lets an unshuffled ``DataLoader`` at
    ``batch_size = instances_per_batch * k`` reproduce the same per-step grouping the RL
    pipeline gets from POMO's multi-start. Row-level shuffling would break the pairing
    without changing the loss value -- cross-entropy is a mean over rows -- so it would
    only show up as a silent change in gradient variance.

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


class CVRPSyncedDataset(_TaggedFileReader, Dataset):
    """One row per instance, carrying its ``k`` label-feasible start nodes.

    This is the reinforcement-learning view of the same draw: POMO's multi-start expands
    each row into ``k`` rollouts, so the starts travel with the instance instead of being
    enumerated as separate rows. No reference trajectory is exposed -- the training target
    is the reward, not the label.

    Args:
        path: Path to a tagged CVRP solution file.
        instance_range: Half-open ``(start, stop)`` range of instances to draw from.
            Defaults to the whole file.
        instances: Number of instances to expose this epoch. ``None`` exposes all of them.
        k: Starts per instance.
        seed: Seed for the draw; callers pass ``base_seed + epoch``.
    """

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
        """Number of instances exposed by this epoch."""
        return int(self.instances.size)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        """Return instance ``index`` with its start nodes.

        Args:
            index: Position in ``[0, len(self))``.

        Returns:
            A dict with the instance fields plus ``start_nodes`` ``[k]`` and ``num_routes``.
        """
        instance = int(self.instances[int(index)])
        record = self._read_record(instance)
        record["start_nodes"] = feasible_starts(record["tour"], record["flags"])[self.slots[index]]
        record["num_routes"] = np.int64(self.num_routes[instance])
        record.pop("tour")
        record.pop("flags")
        return record

    @staticmethod
    def collate_fn(batch: list[dict[str, np.ndarray]]) -> TensorDict:
        """Collate instances into a :class:`~tensordict.TensorDict` the RL env can consume.

        Differs from :meth:`~src.envs.dataset.CVRPSolutionDataset.collate_fn` in two
        deliberate ways. ``demand`` is **normalised by capacity here**, because
        ``POMO.shared_step`` passes the batch straight to ``env.reset`` and never goes
        through :meth:`~src.envs.InstanceInputMixin.make_input_td`, which is where the
        supervised path applies the same division; unnormalised demand would scale the
        action mask by ``capacity``. And no ``actions``/``valid`` are emitted, since the
        reinforcement-learning target is label-free.

        Args:
            batch: A list of items as returned by :meth:`__getitem__`.

        Returns:
            A TensorDict with ``depot`` ``[B,2]``, ``locs`` ``[B,N,2]``, ``demand`` ``[B,N]``
            (normalised), ``capacity`` ``[B,1]``, ``start_nodes`` ``[B,k]``, ``cost`` ``[B]``
            and ``num_routes`` ``[B]``.
        """
        size = len(batch)
        capacity = np.stack([item["capacity"] for item in batch]).reshape(size, 1)
        demand = np.stack([item["demand"] for item in batch])

        return TensorDict(
            {
                "depot": torch.as_tensor(np.stack([item["depot"] for item in batch])),
                "locs": torch.as_tensor(np.stack([item["locs"] for item in batch])),
                "demand": torch.as_tensor(demand / capacity),
                "capacity": torch.as_tensor(capacity),
                "start_nodes": torch.as_tensor(np.stack([item["start_nodes"] for item in batch])),
                "cost": torch.as_tensor(np.stack([item["cost"] for item in batch])),
                "num_routes": torch.as_tensor(
                    np.stack([item["num_routes"] for item in batch]), dtype=torch.int64
                ),
            },
            batch_size=[size],
        )
