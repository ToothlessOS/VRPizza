"""Lazy, trajectory-level dataset over the tagged CVRP solution files.

A record in ``data/CVRP`` yields not one training example but ``2R`` of them, where ``R``
is the number of routes in its reference solution. The first action of a CVRP rollout
must traverse a depot edge the solution already contains, so the only legal starting
points are a route's first customer (travelled forwards) and its last customer
(travelled backwards) -- and for Euclidean distances both reproduce the reference cost
exactly. See :meth:`CVRPSolutionDataset.build_trajectory`.

Rather than sampling one start per instance, this dataset indexes *every* such trajectory,
so an instance with ``R = 11`` contributes 22 addressable training examples instead of
one. The global trajectory index maps back to ``(instance, route, direction)`` through a
cached per-file route-count array.
"""

from __future__ import annotations

import os

from pathlib import Path

import numpy as np
import torch

from tensordict.tensordict import TensorDict
from torch.utils.data import Dataset, get_worker_info

from src.envs.parser import InstanceLayout, resolve_data_path, scan_offsets

__all__ = ["CVRPSolutionDataset", "build_index"]

_OFFSETS_SUFFIX = ".offsets.npy"
_ROUTES_SUFFIX = ".num_routes.npy"


def build_index(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray, InstanceLayout]:
    """Build (or load) the byte-offset and route-count index for a CVRP solution file.

    Args:
        path: Path to a tagged CVRP ``.txt`` file.

    Returns:
        A tuple ``(offsets, num_routes, layout)`` where ``offsets`` holds line byte
        offsets with a trailing file-size sentinel and ``num_routes[i]`` is the number
        of routes in instance ``i``.
    """
    path = resolve_data_path(path)
    layout = InstanceLayout.from_tokens(_first_line_tokens(path))
    offsets_path = Path(path + _OFFSETS_SUFFIX)
    routes_path = Path(path + _ROUTES_SUFFIX)

    if offsets_path.exists():
        offsets = np.load(offsets_path)
    else:
        offsets = scan_offsets(path)
        np.save(offsets_path, offsets)

    if routes_path.exists():
        num_routes = np.load(routes_path)
    else:
        num_routes = _scan_num_routes(path, offsets, layout)
        np.save(routes_path, num_routes)

    return offsets, num_routes, layout


def _first_line_tokens(path: str) -> list[str]:
    """Return the comma-split tokens of the first record in ``path``."""
    with open(path) as handle:
        return handle.readline().split(",")


def _scan_num_routes(path: str, offsets: np.ndarray, layout: InstanceLayout) -> np.ndarray:
    """Count the routes of every instance by scanning the file once.

    Reads bytes rather than decoded text and uses ``bytes.count`` on the flag slice, so
    the 4.6 GB training file is indexed in well under two minutes.

    Args:
        path: Path to the file.
        offsets: Line byte offsets as returned by :func:`~src.envs.parser.scan_offsets`.
        layout: The file's :class:`~src.envs.parser.InstanceLayout`.

    Returns:
        An int16 array of per-instance route counts.
    """
    if not layout.has_tour:
        raise ValueError(f"{path} carries no reference tour; it is usable for evaluation only")

    n = layout.num_loc
    start = layout.node_flag_idx + 1 + n
    stop = start + n
    counts = np.empty(offsets.size - 1, dtype=np.int16)

    with open(path, "rb") as handle:
        for i in range(offsets.size - 1):
            tokens = handle.read(offsets[i + 1] - offsets[i]).split(b",")
            flags = tokens[start:stop]
            # Only the very last token of a record can carry a trailing newline.
            counts[i] = flags[:-1].count(b"1") + (1 if flags[-1].strip() == b"1" else 0)

    return counts


class CVRPSolutionDataset(Dataset):
    """A trajectory-level dataset of supervised CVRP examples.

    Each item is one ``(instance, optimal start)`` pair. The dataset length is
    ``sum(2 * num_routes[i])`` over the selected instance range, so an ``N=100`` file with
    a mean of 10.5 routes exposes roughly 21 training examples per instance.

    Args:
        path: Path to a tagged CVRP solution file.
        instance_range: Half-open ``(start, stop)`` range of instance indices to draw
            from. Defaults to the whole file.
        size: Number of trajectories to expose this epoch. When smaller than the pool the
            subset is drawn without replacement using ``seed``, so successive epochs
            cover fresh data. ``None`` exposes the whole pool.
        seed: Seed for the subset draw.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        instance_range: tuple[int, int] | None = None,
        size: int | None = None,
        seed: int = 0,
    ) -> None:
        self.path = resolve_data_path(path)
        self.offsets, self.num_routes, self.layout = build_index(self.path)
        self.num_instances = int(self.num_routes.size)

        start, stop = instance_range if instance_range is not None else (0, self.num_instances)
        self.instance_range = (int(start), int(stop))

        # Local cumulative trajectory counts, so a pool index maps to an instance with a
        # single searchsorted. Each instance contributes `2R` trajectories -- each route can
        # be served first from either end -- and the `divmod(slot, 2)` in `__getitem__`
        # unpacks exactly that, so the running total is over `2R` and not `R`.
        self._csum = np.concatenate(
            [[0], 2 * np.cumsum(self.num_routes[start:stop].astype(np.int64))]
        )
        pool_size = int(self._csum[-1])

        if size is None or size >= pool_size:
            self._pool: np.ndarray | None = None
            self._length = pool_size
        else:
            rng = np.random.default_rng(seed)
            self._pool = rng.choice(pool_size, size=int(size), replace=False).astype(np.int64)
            self._length = int(size)

        self._fd: int | None = None
        self._fd_owner: int = -2  # sentinel distinct from the main process (-1)

    # -- introspection -----------------------------------------------------------

    @property
    def num_loc(self) -> int:
        """Number of customers per instance."""
        return self.layout.num_loc

    @property
    def pool_size(self) -> int:
        """Total number of trajectories in the selected instance range."""
        return int(self._csum[-1])

    @property
    def mean_trajectory_length(self) -> float:
        """Mean action-sequence length, ``N + mean(R)``."""
        start, stop = self.instance_range
        return self.layout.num_loc + float(self.num_routes[start:stop].mean())

    # -- data access -------------------------------------------------------------

    def _worker_fd(self) -> int:
        """Return a file descriptor private to the current worker.

        Forked DataLoader workers inherit a parent's file descriptor, so a single shared
        handle would interleave reads between workers and return corrupt lines. Each
        worker opens its own descriptor here, and reads go through :func:`os.pread`,
        which leaves the shared file offset untouched.

        Returns:
            An open read-only file descriptor owned by the calling process.
        """
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

    def __len__(self) -> int:
        """Number of trajectories exposed by this dataset."""
        return self._length

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        """Return the trajectory at ``index`` as flat numpy arrays.

        Args:
            index: Position in ``[0, len(self))``.

        Returns:
            A dict with the instance fields (``depot``, ``locs``, ``demand``,
            ``capacity``, ``cost``) plus ``actions``, ``start_node`` and ``num_routes``.
        """
        pool_index = int(self._pool[index]) if self._pool is not None else int(index)
        route_slot = int(np.searchsorted(self._csum, pool_index, side="right")) - 1
        instance = self.instance_range[0] + route_slot
        slot = pool_index - int(self._csum[route_slot])
        route_index, direction = divmod(slot, 2)

        record = self._read_record(instance)
        actions = self.build_trajectory(record["tour"], record["flags"], route_index, direction)
        record["actions"] = actions
        record["start_node"] = np.int64(actions[0])
        record["num_routes"] = self.num_routes[instance]
        record.pop("tour")
        record.pop("flags")
        return record

    @staticmethod
    def build_trajectory(
        tour: np.ndarray, flags: np.ndarray, route_index: int, direction: int
    ) -> np.ndarray:
        """Build the full depot-returning action sequence for one optimal start.

        ``flags`` marks route starts by tour position, so the routes are recovered by
        cutting ``tour`` at those positions. The chosen route is served first -- forwards
        from its first customer when ``direction == 0``, backwards from its last customer
        when ``direction == 1`` -- and the remaining routes follow in their original
        relative order. Both variants reproduce the reference cost exactly, and every
        route is terminated by a depot visit (action ``0``).

        Args:
            tour: Permutation of ``1..N`` in visit order.
            flags: Binary route-start flags indexed by tour position.
            route_index: Which route to serve first.
            direction: ``0`` for forwards, ``1`` for backwards.

        Returns:
            An int64 action sequence of length ``N + R``.
        """
        cuts = np.flatnonzero(flags == 1)
        if cuts.size == 0 or cuts[0] != 0:
            raise ValueError("Route flags must begin with a 1 marking the first route")

        routes = np.split(tour, cuts[1:])
        order = [route_index] + [k for k in range(len(routes)) if k != route_index]

        sequence: list[int] = []
        for position, k in enumerate(order):
            route = routes[k][::-1] if (position == 0 and direction == 1) else routes[k]
            sequence.extend(route.tolist())
            sequence.append(0)

        return np.asarray(sequence, dtype=np.int64)

    @staticmethod
    def collate_fn(batch: list[dict[str, np.ndarray]]) -> TensorDict:
        """Collate trajectories into a :class:`~tensordict.TensorDict`, padding with depot.

        Action sequences of different instances have different lengths (``N + R`` varies
        with the route count), so they are right-padded. Padding with ``0`` is safe:
        a depot-to-depot hop has zero length, so the padded reward is unchanged, and the
        ``valid`` mask keeps the padded steps out of the loss.

        Args:
            batch: A list of items as returned by :meth:`__getitem__`.

        Returns:
            A TensorDict with batched instance fields, ``actions`` ``[B, T]``,
            ``valid`` ``[B, T]``, ``num_routes`` ``[B]`` and ``start_node`` ``[B]``.
        """
        size = len(batch)
        length = max(item["actions"].size for item in batch)

        actions = np.zeros((size, length), dtype=np.int64)
        valid = np.zeros((size, length), dtype=bool)
        for i, item in enumerate(batch):
            span = item["actions"].size
            actions[i, :span] = item["actions"]
            valid[i, :span] = True

        return TensorDict(
            {
                "depot": torch.as_tensor(np.stack([b["depot"] for b in batch])),
                "locs": torch.as_tensor(np.stack([b["locs"] for b in batch])),
                "demand": torch.as_tensor(np.stack([b["demand"] for b in batch])),
                "capacity": torch.as_tensor(np.stack([b["capacity"] for b in batch])),
                "cost": torch.as_tensor(np.stack([b["cost"] for b in batch])),
                "actions": torch.as_tensor(actions),
                "valid": torch.as_tensor(valid),
                "num_routes": torch.as_tensor(np.stack([b["num_routes"] for b in batch])),
                "start_node": torch.as_tensor(np.stack([b["start_node"] for b in batch])),
            },
            batch_size=[size],
        )
