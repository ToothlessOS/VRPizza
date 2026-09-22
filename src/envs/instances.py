"""Whole-file instances for evaluation, read once and held in memory.

This exists alongside :mod:`src.envs.dataset` rather than inside it because evaluation and
supervised training want opposite things from the same files. Training needs a lazily indexed,
trajectory-level pool it can sample from for 200 epochs -- hence byte offsets, ``os.pread``
and per-worker file descriptors. Evaluation reads each file exactly once, in order, and needs
one record per *instance* rather than the ``2R`` optimal starts a route can be entered from.
Indexing all five test files costs 1.4 s and 11 MiB, so the lazy machinery would be pure
overhead.

Two properties of the corpus shape the code here:

* ``InstanceLayout`` stores *absolute* token indices, so a layout derived from one record is
  only valid for another record with the same number of customers. ``vrplib_192instances.txt``
  mixes sizes from 15 to 1000, and on that file a file-wide layout raises on 19 of its 192
  records and silently mis-slices ``locs`` and ``demand`` on most of the rest. Every record
  therefore gets its own layout.
* Reading is sequential, so nothing is written next to the data -- the shared
  :func:`~src.envs.dataset.build_index` cache would drop ``.offsets.npy`` and
  ``.num_routes.npy`` files into the test-data directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.envs.parser import InstanceLayout, resolve_data_path

__all__ = ["InstanceSet", "load_instances"]


@dataclass(frozen=True)
class InstanceSet:
    """Every record of one CVRP file, in file order, in the file's own units.

    Attributes:
        name: The file's stem, used as the dataset label in reports.
        path: Resolved path the records were read from.
        num_loc: ``[M]`` int32 customer counts. Not constant when the file mixes sizes.
        capacity: ``[M]`` float32 vehicle capacities, one per record.
        cost: ``[M]`` float32 reference objectives. NaN when the file carries none.
        depot: ``M`` arrays of shape ``[2]``, float32.
        locs: ``M`` arrays of shape ``[N_i, 2]``, float32, depot excluded.
        demand: ``M`` arrays of shape ``[N_i]``, float32.
        tour: ``M`` arrays of shape ``[N_i]`` giving the reference tour as a permutation of
            ``1..N_i``, or ``None`` for a file whose records carry no solution.
        flags: ``M`` arrays of shape ``[N_i]`` marking route starts by tour position, or
            ``None`` alongside ``tour``.
    """

    name: str
    path: str
    num_loc: np.ndarray
    capacity: np.ndarray
    cost: np.ndarray
    depot: list[np.ndarray]
    locs: list[np.ndarray]
    demand: list[np.ndarray]
    tour: list[np.ndarray] | None = None
    flags: list[np.ndarray] | None = None

    def __len__(self) -> int:
        """Number of records read."""
        return int(self.num_loc.size)

    @property
    def has_tour(self) -> bool:
        """Whether the records carry reference tours."""
        return self.tour is not None

    def groups_by_n(self) -> dict[int, np.ndarray]:
        """Partition instance positions by customer count, in ascending size order.

        Multi-start inference batches a whole instance at once, so every member of a batch
        must have the same ``N``: the decoder expands the batch by ``num_starts = N`` and the
        environment's action mask is ``[B, N+1]``. Padding a mixed batch up to the largest
        ``N`` would instead spend starts on padding and cost ``(max_n / N) ** 3`` extra decode
        time on the small members, so sizes are grouped rather than padded.

        Returns:
            A dict mapping ``N`` to the ascending positions of the instances of that size.
            Iterating it yields sizes in ascending order, which puts the cheap groups first.
        """
        groups: dict[int, list[int]] = {}
        for position, num_loc in enumerate(self.num_loc.tolist()):
            groups.setdefault(int(num_loc), []).append(position)
        return {
            n: np.asarray(positions, dtype=np.int64) for n, positions in sorted(groups.items())
        }

    def model_records(
        self, positions: np.ndarray, normalize: bool = False
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        """Return the requested records with coordinates in the units the model expects.

        The network is trained on coordinates in ``[0, 1]`` and embeds them linearly, so the
        scale it is fed is part of the input distribution rather than a free choice. The LKH
        files already occupy ``[0, 1]`` and are passed through untouched. ``vrplib`` is in raw
        integer coordinates whose per-instance extent varies widely, and feeding those
        unscaled is not a rounding error: measured over eight instances, the mean gap is 69.4%
        raw against 4.4% rescaled.

        A single pooled scalar per instance -- ``(x - min) / (max - min)`` over both axes --
        preserves the aspect ratio and so the Euclidean metric up to a constant factor. The
        reference ``cost`` is divided by the same scalar, which leaves the optimality gap
        unchanged while keeping ``tour * scale`` in the file's own units. Per-axis scaling
        would break that back-conversion, since no single scalar would invert it.

        Args:
            positions: Instance positions to return.
            normalize: Whether to rescale each instance into ``[0, 1]``.

        Returns:
            A pair ``(records, scale)``. ``records`` are dicts of float32 arrays keyed
            ``depot``, ``locs``, ``demand``, ``capacity`` and ``cost``; ``scale`` is ``[P]``
            and converts a length in model units back to the file's units (``1.0`` when
            ``normalize`` is false).
        """
        records: list[dict[str, np.ndarray]] = []
        scales = np.ones(len(positions), dtype=np.float64)

        for slot, position in enumerate(np.asarray(positions).tolist()):
            depot = self.depot[position]
            locs = self.locs[position]
            cost = self.cost[position]

            if normalize:
                coords = np.vstack([depot[None], locs])
                low = float(coords.min())
                extent = float(coords.max()) - low
                # Degenerate only for a file of coincident points, which no CVRP instance is;
                # leaving it at 1.0 keeps the scales finite rather than dividing by zero.
                scale = extent if extent > 0.0 else 1.0
                depot = (depot - low) / scale
                locs = (locs - low) / scale
                cost = cost / scale
                scales[slot] = scale

            records.append(
                {
                    "depot": np.asarray(depot, dtype=np.float32),
                    "locs": np.asarray(locs, dtype=np.float32),
                    "demand": np.asarray(self.demand[position], dtype=np.float32),
                    "capacity": np.float32(self.capacity[position]),
                    "cost": np.float32(cost),
                }
            )

        return records, scales

    def ref_cost_from_tour(self, position: int) -> float:
        """Recompute an instance's objective from its reference tour.

        A cross-check that validates the parser against the data rather than against itself:
        the stored ``cost`` is an independent number produced by LKH or HGS, so agreement to
        machine precision on a file whose coordinates are in ``[0, 1]`` shows the coordinates,
        the demand section and the tour were all sliced at the right offsets.

        Args:
            position: Instance position in file order.

        Returns:
            The tour length in the file's own units, or ``nan`` when the file carries no tour.
        """
        if self.tour is None or self.flags is None:
            return float("nan")

        tour, flags = self.tour[position], self.flags[position]
        coords = np.vstack([self.depot[position][None], self.locs[position]])

        cuts = np.flatnonzero(flags == 1)
        if cuts.size == 0 or cuts[0] != 0:
            raise ValueError(f"instance {position} has no route start at tour position 0")

        total = 0.0
        for route in np.split(tour, cuts[1:]):
            # Depot out, the route's customers in order, depot back. `coords[0]` is the depot
            # because `locs` excludes it, matching the file's own node indexing.
            path = coords[np.concatenate([[0], route, [0]])]
            total += float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        return total


def load_instances(
    path: str | os.PathLike, limit: int | None = None, offset: int = 0
) -> InstanceSet:
    """Read a contiguous run of records from a tagged CVRP file, in order.

    Args:
        path: Path to a tagged CVRP ``.txt`` file, absolute or relative to the repository
            root.
        limit: Read at most ``limit`` records. ``None`` reads to the end.
        offset: Skip this many records first. Both training runs carved their validation set
            from the tail of the 4.6 GB HGS training file -- ``val_range`` is
            ``(999_000, 1_000_000)`` -- so reaching it is what allows the scored numbers to be
            checked against the ``val/reward`` the checkpoints were selected on. The skipped
            records are read but not parsed, which for a million lines is seconds rather than
            minutes and keeps peak memory proportional to ``limit`` rather than to the file.

    Returns:
        The parsed :class:`InstanceSet`.

    Raises:
        ValueError: If the file mixes records that carry a reference tour with records that
            do not, which the per-file ``tour`` field could not represent.
    """
    resolved = resolve_data_path(path)

    num_loc: list[int] = []
    capacity: list[np.float32] = []
    cost: list[np.float32] = []
    depot: list[np.ndarray] = []
    locs: list[np.ndarray] = []
    demand: list[np.ndarray] = []
    tours: list[np.ndarray] = []
    flags: list[np.ndarray] = []

    # Decided by the first parsed record, then enforced: the file either carries a solution
    # section on every line or on none of them.
    carries_tour: bool | None = None

    with open(resolved) as handle:
        for _ in range(offset):
            if not handle.readline():
                break

        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break

            layout = InstanceLayout.from_tokens(line.split(","))
            record = layout.parse(line)

            if carries_tour is None:
                carries_tour = layout.has_tour
            elif carries_tour != layout.has_tour:
                raise ValueError(
                    f"{resolved} record {offset + index} "
                    f"{'has' if layout.has_tour else 'has no'} reference tour, unlike the "
                    f"first record read; the file is not uniform"
                )

            num_loc.append(layout.num_loc)
            capacity.append(record["capacity"])
            cost.append(record["cost"])
            depot.append(record["depot"])
            locs.append(record["locs"])
            demand.append(record["demand"])
            if carries_tour:
                tours.append(record["tour"])
                flags.append(record["flags"])

    return InstanceSet(
        name=Path(resolved).stem,
        path=resolved,
        num_loc=np.asarray(num_loc, dtype=np.int32),
        capacity=np.asarray(capacity, dtype=np.float32),
        cost=np.asarray(cost, dtype=np.float32),
        depot=depot,
        locs=locs,
        demand=demand,
        tour=tours or None,
        flags=flags or None,
    )
