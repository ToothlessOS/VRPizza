"""Parsing utilities for the tagged CVRP instance files under ``data/CVRP``.

Each line of a ``.txt`` file holds one instance as a flat, comma-separated, tagged
record::

    depot,x,y,customer,x1,y1,...,xN,yN,capacity,<cap>,demand,<d1..dN>,cost,<ref>,node_flag,<A>,<B>

``A`` is the reference tour -- a permutation of ``1..N`` in visit order -- and ``B`` is a
binary route-start flag array indexed by *tour position*, so cutting ``A`` at the
positions where ``B == 1`` recovers the routes of the solution.

Two quirks of the corpus motivate the dynamic label scanning done here:

* The HGS training files carry an extra leading ``0`` in the demand section (a depot
  slot) that the LKH test files do not.
* ``vrplib_192instances.txt`` is a Python-literal dump: its labels appear in a different
  order (``demand`` before ``capacity``), its tokens are quoted and bracketed, and it has
  no tour at all.

Locating labels by scanning for non-numeric tokens rather than by fixed offsets keeps a
single parser working across all of them.
"""

from __future__ import annotations

import mmap
import os

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "LABELS",
    "InstanceLayout",
    "clean_token",
    "project_root",
    "resolve_data_path",
    "scan_offsets",
]

#: Token values that delimit the tagged sections of a record.
LABELS: tuple[str, ...] = (
    "depot",
    "customer",
    "capacity",
    "demand",
    "cost",
    "node_flag",
    "end",
)

_SCAN_CHUNK = 1 << 26  # 64 MiB


def project_root() -> Path:
    """Return the repository root, located by walking up from this file.

    Returns:
        The directory containing the ``.project-root`` marker, or the current working
        directory if no marker is found.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / ".project-root").exists():
            return parent
    return Path.cwd()


def resolve_data_path(path: str | os.PathLike) -> str:
    """Resolve a data path against the repository root rather than the working directory.

    The data files live at a fixed place in the repository, and ``rootutils.setup_root``
    only puts the repository on ``sys.path`` -- it does not change the working directory.
    Without this, running a script from anywhere but the repository root fails with a
    ``FileNotFoundError`` on the data path.

    Args:
        path: An absolute path, or a path relative to the repository root.

    Returns:
        The resolved path as a string.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    rooted = project_root() / candidate
    # Prefer the repository-relative location, but honour a working-directory-relative
    # path if that is where the file actually lives.
    if rooted.exists() or not candidate.exists():
        return str(rooted)
    return str(candidate)


def clean_token(token: str) -> str:
    """Strip whitespace, list brackets and quotes from a raw token.

    Args:
        token: A raw comma-split token, possibly quoted or bracketed as in the
            ``vrplib`` literal format.

    Returns:
        The bare token, safe to compare against :data:`LABELS`.
    """
    return token.strip().strip("[]").strip("'\"").strip()


@dataclass(frozen=True)
class InstanceLayout:
    """Token layout of one record, derived by scanning for its label tokens.

    Attributes:
        num_loc: Number of customers ``N``.
        num_tokens: Total number of comma-separated tokens in the record.
        depot_idx: Index of the ``depot`` label token.
        customer_idx: Index of the ``customer`` label token.
        capacity_idx: Index of the ``capacity`` label token.
        demand_idx: Index of the ``demand`` label token.
        cost_idx: Index of the ``cost`` label token, or ``None``.
        node_flag_idx: Index of the ``node_flag`` label token, or ``None`` when the
            record carries no solution.
    """

    num_loc: int
    num_tokens: int
    depot_idx: int
    customer_idx: int
    capacity_idx: int
    demand_idx: int
    cost_idx: int | None
    node_flag_idx: int | None

    @property
    def has_tour(self) -> bool:
        """Whether the record carries a reference tour (i.e. a ``node_flag`` section)."""
        return self.node_flag_idx is not None

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> "InstanceLayout":
        """Derive the layout of a record from its tokens.

        Args:
            tokens: The comma-split tokens of a single record.

        Returns:
            The :class:`InstanceLayout` describing where each section lives.

        Raises:
            ValueError: If a required label is missing from the record.
        """
        positions: dict[str, int] = {}
        for idx, token in enumerate(tokens):
            name = clean_token(token)
            if name in LABELS and name not in positions:
                positions[name] = idx

        for required in ("depot", "customer", "capacity", "demand"):
            if required not in positions:
                raise ValueError(f"Missing '{required}' section in record")

        customer_idx = positions["customer"]
        # The customer coordinates run up to whichever label follows them. Bounding the
        # slice this way keeps the parser correct for both label orderings.
        next_after_customer = min(p for p in positions.values() if p > customer_idx)
        num_loc = (next_after_customer - customer_idx - 1) // 2

        return cls(
            num_loc=num_loc,
            num_tokens=len(tokens),
            depot_idx=positions["depot"],
            customer_idx=customer_idx,
            capacity_idx=positions["capacity"],
            demand_idx=positions["demand"],
            cost_idx=positions.get("cost"),
            node_flag_idx=positions.get("node_flag"),
        )

    def demand_end(self, tokens: list[str]) -> int:
        """Return the exclusive end index of the demand section.

        The demand section is bounded by whichever label follows it, which may be
        ``cost`` (standard files) or ``capacity`` (the ``vrplib`` literal dump).

        Args:
            tokens: The comma-split tokens of the same record.

        Returns:
            Index one past the last demand value.
        """
        for idx, token in enumerate(tokens):
            if idx > self.demand_idx and clean_token(token) in LABELS:
                return idx
        return len(tokens)

    def parse(self, line: str) -> dict[str, np.ndarray]:
        """Parse one record into flat numpy arrays.

        Args:
            line: A single raw line from a CVRP ``.txt`` file.

        Returns:
            A dict with keys ``depot`` ``[2]``, ``locs`` ``[N, 2]``, ``demand`` ``[N]``,
            ``capacity`` ``[]``, ``cost`` ``[]`` and -- when the record carries a
            solution -- ``tour`` ``[N]`` and ``flags`` ``[N]``. All values are float32
            except ``tour`` and ``flags``, which are int64.
        """
        tokens = line.split(",")
        n = self.num_loc

        depot = np.array([float(t) for t in tokens[self.depot_idx + 1 : self.depot_idx + 3]])
        locs = np.array(
            [float(t) for t in tokens[self.customer_idx + 1 : self.customer_idx + 1 + 2 * n]]
        )

        # The HGS training files prepend a depot demand slot; taking the trailing N
        # values drops it without branching on the record's provenance.
        demand_full = np.array(
            [float(t) for t in tokens[self.demand_idx + 1 : self.demand_end(tokens)]]
        )

        cost = (
            np.float32(float(tokens[self.cost_idx + 1])) if self.cost_idx else np.float32(np.nan)
        )
        out = {
            "depot": depot.astype(np.float32),
            "locs": locs.reshape(n, 2).astype(np.float32),
            "demand": demand_full[-n:].astype(np.float32),
            "capacity": np.float32(float(tokens[self.capacity_idx + 1])),
            "cost": cost,
        }

        if self.has_tour:
            start = self.node_flag_idx + 1
            out["tour"] = np.array([int(t) for t in tokens[start : start + n]], dtype=np.int64)
            out["flags"] = np.array(
                [int(t) for t in tokens[start + n : start + 2 * n]], dtype=np.int64
            )

        return out


def scan_offsets(path: str) -> np.ndarray:
    """Index the byte offset of every line in a large text file.

    The file is memory-mapped and scanned in chunks, so peak memory stays bounded
    regardless of file size (the 4.6 GB training file is indexed in roughly 3 s).

    Args:
        path: Path to the text file.

    Returns:
        An int64 array of length ``num_lines + 1``. Line ``i`` occupies
        ``[offsets[i], offsets[i + 1])``, so the trailing sentinel is the file size.
    """
    size = os.path.getsize(path)
    starts: list[np.ndarray] = []
    with open(path, "rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for begin in range(0, size, _SCAN_CHUNK):
                end = min(begin + _SCAN_CHUNK, size)
                chunk = np.frombuffer(mm[begin:end], dtype=np.uint8)
                newlines = np.flatnonzero(chunk == 10)
                if newlines.size:
                    starts.append(newlines.astype(np.int64) + begin + 1)

    line_starts = np.concatenate([np.zeros(1, dtype=np.int64), *starts])
    line_starts = line_starts[line_starts < size]
    return np.append(line_starts, size)


def read_layout(path: str) -> InstanceLayout:
    """Read the first record of ``path`` and return its layout.

    Args:
        path: Path to a CVRP ``.txt`` file.

    Returns:
        The :class:`InstanceLayout` shared by every record in the file.
    """
    with open(path) as handle:
        return InstanceLayout.from_tokens(handle.readline().split(","))
