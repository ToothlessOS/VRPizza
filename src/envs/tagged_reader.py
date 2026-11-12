"""Fork-safe access to the records of a tagged CVRP solution file.

This is the reader shared by every file-backed dataset: the canonical LEHD partial dataset,
the supervised 16-start dataset, and the step-matched LEHD dataset. It was originally a private
class in :mod:`src.envs.synced_dataset`; it lives here so the data-synced modules can be
retired without taking the LEHD reader with them.

Subclasses must set ``path``, ``offsets`` and ``layout`` before reading. The per-worker file
descriptor is the same mechanism as :class:`~src.envs.dataset.CVRPSolutionDataset`: forked
DataLoader workers inherit a parent's descriptor, so a single shared handle would interleave
reads between workers and return corrupt lines. Each worker opens its own descriptor, and reads
go through :func:`os.pread`, which leaves the shared file offset untouched.
"""

from __future__ import annotations

import os

import numpy as np

from torch.utils.data import get_worker_info

__all__ = ["TaggedFileReader"]


class TaggedFileReader:
    """Lazy, fork-safe reads from a tagged CVRP solution file.

    Subclasses set ``self.path``, ``self.offsets`` (with a trailing file-size sentinel) and
    ``self.layout`` (an :class:`~src.envs.parser.InstanceLayout`) before calling
    :meth:`_read_record`.
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
