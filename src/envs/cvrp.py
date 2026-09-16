"""A CVRP environment backed by a dataset of reference solutions.

This wraps rl4co's :class:`~rl4co.envs.CVRPEnv` and replaces its data source: instead of
sampling random instances from a generator, it draws ``(instance, optimal start)``
trajectories from the tagged solution files in ``data/CVRP``.

The whole integration point is :meth:`CVRPEnv.dataset`. ``RL4COLitModule.setup`` calls
``env.dataset(size, phase=...)`` and wraps whatever comes back in a DataLoader, so
overriding that one method is enough to drive supervised learning from these files
without touching the Lightning module.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from rl4co.envs.routing.cvrp.env import CVRPEnv as _RL4COCVRPEnv
from tensordict.tensordict import TensorDict

from src.envs.dataset import CVRPSolutionDataset, build_index
from src.envs.parser import resolve_data_path

__all__ = ["CVRPEnv"]


def _resolve_size(batch_size: int | Sequence | None) -> int | None:
    """Normalise the size argument rl4co passes to :meth:`CVRPEnv.dataset`.

    ``RL4COLitModule`` passes ``train_data_size`` positionally as ``batch_size``, while
    the environment base class defaults it to an empty list.

    Args:
        batch_size: An int, a torch.Size, an empty sequence, or ``None``.

    Returns:
        The size as an int, or ``None`` meaning "use the whole pool".
    """
    if batch_size is None:
        return None
    if isinstance(batch_size, int):
        return batch_size
    if isinstance(batch_size, torch.Size):
        batch_size = tuple(batch_size)
    if len(batch_size) == 0:
        return None
    return int(batch_size[0])


class CVRPEnv(_RL4COCVRPEnv):
    """A CVRP environment whose instances come from a reference-solution dataset.

    Each instance in the file yields ``2R`` supervised trajectories rather than one,
    where ``R`` is its route count; see
    :meth:`~src.envs.dataset.CVRPSolutionDataset.build_trajectory` for why exactly
    ``2R`` starts are the ones that reproduce the reference cost.

    Args:
        data_path: Path to the training file, e.g.
            ``data/CVRP/vrp100_hgs_train_100w-001.txt``.
        val_data_path: Optional separate file for validation and testing. When omitted,
            the tail of ``data_path`` is held out instead, which is what the ``N=50``
            data needs since it has no companion test file.
        num_loc: Number of customers. Inferred from the file when omitted.
        capacity: Vehicle capacity in the file's own demand units. Inferred from the
            file when omitted. Demands are normalised by this and the environment's
            ``vehicle_capacity`` is left at 1.0, which is what rl4co's action mask
            assumes.
        val_instances: Number of held-out instances carved from the tail of
            ``data_path`` when ``val_data_path`` is not given.
        seed: Base seed for subset draws. The training subset is re-drawn each epoch
            from ``seed + epoch``; validation keeps a fixed subset.
        check_solution: Whether :meth:`get_reward` validates every solution. Defaults to
            ``False`` because the check is a Python loop over the decode steps; the test
            suite enables it explicitly.
        **kwargs: Forwarded to :class:`~rl4co.envs.CVRPEnv`.
    """

    def __init__(
        self,
        data_path: str,
        val_data_path: str | None = None,
        num_loc: int | None = None,
        capacity: float | None = None,
        val_instances: int = 10_000,
        seed: int = 0,
        check_solution: bool = False,
        **kwargs,
    ) -> None:
        self.data_path = resolve_data_path(data_path)
        self.val_data_path = (
            resolve_data_path(val_data_path) if val_data_path is not None else None
        )
        self.seed = int(seed)
        self._epoch = 0

        _, num_routes, layout = build_index(self.data_path)
        if not layout.has_tour:
            raise ValueError(
                f"{self.data_path} carries no reference tour; it can be used for evaluation "
                "but not for supervised training"
            )

        self.num_loc = int(num_loc) if num_loc is not None else layout.num_loc
        if self.num_loc != layout.num_loc:
            raise ValueError(
                f"num_loc={self.num_loc} does not match the {layout.num_loc} customers "
                f"found in {self.data_path}"
            )

        self._layout = layout
        self.capacity = float(capacity) if capacity is not None else self._file_capacity()

        instances = int(num_routes.size)
        held_out = min(int(val_instances), instances // 10) if self.val_data_path is None else 0
        self.train_range = (0, instances - held_out)
        self.val_range = (instances - held_out, instances)

        super().__init__(
            generator_params={"num_loc": self.num_loc, "capacity": self.capacity},
            check_solution=check_solution,
            **kwargs,
        )

    # -- data ---------------------------------------------------------------------

    def _file_capacity(self) -> float:
        """Read the vehicle capacity from the first record of the training file.

        Capacities are per-file and do not follow rl4co's ``CAPACITIES`` table, so the
        value has to come from the data rather than from ``num_loc``.
        """
        with open(self.data_path) as handle:
            return float(handle.readline().split(",")[self._layout.capacity_idx + 1])

    def dataset(
        self, batch_size: int | Sequence = (), phase: str = "train", filename: str | None = None
    ) -> CVRPSolutionDataset:
        """Build the dataset for a phase.

        This is the single integration point with :class:`~rl4co.models.RL4COLitModule`,
        which calls it as ``env.dataset(size, phase=phase)`` from ``setup`` and again from
        ``on_train_epoch_end`` to refresh the training set each epoch.

        Args:
            batch_size: Number of trajectories to expose. ``None`` or an empty sequence
                exposes the whole pool. rl4co passes the phase's data size here.
            phase: One of ``"train"``, ``"val"`` or ``"test"``.
            filename: Optional path overriding the configured file, used verbatim.

        Returns:
            A :class:`~src.envs.dataset.CVRPSolutionDataset`.
        """
        size = _resolve_size(batch_size)

        if filename is not None:
            return CVRPSolutionDataset(filename, size=size, seed=self.seed)

        if phase == "train":
            self._epoch += 1
            return CVRPSolutionDataset(
                self.data_path,
                instance_range=self.train_range,
                size=size,
                seed=self.seed + self._epoch,
            )

        path = self.val_data_path or self.data_path
        instance_range = None if self.val_data_path is not None else self.val_range
        return CVRPSolutionDataset(path, instance_range=instance_range, size=size, seed=self.seed)

    # -- supervised-learning helpers ----------------------------------------------

    def make_input_td(self, batch: TensorDict) -> TensorDict:
        """Build the environment input from a dataset batch.

        Returns a fresh TensorDict on every call: ``CVRPEnv.reset`` mutates its input in
        place, widening ``locs`` from ``[B, N, 2]`` to ``[B, N+1, 2]``, so a batch that
        had already been reset cannot be reset again.

        Args:
            batch: A batch produced by :meth:`~src.envs.dataset.CVRPSolutionDataset.collate_fn`.

        Returns:
            A TensorDict with ``depot`` ``[B, 2]``, ``locs`` ``[B, N, 2]`` (depot excluded,
            as rl4co expects) and ``demand`` ``[B, N]`` normalised by capacity.
        """
        return TensorDict(
            {
                "depot": batch["depot"],
                "locs": batch["locs"],
                "demand": batch["demand"] / batch["capacity"][:, None],
            },
            batch_size=batch.batch_size,
            device=batch.device,
        )

    def build_expert_actions(
        self, batch: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the reference trajectories carried by a batch.

        The dataset builds each trajectory when it is read, so this is a thin accessor
        that keeps the supervised-learning vocabulary on the environment.

        Args:
            batch: A batch produced by this environment's dataset.

        Returns:
            ``(actions, num_routes, valid)`` with shapes ``[B, N+R_max]``, ``[B]`` and
            ``[B, N+R_max]``. ``valid`` marks the real steps; the decoder consumes only
            ``N + R - 1`` of them, since the final depot return is implicit.
        """
        return batch["actions"], batch["num_routes"], batch["valid"]

    @staticmethod
    def reference_cost(batch: TensorDict) -> torch.Tensor:
        """Return the reference solution cost carried by a batch.

        Args:
            batch: A batch produced by a dataset over a file with a ``cost`` column.

        Returns:
            A ``[B]`` tensor of reference objectives, positive. Note the HGS training
            files carry a reference that differs from the stored coordinates' true
            Euclidean length by up to ~6e-5 relative, while the LKH test files match to
            machine precision.
        """
        return batch["cost"].to(torch.float32)

    def num_trajectories(self, phase: str = "train") -> int:
        """Return the number of supervised trajectories available for a phase.

        Args:
            phase: One of ``"train"``, ``"val"`` or ``"test"``.

        Returns:
            The pool size, i.e. ``sum(2 * R)`` over the phase's instance range.
        """
        path = self.data_path if phase == "train" else (self.val_data_path or self.data_path)
        _, num_routes, _ = build_index(path)
        start, stop = self.train_range if phase == "train" else self.val_range
        if self.val_data_path is not None:
            start, stop = 0, int(num_routes.size)
        return int(2 * np.sum(num_routes[start:stop]))
