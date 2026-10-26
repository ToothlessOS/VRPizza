"""CVRP environments that start an episode from a LEHD partial solution.

Two pieces live here, shared by training and RRC:

* :class:`PartialResetMixin` builds the post-start environment state directly from a reduced
  problem: the forced start is already served, the depot is marked visited so ``done`` fires
  on the last window customer, and ``used_capacity`` carries the demand served earlier in the
  start's route. A plain ``env.reset`` cannot express that because it starts at the depot with
  an empty load, and stepping the start through the environment would reset the capacity at
  the wrong moment.
* :class:`LEHDCVRPEnv` serves the partial dataset for training while leaving validation and
  testing on the parent's full-trajectory dataset, so the POMO validation metrics remain
  comparable with the existing ``sl-`` and ``rl-`` runs. :class:`LEHDEvalCVRPEnv` is the
  dataset-free variant used by RRC.

The state is built by resetting the reduced batch and then overwriting five fields. That is
deliberate: it reuses rl4co's specification checks, the depot-prepended ``locs`` and the
demand normalisation, and the five overwrites are exactly the difference between "start at the
depot" and "start mid-route". The action mask is recomputed afterwards, so it sees the partial
load and the already-served start.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from tensordict.tensordict import TensorDict

from src.envs.cvrp import CVRPEnv, _resolve_size
from src.envs.cvrp_eval import EvalCVRPEnv
from src.envs.dataset import CVRPSolutionDataset
from src.envs.lehd_dataset import LEHDPartialDataset

__all__ = [
    "LEHDCVRPEnv",
    "LEHDEvalCVRPEnv",
    "PartialResetMixin",
]


class PartialResetMixin:
    """Build an environment state whose forced start is already served.

    The host class must provide ``reset``, ``make_input_td`` and ``get_action_mask``, which
    every CVRP environment in this repo does.

    Args:
        batch: A reduced batch carrying ``depot``, ``locs``, ``demand``, ``capacity``,
            ``start_node`` and ``used_capacity0``. ``used_capacity0`` is in raw demand units,
            as emitted by :class:`~src.envs.lehd_dataset.LEHDPartialDataset`.

    Returns:
        A ``TensorDict`` whose ``current_node`` is the forced start, whose ``visited`` marks
        the depot and the start, and whose ``used_capacity`` is normalised by the instance
        capacity. ``done`` is false; it becomes true when every window customer is visited.
    """

    def reset_partial(self, batch: TensorDict) -> TensorDict:
        start = batch["start_node"].to(torch.long)
        used = batch["used_capacity0"].to(torch.float32)
        capacity = batch["capacity"].to(torch.float32)
        size = int(start.shape[0])

        td = self.reset(self.make_input_td(batch))
        visited = td["visited"]
        visited[:, 0] = 1  # depot pre-visited: done fires on the last window customer
        visited[torch.arange(size, device=visited.device), start] = 1
        td["current_node"] = start[:, None]
        # Normalise here rather than at the call site, matching make_input_td's demand / capacity.
        td["used_capacity"] = (used / capacity)[:, None]
        td["done"] = torch.zeros((size, 1), dtype=torch.bool, device=td.device)
        td["action_mask"] = self.get_action_mask(td)
        return td


class LEHDCVRPEnv(PartialResetMixin, CVRPEnv):
    """A CVRP environment serving LEHD partial solutions for training.

    Args:
        data_path: Path to the reference-solution file.
        val_data_path: Optional separate validation file, forwarded to
            :class:`~src.envs.CVRPEnv`.
        num_loc: Number of customers, inferred from the file when omitted.
        capacity: Vehicle capacity, inferred from the file when omitted.
        val_instances: Held-out instances carved from the tail.
        seed: Base seed; training re-draws with ``seed + epoch``.
        check_solution: Forwarded to :class:`~src.envs.CVRPEnv`.
        effective_batch_size: Optimizer batch size used to key the one-``w``-per-batch draw.
            With gradient accumulation pass ``microbatch * accumulate`` here and the
            microbatch as the Lightning module's ``batch_size``.
        **kwargs: Forwarded to :class:`~src.envs.CVRPEnv`.
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
        effective_batch_size: int = 1024,
        **kwargs,
    ) -> None:
        self.effective_batch_size = int(effective_batch_size)
        if self.effective_batch_size <= 0:
            raise ValueError(
                f"effective_batch_size must be positive, got {self.effective_batch_size}"
            )
        super().__init__(
            data_path=data_path,
            val_data_path=val_data_path,
            num_loc=num_loc,
            capacity=capacity,
            val_instances=val_instances,
            seed=seed,
            check_solution=check_solution,
            **kwargs,
        )

    def dataset(
        self, batch_size: int | Sequence = (), phase: str = "train", filename: str | None = None
    ) -> LEHDPartialDataset | CVRPSolutionDataset:
        """Build the partial dataset for training, the parent's for validation/testing.

        Args:
            batch_size: Number of partial solutions to expose for training; rl4co passes the
                phase's data size here. Validation and testing interpret it as a trajectory
                count, exactly as :class:`~src.envs.CVRPEnv` does.
            phase: One of ``"train"``, ``"val"`` or ``"test"``.
            filename: Optional path overriding the configured file. Naming a file falls
                through to the parent for every phase.

        Returns:
            :class:`~src.envs.lehd_dataset.LEHDPartialDataset` for training, and
            :class:`~src.envs.dataset.CVRPSolutionDataset` otherwise.
        """
        if phase == "train" and filename is None:
            self._epoch += 1
            return LEHDPartialDataset(
                self.data_path,
                instance_range=self.train_range,
                partials=_resolve_size(batch_size),
                batch_size=self.effective_batch_size,
                seed=self.seed + self._epoch,
            )
        return super().dataset(batch_size, phase=phase, filename=filename)


class LEHDEvalCVRPEnv(PartialResetMixin, EvalCVRPEnv):
    """Dataset-free CVRP environment for RRC reconstructions.

    Args:
        num_loc: Number of customers in the **full** instance. The reduced partial batch has
            its own width ``w``; this value is only used by multi-start helpers, which RRC
            does not call on a partial.
        capacity: Vehicle capacity in the file's demand units, forwarded to the generator.
        check_solution: Forwarded to :class:`~src.envs.EvalCVRPEnv`.
        **kwargs: Forwarded to :class:`~src.envs.EvalCVRPEnv`.
    """

    def __init__(
        self,
        num_loc: int,
        capacity: float = 1.0,
        check_solution: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            num_loc=num_loc, capacity=capacity, check_solution=check_solution, **kwargs
        )
