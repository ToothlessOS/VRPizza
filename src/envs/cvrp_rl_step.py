"""REINFORCE CVRP for the step-matched comparison: generated instances, 16 starts.

The canonical RL arm (:mod:`src.envs.cvrp_rl`) trains on generated instances with POMO's full
``N`` starts. The step-matched arm keeps the generated (label-free) data source but reduces the
multi-start to 16, so one optimizer step is 125 instances x 16 starts = 2,000 rollouts -- the
same decoded-sequence budget as the supervised arms. Validation and testing deliberately fall
through to the parent's file-backed dataset, where ``make_input_td`` emits no ``start_nodes``
and the env keeps POMO's N=100 starts.

Instances *and* start nodes are drawn from a dedicated ``seed + epoch`` RNG. Seeding the global
torch RNG once is not enough: ``setup()`` builds the first dataset before Lightning restores
``current_epoch``, so a resumed run would regenerate epoch 0's data. The local
:func:`torch.random.fork_rng` block does not perturb the global RNG used elsewhere.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from rl4co.data.dataset import TensorDictDataset

from src.envs.cvrp import CVRPEnv, _resolve_size
from src.envs.start_nodes import StartNodeMixin

__all__ = ["GeneratedStepRLCVRPEnv"]


class GeneratedStepRLCVRPEnv(StartNodeMixin, CVRPEnv):
    """A CVRP environment trained on generated instances with a per-batch 16-start multi-start.

    Args:
        data_path: Reference-solution file used only to carve the validation/test tail and to
            supply the reference cost.
        val_data_path: Optional separate validation file, forwarded to
            :class:`~src.envs.CVRPEnv`.
        num_loc: Number of customers, inferred from the file when omitted.
        capacity: Vehicle capacity, inferred from the file when omitted.
        val_instances: Held-out instances carved from the tail of ``data_path``.
        seed: Base seed for the generated data and start nodes.
        check_solution: Forwarded to :class:`~src.envs.CVRPEnv`.
        starts_per_instance: Starts per generated instance. Must be at least 2 for POMO's
            shared baseline; duplicates are sampled without replacement.
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
        starts_per_instance: int = 16,
        **kwargs,
    ) -> None:
        self.starts_per_instance = int(starts_per_instance)
        if self.starts_per_instance < 2:
            raise ValueError(
                "starts_per_instance must be >= 2; POMO's shared baseline is identically "
                "zero with one start"
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
    ) -> TensorDictDataset:
        """Build the generated training set, or defer val/test to the file-backed parent.

        Args:
            batch_size: Number of generated instances (train) or trajectories (val/test).
                rl4co passes the phase's data size here.
            phase: One of ``"train"``, ``"val"`` or ``"test"``.
            filename: Optional path overriding the configured file. Naming a file sends every
                phase, training included, to the parent's file-backed dataset.

        Returns:
            A :class:`~rl4co.data.dataset.TensorDictDataset` carrying the generated instance
            fields plus ``start_nodes`` ``[B, starts_per_instance]`` for training, and the
            parent's dataset otherwise.
        """
        if phase == "train" and filename is None:
            self._epoch += 1
            size = _resolve_size(batch_size)
            if size is None:
                raise ValueError("generated training requires an explicit train_data_size")
            return self._generated_dataset(int(size))
        return super().dataset(batch_size, phase=phase, filename=filename)

    def _generated_dataset(self, size: int) -> TensorDictDataset:
        """Generate one epoch of instances and attach distinct start nodes."""
        epoch_seed = self.seed + self._epoch
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(epoch_seed)
            td = self.generator(size)

        num_loc = int(td["locs"].shape[-2])
        if self.starts_per_instance > num_loc:
            raise ValueError(
                f"starts_per_instance={self.starts_per_instance} exceeds num_loc={num_loc}"
            )
        # Per-instance draws, seeded by ``[seed, epoch]``; distinct customer ids in 1..N.
        rng = np.random.default_rng([epoch_seed, self.starts_per_instance])
        starts = np.stack(
            [
                rng.choice(num_loc, size=self.starts_per_instance, replace=False) + 1
                for _ in range(size)
            ]
        )
        td = td.clone()
        td.set("start_nodes", torch.as_tensor(starts, dtype=torch.long))
        return TensorDictDataset(td)
