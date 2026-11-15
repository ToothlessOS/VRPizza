"""CVRP environments for local-reconstruction SIL.

The SIL-RC pipeline has two data sources:

* a **warm-up** stage on generated CVRP20 instances (the paper's POMO warm-up);
* a **SIL** stage on pseudo-labels built from generated CVRP100 instances, with the
  same held-out HGS100 tail for validation as every other arm.

The environment therefore needs one generator for the warm-up and another for the
label instances, and ``reset_partial`` for the teacher-forced partial windows.
The label store itself lives in the Lightning module, not here: it is large
(1M instances for the strict run) and has its own persistence, so keeping a copy
on the env would create a second source of truth.

:class:`SILRCCVRPEnv` is the strict arm's environment: the warm-up batch is POMO's
normal ``N`` starts. :class:`SILRCStepCVRPEnv` is the step-matched arm's: the
warm-up batch carries 16 start nodes per instance, so one update is 125 instances
x 16 starts = 2,000 rollouts. Both send validation/test to the file-backed
parent, which is what keeps the validation subset identical to the other arms.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from rl4co.data.dataset import TensorDictDataset
from rl4co.envs.routing.cvrp.generator import CVRPGenerator

from src.envs.cvrp import _resolve_size
from src.envs.cvrp_rl import RLCVRPEnv
from src.envs.cvrp_rl_step import GeneratedStepRLCVRPEnv
from src.envs.cvrp_sl_lehd import PartialResetMixin

__all__ = ["SILRCCVRPEnv", "SILRCStepCVRPEnv"]


class SILRCLabelMixin:
    """Shared label-instance generation for both SIL-RC environments.

    Args:
        warmup_num_loc: Customers per warm-up instance.
        warmup_capacity: Warm-up vehicle capacity, in the generator's raw units.
        **kwargs: Forwarded to the environment base.
    """

    def __init__(
        self,
        *args,
        warmup_num_loc: int = 20,
        warmup_capacity: float = 30.0,
        **kwargs,
    ) -> None:
        self.warmup_num_loc = int(warmup_num_loc)
        self.warmup_capacity = float(warmup_capacity)
        # The environment's own generator is built by `CVRPEnv.__init__` with the
        # file's size/capacity and is used for the SIL labels. The warm-up needs a
        # different problem size, so it gets its own generator.
        self.warmup_generator = CVRPGenerator(
            num_loc=self.warmup_num_loc, capacity=self.warmup_capacity
        )
        super().__init__(*args, **kwargs)

    def sample_instances(self, count: int, seed: int):
        """Generate ``count`` full-size instances for the label store."""
        from src.sil_rc.labels import generate_instances

        return generate_instances(self.generator, int(count), int(seed))

    def warmup_dataset(self, size: int, seed: int) -> TensorDictDataset:
        """One epoch of generated warm-up instances, seeded by ``seed``."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            td = self.warmup_generator(int(size))
        return TensorDictDataset(td)


class SILRCCVRPEnv(SILRCLabelMixin, PartialResetMixin, RLCVRPEnv):
    """Warm-up on generated CVRP20 with POMO's N starts; SIL via installed labels.

    Args:
        data_path: Tagged solution file used only for the held-out validation/test
            tail.
        val_data_path: Optional separate validation file.
        num_loc: Full SIL instance size, inferred from the file when omitted.
        capacity: Full-instance capacity, inferred from the file when omitted.
        val_instances: Held-out instances carved from the tail of ``data_path``.
        seed: Base seed for the generated warm-up and the validation split.
        warmup_num_loc: Warm-up customers, 20 by default.
        warmup_capacity: Warm-up capacity, 30 by default (rl4co's CVRP20 default).
        **kwargs: Forwarded to :class:`~src.envs.RLCVRPEnv`.
    """

    def dataset(
        self, batch_size: int | Sequence = (), phase: str = "train", filename: str | None = None
    ):
        """Generated CVRP20 for training; the file-backed split otherwise."""
        if phase == "train" and filename is None:
            self._epoch += 1
            size = _resolve_size(batch_size)
            if size is None:
                raise ValueError("warm-up training requires an explicit train_data_size")
            return self.warmup_dataset(int(size), self.seed + self._epoch)
        return super().dataset(batch_size, phase=phase, filename=filename)


class SILRCStepCVRPEnv(SILRCLabelMixin, PartialResetMixin, GeneratedStepRLCVRPEnv):
    """Step-matched variant: 125 CVRP20 warm-up instances x 16 starts per update.

    The parent's ``_generated_dataset`` is overridden only to draw from the
    warm-up generator instead of the full-size one; the start-node draw and the
    file-backed validation path are the parent's.

    Args:
        **kwargs: Same as :class:`SILRCCVRPEnv`, plus the parent's
            ``starts_per_instance``.
    """

    def warmup_dataset(self, size: int, seed: int) -> TensorDictDataset:
        """Generated CVRP20 instances with 16 distinct start nodes each."""
        epoch_seed = int(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(epoch_seed)
            td = self.warmup_generator(int(size))

        num_loc = int(td["locs"].shape[-2])
        if self.starts_per_instance > num_loc:
            raise ValueError(
                f"starts_per_instance={self.starts_per_instance} exceeds "
                f"warmup num_loc={num_loc}"
            )
        rng = np.random.default_rng([epoch_seed, self.starts_per_instance])
        starts = np.stack(
            [
                rng.choice(num_loc, size=self.starts_per_instance, replace=False) + 1
                for _ in range(int(size))
            ]
        )
        td = td.clone()
        td.set("start_nodes", torch.as_tensor(starts, dtype=torch.long))
        return TensorDictDataset(td)

    def _generated_dataset(self, size: int) -> TensorDictDataset:
        """The parent's per-epoch hook, routed through the warm-up generator."""
        return self.warmup_dataset(size, self.seed + self._epoch)
