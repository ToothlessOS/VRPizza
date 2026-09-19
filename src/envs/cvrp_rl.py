"""A CVRP environment that trains on generated instances and validates on the SL split.

:class:`~src.envs.CVRPEnv` draws every phase -- train, val and test -- from the reference-solution
files, which is what supervised learning wants and what reinforcement learning does not: POMO
trains on instances sampled fresh from the generator each epoch, and only needs the files in order
to be scored on the same held-out instances the supervised model is scored on.

That is the whole difference this class introduces. It overrides the single integration point
rl4co uses, :meth:`dataset`, and sends the training phase to the generator while leaving val and
test to the parent, so the validation subset matches :class:`~src.models.POMOSL`'s by
construction -- same class, same held-out range, same size, same seed -- rather than by a pair of
constants that could drift apart.

The reference file is still required: ``CVRPEnv.__init__`` carves the held-out range out of it and
refuses a file without a reference tour. Training does not read it.
"""

from __future__ import annotations

from collections.abc import Sequence

from rl4co.data.dataset import TensorDictDataset
from rl4co.envs.routing.cvrp.env import CVRPEnv as _RL4COCVRPEnv

from src.envs.cvrp import CVRPEnv
from src.envs.dataset import CVRPSolutionDataset

__all__ = ["RLCVRPEnv"]


class RLCVRPEnv(CVRPEnv):
    """A CVRP environment whose training set is generated and whose validation set is not.

    Args:
        data_path: Path to the reference-solution file the validation split is carved from.
            Normally the same file :class:`~src.models.POMOSL` trains on.
        val_data_path: Optional separate file for validation and testing, forwarded to
            :class:`~src.envs.CVRPEnv`.
        num_loc: Number of customers. Inferred from the file when omitted.
        capacity: Vehicle capacity in the file's own demand units. Inferred from the file when
            omitted.
        val_instances: Number of held-out instances carved from the tail of ``data_path`` when
            ``val_data_path`` is not given. Must match the supervised run for the two validation
            sets to be the same subset.
        seed: Base seed for subset draws. The validation subset is fixed; training draws fresh
            instances from the generator every epoch.
        **kwargs: Forwarded to :class:`~src.envs.CVRPEnv`.
    """

    def dataset(
        self, batch_size: int | Sequence = (), phase: str = "train", filename: str | None = None
    ) -> CVRPSolutionDataset | TensorDictDataset:
        """Build the dataset for a phase.

        Args:
            batch_size: Number of instances (train) or trajectories (val/test) to expose. rl4co
                passes the phase's data size here.
            phase: One of ``"train"``, ``"val"`` or ``"test"``.
            filename: Optional path overriding the configured file, used verbatim. A caller who
                names a file gets that file for every phase, training included.

        Returns:
            A generator-backed :class:`~rl4co.data.dataset.TensorDictDataset` for training, and a
            :class:`~src.envs.dataset.CVRPSolutionDataset` for validation and testing.
        """
        if phase == "train" and filename is None:
            # `train_file` is unset, so the base implementation samples `batch_size` instances
            # from `self.generator` and wraps them. `RL4COLitModule.on_train_epoch_end` calls
            # this again at the end of every epoch, so training keeps seeing fresh instances
            # while the validation set below stays fixed.
            return _RL4COCVRPEnv.dataset(self, batch_size, phase="train")
        return super().dataset(batch_size, phase=phase, filename=filename)
