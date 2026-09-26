"""Reinforcement-learning CVRP whose training data is the supervised pipeline's, bounded by its labels.

:class:`~src.envs.RLCVRPEnv` trains on instances sampled fresh from rl4co's generator and
uses the reference file only for validation, which is POMO's original setup. That leaves two
variables uncontrolled against :class:`~src.models.POMOSL`: the instances come from a
different distribution than the HGS file, and the multi-start begins from all ``N``
customers rather than from the starts the reference solution makes legal.

This environment removes both. Training instances come from the same file and the same
``train_range`` the supervised run uses, drawn through the shared, instance-grouped sampler
in :mod:`src/envs.synced_dataset`, and each instance's multi-start is restricted to ``k`` of
its ``2R`` label-feasible starts. Validation is deliberately left alone -- it is shared code
(:class:`~src.models.components.pomo_inference.POMOInferenceMixin`), so ``val/reward``,
``val/reward_greedy`` and ``val/gap_ref`` keep meaning what they mean in the other runs.

The mechanics are two overrides that change together:

* :meth:`get_num_starts` reports ``k`` when the batch carries start nodes. The RL batch is
  the only one that does; the validation batch is built by
  :meth:`~src.envs.InstanceInputMixin.make_input_td`, which carries no such key, so
  validation keeps POMO's ``N`` starts and stays comparable.
* :meth:`select_start_nodes` returns the batch's own start nodes instead of rl4co's fixed
  ``arange(k) % num_loc + 1``.

Both read the *post-reset* state, and rl4co's ``CVRPEnv._reset`` rebuilds a fresh
``TensorDict`` from an explicit key list, so :meth:`_reset` re-attaches ``start_nodes``
before either can see it.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.envs.cvrp import CVRPEnv, _resolve_size
from src.envs.dataset import CVRPSolutionDataset
from src.envs.synced_dataset import CVRPSyncedDataset

__all__ = ["SyncedRLCVRPEnv"]


class SyncedRLCVRPEnv(CVRPEnv):
    """A CVRP environment trained on the supervised file, with label-bounded multi-start.

    Subclasses :class:`~src.envs.CVRPEnv` rather than :class:`~src.envs.RLCVRPEnv`: the
    latter's only content is the branch that routes training to the generator, which this
    class replaces outright.

    Args:
        data_path: Path to the reference-solution file both pipelines draw from.
        val_data_path: Optional separate file for validation and testing, forwarded to
            :class:`~src.envs.CVRPEnv`.
        num_loc: Number of customers. Inferred from the file when omitted.
        capacity: Vehicle capacity in the file's own demand units. Inferred when omitted.
        val_instances: Held-out instances carved from the tail of ``data_path``. Must match
            the supervised run for the two validation sets to remain the same subset.
        seed: Base seed for the draw. Training re-draws with ``seed + epoch``.
        check_solution: Forwarded to :class:`~src.envs.CVRPEnv`.
        starts_per_instance: How many of each instance's ``2R`` feasible starts the
            multi-start uses. Must not exceed the smallest ``2R`` in the training range.
            Named to avoid confusion with ``POMO.num_starts``, which is a different thing:
            that one is the model's, this one is the environment's.
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
        # Set before `super().__init__`, which is what sets the counters this draws on.
        self.starts_per_instance = int(starts_per_instance)
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

    # -- data ---------------------------------------------------------------------

    def dataset(
        self, batch_size: int | Sequence = (), phase: str = "train", filename: str | None = None
    ) -> CVRPSyncedDataset | CVRPSolutionDataset:
        """Build the dataset for a phase.

        Args:
            batch_size: Number of *instances* to expose for training; rl4co passes the
                phase's data size here. Validation and testing interpret it as a trajectory
                count, exactly as :class:`~src.envs.CVRPEnv` does.
            phase: One of ``"train"``, ``"val"`` or ``"test"``.
            filename: Optional path overriding the configured file, used verbatim. Naming a
                file sends every phase, training included, to the supervised dataset.

        Returns:
            A :class:`~src.envs.synced_dataset.CVRPSyncedDataset` for training, and a
            :class:`~src.envs.dataset.CVRPSolutionDataset` for validation and testing.
        """
        if phase == "train" and filename is None:
            self._epoch += 1
            return CVRPSyncedDataset(
                self.data_path,
                instance_range=self.train_range,
                instances=_resolve_size(batch_size),
                k=self.starts_per_instance,
                seed=self.seed + self._epoch,
            )
        return super().dataset(batch_size, phase=phase, filename=filename)

    def num_instances(self, phase: str = "train") -> int:
        """Return how many instances a phase's draw selects.

        :meth:`~src.envs.CVRPEnv.num_trajectories` reports the ``2R`` trajectory pool, which
        is the wrong unit for this environment's training phase.

        Args:
            phase: One of ``"train"``, ``"val"`` or ``"test"``.

        Returns:
            The number of instances in the phase's range.
        """
        start, stop = self.train_range if phase == "train" else self.val_range
        return int(stop - start)

    # -- multi-start --------------------------------------------------------------

    def _reset(self, td=None, batch_size=None):
        """Reset the environment, carrying ``start_nodes`` across.

        rl4co's ``CVRPEnv._reset`` builds a fresh ``TensorDict`` listing its own keys, so
        anything else on the input -- including the start nodes -- is dropped. Because
        :meth:`get_num_starts` and :meth:`select_start_nodes` only ever see the post-reset
        state, the key has to be re-attached here.
        """
        td_reset = super()._reset(td, batch_size)
        if td is not None and "start_nodes" in td.keys():
            td_reset.set("start_nodes", td["start_nodes"])
        return td_reset

    def get_num_starts(self, td) -> int:
        """Return the batch's start count, or POMO's ``N`` when the batch carries none."""
        if "start_nodes" in td.keys():
            return int(td["start_nodes"].size(-1))
        return super().get_num_starts(td)

    def select_start_nodes(self, td, num_starts: int):
        """Return the batch's own start nodes as a flat ``[num_starts * B]`` action tensor.

        The layout must be *start-major* -- flat index ``k * B + b``. ``batchify`` expands a
        batch of ``B`` by ``repeats`` as ``x.expand(k, B, ...).contiguous().view(k * B, ...)``
        (``rl4co/utils/ops.py:11-14``) and ``unbatchify`` inverts it as
        ``view(k, B).permute(1, 0)``, so any other ordering would silently pair each instance
        with another instance's starts.

        Args:
            td: The post-reset state, carrying ``start_nodes`` ``[B, k]`` for training.
            num_starts: The count the decoder is expanding by.

        Returns:
            A ``[num_starts * B]`` int tensor, or rl4co's fixed selection when the batch has
            no start nodes, which is the case for validation and testing.
        """
        if "start_nodes" not in td.keys():
            return super().select_start_nodes(td, num_starts)

        start_nodes = td["start_nodes"]
        if int(start_nodes.size(-1)) != int(num_starts):
            raise ValueError(
                f"batch carries {start_nodes.size(-1)} start nodes but the decoder is "
                f"expanding by num_starts={num_starts}; the reward would be unbatchified "
                "incorrectly"
            )
        return start_nodes.transpose(0, 1).reshape(-1)
