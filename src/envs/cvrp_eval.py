"""A CVRP environment for scoring a checkpoint on batches a driver built.

:class:`~src.envs.CVRPEnv` is the training environment: it owns a data file, carves a
held-out split out of it and serves datasets to ``RL4COLitModule``. None of that is wanted
when the only goal is to run the POMO rollout over instances that are already in memory --
and some of it actively gets in the way, since that class refuses a file without a reference
tour (``vrplib_192instances.txt`` has none) and its index cache writes sidecar files next to
the data.

What the rollout does need is exactly what :class:`~src.envs.cvrp.InstanceInputMixin`
provides, which is why this class is that mixin plus rl4co's environment and nothing else.
It holds no data path and builds no dataset.

This exists as a separate class rather than as a flag on :class:`~src.envs.CVRPEnv` so that
the training path keeps its guarantees: a training run that silently lost its reference tour
or its validation split would be a much worse failure than an evaluation script that has to
name its inputs explicitly.
"""

from __future__ import annotations

from typing import NoReturn

from rl4co.envs.routing.cvrp.env import CVRPEnv as _RL4COCVRPEnv

from src.envs.cvrp import InstanceInputMixin

__all__ = ["EvalCVRPEnv"]


class EvalCVRPEnv(InstanceInputMixin, _RL4COCVRPEnv):
    """A CVRP environment with no dataset, for evaluating pre-built instance batches.

    Args:
        num_loc: Number of customers ``N``. Load-bearing rather than cosmetic: rl4co selects
            multi-start nodes as ``arange(num_starts) % generator.num_loc + 1``, so a value
            below ``N`` silently confines every start to the first ``num_loc`` customers and
            degrades the rollout without raising. Callers must pass the size of the batch
            they actually intend to score; a batch of a different size is not rejected here,
            so pair this with the assertion in the driver's batch builder.
        capacity: Vehicle capacity in the file's demand units, forwarded to the generator.
            Note this is *not* what bounds the action mask: demands are normalised by the
            batch's own capacity in
            :meth:`~src.envs.cvrp.InstanceInputMixin.make_input_td` and the environment's
            ``vehicle_capacity`` stays at the generator default of ``1.0``. Passing a
            capacity here would leave the mask comparing normalised demands against a
            full-scale capacity, which would make every action appear feasible.
        check_solution: Whether ``get_reward`` validates each produced solution. Defaults to
            ``False`` because the check is a Python loop over the decode steps; worth turning
            on for a smoke test, too slow to leave on for a full benchmark.
        **kwargs: Forwarded to :class:`~rl4co.envs.CVRPEnv`.
    """

    def __init__(
        self,
        num_loc: int,
        capacity: float = 1.0,
        check_solution: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            generator_params={"num_loc": int(num_loc), "capacity": float(capacity)},
            check_solution=check_solution,
            **kwargs,
        )

    def dataset(self, *args, **kwargs) -> NoReturn:
        """Refuse to build a dataset.

        Deliberate rather than omitted: Lightning calls ``env.dataset`` from ``setup`` and
        ``on_train_epoch_end``, so a future ``trainer.validate`` against this environment
        fails here, loudly, instead of resolving to rl4co's generator and quietly scoring
        random instances.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "EvalCVRPEnv has no dataset; batches are built by the caller and passed "
            "straight to the policy"
        )
