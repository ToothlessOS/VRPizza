"""Tree node and incumbent types for the POMO-guided CVRP search.

The specification's node carries a ``state: CVRPState`` and leaves that type unspecified. What
it must be is the *mutable* part of an rl4co CVRP state: ``_step`` writes only ``visited``,
``used_capacity``, ``current_node`` and ``action_mask``, while ``locs`` and ``demand`` are read
and never written. So :class:`CVRPState` holds the four mutable fields and each node points at a
shared :class:`InstanceBank` for the rest, rather than cloning a whole ``TensorDict`` per node.

The specification's container types for the policy information are also worth deviating from, for
a reason that turned out to be performance before it was memory. ``action_priors: dict[int,
float]`` at ``N=100`` costs on the order of 14 KB per node -- boxed keys and boxed floats --
against ~900 B for the state itself, so the specification's own container would dominate node
memory by more than an order of magnitude and hold the search to a batch of one instance. It is
therefore a dense ``[N+1]`` tensor with infeasible actions at zero.

``ranked_customer_actions`` is a tuple of small ints rather than either the specification's list
or a tensor, and the deciding factor was synchronisation rather than size. Expansion reads exactly
one entry per visit, and reading that entry off a device tensor forces a device synchronisation
on every expansion -- tens of thousands per round in a search that is otherwise a pure tensor
pipeline. As a tuple it is pure host memory, and CPython's cache of small integers means the
entries cost pointers rather than objects, so it comes out cheaper than either alternative.

Every invariant the specification states for both fields still holds; only the containers
changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from tensordict.tensordict import TensorDict
from torch import Tensor

__all__ = ["DEPOT_ACTION", "CVRPState", "InstanceBank", "Incumbent", "MCTSNode"]

#: The depot is action index 0 in rl4co's CVRP action space, sharing the ``[N+1]`` axis with the
#: customers. It is the only action that ends a route and resets ``used_capacity``.
DEPOT_ACTION = 0


@dataclass
class InstanceBank:
    """The immutable per-instance data the decoder and the environment read but never write.

    Fields:
        aug_locs: ``[A, N+1, 2]`` locations, depot first, one row per dihedral augmentation.
            ``StateAugmentation`` with its default ``feats`` transforms ``locs`` only, so the
            demand below is shared across all ``A`` rows.
        demand: ``[N]`` demands, already normalised by the instance's capacity.
        n_loc: ``N``, the customer count.
        num_augment: ``A``.
    """

    aug_locs: Tensor
    demand: Tensor
    n_loc: int
    num_augment: int

    @property
    def vehicle_capacity(self) -> Tensor:
        """``[1]`` capacity, always 1.0 because the bank's demands are normalised.

        Load-bearing rather than cosmetic: ``VRPContext._state_embedding`` reads this field to
        embed the remaining capacity, so a materialised state that omits it cannot be decoded.
        """
        return torch.ones(1, dtype=self.aug_locs.dtype, device=self.aug_locs.device)


@dataclass
class CVRPState:
    """The mutable part of one CVRP partial solution.

    Every field is a small tensor; the shared instance data lives in an :class:`InstanceBank`.
    :meth:`clone` is what makes a child independent of its parent, which is the specification's
    section 5.1 requirement -- the alternative, :meth:`tensordict.TensorDict.clone` on a whole
    rl4co state, costs about three times as much at ``N=100`` and six times at ``N=1000``,
    because it copies ``locs`` and ``demand`` that no step ever modifies.

    Fields:
        visited: ``[N+1]`` uint8, index 0 the depot slot.
        used_capacity: ``[1]`` normalised load of the current vehicle. Reset to 0 by a depot
            action, even on a completed episode, which is harmless: the objective reads only
            ``locs`` and the action sequence.
        current_node: ``[1]`` int64, 0 at the depot.
        action_mask: ``[N+1]`` bool, True where the action is feasible.
        done: Whether every customer has been served.
    """

    visited: Tensor
    used_capacity: Tensor
    current_node: Tensor
    action_mask: Tensor
    done: bool = False

    def clone(self) -> CVRPState:
        """Return a deep copy, so stepping the copy cannot disturb the original."""
        return CVRPState(
            visited=self.visited.clone(),
            used_capacity=self.used_capacity.clone(),
            current_node=self.current_node.clone(),
            action_mask=self.action_mask.clone(),
            done=self.done,
        )

    def feasible_actions(self) -> list[int]:
        """Indices of the currently feasible actions, ascending."""
        return self.action_mask.nonzero().flatten().tolist()

    def depot_legal(self) -> bool:
        """Whether a depot return is feasible from this state."""
        return bool(self.action_mask[DEPOT_ACTION].item())


def _first_row(field: Tensor, trailing: int) -> Tensor:
    """Index the leading batch axis off a field that may or may not still have one.

    TensorDict indexing drops the batch dimension, so a field read from ``td[0]`` arrives one
    rank lower than the same field read from a ``[1]``-batched ``td``. Rather than force callers
    to know which they have, restore the axis when it is missing.
    """
    if field.dim() < trailing + 1:
        field = field.unsqueeze(0)
    return field[0].clone()


def state_from_td(td: TensorDict) -> CVRPState:
    """Extract a single-row :class:`CVRPState` from a one-row environment state.

    Accepts either a ``[1]``-batched state or a bare row, since both arrive in practice: the
    round loop indexes a ``[L]`` batch down to one row, while ``env.step`` returns the batch.

    Args:
        td: A one-row rl4co CVRP state.

    Returns:
        The state, detached from that tensor's storage.
    """
    return CVRPState(
        visited=_first_row(td["visited"], trailing=1),
        used_capacity=_first_row(td["used_capacity"], trailing=1),
        current_node=_first_row(td["current_node"], trailing=1),
        action_mask=_first_row(td["action_mask"], trailing=1),
        done=bool(_first_row(td["done"], trailing=0).item()),
    )


def batch_states(
    states: list[CVRPState], aug_index: Tensor, banks: list[InstanceBank]
) -> TensorDict:
    """Materialise a batched rl4co state from per-node states.

    This is the inverse of :func:`state_from_td` at batch scale, and the point at which the
    compact representation has to become a real ``TensorDict`` again: the decoder, the
    environment's mask and the objective all expect one.

    Args:
        states: ``L`` states, each belonging to the instance named by ``aug_index``.
        aug_index: ``[L]`` indices into each state's bank's ``aug_locs``.
        banks: The bank backing each row, ``L`` of them.

    Returns:
        A ``[L]``-batched state carrying exactly the keys the decoder, ``env.step`` and
        ``env.get_reward`` read. ``depot`` is included for completeness even though nothing in
        the current call path reads it, since ``check_solution_validity`` is free to.
    """
    locs = torch.stack([bank.aug_locs[a] for bank, a in zip(banks, aug_index.tolist())])
    demand = torch.stack([bank.demand for bank in banks])
    return TensorDict(
        {
            "depot": locs[:, 0],
            "locs": locs,
            "demand": demand,
            "current_node": torch.stack([s.current_node for s in states]),
            "used_capacity": torch.stack([s.used_capacity for s in states]),
            "visited": torch.stack([s.visited for s in states]),
            "vehicle_capacity": torch.ones(
                len(states), 1, dtype=locs.dtype, device=locs.device
            ),
            "action_mask": torch.stack([s.action_mask for s in states]),
            "done": torch.tensor(
                [s.done for s in states], dtype=torch.bool, device=locs.device
            ).unsqueeze(-1),
            "action": torch.zeros(len(states), dtype=torch.long, device=locs.device),
        },
        batch_size=[len(states)],
        device=locs.device,
    )


@dataclass(eq=False)
class MCTSNode:
    """One partial CVRP solution in the search tree.

    Field names and meanings follow the specification's section 5, with the two container
    substitutions described in the module docstring. ``prefix`` is not in the specification but
    is needed for the objective: ``env.get_reward`` scores a whole action sequence from the
    root, and a rollout only produces the tail of it.

    Args:
        state: This node's partial solution.
        instance_row: Which instance of the batch this node belongs to.
        aug_index: Which dihedral augmentation of that instance this node's coordinates are in.
        parent: The node this one was reached from, or ``None`` at a root.
        action_from_parent: The action taken from the parent, or ``None`` at a root.
        prefix: Actions from the root through ``action_from_parent`` inclusive, so that
            concatenating a rollout's actions onto it reproduces the full sequence a root-based
            rollout would have produced.
    """

    state: CVRPState
    instance_row: int
    aug_index: int
    parent: MCTSNode | None = None
    action_from_parent: int | None = None
    prefix: tuple[int, ...] = ()

    #: The parent's prior for the action that reached this node, as a Python float.
    #:
    #: Held separately from ``action_priors`` for one reason: PUCT reads exactly one prior per
    #: child, and reading it out of a device tensor with ``float(...)`` forces a device
    #: synchronisation. Selection runs once per level of every descent, so that would be
    #: hundreds of thousands of synchronisations in a run -- enough to dominate the search, as
    #: measured. One float per node, filled in a batched gather at child creation, removes all
    #: of them.
    prior_from_parent: float = 0.0

    # Policy information, populated once per node.
    policy_initialized: bool = False
    action_priors: Tensor | None = None
    #: Feasible customer actions in descending prior order, depot excluded (section 5.1).
    #:
    #: A tuple of small ints rather than the specification's list, and not a device tensor.
    #: Expansion reads exactly one entry per visit, so a tensor would synchronise on every
    #: expansion; a tuple costs pointers, because CPython caches integers in this range, and
    #: ``num_customers`` is then just its length.
    ranked_customer_actions: tuple[int, ...] = ()
    num_customers: int = 0
    depot_legal: bool = False

    # Tree structure.
    children: dict[int, MCTSNode] = field(default_factory=dict)
    expanded_customer_count: int = 0

    # MCTS statistics. Higher Q is better.
    N: int = 0
    W: float = 0.0
    Q: float = 0.0

    terminal: bool = False

    def ranking(self) -> list[int]:
        """The feasible customer actions in descending prior order, depot excluded.

        A list view of ``ranked_customer_actions``, for callers that want the specification's
        list semantics.
        """
        return list(self.ranked_customer_actions)

    def customer_children(self) -> list[MCTSNode]:
        """The children reached by a customer action, i.e. all of them but the depot's."""
        return [c for a, c in self.children.items() if a != DEPOT_ACTION]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"MCTSNode(depth={len(self.prefix)}, N={self.N}, Q={self.Q:.4f}, "
            f"children={len(self.children)}, expanded={self.expanded_customer_count}, "
            f"terminal={self.terminal})"
        )


@dataclass
class Incumbent:
    """The best complete solution seen so far, per instance.

    The specification keeps a scalar incumbent. This is the batched form of the same thing: the
    search runs every root of every instance in a batch, so the incumbent is one entry per
    instance. It is seeded from the ordinary POMO baseline, which is why the search can never
    return something worse than the baseline -- the guarantee is structural, not a property the
    search has to earn.

    Args:
        baseline_cost: ``[B]`` the baseline's cost per instance, for reporting.
        cost: ``[B]`` best cost found so far, initialised to the baseline.
        actions: Per instance, the action sequence achieving :attr:`cost`, or ``None`` before a
            baseline is seeded.
        improved: ``[B]`` whether any rollout beat the baseline. False is a valid outcome, not
            an error.
    """

    baseline_cost: Tensor
    cost: Tensor
    actions: list[list[int] | None] = field(default_factory=list)
    improved: Tensor | None = None

    @classmethod
    def from_costs(cls, costs: Tensor, actions: list[list[int]]) -> Incumbent:
        """Seed an incumbent from one complete solution per instance.

        Args:
            costs: ``[B]`` costs of those solutions.
            actions: ``B`` action sequences.

        Returns:
            An incumbent at those solutions, with :attr:`improved` all False.
        """
        return cls(
            baseline_cost=costs.clone(),
            cost=costs.clone(),
            actions=list(actions),
            improved=torch.zeros_like(costs, dtype=torch.bool),
        )

    def update_if_better(self, instance_row: int, cost: float, actions: list[int]) -> bool:
        """Record a solution if it beats the incumbent for that instance.

        Args:
            instance_row: Which instance the solution belongs to.
            cost: The solution's total route length.
            actions: The complete action sequence, from the root.

        Returns:
            Whether the incumbent improved.
        """
        if cost < float(self.cost[instance_row].item()):
            self.cost[instance_row] = cost
            self.actions[instance_row] = list(actions)
            self.improved[instance_row] = True
            return True
        return False
