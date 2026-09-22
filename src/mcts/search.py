"""POMO-prior PUCT search over CVRP partial solutions.

The specification in ``prompts/MCTS.md`` describes a per-instance loop running one simulation at
a time. This module implements the same algorithm, but every root of an instance advances in
lockstep, because the per-instance loop cannot be batched and the batch is the whole performance
story: a baseline ``vrp100`` batch decodes 256 instances x 8 augmentations x 100 starts =
204,800 rows, and one simulation of one root is a single row. Lockstep rounds hold the decoder
call width at ``instances x 8N`` no matter how far the trees diverge.

Three properties make the restructuring an exact reordering rather than an approximation:

* Every root yields exactly one leaf per simulation, so a round is a fixed-size batch.
* Roots are independent within a round and every backup completes before the next round begins,
  so no root ever observes another's half-finished round.
* The incumbent is write-only during the search -- no decision reads it -- so sharing one across
  roots cannot move a trajectory.

The one place the reordering is visible in the numbers rather than in the control flow is float
arithmetic: batched GEMM and attention under TF32 can flip an argmax on a near-tie, so a rollout
batched over 800 rows need not be bit-identical to the same rollout evaluated alone. The
evaluation driver already documents that property of the pipeline as a whole, and it is why
"equivalent" here means "equivalent up to batch-shape float noise, greedy only".

**Timing of policy initialisation.** Section 6 says priors are computed "when a node first needs
expansion or selection". Here they are computed at creation, batched across the round's new
nodes. Every node is initialised exactly once either way, and a node's priors are a deterministic
function of a state fixed at creation, so the results agree; the reason for the deviation is that
the lazy version would put a model call inside the descent phase and serialise the round.

Creating a node's depot child at initialisation has a subtlety worth spelling out, because it is
the one place where a plausible implementation silently reads uninitialised data. A depot child
is registered in its parent's ``children`` immediately, so PUCT can select it -- but its own
priors are not known until it too is initialised, and a node whose priors are missing cannot be
expanded. Rather than rely on an argument about visit counts to show the selection can never
happen first, :func:`initialize_policy` runs to a fixed point: it initialises the depot children
it just created before returning. The chain is at most two links long, because a depot child sits
at the depot, where the mask forbids another depot action while any customer remains feasible.
The extra cost is one decoder call per round over one row per new node, against roughly ``N+R``
decoder calls per round for the rollouts.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
from tensordict.tensordict import TensorDict
from rl4co.utils.ops import batchify
from torch import Tensor

from src.mcts.config import POMOMCTSConfig
from src.mcts.node import (
    DEPOT_ACTION,
    Incumbent,
    InstanceBank,
    MCTSNode,
    batch_states,
    state_from_td,
)
from src.mcts.oracle import POMOOracle
from src.mcts.stats import SearchStatistics

__all__ = [
    "MCTSResult",
    "allowed_customer_width",
    "backup",
    "cfg_value",
    "initialize_policy",
    "maybe_expand_one",
    "select_child_puct",
    "step_children",
    "solve_batch_with_pomo_mcts",
]

#: ``PrecomputedCache`` field names, in declaration order. ``graph_context`` is the integer 0 for
#: these checkpoints, since both train with ``use_graph_context=False``, so it has no rows to
#: select and is passed through untouched.
_CACHE_FIELDS = ("node_embeddings", "graph_context", "glimpse_key", "glimpse_val", "logit_key")


@dataclass
class MCTSResult:
    """The search's answer for a batch of instances.

    Args:
        cost: ``[B]`` incumbent cost per instance, never worse than ``baseline_cost``.
        actions: ``B`` action sequences achieving :attr:`cost`.
        baseline_cost: ``[B]`` the ordinary POMO rollout's cost, for the paired comparison.
        baseline_actions: ``B`` action sequences achieving :attr:`baseline_cost`.
        stats: Cost accounting for the run.
        roots: The tree's roots, only when ``keep_tree`` is set -- the verification harness needs
            them to check the specification's node invariants, a benchmark does not.
    """

    cost: Tensor
    actions: list[list[int]]
    baseline_cost: Tensor
    baseline_actions: list[list[int]]
    stats: SearchStatistics = field(default_factory=SearchStatistics)
    roots: list[MCTSNode] | None = None

    @property
    def improved(self) -> Tensor:
        """``[B]`` whether the search beat the baseline on each instance."""
        return self.cost < self.baseline_cost


# ---------------------------------------------------------------------------------------
# Tree rules, one function per clause of the specification
# ---------------------------------------------------------------------------------------


def allowed_customer_width(node: MCTSNode, cfg: POMOMCTSConfig) -> int:
    """``K(s)``, the number of customer children this node may currently expose (section 7.1).

    The depot child is deliberately not counted here. Section 7.3 gives it a slot outside the
    frontier, which is what keeps a route's return available even when the customer frontier is
    saturated.
    """
    return cfg.customer_width(node.num_customers, node.N)


def maybe_expand_one(node: MCTSNode, cfg: POMOMCTSConfig) -> int | None:
    """The next customer action to materialise, or ``None`` if the frontier is full (section 7.5).

    Section 7.4: the next action is always ``ranked_customer_actions[expanded_customer_count]``,
    deterministically, with no resampling on later visits and no discarding of earlier children.

    The child itself is not created here. Round batching creates every root's child in a single
    environment step, so this answers "which action" and the caller does the stepping.
    """
    if node.terminal:
        return None
    if node.expanded_customer_count >= allowed_customer_width(node, cfg):
        return None
    return int(node.ranked_customer_actions[node.expanded_customer_count])


def select_child_puct(node: MCTSNode, c_puct: float) -> tuple[int, MCTSNode]:
    """Pick among materialised children by PUCT (section 8.2).

    ``Q(s,a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s,a))``, maximised. Actions are walked in
    ascending order and the comparison is strict, so ties go to the lowest action index and the
    choice is a deterministic function of the tree -- section 14.4 asks for exactly that.

    The prior is read from ``child.prior_from_parent``, a plain float, rather than out of the
    parent's prior vector. That is not a micro-optimisation: indexing a device tensor here forces
    a synchronisation on every level of every descent, and the search is otherwise a pure
    tensor pipeline where those synchronisations would dominate the wall clock.

    Args:
        node: The node to select from. Children must be materialised and priors set.
        c_puct: Exploration weight.

    Returns:
        ``(action, child)``.

    Raises:
        RuntimeError: If the node has no children. CVRP's mask always leaves a feasible action,
            so this means the node was expanded and initialised out of order.
    """
    sqrt_visits = math.sqrt(node.N)
    best_action, best_child, best_score = None, None, -math.inf
    for action, child in node.children.items():
        score = child.Q + c_puct * child.prior_from_parent * sqrt_visits / (1 + child.N)
        if score > best_score or (score == best_score and action < best_action):
            best_action, best_child, best_score = action, child, score
    if best_child is None:  # pragma: no cover - CVRP masks guarantee a feasible action
        raise RuntimeError("node has no children and no expandable action")
    return best_action, best_child


def backup(path: list[MCTSNode], value: float) -> None:
    """Propagate a rollout's value to every node on its path (section 10).

    The same value is added at every node, with no sign alternation: CVRP is a single-agent
    minimisation, not a two-player zero-sum game, and the objective is global route length rather
    than a per-step reward.
    """
    for node in path:
        node.N += 1
        node.W += value
        node.Q = node.W / node.N


def cfg_value(cfg: POMOMCTSConfig, cost: float, reference: float) -> float:
    """Map a cost to the scalar the tree maximises (section 8.1).

    CVRP minimises and PUCT maximises, so the value has to be oriented. The specification's
    normalised form makes an improving solution slightly positive and everything else negative --
    including the baseline itself, at exactly zero, since ``C_ref`` *is* the baseline.

    The consequence is worth stating rather than discovering later. Unvisited children carry
    ``Q = 0`` by section 5.1, and 0 is the best value an unvisited child can be credited with, so
    an unvisited child always outranks a visited one that failed to improve on the baseline, and
    a node's exploration bonus shrinks only as ``1/(1 + N)``. With the spread of
    ``(C_ref - C)/C_ref`` around 0.02-0.06 and the root bonus around 0.05-0.4, ``c_puct = 1.0``
    makes the search close to prior-ordered breadth expansion for the first several visits of
    each child. That is the specification's default and it is kept; it is also why ``c_puct`` is
    the first field to sweep.
    """
    if cfg.value_form == "negative_cost":
        return -cost
    return (reference - cost) / max(abs(reference), 1e-8)


# ---------------------------------------------------------------------------------------
# Policy initialisation
# ---------------------------------------------------------------------------------------


def _device_of(nodes: list[MCTSNode]):
    """The device a node's tensors live on, for building index tensors that have to match.

    Every index tensor here is passed to ``index_select`` or ``TensorDict.set`` against a state
    that is already on the accelerator, and a CPU index against a CUDA target raises rather than
    being moved implicitly.
    """
    return nodes[0].state.visited.device


def _cache_rows(nodes: list[MCTSNode], num_instances: int) -> Tensor:
    """Decoder cache row for each node.

    ``StateAugmentation`` tiles augmentation-major: ``_batchify_single`` expands to ``[A, B, ...]``
    and views to ``[A*B, ...]``, so the augmented row for instance ``b`` under augmentation ``a``
    is ``a*B + b``, not ``b*A + a``. Its own docstring claims the latter and is wrong. The
    convention is pinned by ``unbatchify(reward, (num_augment, num_starts))`` producing
    ``[b, aug, start]``, which only holds if the flat index is augmentation-major.

    Note this is *not* the augmentation index, and passing one where the other belongs is a
    quiet off-by-a-factor-of-``B``: the cache has ``A*B`` rows but a bank has ``A``.
    """
    return torch.tensor(
        [node.aug_index * num_instances + node.instance_row for node in nodes],
        dtype=torch.long,
        device=_device_of(nodes),
    )


def _aug_indices(nodes: list[MCTSNode]) -> Tensor:
    """Row of each node's :class:`~src.mcts.node.InstanceBank` ``aug_locs`` this node sits in."""
    return torch.tensor(
        [node.aug_index for node in nodes], dtype=torch.long, device=_device_of(nodes)
    )


def _index_cache(cached, rows: Tensor):
    """Select cache rows, keeping the batch dimension and passing non-tensor fields through."""
    return type(cached)(
        **{
            name: (value.index_select(0, rows) if torch.is_tensor(value) else value)
            for name, value in zip(_CACHE_FIELDS, cached.fields)
        }
    )


def _make_child(parent: MCTSNode, action: int, prior: float) -> MCTSNode:
    """A child of ``parent`` reached by ``action``, registered immediately.

    The state is seeded with a copy of the parent's because it is not known until the
    environment steps; :func:`step_children` overwrites it. Keeping creation and stepping apart
    is what lets a round step every root's child in one environment call. Registration happens
    here rather than after stepping so that a caller's view of ``children`` is never stale.

    Args:
        parent: The node being expanded.
        action: The action leading to the child.
        prior: The parent's prior for that action, already read off the device by the caller.
    """
    child = MCTSNode(
        state=parent.state.clone(),
        instance_row=parent.instance_row,
        aug_index=parent.aug_index,
        parent=parent,
        action_from_parent=action,
        prefix=parent.prefix + (action,),
        prior_from_parent=float(prior),
    )
    parent.children[action] = child
    if action != DEPOT_ACTION:
        parent.expanded_customer_count += 1
    return child


def step_children(
    requests: list[tuple[MCTSNode, int]], env, banks: list[InstanceBank], num_instances: int
) -> list[MCTSNode]:
    """Create children in one batched environment step.

    Stepping through ``env.step`` rather than reimplementing ``_step`` is deliberate. The
    capacity reset, the ``visited`` scatter and the three masking rules are subtle enough that a
    hand-rolled copy would be the likeliest place in this module to be wrong -- and the batched
    step is faster than one call per root besides.

    Args:
        requests: ``(parent, action)`` pairs.
        env: The CVRP environment.
        banks: One bank per instance, indexed by ``instance_row``.
        num_instances: ``B``, for the cache row mapping.

    Returns:
        The new children, in request order, with their states filled in.
    """
    if not requests:
        return []

    parents = [parent for parent, _ in requests]
    actions = torch.tensor(
        [action for _, action in requests], dtype=torch.long, device=_device_of(parents)
    )
    # One batched gather and one host transfer, rather than a device read per child: the priors
    # are needed as floats on the child for PUCT, and reading them one at a time would
    # synchronise once per node created.
    priors = (
        torch.stack([parent.action_priors for parent in parents])
        .gather(1, actions.unsqueeze(1))
        .squeeze(1)
        .tolist()
    )
    children = [
        _make_child(parent, action, prior)
        for (parent, action), prior in zip(requests, priors)
    ]

    td = batch_states(
        [parent.state for parent in parents],
        _aug_indices(parents),
        [banks[parent.instance_row] for parent in parents],
    )
    td.set("action", actions)
    stepped = env.step(td)["next"]
    for index, child in enumerate(children):
        child.state = state_from_td(stepped[index])
    return children


def initialize_policy(
    nodes: list[MCTSNode],
    oracle: POMOOracle,
    cfg: POMOMCTSConfig,
    env,
    banks: list[InstanceBank],
    num_instances: int,
) -> None:
    """Populate priors, ranking, depot legality and terminality (section 6).

    Terminal nodes short-circuit before any decoding, per section 6.1: a completed solution has
    no children to rank, so decoding one would spend a call to discover nothing.

    Already-initialised nodes are skipped, so the function is idempotent.

    Args:
        nodes: Nodes needing policy information.
        oracle: The policy wrapper, with its root cache bound.
        cfg: Search configuration.
        env: The CVRP environment, used to step depot children.
        banks: One bank per instance.
        num_instances: ``B``.
    """
    queue = [node for node in nodes if not node.policy_initialized]
    while queue:
        for node in queue:
            if node.state.done:
                node.terminal = True
                node.policy_initialized = True

        live = [node for node in queue if not node.terminal]
        depot_requests: list[tuple[MCTSNode, int]] = []

        if live:
            td = batch_states(
                [node.state for node in live],
                _aug_indices(live),
                [banks[node.instance_row] for node in live],
            )
            priors = oracle.priors(
                td, _index_cache(oracle.root_cache, _cache_rows(live, num_instances))
            )

            # Everything below is computed for the whole batch at once. The per-node version
            # reads a device scalar per node per field, which is three synchronisations per node
            # created -- tens of thousands in a run, against a search that is otherwise a pure
            # tensor pipeline.
            mask = td["action_mask"]
            # The depot is excluded from the ranking by making it unsortable rather than by
            # filtering, so the ranking stays a rectangular tensor. It lands immediately after
            # the feasible customers, and `num_customers` cuts it off.
            scores = priors.masked_fill(~mask, float("-inf"))
            scores[:, DEPOT_ACTION] = float("-inf")
            # Descending by prior; stable, so equal priors keep ascending action order and the
            # ranking is a deterministic function of the state.
            ranking = torch.argsort(-scores, dim=1, stable=True)

            num_customers = mask[:, 1:].sum(-1).tolist()
            depot_legal = mask[:, DEPOT_ACTION].tolist()
            # One host transfer for the whole batch. The ranking is consumed one entry at a time
            # during expansion, and reading a single entry off the device would synchronise on
            # every expansion; as a tuple of small ints it costs pointers, since CPython caches
            # integers in this range.
            ranking_rows = ranking.tolist()

            for index, node in enumerate(live):
                # Infeasible actions come out of `process_logits` as exp(-inf) = 0, so the prior
                # vector already carries the specification's zero-on-invalid property, and an
                # invalid action can never be ranked.
                node.action_priors = priors[index]
                node.depot_legal = bool(depot_legal[index])
                count = int(num_customers[index])
                node.ranked_customer_actions = tuple(ranking_rows[index][:count])
                node.num_customers = count
                node.policy_initialized = True

                # Section 7.3: the depot child is created as soon as the node is initialised,
                # whenever it is legal, and never consumes a customer slot.
                if node.depot_legal and DEPOT_ACTION not in node.children:
                    depot_requests.append((node, DEPOT_ACTION))

        # Initialise the depot children just created before returning, so no node with children
        # ever lacks the priors needed to expand them. The chain is at most two links long.
        queue = (
            step_children(depot_requests, env, banks, num_instances) if depot_requests else []
        )


# ---------------------------------------------------------------------------------------
# The round loop
# ---------------------------------------------------------------------------------------


def _descend(
    root: MCTSNode, cfg: POMOMCTSConfig, stats: SearchStatistics | None = None
) -> tuple[list[MCTSNode], int | None]:
    """Walk down from a root to the node this simulation will evaluate.

    Section 8.3's decision order, with steps 1 and 2 already done at initialisation: expand one
    customer child if widening permits, otherwise select an existing child by PUCT.

    Args:
        root: The root to descend from.
        cfg: Search configuration.
        stats: Optional sink, so a caller can tell how often each branch was taken. The split
            between expansions and PUCT selections is the cheapest way to see whether a budget
            was large enough for the search to do anything but build depth-1 trees.

    Returns:
        ``(path, expand_action)``. When ``expand_action`` is set, the simulation creates that
        child of ``path[-1]`` and evaluates it; otherwise ``path[-1]`` is terminal and is
        evaluated as it stands (section 9.3).
    """
    node = root
    path = [node]
    while True:
        if node.terminal:
            if stats is not None:
                stats.terminal_evaluations += 1
            return path, None
        action = maybe_expand_one(node, cfg)
        if action is not None:
            if stats is not None:
                stats.expansions += 1
            return path, action
        if stats is not None:
            stats.puct_selections += 1
        _, child = select_child_puct(node, cfg.c_puct)
        node = child
        path.append(node)


def solve_batch_with_pomo_mcts(
    batch: TensorDict,
    model,
    env,
    cfg: POMOMCTSConfig,
    stats: SearchStatistics | None = None,
) -> MCTSResult:
    """Search every instance of a batch, returning the best solution found for each.

    The ordinary POMO rollout is computed here too, from the same roots and in the same pass: it
    supplies the reference cost for the value normalisation, the incumbent's seed solution (with
    its actions, which ``_inference_step`` discards) and the baseline row of the comparison. That
    is why the returned cost can never be worse than the baseline -- the guarantee is structural,
    not something the search has to earn, so a run showing "MCTS >= POMO" is evidence of nothing
    and only a matched-compute control says anything.

    Args:
        batch: Instances in the format ``InstanceInputMixin.make_input_td`` expects.
        model: A fitted ``POMOSL`` or ``POMORL`` in eval mode.
        env: The CVRP environment, sized to the batch's customer count.
        cfg: Search configuration.
        stats: Optional statistics sink to accumulate into.

    Returns:
        The search's answer. See :class:`MCTSResult`.
    """
    stats = stats if stats is not None else SearchStatistics()
    num_instances = int(batch.shape[0])
    stats.instances += num_instances

    torch.manual_seed(cfg.seed)
    oracle = POMOOracle(model, env, cfg, stats)

    reset = env.reset(env.make_input_td(batch))
    augment = getattr(model, "augment", None)
    num_augment = int(model.num_augment) if augment is not None else 1
    td_aug = augment(reset) if augment is not None else reset

    n_loc = int(td_aug["demand"].shape[-1])
    num_starts = int(env.get_num_starts(reset))
    if cfg.max_starts is not None:
        num_starts = min(num_starts, int(cfg.max_starts))

    banks = [
        InstanceBank(
            aug_locs=torch.stack(
                [td_aug[aug_index * num_instances + row]["locs"] for aug_index in range(num_augment)]
            ),
            demand=td_aug[row]["demand"].clone(),
            n_loc=n_loc,
            num_augment=num_augment,
        )
        for row in range(num_instances)
    ]

    # One encoder pass for the whole forest. The cache depends only on the augmented
    # coordinates, so every node of every tree rooted at one row reuses that row.
    oracle.root_cache = oracle.encode(td_aug)
    stats.roots += num_instances * num_augment * num_starts

    # -- roots -----------------------------------------------------------------------
    # Forced first action, no mask consulted, exactly as rl4co's `pre_decoder_hook` does it. The
    # start permutation comes from the environment so a change there cannot silently diverge.
    starts = env.select_start_nodes(reset, num_starts=num_starts).flatten()
    orders = _root_order(num_instances, num_augment, num_starts)
    root_actions = torch.tensor(
        [int(starts[start * num_instances + row]) for row, _, start in orders],
        dtype=torch.long,
        device=reset.device,
    )

    td_roots = batchify(td_aug, num_starts)
    td_roots.set("action", root_actions)
    env.step(td_roots)
    cached_roots = oracle.root_cache.batchify(num_starts)

    roots = [
        MCTSNode(
            state=state_from_td(td_roots[index]),
            instance_row=row,
            aug_index=aug_index,
            prefix=(int(root_actions[index]),),
        )
        for index, (row, aug_index, _) in enumerate(orders)
    ]

    # -- baseline --------------------------------------------------------------------
    started = time.time()
    baseline = oracle.complete(td_roots, cached_roots, [[node.prefix[0]] for node in roots])
    stats.rollouts += len(roots)
    stats.invalid_rollouts += int((~baseline.valid).sum().item())
    stats.baseline_seconds += time.time() - started

    baseline_cost, baseline_actions = _best_per_instance(
        baseline.cost, baseline.actions, orders, num_instances
    )
    incumbent = Incumbent.from_costs(baseline_cost, baseline_actions)

    # -- rounds ----------------------------------------------------------------------
    started = time.time()
    pending: list[MCTSNode] = list(roots)
    for _ in range(max(0, cfg.sims_per_root)):
        # Priors for everything created in the previous round, batched into one decoder call.
        initialize_policy(pending, oracle, cfg, env, banks, num_instances)
        pending = []

        paths: list[list[MCTSNode]] = []
        requests: list[tuple[MCTSNode, int]] = []
        planned: list[bool] = []
        for root in roots:
            path, action = _descend(root, cfg, stats)
            paths.append(path)
            planned.append(action is not None)
            if action is not None:
                requests.append((path[-1], action))

        created = step_children(requests, env, banks, num_instances)
        created_iter = iter(created)

        leaves: list[MCTSNode] = []
        leaf_paths: list[list[MCTSNode]] = []
        for path, expands in zip(paths, planned):
            if expands:
                child = next(created_iter)
                leaves.append(child)
                leaf_paths.append(path + [child])
            else:
                leaves.append(path[-1])
                leaf_paths.append(path)

        leaf_rows = _cache_rows(leaves, num_instances)
        leaf_td = batch_states(
            [node.state for node in leaves],
            _aug_indices(leaves),
            [banks[node.instance_row] for node in leaves],
        )
        rolled = oracle.complete(
            leaf_td,
            _index_cache(oracle.root_cache, leaf_rows),
            [list(node.prefix) for node in leaves],
        )

        stats.simulations += 1
        stats.rollouts += len(leaves)
        stats.invalid_rollouts += int((~rolled.valid).sum().item())
        if cfg.store_rollout_solutions:
            stats.solutions.extend(rolled.actions)

        for index, node in enumerate(leaves):
            if not bool(rolled.valid[index]):
                continue
            cost = float(rolled.cost[index])
            value = cfg_value(cfg, cost, float(incumbent.baseline_cost[node.instance_row]))
            if incumbent.update_if_better(node.instance_row, cost, rolled.actions[index]):
                stats.improvements += 1
            backup(leaf_paths[index], value)

        pending.extend(created)

    stats.search_seconds += time.time() - started

    return MCTSResult(
        cost=incumbent.cost,
        actions=[actions if actions is not None else [] for actions in incumbent.actions],
        baseline_cost=baseline_cost,
        baseline_actions=baseline_actions,
        stats=stats,
        roots=roots if cfg.keep_tree else None,
    )


def _root_order(num_instances: int, num_augment: int, num_starts: int):
    """``(instance_row, aug_index, start)`` per root, in ``batchify`` order.

    ``batchify(td_aug, num_starts)`` tiles repeat-major, so the flat index of ``(b, a, s)`` is
    ``s*(B*A) + a*B + b``. Getting this wrong permutes which instance each root belongs to, and
    the mistake is invisible in any aggregate: the numbers stay plausible while the pairing
    quietly breaks.
    """
    per_start = num_instances * num_augment
    return [
        (index % num_instances, (index // num_instances) % num_augment, index // per_start)
        for index in range(per_start * num_starts)
    ]


def _best_per_instance(
    cost: Tensor, actions: list[list[int]], orders, num_instances: int
) -> tuple[Tensor, list[list[int]]]:
    """Reduce the per-root baseline to the best root per instance.

    The same reduction ``POMOInferenceMixin._inference_step`` performs with
    ``unbatchify(reward, (num_augment, num_starts)).max(...)``, expressed over the flat root list
    so the winning root's action sequence -- which that method discards -- stays available to
    seed the incumbent.
    """
    best_cost = torch.full((num_instances,), float("inf"), device=cost.device, dtype=cost.dtype)
    best_actions: list[list[int]] = [[] for _ in range(num_instances)]
    for index, (row, _, _) in enumerate(orders):
        value = float(cost[index])
        if value < float(best_cost[row]):
            best_cost[row] = cost[index]
            best_actions[row] = list(actions[index])
    return best_cost, best_actions
