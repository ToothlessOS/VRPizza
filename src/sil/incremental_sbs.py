"""Incremental stochastic beam search and the Gumbeldore trie.

A port of ``core/incremental_sbs.py`` from the Gumbeldore repository, keeping the full
policy-update machinery:

1. **Without replacement** (``log_prob_update_type="wor"``): the log-probability mass of every
   sampled trajectory is subtracted from every edge on its path, so later rounds sample from
   the residual distribution over unsampled sequences.
2. **Simple advantage update** (``"gumbeldore"``): the outcome advantage of each sampled leaf
   is added to every edge on its path as a logit perturbation.
3. **Theory update** (``"theory_gumbeldore"``): each internal edge gets
   ``sigma * (E[outcome | node] - E[outcome | parent])``, estimated from the sampled leaves.

Only (1) is used by the SIL experiment driver in stage 1; (2) and (3) are ported, unit-checked
and left dormant for the next stage. The keyed residual mass is the mechanism that makes a
sequence unsampleable once it has been drawn, which is the part under comparison with plain
beam search.
"""

from __future__ import annotations

import math
import sys
import typing
from typing import Callable, List, Optional, Tuple

import numpy as np

from src.sil import sbs

__all__ = [
    "IncrementalSBS",
    "gumbel_log_survival",
    "gumbel_without_replacement_expectation",
    "log_subtract",
]

sys.setrecursionlimit(10000)  # Trie updates recurse along a path; CVRP depth is ~2N.

_UPDATE_TYPES = ("wor", "gumbeldore", "theory_gumbeldore")


def log_subtract(x: float, y: float) -> float:
    """Return ``log(exp(x) - exp(y))``, or ``-inf`` when ``x <= y``.

    Inspired by https://stackoverflow.com/questions/778047.

    Scalar ``math`` rather than NumPy: this runs once per trie edge on every sampled leaf, and
    the NumPy ufunc call overhead dominated it. The values are computed in double precision like
    the NumPy expression (within 1 ulp of it) and land in a float32 mass array either way.
    """
    delta = float(y) - float(x)
    if delta >= 0.0:
        return -np.inf
    return float(x) + math.log1p(-math.exp(delta))


def gumbel_log_survival(x: np.ndarray) -> np.ndarray:
    """Compute ``log P(g > x) = log(1 - exp(-exp(-x)))`` for a standard Gumbel.

    Adapted from https://github.com/wouterkool/stochastic-beam-search/blob/stochastic-beam-search/fairseq/gumbel.py.
    In practice ``x := kappa - phi_i``.

    Args:
        x: Values to evaluate the log survival at.

    Returns:
        The log survival, elementwise.
    """
    with np.errstate(divide="ignore"):
        y = np.exp(-x)
        return np.where(
            # y < 1e-4, so O(y^6) <= 1e-24 and the series expansion is safe.
            x >= 10,
            -x - y / 2 + y**2 / 24 - y**4 / 2880,
            np.log(-np.expm1(-np.exp(-x))),
        )


def gumbel_without_replacement_expectation(
    sbs_leaves: List[sbs.BeamLeaf],
    leaf_eval_fn: Callable,
    normalize_importance_weights: bool = True,
) -> float:
    """Estimate the expected outcome of the policy from SBS leaves.

    Uses the importance-weighted SBS estimator from Kool et al. (2019), Section 4.2 and
    Appendix C.

    Args:
        sbs_leaves: Sampled trajectories, with ``state`` of the form used by
            :class:`IncrementalSBS` (the leaf evaluation reads ``state[1]``).
        leaf_eval_fn: Maps a leaf's client state to its outcome; maximise the outcome.
        normalize_importance_weights: Normalise the importance weights by their sum.

    Returns:
        The estimated expectation as a Python float.
    """
    if len(sbs_leaves) == 1:
        return leaf_eval_fn(sbs_leaves[0].state[1])

    sbs_leaves = sorted(sbs_leaves, key=lambda x: x.gumbel, reverse=True)
    log_probs = np.array([leaf.log_probability for leaf in sbs_leaves])
    outcomes = np.array([leaf_eval_fn(leaf.state[1]) for leaf in sbs_leaves])

    # The last leaf has the smallest Gumbel and is discarded.
    kappa = sbs_leaves[-1].gumbel
    log_probs = log_probs[:-1]
    _outcomes = outcomes[:-1]
    importance_weights = np.exp(log_probs - gumbel_log_survival(kappa - log_probs))
    if normalize_importance_weights:
        importance_weights = importance_weights / np.sum(importance_weights)
    return float(np.sum(importance_weights * _outcomes))


class _TrieNode:
    """One node of the persistent search trie.

    Attributes:
        parent: Parent node, or ``None`` at the root.
        index_in_parent: Action index of this node in the parent, or ``None`` at the root.
        children: Lazily created child nodes. ``None`` means never expanded; an empty list
            means a leaf.
        unsampled_log_masses: ``[A]`` current residual absolute log mass of each child, or
            ``None`` if the node has never been sampled from.
        children_advantages: ``[A]`` accumulated logit perturbations per child edge.
        children_visit_counts: ``[A]`` number of sampled leaves that traversed each edge.
        sbs_child_state_cache: ``[A]`` cached ``(client_state, is_leaf)`` pairs.
        ancestors: Parent chain, used by the theory update.
        advantage_touched_flag: Whether the node was already updated in the current round.
    """

    # `__slots__` because a chunk's trie holds ~10^5-10^6 nodes and the plain class pays a dict
    # per instance. `ancestors` is a cached property below, not a slot.
    __slots__ = (
        "parent",
        "index_in_parent",
        "children",
        "unsampled_log_masses",
        "children_advantages",
        "children_visit_counts",
        "sbs_child_state_cache",
        "advantage_touched_flag",
        "live_child_edges",
        "_ancestors_cache",
    )

    def __init__(self, parent: Optional["_TrieNode"], index_in_parent: Optional[int]) -> None:
        self.parent = parent
        self.index_in_parent = index_in_parent
        self.children = None
        self.unsampled_log_masses = None
        self.children_advantages = None
        self.children_visit_counts = None
        self.sbs_child_state_cache: Optional[List[Tuple[sbs.State, bool]]] = None
        self.advantage_touched_flag = False
        # Number of outgoing edges whose residual mass is not `-inf`, maintained by
        # `mark_mass_sampled`; `None` until the node is first expanded.
        self.live_child_edges: Optional[int] = None
        self._ancestors_cache: Optional[List["_TrieNode"]] = None

    @property
    def ancestors(self) -> List["_TrieNode"]:
        """Parent chain, nearest first. Built lazily: only the dormant theory update reads it."""
        if self._ancestors_cache is None:
            self._ancestors_cache = (
                [] if self.parent is None else [self.parent] + self.parent.ancestors
            )
        return self._ancestors_cache

    def initial_log_mass_if_not_sampled(self) -> float:
        """Return this node's initial log mass, assuming no sample has been drawn from it."""
        return (
            self.parent.unsampled_log_masses[self.index_in_parent] if self.parent else 0.0
        )

    def mark_leaf(self) -> None:
        """Mark this node as a terminal leaf."""
        self.children = []

    def init_masses(self, child_log_probs: np.ndarray, initial_log_mass: float) -> None:
        """Install the initial residual masses of this node's outgoing edges.

        Also seeds :attr:`live_child_edges`, which makes :meth:`exhausted` O(1). A NaN mass
        counts as live, exactly as the previous ``all(np.isneginf(...))`` scan did.
        """
        masses = child_log_probs + initial_log_mass
        self.unsampled_log_masses = masses
        self.live_child_edges = int(masses.size - np.count_nonzero(np.isneginf(masses)))

    def exhausted(self) -> bool:
        """Return whether all probability mass at this node has been sampled."""
        # Distinguish [] (leaf) from None (never expanded).
        if self.children is not None and not len(self.children):
            return True
        if self.unsampled_log_masses is None:
            return False
        return self.live_child_edges == 0

    def mark_mass_sampled(self, log_mass: float) -> None:
        """Subtract ``log_mass`` from this node's edge and all ancestors' edges."""
        node = self
        while node.parent is not None:
            parent = node.parent
            edge = node.index_in_parent
            masses = parent.unsampled_log_masses
            if node.exhausted():
                new_log_mass = -np.inf  # Forbid ever sampling this edge again.
            else:
                new_log_mass = log_subtract(masses[edge], log_mass)
            # Keep `live_child_edges` exact: decrement only when a finite edge becomes -inf.
            # Comparing to -inf is equivalent to `np.isneginf` here and ~30x cheaper per edge.
            if new_log_mass == -np.inf and masses[edge] != -np.inf:
                parent.live_child_edges -= 1
            masses[edge] = new_log_mass
            node = parent

    def add_simple_advantage(self, mass: float) -> None:
        """Recursively add ``mass`` to this node's edge and all ancestors' edges."""
        if not self.parent:
            return
        # A trie first driven by WOR does not allocate the advantage arrays; allocate on first
        # use so an update-type switch on a reused searcher still works.
        if self.parent.children_advantages is None:
            self.parent.children_advantages = np.zeros(
                len(self.parent.unsampled_log_masses)
            )
            self.parent.children_visit_counts = np.full(
                len(self.parent.unsampled_log_masses), 1e-8
            )
        self.parent.children_advantages[self.index_in_parent] += mass
        self.parent.children_visit_counts[self.index_in_parent] += 1
        self.parent.add_simple_advantage(mass)

    def obtain_locally_estimated_advantage(
        self,
        root_expectation: float,
        advantage_constant: float,
        beam_leaves: List[sbs.BeamLeaf],
        leaf_eval_fn: Callable,
    ) -> None:
        """Apply the theory update's local expectation difference once per node per round."""
        if not self.parent or self.advantage_touched_flag:
            return
        # Leaf nodes already have their mass removed; only internal nodes are updated.
        if len(self.children):
            self.advantage_touched_flag = True
            shared_parent_leaves = [
                leaf for leaf in beam_leaves if self.parent in leaf.state[0].ancestors
            ]
            if len(shared_parent_leaves) >= 2:
                expectation_parent = gumbel_without_replacement_expectation(
                    shared_parent_leaves, leaf_eval_fn
                )
                shared_node_leaves = [
                    leaf for leaf in shared_parent_leaves if self in leaf.state[0].ancestors
                ]
                expectation_node = gumbel_without_replacement_expectation(
                    shared_node_leaves, leaf_eval_fn
                )
                if self.parent.children_advantages is None:
                    self.parent.children_advantages = np.zeros(
                        len(self.parent.unsampled_log_masses)
                    )
                    self.parent.children_visit_counts = np.full(
                        len(self.parent.unsampled_log_masses), 1e-8
                    )
                self.parent.children_advantages[self.index_in_parent] += (
                    advantage_constant * (expectation_node - expectation_parent)
                )

        self.parent.obtain_locally_estimated_advantage(
            root_expectation, advantage_constant, beam_leaves, leaf_eval_fn
        )


class IncrementalSBS:
    """Run SBS rounds against a persistent trie for a batch of initial states.

    Construct one instance per batch of problem instances; the trie, residual masses and
    advantages persist across calls to :meth:`perform_incremental_sbs`.

    Args:
        initial_states: One root state per instance.
        child_log_probability_fn: Client callback; see :func:`src.sil.sbs.stochastic_beam_search`.
        child_transition_fn: Client callback; see
            :func:`src.sil.sbs.stochastic_beam_search`.
        leaf_evaluation_fn: Maps a finished client state to an outcome that should be
            **maximised** (for routing, the negated tour length). Required for the
            ``gumbeldore`` and ``theory_gumbeldore`` updates.
        memory_aggressive: Drop cached client states after they are passed, trading
            re-transitioning for memory. Not used by CVRP.
    """

    def __init__(
        self,
        initial_states: List[sbs.State],
        child_log_probability_fn: Callable[[List[sbs.State]], List[np.ndarray]],
        child_transition_fn: Callable[
            [List[Tuple[sbs.State, int]]], List[Tuple[sbs.State, bool]]
        ],
        leaf_evaluation_fn: Optional[Callable[[sbs.State], float]] = None,
        memory_aggressive: bool = False,
    ):
        self.root_nodes = [
            (_TrieNode(None, None), state) for state in initial_states
        ]
        self.root_nodes_exhausted = [False] * len(self.root_nodes)
        self.leaf_evaluation_fn = leaf_evaluation_fn
        self.child_log_probability_fn = child_log_probability_fn
        self.child_transition_fn = child_transition_fn
        self.memory_aggressive = memory_aggressive

    def perform_incremental_sbs(
        self,
        beam_width: int,
        num_rounds: int,
        log_prob_update_type: str = "wor",
        advantage_constant: float = 1.0,
        min_max_normalize_advantage: bool = False,
        expected_value_use_simple_mean: bool = False,
        use_pure_outcomes: bool = False,
        normalize_advantage_by_visit_count: bool = False,
        perform_first_round_deterministic: bool = False,
        min_nucleus_top_p: float = 1.0,
        rng: np.random.Generator | None = None,
        max_steps: int | None = None,
    ) -> List[List[sbs.BeamLeaf]]:
        """Run ``num_rounds`` rounds of SBS and return all sampled leaves.

        Args:
            beam_width: Beam width for one round.
            num_rounds: Number of rounds. Mass is subtracted after every round, which is what
                makes the concatenated leaves a without-replacement sample.
            log_prob_update_type: ``"wor"``, ``"gumbeldore"`` or ``"theory_gumbeldore"``.
            advantage_constant: ``sigma`` for the advantage updates.
            min_max_normalize_advantage: Normalise advantages to ``[-1, 1]``.
            expected_value_use_simple_mean: Use the arithmetic mean instead of the
                importance-weighted expectation.
            use_pure_outcomes: Do not centre outcomes on the expectation.
            normalize_advantage_by_visit_count: Divide accumulated advantages by visit counts.
            perform_first_round_deterministic: Make round 0 a deterministic beam search.
            min_nucleus_top_p: If below 1, anneal top-p from this value to 1 across rounds.
            rng: Optional NumPy generator threaded into the Gumbel draws.
            max_steps: Optional safety bound forwarded to each SBS round.

        Returns:
            One list of leaves per instance, concatenating every round in round order. Leaf
            states are the client states (the trie node is stripped).
        """
        if log_prob_update_type not in _UPDATE_TYPES:
            raise ValueError(
                f"log_prob_update_type must be one of {_UPDATE_TYPES}, "
                f"got {log_prob_update_type!r}"
            )
        if log_prob_update_type != "wor" and self.leaf_evaluation_fn is None:
            raise ValueError(f"the {log_prob_update_type!r} update needs a leaf_evaluation_fn")

        leaves_batch: List[List[sbs.BeamLeaf]] = [[] for _ in range(len(self.root_nodes))]

        for round_idx in range(num_rounds):
            unexhausted_root_idcs = [
                i for i, exhausted in enumerate(self.root_nodes_exhausted) if not exhausted
            ]
            root_nodes_to_search = [self.root_nodes[i] for i in unexhausted_root_idcs]

            is_deterministic_round = bool(
                perform_first_round_deterministic and round_idx == 0
            )
            top_p_round = (
                1.0
                if num_rounds == 1
                else (1 - round_idx / (num_rounds - 1.0)) * min_nucleus_top_p
                + 1.0 * (round_idx / (num_rounds - 1.0))
            )

            round_beam_leaves_batch = sbs.stochastic_beam_search(
                child_log_probability_fn=self.wrap_child_log_probability_fn(
                    self.child_log_probability_fn,
                    normalize_advantage_by_visit_count,
                    # WOR never reads the advantage/visit arrays; the gumbeldore and theory
                    # updates do. Skipping two [A] allocations per expanded trie node is a
                    # measurable share of the WOR arm.
                    track_advantages=log_prob_update_type != "wor",
                ),
                child_transition_fn=self.wrap_child_transition_fn(
                    self.child_transition_fn, self.memory_aggressive
                ),
                root_states=root_nodes_to_search,
                beam_width=beam_width,
                deterministic=is_deterministic_round,
                top_p=1 if is_deterministic_round else top_p_round,
                rng=rng,
                max_steps=max_steps,
            )

            for j, beam_leaves in enumerate(round_beam_leaves_batch):
                batch_idx = unexhausted_root_idcs[j]
                if not is_deterministic_round and log_prob_update_type == "gumbeldore":
                    _, _, normalized_advantages = self.compute_expected_outcome(
                        sbs_leaves=beam_leaves,
                        simple_mean=expected_value_use_simple_mean,
                        min_max_normalize_advantage=min_max_normalize_advantage,
                        use_pure_outcomes=use_pure_outcomes,
                    )

                client_beam_leaves = []
                for i, beam_leaf in enumerate(beam_leaves):
                    leaf_node, client_state = beam_leaf.state

                    # All update types sample without replacement.
                    log_sampled_mass = leaf_node.initial_log_mass_if_not_sampled()
                    leaf_node.mark_mass_sampled(log_sampled_mass)

                    if not is_deterministic_round:
                        if log_prob_update_type == "gumbeldore":
                            leaf_node.add_simple_advantage(
                                advantage_constant * normalized_advantages[i]
                            )
                        elif log_prob_update_type == "theory_gumbeldore":
                            root_expectation = gumbel_without_replacement_expectation(
                                beam_leaves, self.leaf_evaluation_fn
                            )
                            leaf_node.obtain_locally_estimated_advantage(
                                root_expectation,
                                advantage_constant,
                                beam_leaves,
                                self.leaf_evaluation_fn,
                            )

                    client_beam_leaves.append(beam_leaf._replace(state=client_state))

                leaves_batch[batch_idx].extend(client_beam_leaves)
                self.root_nodes_exhausted[batch_idx] = self.root_nodes[batch_idx][
                    0
                ].exhausted()

        return leaves_batch

    def compute_expected_outcome(
        self,
        sbs_leaves: List[sbs.BeamLeaf],
        simple_mean: bool = False,
        normalize_importance_weights: bool = True,
        min_max_normalize_advantage: bool = False,
        use_pure_outcomes: bool = False,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute the expected outcome and the per-leaf advantages.

        Args:
            sbs_leaves: Leaves from one SBS round.
            simple_mean: Use the arithmetic mean instead of the importance-weighted estimator.
            normalize_importance_weights: Normalise the importance weights by their sum.
            min_max_normalize_advantage: Rescale advantages so the best/worst are ``1``/``-1``.
            use_pure_outcomes: Return raw outcomes instead of advantages.

        Returns:
            ``(expected_outcome, outcomes, advantages)``.
        """
        log_probs = np.array([leaf.log_probability for leaf in sbs_leaves])
        outcomes = np.array([self.leaf_evaluation_fn(leaf.state[1]) for leaf in sbs_leaves])
        min_outcome = np.min(outcomes)
        max_outcome = np.max(outcomes)

        if simple_mean:
            expected_outcome = float(np.mean(outcomes))
        else:
            kappa = sbs_leaves[-1].gumbel
            log_probs = log_probs[:-1]
            _outcomes = outcomes[:-1]
            if len(log_probs) == 0:
                # One leaf only: there is nothing to weight. The reference returns 0 here,
                # and the single leaf then carries its full outcome as advantage.
                expected_outcome = 0.0
            else:
                importance_weights = np.exp(
                    log_probs - gumbel_log_survival(kappa - log_probs)
                )
                if normalize_importance_weights:
                    importance_weights = importance_weights / np.sum(importance_weights)
                expected_outcome = float(np.sum(importance_weights * _outcomes))

        if not use_pure_outcomes:
            advantages = outcomes - expected_outcome
            if min_max_normalize_advantage:
                advantages = np.where(
                    advantages > 0,
                    advantages / (np.abs(max_outcome - expected_outcome) + 1e-7),
                    advantages / (np.abs(min_outcome - expected_outcome) + 1e-7),
                )
        else:
            advantages = outcomes

        return expected_outcome, outcomes, advantages

    @staticmethod
    def wrap_child_log_probability_fn(
        child_log_probability_fn,
        normalize_advantage_by_visit_count: bool,
        track_advantages: bool = True,
    ) -> Callable[[List[Tuple[_TrieNode, sbs.State]]], List[np.ndarray]]:
        """Wrap a client callback so it reads and writes the trie's residual masses.

        ``track_advantages=False`` skips the per-node advantage/visit arrays. Only the WOR update
        is valid with it, because the gumbeldore and theory updates accumulate into those arrays.
        """

        def wrapper_child_log_probability_fn(
            node_state_tuples: List[Tuple[_TrieNode, sbs.State]],
        ) -> List[np.ndarray]:
            results = [None] * len(node_state_tuples)
            unexpanded_client_states = []
            unexpanded_indices = []
            cached_indices = []
            cached_masses = []
            cached_advantages = []

            for i, (node, client_state) in enumerate(node_state_tuples):
                if node.unsampled_log_masses is None:
                    unexpanded_client_states.append(client_state)
                    unexpanded_indices.append(i)
                else:
                    cached_indices.append(i)
                    cached_masses.append(node.unsampled_log_masses)
                    if track_advantages:
                        advantages = node.children_advantages
                        if normalize_advantage_by_visit_count:
                            advantages = (
                                node.children_advantages / node.children_visit_counts
                            )
                        cached_advantages.append(advantages)

            if cached_indices:
                # Normalise every already-expanded node in one block. Cast the float32 masses to
                # float64 first: the scalar expression added a float64 advantage array, so the
                # normalisation ran in float64, and WOR's result must not drift.
                log_unnormalized = np.stack(cached_masses).astype(np.float64)
                if cached_advantages:
                    log_unnormalized += np.stack(cached_advantages)
                max_log = log_unnormalized.max(axis=1)
                exhausted = ~np.isfinite(max_log)
                # Every child of such a row is exhausted or non-finite; returning all -inf lets
                # the search drop the node instead of producing NaNs from 0/0.
                safe_max = np.where(exhausted, 0.0, max_log)
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    unnormalized = np.exp(log_unnormalized - safe_max[:, None])
                    normalised = np.log(
                        unnormalized / np.sum(unnormalized, axis=1, keepdims=True)
                    )
                normalised[exhausted] = -np.inf
                for row, index in enumerate(cached_indices):
                    results[index] = normalised[row]

            if unexpanded_client_states:
                client_fn_results = child_log_probability_fn(unexpanded_client_states)
                for i, log_probs in zip(unexpanded_indices, client_fn_results):
                    results[i] = log_probs
                    node = node_state_tuples[i][0]
                    node.init_masses(
                        log_probs, node.initial_log_mass_if_not_sampled()
                    )
                    if track_advantages:
                        node.children_advantages = np.zeros(
                            len(node.unsampled_log_masses)
                        )
                        node.children_visit_counts = np.full(
                            len(node.unsampled_log_masses), 1e-8
                        )

            return typing.cast(List[np.ndarray], results)

        return wrapper_child_log_probability_fn

    @staticmethod
    def wrap_child_transition_fn(child_transition_fn, memory_aggressive: bool) -> Callable:
        """Wrap a client callback so it creates and caches trie children."""

        def wrapper_child_transition_fn(
            node_state_action_index_pairs: List[Tuple[Tuple[_TrieNode, sbs.State], int]],
        ) -> List[Tuple[Tuple[_TrieNode, sbs.State], bool]]:
            results = [None] * len(node_state_action_index_pairs)
            unexpanded_client_state_index_pairs = []
            unexpanded_indices = []

            for i, ((node, client_state), child_index) in enumerate(
                node_state_action_index_pairs
            ):
                if node.children is None:
                    num_children = len(node.unsampled_log_masses)
                    node.children = [None] * num_children
                    node.sbs_child_state_cache = [None] * num_children

                if (
                    node.children[child_index] is None
                    or node.sbs_child_state_cache[child_index] is None
                ):
                    unexpanded_client_state_index_pairs.append((client_state, child_index))
                    unexpanded_indices.append(i)
                else:
                    child_client_state, child_is_leaf = node.sbs_child_state_cache[
                        child_index
                    ]
                    node.children[child_index].advantage_touched_flag = False
                    results[i] = (
                        (node.children[child_index], child_client_state),
                        child_is_leaf,
                    )

            if unexpanded_client_state_index_pairs:
                client_fn_results = child_transition_fn(
                    unexpanded_client_state_index_pairs
                )
                for i, (child_client_state, child_is_leaf) in zip(
                    unexpanded_indices, client_fn_results
                ):
                    (node, _), child_index = node_state_action_index_pairs[i]
                    child_node = node.children[child_index]
                    if node.children[child_index] is None:
                        child_node = _TrieNode(parent=node, index_in_parent=child_index)
                        if child_is_leaf:
                            child_node.mark_leaf()
                        node.children[child_index] = child_node

                    node.sbs_child_state_cache[child_index] = (
                        child_client_state,
                        child_is_leaf,
                    )
                    results[i] = ((child_node, child_client_state), child_is_leaf)

            if memory_aggressive:
                # Drop the parent's cached client state after transitioning through it.
                for (node, _), _ in node_state_action_index_pairs:
                    if node.parent is not None:
                        node.parent.sbs_child_state_cache[node.index_in_parent] = None

            return typing.cast(
                List[Tuple[Tuple[_TrieNode, sbs.State], bool]], results
            )

        return wrapper_child_transition_fn
