"""Stochastic beam search (SBS) for a batched set of search states.

A near-verbatim port of ``core/stochastic_beam_search.py`` from the Gumbeldore repository
(https://github.com/rodrigoaraujo01/gumbeldore), itself based on the UniqueRandomizer
implementation:

    https://github.com/google-research/unique-randomizer/blob/master/unique_randomizer/stochastic_beam_search.py

The algorithm is unchanged; only the import surface, the optional random generator and a
couple of defensive guards differ. The reference implementation is NumPy and operates on an
abstract ``State``; the CVRP adapter that supplies the child probabilities and transitions
lives in :mod:`src.sil.trajectory`.

The implementation is slightly generalised from the description in the SBS paper: it handles
leaves at different depths, which CVRP episodes require because route boundaries cost an
extra depot action.
"""

from __future__ import annotations

import typing
from typing import Any, Callable, List, Tuple

import numpy as np

__all__ = [
    "BeamLeaf",
    "sample_gumbels_with_maximum",
    "softmax",
    "stochastic_beam_search",
    "top_p_filtering",
]

State = Any  # One node (partial trajectory) of the search tree.

#: A sampled leaf, as returned by :func:`stochastic_beam_search`.
BeamLeaf = typing.NamedTuple(
    "BeamLeaf",
    [("state", State), ("log_probability", float), ("gumbel", float)],
)


def sample_gumbels_with_maximum(
    log_probabilities: np.ndarray,
    target_max: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample Gumbels conditioned on their maximum being ``target_max``.

    Based on https://gist.github.com/wouterkool/a3bb2aae8d6a80f985daae95252a8aa8 and
    equations (23)/(24) in Appendix B.3 of the SBS paper.

    Args:
        log_probabilities: ``[A]`` log-probabilities of the items to sample Gumbels for.
        target_max: The desired maximum sampled Gumbel, normally the parent's Gumbel.
        rng: Optional NumPy generator. Defaults to the global ``np.random`` module, matching
            the reference implementation; passing a generator makes generation reproducible
            without seeding NumPy globally.

    Returns:
        ``[A]`` sampled Gumbels with maximum ``target_max``.
    """
    rng = np.random if rng is None else rng
    gumbels = rng.gumbel(loc=log_probabilities)
    max_gumbel = np.max(gumbels)

    # Numpy may warn "divide by zero encountered in log1p": one element of
    # `gumbels - max_gumbel` is zero by construction. The math fixes itself later, and that
    # element ends up shifted to `target_max`. The warning is disabled locally.
    with np.errstate(divide="ignore"):
        v = target_max - gumbels + np.log1p(-np.exp(gumbels - max_gumbel))
        ret_gumbels = target_max - np.maximum(v, 0) - np.log1p(np.exp(-np.abs(v)))
    return ret_gumbels


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax over a one-dimensional array (the reference implementation's axis)."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def top_p_filtering(log_probs: np.ndarray, top_p: float) -> np.ndarray:
    """Nucleus-truncate ``log_probs`` in place, keeping the smallest set with mass ``top_p``."""
    sorted_log_probs_indices = np.argsort(-log_probs)  # descending
    cumulative_probs = np.cumsum(softmax(log_probs[sorted_log_probs_indices]))

    # Remove tokens with cumulative probability above the threshold.
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift right so the first token above the threshold is kept.
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_log_probs_indices[sorted_indices_to_remove]
    log_probs[indices_to_remove] = -np.inf
    return log_probs


def stochastic_beam_search(
    child_log_probability_fn: Callable[[List[State]], List[np.ndarray]],
    child_transition_fn: Callable[[List[Tuple[State, int]]], List[Tuple[State, bool]]],
    root_states: List[State],
    beam_width: int,
    deterministic: bool = False,
    top_p: float = 0.0,
    rng: np.random.Generator | None = None,
    max_steps: int | None = None,
) -> List[List[BeamLeaf]]:
    """Run SBS over a batch of root states.

    States may be anything, but must support:

    1. a consistent ordering of all children (the action axis);
    2. enumeration of the log-probability of every child;
    3. a transition to the child at a given index.

    Args:
        child_log_probability_fn: Maps a list of states to a list of ``[A]`` child
            log-probability arrays, one per state. Infeasible actions must be ``-inf``.
        child_transition_fn: Maps a list of ``(state, action_index)`` pairs to
            ``(child_state, is_leaf)`` pairs. A leaf state is a complete trajectory and is
            returned by the search rather than expanded.
        root_states: ``B`` root states; roots may not be leaves.
        beam_width: Number of leaves to keep per instance.
        deterministic: If true, use cumulative log-probability instead of Gumbels, which
            reduces the algorithm to ordinary beam search.
        top_p: Optional nucleus truncation of each node's children, applied before sampling.
        rng: Optional NumPy generator used for the Gumbel draws.
        max_steps: Optional safety bound on expansion levels. Exceeding it raises instead of
            looping forever on a malformed state.

    Returns:
        ``B`` lists of up to ``beam_width`` leaves, each sorted by decreasing Gumbel (which is
        the order one would get by sampling one-at-a-time without replacement).
    """
    k = beam_width
    if beam_width <= 0:
        return []

    batch_len = len(root_states)
    # Nodes currently on the beam, split into leaves and internal (expandable) states. Entries
    # are index-aligned within one instance.
    batch_leaf_log_probs = [[] for _ in range(batch_len)]
    batch_leaf_gumbels = [[] for _ in range(batch_len)]
    batch_leaf_states = [[] for _ in range(batch_len)]
    batch_internal_log_probs = [[0.0] for _ in range(batch_len)]
    batch_internal_gumbels = [[0.0] for _ in range(batch_len)]
    batch_internal_states = [[root_state] for root_state in root_states]

    steps = 0
    # Expand internal nodes until none are left.
    while any(len(internal_states) for internal_states in batch_internal_states):
        if max_steps is not None and steps >= max_steps:
            raise RuntimeError(
                f"stochastic beam search exceeded max_steps={max_steps}; a state is probably "
                "not terminating"
            )
        steps += 1

        # One policy call for every internal node across all instances.
        flat_child_log_probs_list = child_log_probability_fn(
            [state for internal_states in batch_internal_states for state in internal_states]
        )
        batch_child_log_probs_list = []
        for internal_states in batch_internal_states:
            batch_child_log_probs_list.append(
                flat_child_log_probs_list[: len(internal_states)]
            )
            flat_child_log_probs_list = flat_child_log_probs_list[len(internal_states) :]
        assert not flat_child_log_probs_list, "Flat child log probs list should be empty"

        for batch_idx in range(batch_len):
            if not len(batch_child_log_probs_list[batch_idx]):
                continue  # This instance is already finished.

            all_log_probs = []
            all_gumbels = []
            all_states = []  # Parent state of each flattened candidate.
            all_child_indices = []

            for node_state, node_log_prob, node_gumbel, child_log_probs in zip(
                batch_internal_states[batch_idx],
                batch_internal_log_probs[batch_idx],
                batch_internal_gumbels[batch_idx],
                batch_child_log_probs_list[batch_idx],
            ):
                if 1 > top_p > 0:
                    child_log_probs = top_p_filtering(child_log_probs.copy(), top_p)
                    with np.errstate(divide="ignore"):
                        child_log_probs = np.log(softmax(child_log_probs))
                log_probabilities = child_log_probs + node_log_prob
                # Keep only feasible actions. `isfinite` also rejects a NaN that a corrupt
                # state could produce; `!= -inf` would let it through.
                good_indices = np.where(np.isfinite(log_probabilities))[0]
                log_probabilities = log_probabilities[good_indices]
                if deterministic:
                    gumbels = log_probabilities
                else:
                    gumbels = sample_gumbels_with_maximum(
                        log_probabilities, node_gumbel, rng=rng
                    )

                all_log_probs.extend(log_probabilities)
                all_gumbels.extend(gumbels)
                all_states.extend([node_state] * len(log_probabilities))
                all_child_indices.extend(good_indices)

            # Combine the new children with the leaves already on the beam and keep the top k.
            leaf_gumbels = batch_leaf_gumbels[batch_idx]
            leaf_log_probs = batch_leaf_log_probs[batch_idx]
            leaf_states = batch_leaf_states[batch_idx]
            num_internal_candidates = len(all_gumbels)
            num_leaf_candidates = len(leaf_gumbels)
            if beam_width >= num_internal_candidates + num_leaf_candidates:
                # Everything fits: no existing leaf is dropped.
                to_expand_states = list(zip(all_states, all_child_indices))
                to_expand_log_probs = all_log_probs
                to_expand_gumbels = all_gumbels
            else:
                all_gumbels.extend(leaf_gumbels)
                top_k_indices = np.argpartition(all_gumbels, -k)[-k:]
                to_expand_states = []
                to_expand_log_probs = []
                to_expand_gumbels = []
                leaf_indices = []  # Existing leaves that persist.
                for i in top_k_indices:
                    if i >= num_internal_candidates:
                        leaf_indices.append(i - num_internal_candidates)
                    else:
                        to_expand_states.append((all_states[i], all_child_indices[i]))
                        to_expand_log_probs.append(all_log_probs[i])
                        to_expand_gumbels.append(all_gumbels[i])

                leaf_log_probs = [leaf_log_probs[i] for i in leaf_indices]
                leaf_gumbels = [leaf_gumbels[i] for i in leaf_indices]
                leaf_states = [leaf_states[i] for i in leaf_indices]

            # Expand the selected non-leaf candidates.
            internal_log_probs = []
            internal_gumbels = []
            internal_states = []
            if len(to_expand_states):
                child_states = child_transition_fn(to_expand_states)
                for log_prob, gumbel, (child_state, is_leaf) in zip(
                    to_expand_log_probs, to_expand_gumbels, child_states
                ):
                    if is_leaf:
                        leaf_log_probs.append(log_prob)
                        leaf_gumbels.append(gumbel)
                        leaf_states.append(child_state)
                    else:
                        internal_log_probs.append(log_prob)
                        internal_gumbels.append(gumbel)
                        internal_states.append(child_state)

            batch_leaf_log_probs[batch_idx] = leaf_log_probs
            batch_leaf_gumbels[batch_idx] = leaf_gumbels
            batch_leaf_states[batch_idx] = leaf_states
            batch_internal_log_probs[batch_idx] = internal_log_probs
            batch_internal_gumbels[batch_idx] = internal_gumbels
            batch_internal_states[batch_idx] = internal_states

    # Pack the leaves and sort by decreasing Gumbel, the order a sequential without-
    # replacement sampler would produce.
    results = []
    for batch_idx in range(batch_len):
        sampled_nodes = [
            BeamLeaf(state=state, log_probability=log_prob, gumbel=gumbel)
            for log_prob, gumbel, state in zip(
                batch_leaf_log_probs[batch_idx],
                batch_leaf_gumbels[batch_idx],
                batch_leaf_states[batch_idx],
            )
        ]
        results.append(sorted(sampled_nodes, key=lambda x: x.gumbel, reverse=True))

    return results
