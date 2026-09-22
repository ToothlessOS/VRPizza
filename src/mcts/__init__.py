"""POMO-guided MCTS for CVRP inference.

An inference-time search that spends extra computation on promising partial routes instead of
treating POMO's multi-start rollouts as equally worth finishing. See ``prompts/MCTS.md`` for the
specification these modules implement and :mod:`src.mcts.search` for where the implementation
deliberately departs from it and why.
"""

from src.mcts.config import POMOMCTSConfig
from src.mcts.node import (
    DEPOT_ACTION,
    CVRPState,
    Incumbent,
    InstanceBank,
    MCTSNode,
    batch_states,
    state_from_td,
)
from src.mcts.oracle import POMOOracle
from src.mcts.search import (
    MCTSResult,
    allowed_customer_width,
    backup,
    cfg_value,
    initialize_policy,
    maybe_expand_one,
    select_child_puct,
    solve_batch_with_pomo_mcts,
    step_children,
)
from src.mcts.stats import SearchStatistics

__all__ = [
    "DEPOT_ACTION",
    "CVRPState",
    "Incumbent",
    "InstanceBank",
    "MCTSNode",
    "MCTSResult",
    "POMOMCTSConfig",
    "POMOOracle",
    "SearchStatistics",
    "allowed_customer_width",
    "backup",
    "batch_states",
    "cfg_value",
    "initialize_policy",
    "maybe_expand_one",
    "select_child_puct",
    "solve_batch_with_pomo_mcts",
    "state_from_td",
    "step_children",
]
