"""Self-improvement learning (SIL) with stochastic beam search for CVRP.

The search is a port of the Gumbeldore implementation (see :mod:`src.sil.sbs` and
:mod:`src.sil.incremental_sbs`); :mod:`src.sil.trajectory` adapts it to an rl4co POMO policy
and environment, :mod:`src.sil.search` turns sampled trajectories into pseudo-expert records,
and :mod:`src.sil.dataset` serves those records as supervised trajectories.
"""

from src.sil.dataset import SILSolutionDataset
from src.sil.incremental_sbs import IncrementalSBS
from src.sil.sbs import (
    BeamLeaf,
    sample_gumbels_with_maximum,
    stochastic_beam_search,
    top_p_filtering,
)
from src.sil.search import SILRecord, SILSearchConfig, generate_best_solutions
from src.sil.trajectory import SILSearchAdapter, SILState

__all__ = [
    "BeamLeaf",
    "IncrementalSBS",
    "SILRecord",
    "SILSearchAdapter",
    "SILSearchConfig",
    "SILSolutionDataset",
    "SILState",
    "generate_best_solutions",
    "sample_gumbels_with_maximum",
    "stochastic_beam_search",
    "top_p_filtering",
]
