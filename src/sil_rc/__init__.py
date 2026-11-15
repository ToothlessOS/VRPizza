"""Self-improvement learning by local reconstruction (SIL-RC).

A second SIL pipeline, distinct from the WOR-SBS port in :mod:`src.sil`: labels
are improved by LEHD-style random reconstruction with the policy itself, and the
policy is trained by teacher forcing on random sub-paths of those labels. See
``docs/SIL_RC.md`` and ``prompts/another_SIL_method.md``.
"""

from src.sil_rc.dataset import SILPartialDataset
from src.sil_rc.insertion import random_insertion_routes, validate_routes
from src.sil_rc.labels import (
    LABEL_GEN_BLOCK,
    LABEL_STORE_VERSION,
    LabelStore,
    generate_instances,
    initial_labels,
    recompute_costs,
)
from src.sil_rc.reconstruction import improve_labels

__all__ = [
    "LABEL_GEN_BLOCK",
    "LABEL_STORE_VERSION",
    "LabelStore",
    "SILPartialDataset",
    "generate_instances",
    "improve_labels",
    "initial_labels",
    "random_insertion_routes",
    "recompute_costs",
    "validate_routes",
]
