"""One full local-reconstruction pass over a pseudo-label store.

This is the SIL/SIT equivalent of the reference's ``repair_all_data``: every
instance's label is improved by ``iterations`` RRC rounds with the current policy,
in chunks of ``repair_batch``. The pass is completed before the caller starts the
cycle's training block, and labels are updated in place, so the training set is
the repaired one.

The RRC budget is the reference's ``budget`` (100 for the paper/code). At N=100
the reference additionally reconstructs up to ``min(floor(N/L), routes, 2)``
non-overlapping windows per round; :class:`~src.rrc.runner.run_rrc_from_routes`
does one window per round, so the same budget means ~1.5x fewer segment
reconstructions. The plan records this as a deliberate deviation; raise
``iterations`` to ~150 if exact segment parity is wanted.

Seeding: ``RRCConfig.seed = seed + cycle`` and each instance is keyed by its
**global** label index (passed as ``positions``), so changing ``repair_batch``
does not change which windows are drawn. The decoder itself is greedy.
"""

from __future__ import annotations

import numpy as np
import torch

from src.rrc import RRCConfig, RRCStats, run_rrc_from_routes
from src.sil_rc.labels import LabelStore

__all__ = ["improve_labels"]


def improve_labels(
    store: LabelStore,
    model,
    env,
    repair_batch: int = 1024,
    iterations: int = 100,
    cycle: int = 0,
    seed: int = 0,
    w_min: int = 4,
    w_max: int | None = None,
    device=None,
) -> dict:
    """Run one full repair pass over every label in ``store``.

    Args:
        store: Pseudo-label store, updated in place.
        model: A POMO-family Lightning module whose ``policy`` does the greedy
            reconstruction. The caller is responsible for loading the best-policy
            snapshot before this call.
        env: A CVRP environment with ``reset_partial``.
        repair_batch: Instances per reconstruction batch. A memory knob; the
            per-instance window draws are keyed by the global label index.
        iterations: RRC rounds per instance (the reference's ``budget``).
        cycle: Cycle index, folded into the RRC seed.
        seed: Base label seed.
        w_min: Minimum window length.
        w_max: Maximum window length; defaults to the store's ``N``.
        device: Torch device; defaults to the model's.

    Returns:
        A diagnostics dict with the reference-style counters and the post-pass
        label costs.

    Raises:
        ValueError: If the batch size or budget is not positive.
    """
    repair_batch = int(repair_batch)
    iterations = int(iterations)
    if repair_batch <= 0 or iterations <= 0:
        raise ValueError("repair_batch and iterations must be positive")
    if device is None:
        device = model.device

    config = RRCConfig(
        budgets=(iterations,),
        w_min=int(w_min),
        w_max=w_max,
        seed=int(seed) + int(cycle),
        init="greedy",  # unused by run_rrc_from_routes; kept for config validation
        num_augment=1,
        rrc_num_augment=1,
    )

    stats = RRCStats()
    improved = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), torch.autocast(
            device_type=torch.device(device).type, enabled=False
        ):
            for start in range(0, len(store), repair_batch):
                stop = min(start + repair_batch, len(store))
                indices = np.arange(start, stop, dtype=np.int64)
                routes = store.routes_for(indices)
                batch = store.batch(indices, device=device)
                result = run_rrc_from_routes(
                    model, env, batch, routes, config, positions=indices
                )
                new_costs = result.budget_costs[:, -1]
                improved += int(
                    (new_costs < result.initial_cost - 1e-9).sum()
                )
                store.update(indices, result.final_routes, new_costs)
                stats.merge(result.stats)
    finally:
        model.train(was_training)

    store.meta["cycle"] = int(cycle)
    store.meta["recon_iterations"] = iterations
    return {
        "recon_iterations": stats.iterations,
        "recon_rounds": iterations,
        "recon_accepted": stats.accepted,
        "recon_seconds": stats.seconds,
        "label_improved": improved,
        "label_mean_cost": store.mean_cost,
        "label_best_cost": store.best_cost,
        "labels": len(store),
    }
