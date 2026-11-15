"""Turn a POMO policy into a SIL pseudo-expert dataset with beam search or WOR-SBS.

For each freshly generated CVRP instance, the policy explores the tree with one of two sample
generators and the cheapest complete trajectory seen is kept as that instance's pseudo-expert.
The instances come from the same ``CVRPGenerator`` the RL runs train on, so SIL sees the same
distribution as the warm-up phase.

* ``search_type="wor"``: ``IncrementalSBS`` with residual-mass subtraction, sampled without
  replacement across rounds. This is the stage-1 Gumbeldore sampler.
* ``search_type="beam_search"``: deterministic SBS, i.e. ordinary beam search over the same
  tree and adapter. One round.

The advantage-based trie update is ported in :mod:`src.sil.incremental_sbs` but deliberately
not reachable from here: ``search_type`` is validated to the two sampling arms under
comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from src.lehd.subpath import actions_to_routes, cost_from_flags, flatten_routes
from src.sil.incremental_sbs import IncrementalSBS
from src.sil.sbs import stochastic_beam_search
from src.sil.trajectory import SILSearchAdapter

__all__ = ["SILRecord", "SILSearchConfig", "generate_best_solutions"]

_SEARCH_TYPES = ("wor", "beam_search")


@dataclass(frozen=True)
class SILSearchConfig:
    """Sampler configuration for one SIL generation round.

    Args:
        search_type: ``"wor"`` or ``"beam_search"``. The advantage-update variants are not
            exposed until stage 2.
        beam_width: Beam width per round. A round keeps up to this many leaves per instance.
        num_rounds: WOR rounds. Ignored by ``beam_search``.
        log_prob_update_type: Passed to :class:`~src.sil.incremental_sbs.IncrementalSBS`.
            Dormant; keep ``"wor"``.
        advantage_constant: ``sigma`` for the dormant advantage updates.
        min_max_normalize_advantage: Dormant.
        expected_value_use_simple_mean: Dormant.
        use_pure_outcomes: Dormant.
        normalize_advantage_by_visit_count: Dormant.
        top_p: Nucleus truncation. Disabled at 1.0.
        min_nucleus_top_p: Starting top-p for the linear anneal across WOR rounds.
        perform_first_round_deterministic: Make WOR round 0 a plain beam search.
        chunk_size: Instances per encoded chunk. The encoder cache is per instance and is
            broadcast across the level's beam, so this no longer multiplies the decoder's cache
            footprint. It bounds the residual-memory cost of the WOR trie (roughly
            ``chunk_size * beam_width * depth`` nodes), which is why 32 is still the practical
            value; larger chunks measured no faster.
        temperature: Decoding temperature.
        tanh_clipping: Logit clipping. ``None`` uses the policy's own value.
        seed: Seed for instance generation and the Gumbel draws.
    """

    search_type: str = "wor"
    beam_width: int = 32
    num_rounds: int = 4

    log_prob_update_type: str = "wor"
    advantage_constant: float = 3.0
    min_max_normalize_advantage: bool = False
    expected_value_use_simple_mean: bool = False
    use_pure_outcomes: bool = False
    normalize_advantage_by_visit_count: bool = False

    top_p: float = 1.0
    min_nucleus_top_p: float = 1.0
    perform_first_round_deterministic: bool = False

    chunk_size: int = 32
    temperature: float = 1.0
    tanh_clipping: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.search_type not in _SEARCH_TYPES:
            raise ValueError(
                f"search_type must be one of {_SEARCH_TYPES}, got {self.search_type!r}"
            )
        if self.log_prob_update_type != "wor":
            raise ValueError(
                "the advantage-based trie update is ported but not part of the stage-1 "
                f"experiment; log_prob_update_type must be 'wor', got "
                f"{self.log_prob_update_type!r}"
            )
        if self.beam_width <= 0:
            raise ValueError(f"beam_width must be positive, got {self.beam_width}")
        if self.search_type == "wor" and self.num_rounds <= 0:
            raise ValueError(f"num_rounds must be positive, got {self.num_rounds}")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")


@dataclass
class SILRecord:
    """One generated instance together with its best sampled solution.

    Args:
        depot: ``[2]`` depot coordinates.
        locs: ``[N, 2]`` customer coordinates, depot excluded.
        demand: ``[N]`` demands **normalised** by the vehicle capacity, as the rl4co generator
            emits them. :attr:`capacity` is therefore ``1.0``.
        cost: Tour length of the stored solution, computed from its coordinates.
        tour: ``[N]`` customer ids in visit order.
        flags: ``[N]`` route-start flags indexed by tour position.
        num_routes: Number of routes in the solution.
    """

    depot: np.ndarray
    locs: np.ndarray
    demand: np.ndarray
    cost: float
    tour: np.ndarray
    flags: np.ndarray
    num_routes: int
    capacity: float = 1.0


@torch.no_grad()
def generate_best_solutions(
    policy,
    env,
    num_instances: int,
    cfg: SILSearchConfig,
    device: torch.device | str = "cuda",
) -> tuple[List[SILRecord], dict]:
    """Generate ``num_instances`` CVRP instances and keep one best solution per instance.

    Args:
        policy: The policy to search with, in eval mode.
        env: A CVRP environment backed by the generator.
        num_instances: Instances to generate. The search cost scales linearly with this,
            ``beam_width`` and ``num_rounds``.
        cfg: Sampler configuration.
        device: Device for the decoder and the environment.

    Returns:
        ``(records, metrics)``. ``metrics`` carries the mean and best generated tour length,
        the instance count and the wall-clock search time.

    Raises:
        RuntimeError: If an instance yields no sampled leaf, which means the search tree was
            malformed rather than merely unsolved.
    """
    if num_instances <= 0:
        raise ValueError(f"num_instances must be positive, got {num_instances}")

    start = time.perf_counter()
    n_loc = int(env.generator.num_loc)

    # The generator consumes the global torch RNG; fork it so a generation round does not
    # perturb the training stream and is reproducible from `seed` alone.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(cfg.seed))
        generated = env.generator(batch_size=[num_instances])

    # `env.reset` mutates and widens its input, so the record fields are copied first.
    depot_all = generated["depot"].float().cpu().numpy()
    locs_all = generated["locs"].float().cpu().numpy()
    demand_all = generated["demand"].float().cpu().numpy()

    records: List[SILRecord] = []
    max_steps = 4 * (n_loc + 1)

    for chunk_start in range(0, num_instances, cfg.chunk_size):
        chunk_stop = min(chunk_start + cfg.chunk_size, num_instances)
        chunk = generated[chunk_start:chunk_stop].to(device).clone()
        root_td = env.reset(chunk)

        adapter = SILSearchAdapter(
            policy,
            env,
            root_td,
            device=device,
            temperature=cfg.temperature,
            tanh_clipping=cfg.tanh_clipping,
        )
        roots = adapter.root_states(root_td)
        rng = np.random.default_rng([int(cfg.seed), chunk_start])

        if cfg.search_type == "wor":
            searcher = IncrementalSBS(
                roots,
                adapter.child_log_probability_fn,
                adapter.child_transition_fn,
                leaf_evaluation_fn=lambda state: state.to_max_evaluation(),
            )
            leaves_batch = searcher.perform_incremental_sbs(
                beam_width=cfg.beam_width,
                num_rounds=cfg.num_rounds,
                log_prob_update_type=cfg.log_prob_update_type,
                advantage_constant=cfg.advantage_constant,
                min_max_normalize_advantage=cfg.min_max_normalize_advantage,
                expected_value_use_simple_mean=cfg.expected_value_use_simple_mean,
                use_pure_outcomes=cfg.use_pure_outcomes,
                normalize_advantage_by_visit_count=cfg.normalize_advantage_by_visit_count,
                perform_first_round_deterministic=cfg.perform_first_round_deterministic,
                min_nucleus_top_p=cfg.min_nucleus_top_p,
                rng=rng,
                max_steps=max_steps,
            )
        else:
            leaves_batch = stochastic_beam_search(
                child_log_probability_fn=adapter.child_log_probability_fn,
                child_transition_fn=adapter.child_transition_fn,
                root_states=roots,
                beam_width=cfg.beam_width,
                deterministic=True,
                top_p=1.0,
                rng=rng,
                max_steps=max_steps,
            )

        for local_idx, leaves in enumerate(leaves_batch):
            if not leaves:
                raise RuntimeError(
                    f"no leaf sampled for instance {chunk_start + local_idx}; the search "
                    "tree did not terminate in a solution"
                )
            best = min(leaves, key=lambda leaf: leaf.state.cost)
            global_idx = chunk_start + local_idx
            records.append(
                _record_from_actions(
                    global_idx,
                    best.state.actions,
                    depot_all[global_idx],
                    locs_all[global_idx],
                    demand_all[global_idx],
                    n_loc,
                )
            )

    costs = np.asarray([record.cost for record in records], dtype=np.float64)
    metrics = {
        "gen_mean_cost": float(costs.mean()),
        "gen_best_cost": float(costs.min()),
        "gen_instances": len(records),
        "search_seconds": float(time.perf_counter() - start),
    }
    return records, metrics


def _record_from_actions(
    index: int,
    actions,
    depot: np.ndarray,
    locs: np.ndarray,
    demand: np.ndarray,
    n_loc: int,
) -> SILRecord:
    """Convert one sampled action sequence into a record in the SL dataset's representation."""
    routes = actions_to_routes(np.asarray(actions, dtype=np.int64), n_loc)
    tour, flags = flatten_routes(routes)
    cost = cost_from_flags(tour, flags, depot, locs)
    return SILRecord(
        depot=np.asarray(depot, dtype=np.float32),
        locs=np.asarray(locs, dtype=np.float32),
        demand=np.asarray(demand, dtype=np.float32),
        cost=float(cost),
        tour=tour,
        flags=flags,
        num_routes=len(routes),
    )
