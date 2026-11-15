"""Pseudo-label generation, storage and persistence for local-reconstruction SIL.

The reference SIL/SIT trainer keeps one improved solution per instance in a flat
``node_flag`` tensor, generated once and improved in place across cycles
(``best_student_list`` in ``Trainer.py``), and saves it to ``repaired_data*.pt``
for resume. This module is the equivalent store for the POMO port.

Layout per instance (``M`` instances, ``N`` customers):

* ``depot [M,2]`` float32, ``locs [M,N,2]`` float32;
* ``demand [M,N]`` float32 in the file's raw units, ``capacity [M]`` float32;
* ``tour [M,N]`` int16 (customer ids ``1..N`` in visit order);
* ``flags [M,N]`` uint8 (``1`` at each route's first customer);
* ``cost [M]`` float32, the exact full-tour length replayed from ``tour``/``flags``.

The dtypes keep a 1M-instance store around 1.2 GB. The tour/flag representation
is the same one ``src.lehd.subpath`` uses, so ``sample_window``,
``split_routes``, ``flatten_routes`` and ``cost_from_flags_torch`` apply without
translation.

Instance generation is keyed by ``(LABEL_SEED, block_index)`` and a fixed block
size, so the set is independent of the repair batch, DataLoader workers and
whether the smoke path or the full run is used. The block size is a module
constant and therefore part of the reproduction recipe: the rl4co generator
consumes its RNG by batch shape.
"""

from __future__ import annotations

import json
import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from tensordict.tensordict import TensorDict

from src.lehd.subpath import flatten_routes, split_routes

__all__ = [
    "LABEL_GEN_BLOCK",
    "LABEL_STORE_VERSION",
    "LabelStore",
    "generate_instances",
    "initial_labels",
    "recompute_costs",
]

#: Instances drawn per generated block. Part of the reproduction recipe.
LABEL_GEN_BLOCK = 10_000

#: Bumped when the on-disk layout or metadata changes.
LABEL_STORE_VERSION = 1


def _block_seed(seed: int, block: int) -> int:
    """A deterministic torch seed for one generation block."""
    state = np.random.SeedSequence([int(seed), int(block)]).generate_state(
        1, dtype=np.uint32
    )
    return int(state[0])


@dataclass
class LabelStore:
    """The M-instance pseudo-label dataset plus its metadata."""

    depot: np.ndarray
    locs: np.ndarray
    demand: np.ndarray
    capacity: np.ndarray
    tour: np.ndarray
    flags: np.ndarray
    cost: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.depot = np.asarray(self.depot, dtype=np.float32)
        self.locs = np.asarray(self.locs, dtype=np.float32)
        self.demand = np.asarray(self.demand, dtype=np.float32)
        self.capacity = np.asarray(self.capacity, dtype=np.float32)
        self.tour = np.asarray(self.tour, dtype=np.int16)
        self.flags = np.asarray(self.flags, dtype=np.uint8)
        self.cost = np.asarray(self.cost, dtype=np.float32)

        count = self.depot.shape[0]
        expected = (count, self.locs.shape[1] if self.locs.ndim == 3 else -1)
        if self.depot.shape != (count, 2):
            raise ValueError(f"depot must be [M, 2], got {self.depot.shape}")
        if self.locs.ndim != 3 or self.locs.shape[0] != count or self.locs.shape[2] != 2:
            raise ValueError(f"locs must be [M, N, 2], got {self.locs.shape}")
        if self.demand.shape != (count, expected[1]):
            raise ValueError(f"demand must be [M, N], got {self.demand.shape}")
        if self.capacity.shape != (count,):
            raise ValueError(f"capacity must be [M], got {self.capacity.shape}")
        if self.tour.shape != (count, expected[1]) or self.flags.shape != self.tour.shape:
            raise ValueError(
                f"tour/flags must be [M, N], got {self.tour.shape}/{self.flags.shape}"
            )
        if self.cost.shape != (count,):
            raise ValueError(f"cost must be [M], got {self.cost.shape}")
        if self.num_loc >= np.iinfo(np.int16).max:
            raise ValueError(
                f"num_loc={self.num_loc} exceeds the int16 tour storage; widen `tour`"
            )

    # -- introspection -----------------------------------------------------------

    def __len__(self) -> int:
        """Number of labelled instances."""
        return int(self.depot.shape[0])

    @property
    def num_loc(self) -> int:
        """Customers per instance."""
        return int(self.locs.shape[1])

    @property
    def mean_cost(self) -> float:
        """Mean label length; 0.0 for an unlabelled store."""
        return float(self.cost.mean()) if self.cost.size else 0.0

    @property
    def best_cost(self) -> float:
        """Shortest label length; 0.0 for an unlabelled store."""
        return float(self.cost.min()) if self.cost.size else 0.0

    # -- access ------------------------------------------------------------------

    def routes_for(self, index: int | Sequence[int]):
        """Return one route list, or a list of them, in the ``split_routes`` form."""
        if np.isscalar(index):
            row = int(index)
            return split_routes(
                self.tour[row].astype(np.int64), self.flags[row].astype(np.int64)
            )
        return [self.routes_for(int(row)) for row in index]

    def batch(self, indices: Sequence[int], device=None) -> TensorDict:
        """A complete-instance batch for ``run_rrc_from_routes``."""
        index = np.asarray(indices, dtype=np.int64)
        return TensorDict(
            {
                "depot": torch.as_tensor(self.depot[index], dtype=torch.float32),
                "locs": torch.as_tensor(self.locs[index], dtype=torch.float32),
                "demand": torch.as_tensor(self.demand[index], dtype=torch.float32),
                "capacity": torch.as_tensor(self.capacity[index], dtype=torch.float32),
                "cost": torch.as_tensor(self.cost[index], dtype=torch.float32),
            },
            batch_size=[int(index.size)],
            device=device,
        )

    def update(
        self,
        indices: Sequence[int],
        routes: Sequence[Sequence[np.ndarray]],
        costs: Sequence[float],
    ) -> None:
        """Write improved routes and their exact costs back into the store."""
        index = np.asarray(indices, dtype=np.int64)
        if len(routes) != index.size or len(costs) != index.size:
            raise ValueError(
                f"got {len(routes)} route lists and {len(costs)} costs for {index.size} rows"
            )
        n = self.num_loc
        for row, label_index in enumerate(index.tolist()):
            customers, flags = flatten_routes(routes[row])
            if customers.size != n or not np.array_equal(
                np.sort(customers), np.arange(1, n + 1, dtype=np.int64)
            ):
                raise ValueError(
                    f"row {label_index} does not cover every customer exactly once"
                )
            self.tour[label_index] = customers.astype(np.int16)
            self.flags[label_index] = flags.astype(np.uint8)
            self.cost[label_index] = np.float32(costs[row])

    # -- persistence -------------------------------------------------------------

    def save(self, path: str | os.PathLike) -> None:
        """Write the arrays and metadata atomically to one ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        meta = dict(self.meta)
        meta["version"] = LABEL_STORE_VERSION
        meta["count"] = len(self)
        meta["num_loc"] = self.num_loc
        with open(tmp, "wb") as handle:
            np.savez(
                handle,
                depot=self.depot,
                locs=self.locs,
                demand=self.demand,
                capacity=self.capacity,
                tour=self.tour,
                flags=self.flags,
                cost=self.cost,
                meta=np.asarray(json.dumps(meta, sort_keys=True)),
            )
        os.replace(tmp, path)

    @classmethod
    def load(
        cls, path: str | os.PathLike, expected: dict | None = None
    ) -> "LabelStore":
        """Load a store, refusing a layout or configuration that does not match."""
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            store = cls(
                depot=data["depot"],
                locs=data["locs"],
                demand=data["demand"],
                capacity=data["capacity"],
                tour=data["tour"],
                flags=data["flags"],
                cost=data["cost"],
                meta=meta,
            )
        if int(meta.get("version", -1)) != LABEL_STORE_VERSION:
            raise ValueError(
                f"{path}: label-store version {meta.get('version')} != "
                f"{LABEL_STORE_VERSION}"
            )
        # Sidecars written before the backbone field existed are POMO-backbone
        # stores; defaulting here keeps an in-progress POMO run resumable.
        meta.setdefault("backbone", "pomo")
        meta.setdefault("policy", "pomo-am-e6-h8-ff512")
        if expected is not None:
            mismatched = {
                key: (expected[key], meta.get(key))
                for key in expected
                if meta.get(key) != expected[key]
            }
            if mismatched:
                raise ValueError(f"{path}: label-store config mismatch {mismatched}")
        return store


def generate_instances(
    generator,
    count: int,
    seed: int,
    block_size: int = LABEL_GEN_BLOCK,
) -> LabelStore:
    """Generate a fixed CVRP instance set and return an unlabelled store.

    Args:
        generator: An rl4co ``CVRPGenerator`` (normally ``env.generator``).
        count: Number of instances.
        seed: Base seed; block ``b`` is drawn under ``_block_seed(seed, b)``.
        block_size: Instances per generated block. Part of the recipe.

    Returns:
        A :class:`LabelStore` with zeroed tours/flags/costs.
    """
    count = int(count)
    block_size = int(block_size)
    if count <= 0 or block_size <= 0:
        raise ValueError("count and block_size must be positive")

    num_loc = int(getattr(generator, "num_loc"))
    depot = np.empty((count, 2), dtype=np.float32)
    locs = np.empty((count, num_loc, 2), dtype=np.float32)
    demand = np.empty((count, num_loc), dtype=np.float32)
    capacity = np.empty((count,), dtype=np.float32)

    for block, start in enumerate(range(0, count, block_size)):
        stop = min(start + block_size, count)
        size = stop - start
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_block_seed(seed, block))
            td = generator(size)

        cap = td["capacity"].reshape(-1).to(torch.float32)
        # rl4co normalises demand by capacity; the reference stores integer demands.
        raw = (td["demand"].to(torch.float32) * cap[:, None]).round()
        if not bool(torch.isfinite(raw).all()):
            raise ValueError("the generator produced non-finite demands")
        if bool((raw > cap[:, None] + 1e-6).any()) or bool((raw < 0).any()):
            raise ValueError("the generator produced a demand outside [0, capacity]")

        depot[start:stop] = td["depot"].detach().cpu().numpy().astype(np.float32)
        locs[start:stop] = td["locs"].detach().cpu().numpy().astype(np.float32)
        demand[start:stop] = raw.detach().cpu().numpy().astype(np.float32)
        capacity[start:stop] = cap.detach().cpu().numpy().astype(np.float32)

    meta = {
        "label_seed": int(seed),
        "block_size": block_size,
        "generator_num_loc": num_loc,
        "capacity": float(capacity.mean()) if count else 0.0,
    }
    return LabelStore(
        depot=depot,
        locs=locs,
        demand=demand,
        capacity=capacity,
        tour=np.zeros((count, num_loc), dtype=np.int16),
        flags=np.zeros((count, num_loc), dtype=np.uint8),
        cost=np.zeros((count,), dtype=np.float32),
        meta=meta,
    )


def recompute_costs(
    store: LabelStore,
    indices: Sequence[int] | None = None,
    batch_size: int = 1024,
    device=None,
) -> None:
    """Replay ``tour``/``flags`` and refresh ``cost`` exactly.

    Uses the same float64 vectorised cost as the RRC acceptance check, so the stored
    cost and the next cycle's ``initial_cost`` agree.
    """
    from src.rrc.runner import cost_from_flags_torch

    index = (
        np.arange(len(store), dtype=np.int64)
        if indices is None
        else np.asarray(indices, dtype=np.int64)
    )
    for start in range(0, index.size, batch_size):
        chunk = index[start : start + batch_size]
        customers = torch.as_tensor(store.tour[chunk].astype(np.int64), device=device)
        flags = torch.as_tensor(store.flags[chunk].astype(np.int64), device=device).bool()
        depot = torch.as_tensor(store.depot[chunk], dtype=torch.float32, device=device)
        locs = torch.as_tensor(store.locs[chunk], dtype=torch.float32, device=device)
        costs = cost_from_flags_torch(customers, flags, depot, locs)
        store.cost[chunk] = costs.detach().cpu().numpy().astype(np.float32)


def initial_labels(
    store: LabelStore,
    model,
    env,
    init: str = "greedy",
    seed: int = 0,
    batch_size: int = 1024,
    order: str = "angle",
    device=None,
) -> dict:
    """Fill an unlabelled store with random-insertion or model-greedy labels.

    ``init="random_insertion"`` is the reference's from-scratch bootstrap: the
    deterministic polar-angle cheapest insertion, with no model call. ``init="greedy"``
    is the reference's warm-up path: one greedy rollout per instance with the current
    policy. Costs are always recomputed exactly from the resulting tours.

    Args:
        store: The unlabelled instance store, updated in place.
        model: A POMO-family Lightning module (used by ``init="greedy"``).
        env: A CVRP environment with ``make_input_td`` and ``reset``.
        init: ``"random_insertion"`` or ``"greedy"``.
        seed: Base seed; the random insertion order (when ``order="random"``) and
            nothing else are drawn from it.
        batch_size: Instances per forward pass for the greedy path.
        order: Insertion order for the random-insertion path.
        device: Torch device for the greedy path; defaults to the model's.

    Returns:
        A small diagnostics dict (``init``, ``label_mean_cost``, ``label_best_cost``).

    Raises:
        ValueError: If ``init`` is not one of the two supported values.
    """
    from src.lehd.subpath import actions_to_routes
    from src.rrc.runner import greedy_rollout

    if init not in ("random_insertion", "greedy"):
        raise ValueError(
            f"init must be 'random_insertion' or 'greedy', got {init!r}"
        )

    if init == "random_insertion":
        from src.sil_rc.insertion import random_insertion_routes

        for row in range(len(store)):
            rng = np.random.default_rng([int(seed), int(row)])
            routes = random_insertion_routes(
                store.depot[row],
                store.locs[row],
                store.demand[row],
                float(store.capacity[row]),
                order=order,
                rng=rng,
            )
            # The insertion helper indexes customers from 0; the solution
            # representation is 1-based (customer ids 1..N, 0 is the depot).
            customers, flags = flatten_routes([route + 1 for route in routes])
            store.tour[row] = customers.astype(np.int16)
            store.flags[row] = flags.astype(np.uint8)
        recompute_costs(store, device=device)
    else:
        if device is None:
            device = model.device
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(store), batch_size):
                    chunk = np.arange(start, min(start + batch_size, len(store)))
                    batch = store.batch(chunk, device=device)
                    _, actions = greedy_rollout(model, env, batch)
                    routes = [
                        actions_to_routes(
                            actions[row].detach().cpu().numpy(), store.num_loc
                        )
                        for row in range(actions.shape[0])
                    ]
                    flats = [flatten_routes(route_list) for route_list in routes]
                    store.tour[chunk] = np.stack([flat[0] for flat in flats]).astype(
                        np.int16
                    )
                    store.flags[chunk] = np.stack([flat[1] for flat in flats]).astype(
                        np.uint8
                    )
                recompute_costs(store, device=device)
        finally:
            model.train(was_training)

    store.meta["init"] = init
    return {
        "init": init,
        "label_mean_cost": store.mean_cost,
        "label_best_cost": store.best_cost,
    }
