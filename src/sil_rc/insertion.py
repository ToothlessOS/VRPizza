"""GLOP's capacity-feasible cheapest-insertion constructor, in NumPy.

The reference SIL repo calls a compiled C++ implementation
(``utils/insertion/src/cvrp.cpp``) through ``cvrp_random_insertion``. That
extension is built for Python 3.10 / NumPy 1.23 and cannot be imported in this
project's Python 3.13 / NumPy 2.x virtualenv, so the algorithm is ported here.

The constructor is deterministic despite its name: customers are inserted in
polar-angle order around the depot (the reference's default when ``order`` is
``None``), and each is placed at the cheapest position across the currently open
routes. A new route is opened only when no feasible insertion beats the
``2 * d(depot, customer) * exploration`` fallback. The ``exploration`` factor
scales that *decision* threshold only; the route length always uses the true
depot edges, exactly as the C++ does.

The port is per instance. That is fine for the step-matched dataset
(12,500 instances per cycle) and for smoke runs; at the reference's 1M labels
the random-insertion bootstrap is a one-off cost measured in hours, which is why
the strict warm-up path defaults to model-greedy labels instead.
"""

from __future__ import annotations

import numpy as np

__all__ = ["random_insertion_routes", "validate_routes"]


def _polar_order(depot: np.ndarray, locs: np.ndarray) -> np.ndarray:
    """Customers in increasing polar angle around the depot, as the reference does."""
    dx = locs[:, 0] - depot[0]
    dy = locs[:, 1] - depot[1]
    return np.argsort(np.arctan2(dy, dx)).astype(np.int64)


def _insertion_order(
    depot: np.ndarray,
    locs: np.ndarray,
    order: str | np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resolve the customer insertion order."""
    n = int(locs.shape[0])
    if order is None or order == "angle":
        return _polar_order(depot, locs)
    if isinstance(order, str):
        if order != "random":
            raise ValueError(f"order must be 'angle', 'random' or an index array, got {order!r}")
        return rng.permutation(n).astype(np.int64)
    order = np.asarray(order, dtype=np.int64)
    if order.shape != (n,) or not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError("an explicit order must be a permutation of 0..n-1")
    return order


def random_insertion_routes(
    depot: np.ndarray,
    locs: np.ndarray,
    demand: np.ndarray,
    capacity: float,
    order: str | np.ndarray | None = None,
    exploration: float = 1.0,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Build a capacity-feasible CVRP solution by cheapest insertion.

    Args:
        depot: ``[2]`` depot coordinates.
        locs: ``[N, 2]`` customer coordinates.
        demand: ``[N]`` customer demands, in the same units as ``capacity``.
        capacity: Vehicle capacity. Per-instance capacities are supported by
            calling this once per instance with its own value.
        order: ``"angle"`` (default), ``"random"``, or an explicit permutation of
            ``0..N-1``.
        exploration: Multiplier on the ``2 * d(depot, customer)`` threshold that
            decides whether to open a new route. The reference's training wrapper
            uses 1.0.
        rng: NumPy generator used only by ``order="random"``.

    Returns:
        One int64 array per route, each holding 0-based customer indices. The
        union is exactly ``0..N-1`` and every route's demand is at most
        ``capacity``.

    Raises:
        ValueError: If the input shapes disagree or a demand exceeds capacity.
    """
    depot = np.asarray(depot, dtype=np.float64)
    locs = np.asarray(locs, dtype=np.float64)
    demand = np.asarray(demand, dtype=np.float64)
    if depot.shape != (2,):
        raise ValueError(f"depot must be [2], got {depot.shape}")
    if locs.ndim != 2 or locs.shape[1] != 2:
        raise ValueError(f"locs must be [N, 2], got {locs.shape}")
    if demand.shape != (locs.shape[0],):
        raise ValueError(f"demand must be [{locs.shape[0]}], got {demand.shape}")
    if np.any(demand > capacity + 1e-9):
        raise ValueError("a customer demand exceeds the vehicle capacity")

    rng = np.random.default_rng() if rng is None else rng
    n = int(locs.shape[0])
    insertion_order = _insertion_order(depot, locs, order, rng)

    # Routes hold 0-based customer indices; route_demand[c] tracks each route's load.
    routes: list[list[int]] = []
    route_demand: list[float] = []

    for customer in insertion_order.tolist():
        currdemand = float(demand[customer])
        customer_xy = locs[customer]
        depotdist = float(np.linalg.norm(customer_xy - depot))
        # The reference initialises with the new-route fallback, scaled by
        # `exploration` for the *decision* only.
        mincost = 2.0 * depotdist * float(exploration)
        best_route: int | None = None
        best_position: int | None = None

        for route_index, route in enumerate(routes):
            if route_demand[route_index] + currdemand > capacity + 1e-9:
                continue
            for position in range(len(route) + 1):
                previous_xy = depot if position == 0 else locs[route[position - 1]]
                next_xy = depot if position == len(route) else locs[route[position]]
                delta = (
                    float(np.linalg.norm(previous_xy - customer_xy))
                    + float(np.linalg.norm(customer_xy - next_xy))
                    - float(np.linalg.norm(previous_xy - next_xy))
                )
                if delta < mincost:
                    mincost = delta
                    best_route = route_index
                    best_position = position

        if best_route is None:
            routes.append([customer])
            route_demand.append(currdemand)
        else:
            assert best_position is not None
            routes[best_route].insert(best_position, customer)
            route_demand[best_route] += currdemand

    result = [np.asarray(route, dtype=np.int64) for route in routes]
    validate_routes(result, demand, capacity)
    return result


def validate_routes(
    routes: list[np.ndarray], demand: np.ndarray, capacity: float
) -> None:
    """Assert that ``routes`` cover every customer once and respect capacity.

    Args:
        routes: Route list as returned by :func:`random_insertion_routes`.
        demand: ``[N]`` customer demands.
        capacity: Vehicle capacity.

    Raises:
        ValueError: If a customer is missing or repeated, or a route exceeds
            capacity.
    """
    demand = np.asarray(demand, dtype=np.float64)
    flat = np.concatenate(routes) if routes else np.empty(0, dtype=np.int64)
    if not np.array_equal(np.sort(flat), np.arange(demand.size, dtype=np.int64)):
        raise ValueError("routes do not cover every customer exactly once")
    for route_index, route in enumerate(routes):
        load = float(demand[np.asarray(route, dtype=np.int64)].sum())
        if load > capacity + 1e-9:
            raise ValueError(
                f"route {route_index} carries {load:.6f} > capacity {capacity:.6f}"
            )
