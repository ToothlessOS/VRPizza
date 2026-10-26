"""Partial-solution geometry for LEHD-style supervised CVRP training and RRC.

The reference implementation
(``CVRP/VRPEnv.py:sampling_subpaths`` and ``sampling_subpaths_repair``) builds a training
example by cutting a **contiguous circular window of ``w`` customers ending at a route end**
out of an optimal solution, then reindexing the window's customers by ascending original id.
The first window customer is the forced start; the rest are the reconstruction target, and
the capacity at the start is the original capacity minus the demand already served earlier
in the same route.

This module owns that geometry once. Both ``src.envs.lehd_dataset`` (training) and
``src.rrc`` (inference) call it, so the flag, capacity and reindexing conventions cannot
drift between the two -- a drift that would otherwise be invisible until the RRC results
stopped matching the training distribution.

Representation
--------------
A solution is a list of routes, each a 1-D int array of customer ids in ``1..N``. Flattening
a route list produces the customer sequence and a binary ``flags`` array with ``1`` at each
route's first customer, which is exactly the ``node_flag`` layout of the data files. The
customer sequence is circular: route order is meaningful only up to rotation, and the
function that picks a window rotates the route list so the chosen route end is last. That
makes the window a suffix, which is what makes the splice in RRC exact.

A route end is a customer after which the vehicle returns to the depot; in the flat
representation it is the last customer of a route (the next position is a route start).
Sampling one route end uniformly is the same as sampling one route uniformly and taking its
last customer, which is what :func:`sample_window` does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "RouteWindow",
    "actions_to_routes",
    "cost_from_flags",
    "flat_to_routes",
    "flatten_routes",
    "parse_window_actions",
    "sample_window",
    "split_routes",
    "transform_routes",
]


@dataclass(frozen=True)
class RouteWindow:
    """One sampled partial solution, in the reduced problem's local indexing.

    Attributes:
        prefix_customers: ``[N-w]`` customers before the window, in solution order.
        prefix_flags: ``[N-w]`` route-start flags for ``prefix_customers``.
        window_customers: ``[w]`` customers of the window, in solution order.
        window_flags: ``[w]`` route-start flags for ``window_customers``. The first entry is
            the forced start's flag; it is not represented by an action and must be preserved
            when the window is spliced back.
        start: The first window customer (original id).
        start_flag: Whether ``start`` begins a route.
        prefix_demand: Demand served earlier in the start's route, in the file's demand units.
        used_capacity0: ``prefix_demand + demand[start]``, in raw demand units. The caller
            divides by the instance capacity to match the environment's normalised demand.
        sorted_ids: ``[w]`` window customer ids sorted ascending; local id ``i`` maps to
            ``sorted_ids[i-1]``.
        local_ids: ``[w]`` local id of each window position (same order as
            ``window_customers``).
        locs_w: ``[w, 2]`` coordinates of the window customers, sorted by original id.
        demand_w: ``[w]`` demands of the window customers, sorted by original id.
        start_local: Local id of the forced start (``local_ids[0]``).
        actions: ``[T]`` teacher action sequence after the forced start. Local customer ids
            are ``1..w``; ``0`` is a depot action inserted wherever the window's next
            customer starts a new route. The final return to the depot is implicit and is
            deliberately absent.
    """

    prefix_customers: np.ndarray
    prefix_flags: np.ndarray
    window_customers: np.ndarray
    window_flags: np.ndarray
    start: int
    start_flag: int
    prefix_demand: float
    used_capacity0: float
    sorted_ids: np.ndarray
    local_ids: np.ndarray
    locs_w: np.ndarray
    demand_w: np.ndarray
    start_local: int
    actions: np.ndarray


def split_routes(tour: np.ndarray, flags: np.ndarray) -> list[np.ndarray]:
    """Cut a flat customer sequence at its route-start flags.

    Args:
        tour: ``[N]`` permutation of ``1..N`` in visit order.
        flags: ``[N]`` binary route-start flags indexed by tour position.

    Returns:
        One int64 array per route.

    Raises:
        ValueError: If the shapes disagree or the first position is not a route start.
    """
    tour = np.asarray(tour, dtype=np.int64)
    flags = np.asarray(flags, dtype=np.int64)
    if tour.ndim != 1 or flags.shape != tour.shape:
        raise ValueError(f"tour {tour.shape} and flags {flags.shape} are not aligned")
    cuts = np.flatnonzero(flags == 1)
    if cuts.size == 0 or cuts[0] != 0:
        raise ValueError("route flags must begin with a 1 marking the first route")
    return [route.copy() for route in np.split(tour, cuts[1:])]


def flatten_routes(routes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten routes into a customer sequence and its route-start flags.

    Args:
        routes: One int array per route, at least one route.

    Returns:
        ``(customers, flags)`` as int64 arrays. ``flags[0] == 1``.

    Raises:
        ValueError: If ``routes`` is empty or contains an empty route.
    """
    if not routes:
        raise ValueError("at least one route is required")
    sizes = [int(np.asarray(route).size) for route in routes]
    if any(size == 0 for size in sizes):
        raise ValueError(f"empty route in lengths {sizes}")
    customers = np.concatenate([np.asarray(route, dtype=np.int64) for route in routes])
    flags = np.zeros(customers.size, dtype=np.int64)
    flags[np.cumsum([0] + sizes[:-1])] = 1
    return customers, flags


def transform_routes(
    routes: list[np.ndarray], rng: np.random.Generator
) -> list[np.ndarray]:
    """Apply LEHD's label-preserving random flips to a solution.

    With probability 0.5 the whole solution is reversed (route order and every route), and
    then each route is reversed independently with probability 0.5. This is the distribution
    of ``vrp_whole_and_solution_subrandom_inverse`` without its rolled-flag implementation:
    the flags are recomputed from the transformed route list, which is equivalent and harder
    to get wrong.

    Args:
        routes: The solution to transform.
        rng: Per-instance generator. Callers seed it by instance so worker order and worker
            count cannot change the draw.

    Returns:
        A new route list; the input is not modified.
    """
    if rng.random() < 0.5:
        routes = [np.asarray(route, dtype=np.int64)[::-1] for route in reversed(routes)]
    else:
        routes = [np.asarray(route, dtype=np.int64) for route in routes]

    out: list[np.ndarray] = []
    for route in routes:
        if rng.random() < 0.5:
            route = route[::-1]
        out.append(route.copy())
    return out


def sample_window(
    routes: list[np.ndarray],
    locs: np.ndarray,
    demand: np.ndarray,
    w: int,
    rng: np.random.Generator,
) -> RouteWindow:
    """Sample one partial solution of length ``w`` ending at a route end.

    The solution is transformed with :func:`transform_routes`, one route is chosen uniformly,
    and the route list is rotated so that route is last. The window is then the last ``w``
    customers of the flattened sequence. Rotating to make the chosen route last does not
    change the circular predecessor set of its end, so this samples the same window
    distribution as LEHD's shift-plus-doubled-buffer construction, while keeping the
    representation linear and the splice exact.

    Args:
        routes: The solution, as a list of customer-id arrays.
        locs: ``[N, 2]`` customer coordinates, indexed by ``id - 1``.
        demand: ``[N]`` customer demands in the file's units, indexed by ``id - 1``.
        w: Window length in customers, ``2 <= w <= N``.
        rng: Per-instance generator.

    Returns:
        The :class:`RouteWindow` in the reduced problem's local indexing.

    Raises:
        ValueError: If ``w`` is out of range or the start's route cannot be located in the
            prefix.
    """
    locs = np.asarray(locs)
    demand = np.asarray(demand, dtype=np.float64)
    routes = transform_routes(routes, rng)

    # Rotate the route list so the chosen route is last. The cyclic order of the other
    # routes is preserved, which is what keeps the window's circular predecessors intact.
    chosen = int(rng.integers(len(routes)))
    ordered = routes[chosen + 1 :] + routes[: chosen + 1]

    customers, flags = flatten_routes(ordered)
    n = int(customers.size)
    w = int(w)
    if not 2 <= w <= n:
        raise ValueError(f"window length {w} is outside [2, {n}]")

    prefix_customers = customers[:-w].copy()
    prefix_flags = flags[:-w].copy()
    window_customers = customers[-w:].copy()
    window_flags = flags[-w:].copy()

    start = int(window_customers[0])
    start_flag = int(window_flags[0])

    if start_flag == 1:
        # The start begins its route; nothing was served in it before the window.
        prefix_demand = 0.0
    else:
        route_starts = np.flatnonzero(prefix_flags == 1)
        if route_starts.size == 0:
            raise ValueError(
                f"start {start} is mid-route but the prefix carries no route start"
            )
        route_start = int(route_starts[-1])
        prefix_demand = float(demand[prefix_customers[route_start:] - 1].sum())

    used_capacity0 = prefix_demand + float(demand[start - 1])

    # Reindex by ascending original id. The window is a permutation of sorted_ids, so
    # searchsorted gives each position's 1-based rank directly.
    sorted_ids = np.sort(window_customers)
    local_ids = np.searchsorted(sorted_ids, window_customers) + 1
    locs_w = locs[sorted_ids - 1]
    demand_w = demand[sorted_ids - 1]
    start_local = int(local_ids[0])

    # Teacher actions after the forced start. A depot action is inserted wherever the next
    # window customer starts a route; the final return to the depot is implicit.
    actions: list[int] = []
    for position in range(1, w):
        if window_flags[position] == 1:
            actions.append(0)
        actions.append(int(local_ids[position]))

    return RouteWindow(
        prefix_customers=prefix_customers,
        prefix_flags=prefix_flags,
        window_customers=window_customers,
        window_flags=window_flags,
        start=start,
        start_flag=start_flag,
        prefix_demand=prefix_demand,
        used_capacity0=used_capacity0,
        sorted_ids=sorted_ids,
        local_ids=local_ids,
        locs_w=locs_w,
        demand_w=demand_w,
        start_local=start_local,
        actions=np.asarray(actions, dtype=np.int64),
    )


def parse_window_actions(
    start: int,
    start_flag: int,
    actions: np.ndarray,
    sorted_ids: np.ndarray,
    n_customers: int,
) -> list[tuple[int, int]]:
    """Turn a decoded action sequence back into the window's ``(customer, flag)`` pairs.

    The forced start is not present in ``actions``: it is prepended with its inherited flag.
    A depot action raises the flag of the following customer. Decoding is stopped as soon as
    all ``n_customers`` window customers have been collected, which drops the depot padding
    the batched decoder emits for rows that finish early.

    Args:
        start: The forced start customer (original id).
        start_flag: The start's route-start flag, inherited from the sampled window.
        actions: Decoded local actions; ``0`` is depot, ``1..w`` are local customer ids.
        sorted_ids: Local id ``i`` maps to ``sorted_ids[i-1]``.
        n_customers: Window length ``w``.

    Returns:
        ``w`` pairs ``(customer, flag)``, starting with the forced start.

    Raises:
        ValueError: If the actions end before all window customers were collected, or a local
            id is out of range.
    """
    pairs: list[tuple[int, int]] = [(int(start), int(start_flag))]
    seen = {int(start)}
    pending_depot = False
    for raw in np.asarray(actions, dtype=np.int64).tolist():
        action = int(raw)
        if action == 0:
            pending_depot = True
            continue
        if not 1 <= action <= len(sorted_ids):
            raise ValueError(f"decoded local id {action} outside [1, {len(sorted_ids)}]")
        customer = int(sorted_ids[action - 1])
        if customer in seen:
            raise ValueError(f"decoded customer {customer} twice")
        seen.add(customer)
        pairs.append((customer, 1 if pending_depot else 0))
        pending_depot = False
        if len(pairs) == n_customers:
            break
    if len(pairs) != n_customers:
        raise ValueError(
            f"decoded {len(pairs) - 1} customers after the start, expected {n_customers - 1}"
        )
    if seen != set(int(value) for value in sorted_ids):
        raise ValueError("decoded customers are not the window's customer set")
    return pairs


def flat_to_routes(customers: np.ndarray, flags: np.ndarray) -> list[np.ndarray]:
    """Inverse of :func:`flatten_routes`."""
    return split_routes(customers, flags)


def actions_to_routes(actions: np.ndarray, n_customers: int) -> list[np.ndarray]:
    """Split a complete rollout's action sequence into routes.

    Trailing depot actions are decoder padding after the episode is done; internal zeros are
    route boundaries. Consecutive zeros are dropped, which is harmless because the environment
    never emits two depot actions in a row while customers remain.

    Args:
        actions: ``[T]`` actions from a POMO rollout; ``0`` is depot, customers are ``1..N``.
        n_customers: Expected number of customers, ``N``.

    Returns:
        One int64 array per route.

    Raises:
        ValueError: If the non-padding actions do not account for exactly ``n_customers``
            customers.
    """
    sequence = np.asarray(actions, dtype=np.int64).tolist()
    while sequence and sequence[-1] == 0:
        sequence.pop()

    routes: list[list[int]] = []
    current: list[int] = []
    for action in sequence:
        if action == 0:
            if current:
                routes.append(current)
                current = []
            continue
        current.append(int(action))
    if current:
        routes.append(current)

    total = sum(len(route) for route in routes)
    if total != int(n_customers):
        raise ValueError(f"action sequence covers {total} customers, expected {n_customers}")
    return [np.asarray(route, dtype=np.int64) for route in routes]


def cost_from_flags(
    customers: np.ndarray,
    flags: np.ndarray,
    depot: np.ndarray,
    locs: np.ndarray,
) -> float:
    """Length of the complete solution encoded by a customer sequence and its flags.

    Each route contributes ``depot -> first + edges + last -> depot``. The sequence must begin
    at a route start; the last customer is always followed by a depot return.

    Args:
        customers: ``[N]`` customer ids in visit order.
        flags: ``[N]`` route-start flags.
        depot: ``[2]`` depot coordinates.
        locs: ``[N, 2]`` customer coordinates, indexed by ``id - 1``.

    Returns:
        The tour length in the coordinates' units, as a Python float.

    Raises:
        ValueError: If the shapes disagree or the first position is not a route start.
    """
    customers = np.asarray(customers, dtype=np.int64)
    flags = np.asarray(flags, dtype=np.int64)
    if customers.shape != flags.shape or customers.ndim != 1:
        raise ValueError(
            f"customers {customers.shape} and flags {flags.shape} are not aligned"
        )
    if flags.size == 0 or flags[0] != 1:
        raise ValueError("the solution must begin at a route start")

    coords = np.vstack(
        [np.asarray(depot, dtype=np.float64)[None, :], np.asarray(locs, dtype=np.float64)]
    )
    sequence = coords[customers]
    depot_xy = coords[0]

    # Depot -> first customer of every route.
    start_in = float(np.linalg.norm(sequence[flags == 1] - depot_xy, axis=1).sum())

    # Each position's outgoing edge: the next customer, or the depot when the next position
    # starts a route (and after the last position).
    next_is_depot = np.empty(customers.size, dtype=bool)
    next_is_depot[:-1] = flags[1:] == 1
    next_is_depot[-1] = True
    next_xy = np.empty_like(sequence)
    next_xy[:-1] = sequence[1:]
    next_xy[-1] = depot_xy
    next_xy[next_is_depot] = depot_xy

    return start_in + float(np.linalg.norm(next_xy - sequence, axis=1).sum())
