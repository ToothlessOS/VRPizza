"""Batched Random Re-Construct inference.

The loop mirrors the reference implementation
(``CVRP/VRPTester.py:_test_one_batch``): keep an incumbent complete solution, and each
iteration flip it, choose a route end, cut a ``w``-customer window ending there, reconstruct
that window greedily from its forced first customer, splice it back, and accept iff the full
solution is strictly shorter. The budget is the number of iterations.

Two deliberate choices:

* **Acceptance is on the exact full-tour cost**, not on the partial problem's reward. For a
  mid-route start the partial reward prepends a spurious depot-to-start edge; it is constant
  across augmentations of the same window (so it is fine for choosing among them) but it is
  not the quantity the incumbent is compared on. The geometry is built so the prefix is
  unchanged, which makes the exact full-cost comparison both safe and simple.
* **The window is a suffix.** Rotating the route list so the chosen route end is last does
  not change the window's circular predecessors, and it makes the splice ``prefix + window``
  with no suffix to track. The next iteration re-rotates, so every route end remains
  reachable.

Multi-start is an initial-solution strategy, not a reconstruction strategy: the start of a
window is fixed by the splice (the prefix must remain connected), so forcing different starts
would break the partial problem. ``best_multistart`` is therefore used to produce the
incumbent and the matched plain baseline; reconstructions run a single greedy trajectory.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from rl4co.data.transforms import StateAugmentation
from rl4co.utils.ops import unbatchify
from tensordict.tensordict import TensorDict

from src.lehd.subpath import (
    actions_to_routes,
    flat_to_routes,
    flatten_routes,
    parse_window_actions,
    sample_window,
)
from src.rrc.config import RRCConfig, RRCResult, RRCStats

__all__ = [
    "best_multistart",
    "cost_from_flags_torch",
    "greedy_rollout",
    "run_rrc",
]


def _augmentation(num_augment: int) -> StateAugmentation | None:
    """Dihedral augmentation, or ``None`` when it is disabled."""
    if num_augment <= 1:
        return None
    return StateAugmentation(num_augment=num_augment, augment_fn="dihedral8")


@torch.no_grad()
def best_multistart(
    model,
    env,
    batch: TensorDict,
    num_starts: int | None = None,
    num_augment: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run POMO's ``best of augmentations x starts`` rollout and return its best trajectories.

    Args:
        model: A fitted POMO-family Lightning module with a ``policy``.
        env: The full-instance evaluation environment, with ``num_loc = N``.
        batch: A batch of complete instances.
        num_starts: POMO starts, or ``None`` for ``env.get_num_starts(td) = N``.
        num_augment: Dihedral augmentations; 1 disables them.

    Returns:
        ``(cost, actions)`` with ``cost [B]`` (negative reward, model units) and
        ``actions [B, T]`` for the best rollout of each instance.

    Raises:
        ValueError: If ``num_starts`` is not positive.
    """
    td = env.reset(env.make_input_td(batch))
    augment = _augmentation(num_augment)
    if augment is not None:
        td = augment(td)

    starts = env.get_num_starts(td) if num_starts is None else int(num_starts)
    if starts <= 0:
        raise ValueError(f"num_starts must be positive, got {starts}")

    out = model.policy(td, env, phase="test", num_starts=starts)
    # [B, A, S] and [B, A, S, T]; the layouts are aligned because both were batchified and
    # unbatchified with the same (augmentations, starts) shape.
    reward = unbatchify(out["reward"], (num_augment, starts))
    actions = unbatchify(out["actions"], (num_augment, starts))
    size = reward.shape[0]
    flat_reward = reward.reshape(size, -1)
    best = flat_reward.argmax(dim=1)
    cost = -flat_reward.gather(1, best[:, None]).squeeze(1)
    width = actions.shape[-1]
    best_actions = actions.reshape(size, -1, width).gather(
        1, best[:, None, None].expand(-1, 1, width)
    ).squeeze(1)
    return cost, best_actions


@torch.no_grad()
def greedy_rollout(model, env, batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    """One greedy complete rollout, the LEHD-style initial incumbent.

    The depot is masked at reset, so the first action is a customer and the construction is a
    single-start greedy solution -- which is what the reference RRC starts from.

    Args:
        model: A fitted POMO-family Lightning module.
        env: The full-instance evaluation environment.
        batch: A batch of complete instances.

    Returns:
        ``(cost, actions)`` with ``cost [B]`` and ``actions [B, T]``.
    """
    td = env.reset(env.make_input_td(batch))
    out = model.policy(td, env, phase="test", decode_type="greedy")
    return -out["reward"], out["actions"]


def cost_from_flags_torch(
    customers: torch.Tensor,
    flags: torch.Tensor,
    depot: torch.Tensor,
    locs: torch.Tensor,
) -> torch.Tensor:
    """Vectorised full-tour length for batched ``(customers, flags)`` solutions.

    Mirrors :func:`src.lehd.subpath.cost_from_flags`: each route contributes
    ``depot -> first + edges + last -> depot``. Computed in float64 on the device; the batch
    is small and the acceptance comparison should not be decided by float32 noise.

    Args:
        customers: ``[B, N]`` customer ids in ``1..N``.
        flags: ``[B, N]``, ``1`` at each route's first customer.
        depot: ``[B, 2]`` depot coordinates.
        locs: ``[B, N, 2]`` customer coordinates, indexed by ``id - 1``.

    Returns:
        ``[B]`` float64 tour lengths.
    """
    customers = customers.long()
    flags = flags.bool()
    coords = torch.cat([depot[:, None, :], locs], dim=1).double()
    sequence = coords.gather(1, customers[..., None].expand(-1, -1, 2))
    depot_xy = coords[:, 0]

    start_in = torch.where(
        flags,
        (sequence - depot_xy[:, None, :]).norm(dim=-1),
        torch.zeros((), dtype=torch.float64, device=sequence.device),
    ).sum(dim=1)

    next_is_depot = torch.empty_like(flags)
    next_is_depot[:, :-1] = flags[:, 1:]
    next_is_depot[:, -1] = True
    next_xy = torch.empty_like(sequence)
    next_xy[:, :-1] = sequence[:, 1:]
    next_xy[:, -1] = depot_xy
    next_xy = torch.where(next_is_depot[..., None], depot_xy[:, None, :], next_xy)

    return start_in + (next_xy - sequence).norm(dim=-1).sum(dim=1)


def _routes_to_flat(routes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten one instance's routes for the batched cost function."""
    return flatten_routes(routes)


@torch.no_grad()
def run_rrc(
    model,
    env,
    batch: TensorDict,
    config: RRCConfig,
    positions: np.ndarray | None = None,
) -> RRCResult:
    """Run RRC on a batch of complete instances.

    Args:
        model: A fitted POMO-family Lightning module; its training objective may be the LEHD
            one or a baseline's.
        env: The full-instance evaluation environment. The same object is used for the
            initial rollout and for the reduced reconstructions; ``reset_partial`` takes the
            reduced batch's width from its data, so the environment's ``num_loc`` is only used
            by the full-instance start selection.
        batch: Complete instances with ``depot [B,2]``, ``locs [B,N,2]``, raw
            ``demand [B,N]``, ``capacity [B]`` and ``cost [B]``.
        config: The RRC protocol.
        positions: Original instance positions, used to seed the per-instance draws so the
            same instance gets the same flips/route-end choices in every checkpoint's run.
            Defaults to ``arange(B)``.

    Returns:
        :class:`~src.rrc.config.RRCResult` with the exact initial cost and the incumbent cost
        after each configured budget.

    Raises:
        AssertionError: If the model's reported reward and the replayed tour length disagree,
            or if a label-preserving transform changed the incumbent's cost. Both indicate a
            parsing or indexing bug that would otherwise silently corrupt the acceptance.
    """
    started = time.time()
    size = int(batch.batch_size[0])
    n_loc = int(batch["locs"].shape[1])
    if positions is None:
        positions = np.arange(size, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    if positions.size != size:
        raise ValueError(f"positions has {positions.size} entries for a batch of {size}")

    w_max = n_loc if config.w_max is None else min(int(config.w_max), n_loc)
    if config.w_min > w_max:
        raise ValueError(f"w_min={config.w_min} exceeds the usable maximum {w_max}")

    # -- initial incumbent and matched plain baseline ---------------------------------
    if config.init == "multistart":
        reward, actions = best_multistart(
            model, env, batch, config.num_starts, config.num_augment
        )
    else:
        reward, actions = greedy_rollout(model, env, batch)

    routes = [actions_to_routes(actions[row].detach().cpu().numpy(), n_loc) for row in range(size)]
    flats = [_routes_to_flat(route_list) for route_list in routes]
    customers = torch.as_tensor(
        np.stack([flat[0] for flat in flats]), device=batch.device
    )
    flags = torch.as_tensor(
        np.stack([flat[1] for flat in flats]), device=batch.device
    ).bool()
    costs = cost_from_flags_torch(customers, flags, batch["depot"], batch["locs"])
    if not torch.allclose(costs, reward.double(), rtol=1e-4, atol=1e-4):
        raise AssertionError(
            "the initial rollout's reward and the replayed route length disagree; the action "
            "sequence and the coordinates are not in the same frame"
        )
    initial_cost = costs.detach().cpu().numpy().copy()

    budgets = config.budgets
    budget_costs = np.full((size, len(budgets)), np.nan, dtype=np.float64)
    budget_index = {budget: index for index, budget in enumerate(budgets)}
    stats = RRCStats()

    # Geometry is CPU-side; the tensors are moved once per chunk.
    locs_cpu = batch["locs"].detach().cpu().numpy()
    demand_cpu = batch["demand"].detach().cpu().numpy()
    depot_cpu = batch["depot"].detach().cpu().numpy()
    capacity_cpu = batch["capacity"].detach().cpu().numpy()

    for iteration in range(config.max_budget):
        # One w per batch, matching LEHD's sampler; the same draw across budgets/checkpoints.
        w = int(
            np.random.default_rng([config.seed, iteration]).integers(
                config.w_min, w_max + 1
            )
        )
        windows = [
            sample_window(
                routes[row],
                locs_cpu[row],
                demand_cpu[row],
                w,
                np.random.default_rng([config.seed, iteration, int(positions[row])]),
            )
            for row in range(size)
        ]

        reduced = TensorDict(
            {
                "depot": torch.as_tensor(
                    depot_cpu, dtype=torch.float32, device=batch.device
                ),
                "locs": torch.as_tensor(
                    np.stack([window.locs_w for window in windows]),
                    dtype=torch.float32,
                    device=batch.device,
                ),
                "demand": torch.as_tensor(
                    np.stack([window.demand_w for window in windows]),
                    dtype=torch.float32,
                    device=batch.device,
                ),
                "capacity": torch.as_tensor(
                    capacity_cpu, dtype=torch.float32, device=batch.device
                ),
                "start_node": torch.as_tensor(
                    np.asarray([window.start_local for window in windows], dtype=np.int64),
                    device=batch.device,
                ),
                "used_capacity0": torch.as_tensor(
                    np.asarray(
                        [window.used_capacity0 for window in windows], dtype=np.float32
                    ),
                    device=batch.device,
                ),
            },
            batch_size=[size],
            device=batch.device,
        )
        td = env.reset_partial(reduced)

        augment = _augmentation(config.rrc_num_augment)
        if augment is not None:
            td = augment(td)
            out = model.policy(td, env, phase="test", decode_type="greedy")
            rewards = unbatchify(out["reward"], (config.rrc_num_augment,))
            actions = unbatchify(out["actions"], (config.rrc_num_augment,))
            best = rewards.argmax(dim=1)
            width = actions.shape[-1]
            selected = actions.gather(
                1, best[:, None, None].expand(-1, 1, width)
            ).squeeze(1)
        else:
            out = model.policy(td, env, phase="test", decode_type="greedy")
            selected = out["actions"]

        stats.decoder_rows += int(out["actions"].numel())
        decoded = selected.detach().cpu().numpy()

        # Parse the reconstructed window, then build both candidates in one tensor each:
        # the accepted candidate and the transformed incumbent (kept if rejected).
        candidate_customers = np.empty((size, n_loc), dtype=np.int64)
        candidate_flags = np.empty((size, n_loc), dtype=np.int64)
        transformed_customers = np.empty((size, n_loc), dtype=np.int64)
        transformed_flags = np.empty((size, n_loc), dtype=np.int64)
        for row, window in enumerate(windows):
            pairs = parse_window_actions(
                window.start, window.start_flag, decoded[row], window.sorted_ids, w
            )
            candidate_customers[row, : n_loc - w] = window.prefix_customers
            candidate_flags[row, : n_loc - w] = window.prefix_flags
            candidate_customers[row, n_loc - w :] = [pair[0] for pair in pairs]
            candidate_flags[row, n_loc - w :] = [pair[1] for pair in pairs]
            transformed_customers[row] = np.concatenate(
                [window.prefix_customers, window.window_customers]
            )
            transformed_flags[row] = np.concatenate(
                [window.prefix_flags, window.window_flags]
            )

        candidate_customers_t = torch.as_tensor(candidate_customers, device=batch.device)
        candidate_flags_t = torch.as_tensor(candidate_flags, device=batch.device).bool()
        transformed_customers_t = torch.as_tensor(transformed_customers, device=batch.device)
        transformed_flags_t = torch.as_tensor(transformed_flags, device=batch.device).bool()

        candidate_cost = cost_from_flags_torch(
            candidate_customers_t, candidate_flags_t, batch["depot"], batch["locs"]
        )
        transformed_cost = cost_from_flags_torch(
            transformed_customers_t, transformed_flags_t, batch["depot"], batch["locs"]
        )
        if not torch.allclose(transformed_cost, costs, rtol=1e-5, atol=1e-5):
            raise AssertionError(
                "a label-preserving flip/rotation changed the incumbent's cost; the route "
                "representation or the flag reconstruction is wrong"
            )

        accept = candidate_cost < costs
        costs = torch.where(accept, candidate_cost, costs)
        chosen_customers = torch.where(
            accept[:, None], candidate_customers_t, transformed_customers_t
        )
        chosen_flags = torch.where(accept[:, None], candidate_flags_t, transformed_flags_t)
        routes = [
            flat_to_routes(
                chosen_customers[row].detach().cpu().numpy(),
                chosen_flags[row].detach().cpu().numpy().astype(np.int64),
            )
            for row in range(size)
        ]

        stats.iterations += 1
        stats.accepted += int(accept.sum())
        if iteration + 1 in budget_index:
            budget_costs[:, budget_index[iteration + 1]] = (
                costs.detach().cpu().numpy().astype(np.float64)
            )

    stats.seconds = time.time() - started
    return RRCResult(initial_cost=initial_cost, budget_costs=budget_costs, stats=stats)
