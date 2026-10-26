#!/usr/bin/env python
"""Verify the LEHD partial-solution data pipeline, without pytest.

Run from the project root::

    .venv/bin/python scripts/verify_lehd_data.py

The checks pin the geometry the training loss depends on: the window really ends at a route
end, the action replay reproduces the label, the capacity prefix is the demand already served
in the start's route, the reduced indexing round-trips, and the one-``w``-per-effective-batch
draw is stable across dataset instances and DataLoader workers. A final group drives the
partial state through the actual POMO policy and asserts the teacher-forced log-likelihoods
are finite and aligned with the target length.
"""

from __future__ import annotations

import itertools
import sys

from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envs.cvrp_sl_lehd import LEHDEvalCVRPEnv  # noqa: E402
from src.envs.lehd_dataset import LEHDPartialDataset  # noqa: E402
from src.envs.parser import InstanceLayout  # noqa: E402
from src.lehd.subpath import (  # noqa: E402
    actions_to_routes,
    cost_from_flags,
    flat_to_routes,
    flatten_routes,
    parse_window_actions,
    sample_window,
    split_routes,
)
from src.models.pomo_sl import POMOSL  # noqa: E402

DATA = Path("data/CVRP/testing dataset/vrp100_test_lkh.txt")

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record and report the outcome of a single assertion."""
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")
    if not condition:
        failures.append(name)


def sample_records(count: int) -> list[dict[str, np.ndarray]]:
    """Parse the first ``count`` records of the LKH test file."""
    records = []
    with open(DATA) as handle:
        for line in itertools.islice(handle, count):
            layout = InstanceLayout.from_tokens(line.split(","))
            records.append(layout.parse(line))
    return records


def reference_cost(routes: list[np.ndarray], depot: np.ndarray, locs: np.ndarray) -> float:
    """Independent depot-to-depot route-length oracle."""
    coords = np.vstack([depot[None, :], locs])
    total = 0.0
    for route in routes:
        path = coords[np.concatenate([[0], route, [0]])]
        total += float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    return total


def verify_geometry() -> None:
    """Check the window/replay/prefix/splice invariants on real records."""
    print("\n[1/4] window geometry, replay, capacity prefix and splice")
    records = sample_records(12)
    windows = 0
    failures_before = len(failures)
    for index, record in enumerate(records):
        routes = split_routes(record["tour"], record["flags"])
        n = int(record["locs"].shape[0])
        base_cost = reference_cost(routes, record["depot"], record["locs"])
        for seed in range(12):
            rng = np.random.default_rng([seed, index])
            w = int(rng.integers(4, n + 1))
            window = sample_window(routes, record["locs"], record["demand"], w, rng)
            windows += 1

            # The transformed solution starts at a route start and the window is its suffix.
            full_customers = np.concatenate(
                [window.prefix_customers, window.window_customers]
            )
            full_flags = np.concatenate([window.prefix_flags, window.window_flags])
            if full_flags[0] != 1:
                check(f"record {index} seed {seed}: transformed solution starts a route", False)
            if np.sort(window.window_customers).tolist() != np.sort(
                window.sorted_ids
            ).tolist():
                check(f"record {index} seed {seed}: window ids are a permutation", False)

            # Target replay reproduces the window exactly.
            pairs = parse_window_actions(
                window.start, window.start_flag, window.actions, window.sorted_ids, w
            )
            if [pair[0] for pair in pairs] != window.window_customers.tolist():
                check(f"record {index} seed {seed}: replay customers", False)
            if [pair[1] for pair in pairs] != window.window_flags.tolist():
                check(f"record {index} seed {seed}: replay flags", False)

            # Local-id mapping round-trips and the start is the first window customer.
            if int(window.sorted_ids[window.start_local - 1]) != window.start:
                check(f"record {index} seed {seed}: local id mapping", False)
            if window.start != int(window.window_customers[0]):
                check(f"record {index} seed {seed}: start is first window customer", False)

            # Capacity context: the prefix demand is the start route's prior demand.
            if window.start_flag == 1:
                expected_prefix = 0.0
            else:
                route_starts = np.flatnonzero(window.prefix_flags == 1)
                if route_starts.size == 0:
                    expected_prefix = float("nan")
                else:
                    expected_prefix = float(
                        record["demand"][
                            window.prefix_customers[int(route_starts[-1]) :] - 1
                        ].sum()
                    )
            if not np.isclose(expected_prefix, window.prefix_demand):
                check(f"record {index} seed {seed}: capacity prefix", False)
            if not np.isclose(
                window.used_capacity0,
                window.prefix_demand + float(record["demand"][window.start - 1]),
            ):
                check(f"record {index} seed {seed}: used_capacity0", False)

            # The labelled target is capacity-feasible along its own route boundaries.
            used = window.prefix_demand + float(record["demand"][window.start - 1])
            feasible = True
            for customer, flag in zip(
                window.window_customers[1:], window.window_flags[1:]
            ):
                if flag == 1:
                    used = 0.0
                used += float(record["demand"][int(customer) - 1])
                feasible &= used <= float(record["capacity"]) + 1e-6
            if not feasible:
                check(f"record {index} seed {seed}: target capacity", False)

            # Flipping/rotating is label-preserving, and the splice costs the same.
            if not np.isclose(
                cost_from_flags(
                    full_customers, full_flags, record["depot"], record["locs"]
                ),
                base_cost,
                rtol=1e-9,
                atol=1e-5,
            ):
                check(f"record {index} seed {seed}: splice cost", False)

            # The label's route boundaries survive flat -> routes -> flat.
            rebuilt = flat_to_routes(full_customers, full_flags)
            rebuilt_customers, rebuilt_flags = flatten_routes(rebuilt)
            if not (
                np.array_equal(rebuilt_customers, full_customers)
                and np.array_equal(rebuilt_flags, full_flags)
            ):
                check(f"record {index} seed {seed}: flat/route round-trip", False)

    check(
        f"{windows} sampled windows preserve all geometry invariants",
        len(failures) == failures_before,
        f"{windows} windows over {len(records)} records",
    )


def verify_edge_cases() -> None:
    """Synthetic singleton routes, route-start starts, full-capacity routes, w=4 and w=N."""
    print("\n[2/4] edge cases: singleton routes, route starts, full routes, w bounds")
    rng = np.random.default_rng(7)
    depot = np.array([0.5, 0.5], dtype=np.float32)
    # A singleton route in the middle and a route whose demand exactly equals capacity.
    routes = [
        np.array([1, 2]),
        np.array([3]),
        np.array([4, 5, 6]),
        np.array([7, 8]),
    ]
    locs = rng.random((8, 2), dtype=np.float32)
    demand = np.array([2.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)
    capacity = 15.0
    base_cost = reference_cost(routes, depot, locs)

    ok = True
    for w in (4, 8):
        for seed in range(20):
            window = sample_window(
                routes, locs, demand, w, np.random.default_rng(seed)
            )
            pairs = parse_window_actions(
                window.start, window.start_flag, window.actions, window.sorted_ids, w
            )
            ok &= [pair[0] for pair in pairs] == window.window_customers.tolist()
            ok &= [pair[1] for pair in pairs] == window.window_flags.tolist()
            full_customers = np.concatenate(
                [window.prefix_customers, window.window_customers]
            )
            full_flags = np.concatenate([window.prefix_flags, window.window_flags])
            ok &= np.isclose(
                cost_from_flags(full_customers, full_flags, depot, locs),
                base_cost,
                atol=1e-6,
            )
            ok &= full_flags[0] == 1
            # Full-capacity routes must not make the labelled target infeasible.
            used = window.prefix_demand + float(demand[window.start - 1])
            for customer, flag in zip(
                window.window_customers[1:], window.window_flags[1:]
            ):
                if flag == 1:
                    used = 0.0
                used += float(demand[int(customer) - 1])
                ok &= used <= capacity + 1e-6
    check("singleton routes, route starts, w=4 and w=N", bool(ok))


def verify_dataset_draw() -> None:
    """Check deterministic items, one w per effective batch, and worker independence."""
    print("\n[3/4] dataset draw: determinism, w per effective batch, workers")
    dataset = LEHDPartialDataset(
        DATA, instance_range=(0, 40), partials=32, batch_size=16, seed=5
    )
    same = LEHDPartialDataset(
        DATA, instance_range=(0, 40), partials=32, batch_size=16, seed=5
    )
    check("same seed gives the same w draw", np.array_equal(dataset.w_per_chunk, same.w_per_chunk))

    # Effective chunks are 16 rows; the DataLoader may use a smaller microbatch.
    widths = [int(dataset[index]["num_loc"]) for index in range(len(dataset))]
    chunk_widths = [widths[0:16], widths[16:32]]
    check(
        "one w per effective chunk",
        len(set(chunk_widths[0])) == 1 and len(set(chunk_widths[1])) == 1,
        f"{[chunk_widths[0][0], chunk_widths[1][0]]}",
    )

    # Two dataset instances with the same seed emit identical items.
    identical = True
    for index in range(len(dataset)):
        left = dataset[index]
        right = same[index]
        identical &= all(
            np.array_equal(np.asarray(left[key]), np.asarray(right[key]))
            for key in ("locs", "demand", "actions", "start_node", "used_capacity0")
        )
    check("same seed gives identical items", identical)

    # Worker count must not change an item: compare a two-worker loader against the map
    # dataset. One effective chunk keeps every batch the same width, so the comparison is a
    # plain stack. The spawned workers re-import this module, so the `__main__` guard matters.
    from torch.utils.data import DataLoader

    worker_dataset = LEHDPartialDataset(
        DATA, instance_range=(0, 16), partials=16, batch_size=16, seed=5
    )
    loader = DataLoader(
        worker_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        collate_fn=LEHDPartialDataset.collate_fn,
    )
    loaded = torch.cat([batch["locs"] for batch in loader], dim=0).numpy()
    direct = np.stack([worker_dataset[index]["locs"] for index in range(len(worker_dataset))])
    check("two DataLoader workers emit the same items", np.allclose(loaded, direct))

    # The training-style batch has float32 instance fields and a valid mask over the target.
    batch = LEHDPartialDataset.collate_fn([dataset[index] for index in range(16)])
    check(
        "collated batch dtypes",
        batch["demand"].dtype == torch.float32
        and batch["locs"].dtype == torch.float32
        and batch["actions"].dtype == torch.int64
        and batch["valid"].dtype == torch.bool,
    )
    check(
        "target length matches the valid mask",
        int(batch["valid"].sum(1).max()) == int(batch["actions"].shape[1]),
        f"{int(batch['valid'].sum(1).max())} vs {int(batch['actions'].shape[1])}",
    )


def verify_teacher_forcing() -> None:
    """Drive the partial state through POMO and check the loss alignment and finiteness."""
    print("\n[4/4] partial state, teacher forcing and loss")
    dataset = LEHDPartialDataset(
        DATA, instance_range=(0, 80), partials=16, batch_size=16, seed=9
    )
    batch = LEHDPartialDataset.collate_fn([dataset[index] for index in range(16)])
    env = LEHDEvalCVRPEnv(num_loc=100, capacity=50.0)
    td = env.reset_partial(batch)
    model = POMOSL(env, num_augment=1, eval_greedy=False).eval()

    # The start is visited, the depot is pre-visited, and the class is not done before the
    # first decision.
    check("start and depot are pre-visited", bool((td["visited"].sum(1) >= 2).all()))
    check("partial state is not done", not bool(td["done"].any()))
    check(
        "start is the current node",
        torch.equal(td["current_node"].squeeze(1), batch["start_node"]),
    )

    with torch.no_grad():
        out = model.policy(
            td,
            env,
            phase="train",
            actions=batch["actions"],
            store_all_logp=True,
            return_sum_log_likelihood=False,
        )
        log_likelihood = out["log_likelihood"]
        loss = model.calculate_loss(td, batch, out, valid=batch["valid"], logprobs=None)[
            "loss"
        ]
    check("teacher-forced log-likelihood is finite", bool(torch.isfinite(log_likelihood).all()))
    check("loss is finite", bool(torch.isfinite(loss)))
    check(
        "decoded steps equal the longest target",
        log_likelihood.shape[1] == int(batch["valid"].sum(1).max()),
        f"{tuple(log_likelihood.shape)} vs {int(batch['valid'].sum(1).max())}",
    )

    # Replay the target through the environment: every window customer is served exactly once
    # and the episode ends on the last valid action.
    replay = env.reset_partial(batch)
    for step in range(int(batch["valid"].sum(1).max())):
        action = batch["actions"][:, step]
        replay.set("action", action)
        replay = env.step(replay)["next"]
    check("teacher replay finishes", bool(replay["done"].all()))
    check(
        "teacher replay serves every customer",
        bool((replay["visited"][:, 1:].sum(1) == batch["num_loc"]).all()),
    )

    # The per-row action mean is what the model trains on; it must be finite and positive.
    check("per-row CE is positive", float(loss) > 0.0, f"{float(loss):.4f}")


def main() -> int:
    """Run every verification group and report a summary."""
    verify_geometry()
    verify_edge_cases()
    verify_dataset_draw()
    verify_teacher_forcing()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
