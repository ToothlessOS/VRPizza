#!/usr/bin/env python
"""Verify the SL data pipeline end to end, without pytest.

Run from the project root::

    .venv/bin/python scripts/verify_sl_data.py

The checks, in order, prove that the parser reads every file layout correctly, that the
``2R`` optimal-start trajectories reproduce the reference cost exactly, that replaying
them through the environment yields a valid and correctly-priced solution, and that
padding a batch of unequal-length trajectories leaves the reward and loss untouched.
"""

from __future__ import annotations

import itertools
import sys

from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envs import CVRPEnv, CVRPSolutionDataset  # noqa: E402
from src.envs.parser import InstanceLayout  # noqa: E402

DATA = Path("data/CVRP")
# The HGS training files score slightly different coordinates than they store, so their
# cost column carries ~6e-5 relative noise. The LKH test files match to machine
# precision, which is what makes them a usable oracle for the construction itself.
FILES = {
    DATA / "vrp100_hgs_train_100w-001.txt": 2e-3,
    DATA / "vrp50_train_hgs_n1000000_C40-002.txt": 2e-3,
    DATA / "testing dataset/vrp100_test_lkh.txt": 1e-5,
    DATA / "testing dataset/vrp200_test_lkh.txt": 1e-5,
    DATA / "testing dataset/vrp500_test_lkh.txt": 1e-5,
    DATA / "testing dataset/vrp1000_test_lkh.txt": 1e-5,
}

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record and report the outcome of a single assertion."""
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")
    if not condition:
        failures.append(name)


def sample_lines(path: Path, count: int) -> list[str]:
    """Return the first ``count`` lines of ``path``."""
    with open(path) as handle:
        return list(itertools.islice(handle, count))


def reference_cost(tokens: list[str], layout: InstanceLayout, actions: np.ndarray) -> float:
    """Compute a trajectory's Euclidean length in float64 straight from the raw text."""
    n = layout.num_loc
    coords = np.vstack(
        [
            [float(t) for t in tokens[layout.depot_idx + 1 : layout.depot_idx + 3]],
            np.array(
                [
                    float(t)
                    for t in tokens[layout.customer_idx + 1 : layout.customer_idx + 1 + 2 * n]
                ]
            ).reshape(n, 2),
        ]
    )
    walk = [0, *actions.tolist()]
    return float(
        sum(np.linalg.norm(coords[walk[i + 1]] - coords[walk[i]]) for i in range(len(walk) - 1))
    )


def verify_parser_and_trajectories() -> None:
    """Check the layout detection and the exactness of all ``2R`` optimal starts."""
    print("\n[1/3] parser and optimal-start trajectories")
    for path, tol in FILES.items():
        if not path.exists():
            print(f"  SKIP  {path.name} (missing)")
            continue

        worst = 0.0
        instances = 0
        exact = 0
        total = 0
        for line in sample_lines(path, 60):
            tokens = line.split(",")
            layout = InstanceLayout.from_tokens(tokens)
            record = layout.parse(line)
            if not layout.has_tour:
                continue
            instances += 1
            routes = int(np.sum(record["flags"] == 1))
            for slot in range(2 * routes):
                route_index, direction = divmod(slot, 2)
                actions = CVRPSolutionDataset.build_trajectory(
                    record["tour"], record["flags"], route_index, direction
                )
                deviation = abs(reference_cost(tokens, layout, actions) - float(record["cost"]))
                worst = max(worst, deviation)
                exact += deviation < tol
                total += 1

        check(
            f"{path.name}: {total} trajectories / {instances} instances, "
            f"worst |cost-ref| {worst:.2e}",
            exact == total,
            f"{exact}/{total} within {tol:g}",
        )


def verify_environment_replay() -> None:
    """Replay expert trajectories through the env and compare reward to the reference."""
    print("\n[2/3] environment replay and decode-loop invariant")
    path = DATA / "testing dataset/vrp100_test_lkh.txt"
    if not path.exists():
        print(f"  SKIP  {path.name} (missing)")
        return

    env = CVRPEnv(data_path=str(path), num_loc=100, capacity=50.0, val_instances=1000,
                  check_solution=True)
    dataset = env.dataset(128, phase="train")
    batch = dataset.collate_fn([dataset[i] for i in range(128)])

    actions, num_routes, valid = env.build_expert_actions(batch)
    td = env.reset(env.make_input_td(batch))
    # check_solution=True makes get_reward assert the solution is a valid permutation
    # that respects capacity, so this also proves feasibility of the padded batch.
    reward = env.get_reward(td, actions)
    reference = env.reference_cost(batch)
    error = (reward + reference).abs()
    check(
        "replayed expert reward matches reference",
        bool((error / reference).max() < 1e-5),
        f"max relative {float((error / reference).max()):.2e}",
    )
    check(
        "expert sequences are valid tours within capacity",
        True,
        "check_solution_validity passed for all 128",
    )

    model = _build_model(env)
    out = model.policy(env.reset(env.make_input_td(batch)), env, phase="train", actions=actions,
                       store_all_logp=True, return_sum_log_likelihood=False)
    log_likelihood = out["log_likelihood"]
    expected = 100 + int(num_routes.max()) - 1
    check(
        "decode loop consumes N + R_max - 1 actions",
        log_likelihood.shape[1] == expected,
        f"got {log_likelihood.shape[1]}, expected {expected}",
    )
    check(
        "no -inf or NaN log-probs on padded steps",
        bool(torch.isfinite(log_likelihood).all()),
        f"min {float(log_likelihood.min()):.3f}",
    )


def verify_padding_safety() -> None:
    """Check that padding trajectories of unequal length changes neither reward nor loss."""
    print("\n[3/3] padding safety and store_all_logp equivalence")
    path = DATA / "testing dataset/vrp100_test_lkh.txt"
    if not path.exists():
        print(f"  SKIP  {path.name} (missing)")
        return

    env = CVRPEnv(data_path=str(path), num_loc=100, capacity=50.0, val_instances=1000)
    dataset = env.dataset(256, phase="train")
    batch = dataset.collate_fn([dataset[i] for i in range(256)])
    actions, _, valid = env.build_expert_actions(batch)
    reference = env.reference_cost(batch)

    # Recompute each row's reward from its own unpadded slice and compare.
    padded_reward = -env.get_reward(env.reset(env.make_input_td(batch)), actions)
    worst = 0.0
    for i in range(batch.batch_size[0]):
        span = int(valid[i].sum())
        single = env.reset(env.make_input_td(batch[i : i + 1]))
        unpadded = -env.get_reward(single, actions[i : i + 1, :span])
        worst = max(worst, abs(float(unpadded) - float(padded_reward[i])))
    check("padded reward equals unpadded reward", worst < 1e-5, f"max abs diff {worst:.2e}")

    def is_prefix(row: torch.Tensor) -> bool:
        """Whether a boolean mask is contiguous True values followed by False values."""
        span = int(row.sum())
        return bool(row[:span].all()) and not bool(row[span:].any())

    check(
        "valid mask is a strict prefix of each row",
        all(is_prefix(valid[i]) for i in range(batch.batch_size[0])),
    )

    model = _build_model(env)
    base = env.reset(env.make_input_td(batch))
    with torch.no_grad():
        allp = model.policy(base, env, phase="train", actions=actions, store_all_logp=True,
                            return_sum_log_likelihood=False)["log_likelihood"]
        base2 = env.reset(env.make_input_td(batch))
        summed = model.policy(base2, env, phase="train", actions=actions, store_all_logp=False,
                              return_sum_log_likelihood=False)["log_likelihood"]
    mask = valid[:, : allp.shape[1]].to(allp.dtype)
    ce_all = float(-((allp * mask).sum() / mask.sum()))
    ce_summed = float(-((summed * mask).sum() / mask.sum()))
    check(
        "store_all_logp=True matches store_all_logp=False",
        abs(ce_all - ce_summed) < 1e-5,
        f"{ce_all:.6f} vs {ce_summed:.6f}",
    )
    check("reference cost is positive", bool((reference > 0).all()))


def _build_model(env):
    """Instantiate a POMOSL with a tiny inference configuration for the checks above."""
    from src.models import POMOSL

    return POMOSL(env, batch_size=64, num_augment=1)


def main() -> int:
    """Run every verification group and report a summary."""
    verify_parser_and_trajectories()
    verify_environment_replay()
    verify_padding_safety()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
