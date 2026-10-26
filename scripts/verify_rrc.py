#!/usr/bin/env python
"""Verify Random Re-Construct, without pytest.

Run from the project root::

    .venv/bin/python scripts/verify_rrc.py

The checks exercise the whole inference path on four real CVRP100 instances and a randomly
initialised POMO: the selected multistart trajectory replays to its reported reward, RRC is
monotone and nestable across budgets, greedy and 8x reconstruction run, the window extremes
work, and the LEHD model is bit-identical to POMOSL under the same weights. The
per-iteration feasibility invariants (each window customer exactly once, label-preserving
transforms, exact splice cost) are asserted inside the runner as well, so a violation surfaces
as an exception rather than a quietly worse cost.
"""

from __future__ import annotations

import sys

from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from notebooks.eval_pomo import build_batch  # noqa: E402
from src.envs import LEHDEvalCVRPEnv, load_instances  # noqa: E402
from src.lehd.subpath import (  # noqa: E402
    actions_to_routes,
    cost_from_flags,
    flatten_routes,
)
from src.models import LEHDPOMOSL, POMOSL  # noqa: E402
from src.rrc import RRCConfig, run_rrc  # noqa: E402
from src.rrc.runner import best_multistart  # noqa: E402

DATA = Path("data/CVRP/testing dataset/vrp100_test_lkh.txt")

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record and report the outcome of a single assertion."""
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")
    if not condition:
        failures.append(name)


def make_batch(limit: int = 4):
    """Four complete LKH instances as a batch, with their raw demand and capacity."""
    inst = load_instances(DATA, limit=limit)
    records, scale = inst.model_records(np.arange(limit), normalize=False)
    return build_batch(records), scale, inst


def replay_cost(actions: torch.Tensor, batch) -> float:
    """Exact route length of one instance's action sequence."""
    routes = actions_to_routes(actions.detach().cpu().numpy(), int(batch["locs"].shape[1]))
    customers, flags = flatten_routes(routes)
    return cost_from_flags(
        customers,
        flags,
        batch["depot"][0].detach().cpu().numpy(),
        batch["locs"][0].detach().cpu().numpy(),
    )


def verify_initial_rollout(batch, env, model) -> None:
    """The selected multistart trajectory replays to the reported cost."""
    print("\n[1/5] initial rollout and matched baseline")
    cost, actions = best_multistart(
        model, env, batch, num_starts=10, num_augment=1
    )
    replayed = torch.as_tensor(
        [replay_cost(actions[row], batch[row].unsqueeze(0)) for row in range(len(cost))],
        dtype=cost.dtype,
        device=cost.device,
    )
    check(
        "best-of-starts trajectory replays to the reported cost",
        torch.allclose(cost, replayed, rtol=1e-4, atol=1e-4),
        f"max |diff| {float((cost - replayed).abs().max()):.2e}",
    )
    check("initial costs are finite", bool(torch.isfinite(cost).all()))

    from src.rrc.runner import greedy_rollout

    greedy_cost, greedy_actions = greedy_rollout(model, env, batch)
    check(
        "greedy initial trajectory replays to its reward",
        torch.allclose(
            greedy_cost,
            torch.as_tensor(
                [
                    replay_cost(greedy_actions[row], batch[row].unsqueeze(0))
                    for row in range(len(greedy_cost))
                ],
                dtype=greedy_cost.dtype,
                device=greedy_cost.device,
            ),
            rtol=1e-4,
            atol=1e-4,
        ),
    )


def verify_rrc(batch, env, model) -> None:
    """Monotonicity, nested budgets, and the two initial modes."""
    print("\n[2/5] RRC monotonicity and nested budgets")
    config = RRCConfig(
        budgets=(1, 3), seed=3, init="multistart", num_starts=10, num_augment=1
    )
    result = run_rrc(model, env, batch, config, positions=np.arange(len(batch)))
    check("initial costs finite", bool(np.isfinite(result.initial_cost).all()))
    check("budget costs finite", bool(np.isfinite(result.budget_costs).all()))
    check(
        "costs never increase",
        bool((result.budget_costs <= result.initial_cost[:, None] + 1e-6).all())
        and bool((np.diff(result.budget_costs, axis=1) <= 1e-6).all()),
    )
    first = run_rrc(
        model,
        env,
        batch,
        RRCConfig(budgets=(1,), seed=3, init="multistart", num_starts=10, num_augment=1),
        positions=np.arange(len(batch)),
    )
    check(
        "first budget is reproducible across nested runs",
        np.allclose(first.budget_costs[:, 0], result.budget_costs[:, 0]),
    )
    check(
        "initial incumbent is reproducible",
        np.allclose(first.initial_cost, result.initial_cost),
    )
    check(
        "stats cover the whole loop",
        result.stats.iterations == 3 and result.stats.decoder_rows > 0,
        f"{result.stats.iterations} iterations, {result.stats.decoder_rows} rows",
    )

    greedy = run_rrc(
        model,
        env,
        batch,
        RRCConfig(budgets=(1, 2), seed=3, init="greedy", num_augment=1),
        positions=np.arange(len(batch)),
    )
    check(
        "greedy initial mode is monotone",
        bool((greedy.budget_costs <= greedy.initial_cost[:, None] + 1e-6).all()),
    )
    check(
        "greedy and multistart incumbents differ on at least one instance",
        bool((greedy.initial_cost != result.initial_cost).any()),
    )


def verify_windows(batch, env, model) -> None:
    """The two window extremes and the 8x reconstruction path."""
    print("\n[3/5] window extremes and 8x reconstruction")
    for w_min, w_max, label in ((4, 4, "w=4"), (100, 100, "w=N")):
        result = run_rrc(
            model,
            env,
            batch,
            RRCConfig(
                budgets=(1,),
                w_min=w_min,
                w_max=w_max,
                seed=4,
                init="multistart",
                num_starts=10,
                num_augment=1,
            ),
            positions=np.arange(len(batch)),
        )
        check(
            f"{label}: finite and monotone",
            bool(np.isfinite(result.budget_costs).all())
            and bool((result.budget_costs[:, 0] <= result.initial_cost + 1e-6).all()),
        )

    augmented = run_rrc(
        model,
        env,
        batch,
        RRCConfig(
            budgets=(1,),
            seed=5,
            init="multistart",
            num_starts=10,
            num_augment=1,
            rrc_num_augment=8,
        ),
        positions=np.arange(len(batch)),
    )
    check(
        "8x reconstruction is finite and monotone",
        bool(np.isfinite(augmented.budget_costs).all())
        and bool((augmented.budget_costs[:, 0] <= augmented.initial_cost + 1e-6).all()),
        f"decoder rows {augmented.stats.decoder_rows}",
    )


def verify_model_parity(batch, env) -> None:
    """LEHDPOMOSL and POMOSL are the same network under the same weights."""
    print("\n[4/5] POMO backbone parity and num_starts option")
    plain = POMOSL(env, num_augment=1, eval_greedy=False).eval()
    lehd = LEHDPOMOSL(env, num_augment=1, eval_greedy=False).eval()
    lehd.load_state_dict(plain.state_dict(), strict=True)

    with torch.no_grad():
        plain_reward = plain._inference_step(batch, "test", num_starts=10)["reward"]
        lehd_reward = lehd._inference_step(batch, "test", num_starts=10)["reward"]
        default_reward = lehd._inference_step(batch, "test")["reward"]
        full_reward = lehd._inference_step(batch, "test", num_starts=100)["reward"]
        single_reward = lehd._inference_step(batch, "test", num_starts=1)["reward"]

    check(
        "same weights give identical validation rewards",
        torch.allclose(plain_reward, lehd_reward, rtol=0.0, atol=0.0),
    )
    check(
        "num_starts=None keeps the historical N starts",
        torch.allclose(default_reward, full_reward, rtol=0.0, atol=0.0),
    )
    check("num_starts=1 runs", bool(torch.isfinite(single_reward).all()))


def verify_partial_loss(batch, env) -> None:
    """A teacher-forced partial batch has aligned, finite log-probabilities."""
    print("\n[5/5] partial-state teacher forcing")
    from src.envs.lehd_dataset import LEHDPartialDataset

    dataset = LEHDPartialDataset(
        DATA, instance_range=(0, 8), partials=8, batch_size=8, seed=2
    )
    partial = LEHDPartialDataset.collate_fn([dataset[index] for index in range(8)])
    td = env.reset_partial(partial)
    model = POMOSL(env, num_augment=1, eval_greedy=False).eval()
    with torch.no_grad():
        out = model.policy(
            td,
            env,
            phase="train",
            actions=partial["actions"],
            store_all_logp=True,
            return_sum_log_likelihood=False,
        )
        loss = model.calculate_loss(td, partial, out, valid=partial["valid"])["loss"]
    check(
        "partial teacher forcing is finite",
        bool(torch.isfinite(out["log_likelihood"]).all()) and bool(torch.isfinite(loss)),
    )
    check(
        "partial decode length equals the longest target",
        out["log_likelihood"].shape[1] == int(partial["valid"].sum(1).max()),
    )


def main() -> int:
    """Run every verification group and report a summary."""
    torch.manual_seed(0)
    batch, _scale, inst = make_batch()
    env = LEHDEvalCVRPEnv(num_loc=int(inst.num_loc[0]), capacity=float(inst.capacity[0]))
    model = POMOSL(env, num_augment=1, eval_greedy=False).eval()

    verify_initial_rollout(batch, env, model)
    verify_rrc(batch, env, model)
    verify_windows(batch, env, model)
    verify_model_parity(batch, env)
    verify_partial_loss(batch, env)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
