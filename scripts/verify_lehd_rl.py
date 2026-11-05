#!/usr/bin/env python
"""Verify the LEHD-style RL reward, both baseline arms, and the label boundary.

Run from the project root::

    .venv/bin/python scripts/verify_lehd_rl.py

The checks pin the claims in ``docs/LEHD_RL.md``:

* the reduced closed-tour cost helper matches an independent computation;
* for two different reconstructions of the same window, ``C_full - C_red`` is the same
  constant (the fixed-start invariant that makes the partial reward exact);
* the environment reward includes the pre-served start, and the post-start state is right;
* one sampled rollout (teacher arm) and K sampled rollouts (shared arm) have the shapes and
  rewards the loss assumes;
* the shared arm's loss does not read ``actions`` or ``cost`` — the label-free boundary —
  while the teacher baseline does depend on ``actions``;
* the loss math is right and differentiable;
* the model has the same state dict and validation rewards as plain POMORL.

The suite uses real records from the LKH test file and a one-layer policy on CPU, so it does
not need a GPU, experiment tracking or a trained checkpoint.
"""

from __future__ import annotations

import itertools
import sys

from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl4co.utils.ops import unbatchify  # noqa: E402

from src.envs.cvrp_rl_lehd import LEHDRLCVRPEnv, reduced_window_cost  # noqa: E402
from src.envs.lehd_dataset import collate_partials  # noqa: E402
from src.envs.parser import InstanceLayout  # noqa: E402
from src.lehd.subpath import (  # noqa: E402
    cost_from_flags,
    sample_window,
    split_routes,
)
from src.models import LEHDPOMORL, POMORL  # noqa: E402

DATA = Path("data/CVRP/testing dataset/vrp100_test_lkh.txt")
CAPACITY = 50.0

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


def build_batch(records: list[dict], w: int, seed: int = 0):
    """Build a reduced TensorDict batch and keep the RouteWindows for the invariant check."""
    items, windows = [], []
    for index, record in enumerate(records):
        routes = split_routes(record["tour"], record["flags"])
        rng = np.random.default_rng([seed, index])
        window = sample_window(routes, record["locs"], record["demand"], w, rng)
        windows.append((record, window))
        items.append(
            {
                "depot": record["depot"],
                "locs": window.locs_w,
                "demand": window.demand_w,
                "capacity": record["capacity"],
                "cost": record["cost"],
                "start_node": np.int64(window.start_local),
                "used_capacity0": np.float32(window.used_capacity0),
                "actions": window.actions,
                "num_loc": np.int64(w),
            }
        )
    return collate_partials(items), windows


def reference_reduced_cost(depot, locs, actions) -> float:
    """Independent NumPy cost of ``depot -> first -> ... -> last -> depot``.

    ``actions`` are local ids (0 = depot, 1..w customers) and are read one row at a time.
    """
    coords = np.vstack([np.asarray(depot, dtype=np.float64)[None, :], locs])
    # Prepend the depot index 0, exactly like `reduced_window_cost`; otherwise the closure
    # would be last -> start instead of last -> depot and the depot -> start edge is missing.
    indices = np.concatenate([[0], np.asarray(actions, dtype=np.int64)])
    sequence = coords[indices]
    closed = np.vstack([sequence, sequence[:1]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def full_cost(record: dict, prefix_customers, prefix_flags, customers, flags) -> float:
    """Exact full-tour cost after splicing a reconstructed window into the prefix."""
    return cost_from_flags(
        np.concatenate([prefix_customers, customers]),
        np.concatenate([prefix_flags, flags]),
        record["depot"],
        record["locs"],
    )


def verify_cost_helper() -> None:
    """The helper matches an independent computation for every row."""
    print("\n[1/8] reduced cost helper")
    batch, windows = build_batch(sample_records(4), w=20)
    actions = torch.cat([batch["start_node"][:, None], batch["actions"]], dim=1)
    cost = reduced_window_cost(batch["depot"], batch["locs"], actions)
    for row, (record, window) in enumerate(windows):
        expected = reference_reduced_cost(record["depot"], window.locs_w, actions[row].numpy())
        check(
            f"row {row}: helper == numpy",
            abs(float(cost[row]) - expected) < 1e-5,
            f"{float(cost[row]):.6f} vs {expected:.6f}",
        )


def verify_fixed_start_invariant() -> None:
    """C_full - C_red is the same constant for any reconstruction of a window."""
    print("\n[2/8] fixed-start invariant: C_full - C_red is constant")
    batch, windows = build_batch(sample_records(6), w=30)
    for row, (record, window) in enumerate(windows):
        # Teacher reconstruction.
        teacher_full = full_cost(
            record,
            window.prefix_customers,
            window.prefix_flags,
            window.window_customers,
            window.window_flags,
        )
        teacher_actions = np.concatenate([[window.start_local], window.actions])
        teacher_red = reference_reduced_cost(record["depot"], window.locs_w, teacher_actions)

        # A deliberately different reconstruction: keep the start, permute the rest, and put
        # everything in the start's route. Feasibility is irrelevant to the cost algebra.
        rng = np.random.default_rng([99, row])
        shuffled = window.window_customers[1:].copy()
        rng.shuffle(shuffled)
        customers = np.concatenate([[window.start], shuffled])
        flags = np.zeros(customers.size, dtype=np.int64)
        flags[0] = window.start_flag
        random_full = full_cost(
            record,
            window.prefix_customers,
            window.prefix_flags,
            customers,
            flags,
        )
        local = np.searchsorted(window.sorted_ids, shuffled) + 1
        random_red = reference_reduced_cost(
            record["depot"], window.locs_w, np.concatenate([[window.start_local], local])
        )

        check(
            f"row {row}: offset constant",
            abs((teacher_full - teacher_red) - (random_full - random_red)) < 1e-5,
            f"{(teacher_full - teacher_red):.6f} vs {(random_full - random_red):.6f}",
        )


def make_env() -> LEHDRLCVRPEnv:
    """A dataset-backed environment for the partial-state checks."""
    return LEHDRLCVRPEnv(
        data_path=str(DATA),
        num_loc=100,
        capacity=CAPACITY,
        val_instances=100,
        effective_batch_size=8,
    )


def verify_state_and_reward() -> None:
    """reset_partial builds the post-start state; _get_reward includes the start."""
    print("\n[3/8] post-start state and environment reward")
    env = make_env()
    batch, windows = build_batch(sample_records(3), w=12)
    td = env.reset_partial(batch)
    size = batch.batch_size[0]
    rows = torch.arange(size)

    check(
        "depot visited",
        bool(td["visited"][:, 0].all()),
    )
    check(
        "start visited and current",
        bool(td["visited"][rows, batch["start_node"]].all())
        and bool((td["current_node"].squeeze(-1) == batch["start_node"]).all()),
    )
    expected_used = (batch["used_capacity0"] / batch["capacity"])[:, None]
    check(
        "residual capacity carried",
        torch.allclose(td["used_capacity"], expected_used, atol=1e-6),
        f"{td['used_capacity'].flatten().tolist()} vs {expected_used.flatten().tolist()}",
    )
    check("not done after reset", not bool(td["done"].any()))
    check(
        "first teacher action feasible",
        bool(td["action_mask"].gather(1, batch["actions"][:, :1]).all()),
    )

    actions = torch.cat([batch["start_node"][:, None], batch["actions"]], dim=1)
    expected = -reduced_window_cost(batch["depot"], batch["locs"], actions)
    actual = env.get_reward(td, batch["actions"])
    check("env reward == -C_red", torch.allclose(actual, expected, atol=1e-5))


def make_model(env, *, baseline="teacher", num_samples=1) -> LEHDPOMORL:
    """A one-layer model for CPU checks."""
    return LEHDPOMORL(
        env,
        num_augment=1,
        eval_greedy=False,
        advantage_baseline=baseline,
        num_samples=num_samples,
        policy_kwargs={"num_encoder_layers": 1},
    )


def verify_teacher_rollout() -> None:
    """One sampled rollout, its reward, and teacher-forced evaluation."""
    print("\n[4/8] teacher arm: single sampled rollout")
    env = make_env()
    model = make_model(env).eval()
    batch, _ = build_batch(sample_records(3), w=10)
    td = env.reset_partial(batch)

    with torch.no_grad():
        out = model.policy(td, env, phase="train", decode_type="sampling")
        expected = -reduced_window_cost(
            batch["depot"],
            batch["locs"],
            torch.cat([batch["start_node"][:, None], out["actions"]], dim=1),
        )
        check(
            "reward == -C_red of the sampled actions",
            torch.allclose(out["reward"], expected, atol=1e-5),
        )
        check("reward is [B]", tuple(out["reward"].shape) == (batch.batch_size[0],))

        teacher_cost = reduced_window_cost(
            batch["depot"],
            batch["locs"],
            torch.cat([batch["start_node"][:, None], batch["actions"]], dim=1),
        )
        # The policy steps mutate the state in place, so build a fresh one for the replay.
        forced = model.policy(
            env.reset_partial(batch), env, phase="train", actions=batch["actions"]
        )
        check(
            "teacher-forced reward == -C_teacher",
            torch.allclose(forced["reward"], -teacher_cost, atol=1e-5),
        )


def verify_shared_rollout() -> None:
    """K sampled rollouts, per-sample rewards, and the label-free loss boundary."""
    print("\n[5/8] shared arm: K sampled rollouts and label-freeness")
    env = make_env()
    model = make_model(env, baseline="shared", num_samples=4).eval()
    batch, _ = build_batch(sample_records(3), w=10)
    td = env.reset_partial(batch)

    with torch.no_grad():
        out = model.policy(td, env, phase="train", decode_type="sampling", num_samples=4)
        reward = unbatchify(out["reward"], 4)  # [B, K]
        actions = unbatchify(out["actions"], 4)  # [B, K, T]
        check("shared reward is [B, K]", tuple(reward.shape) == (3, 4))
        expected = torch.stack(
            [
                -reduced_window_cost(
                    batch["depot"],
                    batch["locs"],
                    torch.cat(
                        [batch["start_node"][:, None], actions[:, k]], dim=1
                    ),
                )
                for k in range(4)
            ],
            dim=1,
        )
        check("each sample's reward == -C_red", torch.allclose(reward, expected, atol=1e-5))

        logp = unbatchify(out["log_likelihood"], 4)
        # The shared loss reads only `num_loc` from the batch. Replace the label-bearing
        # fields and assert the loss is unchanged.
        clean = model.calculate_loss(td, batch, {}, reward, logp)
        perturbed_batch = batch.clone()
        perturbed_batch["actions"] = torch.randint(
            0, 11, batch["actions"].shape, dtype=batch["actions"].dtype
        )
        perturbed_batch["cost"] = torch.rand_like(batch["cost"])
        perturbed = model.calculate_loss(td, perturbed_batch, {}, reward, logp)
        check(
            "shared loss ignores actions/cost",
            torch.allclose(clean["loss"], perturbed["loss"]),
            f"{float(clean['loss']):.6f} vs {float(perturbed['loss']):.6f}",
        )

        # Contrast: the teacher arm's baseline really is a function of `actions`.
        original = reduced_window_cost(
            batch["depot"],
            batch["locs"],
            torch.cat([batch["start_node"][:, None], batch["actions"]], dim=1),
        )
        changed = reduced_window_cost(
            batch["depot"],
            batch["locs"],
            torch.cat([batch["start_node"][:, None], perturbed_batch["actions"]], dim=1),
        )
        check("teacher baseline depends on actions", not torch.allclose(original, changed))


def verify_loss_math() -> None:
    """Synthetic checks on both arms, then one backward pass."""
    print("\n[6/8] loss math and backward")
    env = make_env()
    batch, _ = build_batch(sample_records(2), w=6)
    horizon = batch["num_loc"]

    teacher = make_model(env, baseline="teacher", num_samples=1).eval()
    cost = torch.tensor([5.0, 8.0])
    reward = (-cost)[:, None]
    logp = torch.full((2, 1), -3.0)
    out = teacher.calculate_loss(None, batch, {}, reward, logp, teacher_cost=cost)
    check("teacher: matching cost -> zero advantage loss", abs(float(out["loss"])) < 1e-6)
    worse = teacher.calculate_loss(None, batch, {}, -1.1 * cost[:, None], logp, teacher_cost=cost)
    check(
        "teacher: worse sample -> positive regret",
        bool(worse["regret"] > 0) and bool(worse["advantage_abs"] > 0),
        f"regret={float(worse['regret']):.4f}",
    )

    shared = make_model(env, baseline="shared", num_samples=4).eval()
    reward = torch.full((2, 4), -5.0)
    logp = torch.full((2, 4), -3.0)
    out = shared.calculate_loss(None, batch, {}, reward, logp)
    check("shared: identical samples -> zero loss", abs(float(out["loss"])) < 1e-6)
    varied = -torch.tensor(
        [[5.0, 5.5, 4.5, 6.0], [8.0, 7.5, 8.5, 8.0]], dtype=torch.float32
    )
    out = shared.calculate_loss(None, batch, {}, varied, logp)
    check("shared: varied samples -> finite loss", torch.isfinite(out["loss"]))

    model = make_model(env).train()
    td = env.reset_partial(batch)
    out = model.policy(td, env, phase="train", decode_type="sampling")
    cost = reduced_window_cost(
        batch["depot"],
        batch["locs"],
        torch.cat([batch["start_node"][:, None], batch["actions"]], dim=1),
    )
    out = model.calculate_loss(
        td, batch, out, out["reward"][:, None], out["log_likelihood"][:, None],
        teacher_cost=cost,
    )
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check(
        "backward produces finite gradients",
        len(grads) > 0 and all(torch.isfinite(g).all() for g in grads),
        f"{len(grads)} tensors",
    )


def verify_state_dict_parity() -> None:
    """Same architecture, same validation rewards under shared weights."""
    print("\n[7/8] state dict and validation parity with POMORL")
    env = make_env()
    lehd = make_model(env)
    plain = POMORL(
        env,
        num_augment=1,
        eval_greedy=False,
        policy_kwargs={"num_encoder_layers": 1},
    )
    left = lehd.state_dict()
    right = plain.state_dict()
    check(
        "state dict keys match",
        set(left) == set(right) and all(left[k].shape == right[k].shape for k in left),
        f"{len(left)} tensors",
    )

    plain.load_state_dict(lehd.state_dict(), strict=True)
    dataset = env.dataset(4, phase="val")
    batch = dataset.collate_fn([dataset[i] for i in range(2)])
    lehd.eval()
    plain.eval()
    with torch.no_grad():
        a = lehd._inference_step(batch, "val")
        b = plain._inference_step(batch, "val")
    check(
        "val/reward identical under shared weights",
        torch.allclose(a["reward"], b["reward"], atol=1e-6),
        f"{a['reward'].tolist()} vs {b['reward'].tolist()}",
    )


def verify_dataset_contract() -> None:
    """The RL env serves LEHD partials for training and full trajectories for validation."""
    print("\n[8/8] dataset contract")
    env = make_env()
    train = env.dataset(8, phase="train")
    val = env.dataset(4, phase="val")
    check("train dataset is LEHD partials", type(train).__name__ == "LEHDPartialDataset")
    check("val dataset is full trajectories", type(val).__name__ == "CVRPSolutionDataset")
    item = train[0]
    check(
        "partial item has the RL fields",
        {"start_node", "used_capacity0", "actions", "num_loc"} <= set(item),
    )
    batch = collate_partials([item])
    check("collated partial has one window width", int(batch["num_loc"][0]) == item["num_loc"])
    check(
        "collated demand is raw (reset_partial normalises it)",
        torch.allclose(
            batch["demand"][0], torch.as_tensor(item["demand"], dtype=torch.float32)
        ),
    )


def main() -> int:
    """Run every group and report."""
    if not DATA.exists():
        print(f"missing {DATA}; run from the project root")
        return 1

    verify_cost_helper()
    verify_fixed_start_invariant()
    verify_state_and_reward()
    verify_teacher_rollout()
    verify_shared_rollout()
    verify_loss_math()
    verify_state_dict_parity()
    verify_dataset_contract()

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("PASSED: all LEHD-style RL checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
