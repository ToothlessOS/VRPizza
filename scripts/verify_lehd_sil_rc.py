#!/usr/bin/env python
"""Verify the LEHD backbone and its SIL-RC integration without pytest.

Run from the project root::

    .venv/bin/python scripts/verify_lehd_sil_rc.py

Sections 1-2 compare the port against the reference implementation when its
checkout is present (state-dict equality, first/mid/last-step 2N probabilities,
and the exact teacher-forced CE). The remaining sections check the rl4co
policy contract, the RRC loop, a tiny two-phase fit with the backbone guards,
seed reproducibility and the evaluation registry. The reference checks are
skipped, not failed, when the external checkout is absent.
"""

from __future__ import annotations

import importlib.util
import sys

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightning.pytorch import seed_everything  # noqa: E402
from rl4co.envs.routing.cvrp.generator import CVRPGenerator  # noqa: E402
from rl4co.utils.trainer import RL4COTrainer  # noqa: E402
from tensordict.tensordict import TensorDict  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.envs import LEHDEvalCVRPEnv, SILRCCVRPEnv  # noqa: E402
from src.lehd.subpath import flatten_routes  # noqa: E402
from src.models import (  # noqa: E402
    LEHDPolicy,
    PhaseAccumulationCallback,
    SILRCLEHDLitModule,
)
from src.models.pomo_sl import _capture_logprobs  # noqa: E402
from src.rrc import RRCConfig, run_rrc_from_routes  # noqa: E402
from src.sil_rc import (  # noqa: E402
    LabelStore,
    SILPartialDataset,
    generate_instances,
    initial_labels,
)

DATA = Path("data/CVRP/testing dataset/vrp100_test_lkh.txt")
TMP = Path("/tmp/verify_lehd_sil_rc")
REFERENCE_ROOT = Path(
    "/home/toothlessos/Projects/nrp/CIAM-Group NCO_code main single_objective-LEHD"
)
REFERENCE_MODEL = REFERENCE_ROOT / "CVRP/VRPModel.py"
REFERENCE_CKPT = (
    REFERENCE_ROOT / "CVRP/result/20230817_235537_train/checkpoint-40.pt"
)
REFERENCE_PARAMS = {
    "mode": "test",
    "embedding_dim": 128,
    "decoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "ff_hidden_dim": 512,
    "sqrt_embedding_dim": 128**0.5,
}
EXPECTED_PARAMS = 1_415_682
EXPECTED_KEYS = 71

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record and report the outcome of a single assertion."""
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")
    if not condition:
        failures.append(name)


def load_reference():
    """Import the reference model and checkpoint, or return ``None``."""
    if not REFERENCE_MODEL.is_file() or not REFERENCE_CKPT.is_file():
        return None
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("ref_vrpmodel", REFERENCE_MODEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.VRPModel(**REFERENCE_PARAMS).eval()
    checkpoint = torch.load(REFERENCE_CKPT, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model, state


def synthetic_td(env, count: int = 2, seed: int = 0) -> TensorDict:
    """A reset ``TensorDict`` over a generated instance batch."""
    torch.manual_seed(seed)
    generator = CVRPGenerator(num_loc=env.generator.num_loc, capacity=30.0)
    raw = generator(count)
    batch = TensorDict(
        {key: raw[key] for key in ("depot", "locs", "demand")}
        | {"capacity": raw["capacity"].reshape(-1)},
        batch_size=[count],
    )
    return env.reset(env.make_input_td(batch))


def reference_problems(
    td: TensorDict, capacity: float, remaining: torch.Tensor
) -> torch.Tensor:
    """The reference's ``state.problems`` ``[B, N+1, 4]`` for the current state."""
    size = int(td.batch_size[0])
    n = int(td["demand"].shape[-1])
    raw_demand = td["demand"] * capacity
    return torch.cat(
        [
            td["locs"],
            torch.cat([torch.zeros(size, 1), raw_demand], -1)[..., None],
            remaining.reshape(size, 1).expand(size, n + 1)[..., None],
        ],
        dim=-1,
    )


def reference_decoder_probs(reference, encoded, selected, capacity, remaining):
    """The reference decoder's exact 2N probabilities for one state."""
    return reference.decoder(encoded, selected, capacity, remaining)


def compare_unselected(
    ours: dict, reference_probs: torch.Tensor, selected: list[int]
) -> float:
    """Max absolute difference over the unselected 2N entries."""
    n = ours["direct"].shape[-1]
    mask = torch.ones(n, dtype=torch.bool)
    for customer in selected:
        mask[customer - 1] = False
    direct = (ours["direct"][:, mask] - reference_probs[:, :n][:, mask]).abs().max()
    via = (ours["via"][:, mask] - reference_probs[:, n:][:, mask]).abs().max()
    return float(torch.maximum(direct, via))


def verify_architecture() -> None:
    """Parameter/state-dict equality and 2N probability parity."""
    print("\n[1/7] reference architecture and 2N parity")
    policy = LEHDPolicy().eval()
    check(
        "parameter count is the reference's 1,415,682",
        sum(p.numel() for p in policy.parameters()) == EXPECTED_PARAMS,
        str(sum(p.numel() for p in policy.parameters())),
    )
    check(
        "state dict has the reference's 71 tensors",
        len(policy.state_dict()) == EXPECTED_KEYS,
        str(len(policy.state_dict())),
    )

    loaded = load_reference()
    if loaded is None:
        print("  SKIP  reference checkout/checkpoint not available")
        return
    reference, state = loaded
    missing, unexpected = policy.load_state_dict(state, strict=True)
    check(
        "the reference checkpoint loads strictly into LEHDPolicy",
        len(missing) == 0 and len(unexpected) == 0,
        f"missing {len(missing)}, unexpected {len(unexpected)}",
    )
    check(
        "state-dict keys match the reference exactly",
        set(policy.state_dict()) == set(state),
        f"{len(set(state) - set(policy.state_dict()))} missing",
    )

    env = LEHDEvalCVRPEnv(num_loc=20, capacity=30.0)
    td = synthetic_td(env, count=2, seed=0)
    capacity = 30.0
    with torch.no_grad():
        ours = policy.reference_probs(td)
        reference_problems_td = reference_problems(
            td, capacity, torch.full((2,), capacity)
        )
        encoded = reference.encoder(reference_problems_td, capacity)
        reference_probs = reference_decoder_probs(
            reference,
            encoded,
            torch.zeros(2, 0, dtype=torch.long),
            capacity,
            torch.full((2,), capacity),
        )
    diff = compare_unselected(ours, reference_probs, [])
    check("first-step 2N probabilities match the reference", diff < 1e-4, f"max {diff:.2e}")

    # After one direct customer and after one via customer (same post-step state).
    for customer in (1, 2):
        state_td = td.clone()
        state_td["visited"][:, customer] = 1
        state_td["current_node"] = torch.full((2, 1), customer, dtype=torch.long)
        state_td["used_capacity"] = td["demand"][:, customer - 1 : customer]
        remaining = capacity - (td["demand"][:, customer - 1] * capacity)
        with torch.no_grad():
            ours = policy.reference_probs(state_td)
            problems = reference_problems(td, capacity, remaining)
            encoded = reference.encoder(problems, capacity)
            reference_probs = reference_decoder_probs(
                reference,
                encoded,
                torch.full((2, 1), customer, dtype=torch.long),
                capacity,
                remaining,
            )
        diff = compare_unselected(ours, reference_probs, [customer])
        check(
            f"post-customer-{customer} 2N probabilities match",
            diff < 1e-4,
            f"max {diff:.2e}",
        )

    # Last-customer state: one unselected customer remains.
    last = td.clone()
    last["visited"][:, 1:] = 1
    last["visited"][:, 20] = 0
    last["current_node"] = torch.full((2, 1), 19, dtype=torch.long)
    last["used_capacity"] = td["demand"][:, 18:19] * 0.0 + 0.01
    with torch.no_grad():
        ours = policy.reference_probs(last)
        remaining = capacity - last["used_capacity"].reshape(-1) * capacity
        problems = reference_problems(td, capacity, remaining)
        encoded = reference.encoder(problems, capacity)
        selected = torch.arange(1, 20).reshape(1, -1).repeat(2, 1)
        reference_probs = reference_decoder_probs(
            reference, encoded, selected, capacity, remaining
        )
    diff = compare_unselected(ours, reference_probs, list(range(1, 20)))
    check("last-customer 2N probabilities match", diff < 1e-4, f"max {diff:.2e}")

    # The faithful 2N greedy must reproduce the reference tester's loop exactly.
    from src.rrc.runner import cost_from_flags_torch

    rollout_td = synthetic_td(env, count=3, seed=2)
    costs, nodes, flags = policy.reference_rollout(rollout_td)
    size = int(rollout_td.batch_size[0])
    n = int(rollout_td["demand"].shape[-1])
    raw_demand = rollout_td["demand"] * capacity
    problems = reference_problems(
        rollout_td, capacity, torch.full((size,), capacity)
    )
    selected = torch.zeros(size, 0, dtype=torch.long)
    ref_nodes: list[torch.Tensor] = []
    ref_flags: list[torch.Tensor] = []
    with torch.no_grad():
        for step in range(n):
            state = SimpleNamespace(problems=problems)
            _, _, node, flag_teacher, _ = reference(
                state, selected, None, step, raw_data_capacity=torch.tensor([capacity])
            )
            flag = flag_teacher.clone()
            if step == 0:
                flag[:] = 1
            demands = problems.gather(
                1, node[:, None, None].expand(size, 1, 4)
            ).squeeze(1)[:, 2]
            smaller = problems[:, 0, 3] < demands
            flag[smaller] = 1
            problems[flag == 1, :, 3] = capacity
            problems[:, :, 3] = problems[:, :, 3] - demands[:, None]
            selected = torch.cat([selected, node[:, None]], 1)
            ref_nodes.append(node.clone())
            ref_flags.append(flag.clone())
    ref_nodes_t = torch.stack(ref_nodes, 1)
    ref_flags_t = torch.stack(ref_flags, 1)
    ref_costs = cost_from_flags_torch(
        ref_nodes_t, ref_flags_t.bool(), rollout_td["depot"], rollout_td["locs"][:, 1:]
    )
    check(
        "the faithful greedy reproduces the reference tester's nodes and flags",
        torch.equal(nodes, ref_nodes_t) and torch.equal(flags, ref_flags_t.int()),
    )
    check(
        "the faithful greedy reproduces the reference tester's cost",
        torch.allclose(costs, ref_costs, atol=1e-6),
        f"max diff {(costs - ref_costs).abs().max():.2e}",
    )


def verify_marginal() -> None:
    """The N+1 logits are exactly the 2N distribution's next-action marginal."""
    print("\n[2/7] 2N -> N+1 marginalisation")
    torch.manual_seed(0)
    direct = torch.randn(3, 7)
    via = torch.randn(3, 7)
    unselected = torch.ones(3, 7, dtype=torch.bool)
    unselected[0, 0] = False
    unselected[1, 3] = False
    neg_inf = float("-inf")
    two_n = torch.softmax(
        torch.cat([direct.masked_fill(~unselected, neg_inf),
                   via.masked_fill(~unselected, neg_inf)], -1), -1
    )
    log_via = torch.logsumexp(via.masked_fill(~unselected, neg_inf), -1)
    n_plus_one = torch.softmax(
        torch.cat([log_via[:, None], direct.masked_fill(~unselected, neg_inf)], -1), -1
    )
    check(
        "P(depot) equals total via mass",
        torch.allclose(n_plus_one[:, 0], two_n[:, 7:].sum(-1), atol=1e-6),
    )
    check(
        "P(customer) equals the direct entry",
        torch.allclose(n_plus_one[:, 1:], two_n[:, :7], atol=1e-6),
    )


def verify_policy_api() -> None:
    """Greedy, multistart, evaluate and the exact CE against the reference loss."""
    print("\n[3/7] rl4co policy contract and exact CE")
    torch.manual_seed(0)
    env = LEHDEvalCVRPEnv(num_loc=10, capacity=30.0)
    policy = LEHDPolicy(use_checkpointing=False).eval()

    with torch.no_grad():
        greedy = policy(
            synthetic_td(env, count=3, seed=1),
            env,
            phase="test",
            decode_type="greedy",
        )
    check(
        "greedy rollout returns [B] rewards and actions",
        tuple(greedy["reward"].shape) == (3,)
        and greedy["actions"].shape[0] == 3
        and torch.isfinite(greedy["reward"]).all(),
    )

    multi_td = synthetic_td(env, count=3, seed=1)
    starts = env.select_start_nodes(multi_td, num_starts=4)
    with torch.no_grad():
        multi = policy(multi_td, env, phase="val", num_starts=4)
    check(
        "multistart returns [B*S] rewards and the forced first actions",
        tuple(multi["reward"].shape) == (12,)
        and torch.equal(multi["actions"][:, 0], starts),
        f"first actions {multi['actions'][:, 0].tolist()}",
    )

    captured: dict = {}
    with torch.no_grad():
        with _capture_logprobs(captured):
            evaluate = policy(
                synthetic_td(env, count=3, seed=1),
                env,
                phase="train",
                actions=greedy["actions"],
                store_all_logp=True,
                return_sum_log_likelihood=False,
            )
    check(
        "teacher-forced evaluate returns finite [B, T] log-likelihoods",
        tuple(evaluate["log_likelihood"].shape) == (3, greedy["actions"].shape[1])
        and torch.isfinite(evaluate["log_likelihood"]).all(),
    )
    check(
        "the captured distribution is [B, T, N+1]",
        captured.get("logprobs") is not None
        and tuple(captured["logprobs"].shape)
        == (3, greedy["actions"].shape[1], 11),
        str(None if captured.get("logprobs") is None else captured["logprobs"].shape),
    )

    # A finished row in a mixed batch: the empty-unselected fallback must not NaN.
    mixed = synthetic_td(env, count=3, seed=1)
    mixed["visited"][0, 1:] = 1
    mixed["done"][0] = 1
    mixed["action_mask"] = env.get_action_mask(mixed)
    with torch.no_grad():
        out = policy(mixed, env, phase="test", decode_type="greedy")
    check(
        "a done row keeps the depot logit finite",
        torch.isfinite(out["reward"]).all() and torch.equal(out["actions"][0, 0], torch.tensor(0)),
        f"actions {out['actions'][0, :3].tolist()}",
    )

    mixed_sample = synthetic_td(env, count=3, seed=1)
    mixed_sample["visited"][0, 1:] = 1
    mixed_sample["done"][0] = 1
    mixed_sample["action_mask"] = env.get_action_mask(mixed_sample)
    mixed_sample["action_mask"][0, 0] = False  # a fully masked done row
    with torch.no_grad():
        sampled = policy(
            mixed_sample, env, phase="test", decode_type="sampling", num_starts=1
        )
    check(
        "a done row is finite under sampling",
        torch.isfinite(sampled["reward"]).all()
        and torch.equal(sampled["actions"][0, 0], torch.tensor(0)),
        f"actions {sampled['actions'][0, :3].tolist()}",
    )

    costs, nodes, flags = policy.reference_rollout(synthetic_td(env, count=3, seed=1))
    check(
        "the reference greedy rollout covers every customer",
        tuple(nodes.shape) == (3, 10)
        and all(
            set(nodes[row].tolist()) == set(range(1, 11)) for row in range(3)
        )
        and torch.isfinite(costs).all(),
    )

    # Exact CE vs. the reference per-step loss on the same window.
    loaded = load_reference()
    if loaded is None:
        print("  SKIP  exact-CE comparison (reference checkout not available)")
        return
    reference, state = loaded
    reference.mode = "train"  # the reference's forward branches on this flag
    policy.load_state_dict(state, strict=True)

    store = generate_instances(
        CVRPGenerator(num_loc=10, capacity=30.0), count=1, seed=0
    )
    small_env = LEHDEvalCVRPEnv(num_loc=10, capacity=30.0)
    module = SILRCLEHDLitModule(
        small_env,
        num_augment=1,
        eval_greedy=False,
        gradient_checkpointing=False,
    )
    module.policy.load_state_dict(state, strict=True)
    initial_labels(store, module, small_env, init="random_insertion", seed=0, batch_size=1)
    dataset = SILPartialDataset(
        store,
        partials_per_instance=1,
        effective_batch_size=1,
        w_min=6,
        w_max=6,
        seed=1,
    )
    batch = next(
        iter(DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn))
    )
    out = module._sil_step(batch)
    our_ce = float(out["ce_loss"].detach())

    done_dataset = SILPartialDataset(
        store,
        partials_per_instance=1,
        effective_batch_size=1,
        w_min=10,
        w_max=10,
        seed=2,
    )
    done_batch = next(
        iter(
            DataLoader(
                done_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=done_dataset.collate_fn,
            )
        )
    )
    done_out = module._sil_step(done_batch)
    check(
        "a full-route window gives a finite exact CE",
        torch.isfinite(done_out["loss"]).all(),
        f"loss {float(done_out['loss'].detach()):.4f}",
    )

    width = int(batch["num_loc"][0])
    capacity = float(batch["capacity"][0])
    remaining = (batch["capacity"] - batch["used_capacity0"]).reshape(1)
    problems = reference_problems(
        TensorDict(
            {
                "depot": batch["depot"],
                "locs": torch.cat([batch["depot"][:, None, :], batch["locs"]], dim=1),
                "demand": batch["demand"] / capacity,
            },
            batch_size=[1],
        ),
        capacity,
        remaining,
    )
    pairs = [(int(batch["start_node"][0]), 0)]
    pending = False
    for action in batch["actions"][0].tolist():
        if action == 0:
            pending = True
        else:
            pairs.append((action, 1 if pending else 0))
            pending = False
    solution = torch.tensor([pairs], dtype=torch.long)
    selected = torch.tensor([[pairs[0][0]]], dtype=torch.long)
    loss_total = 0.0
    with torch.no_grad():
        for step in range(1, width):
            state = SimpleNamespace(problems=problems)
            loss_node, *_ = reference(
                state, selected, solution, step, raw_data_capacity=batch["capacity"]
            )
            loss_total += float(loss_node)
            node, flag = pairs[step]
            if flag == 1:
                problems[:, :, 3] = capacity
            demand = torch.cat([torch.zeros(1, 1), batch["demand"]], -1).gather(
                1, torch.tensor([[node]])
            )
            if (problems[:, 0, 3] < demand.squeeze(1)).any():
                problems[:, :, 3] = capacity
            problems[:, :, 3] = problems[:, :, 3] - demand
            selected = torch.cat([selected, torch.tensor([[node]])], 1)
    reference_ce = loss_total / (width - 1)
    check(
        "the exact CE matches the reference per-step loss",
        abs(our_ce - reference_ce) < 1e-4,
        f"ours {our_ce:.6f}, reference {reference_ce:.6f}",
    )


def verify_rrc() -> None:
    """RRC from random-insertion labels is monotone and feasible with the LEHD policy."""
    print("\n[4/7] RRC reconstruction")
    torch.manual_seed(0)
    env = LEHDEvalCVRPEnv(num_loc=10, capacity=30.0)
    model = SILRCLEHDLitModule(
        env,
        num_augment=1,
        eval_greedy=False,
        gradient_checkpointing=False,
    )
    store = generate_instances(CVRPGenerator(num_loc=10, capacity=30.0), count=4, seed=3)
    initial_labels(store, model, env, init="random_insertion", seed=3, batch_size=4)
    batch = store.batch(np.arange(4))
    routes = [store.routes_for(index) for index in range(4)]
    config = RRCConfig(budgets=(2,), seed=0, init="greedy", num_augment=1, rrc_num_augment=1)
    result = run_rrc_from_routes(model, env, batch, routes, config, positions=np.arange(4))
    monotone = bool((result.budget_costs[:, 0] <= result.initial_cost + 1e-9).all())
    feasible = True
    for row in range(4):
        customers, _ = flatten_routes(result.final_routes[row])
        feasible &= bool(
            np.array_equal(np.sort(customers), np.arange(1, 11))
        )
    check("RRC costs never increase", monotone)
    check("RRC solutions cover every customer", feasible)


def verify_guards() -> None:
    """Sidecar/checkpoint backbone guards and a tiny two-phase fit."""
    print("\n[5/7] backbone guards and two-phase fit")
    seed_everything(0, workers=True)
    TMP.mkdir(parents=True, exist_ok=True)
    cache = TMP / "labels.npz"
    if cache.exists():
        cache.unlink()

    store = generate_instances(CVRPGenerator(num_loc=10, capacity=30.0), count=3, seed=0)
    store.meta.update({"backbone": "pomo", "policy": "pomo-am-e6-h8-ff512"})
    store.save(cache)
    try:
        LabelStore.load(cache, expected={"backbone": "lehd"})
        rejected = False
    except ValueError:
        rejected = True
    check("a POMO label sidecar is refused by a LEHD run", rejected)

    small_env = LEHDEvalCVRPEnv(num_loc=10, capacity=30.0)
    module = SILRCLEHDLitModule(
        small_env,
        num_augment=1,
        eval_greedy=False,
        gradient_checkpointing=False,
    )
    try:
        module.on_load_checkpoint({"sil_rc_state": {"backbone": "pomo"}})
        checkpoint_rejected = False
    except ValueError:
        checkpoint_rejected = True
    check("a POMO checkpoint is refused by the LEHD module", checkpoint_rejected)

    fit_env = SILRCCVRPEnv(
        data_path=str(DATA),
        num_loc=100,
        capacity=50.0,
        val_instances=10,
        seed=0,
        warmup_num_loc=20,
        warmup_capacity=30.0,
    )
    fit_model = SILRCLEHDLitModule(
        fit_env,
        warmup_epochs=1,
        warmup_batch_size=4,
        warmup_data_size=8,
        cycle_epochs=1,
        labels_per_cycle=6,
        fresh_labels_per_cycle=False,
        init="greedy",
        improve_first_cycle=True,
        recon_iterations=1,
        repair_batch=3,
        partials_per_instance=1,
        sil_batch_size=2,
        sil_effective_batch_size=6,
        lr_gamma=0.97,
        label_seed=0,
        label_cache_path=str(cache),
        batch_size=4,
        train_data_size=8,
        val_batch_size=2,
        val_data_size=4,
        num_augment=1,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )
    trainer = RL4COTrainer(
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        callbacks=[PhaseAccumulationCallback(1, 1, 3)],
        num_sanity_val_steps=0,
        precision="32-true",
    )
    trainer.fit(fit_model)
    check(
        "the two-phase fit creates labels and a best snapshot",
        fit_model._labels is not None
        and fit_model._labels_cycle == 0
        and cache.is_file()
        and fit_model._best_policy_state is not None,
        f"cycle {fit_model._labels_cycle}, best {fit_model._best_metric}",
    )
    check(
        "the exact CE metrics were logged",
        fit_model._label_metrics is not None
        and {"label_mean_cost", "recon_iterations", "labels"} <= set(fit_model._label_metrics),
    )

    # The chunk schedule is part of both guards: a whole-store sidecar or
    # checkpoint cannot silently drive an S=2 run.
    try:
        LabelStore.load(cache, expected={"subcycles_per_cycle": 2})
        sidecar_schedule_rejected = False
    except ValueError:
        sidecar_schedule_rejected = True
    check("an S=1 sidecar is refused by an S=2 run", sidecar_schedule_rejected)

    checkpoint_path = TMP / "s1.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    chunk_cache = TMP / "chunked_labels.npz"
    if chunk_cache.exists():
        chunk_cache.unlink()
    chunked_env = SILRCCVRPEnv(
        data_path=str(DATA),
        num_loc=100,
        capacity=50.0,
        val_instances=10,
        seed=0,
        warmup_num_loc=20,
        warmup_capacity=30.0,
    )
    chunked_model = SILRCLEHDLitModule(
        chunked_env,
        warmup_epochs=1,
        warmup_batch_size=4,
        warmup_data_size=8,
        cycle_epochs=1,
        labels_per_cycle=6,
        subcycles_per_cycle=2,
        subcycle_epochs=1,
        fresh_labels_per_cycle=False,
        init="greedy",
        improve_first_cycle=True,
        recon_iterations=1,
        repair_batch=3,
        partials_per_instance=1,
        sil_batch_size=2,
        sil_effective_batch_size=6,
        lr_gamma=0.97,
        label_seed=0,
        label_cache_path=str(chunk_cache),
        batch_size=4,
        train_data_size=8,
        val_batch_size=2,
        val_data_size=4,
        num_augment=1,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )
    try:
        chunked_model.on_load_checkpoint(
            torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        )
        checkpoint_schedule_rejected = False
    except ValueError:
        checkpoint_schedule_rejected = True
    check(
        "an S=1 checkpoint is refused by an S=2 module",
        checkpoint_schedule_rejected,
    )

    chunked_trainer = RL4COTrainer(
        max_epochs=3,
        accelerator="cpu",
        devices=1,
        logger=False,
        callbacks=[PhaseAccumulationCallback(1, 1, 3)],
        num_sanity_val_steps=0,
        precision="32-true",
    )
    chunked_trainer.fit(chunked_model)
    check(
        "the chunked LEHD fit advances one sub-cycle per epoch",
        chunked_model._labels is not None
        and chunked_model._labels_cycle == 0
        and chunked_model._labels_step == 1
        and chunked_model._labels.meta.get("subcycles_per_cycle") == 2
        and chunk_cache.is_file(),
        f"cycle {chunked_model._labels_cycle}, step {chunked_model._labels_step}",
    )
    check(
        "the chunked LEHD exact-CE metrics report the chunk",
        chunked_model._label_metrics is not None
        and chunked_model._label_metrics.get("labels") == 3,
        str(chunked_model._label_metrics),
    )


def verify_seed_reproducibility() -> None:
    """Re-seeding reproduces the policy and the generated instances."""
    print("\n[6/7] seed reproducibility")
    seed_everything(7, workers=True)
    first = LEHDPolicy()
    seed_everything(7, workers=True)
    second = LEHDPolicy()
    same = all(
        torch.equal(a, b)
        for a, b in zip(first.state_dict().values(), second.state_dict().values())
    )
    check("re-seeding reproduces every policy parameter", same)

    generator = CVRPGenerator(num_loc=10, capacity=30.0)
    store_a = generate_instances(generator, count=5, seed=11, block_size=2)
    store_b = generate_instances(generator, count=5, seed=11, block_size=2)
    check(
        "the same label seed reproduces the instance set",
        np.array_equal(store_a.locs, store_b.locs)
        and np.array_equal(store_a.demand, store_b.demand),
    )


def verify_eval_registry() -> None:
    """The eval driver resolves both LEHD run labels to the LEHD module."""
    print("\n[7/7] evaluation registry")
    spec = importlib.util.spec_from_file_location(
        "eval_pomo", Path("notebooks/eval_pomo.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    sys.modules["eval_pomo"] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    from src.models import SILRCLEHDLitModule as ModuleClass

    check(
        "lehd_sil_rc resolves to the LEHD module",
        module.model_class_for("lehd_sil_rc") is ModuleClass,
    )
    check(
        "lehd_sil_rc_step resolves to the LEHD module",
        module.model_class_for("lehd_sil_rc_step") is ModuleClass,
    )
    check(
        "the existing lehd label still resolves to LEHDPOMOSL",
        module.model_class_for("lehd") is not ModuleClass,
    )

    env = LEHDEvalCVRPEnv(num_loc=10, capacity=30.0)
    first = SILRCLEHDLitModule(
        env, num_augment=1, eval_greedy=False, gradient_checkpointing=False
    )
    second = SILRCLEHDLitModule(
        env, num_augment=1, eval_greedy=False, gradient_checkpointing=False
    )
    missing, unexpected = second.load_state_dict(first.state_dict(), strict=True)
    check(
        "a strict state-dict round-trip uses the same keys",
        len(missing) == 0 and len(unexpected) == 0,
    )


def main() -> int:
    """Run every verification group and report a summary."""
    verify_architecture()
    verify_marginal()
    verify_policy_api()
    verify_rrc()
    verify_guards()
    verify_seed_reproducibility()
    verify_eval_registry()

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
