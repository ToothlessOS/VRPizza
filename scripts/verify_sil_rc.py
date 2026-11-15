#!/usr/bin/env python
"""Verify local-reconstruction SIL without pytest.

Run from the project root::

    .venv/bin/python scripts/verify_sil_rc.py

The checks cover the pieces the pipeline cannot afford to get wrong: the NumPy
insertion port, the seeded instance/label store, the in-memory partial dataset
against the ``sample_window`` oracle, the route-in/route-out RRC loop, one full
label-repair pass, the step protocol constants, and a tiny two-epoch fit followed
by a checkpoint resume that must reuse the label sidecar. Everything runs on tiny
generated instances and a randomly initialised POMO, so the suite stays in tens
of seconds and does not depend on a checkpoint.
"""

from __future__ import annotations

import sys

from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightning.pytorch import Trainer, seed_everything  # noqa: E402
from rl4co.envs.routing.cvrp.generator import CVRPGenerator  # noqa: E402
from rl4co.utils.trainer import RL4COTrainer  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.envs import LEHDEvalCVRPEnv, SILRCCVRPEnv, SILRCStepCVRPEnv  # noqa: E402
from src.lehd.subpath import (  # noqa: E402
    cost_from_flags,
    flatten_routes,
    sample_window,
)
from src.models import (  # noqa: E402
    POMOSL,
    PhaseAccumulationCallback,
    SILRCPOMOLitModule,
)
from src.rrc import RRCConfig, run_rrc_from_routes  # noqa: E402
from src.sil_rc import (  # noqa: E402
    LabelStore,
    SILPartialDataset,
    generate_instances,
    improve_labels,
    initial_labels,
    random_insertion_routes,
    validate_routes,
)
from src.utils import comparison as C  # noqa: E402

DATA = Path("data/CVRP/testing dataset/vrp100_test_lkh.txt")
TMP = Path("/tmp/verify_sil_rc")
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record and report the outcome of a single assertion."""
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")
    if not condition:
        failures.append(name)


def route_cost(route_list, depot: np.ndarray, locs: np.ndarray) -> float:
    """Exact full-tour length of one route list in 1-based customer ids."""
    customers, flags = flatten_routes(route_list)
    return cost_from_flags(customers, flags, depot, locs)


def coverage_ok(route_list, n: int, demand: np.ndarray, capacity: float) -> tuple[bool, bool, str]:
    """Return (coverage, capacity) for one route list."""
    flat = np.concatenate(route_list) if route_list else np.empty(0, dtype=np.int64)
    covered = np.array_equal(np.sort(flat), np.arange(1, n + 1))
    loads = [float(demand[np.asarray(route, dtype=np.int64) - 1].sum()) for route in route_list]
    within = max(loads, default=0.0) <= capacity + 1e-5
    return covered, within, f"max load {max(loads, default=0.0):.3f} / {capacity:.1f}"


def make_generator(num_loc: int = 20) -> CVRPGenerator:
    """A small CVRP generator for the core checks."""
    return CVRPGenerator(num_loc=num_loc, capacity=30.0)


def make_env_model(num_loc: int = 20):
    """A dataset-free CVRP env and a randomly initialised POMO on it."""
    env = LEHDEvalCVRPEnv(num_loc=num_loc, capacity=30.0)
    model = POMOSL(env, num_augment=1, eval_greedy=False).eval()
    return env, model


def verify_insertion() -> None:
    """GLOP-style insertion: feasibility, coverage, determinism and order seeding."""
    print("\n[1/8] random-insertion constructor")
    gen = make_generator()
    store = generate_instances(gen, count=5, seed=11, block_size=2)
    for row in range(len(store)):
        routes = random_insertion_routes(
            store.depot[row],
            store.locs[row],
            store.demand[row],
            float(store.capacity[row]),
        )
        validate_routes(routes, store.demand[row], float(store.capacity[row]))
        covered = np.array_equal(
            np.sort(np.concatenate(routes)), np.arange(store.num_loc)
        )
        loads = [
            float(store.demand[row][route].sum()) for route in routes
        ]
        check(
            f"row {row} covers all customers and respects capacity",
            covered and max(loads) <= float(store.capacity[row]) + 1e-9,
            f"{len(routes)} routes, max load {max(loads):.2f}",
        )
        again = random_insertion_routes(
            store.depot[row],
            store.locs[row],
            store.demand[row],
            float(store.capacity[row]),
        )
        check(
            f"row {row} angle order is deterministic",
            all(np.array_equal(a, b) for a, b in zip(routes, again)),
        )

    rng_a = np.random.default_rng(3)
    rng_b = np.random.default_rng(3)
    random_a = random_insertion_routes(
        store.depot[0], store.locs[0], store.demand[0], float(store.capacity[0]),
        order="random", rng=rng_a,
    )
    random_b = random_insertion_routes(
        store.depot[0], store.locs[0], store.demand[0], float(store.capacity[0]),
        order="random", rng=rng_b,
    )
    check(
        "random order is reproducible from the generator",
        all(np.array_equal(a, b) for a, b in zip(random_a, random_b)),
    )


def verify_generation() -> None:
    """Instance generation is seeded, integral and block-keyed."""
    print("\n[2/8] seeded instance generation")
    gen = make_generator()
    a = generate_instances(gen, count=7, seed=13, block_size=3)
    b = generate_instances(gen, count=7, seed=13, block_size=3)
    c = generate_instances(gen, count=7, seed=14, block_size=3)
    check(
        "same seed reproduces depot/locs/demand/capacity",
        np.array_equal(a.depot, b.depot)
        and np.array_equal(a.locs, b.locs)
        and np.array_equal(a.demand, b.demand)
        and np.array_equal(a.capacity, b.capacity),
    )
    check("a different seed changes the instances", not np.array_equal(a.locs, c.locs))
    check(
        "demands are integral and inside the generator range",
        np.allclose(a.demand, np.round(a.demand))
        and float(a.demand.min()) >= 0
        and float(a.demand.max()) <= float(a.capacity.max()) + 1e-6,
        f"range [{float(a.demand.min()):.1f}, {float(a.demand.max()):.1f}]",
    )
    check(
        "generated capacity matches the SIL full-size capacity",
        np.allclose(a.capacity, np.float32(30.0)),
        f"{float(a.capacity[0]):.3f}",
    )


def verify_labels(env, model) -> None:
    """Both label bootstraps are feasible; the store round-trips and validates."""
    print("\n[3/8] initial labels and label-store persistence")
    gen = make_generator()
    ri = generate_instances(gen, count=6, seed=123, block_size=3)
    info = initial_labels(ri, model, env, init="random_insertion", seed=123)
    check(
        "random-insertion labels are feasible",
        all(
            coverage_ok(
                ri.routes_for(row),
                ri.num_loc,
                ri.demand[row],
                float(ri.capacity[row]),
            )[:2]
            == (True, True)
            for row in range(len(ri))
        ),
        f"mean {info['label_mean_cost']:.4f}",
    )
    replay = np.array(
        [
            route_cost(ri.routes_for(row), ri.depot[row], ri.locs[row])
            for row in range(len(ri))
        ]
    )
    check(
        "stored costs replay exactly",
        np.allclose(replay, ri.cost, rtol=1e-5, atol=1e-5),
        f"max |diff| {np.abs(replay - ri.cost).max():.2e}",
    )

    greedy = generate_instances(gen, count=4, seed=7, block_size=2)
    initial_labels(greedy, model, env, init="greedy", seed=7, batch_size=2)
    check(
        "greedy labels cover every customer",
        all(
            coverage_ok(
                greedy.routes_for(row),
                greedy.num_loc,
                greedy.demand[row],
                float(greedy.capacity[row]),
            )[0]
            for row in range(len(greedy))
        ),
        f"mean {greedy.mean_cost:.4f}",
    )

    TMP.mkdir(parents=True, exist_ok=True)
    path = TMP / "labels.npz"
    ri.meta["cycle"] = 0
    ri.save(path)
    loaded = LabelStore.load(
        path, expected={"num_loc": ri.num_loc, "init": "random_insertion", "count": len(ri)}
    )
    check(
        "label store round-trips",
        np.array_equal(loaded.tour, ri.tour)
        and np.array_equal(loaded.flags, ri.flags)
        and np.array_equal(loaded.cost, ri.cost)
        and loaded.meta["cycle"] == 0,
    )
    try:
        LabelStore.load(path, expected={"init": "greedy"})
    except ValueError:
        check("mismatched metadata is rejected", True)
    else:
        check("mismatched metadata is rejected", False)


def verify_dataset(env, model) -> None:
    """The partial dataset matches the sample_window oracle and is seeded per item."""
    print("\n[4/8] in-memory partial dataset")
    gen = make_generator()
    store = generate_instances(gen, count=5, seed=21, block_size=3)
    initial_labels(store, model, env, init="random_insertion", seed=21)
    seed = 99
    micro, accum = 2, 2
    effective = micro * accum
    k = 2
    dataset = SILPartialDataset(
        store,
        partials_per_instance=k,
        effective_batch_size=effective,
        w_min=4,
        w_max=20,
        seed=seed,
    )
    oracle_ok = True
    for index in range(len(dataset)):
        item = dataset[index]
        position, window_index = divmod(index, k)
        label_index = int(dataset.order[position])
        chunk = index // effective
        w = int(dataset.w_per_chunk[chunk])
        window = sample_window(
            store.routes_for(label_index),
            store.locs[label_index],
            store.demand[label_index],
            w,
            np.random.default_rng([seed, label_index, window_index]),
        )
        oracle_ok &= (
            np.allclose(item["locs"], window.locs_w)
            and np.allclose(item["demand"], window.demand_w)
            and np.array_equal(item["actions"], window.actions)
            and int(item["start_node"]) == window.start_local
            and np.isclose(item["used_capacity0"], window.used_capacity0)
            and int(item["num_loc"]) == w
        )
    check("every item matches the sample_window oracle", oracle_ok)

    same = SILPartialDataset(
        store,
        partials_per_instance=k,
        effective_batch_size=effective,
        w_min=4,
        w_max=20,
        seed=seed,
    )
    check(
        "the same seed reproduces every item",
        all(
            np.array_equal(dataset[i]["actions"], same[i]["actions"])
            and np.array_equal(dataset[i]["locs"], same[i]["locs"])
            for i in range(len(dataset))
        ),
    )
    loader = DataLoader(dataset, batch_size=micro, shuffle=False, collate_fn=dataset.collate_fn)
    batch = next(iter(loader))
    check(
        "collate_partials accepts a micro-batch and pads the target",
        tuple(batch["locs"].shape[:2]) == (micro, int(dataset.w_per_chunk[0]))
        and tuple(batch["actions"].shape[:1]) == (micro,)
        and bool(batch["valid"].any()),
        f"locs {tuple(batch['locs'].shape)}, actions {tuple(batch['actions'].shape)}",
    )
    chunk_widths = {int(dataset.w_per_chunk[index // effective]) for index in range(micro)}
    check("a micro-batch shares one w", len(chunk_widths) == 1)


def verify_rrc_routes(env, model) -> None:
    """run_rrc_from_routes is monotone, splices exactly, and is chunk-stable."""
    print("\n[5/8] route-in/route-out reconstruction")
    gen = make_generator()
    store = generate_instances(gen, count=4, seed=31, block_size=2)
    initial_labels(store, model, env, init="random_insertion", seed=31)
    positions = np.arange(len(store), dtype=np.int64)
    batch = store.batch(positions)
    routes = store.routes_for(positions)
    config = RRCConfig(
        budgets=(3,), w_min=4, w_max=20, seed=5, init="greedy", num_augment=1,
        rrc_num_augment=1,
    )
    result = run_rrc_from_routes(model, env, batch, routes, config, positions=positions)
    check(
        "reconstruction never increases the exact cost",
        bool(np.isfinite(result.budget_costs).all())
        and bool((result.budget_costs <= result.initial_cost[:, None] + 1e-6).all()),
    )
    replayed = np.array(
        [
            route_cost(result.final_routes[row], store.depot[row], store.locs[row])
            for row in range(len(store))
        ]
    )
    check(
        "returned routes replay to the reported final cost",
        np.allclose(replayed, result.budget_costs[:, -1], rtol=1e-5, atol=1e-5),
        f"max |diff| {np.abs(replayed - result.budget_costs[:, -1]).max():.2e}",
    )
    feasible = True
    for row in range(len(store)):
        covered, within, _detail = coverage_ok(
            result.final_routes[row], store.num_loc, store.demand[row], float(store.capacity[row])
        )
        feasible &= covered and within
    check("returned routes are feasible complete solutions", feasible)

    left = run_rrc_from_routes(
        model, env, batch[:2], routes[:2], config, positions=positions[:2]
    )
    right = run_rrc_from_routes(
        model, env, batch[2:], routes[2:], config, positions=positions[2:]
    )
    combined = np.concatenate([left.budget_costs[:, -1], right.budget_costs[:, -1]])
    check(
        "global positions make the draw independent of the repair chunk size",
        np.allclose(combined, result.budget_costs[:, -1], rtol=1e-5, atol=1e-5),
        f"max |diff| {np.abs(combined - result.budget_costs[:, -1]).max():.2e}",
    )


def verify_improve(env, model) -> None:
    """One full repair pass is monotone and reports the reference counters."""
    print("\n[6/8] full label-repair pass")
    gen = make_generator()
    store = generate_instances(gen, count=6, seed=41, block_size=3)
    initial_labels(store, model, env, init="random_insertion", seed=41)
    before = store.cost.copy()
    stats = improve_labels(
        store,
        model,
        env,
        repair_batch=2,
        iterations=2,
        cycle=0,
        seed=41,
        w_min=4,
        w_max=20,
    )
    check(
        "repair is monotone",
        bool((store.cost <= before + 1e-5).all()),
        f"improved {stats['label_improved']} / {len(store)}",
    )
    check(
        "the pass reports every chunk and round",
        stats["labels"] == len(store)
        and stats["recon_iterations"] == 3 * 2
        and stats["recon_rounds"] == 2
        and stats["recon_accepted"] >= 0,
        f"{stats['recon_iterations']} instance-rounds",
    )
    check(
        "post-repair labels remain feasible",
        all(
            coverage_ok(
                store.routes_for(row),
                store.num_loc,
                store.demand[row],
                float(store.capacity[row]),
            )[:2]
            == (True, True)
            for row in range(len(store))
        ),
    )


def verify_step_math() -> None:
    """The step-matched constants give exactly 100 steps per phase epoch."""
    print("\n[7/8] step protocol")
    try:
        C.check_step_math(
            label="sil_rc-warmup",
            rows_per_epoch=C.SIL_RC_WARMUP_INSTANCES_PER_EPOCH,
            micro_batch=C.SIL_RC_WARMUP_MICRO_INSTANCES,
            accumulate=1,
            samples_per_step=C.SIL_RC_WARMUP_SAMPLES_PER_STEP,
        )
        C.check_step_math(
            label="sil_rc-sil",
            rows_per_epoch=C.SIL_RC_ROWS_PER_EPOCH,
            micro_batch=C.SIL_RC_MICRO_BATCH,
            accumulate=C.SIL_RC_ACCUM,
            samples_per_step=C.SAMPLES_PER_STEP,
        )
        check("warm-up and SIL phases each produce 100 steps/epoch", True)
    except ValueError as error:
        check("warm-up and SIL phases each produce 100 steps/epoch", False, str(error))
    check(
        "the schedule is 200 epochs with five 30-epoch SIL cycles",
        C.SIL_RC_WARMUP_EPOCHS + C.SIL_RC_EPOCHS == C.MAX_EPOCHS
        and C.SIL_RC_CYCLES * C.SIL_RC_CYCLE_EPOCHS == C.SIL_RC_EPOCHS,
        f"{C.SIL_RC_WARMUP_EPOCHS} + {C.SIL_RC_EPOCHS} = {C.MAX_EPOCHS}",
    )


def build_smoke_model(cache: Path, env):
    """The tiny two-phase module used by the final fit and resume checks."""
    return SILRCPOMOLitModule(
        env,
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
    )


def verify_phase_fit_and_resume() -> None:
    """A tiny fit creates/repairs labels; a resume loads the sidecar, not a redraw."""
    print("\n[8/8] two-phase fit, best snapshot and resume")
    seed_everything(0, workers=True)
    TMP.mkdir(parents=True, exist_ok=True)
    cache = TMP / "fit_labels.npz"
    ckpt = TMP / "fit.ckpt"
    for path in (cache, ckpt):
        if path.exists():
            path.unlink()

    env = SILRCCVRPEnv(
        data_path=str(DATA),
        num_loc=100,
        capacity=50.0,
        val_instances=10,
        seed=0,
        warmup_num_loc=20,
        warmup_capacity=30.0,
    )
    model = build_smoke_model(cache, env)
    trainer = RL4COTrainer(
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        callbacks=[PhaseAccumulationCallback(1, 1, 3)],
        num_sanity_val_steps=0,
        precision="32-true",
    )
    trainer.fit(model)
    check(
        "the fit creates a feasible label store and a best snapshot",
        model._labels is not None
        and len(model._labels) == 6
        and model._labels_cycle == 0
        and cache.is_file()
        and model._best_policy_state is not None
        and model._best_metric is not None,
        f"cycle {model._labels_cycle}, best {model._best_metric}",
    )
    check(
        "SIL label diagnostics were logged",
        model._label_metrics is not None
        and {"label_mean_cost", "recon_iterations", "labels"} <= set(model._label_metrics),
    )
    trainer.save_checkpoint(ckpt)

    # The resume must load the sidecar; a redraw would call this guard.
    env2 = SILRCCVRPEnv(
        data_path=str(DATA),
        num_loc=100,
        capacity=50.0,
        val_instances=10,
        seed=0,
        warmup_num_loc=20,
        warmup_capacity=30.0,
    )
    env2.sample_instances = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("labels were regenerated instead of loaded")
    )
    model2 = build_smoke_model(cache, env2)
    trainer2 = RL4COTrainer(
        max_epochs=3,
        accelerator="cpu",
        devices=1,
        logger=False,
        callbacks=[PhaseAccumulationCallback(1, 1, 3)],
        num_sanity_val_steps=0,
        precision="32-true",
    )
    try:
        Trainer.fit(trainer2, model2, ckpt_path=str(ckpt), weights_only=False)
        resumed = model2._labels_cycle == 1 and model2._best_metric is not None
        detail = f"cycle {model2._labels_cycle}, best {model2._best_metric}"
    except AssertionError as error:
        resumed = False
        detail = str(error)
    check("a resume loads the sidecar and advances the label cycle", resumed, detail)

    step_env = SILRCStepCVRPEnv(
        data_path=str(DATA),
        num_loc=100,
        capacity=50.0,
        val_instances=10,
        seed=0,
        starts_per_instance=4,
        warmup_num_loc=20,
        warmup_capacity=30.0,
    )
    warmup = step_env.warmup_dataset(5, 123)
    check(
        "the step env attaches 16-style start nodes to its warm-up batch",
        tuple(warmup[0]["start_nodes"].shape) == (4,)
        and tuple(warmup[0]["locs"].shape) == (20, 2),
        f"starts {tuple(warmup[0]['start_nodes'].shape)}",
    )


def main() -> int:
    """Run every verification group and report a summary."""
    torch.manual_seed(0)
    env, model = make_env_model()
    verify_insertion()
    verify_generation()
    verify_labels(env, model)
    verify_dataset(env, model)
    verify_rrc_routes(env, model)
    verify_improve(env, model)
    verify_step_math()
    verify_phase_fit_and_resume()

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
