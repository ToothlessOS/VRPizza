#!/usr/bin/env python
"""Verify the ported SIL search and dataset without pytest.

Run from the project root::

    .venv/bin/python scripts/verify_sil.py

The checks are ordered by dependency: the Gumbel primitives, deterministic SBS against a
brute-force beam, without-replacement behaviour of the trie, parity between the rl4co adapter
and ``env.get_reward``, the dormant advantage updates, and the pseudo-expert dataset round
trip. The full phase-switched training run is exercised by ``notebooks/POMO_sil.py --smoke``.

Everything runs on a handful of tiny CVRP instances with a randomly initialised policy. The
properties under test are properties of the port, not of a checkpoint, and this keeps the
suite to seconds.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envs import EvalCVRPEnv  # noqa: E402
from src.lehd.subpath import actions_to_routes, cost_from_flags, flatten_routes  # noqa: E402
from src.models import POMOSL  # noqa: E402
from src.sil.dataset import SILSolutionDataset  # noqa: E402
from src.sil.incremental_sbs import (  # noqa: E402
    IncrementalSBS,
    gumbel_log_survival,
    gumbel_without_replacement_expectation,
    log_subtract,
)
from src.sil.sbs import (  # noqa: E402
    BeamLeaf,
    sample_gumbels_with_maximum,
    stochastic_beam_search,
)
from src.sil.search import SILSearchConfig, generate_best_solutions  # noqa: E402
from src.sil.trajectory import SILSearchAdapter  # noqa: E402

#: Small enough to brute-force the tree, large enough that the WOR sample is non-trivial.
TREE_DEPTH = 6
TREE_BEAM = 4
TREE_ROUNDS = 3

NUM_LOC = 8
BATCH = 3

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record and report the outcome of a single assertion."""
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")
    if not condition:
        failures.append(name)


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n=== {title} ===")


# -- a synthetic tree with a tractable number of leaves --------------------------------


def toy_log_probs(state: tuple[int, ...]) -> np.ndarray:
    """Two-action log-probabilities that depend on depth and on the previous action."""
    base = 0.75 - 0.02 * len(state)
    if state and state[-1] == 1:
        base = 1.0 - base
    return np.log(np.array([base, 1.0 - base], dtype=np.float64))


def toy_log_probability_fn(states):
    return [toy_log_probs(state) for state in states]


def toy_transition_fn(pairs):
    out = []
    for state, action in pairs:
        child = state + (int(action),)
        out.append((child, len(child) >= TREE_DEPTH))
    return out


def toy_leaf_eval(state: tuple[int, ...]) -> float:
    """An arbitrary maximisation objective with a different value for every path."""
    return float(sum(action * 2.0 ** (-(i + 1)) for i, action in enumerate(state)))


def brute_force_top_k(k: int):
    """Top-k paths of the toy tree by cumulative log-probability."""
    paths = []
    for actions in itertools.product((0, 1), repeat=TREE_DEPTH):
        log_prob = float(sum(toy_log_probs(actions[:i])[a] for i, a in enumerate(actions)))
        paths.append((log_prob, actions))
    paths.sort(key=lambda item: item[0], reverse=True)
    return paths[:k]


# -- primitive checks -----------------------------------------------------------------


def check_gumbel_math() -> None:
    section("Gumbel primitives")
    rng = np.random.default_rng(0)
    log_probs = np.log(rng.dirichlet(np.ones(6)))
    for target in (0.0, -3.5, 12.0):
        gumbels = sample_gumbels_with_maximum(log_probs, target, rng=rng)
        check(
            f"conditioned maximum == {target}",
            abs(float(np.max(gumbels)) - target) < 1e-9,
            f"max={float(np.max(gumbels)):.6f}",
        )
    gumbels = sample_gumbels_with_maximum(log_probs, 0.0, rng=rng)
    check("gumbels are not the log-probabilities", not np.allclose(gumbels, log_probs))

    x, y = -1.0, -3.0
    check(
        "log_subtract equals log(exp(x)-exp(y))",
        abs(log_subtract(x, y) - math.log(math.exp(x) - math.exp(y))) < 1e-9,
    )
    check("log_subtract is -inf below the subtrahend", np.isneginf(log_subtract(-4.0, -3.0)))
    check(
        "gumbel_log_survival(0) == log(1-exp(-1))",
        abs(float(gumbel_log_survival(np.array(0.0))) - math.log(1.0 - math.exp(-1.0))) < 1e-12,
    )


def check_deterministic_beam() -> None:
    section("Deterministic SBS equals top-k beam search")
    leaves = stochastic_beam_search(
        toy_log_probability_fn,
        toy_transition_fn,
        root_states=[()],
        beam_width=TREE_BEAM,
        deterministic=True,
    )[0]
    expected = brute_force_top_k(TREE_BEAM)
    got = [(leaf.log_probability, leaf.state) for leaf in leaves]
    check("beam width leaves are returned", len(leaves) == TREE_BEAM, f"{len(leaves)}")
    check(
        "leaves match the brute-force top-k order and values",
        all(
            abs(got[i][0] - expected[i][0]) < 1e-9 and got[i][1] == expected[i][1]
            for i in range(TREE_BEAM)
        ),
        f"got={[path for _, path in got]}",
    )
    check(
        "leaves are sorted by decreasing cumulative log-probability",
        all(got[i][0] >= got[i + 1][0] for i in range(len(got) - 1)),
    )


def check_without_replacement() -> None:
    section("WOR-SBS samples without replacement")
    searcher = IncrementalSBS(
        initial_states=[()],
        child_log_probability_fn=toy_log_probability_fn,
        child_transition_fn=toy_transition_fn,
        leaf_evaluation_fn=toy_leaf_eval,
    )
    first = searcher.perform_incremental_sbs(
        beam_width=TREE_BEAM,
        num_rounds=TREE_ROUNDS,
        log_prob_update_type="wor",
        rng=np.random.default_rng(1),
    )[0]
    second = searcher.perform_incremental_sbs(
        beam_width=TREE_BEAM,
        num_rounds=TREE_ROUNDS,
        log_prob_update_type="wor",
        rng=np.random.default_rng(2),
    )[0]

    first_paths = [leaf.state for leaf in first]
    second_paths = [leaf.state for leaf in second]
    check(
        "rounds fill the beam",
        len(first_paths) == TREE_BEAM * TREE_ROUNDS,
        f"{len(first_paths)}",
    )
    check("no duplicate paths within the first call", len(set(first_paths)) == len(first_paths))
    check("no duplicate paths within the second call", len(set(second_paths)) == len(second_paths))
    check(
        "the second call returns only unseen paths",
        not (set(first_paths) & set(second_paths)),
        f"overlap={len(set(first_paths) & set(second_paths))}",
    )
    root_masses = searcher.root_nodes[0][0].unsampled_log_masses
    check(
        "the trie retains residual masses",
        root_masses is not None and bool(np.any(np.isfinite(root_masses))),
    )


def check_dormant_updates() -> None:
    section("Dormant trie policy updates")
    for update_type in ("gumbeldore", "theory_gumbeldore"):
        searcher = IncrementalSBS(
            initial_states=[()],
            child_log_probability_fn=toy_log_probability_fn,
            child_transition_fn=toy_transition_fn,
            leaf_evaluation_fn=toy_leaf_eval,
        )
        leaves = searcher.perform_incremental_sbs(
            beam_width=TREE_BEAM,
            num_rounds=2,
            log_prob_update_type=update_type,
            advantage_constant=1.0,
            rng=np.random.default_rng(3),
        )[0]
        root = searcher.root_nodes[0][0]
        check(f"{update_type} returns leaves", len(leaves) == TREE_BEAM * 2)
        check(
            f"{update_type} perturbs at least one edge",
            root.children_advantages is not None
            and bool(np.any(root.children_advantages != 0.0)),
        )

    leaves = [
        BeamLeaf(state=(None, "a"), log_probability=math.log(0.6), gumbel=0.2),
        BeamLeaf(state=(None, "b"), log_probability=math.log(0.4), gumbel=-0.4),
    ]
    expectation = gumbel_without_replacement_expectation(
        leaves, lambda state: 1.0 if state == "a" else 0.0
    )
    check(
        "importance-weighted expectation stays in [0, 1]",
        0.0 <= expectation <= 1.0,
        f"{expectation:.4f}",
    )


# -- CVRP adapter and data checks -----------------------------------------------------


def build_case(seed: int = 0):
    """A tiny generated CVRP batch and a randomly initialised POMO model."""
    torch.manual_seed(seed)
    env = EvalCVRPEnv(num_loc=NUM_LOC, capacity=50.0)
    model = POMOSL(env, num_augment=8, eval_greedy=False).eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        generated = env.generator(batch_size=[BATCH])
    return env, model, generated


def check_adapter(env, model, generated) -> None:
    section("Adapter parity with the rl4co environment")
    root_td = env.reset(generated.clone())
    adapter = SILSearchAdapter(model.policy, env, root_td, device="cpu")
    roots = adapter.root_states(root_td)

    check("one root per instance", len(roots) == BATCH)
    check("roots start at the depot", all(root.current == 0 for root in roots))
    check("the depot action is masked at a root", all(not root.mask[0] for root in roots))

    log_probs = adapter.child_log_probability_fn(roots)
    feasible_sums = [
        float(np.exp(probs[state.mask]).sum()) for probs, state in zip(log_probs, roots)
    ]
    check(
        "feasible log-probabilities sum to one",
        all(abs(total - 1.0) < 1e-5 for total in feasible_sums),
        f"{[round(total, 6) for total in feasible_sums]}",
    )
    check(
        "infeasible actions carry -inf",
        all(np.all(np.isneginf(probs[~state.mask])) for probs, state in zip(log_probs, roots)),
    )

    action = int(np.flatnonzero(roots[0].mask)[0])
    child, is_leaf = adapter.child_transition_fn([(roots[0], action)])[0]
    check("a transition returns exactly one child", child.instance == roots[0].instance)
    check("the child advances to the chosen action", child.current == action)
    check("a single customer does not finish the episode", not is_leaf)
    check("the child records the action", child.actions == (action,))


def check_leaf_costs(env, model, generated) -> None:
    section("Leaf costs equal env.get_reward")
    root_td = env.reset(generated.clone())
    adapter = SILSearchAdapter(model.policy, env, root_td, device="cpu")
    roots = adapter.root_states(root_td)
    leaves_batch = stochastic_beam_search(
        adapter.child_log_probability_fn,
        adapter.child_transition_fn,
        roots,
        beam_width=2,
        deterministic=True,
        max_steps=4 * (NUM_LOC + 1),
    )
    checked = 0
    for i, leaves in enumerate(leaves_batch):
        # A fresh reset for the reward call: the adapter's root_td is shared with the trees.
        td_i = env.reset(generated[i : i + 1].clone())
        for leaf in leaves:
            actions = torch.as_tensor(np.asarray(leaf.state.actions), dtype=torch.long)[None]
            reward = env.get_reward(td_i, actions, check_solution=True)
            check(
                f"instance {i}: leaf cost matches get_reward",
                abs(-float(reward[0]) - leaf.state.cost) < 1e-4,
                f"{-float(reward[0]):.6f} vs {leaf.state.cost:.6f}",
            )
            checked += 1
    check("both instances were checked", checked >= 2, f"{checked} leaves")


def check_generation_and_dataset(env, model, generated) -> None:
    section("Generation, records and the pseudo-expert dataset")
    for search_type in ("beam_search", "wor"):
        cfg = SILSearchConfig(
            search_type=search_type,
            beam_width=2,
            num_rounds=2,
            chunk_size=BATCH,
            seed=7,
        )
        records, metrics = generate_best_solutions(model.policy, env, BATCH, cfg, device="cpu")
        check(f"{search_type}: one record per instance", len(records) == BATCH)
        check(
            f"{search_type}: metrics are finite",
            math.isfinite(metrics["gen_mean_cost"])
            and math.isfinite(metrics["gen_best_cost"]),
        )

        for record in records:
            covered = int(np.unique(record.tour).size)
            check(
                f"{search_type}: record covers every customer",
                covered == NUM_LOC,
                f"{covered}/{NUM_LOC}",
            )
            check(
                f"{search_type}: flags mark each route",
                int(np.sum(record.flags)) == record.num_routes
                and len(record.tour) == NUM_LOC,
            )

        dataset = SILSolutionDataset(records, length=len(records) * 4, seed=0)
        check(
            f"{search_type}: dataset honours the requested length",
            len(dataset) == len(records) * 4,
        )
        for index in range(min(len(dataset), 4)):
            item = dataset[index]
            routes = actions_to_routes(np.asarray(item["actions"]), NUM_LOC)
            tour, flags = flatten_routes(routes)
            cost = cost_from_flags(tour, flags, item["depot"], item["locs"])
            check(
                f"{search_type}: dataset item replays to the record cost",
                abs(cost - float(item["cost"])) < 1e-4,
                f"{cost:.6f} vs {float(item['cost']):.6f}",
            )


def main() -> int:
    """Run every section and report."""
    check_gumbel_math()
    check_deterministic_beam()
    check_without_replacement()
    check_dormant_updates()

    env, model, generated = build_case()
    print(f"\nCVRP verification: N={NUM_LOC}, batch={BATCH}")
    check_adapter(env, model, generated)
    check_leaf_costs(env, model, generated)
    check_generation_and_dataset(env, model, generated)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
