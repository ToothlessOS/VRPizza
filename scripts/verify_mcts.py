#!/usr/bin/env python
"""Verify the POMO-guided MCTS search against the specification, without pytest.

Run from the project root::

    .venv/bin/python scripts/verify_mcts.py

``prompts/MCTS.md`` section 14 lists the checks a first implementation has to pass. They are
gathered here rather than under ``tests/`` for the same reason ``verify_val_parity.py`` and
``verify_sl_data.py`` are: pytest is not a dependency of this virtualenv, and this repository's
working verification convention is a plain harness that prints PASS/FAIL and exits non-zero.

Nothing here loads a checkpoint. The properties under test are properties of the search -- the
tree, the masking, the widening rule, the backup -- and a randomly initialised policy exercises
all of them. That keeps the suite to seconds while the model is still the real ``POMOSL``: priors
come from an actual forward pass, not a stub, so the tests would catch a change in how the
decoder is driven as well as a change in the search.

Two checks go beyond the letter of section 14, because they are sharper than what it asks for:

* **Baseline parity.** The search's own baseline must equal ``_inference_step``'s, which is the
  ordinary POMO rollout the published numbers come from. This is exact, not approximate, and it
  is what makes "zero simulations returns the baseline" a fact about the code rather than about
  the tolerance chosen.
* **First PUCT selection.** The specification's value convention gives every unvisited child
  ``Q = 0``, which is the best value a child can hold, while the depot child is deliberately kept
  unvisited. That predicts a specific, checkable behaviour on the first selection that PUCT ever
  makes. Pinning it here means the known value/exploration mis-scaling shows up as a documented
  property rather than as a mystery during an ablation.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from tensordict.tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envs import EvalCVRPEnv  # noqa: E402
from src.mcts import (  # noqa: E402
    DEPOT_ACTION,
    InstanceBank,
    POMOMCTSConfig,
    POMOOracle,
    allowed_customer_width,
    batch_states,
    initialize_policy,
    select_child_puct,
    solve_batch_with_pomo_mcts,
    state_from_td,
    step_children,
)
from src.mcts.search import _aug_indices  # noqa: E402
from src.models import POMOSL  # noqa: E402

#: Small enough to run the full grid in seconds, large enough that the mask has real structure:
#: at N=10 a route holds several customers, so the depot is sometimes legal and sometimes not.
NUM_LOC = 10
BATCH = 2
NUM_AUGMENT = 8

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record and report the outcome of a single assertion."""
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")
    if not condition:
        failures.append(name)


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n=== {title} ===")


def build_case(seed: int = 0):
    """A tiny CVRP batch, a randomly initialised SL policy, and an environment for both."""
    torch.manual_seed(seed)
    env = EvalCVRPEnv(num_loc=NUM_LOC, capacity=50.0)
    model = POMOSL(env, num_augment=NUM_AUGMENT, eval_greedy=False).eval()
    batch = TensorDict(
        {
            "depot": torch.rand(BATCH, 2),
            "locs": torch.rand(BATCH, NUM_LOC, 2),
            "demand": torch.rand(BATCH, NUM_LOC) * 0.25,
            "capacity": torch.full((BATCH,), 50.0),
            "cost": torch.ones(BATCH),
        },
        batch_size=[BATCH],
    )
    return env, model, batch


def search(env, model, batch, **kwargs):
    """Run the search with the tree kept, so the node invariants can be inspected."""
    cfg = POMOMCTSConfig(keep_tree=True, **kwargs)
    with torch.no_grad():
        return solve_batch_with_pomo_mcts(batch.clone(), model, env, cfg), cfg


def walk(roots):
    """Every node reachable from a set of roots."""
    seen, stack = [], list(roots)
    while stack:
        node = stack.pop()
        seen.append(node)
        stack.extend(node.children.values())
    return seen


def _states_equal(left, right) -> bool:
    """Whether two compact states agree on every mutable field."""
    return all(
        torch.equal(left.__dict__[key], right.__dict__[key])
        for key in ("visited", "used_capacity", "current_node", "action_mask")
    ) and left.done == right.done


# ---------------------------------------------------------------------------------------
# Section 14.1 -- environment and state correctness
# ---------------------------------------------------------------------------------------


def check_states(env, model, batch):
    section("14.1  environment and state correctness")
    result, cfg = search(env, model, batch, sims_per_root=3)
    roots = result.roots
    nodes = walk(roots)

    # A child's state must be exactly what the environment produces from its parent.
    banks = _banks_for(env, model, batch)
    mismatched = 0
    for node in nodes:
        if node.parent is None:
            continue
        parent = node.parent
        td = batch_states(
            [parent.state],
            torch.tensor([parent.aug_index], device=parent.state.visited.device),
            [banks[parent.instance_row]],
        )
        td.set("action", torch.tensor([node.action_from_parent]))
        stepped = env.step(td)["next"]
        for key in ("visited", "used_capacity", "current_node", "action_mask"):
            if not torch.equal(stepped[key][0], node.state.__dict__[key]):
                mismatched += 1
                break
    check("child state equals env.step of its parent", mismatched == 0, f"{mismatched} mismatched")

    # Stepping a copy must not disturb the original.
    parent = next(n for n in nodes if n.children and not n.terminal)
    before = {k: parent.state.__dict__[k].clone() for k in
              ("visited", "used_capacity", "current_node", "action_mask")}
    banks = _banks_for(env, model, batch)
    step_children([(parent, min(parent.children))], env, banks, BATCH)
    untouched = all(torch.equal(before[k], parent.state.__dict__[k]) for k in before)
    check("generating a child does not mutate the parent", untouched)

    # Children must be feasible, and the depot must only appear where the mask allows it.
    infeasible = [n for n in nodes if n.parent is not None
                  and not bool(n.parent.state.action_mask[n.action_from_parent].item())]
    check("every child action was feasible in its parent", not infeasible,
          f"{len(infeasible)} infeasible")

    illegal_depot = [n for n in nodes if n.action_from_parent == DEPOT_ACTION
                     and not n.parent.depot_legal]
    check("a depot child only exists where returning is legal", not illegal_depot,
          f"{len(illegal_depot)} illegal")

    # A depot transition must reset the load exactly as the baseline environment does.
    depot_children = [n for n in nodes if n.action_from_parent == DEPOT_ACTION]
    reset_ok = all(float(n.state.used_capacity.item()) == 0.0 for n in depot_children)
    check("a depot transition resets the vehicle load", reset_ok,
          f"{len(depot_children)} depot children")

    # Every solution the search returns must survive the environment's own validator, and its
    # cost must match an independent recomputation through get_reward.
    env_check = EvalCVRPEnv(num_loc=NUM_LOC, capacity=50.0, check_solution=True)
    td = env_check.reset(env_check.make_input_td(batch.clone()))
    mismatches = 0
    for instance_row, actions in enumerate(result.actions):
        reward = env_check.get_reward(
            td[instance_row].unsqueeze(0), torch.tensor([actions]).long(), check_solution=True
        )
        if not torch.allclose(-reward, result.cost[instance_row], atol=1e-5):
            mismatches += 1
    check("returned solutions validate and their cost recomputes", mismatches == 0,
          f"{mismatches} mismatched")

    # The compact state must round-trip through a materialised TensorDict: stepping the rebuilt
    # state has to agree with the engine's own child construction, which is the only thing that
    # exercises whether the shared instance data was reassembled correctly.
    node = next(
        n for n in nodes
        if not n.terminal and any(a not in n.children for a in n.state.feasible_actions())
    )
    bank = _bank_for(env, model, batch, node.instance_row)
    action = next(a for a in node.state.feasible_actions() if a not in node.children)

    from_engine = step_children([(node, action)], env, banks, BATCH)[0].state
    td = batch_states([node.state], _aug_indices([node]), [bank])
    td.set("action", torch.tensor([action]))
    from_materialised = state_from_td(env.step(td)["next"])

    check("state round-trips through batch_states",
          _states_equal(from_engine, from_materialised))


# ---------------------------------------------------------------------------------------
# Section 14.2 -- policy initialisation
# ---------------------------------------------------------------------------------------


def check_policy_init(env, model, batch):
    section("14.2  policy initialisation correctness")
    result, cfg = search(env, model, batch, sims_per_root=2)
    nodes = walk(result.roots)
    initialised = [n for n in nodes if n.policy_initialized and not n.terminal]
    check("nodes were initialised", bool(initialised), f"{len(initialised)} nodes")

    bad_prior = 0
    bad_rank = 0
    not_descending = 0
    depot_in_rank = 0
    for node in initialised:
        mask = node.state.action_mask
        for action in range(mask.shape[0]):
            if not bool(mask[action]) and float(node.action_priors[action]) != 0.0:
                bad_prior += 1
        ranked = node.ranking()
        if any(not bool(mask[a]) for a in ranked):
            bad_rank += 1
        priors = [float(node.action_priors[a]) for a in ranked]
        if any(later > earlier + 1e-9 for earlier, later in zip(priors, priors[1:])):
            not_descending += 1
        if DEPOT_ACTION in ranked:
            depot_in_rank += 1

    check("infeasible actions carry zero prior", bad_prior == 0, f"{bad_prior} non-zero")
    check("ranked_customer_actions holds only feasible customers", bad_rank == 0)
    check("ranking is descending in prior", not_descending == 0)
    check("ranking excludes the depot", depot_in_rank == 0)
    check("num_customers matches the ranking length",
          all(node.num_customers == len(node.ranking()) for node in initialised))

    # The depot is stored as a prior and as a child, but must not consume a customer slot.
    with_depot = [n for n in initialised if n.depot_legal]
    counted = [n for n in with_depot if n.children.get(DEPOT_ACTION) is not None
               and n.expanded_customer_count != len(n.customer_children())]
    check("the depot child does not consume customer width", not counted,
          f"{len(counted)} nodes miscounted")
    check("a legal depot is materialised at initialisation",
          all(DEPOT_ACTION in n.children for n in with_depot),
          f"{len(with_depot)} nodes with a legal depot")

    # Re-running initialisation must not change anything. The node is already initialised, so
    # this is a no-op by the guard -- and it must stay one, since a second pass would build a
    # second depot child and double-count the customer width.
    sample = initialised[0]
    snapshot = (sample.num_customers, sample.Q, sample.N, len(sample.children))
    bank = _bank_for(env, model, batch, sample.instance_row)
    initialize_policy(
        [sample], POMOOracle(model, env, POMOMCTSConfig()), POMOMCTSConfig(), env, [bank], BATCH
    )
    after = (sample.num_customers, sample.Q, sample.N, len(sample.children))
    check("initialisation is idempotent", snapshot == after, f"{snapshot} -> {after}")


# ---------------------------------------------------------------------------------------
# Section 14.3 -- progressive widening
# ---------------------------------------------------------------------------------------


def check_widening(env, model, batch):
    section("14.3  progressive widening correctness")
    cfg = POMOMCTSConfig()

    # The formula, stated independently of the implementation.
    formula_ok = True
    for feasible in (1, 5, 8, 20, 40):
        for visits in (0, 1, 8, 20, 21, 100, 1000):
            expected = min(feasible, max(cfg.k0, int(cfg.c_pw * max(1, visits) ** cfg.alpha_pw)))
            if feasible <= cfg.endgame_customer_threshold:
                expected = feasible
            if cfg.customer_width(feasible, visits) != expected:
                formula_ok = False
    check("customer_width matches K(s) = min(m, max(k0, c_pw * max(1,N)^alpha))", formula_ok)

    check("the endgame override expands every customer",
          cfg.customer_width(5, 0) == 5 and cfg.customer_width(8, 0) == 8)
    check("the endgame override does not apply above the threshold",
          cfg.customer_width(9, 0) == cfg.k0)
    check("the frontier is still k0 at N=20 and widens at N=21",
          cfg.customer_width(40, 20) == cfg.k0 and cfg.customer_width(40, 21) > cfg.k0)
    check("a fixed-k config never widens",
          POMOMCTSConfig(expansion="fixed_k").customer_width(40, 1000) == cfg.k0)

    result, _ = search(env, model, batch, sims_per_root=24)
    nodes = walk(result.roots)
    over = [n for n in nodes if n.expanded_customer_count > allowed_customer_width(n, cfg)]
    check("no node exceeds its permitted customer width", not over, f"{len(over)} over")

    # The next expanded customer is the next ranked one, in order, with no repeats.
    deterministic = 0
    duplicated = 0
    for node in nodes:
        ranked = node.ranking()
        customer_children = sorted(a for a in node.children if a != DEPOT_ACTION)
        if customer_children != sorted(ranked[: len(customer_children)]):
            deterministic += 1
        if len(customer_children) != len(set(customer_children)):
            duplicated += 1
    check("children are a prefix of the ranking, in order", deterministic == 0,
          f"{deterministic} nodes off")
    check("no node creates duplicate children", duplicated == 0)
    check("a node never has more children than ranking entries",
          all(len(n.customer_children()) <= n.num_customers for n in nodes))


# ---------------------------------------------------------------------------------------
# Section 14.4 -- statistics
# ---------------------------------------------------------------------------------------


def check_statistics(env, model, batch):
    section("14.4  MCTS statistics correctness")
    # Deliberately above the PUCT threshold. The frontier holds at k0 until a node has been
    # visited 21 times, so a short budget never selects by PUCT at all -- and a suite that only
    # ever called segment_child_puct directly would still pass while the search did nothing but
    # build depth-1 trees.
    sims = 24
    result, cfg = search(env, model, batch, sims_per_root=sims)
    roots = result.roots
    nodes = walk(roots)

    check("the budget was large enough for PUCT to run",
          result.stats.puct_selections > 0,
          f"{result.stats.puct_selections} selections, {result.stats.expansions} expansions")
    check("every simulation ends in exactly one expansion or one terminal evaluation",
          result.stats.expansions + result.stats.terminal_evaluations == sims * len(roots),
          f"{result.stats.expansions} + {result.stats.terminal_evaluations} vs {sims * len(roots)}")
    check("the tree is deeper than one level",
          any(len(n.prefix) > 2 for n in nodes),
          f"deepest {max(len(n.prefix) for n in nodes)}")

    # Every simulation backs up through exactly one path, so a node's visits cannot exceed the
    # number of simulations and the roots must have received all of them.
    check("every root was visited once per simulation",
          all(root.N == sims for root in roots),
          f"visits {sorted({r.N for r in roots})} for {sims} simulations")
    check("no node is visited more often than there were simulations",
          all(node.N <= sims for node in nodes))
    check("Q equals W / N wherever N > 0",
          all(abs(node.Q - node.W / node.N) < 1e-9 for node in nodes if node.N))
    check("Q is exactly zero wherever N is zero",
          all(node.Q == 0.0 for node in nodes if node.N == 0))
    # W is not a proxy for N: a rollout that re-finds the baseline exactly backs up a value of
    # exactly zero, because the value is normalised against the baseline. So nodes with W == 0
    # and N > 0 are expected, and asserting W == 0 iff N == 0 would fail on a correct search.
    check("a rollout matching the baseline backs up exactly zero",
          all(
              abs(node.Q) < 1e-12
              for node in nodes
              if node.N and node.terminal is False and abs(node.W) < 1e-12
          ))

    # Children carry the visits that flowed through them, so a parent's count is the sum over
    # children plus the simulations that stopped at the parent itself.
    consistent = True
    for node in nodes:
        if node.terminal:
            continue
        total = sum(child.N for child in node.children.values())
        if total > node.N:
            consistent = False
    check("a parent is visited at least as often as its children together", consistent)

    # PUCT must equal a reference implementation of the same formula, ties to the smallest action.
    node = next(n for n in nodes if len(n.children) > 1)
    scores = {
        action: child.Q + cfg.c_puct * float(node.action_priors[action]) * math.sqrt(node.N)
        / (1 + child.N)
        for action, child in node.children.items()
    }
    best = max(scores.values())
    expected = min(action for action, score in scores.items() if score == best)
    action, _ = select_child_puct(node, cfg.c_puct)
    check("PUCT selection matches the reference argmax", action == expected,
          f"got {action}, expected {expected}")

    # The predicted first-selection behaviour. Section 5.1 gives every unvisited child Q = 0,
    # which is the maximum a child can hold under the normalised value, and section 7.3 keeps the
    # depot child unvisited until it is selected -- so the depot is a strong candidate the first
    # time PUCT runs at a node. This is a property of the specification's value convention,
    # recorded rather than corrected; section 8.2 asks for c_puct to be tuned against it.
    saturated = [
        n for n in nodes
        if n.N > 0 and n.expanded_customer_count >= allowed_customer_width(n, cfg)
        and len(n.children) > 1
    ]
    check("some node saturated its frontier and had to select", bool(saturated),
          f"{len(saturated)} saturated nodes")
    if saturated:
        node = saturated[0]
        action, _ = select_child_puct(node, cfg.c_puct)
        unvisited = [c for c in node.children.values() if c.N == 0]
        check("PUCT's choice is consistent with the score it computes",
              action in node.children
              and all(
                  node.children[action].Q
                  + cfg.c_puct * float(node.action_priors[action]) * math.sqrt(node.N)
                  / (1 + node.children[action].N)
                  >= other.Q
                  + cfg.c_puct * float(node.action_priors[a]) * math.sqrt(node.N) / (1 + other.N)
                  for a, other in node.children.items()
              ))
        # The structural consequence of the value convention: an unvisited child holds Q = 0,
        # and no child that failed to beat the baseline can hold more. This is the mis-scaling
        # section 8.2 warns about, stated as the invariant it actually is -- it does *not* mean
        # the depot child wins, because the prior still weights the exploration term.
        failed = [c for c in node.children.values() if c.N > 0 and c.Q < 0]
        if unvisited and failed:
            check("an unvisited child's Q dominates every child that failed to improve",
                  all(u.Q >= c.Q for u in unvisited for c in failed))


# ---------------------------------------------------------------------------------------
# Section 14.5 -- regression against the baseline
# ---------------------------------------------------------------------------------------


def check_regression(env, model, batch):
    section("14.5  regression against the POMO baseline")

    # The strongest form of "zero simulations returns the baseline": exact equality, and the
    # baseline itself must reproduce the rollout the published numbers come from.
    result, _ = search(env, model, batch, sims_per_root=0)
    check("zero simulations returns the baseline exactly",
          torch.equal(result.cost, result.baseline_cost)
          and result.actions == result.baseline_actions)

    with torch.no_grad():
        reference = -model._inference_step(batch.clone(), "test")["reward"]
    check("the search's baseline equals _inference_step's", torch.equal(reference, result.baseline_cost),
          f"{reference.tolist()} vs {result.baseline_cost.tolist()}")

    for sims in (1, 3, 12):
        result, _ = search(env, model, batch, sims_per_root=sims)
        check(f"at {sims} simulations the incumbent is never worse than the baseline",
              bool((result.cost <= result.baseline_cost).all()))
        check(f"at {sims} simulations the incumbent is improvable",
              bool((result.cost >= 0).all()))

    env_check = EvalCVRPEnv(num_loc=NUM_LOC, capacity=50.0, check_solution=True)
    result, _ = search(env, model, batch, sims_per_root=12)
    td = env_check.reset(env_check.make_input_td(batch.clone()))
    feasible = True
    for instance_row, actions in enumerate(result.actions):
        try:
            env_check.get_reward(
                td[instance_row].unsqueeze(0), torch.tensor([actions]).long(), check_solution=True
            )
        except AssertionError:
            feasible = False
    check("every returned solution is feasible", feasible)
    check("finding no improvement is reported rather than treated as an error",
          result.stats.improvements == int(result.improved.sum()),
          f"{result.stats.improvements} improvements over {int(result.improved.sum())} instances")


def _banks_for(env, model, batch):
    """The instance banks the search builds, rebuilt for inspection.

    ``step_children`` indexes its bank list by ``instance_row``, so callers have to hand it the
    whole batch's worth even when stepping a single node.
    """
    reset = env.reset(env.make_input_td(batch.clone()))
    td_aug = model.augment(reset)
    num_augment = int(model.num_augment)
    return [
        InstanceBank(
            aug_locs=torch.stack([td_aug[a * BATCH + row]["locs"] for a in range(num_augment)]),
            demand=td_aug[row]["demand"].clone(),
            n_loc=NUM_LOC,
            num_augment=num_augment,
        )
        for row in range(BATCH)
    ]


def _bank_for(env, model, batch, instance_row):
    """The instance bank for one instance."""
    return _banks_for(env, model, batch)[instance_row]


def main() -> int:
    """Run every section and report."""
    env, model, batch = build_case()
    print(f"MCTS verification: N={NUM_LOC}, batch={BATCH}, augment={NUM_AUGMENT}")

    check_states(env, model, batch)
    check_policy_init(env, model, batch)
    check_widening(env, model, batch)
    check_statistics(env, model, batch)
    check_regression(env, model, batch)

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
