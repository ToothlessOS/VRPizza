# POMO-Guided MCTS for CVRP: Implementation Specification

## 1. Purpose and scope

Implement an inference-time search module for a pretrained POMO-style autoregressive CVRP policy. The module improves on POMO multi-start + 8x augmentation inference by using Monte Carlo Tree Search (MCTS) to allocate more computation to promising partial routes.

The existing inference / evaluation pipeline is available through `notebooks/eval_pomo.py`

The implementation is intentionally **inference only** for the first version:

- Do not retrain POMO.
- Reuse the existing CVRP environment for transitions, feasibility masks, objective computation, and final-solution validation.
- Treat the pretrained POMO decoder as a source of policy priors and rollout completions.
- Use a custom tree-search implementation, not a general-purpose MCTS library.

The principal search method is **POMO-prior PUCT with progressive widening**:

1. POMO ranks feasible customer actions at each tree node.
2. Initially, only a small top-\(k_0\) prefix of ranked customers becomes tree children.
3. A legal depot-return action is always exposed separately and does not consume a customer-action slot.
4. As a particular node receives more MCTS visits, progressively expose more customer children from the stored POMO ranking.
5. Select among currently materialized children with PUCT.
6. Evaluate new leaves by completing the remaining CVRP solution with POMO.
7. Back up the resulting scalar value through the tree.

The code should be modular enough to support ablations:

- No search / original POMO inference.
- Fixed top-\(k\) MCTS.
- Progressive-widening MCTS.

We should have 4 different configurations:

1. SL
2. SL + MCTS
3. RL
4. RL + MCTS

and the results should besotred to `logs/eval_pomo_cvrp`

---

## 2. Problem assumptions

### 2.1 CVRP state

Use the already implemented CVRP environment. You can refer to `docs/CVRPEnv.md` for more info.

---

## 3. High-level algorithm

For each CVRP instance:

1. Generate POMO multi-start with 8x augmentation (as in the current eval/inference pipeline) initial states, following the existing POMO inference convention for the model/environment.
2. Treat each valid POMO start as a separate MCTS root.
3. Allocate a total simulation/forward-pass/wall-clock budget across roots.
4. For each root, execute MCTS simulations:
   - Selection through existing children using PUCT.
   - Progressive widening when a selected node may expose another customer action.
   - Evaluation of a newly created child or unexpanded leaf using a POMO completion rollout.
   - Backup of value to all nodes/actions on the path.
5. Return the best feasible complete CVRP solution observed across all roots and all simulations.
6. Optionally apply the same allowed local-search postprocessor to every compared inference method, then retain the best feasible postprocessed solution.

The first version should use a fixed total simulation budget per instance. Also report POMO decoder calls and elapsed wall-clock time because a simulation count alone is not a fair cost measure once rollout lengths and batching differ.
---

## 4. POMO multi-start roots

### 4.1 Root interpretation

POMO multi-start inference is a population of initial decoding trajectories. For MCTS, represent each start as its own root state:

\[
s_0^{(r)}, \qquad r \in \{1,\dots,R\}.
\]

A root may be the initial empty CVRP state or a state after the initial POMO-prescribed action(s), depending on the original POMO CVRP inference code. Preserve the baseline convention exactly so that the no-search baseline is comparable.

### 4.2 Root budget allocation

Start with equal allocation:

\[
B_r = \left\lfloor \frac{B}{R} \right\rfloor,
\]

where \(B\) is the total number of simulations per instance.

Later ablations may use a two-stage allocation:

1. Give every root \(B_{\min}\) simulations.
2. Estimate each root using its best rollout or greedy completion.
3. Allocate the remaining budget to roots with the strongest scores.

Do not add adaptive root allocation in the initial correctness-first implementation.

### 4.3 Best-solution policy

Maintain a global incumbent:

```python
best_solution: Solution | None
best_cost: float = float("inf")
```

Whenever a rollout or terminal leaf creates a valid full CVRP solution, compute its objective and update the incumbent if the objective is lower. Return this incumbent—not merely the root's most visited action.

---

## 5. Tree node design

A node represents a fully specified CVRP partial solution.

```python
@dataclass
class MCTSNode:
    state: CVRPState
    parent: MCTSNode | None
    action_from_parent: int | None

    # POMO policy information, initialized once per node.
    policy_initialized: bool = False
    action_priors: dict[int, float] = field(default_factory=dict)
    ranked_customer_actions: list[int] = field(default_factory=list)
    depot_legal: bool = False

    # Tree structure.
    children: dict[int, MCTSNode] = field(default_factory=dict)
    expanded_customer_count: int = 0

    # MCTS statistics. Higher Q must mean better.
    N: int = 0
    W: float = 0.0
    Q: float = 0.0

    terminal: bool = False
```

### 5.1 Required invariants

- `children[action]` is unique for each action.
- `ranked_customer_actions` contains only currently feasible customer actions, in descending POMO prior order, and excludes the depot.
- `expanded_customer_count` equals the number of customer children currently in `children`.
- A depot child, if it exists, does not count toward `expanded_customer_count`.
- `N >= 0` and increments by one for each backed-up simulation that visits the node.
- `W` is the accumulated scalar value across those visits.
- `Q = W / N` for `N > 0`; define `Q = 0.0` for `N == 0`.
- The node's state must be independent of its parent state after creation.

### 5.2 Transposition policy

For version 1, use a strict tree without transpositions. Two action histories leading to equivalent route states are stored separately.

This avoids difficult questions about state hashing, route history, decoder cache identity, and correct sharing of statistics. Add a transposition table only after the baseline is validated.

---

## 6. POMO policy initialization

Policy initialization occurs once when a node first needs expansion or selection.

### 6.1 Required function

```python
def initialize_policy(node: MCTSNode, model, env) -> None:
    """Populate action priors, ranked customer actions, depot legality, terminal flag."""
```

Required behavior:

1. If terminal, set `node.terminal = True` and return.
2. Obtain the environment action mask for this state.
3. Run the POMO decoder to produce logits or probabilities for every action.
4. Apply the feasibility mask before constructing probabilities.
5. Convert valid logits to a probability distribution using softmax, unless the model already returns masked probabilities.
6. Identify legal feasible customers and, separately, whether depot return is legal.
7. Store action priors for every legal customer and the legal depot action.
8. Sort feasible customers by descending prior and store them in `ranked_customer_actions`.
9. Set `policy_initialized = True`.

Conceptually:

\[
P_\theta(a\mid s) = \operatorname{softmax}(z_\theta(s) + \log M(s))_a,
\]

where \(M(s)\) is the feasibility mask. Invalid actions must have zero probability and must never be added as children.

### 6.2 Prior temperature

Initially use the pretrained policy's natural inference distribution:

\[
P_\theta(a\mid s) = \operatorname{softmax}(z_\theta(s))_a.
\]

Optionally expose an inference-only prior temperature \(\tau_p\):

\[
P_{\theta,\tau_p}(a\mid s)
= \operatorname{softmax}\left(\frac{z_\theta(s)}{\tau_p}\right)_a.
\]

- \(\tau_p < 1\): sharper priors, more exploitation.
- \(\tau_p > 1\): flatter priors, more exploration.

Default: \(\tau_p = 1\). Do not tune this until the baseline is correct.

---

## 7. Progressive widening

### 7.1 Customer-width rule

Let \(N(s)\) be the visit count at node \(s\). Let \(m(s)=|\mathcal A_{\mathrm{cust}}(s)|\) be the number of feasible customers at policy initialization. The maximum number of materialized **customer** children is:

\[
K(s) =
\min\left[
 m(s),
 \max\left\{
 k_0,
 \left\lfloor c_{\mathrm{pw}}\,\max(1,N(s))^\alpha \right\rfloor
 \right\}
\right].
\]

Parameters:

- \(k_0\): initial customer frontier size.
- \(c_{\mathrm{pw}}\): widening coefficient.
- \(\alpha\): widening exponent, strictly between 0 and 1.

Recommended initial values for CVRP100:

```python
k0 = 8
c_pw = 2.0
alpha = 0.5
endgame_customer_threshold = 8
```

These are starting points for ablation, not established optimal hyperparameters.

### 7.2 Endgame override

When the number of feasible customers is small, expand all customers:

\[
K(s)=m(s) \quad \text{if } m(s) \le m_{\mathrm{end}}.
\]

Use `m_end = 8` initially. This makes exhaustive local branching available when it is cheap.

This is based on remaining feasible customer count, not fixed decoder depth. Do **not** implement a rule like “top-\(k\) for the first 10 layers, all actions later” as the default algorithm.

### 7.3 Depot rule

The depot action is not part of \(K(s)\). Define the expanded actions as:

\[
\mathcal C(s) =
\operatorname{TopK}_{K(s)}
\left\{
P_\theta(a\mid s): a\in\mathcal A_{\mathrm{cust}}(s)
\right\}
\cup
\{\text{depot if legal}\}.
\]

Operationally:

- Create the depot child as soon as the node is initialized, whenever it is legal.
- Never let a depot child consume one of the top-\(K\) customer slots.
- Respect the environment mask: do not allow an illegal depot transition or consecutive no-op depot behavior.

### 7.4 How to add customer children

The node stores the complete POMO-ranked customer list:

\[
a_{(1)}, a_{(2)}, \ldots, a_{(m)}.
\]

If `expanded_customer_count = j` and widening permits another customer, the next action is deterministically:

\[
a_{(j+1)}.
\]

Do not resample customer rankings on later visits. Do not discard or replace previously created children.

### 7.5 Expansion schedule

Use conservative one-child expansion for the initial implementation:

```python
def maybe_expand_one(node: MCTSNode, env, cfg) -> MCTSNode | None:
    initialize_policy_if_needed(node)

    if node.terminal:
        return None

    # Depot becomes immediately available but does not consume customer width.
    if node.depot_legal and DEPOT_ACTION not in node.children:
        node.children[DEPOT_ACTION] = create_child(node, DEPOT_ACTION, env)

    target_width = allowed_customer_width(node, cfg)
    if node.expanded_customer_count < target_width:
        action = node.ranked_customer_actions[node.expanded_customer_count]
        child = create_child(node, action, env)
        node.children[action] = child
        node.expanded_customer_count += 1
        return child

    return None
```

The returned new child is evaluated immediately by a rollout in the same simulation.

Alternative ablation: materialize all children up to `target_width` at once. Do not use this as the first default because it can consume a large portion of simulations on shallow unvisited children.

---

## 8. Selection with PUCT

### 8.1 Value convention

CVRP minimizes total route cost, but PUCT is written as a maximization rule. Use a scalar value convention in which larger is better.

For the first implementation, use a normalized negative total cost:

\[
V(\text{complete solution}) = -\frac{C(\text{solution})}{S},
\]

where \(S>0\) is a stable instance-level scale. Possible choices:

- `S = 1.0` for an initial correctness test, with careful PUCT tuning.
- A deterministic baseline cost, such as the POMO greedy solution cost for the same instance.
- A coordinate/problem-dependent scaling factor used consistently within an instance.

A practical normalized improvement form is:

\[
V = \frac{C_{\mathrm{ref}} - C}{\max(C_{\mathrm{ref}}, \varepsilon)},
\]

where \(C_{\mathrm{ref}}\) is the greedy POMO cost for this instance. Then positive values mean improvement over the baseline.

For the first version, calculate `reference_cost` once per instance from ordinary POMO inference and use this expression. Ensure the reference solution is feasible.

### 8.2 PUCT equation

For an already materialized child reached through action \(a\), select:

\[
a^* =
\arg\max_{a\in\mathcal C(s)}
\left[
Q(s,a)
+
c_{\mathrm{puct}}\,P_\theta(a\mid s)
\frac{\sqrt{N(s)}}{1+N(s,a)}
\right].
\]

Use:

- `child.Q` for \(Q(s,a)\).
- `node.N` for \(N(s)\).
- `child.N` for \(N(s,a)\).
- `node.action_priors[action]` for \(P_\theta(a\mid s)\).

Initial value:

```python
c_puct = 1.0
```

Tune only after confirming sensible tree behavior. Because value normalization affects the relative size of `Q` and the exploration bonus, `c_puct` must be tuned jointly with the return scale.

### 8.3 Selection order

At each node, use this decision order:

1. Initialize POMO policy if needed.
2. Create legal depot child if it has not yet been created.
3. If progressive widening permits a new customer child, create exactly one new customer child and stop selection; evaluate that new child.
4. Otherwise, choose an existing child by PUCT and descend.

This lets newly permitted customer actions receive at least one evaluation rather than being starved by strong existing children.

---

## 9. Leaf evaluation / rollout

### 9.1 Initial rollout evaluator

For a newly expanded node or a leaf with no expandable actions:

1. Clone the partial CVRP state.
2. Finish it using the pretrained POMO decoder.
3. Use deterministic greedy decoding first.
4. Validate the finished solution with the environment.
5. Compute total route cost.
6. Convert the cost to the normalized scalar value described above.
7. Update the global incumbent if the solution is valid and cheaper.

```python
def evaluate_leaf(node, model, env, reference_cost) -> tuple[float, Solution]:
    completed_solution = pomo_greedy_complete(node.state, model, env)
    assert validate_solution(completed_solution, env)
    cost = compute_cost(completed_solution, env)
    value = (reference_cost - cost) / max(reference_cost, 1e-8)
    return value, completed_solution
```

### 9.2 Rollout variants

Keep this interface extensible for later experiments:

```python
rollout_mode: Literal[
    "greedy",
    "sample",
    "best_of_n_samples",
    "greedy_plus_local_search",
]
```

Recommended experiment order:

1. Greedy POMO completion.
2. Best of \(r\) stochastic POMO completions from the leaf.
3. Greedy completion plus a local-search method, applied consistently across baselines.

Do not use a top-\(k\)-restricted rollout policy in the first implementation. Progressive widening applies to **tree expansion**; allow the rollout to use the original feasible-action POMO policy. This avoids pruning the same actions both in the tree and in the evaluator.

### 9.3 Terminal nodes

If a selected/expanded node is terminal, do not rollout further. Evaluate its completed solution directly and back up the resulting value.

---

## 10. Backup

After evaluating a complete solution, back up its scalar value through every node on the selected path, including the leaf/root according to a single consistent convention.

```python
def backup(path: list[MCTSNode], value: float) -> None:
    for node in reversed(path):
        node.N += 1
        node.W += value
        node.Q = node.W / node.N
```

Use the same scalar value at every node in the initial implementation because this is a deterministic planning problem and the final objective is global route cost.

Do not negate the value on alternating levels: CVRP here is a single-agent minimization problem, not a two-player zero-sum game.

---

## 11. Full simulation pseudocode

```python
def run_simulation(root, model, env, cfg, reference_cost, incumbent):
    node = root
    path = [node]

    while True:
        initialize_policy_if_needed(node, model, env)

        if node.terminal:
            value, solution = evaluate_terminal(node, env, reference_cost)
            incumbent.update_if_better(solution)
            backup(path, value)
            return

        new_child = maybe_expand_one(node, env, cfg)
        if new_child is not None:
            path.append(new_child)
            value, solution = evaluate_leaf(
                new_child, model, env, reference_cost
            )
            incumbent.update_if_better(solution)
            backup(path, value)
            return

        action, child = select_child_puct(node, cfg.c_puct)
        node = child
        path.append(node)
```

Important implementation note: depending on exactly when `N` is incremented, compute widening using either the current `node.N` or `node.N + 1`, but use the chosen convention consistently. The specification above assumes `N` is updated during backup, so during descent the current count reflects prior completed simulations. `max(1, N)` prevents zero-width behavior before the first backup.

---

## 12. Multi-root inference pseudocode

```python
def solve_instance_with_pomo_mcts(instance, model, env, cfg):
    baseline_solution = pomo_original_inference(instance, model, env)
    validate_solution(baseline_solution, env)
    reference_cost = baseline_solution.cost

    roots = make_pomo_roots(instance, model, env, cfg)
    incumbent = Incumbent.from_solution(baseline_solution)

    budgets = equal_root_budgets(cfg.total_simulations, len(roots))

    for root, root_budget in zip(roots, budgets):
        for _ in range(root_budget):
            run_simulation(
                root=root,
                model=model,
                env=env,
                cfg=cfg,
                reference_cost=reference_cost,
                incumbent=incumbent,
            )

    validate_solution(incumbent.solution, env)
    return incumbent.solution, collect_search_statistics(roots, incumbent)
```

Version 1 may execute roots sequentially. Version 2 may interleave simulations across roots or batch model evaluations across roots.

---

## 13. Configuration

```python
@dataclass
class POMOMCTSConfig:
    total_simulations: int = 1000

    # Progressive widening.
    k0: int = 8
    c_pw: float = 2.0
    alpha_pw: float = 0.5
    endgame_customer_threshold: int = 8

    # PUCT.
    c_puct: float = 1.0
    prior_temperature: float = 1.0

    # Rollout.
    rollout_mode: str = "greedy"
    rollout_samples: int = 1

    # Multi-root execution.
    root_allocation: str = "equal"

    # Engineering / experimental reproducibility.
    seed: int = 1234
    validate_every_solution: bool = True
    store_rollout_solutions: bool = False
```

Avoid hard-coding CVRP100-specific dimensions into the MCTS engine. The engine should work for CVRP20/50/100, subject to environment/model compatibility.

---

## 14. Required tests

### 14.1 Environment and state correctness

- A child state must match the environment's result of stepping its parent state with `action_from_parent`.
- No parent-state mutation occurs when generating a child.
- Customer actions added to a node are feasible according to the environment mask.
- A depot action is created only if legal according to the environment mask.
- A depot transition changes/reset capacity exactly as in baseline environment stepping.
- Every final rollout solution is validated: all customers served exactly once, capacity constraints satisfied, valid depot behavior, and correct objective.

### 14.2 Policy initialization correctness

- Invalid actions have no stored prior and never appear in `ranked_customer_actions`.
- `ranked_customer_actions` excludes the depot.
- Customer ranking is descending in POMO prior probability.
- A legal depot action has a stored prior but does not consume customer width.
- Policy initialization runs at most once per node unless an intentionally uncached mode is enabled.

### 14.3 Progressive widening correctness

- At `N=0` or `N=1`, the node can expose `k0` customer children over successive qualifying expansion events.
- The customer-child count never exceeds `allowed_customer_width(node)`.
- The next expanded customer is exactly the next previously unexpanded customer in `ranked_customer_actions`.
- Depot child count does not affect `expanded_customer_count`.
- When feasible customer count is at most `endgame_customer_threshold`, all customers become eligible for expansion.
- No node creates duplicate children.

### 14.4 MCTS statistics correctness

- Every completed simulation increments `N` by one along its full path.
- `W` and `Q` match the backed-up rollout values.
- The returned action from PUCT maximizes the implemented score, including ties under a deterministic tie-breaking rule.
- On tiny deterministic hand-built instances, search eventually improves or matches a known greedy baseline under a sufficient simulation budget.

### 14.5 Regression comparisons

At zero simulations, return exactly the original POMO baseline solution.

At any positive budget:

- The returned solution must be feasible.
- The returned incumbent must not be worse than the stored baseline if the baseline is initialized as the incumbent.
- Record if MCTS produces no incumbent improvement; this is a valid outcome, not an error.