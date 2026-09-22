# The CVRP action space

Notes on the action space of the CVRP environments, written for anyone reading or changing
`notebooks/eval_pomo.py`, `src/envs/cvrp.py` or `src/envs/cvrp_eval.py`. Everything here was
checked against the installed rl4co (0.6.0, torch 2.14.0) at revision `362fba1`, by driving the
environment directly; the probe snippets are at the end so the claims can be re-checked after a
dependency bump rather than believed.

This document covers the *space*. For how to run the evaluation that drives it, and how to read
its tables, see `docs/EvalPomo.md`.

## The space is rl4co's, unchanged

Neither `CVRPEnv` nor `EvalCVRPEnv` overrides anything action-related. Both are
`InstanceInputMixin` plus rl4co's `CVRPEnv`, and the mixin supplies only `make_input_td` and
`reference_cost` — data plumbing. The action space is therefore rl4co's verbatim, and it is not
a place to look for differences between the SL and RL paths: those two differ in their data and
their objective, not in their actions.

The definition is `rl4co/envs/routing/cvrp/env.py:133-144` (`get_action_mask`) together with
`env.py:68-100` (`_step`). It is a masked, discrete, variable-length sequence over `N+1`
actions per step:

| action | meaning | `locs` column (after `reset`) | `demand` column |
| --- | --- | --- | --- |
| `0` | return to the depot, ending the current route | `0` | none — clamped to `0` and zeroed by the `(current_node != 0)` factor |
| `k`, `1..N` | serve customer `k` | `k` | `k - 1` |

The `-1` shift is the one thing worth internalising: `_reset` prepends the depot to `locs`
(`env.py:112`) so `locs[k]` is customer `k`, while `demand` stays customer-only, so
`_step` reads `demand[k-1]` (`env.py:73-75`). The file's own 1-based customer numbering is
preserved throughout, which is why the supervised trajectories in `src/envs/dataset.py` need no
remapping.

The mask is `[B, N+1]` bool and it **is** the space — the policy softmaxes over `-inf`-masked
logits (`rl4co/models/nn/attention.py:32-34`), so `action_spec` is never consulted. Note that
`action_spec.high` is `num_loc + 1` (`env.py:220-225`), one past the largest legal action; it
prints `5` for `num_loc=4`. Harmless today, a landmine if a torchrl sampler is ever pointed at
the environment.

## The three masking rules

1. `exceeds_cap = demand + used_capacity > vehicle_capacity + 1e-5`. The `1e-5` slack permits a
   solution to exceed capacity by that relative amount.
2. A customer is masked if it is already visited or does not fit in the remaining capacity.
3. The depot is masked **only** when the vehicle is already at the depot with at least one
   feasible customer left. At a customer the depot is always available, so a route may be cut
   short at any point.

Two structural consequences:

**The mask can never be all-`False`.** At a customer the depot is a universal fallback; at the
depot either a customer is feasible or the depot itself is. There is no state that yields a
NaN-logit softmax.

**No two depot actions can be consecutive**, which is what makes the route count `R`
unambiguous. The depot is overloaded — it means both "refill and start another route" and
"return home at the end" — and the mask is the only thing separating them.

## Termination, reward, and trajectory length

`done = visited.sum(-1) == visited.size(-1)`, where `visited` is `[B, N+1]` and includes a slot
for the depot (`env.py:84-87`, `env.py:121-125`). It follows that:

* If the vehicle has touched the depot at any earlier point, the episode ends on the **last
  customer** and no closing depot action is taken.
* If it has not — the single-route case, where the whole instance fits in one vehicle — one
  final depot step is taken, forced by the mask since every customer is already visited.

So the action sequence is `N + R - 1` long when `R >= 2` and `N + R` when `R == 1`. This matches
the "the decoder consumes only `N + R - 1` of them" note on
`CVRPEnv.build_expert_actions` for the multi-route case, and is one step longer for
single-route instances. The discrepancy costs nothing, for two reasons:

* **The reward is identical either way.** `_get_reward` computes
  `-get_tour_length([depot] + locs[actions])`, and `get_tour_length` rolls the sequence
  cyclically (`rl4co/utils/ops.py:92`). Since the sequence starts at the depot, that closure
  edge supplies the missing return leg. Measured: serving `1 | depot | 2,3,4` on a four-customer
  instance gives `-9.478708` both with and without an appended depot action, against a
  hand-computed closed tour of `-9.478708664619075`.
* **The final depot step is mask-forced anyway**, so it carries no information the model could
  learn. The supervised path drops it; the rollout takes it. Nothing differs.

The practical caution: step-indexed code cannot assume a fixed trajectory length, and
`check_solution_validity` tolerates both, since it only requires `T >= N` with `0`s in the
excess (`env.py:157-185`).

## The unit coupling that actually matters

Everything above rests on `demand` and `vehicle_capacity` being in the same units, and that
rests on exactly one line: the `demand / batch["capacity"][:, None]` in
`InstanceInputMixin.make_input_td`. Demands arrive from the files in their own units and are
normalised per instance by that instance's own capacity; `vehicle_capacity` is then read from
`generator.vehicle_capacity`, hard-wired to `1.0` by `CVRPGenerator`'s default
(`rl4co/envs/routing/cvrp/generator.py:65,74`) and never overridden here.

The `capacity=` argument that both environments accept does **not** set it. In this rl4co
version `CVRPGenerator.__init__` takes `vehicle_capacity` and `capacity` as separate parameters,
and only `capacity` is assigned from the kwarg. Verified:
`EvalCVRPEnv(num_loc=5, capacity=100.0)` yields `generator.vehicle_capacity == 1.0` and
`generator.capacity == 100.0`. So `eval_pomo.py:347` passing `inst.capacity[positions].max()`
cannot widen the mask, and neither can the training environment's.

`capacity` is not dead, though — it is just not part of the action space. It sets the
normalisation scale in `CVRPGenerator._generate`, which is what keeps RL's generated training
instances commensurate with the file-derived validation instances. For `EvalCVRPEnv`, which
never generates, it is inert.

> **Known inaccuracy.** The docstring at `src/envs/cvrp_eval.py:44-47` says passing a capacity
> "would leave the mask comparing normalised demands against a full-scale capacity, which would
> make every action appear feasible". That is not what happens, and it contradicts the accurate
> account at `src/envs/cvrp.py:128-131`. The risk is directional: a reader who believes the
> docstring may "correct" the code to pass `vehicle_capacity=`, which *would* genuinely open the
> mask. The wording should be fixed.

## The step-0 exception: with multi-start, the first action is not sampled

Under the `multistart` decode type, `pre_decoder_hook` assigns the first action as
`arange(num_starts) % num_loc + 1` and pins its log-probability to `0`
(`rl4co/utils/decoding.py:306-320`). It does not consult the mask. With `num_starts == N`, step 0
is therefore a fixed permutation — each customer starts exactly one rollout — not a draw from
the space.

Two things follow. First, `num_loc` must equal the batch's `N`, or
`select_start_nodes` silently confines every start to the first `num_loc` customers and weakens
the rollout; `cvrp_eval.py:34-40` documents this and `eval_pomo.py:345-347` honours it by
building one environment per size group. Second, and less obviously, the "best of `8 x N`"
claim depends on it.

Because the forced start bypasses the mask, a customer whose demand exceeds capacity would be
forced anyway — and the failure mode is not an exception. The rollout parks at the depot:
with no feasible customer, the depot is the only allowed action, `done` never fires, and the
step limit is reached with an incomplete tour. Verified: with all four demands at `1.5` against
`vehicle_capacity = 1.0`, the mask is `[1,0,0,0,0]` at every step and stays there. Since
`EvalCVRPEnv` defaults `check_solution=False`, this would be scored silently.

The protection is `build_batch`'s demand-versus-capacity guard (`notebooks/eval_pomo.py:234`),
which lives in the driver rather than the environment. It is load-bearing. Passing
`check_solution=True` on the `--limit 8` smoke run is the cheap way to confirm it holds, since
the check is a Python loop over decode steps.

## Other things that are not bugs

* **Early depot returns are always legal** (rule 3), so the rollout's route count is not bounded
  by the reference `R`. A weak policy pays for the extra routes in tour length, which is the
  right way to score it.
* **The `1e-5` capacity slack** means a solution may exceed capacity by that relative amount.
  Immaterial at these scales, but it is why an exact-capacity fill does not get masked.
* **Reset mutates its input.** `_reset` widens `locs` in place, which is why `make_input_td`
  returns a fresh TensorDict every call and why `eval_pomo.py:360-362` builds a separate batch
  per model instead of sharing one.

## Re-checking these claims

Each snippet is standalone; run with the project virtualenv (`.venv/bin/python`).

```python
import torch
from tensordict.tensordict import TensorDict
from src.envs.cvrp_eval import EvalCVRPEnv

env = EvalCVRPEnv(num_loc=4, capacity=100.0)
print(env.generator.vehicle_capacity, env.generator.capacity, env.action_spec.high.item())
# 1.0 100.0 5  -> the kwarg sets `capacity`, the mask's bound stays 1.0, spec is off by one

batch = TensorDict({
    "depot": torch.zeros(1, 2),
    "locs": torch.tensor([[[1., 0.], [1., 1.], [0., 1.], [2., 2.]]]),
    "demand": torch.tensor([[40., 40., 40., 40.]]),   # normalised to 0.4 on the way in
    "capacity": torch.tensor([100.]),
}, batch_size=[1])

td = env.reset(env.make_input_td(batch))
print(td["action_mask"].int().tolist(), td["vehicle_capacity"].flatten().tolist())
# [[0, 1, 1, 1, 1]] [1.0]  -> depot masked at the start, all customers feasible

for a in (1, 0, 2, 3, 4):                             # two routes, so no final depot
    td.set("action", torch.tensor([a]))
    td = env.step(td)["next"]
    print(a, bool(td["done"].item()), td["action_mask"].int().tolist()[0])
# 1 False [1,0,1,1,1]   depot available at a customer
# 0 False [0,0,1,1,1]   no consecutive depots; used_capacity back to 0
# 2 False [1,0,0,1,1]
# 3 False [1,0,0,0,0]
# 4 True  [1,0,0,0,0]   ends on the last customer, all customers visited

acts = torch.tensor([[1, 0, 2, 3, 4]])
print(env.get_reward(td.clone(), acts).item(),
      env.get_reward(td.clone(), torch.tensor([[1, 0, 2, 3, 4, 0]])).item())
# -9.478708267211914 -9.478708267211914
# -> identical, i.e. the missing return leg comes from the cyclic roll. Hand-computed closed
#    tour: -9.478708664619075 (the gap is float32).
```
