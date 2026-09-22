"""Configuration for the POMO-guided CVRP search.

The field names follow the specification in ``prompts/MCTS.md`` section 13 so the ablation
vocabulary in that document maps onto this dataclass one-to-one. Two of the spec's fields carry
meanings that need stating rather than assuming:

``sims_per_root`` versus ``total_simulations``
    The spec allots a total budget ``B`` equally across roots, ``B_r = floor(B / R)``, with
    ``R = 8N`` roots per instance. At ``vrp100`` that is 800 roots, so the spec's default of
    1000 simulations gives every root exactly one, and for ``N >= 126`` it gives every root
    zero -- the search then never calls the model and silently returns the baseline. A budget
    that low is also below the point where the search does anything: with the recommended
    widening constants the frontier stays at ``k0`` until a node has been visited 21 times, so
    the first eight simulations at a root each simply create one child and PUCT selection does
    not run until the ninth. ``sims_per_root`` is therefore the real knob and
    :meth:`POMOMCTSConfig.total_simulations` derives the spec's quantity from it.

``prior_temperature``
    The spec writes the prior as ``softmax(z)``. What this module actually computes is
    ``softmax(tanh_clipping * tanh(z) / tau)``, because rl4co's ``process_logits`` applies the
    policy's tanh clipping before the softmax and both checkpoints carry ``tanh_clipping=10.0``.
    Tanh is strictly monotone, so the spec's section 7.4 *ranking* -- which is all progressive
    widening depends on -- is preserved exactly, but the distribution is not the spec's. Set
    ``prior_tanh_clipping=0.0`` to get the spec's form literally.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["POMOMCTSConfig"]


@dataclass
class POMOMCTSConfig:
    """Search hyperparameters, defaults as recommended by the specification.

    Args:
        sims_per_root: Simulations per root. This, not ``total_simulations``, is the budget.
            The default of 32 clears both thresholds that make the search meaningful: PUCT
            selection first runs at 9 (the depot child is created at initialisation and does
            not consume a customer slot, so the first eight simulations at a root each create
            one customer child and return), and progressive widening first exceeds ``k0`` at
            21. It remains well short of the endgame override, which needs a node deep enough
            that at most ``endgame_customer_threshold`` customers remain.
        k0: Initial customer frontier size.
        c_pw: Widening coefficient.
        alpha_pw: Widening exponent, strictly between 0 and 1.
        endgame_customer_threshold: Feasible-customer count at or below which every customer
            becomes expandable, bypassing the widening rule.
        expansion: ``"one_child"`` (spec default, section 7.5), ``"progressive"`` to materialise
            the whole permitted frontier at once, or ``"fixed_k"`` to hold the frontier at
            ``k0`` regardless of visits -- the no-widening ablation.
        c_puct: PUCT exploration weight. Note this trades off directly against the scale of the
            backed-up value: ``argmax(k*Q + c*P*U) == argmax(Q + (c/k)*P*U)``, so scaling the
            value and scaling ``c_puct`` are the same adjustment. With the spec's normalised
            value the spread is ~0.02-0.06 while the bonus at a root child is ~0.05-0.4, so at
            the default 1.0 the search is close to prior-ordered breadth expansion and ``Q``
            barely discriminates. The default is the spec's; the first ablation to run is a
            sweep of this field.
        prior_temperature: Prior temperature ``tau_p``. Below 1 sharpens, above 1 flattens.
        prior_tanh_clipping: Tanh clipping for the prior computation. Defaults to the policy's
            own value so priors and rollouts agree; 0.0 gives the spec's literal ``softmax(z)``.
        rollout_mode: ``"greedy"``, ``"sample"`` or ``"best_of_n_samples"``.
            ``"greedy_plus_local_search"`` is accepted by the type but raises, because no local
            search exists in this repository (rl4co's CVRP local search needs a C++ extension
            that is not built here).
        rollout_samples: Sample count for ``"best_of_n_samples"``.
        rollout_temperature: Temperature for stochastic rollouts.
        value_form: ``"normalized_improvement"`` for the spec's section 8.1
            ``(C_ref - C) / C_ref``, or ``"negative_cost"`` for ``-C``.
        reference: What ``C_ref`` is measured against. ``"baseline_best"`` is the spec's
            section 12 reading (the ordinary POMO rollout's best solution), ``"baseline_mean"``
            the mean over roots. With ``"baseline_best"`` almost every leaf has a negative
            value, since beating the best of 8N rollouts is rare; ``"baseline_mean"`` centres
            the value near zero, which makes ``Q`` informative earlier.
        seed: Seed for stochastic rollouts.
        validate_every_solution: Check every rollout against the environment's own solution
            validator. The specification recommends ``True``; it defaults to ``False`` here for
            a measured cost reason, and turning it on is the right first move for a smoke test.
            ``check_solution_validity`` is a Python loop over decode steps, so validating a
            round costs one interpreter iteration per row per step: at 32 instances x 8N roots
            that is millions of iterations *per round*, which outweighs the search itself. The
            driver validates the final incumbent unconditionally instead, which is one check per
            instance rather than one per rollout.
        store_rollout_solutions: Keep every rollout's action sequence on the statistics object.
            Off by default: at 8N roots and 32 simulations that is ~25x the solutions retained.
        keep_tree: Attach the root nodes to the result so a caller can inspect the tree. Off by
            default because the tree is large -- 8N roots times ``sims_per_root`` nodes -- and a
            benchmark has no use for it. The verification harness turns it on.
    """

    sims_per_root: int = 32

    # Progressive widening.
    k0: int = 8
    c_pw: float = 2.0
    alpha_pw: float = 0.5
    endgame_customer_threshold: int = 8
    expansion: str = "one_child"

    # PUCT.
    c_puct: float = 1.0
    prior_temperature: float = 1.0
    prior_tanh_clipping: float | None = None  # None means "use the policy's own value".

    # Rollout.
    rollout_mode: str = "greedy"
    rollout_samples: int = 1
    rollout_temperature: float = 1.0

    # Value.
    value_form: str = "normalized_improvement"
    reference: str = "baseline_best"

    # Root selection.
    max_starts: int | None = None  # None means all N starts per augmentation.

    # Engineering.
    seed: int = 1234
    validate_every_solution: bool = False
    store_rollout_solutions: bool = False
    keep_tree: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha_pw < 1.0:
            raise ValueError(f"alpha_pw must lie strictly between 0 and 1, got {self.alpha_pw}")
        if self.sims_per_root < 0:
            raise ValueError(f"sims_per_root must be non-negative, got {self.sims_per_root}")
        if self.expansion not in ("one_child", "progressive", "fixed_k"):
            raise ValueError(f"unknown expansion {self.expansion!r}")
        if self.rollout_mode not in (
            "greedy",
            "sample",
            "best_of_n_samples",
            "greedy_plus_local_search",
        ):
            raise ValueError(f"unknown rollout_mode {self.rollout_mode!r}")
        if self.value_form not in ("normalized_improvement", "negative_cost"):
            raise ValueError(f"unknown value_form {self.value_form!r}")
        if self.reference not in ("baseline_best", "baseline_mean"):
            raise ValueError(f"unknown reference {self.reference!r}")

    def total_simulations(self, num_roots: int) -> int:
        """The spec's ``B``, derived from :attr:`sims_per_root` and the root count.

        Args:
            num_roots: Roots per instance, ``num_augment * num_starts``.

        Returns:
            The total simulation budget the spec's section 13 would have carried.
        """
        return self.sims_per_root * int(num_roots)

    def customer_width(self, feasible_customers: int, visits: int) -> int:
        """Maximum number of materialised customer children, per the spec's section 7.1-7.2.

        ``K(s) = min(m, max(k0, floor(c_pw * max(1, N)^alpha)))``, with the endgame override
        ``K(s) = m`` once ``m <= endgame_customer_threshold``.

        Args:
            feasible_customers: ``m``, the number of feasible customer actions at the node.
            visits: ``N(s)``, the node's visit count.

        Returns:
            The permitted number of customer children. Zero when the node has no feasible
            customer, which the caller must treat as "nothing to expand" rather than as a
            width of zero to grow into.
        """
        if feasible_customers <= 0:
            return 0
        if feasible_customers <= self.endgame_customer_threshold:
            return feasible_customers
        if self.expansion == "fixed_k":
            return min(feasible_customers, self.k0)
        widened = int(self.c_pw * max(1, visits) ** self.alpha_pw)
        return min(feasible_customers, max(self.k0, widened))
