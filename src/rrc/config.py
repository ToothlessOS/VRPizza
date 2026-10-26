"""Configuration and result types for Random Re-Construct inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["RRCConfig", "RRCResult", "RRCStats"]


@dataclass(frozen=True)
class RRCConfig:
    """How to run RRC.

    Args:
        budgets: Iteration counts to report, sorted ascending. The loop runs once to the
            largest budget and snapshots the incumbent at each entry, so the budget curve is
            nested and paired rather than a set of independent runs.
        w_min: Minimum window length in customers. LEHD uses 4.
        w_max: Maximum window length, or ``None`` for the instance's ``N``.
        seed: Base seed for the flips, route-end choice and per-iteration ``w``. The same
            sequence is reused across budgets and checkpoints (common random numbers).
        init: ``"multistart"`` for POMO's best-of-(augmentations x starts), or ``"greedy"``
            for LEHD's single greedy construction. The multistart call also produces the
            matched plain baseline the RRC row is compared against.
        num_starts: POMO starts for the initial incumbent. ``None`` means ``N``; ``1`` means a
            single rollout.
        num_augment: Dihedral augmentations for the initial incumbent / plain baseline.
        rrc_num_augment: Dihedral augmentations for each reconstruction. ``1`` is faithful to
            LEHD; ``8`` chooses the best of eight candidates per window.
    """

    budgets: tuple[int, ...] = (50, 100, 300, 500, 1000)
    w_min: int = 4
    w_max: int | None = None
    seed: int = 12
    init: str = "multistart"
    num_starts: int | None = None
    num_augment: int = 8
    rrc_num_augment: int = 1

    def __post_init__(self) -> None:
        budgets = tuple(sorted({int(budget) for budget in self.budgets if int(budget) > 0}))
        if not budgets:
            raise ValueError("at least one positive RRC budget is required")
        object.__setattr__(self, "budgets", budgets)
        if self.init not in ("multistart", "greedy"):
            raise ValueError(f"init must be 'multistart' or 'greedy', got {self.init!r}")
        if self.w_min < 2:
            raise ValueError(f"w_min must be at least 2, got {self.w_min}")
        if self.w_max is not None and self.w_max < self.w_min:
            raise ValueError(f"w_max={self.w_max} is below w_min={self.w_min}")
        for name, value in (
            ("num_augment", self.num_augment),
            ("rrc_num_augment", self.rrc_num_augment),
        ):
            if value not in (1, 8):
                raise ValueError(f"{name} must be 1 or 8 (dihedral8 needs exactly 8), got {value}")
        if self.num_starts is not None and self.num_starts <= 0:
            raise ValueError(f"num_starts must be positive, got {self.num_starts}")

    @property
    def max_budget(self) -> int:
        """Largest iteration count; the loop runs this far once."""
        return int(self.budgets[-1])


@dataclass
class RRCStats:
    """Compute and acceptance accounting for one RRC run."""

    iterations: int = 0
    accepted: int = 0
    decoder_rows: int = 0
    seconds: float = 0.0

    def merge(self, other: "RRCStats") -> None:
        """Accumulate another run's counters."""
        self.iterations += other.iterations
        self.accepted += other.accepted
        self.decoder_rows += other.decoder_rows
        self.seconds += other.seconds


@dataclass
class RRCResult:
    """Per-instance costs before and after each budget.

    Attributes:
        initial_cost: ``[B]`` exact full-tour length of the initial incumbent, in the model's
            coordinate units.
        budget_costs: ``[B, len(budgets)]`` incumbent cost after each budget.
        stats: Acceptance and compute counters.
    """

    initial_cost: np.ndarray
    budget_costs: np.ndarray
    stats: RRCStats = field(default_factory=RRCStats)
