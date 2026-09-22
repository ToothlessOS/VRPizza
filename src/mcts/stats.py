"""Cost accounting for the search.

The specification asks for decoder calls and wall-clock time alongside the simulation count,
for a specific reason: a simulation count is not a comparable unit of work. One simulation at a
root is a full greedy completion of roughly ``N + R`` decode steps, one at depth 20 is roughly
``N + R - 20``, and the batched form runs many of them per decoder call. Reporting only "sims"
would make the search look cheaper than it is and make two budgets incomparable.

The counter to keep in mind when reading these numbers is that the batch is not constant. A
decoder call covers every root of every instance in the current sub-batch, so ``decoder_calls``
being low is not by itself evidence of cheapness -- ``decoder_rows`` is the quantity that
corresponds to work, and ``decoder_rows / decoder_calls`` is the realised batch width, which is
what says whether the GPU was actually used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

__all__ = ["SearchStatistics"]


@dataclass
class SearchStatistics:
    """Everything the specification's section 3 asks to report, accumulated over a run.

    Fields:
        instances: Instances searched.
        roots: Tree roots, i.e. ``num_augment * num_starts`` per instance.
        simulations: Simulations actually executed.
        expansions: Simulations that materialised a new child instead of descending.
        terminal_evaluations: Simulations whose selected node was already a complete solution.
        puct_selections: Descents through an existing child by PUCT. This counts *levels*, not
            simulations: one descent can pass through many saturated nodes, so it is bounded
            below by nothing and above by the total tree depth walked.
        decoder_calls: Calls into the policy's decoder, across priors, rollouts and baseline.
        decoder_rows: Rows decoded, summed over those calls.
        rollouts: Completed solutions evaluated.
        improvements: Rollouts that beat the incumbent.
        invalid_rollouts: Rollouts that failed to terminate and were discarded.
        baseline_seconds: Wall clock in the baseline pass, which the search needs anyway as its
            reference cost and incumbent seed.
        search_seconds: Wall clock in the simulation rounds.
        solutions: Rollout action sequences, only when ``store_rollout_solutions`` is set.

    ``puct_selections`` is the field to read first when a budget produces no improvement. The
    widening rule holds the frontier at ``k0`` until a node has been visited 21 times, and the
    depot child is created at initialisation without consuming a customer slot, so the first
    eight simulations at a root each materialise a child and return; PUCT selection does not run
    at all until the ninth. A run whose ``puct_selections`` is zero spent its whole budget
    building depth-1 trees, and its result says nothing about the search.
    """

    instances: int = 0
    roots: int = 0
    simulations: int = 0
    expansions: int = 0
    terminal_evaluations: int = 0
    puct_selections: int = 0
    decoder_calls: int = 0
    decoder_rows: int = 0
    rollouts: int = 0
    improvements: int = 0
    invalid_rollouts: int = 0
    baseline_seconds: float = 0.0
    search_seconds: float = 0.0
    solutions: list = field(default_factory=list)

    def record_decoder_call(self, rows: int) -> None:
        """Count one decoder invocation covering ``rows`` rows."""
        self.decoder_calls += 1
        self.decoder_rows += int(rows)

    @property
    def total_seconds(self) -> float:
        """Baseline plus search wall clock."""
        return self.baseline_seconds + self.search_seconds

    @property
    def mean_batch_width(self) -> float:
        """Realised rows per decoder call; the batching efficiency of the run."""
        if not self.decoder_calls:
            return 0.0
        return self.decoder_rows / self.decoder_calls

    @property
    def decoder_rows_per_instance(self) -> float:
        """Rows decoded per instance, for comparison against the baseline's single pass."""
        if not self.instances:
            return 0.0
        return self.decoder_rows / self.instances

    def merge(self, other: SearchStatistics) -> None:
        """Fold another statistics object into this one, in place."""
        self.instances += other.instances
        self.roots += other.roots
        self.simulations += other.simulations
        self.expansions += other.expansions
        self.terminal_evaluations += other.terminal_evaluations
        self.puct_selections += other.puct_selections
        self.decoder_calls += other.decoder_calls
        self.decoder_rows += other.decoder_rows
        self.rollouts += other.rollouts
        self.improvements += other.improvements
        self.invalid_rollouts += other.invalid_rollouts
        self.baseline_seconds += other.baseline_seconds
        self.search_seconds += other.search_seconds
        self.solutions.extend(other.solutions)

    def as_dict(self) -> dict:
        """Flatten for storage, dropping the per-solution list."""
        data = asdict(self)
        data.pop("solutions", None)
        data["total_seconds"] = self.total_seconds
        data["mean_batch_width"] = self.mean_batch_width
        data["decoder_rows_per_instance"] = self.decoder_rows_per_instance
        return data

    def format_lines(self, prefix: str = "") -> list[str]:
        """Human-readable summary lines for the run report."""
        return [
            f"{prefix}simulations        {self.simulations:>10d}  over {self.roots} roots",
            f"{prefix}expansions         {self.expansions:>10d}  "
            f"({self.terminal_evaluations} terminal, PUCT levels {self.puct_selections})",
            f"{prefix}decoder calls      {self.decoder_calls:>10d}",
            f"{prefix}decoder rows       {self.decoder_rows:>10d}  "
            f"({self.decoder_rows_per_instance:.0f}/instance, "
            f"{self.mean_batch_width:.0f} rows/call)",
            f"{prefix}rollouts           {self.rollouts:>10d}  "
            f"({self.improvements} improved, {self.invalid_rollouts} invalid)",
            f"{prefix}wall clock         {self.total_seconds:>10.1f}s  "
            f"(baseline {self.baseline_seconds:.1f}s, search {self.search_seconds:.1f}s)",
        ]
