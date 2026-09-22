"""The network-facing half of the search: priors and completions from a pretrained POMO policy.

Everything that touches the model lives here, so the tree logic in :mod:`src.mcts.search` is
testable against a stub and the rl4co coupling stays in one file. Two properties of that
coupling are worth stating because the whole design rests on them.

**The decoder is stateless with respect to decode progress.** ``policy.decoder(td, cached,
num_starts)`` takes the current environment state and a *static* ``PrecomputedCache`` built once
from ``policy.encoder``; there is no incremental key/value cache to keep in step. A tree node can
therefore be decoded exactly as it stands, with no replay of its prefix, and one cache serves
every node of a tree, since the cache depends only on the instance's augmented coordinates.

**Finished rows need no special handling in a batched rollout.** An episode ends when every
customer is served, and at that point the environment's mask leaves the depot as the *only*
feasible action -- verified against ``CVRPEnv.get_action_mask`` at ``done=True``, where the mask
is ``[True, False, ...]``. So a row that finishes early keeps taking a cost-neutral depot no-op
while its siblings run, rather than producing an all-masked softmax and a NaN. That is what lets
leaves at different depths share one batch, which is what makes the round loop cheap.

The probability handling deliberately reuses rl4co's own ``process_logits`` rather than
reimplementing the mask-then-softmax step, so the priors and the rollout cannot drift from what
the checkpoints were validated with. Note it rebinds its input when ``tanh_clipping > 0`` and
mutates in place only when ``tanh_clipping == 0``, so the decoder's returned tensor must not be
kept either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict.tensordict import TensorDict
from torch import Tensor

from src.mcts.config import POMOMCTSConfig
from src.mcts.stats import SearchStatistics

__all__ = ["CompletionResult", "POMOOracle"]

#: Decode steps a completion may take before it is declared stuck. A well-formed CVRP episode
#: needs at most ``N + R`` steps with one route per customer, so ``2N`` is generous; exceeding it
#: means the state was parked (a demand larger than the capacity freezes the mask on the depot
#: and ``done`` never fires), which is possible for callers that do not screen their batches the
#: way ``notebooks/eval_pomo.py`` does.
_MAX_STEPS_FACTOR = 2


@dataclass
class CompletionResult:
    """One completed solution per row.

    Args:
        cost: ``[L]`` total route length.
        actions: ``L`` full action sequences, root prefix included, no padding.
        valid: ``[L]`` bool, False for rows that hit the step cap without terminating. Invalid
            rows carry a cost of infinity so a caller that forgets the flag still cannot select
            them.
    """

    cost: Tensor
    actions: list[list[int]]
    valid: Tensor


class POMOOracle:
    """Policy priors and greedy completions for a pretrained POMO checkpoint.

    Args:
        model: A fitted ``POMOSL`` or ``POMORL``, in eval mode.
        env: The CVRP environment the model is pointed at.
        cfg: Search configuration.
        stats: Optional statistics sink; every decoder call is recorded here.
    """

    def __init__(
        self,
        model,
        env,
        cfg: POMOMCTSConfig,
        stats: SearchStatistics | None = None,
    ) -> None:
        self.model = model
        self.policy = model.policy
        self.env = env
        self.cfg = cfg
        self.stats = stats

        # Fall back to the policy's own preprocessing so priors and rollouts agree; the spec's
        # literal softmax(z) is available by setting the config field to 0.0.
        self.tanh_clipping = (
            self.policy.tanh_clipping
            if cfg.prior_tanh_clipping is None
            else cfg.prior_tanh_clipping
        )

    # -- caching ---------------------------------------------------------------------

    def encode(self, td_aug: TensorDict):
        """Encode augmented roots once and precompute the decoder cache.

        The decoder's cache depends only on the encoder's output for the augmented coordinates,
        so this is the single encoder pass for a whole forest: every node of every tree rooted at
        one of these rows reuses its row.

        Args:
            td_aug: ``[B*A]``-batched reset-and-augmented environment state.

        Returns:
            A ``PrecomputedCache`` with one row per augmented instance.
        """
        hidden, _ = self.policy.encoder(td_aug)
        return self.policy.decoder._precompute_cache(hidden, num_starts=1)

    @torch.no_grad()
    def priors(self, td: TensorDict, cached) -> Tensor:
        """Masked action probabilities at every row of a batch.

        Args:
            td: ``[L]``-batched environment state.
            cached: The matching ``PrecomputedCache`` rows.

        Returns:
            ``[L, N+1]`` probabilities summing to one per row, exactly zero on infeasible
            actions. Row ``i`` is normalised *given* that row's mask, so the values are the
            policy's conditional distribution rather than a joint one.
        """
        logits, mask = self.policy.decoder(td, cached, num_starts=1)
        self._count(td.shape[0])
        logprobs = self._log_probs(logits, mask, temperature=self.cfg.prior_temperature)
        return logprobs.exp()

    def _log_probs(self, logits: Tensor, mask: Tensor, temperature: float) -> Tensor:
        """rl4co's own mask-then-softmax, so priors equal what the rollout uses."""
        from rl4co.utils.decoding import process_logits

        return process_logits(
            logits,
            mask,
            temperature=temperature,
            tanh_clipping=self.tanh_clipping,
            mask_logits=self.policy.mask_logits,
        )

    # -- rollouts --------------------------------------------------------------------

    @torch.no_grad()
    def complete(
        self,
        td: TensorDict,
        cached,
        prefixes: list[list[int]],
    ) -> CompletionResult:
        """Finish partial solutions greedily, in one batch.

        Args:
            td: ``[L]``-batched state to complete from. Mutated in place by the stepping; the
                ``locs`` it carries are what the objective is measured on, and those are never
                written.
            cached: The matching cache rows.
            prefixes: ``L`` action sequences from each row's root through its current state. They
                are prepended to the completion because ``env.get_reward`` scores a whole
                trajectory from the root, not a suffix.

        Returns:
            The completed solutions. See :class:`CompletionResult`.
        """
        samples = max(1, self.cfg.rollout_samples) if self._is_best_of() else 1
        if samples > 1:
            td = self._repeat(td, samples)
            cached = self._repeat_cache(cached, samples)
            prefixes = [p for p in prefixes for _ in range(samples)]

        rows = td.shape[0]
        collected: list[list[int]] = [[] for _ in range(rows)]
        done = td["done"].clone()
        cap = _MAX_STEPS_FACTOR * int(td["demand"].shape[-1])

        steps = 0
        while not bool(done.all()) and steps < cap:
            logits, mask = self.policy.decoder(td, cached, num_starts=1)
            self._count(rows)
            logprobs = self._log_probs(
                logits, mask, temperature=self.cfg.rollout_temperature
            )
            action = self._select(logprobs)
            # Transfer both vectors once per step rather than reading them row by row. Indexing
            # a device tensor is a synchronisation, and at thousands of rows over a hundred-odd
            # steps that is hundreds of thousands of them -- enough to dominate the wall clock
            # of a search that is otherwise a pure tensor pipeline.
            finished = done.squeeze(-1).tolist() if done.dim() > 1 else done.tolist()
            chosen = action.tolist()
            for row in range(rows):
                if not finished[row]:
                    collected[row].append(chosen[row])
            td.set("action", action)
            self.env.step(td)
            done = td["done"]
            steps += 1

        # A row still unfinished after the cap never terminated; park its action so the sequence
        # stays well formed, and mark it invalid below rather than letting it pass as a solution.
        hit_cap = ~done.squeeze(-1) if done.dim() > 1 else ~done
        valid = ~hit_cap

        full = [prefixes[row] + collected[row] for row in range(rows)]
        width = max(len(seq) for seq in full) if full else 0
        padded = torch.zeros(rows, width, dtype=torch.long, device=td.device)
        for row, seq in enumerate(full):
            if seq:
                padded[row, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=td.device)

        cost = -self.env.get_reward(
            td, padded, check_solution=bool(self.cfg.validate_every_solution and width)
        )
        cost = torch.where(valid, cost, torch.full_like(cost, float("inf")))

        if samples > 1:
            return self._best_of(cost, full, valid, samples)
        return CompletionResult(cost=cost, actions=full, valid=valid)

    def _is_best_of(self) -> bool:
        return self.cfg.rollout_mode == "best_of_n_samples"

    def _select(self, logprobs: Tensor) -> Tensor:
        """Greedy unless the rollout is stochastic."""
        if self.cfg.rollout_mode in ("sample", "best_of_n_samples"):
            return torch.multinomial(logprobs.exp(), 1).squeeze(-1)
        return logprobs.argmax(dim=-1)

    def _best_of(
        self, cost: Tensor, actions: list[list[int]], valid: Tensor, samples: int
    ) -> CompletionResult:
        """Reduce ``samples`` draws per row back to one, keeping the cheapest."""
        rows = cost.shape[0] // samples
        cost = cost.view(rows, samples)
        valid = valid.view(rows, samples)
        best = cost.argmin(dim=1)
        keep = [actions[row * samples + int(best[row])] for row in range(rows)]
        return CompletionResult(
            cost=cost.gather(1, best.unsqueeze(1)).squeeze(1),
            actions=keep,
            valid=valid.gather(1, best.unsqueeze(1)).squeeze(1),
        )

    @staticmethod
    def _repeat(td: TensorDict, samples: int) -> TensorDict:
        """Tile a batch along a new sampling axis, sample-major like rl4co's own ``batchify``."""
        return TensorDict(
            {key: value.repeat_interleave(samples, dim=0) for key, value in td.items()},
            batch_size=[td.shape[0] * samples],
            device=td.device,
        )

    @staticmethod
    def _repeat_cache(cached, samples: int):
        """Tile a cache the same way :meth:`_repeat` tiles its state."""
        fields = ("node_embeddings", "graph_context", "glimpse_key", "glimpse_val", "logit_key")
        return type(cached)(
            **{
                name: (
                    value.repeat_interleave(samples, dim=0) if torch.is_tensor(value) else value
                )
                for name, value in zip(fields, cached.fields)
            }
        )

    # -- bookkeeping -----------------------------------------------------------------

    def _count(self, rows: int) -> None:
        if self.stats is not None:
            self.stats.record_decoder_call(int(rows))
