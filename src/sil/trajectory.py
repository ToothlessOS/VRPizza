"""The bridge between Gumbeldore's SBS states and an rl4co CVRP policy.

The search code in :mod:`src.sil.sbs` and :mod:`src.sil.incremental_sbs` is framework- and
problem-agnostic: it asks for child log-probabilities of a batch of states and for the
transition of a batch of ``(state, action)`` pairs. This module supplies both for a POMO
policy over :class:`~rl4co.envs.routing.cvrp.env.CVRPEnv`.

Two properties of the policy make the adapter cheap:

* The encoder depends only on the instance, and the decoder is stateless with respect to decode
  progress, so one ``PrecomputedCache`` per chunk serves every node of every tree. This is the
  same property :class:`~src.mcts.oracle.POMOOracle` relies on.
* rl4co's CVRP action space is a fixed ``[N+1]`` axis at every depth (``0`` is the depot), so
  states at different depths share one action ordering. Gumbeldore's own CVRP trajectory needs
  variable-length action vectors; here no re-indexing or same-level restriction is needed.

States are kept on the CPU as small NumPy arrays. Only the decoder input and the environment
step are moved to the device, one bulk transfer each per SBS level; per-state tensors would
create tens of thousands of tiny allocations and synchronisations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from rl4co.utils.decoding import process_logits

__all__ = ["SILSearchAdapter", "SILState"]

#: Field order of rl4co's ``PrecomputedCache``. `fields` returns the values, not the names.
_CACHE_FIELDS = ("node_embeddings", "graph_context", "glimpse_key", "glimpse_val", "logit_key")


@dataclass(frozen=True)
class SILState:
    """One partial CVRP solution, as seen by the search.

    Args:
        instance: Row of the generation chunk this state belongs to.
        visited: ``[N+1]`` uint8, index ``0`` is the depot slot.
        current: Current node index, ``0`` at the depot.
        used: Normalised load of the current vehicle.
        mask: ``[N+1]`` bool, ``True`` where the action is feasible (rl4co's convention).
        done: Whether the episode has terminated.
        actions: Actions from the root through this state, inclusive.
        cost: Cumulative Euclidean length, including the implicit final return edge for
            terminal states.
    """

    instance: int
    visited: np.ndarray
    current: int
    used: float
    mask: np.ndarray
    done: bool
    actions: tuple[int, ...]
    cost: float

    def to_max_evaluation(self) -> float:
        """Outcome to maximise: minus the tour length."""
        return -self.cost


def _index_cache(cached, rows: torch.Tensor):
    """Select cache rows, leaving non-tensor fields (the graph context) untouched."""
    return type(cached)(
        **{
            name: (value.index_select(0, rows) if torch.is_tensor(value) else value)
            for name, value in zip(_CACHE_FIELDS, cached.fields)
        }
    )


class SILSearchAdapter:
    """Policy and environment callbacks for one chunk of CVRP instances.

    Args:
        policy: An :class:`~rl4co.models.zoo.am.AttentionModelPolicy`, in eval mode.
        env: The CVRP environment used for transitions.
        td: The chunk's **reset** environment state (``[B]``-batched), already on ``device``.
            It is used only to encode the instances and to seed the root states.
        device: Device for the decoder and the environment step.
        temperature: Decoding temperature. Defaults to the policy's own.
        tanh_clipping: Logit clipping. Defaults to the policy's own; ``0`` disables it.
    """

    def __init__(
        self,
        policy,
        env,
        td: TensorDict,
        device: torch.device | str = "cuda",
        temperature: float = 1.0,
        tanh_clipping: float | None = None,
    ) -> None:
        if not getattr(policy, "mask_logits", True):
            raise ValueError(
                "SIL search expects the policy to mask infeasible actions; "
                "policy.mask_logits is False"
            )
        self.policy = policy
        self.env = env
        self.device = torch.device(device)
        self.temperature = float(temperature)
        self.tanh_clipping = (
            policy.tanh_clipping if tanh_clipping is None else float(tanh_clipping)
        )
        self.mask_logits = bool(policy.mask_logits)

        self.locs = td["locs"].detach().to(self.device)  # [B, N+1, 2]
        self.demand = td["demand"].detach().to(self.device)  # [B, N] normalised
        self.n_loc = int(self.demand.shape[-1])
        self.locs_np = self.locs.float().cpu().numpy()
        self._demand_np = self.demand.float().cpu().numpy()

        with torch.no_grad():
            # The chunk may arrive on the CPU (the generator's default), so the encoder sees
            # a device copy rather than the caller's tensors.
            hidden, _ = policy.encoder(td.to(self.device))
            self.cached = policy.decoder._precompute_cache(hidden, num_starts=1)

    # -- roots ---------------------------------------------------------------------

    def root_states(self, td: TensorDict) -> List[SILState]:
        """Build one root state per row of the reset chunk.

        The depot is not yet visited at reset, so the episode terminates only after a depot
        action has been taken at least once.
        """
        visited = td["visited"].cpu().numpy().astype(np.uint8)
        current = td["current_node"].reshape(-1).cpu().numpy().astype(np.int64)
        used = td["used_capacity"].reshape(-1).float().cpu().numpy()
        mask = td["action_mask"].cpu().numpy().astype(bool)
        done = td["done"].reshape(td.shape[0], -1)[:, 0].cpu().numpy().astype(bool)

        return [
            SILState(
                instance=i,
                visited=visited[i].copy(),
                current=int(current[i]),
                used=float(used[i]),
                mask=mask[i].copy(),
                done=bool(done[i]),
                actions=(),
                cost=0.0,
            )
            for i in range(td.shape[0])
        ]

    # -- SBS callbacks -------------------------------------------------------------

    @torch.no_grad()
    def child_log_probability_fn(self, states: List[SILState]) -> List[np.ndarray]:
        """Return ``[N+1]`` log-probabilities for every state, ``-inf`` on infeasible actions."""
        if not states:
            return []

        rows = torch.as_tensor(
            [state.instance for state in states], dtype=torch.long, device=self.device
        )
        current = torch.as_tensor(
            [state.current for state in states], dtype=torch.long, device=self.device
        )[:, None]
        used = torch.as_tensor(
            [state.used for state in states], dtype=torch.float32, device=self.device
        )[:, None]
        mask = torch.as_tensor(
            np.stack([state.mask for state in states]), device=self.device
        )
        vehicle_capacity = torch.ones_like(used)

        # StaticEmbedding contexts read only the current node, the residual capacity and the
        # action mask; node embeddings come from the cache.
        td = TensorDict(
            {
                "current_node": current,
                "used_capacity": used,
                "vehicle_capacity": vehicle_capacity,
                "action_mask": mask,
            },
            batch_size=[len(states)],
            device=self.device,
        )

        cached = _index_cache(self.cached, rows)
        logits, decoder_mask = self.policy.decoder(td, cached, num_starts=1)
        logprobs = process_logits(
            logits,
            decoder_mask,
            temperature=self.temperature,
            tanh_clipping=self.tanh_clipping,
            mask_logits=self.mask_logits,
        )
        # One bulk transfer; slicing a CPU tensor afterwards is free of device syncs.
        return list(logprobs.float().cpu().numpy())

    @torch.no_grad()
    def child_transition_fn(
        self, pairs: List[Tuple[SILState, int]]
    ) -> List[Tuple[SILState, bool]]:
        """Step the environment for every ``(state, action)`` pair in one batch.

        Returns:
            ``(child_state, is_leaf)`` pairs, index-aligned with ``pairs``. ``is_leaf`` is the
            environment's ``done`` flag; the child's cost includes the implicit return edge
            when the episode terminates at a customer.
        """
        if not pairs:
            return []

        parents = [state for state, _ in pairs]
        actions_np = np.asarray([action for _, action in pairs], dtype=np.int64)
        rows_np = np.asarray([state.instance for state in parents], dtype=np.int64)
        rows = torch.as_tensor(rows_np, dtype=torch.long, device=self.device)

        td = TensorDict(
            {
                "locs": self.locs[rows],
                "demand": self.demand[rows],
                "current_node": torch.as_tensor(
                    [state.current for state in parents],
                    dtype=torch.long,
                    device=self.device,
                )[:, None],
                "used_capacity": torch.as_tensor(
                    [state.used for state in parents],
                    dtype=torch.float32,
                    device=self.device,
                )[:, None],
                "vehicle_capacity": torch.ones(
                    (len(pairs), 1), dtype=torch.float32, device=self.device
                ),
                "visited": torch.as_tensor(
                    np.stack([state.visited for state in parents]), device=self.device
                ),
                "action_mask": torch.as_tensor(
                    np.stack([state.mask for state in parents]), device=self.device
                ),
                "action": torch.as_tensor(actions_np, device=self.device),
            },
            batch_size=[len(pairs)],
            device=self.device,
        )
        stepped = self.env.step(td)["next"]

        visited = stepped["visited"].cpu().numpy().astype(np.uint8)
        current = stepped["current_node"].reshape(-1).cpu().numpy().astype(np.int64)
        used = stepped["used_capacity"].reshape(-1).float().cpu().numpy()
        mask = stepped["action_mask"].cpu().numpy().astype(bool)
        done = stepped["done"].reshape(len(pairs), -1)[:, 0].cpu().numpy().astype(bool)

        # Every action contributes one edge; a terminal state at a customer additionally
        # closes its route back to the depot, which `env.get_reward` includes but the action
        # sequence does not (the decoder stops at that customer).
        edge = np.linalg.norm(
            self.locs_np[rows_np, [state.current for state in parents]]
            - self.locs_np[rows_np, actions_np],
            axis=-1,
        )
        return_edge = np.linalg.norm(
            self.locs_np[rows_np, current] - self.locs_np[rows_np, 0], axis=-1
        )
        return_edge = np.where(done & (current != 0), return_edge, 0.0)

        out = []
        for i, (parent, action) in enumerate(pairs):
            out.append(
                (
                    SILState(
                        instance=parent.instance,
                        visited=visited[i].copy(),
                        current=int(current[i]),
                        used=float(used[i]),
                        mask=mask[i].copy(),
                        done=bool(done[i]),
                        actions=parent.actions + (int(action),),
                        cost=float(parent.cost + edge[i] + return_edge[i]),
                    ),
                    bool(done[i]),
                )
            )
        return out
