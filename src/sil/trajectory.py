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

States are kept on the CPU as small NumPy arrays whose ``visited``/``mask`` are views into the
transition's shared ``[n, N+1]`` block. Each SBS level moves the decoder input and the
environment step to the device in bulk: one decoder call for the whole level (batched as
``[B, K]`` with the ``[B]`` cache broadcasting), and one transition call for the level's
expansion pairs. Per-state tensors or per-state cache rows would create tens of thousands of
allocations and device synchronisations.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from rl4co.utils.decoding import process_logits

__all__ = ["SILSearchAdapter", "SILState"]

#: Field order of rl4co's ``PrecomputedCache``. `fields` returns the values, not the names.
_CACHE_FIELDS = ("node_embeddings", "graph_context", "glimpse_key", "glimpse_val", "logit_key")


class _ActionPath:
    """A persistent linked list of actions from the root to one state.

    The search creates one state per expanded child -- ~10^4 per level, ~10^7 per generation
    round -- so the path is extended in O(1) here and only walked when a leaf's action sequence
    is actually needed (:func:`src.sil.search._record_from_actions`, and the verifier). Storing
    ``parent.actions + (action,)`` on every state would copy an O(depth) tuple per node and would
    also keep every ancestor alive.
    """

    __slots__ = ("prev", "action")

    def __init__(self, prev: "_ActionPath | None", action: int) -> None:
        self.prev = prev
        self.action = action


class SILState:
    """One partial CVRP solution, as seen by the search.

    A plain ``__slots__`` class rather than a dataclass: construction is the single most frequent
    object operation in the search, and the dataclass machinery (per-field ``object.__setattr__``
    for ``frozen``) is measurably slower. States are never mutated.

    Args:
        instance: Row of the generation chunk this state belongs to.
        visited: ``[N+1]`` uint8, index ``0`` is the depot slot. Normally a **view** into the
            transition's shared array, not a copy; the base array must therefore never be
            recycled (the WOR trie caches states across rounds).
        current: Current node index, ``0`` at the depot.
        used: Normalised load of the current vehicle.
        mask: ``[N+1]`` bool, ``True`` where the action is feasible (rl4co's convention). Also a
            view into a shared array.
        done: Whether the episode has terminated.
        cost: Cumulative Euclidean length, including the implicit final return edge for terminal
            states.
        _action_path: Linked action history, or ``None`` at a root.
    """

    __slots__ = (
        "instance",
        "visited",
        "current",
        "used",
        "mask",
        "done",
        "cost",
        "_action_path",
    )

    def __init__(
        self,
        instance: int,
        visited: np.ndarray,
        current: int,
        used: float,
        mask: np.ndarray,
        done: bool,
        cost: float,
        _action_path: _ActionPath | None = None,
    ) -> None:
        self.instance = instance
        self.visited = visited
        self.current = current
        self.used = used
        self.mask = mask
        self.done = done
        self.cost = cost
        self._action_path = _action_path

    @property
    def actions(self) -> tuple[int, ...]:
        """Actions from the root through this state, inclusive."""
        actions = []
        path = self._action_path
        while path is not None:
            actions.append(path.action)
            path = path.prev
        actions.reverse()
        return tuple(actions)

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
        grouped_decoder: Decode the level as one ``[B, K]`` batch with the per-instance cache
            left at ``[B]`` so it broadcasts, instead of a flat ``[B*K]`` batch whose cache is
            re-materialised per row. Set ``False`` to use the previous per-row path.
    """

    def __init__(
        self,
        policy,
        env,
        td: TensorDict,
        device: torch.device | str = "cuda",
        temperature: float = 1.0,
        tanh_clipping: float | None = None,
        grouped_decoder: bool = True,
    ) -> None:
        if not getattr(policy, "mask_logits", True):
            raise ValueError(
                "SIL search expects the policy to mask infeasible actions; "
                "policy.mask_logits is False"
            )
        if grouped_decoder and getattr(policy.decoder, "is_dynamic_embedding", False):
            # Sharing one cache row across a beam is only valid when the keys/values depend on
            # the instance alone. CVRP's StaticEmbedding satisfies that; fail loudly otherwise.
            raise ValueError(
                "the grouped decoder needs a static embedding; "
                f"{type(policy.decoder).__name__} uses a dynamic embedding"
            )
        self.policy = policy
        self.env = env
        self.device = torch.device(device)
        self.temperature = float(temperature)
        self.tanh_clipping = (
            policy.tanh_clipping if tanh_clipping is None else float(tanh_clipping)
        )
        self.mask_logits = bool(policy.mask_logits)
        self.grouped_decoder = bool(grouped_decoder)
        self.n_instances = int(td.shape[0])

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
                visited=visited[i],
                current=int(current[i]),
                used=float(used[i]),
                mask=mask[i],
                done=bool(done[i]),
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
        if self.grouped_decoder:
            return self._log_probs_grouped(states)
        return self._log_probs_flat(states)

    def _log_probs_flat(self, states: List[SILState]) -> List[np.ndarray]:
        """One decoder call over a flat ``[n]`` batch, re-indexing the cache per row."""
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

    def _log_probs_grouped(self, states: List[SILState]) -> List[np.ndarray]:
        """One decoder call over a ``[B, K]`` batch with a broadcast ``[B]`` cache.

        States are grouped by instance into a rectangular block; short groups are padded by
        duplicating one of their real rows (an all-masked dummy can produce NaNs in the
        attention). ``K`` is at least 2 because rl4co's ``gather_by_index`` squeezes a
        singleton index dimension, which breaks the context embedding's concatenation.
        """
        n_states = len(states)
        n_actions = self.n_loc + 1
        instances = np.fromiter(
            (state.instance for state in states), dtype=np.int64, count=n_states
        )
        # Stable: states that share an instance keep their input order.
        order = np.argsort(instances, kind="stable")
        uniq, counts = np.unique(instances[order], return_counts=True)
        batch = int(uniq.size)
        k = max(2, int(counts.max()))

        mask_np = np.stack([states[i].mask for i in order])
        current_np = np.fromiter(
            (states[i].current for i in order), dtype=np.int64, count=n_states
        )
        used_np = np.fromiter(
            (states[i].used for i in order), dtype=np.float32, count=n_states
        )

        mask = np.empty((batch, k, n_actions), dtype=bool)
        current = np.empty((batch, k), dtype=np.int64)
        used = np.empty((batch, k), dtype=np.float32)
        offset = 0
        for group, count in enumerate(counts):
            count = int(count)
            mask[group, :count] = mask_np[offset : offset + count]
            current[group, :count] = current_np[offset : offset + count]
            used[group, :count] = used_np[offset : offset + count]
            mask[group, count:] = mask[group, count - 1]
            current[group, count:] = current[group, count - 1]
            used[group, count:] = used[group, count - 1]
            offset += count

        td = TensorDict(
            {
                "current_node": torch.as_tensor(current[:, :, None], device=self.device),
                "used_capacity": torch.as_tensor(used[:, :, None], device=self.device),
                "vehicle_capacity": torch.ones(
                    (batch, k, 1), dtype=torch.float32, device=self.device
                ),
                "action_mask": torch.as_tensor(mask, device=self.device),
            },
            batch_size=[batch, k],
            device=self.device,
        )

        if batch == self.n_instances and np.array_equal(
            uniq, np.arange(self.n_instances)
        ):
            cached = self.cached
        else:
            # WOR rounds 2+ search only the unexhausted roots, so the level's instances are a
            # subset of the chunk; select the matching cache rows rather than pairing blindly.
            cached = _index_cache(
                self.cached,
                torch.as_tensor(uniq, dtype=torch.long, device=self.device),
            )

        logits, decoder_mask = self.policy.decoder(td, cached, num_starts=1)
        logprobs = (
            process_logits(
                logits,
                decoder_mask,
                temperature=self.temperature,
                tanh_clipping=self.tanh_clipping,
                mask_logits=self.mask_logits,
            )
            .float()
            .cpu()
            .numpy()
        )

        out: List[np.ndarray] = [None] * n_states
        offset = 0
        for group, count in enumerate(counts):
            count = int(count)
            for j in range(count):
                out[order[offset + j]] = logprobs[group, j]
            offset += count
        return out

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
        actions_np = np.fromiter(
            (action for _, action in pairs), dtype=np.int64, count=len(pairs)
        )
        rows_np = np.fromiter(
            (state.instance for state in parents), dtype=np.int64, count=len(parents)
        )
        parent_current = np.fromiter(
            (state.current for state in parents), dtype=np.int64, count=len(parents)
        )
        parent_used = np.fromiter(
            (state.used for state in parents), dtype=np.float32, count=len(parents)
        )
        rows = torch.as_tensor(rows_np, dtype=torch.long, device=self.device)

        td = TensorDict(
            {
                "locs": self.locs[rows],
                "demand": self.demand[rows],
                "current_node": torch.as_tensor(
                    parent_current, dtype=torch.long, device=self.device
                )[:, None],
                "used_capacity": torch.as_tensor(
                    parent_used, dtype=torch.float32, device=self.device
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
            self.locs_np[rows_np, parent_current] - self.locs_np[rows_np, actions_np],
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
                        visited=visited[i],
                        current=int(current[i]),
                        used=float(used[i]),
                        mask=mask[i],
                        done=bool(done[i]),
                        cost=float(parent.cost + edge[i] + return_edge[i]),
                        _action_path=_ActionPath(parent._action_path, int(action)),
                    ),
                    bool(done[i]),
                )
            )
        return out
