"""LEHD's light encoder / heavy decoder, as an rl4co constructive policy.

The reference is the CVRP model in
``CIAM-Group NCO_code main single_objective-LEHD/CVRP/VRPModel.py``: a one-layer
encoder, a six-layer decoder that re-encodes ``[depot, unselected customers,
last-selected]`` at every decoding step, and a two-channel output head (direct
arrival vs. arrival via the depot). The port keeps the reference's module names
and shapes, so the published checkpoint loads with ``strict=True`` and the
parameter count is exactly 1,415,682.

The one interface adaptation is the action space. The reference scores a ``2N``
``(customer, via_depot)`` distribution; every other component of this repo
(the CVRP environment, the sub-path labels, the RRC parser and the loss) uses
POMO's ``0 = depot, 1..N = customers`` space. This policy therefore exposes the
reference's 2N distribution as ``N+1`` logits by marginalising at each step::

    logits[0]   = logsumexp_u z_via(u)
    logits[c]   = z_direct(c)          for an unselected customer
    logits[c]   = -inf                 for a visited customer

A softmax over those logits is exactly the next-action marginal of the
reference's joint 2N softmax, so ``ConstructivePolicy.forward`` can drive
sampling, greedy rollout, teacher forcing, multistart, rewards and
``get_log_likelihood`` without a special case. A via-depot action is decoded in
two POMO steps (depot, then customer); the exact reference teacher-forced CE is
recovered separately by :func:`src.models.sil_rc_lehd.lehd_exact_ce_step`, which
scores the via channel of the *next* customer at the pre-depot state and skips
the structural customer token. :meth:`LEHDPolicy.reference_rollout` is the
faithful 2N greedy rollout for paper-parity evaluation.

Two reference quirks are deliberately configurable rather than hard-wired:

* ``detach_encoder_nodes`` mirrors the reference decoder's detached customer
  gathers (only the depot context carries gradient to the encoder). It is off
  for the SIL-RC runs, which train the encoder, and on for exact-reproduction
  experiments.
* the reference's ``+1e-7`` probability floor is applied only by
  :meth:`reference_probs`; training uses log-softmax.

See ``docs/LEHD_SIL_RC.md`` for the full deviation list.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensordict.tensordict import TensorDict

from rl4co.models.common.constructive.autoregressive.policy import (
    AutoregressivePolicy,
)
from rl4co.models.common.constructive.base import (
    ConstructiveDecoder,
    ConstructiveEncoder,
)
from rl4co.utils.ops import gather_by_index

__all__ = [
    "LEHDDecoder",
    "LEHDEncoder",
    "LEHDFeedForward",
    "LEHDLayer",
    "LEHDPolicy",
]

#: The reference's probability floor (``VRPModel.py:264-267``).
PROB_FLOOR = 1e-7
PROB_FLOOR_THRESHOLD = 1e-5


def _reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    """``[B, L, H * D] -> [B, H, L, D]``, as the reference's ``reshape_by_heads``."""
    batch, length = qkv.shape[:2]
    return qkv.reshape(batch, length, head_num, -1).transpose(1, 2)


def _multi_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full self-attention, optionally masking invalid key positions.

    The reference has no mask because every row has the same number of tokens.
    The port keeps every customer in the sequence and masks the visited ones, so
    a key mask is needed; for valid keys the result is identical.
    """
    key_dim = q.size(-1)
    scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(key_dim)
    if key_mask is not None:
        scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v)
    batch, heads, length, dim = out.shape
    return out.transpose(1, 2).reshape(batch, length, heads * dim)


class LEHDFeedForward(nn.Module):
    """The reference's ``Feed_Forward_Module``: ReLU FFN, no normalization."""

    def __init__(self, embedding_dim: int = 128, ff_hidden_dim: int = 512) -> None:
        super().__init__()
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(F.relu(self.W1(x)))


class LEHDLayer(nn.Module):
    """One reference encoder/decoder block.

    The reference's ``EncoderLayer`` and ``DecoderLayer`` are structurally
    identical, so one class serves both and the state-dict names
    (``Wq``, ``Wk``, ``Wv``, ``multi_head_combine``, ``feedForward.W1/W2``)
    match either.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        head_num: int = 8,
        qkv_dim: int = 16,
        ff_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.head_num = int(head_num)
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.feedForward = LEHDFeedForward(embedding_dim, ff_hidden_dim)

    def forward(
        self, x: torch.Tensor, key_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        q = _reshape_by_heads(self.Wq(x), self.head_num)
        k = _reshape_by_heads(self.Wk(x), self.head_num)
        v = _reshape_by_heads(self.Wv(x), self.head_num)
        attended = self.multi_head_combine(_multi_head_attention(q, k, v, key_mask))
        out = x + attended
        return out + self.feedForward(out)


class LEHDEncoder(ConstructiveEncoder):
    """The reference's light encoder: ``Linear(3, E)`` plus one attention block.

    The reference reads ``(x, y, demand / capacity)`` for every node, with the
    depot at index 0 and demand 0. The repo's ``make_input_td`` has already
    divided demand by capacity, so the division is not repeated here.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_layers: int = 1,
        head_num: int = 8,
        qkv_dim: int = 16,
        ff_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_layers = int(num_layers)
        self.embedding = nn.Linear(3, embedding_dim, bias=True)
        self.layers = nn.ModuleList(
            [
                LEHDLayer(embedding_dim, head_num, qkv_dim, ff_hidden_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(self, td: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ``locs [B, N+1, 2]`` and normalised ``demand [B, N]``."""
        locs = td["locs"]
        demand = td["demand"]
        depot_demand = torch.zeros(
            (*demand.shape[:-1], 1), dtype=demand.dtype, device=demand.device
        )
        features = torch.cat([locs, torch.cat([depot_demand, demand], -1)[..., None]], -1)
        initial = self.embedding(features)
        out = initial
        for layer in self.layers:
            out = layer(out)
        return out, initial


class LEHDDecoder(ConstructiveDecoder):
    """The reference's heavy decoder, batched and adapted to the N+1 actions.

    At each step the decoder builds

    ``[encoded depot + remaining capacity, encoded customers, encoded current
    node + remaining capacity]``

    runs ``num_layers`` self-attention blocks over it, and projects every
    position to two logits (direct, via-depot). Visited customers are masked as
    attention keys, which is the batched equivalent of the reference's
    ``_get_new_data`` compaction: there is no positional encoding, the unselected
    customers keep ascending-id order, so the outputs at unselected positions are
    identical to the reference's.

    Args:
        embedding_dim: Encoder/decoder width, 128 in the reference.
        num_layers: Decoder blocks, 6 in the reference.
        head_num: Attention heads, 8.
        qkv_dim: Per-head width, 16.
        ff_hidden_dim: FFN hidden width, 512.
        detach_encoder_nodes: Mirror the reference's detached customer gathers.
        use_checkpointing: Checkpoint the per-step block stack while training.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_layers: int = 6,
        head_num: int = 8,
        qkv_dim: int = 16,
        ff_hidden_dim: int = 512,
        detach_encoder_nodes: bool = False,
        use_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_layers = int(num_layers)
        self.detach_encoder_nodes = bool(detach_encoder_nodes)
        self.use_checkpointing = bool(use_checkpointing)
        self.embedding_first_node = nn.Linear(embedding_dim + 1, embedding_dim, bias=True)
        self.embedding_last_node = nn.Linear(embedding_dim + 1, embedding_dim, bias=True)
        self.layers = nn.ModuleList(
            [
                LEHDLayer(embedding_dim, head_num, qkv_dim, ff_hidden_dim)
                for _ in range(num_layers)
            ]
        )
        self.Linear_final = nn.Linear(embedding_dim, 2, bias=True)

        # Side channel used by the exact teacher-forced CE. It is empty unless
        # `set_capture(True)` is called around one policy forward.
        self._capture_enabled = False
        self._captures: list[dict[str, torch.Tensor]] | None = None

    # -- capture side channel ---------------------------------------------------

    def set_capture(self, enabled: bool) -> None:
        """Enable/disable raw-logit capture for the next forward pass."""
        self._capture_enabled = bool(enabled)
        if not enabled:
            self._captures = None

    def pop_captures(self) -> list[dict[str, torch.Tensor]]:
        """Return and clear the captures of the last forward pass."""
        captures = self._captures or []
        self._captures = None
        return captures

    # -- rl4co hooks ------------------------------------------------------------

    def pre_decoder_hook(
        self, td: TensorDict, env, hidden: torch.Tensor, num_starts: int = 0
    ):
        """Align the encoder output with the (possibly batchified) td.

        The encoder ran on the unbatched ``td``. Multistart decoding batchifies
        ``td`` start-major (``rl4co.utils.ops.batchify``), so the shared encoder
        output is repeated the same way here.
        """
        if hidden is not None and num_starts > 1:
            hidden = hidden.repeat(num_starts, 1, 1)
        self._captures = [] if self._capture_enabled else None
        return td, env, hidden

    def _run_layers(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_mask)
        return x

    def _step_logits(
        self,
        td: TensorDict,
        hidden: torch.Tensor,
        capture: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(direct, via, unselected, logZ)`` for the current state.

        ``direct``/``via`` are the raw ``Linear_final`` channels for every
        customer, ``unselected`` is ``[B, N]`` and ``logZ`` is the reference's
        joint 2N normaliser over the unselected customers.
        """
        visited = td["visited"]
        unselected = visited[..., 1:] == 0  # [B, N]
        current = td["current_node"].squeeze(-1).long()
        remaining = (td["vehicle_capacity"] - td["used_capacity"]).to(hidden.dtype)

        customers = hidden[:, 1:]
        if self.detach_encoder_nodes:
            customers = customers.detach()
        depot = hidden[:, 0]
        last = gather_by_index(customers, (current - 1).clamp_min(0))
        last = torch.where((current == 0)[:, None], depot, last)

        first_context = self.embedding_first_node(
            torch.cat([depot, remaining], -1)
        )
        last_context = self.embedding_last_node(
            torch.cat([last, remaining], -1)
        )
        sequence = torch.cat(
            [first_context[:, None], customers, last_context[:, None]], dim=1
        )
        ones = torch.ones_like(unselected[:, :1])
        key_mask = torch.cat([ones, unselected, ones], dim=1)

        if self.use_checkpointing and self.training and torch.is_grad_enabled():
            out = torch.utils.checkpoint.checkpoint(
                self._run_layers, sequence, key_mask, use_reentrant=False
            )
        else:
            out = self._run_layers(sequence, key_mask)

        out = self.Linear_final(out)  # [B, N+2, 2]
        direct = out[:, 1:-1, 0]  # [B, N]
        via = out[:, 1:-1, 1]  # [B, N]
        neg_inf = float("-inf")
        log_z = torch.logsumexp(
            torch.cat(
                [
                    direct.masked_fill(~unselected, neg_inf),
                    via.masked_fill(~unselected, neg_inf),
                ],
                dim=-1,
            ),
            dim=-1,
        )
        if capture and self._captures is not None:
            # `process_logits` mutates the returned N+1 tensor in place; these
            # are separate tensors, and cloning keeps that guarantee explicit.
            self._captures.append(
                {
                    "direct": direct.clone(),
                    "via": via.clone(),
                    "unselected": unselected.clone(),
                    "logz": log_z.clone(),
                }
            )
        return direct, via, unselected, log_z

    def forward(
        self, td: TensorDict, hidden: torch.Tensor, num_starts: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(N+1 logits, action mask)`` for the current state.

        The depot logit is the total via-depot mass, not the joint 2N
        normaliser: the latter would double-count the direct mass in the N+1
        softmax. The joint normaliser is still captured for the exact CE.
        """
        direct, via, unselected, _log_z = self._step_logits(
            td, hidden, capture=self._captures is not None
        )
        neg_inf = float("-inf")
        depot_via = torch.logsumexp(via.masked_fill(~unselected, neg_inf), dim=-1)
        # An empty unselected set happens only after every customer is served;
        # the depot stays feasible and must get a finite logit, or the log-softmax
        # over the fully masked row is NaN.
        depot_logit = torch.where(
            unselected.any(dim=-1), depot_via, torch.zeros_like(depot_via)
        )
        customer_logits = direct.masked_fill(~unselected, neg_inf)
        logits = torch.cat([depot_logit[:, None], customer_logits], dim=-1)
        return logits, td["action_mask"]

    # -- faithful reference distribution ----------------------------------------

    @torch.no_grad()
    def reference_probs(self, td: TensorDict, hidden: torch.Tensor) -> dict:
        """The reference's exact 2N distribution for the current state.

        Returns a dict with ``direct [B, N]``, ``via [B, N]``, ``unselected
        [B, N]`` and ``logz [B]``. The probability floor is applied, matching
        ``VRPModel.py:264-267``; selected customers keep probability 0 here
        (the reference's flat representation marks them ``-2``).
        """
        direct, via, unselected, log_z = self._step_logits(td, hidden, capture=False)
        neg_inf = float("-inf")
        probs = torch.softmax(
            torch.cat(
                [
                    direct.masked_fill(~unselected, neg_inf),
                    via.masked_fill(~unselected, neg_inf),
                ],
                dim=-1,
            ),
            dim=-1,
        )
        probs = torch.where(
            (probs <= PROB_FLOOR_THRESHOLD) & unselected.repeat(1, 2),
            probs + PROB_FLOOR,
            probs,
        )
        n = direct.shape[-1]
        return {
            "direct": probs[:, :n],
            "via": probs[:, n:],
            "unselected": unselected,
            "logz": log_z,
        }


class LEHDPolicy(AutoregressivePolicy):
    """rl4co policy with the reference LEHD encoder/decoder.

    Args:
        embedding_dim: 128, as in the reference config.
        num_encoder_layers: 1 (the reference hard-codes one encoder layer).
        num_decoder_layers: 6.
        num_heads: 8.
        qkv_dim: 16.
        ff_hidden_dim: 512.
        env_name: Passed to ``AutoregressivePolicy``; unused by the network.
        use_checkpointing: Per-step gradient checkpointing in the decoder.
        detach_encoder_nodes: Reference detached-gather behaviour.
        **kwargs: Forwarded to ``AutoregressivePolicy`` (decode types and so on).
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        qkv_dim: int = 16,
        ff_hidden_dim: int = 512,
        env_name: str = "cvrp",
        use_checkpointing: bool = True,
        detach_encoder_nodes: bool = False,
        **kwargs,
    ) -> None:
        self.embedding_dim = int(embedding_dim)
        self.num_encoder_layers = int(num_encoder_layers)
        self.num_decoder_layers = int(num_decoder_layers)
        self.num_heads = int(num_heads)
        self.qkv_dim = int(qkv_dim)
        self.ff_hidden_dim = int(ff_hidden_dim)
        encoder = LEHDEncoder(
            embedding_dim=embedding_dim,
            num_layers=num_encoder_layers,
            head_num=num_heads,
            qkv_dim=qkv_dim,
            ff_hidden_dim=ff_hidden_dim,
        )
        decoder = LEHDDecoder(
            embedding_dim=embedding_dim,
            num_layers=num_decoder_layers,
            head_num=num_heads,
            qkv_dim=qkv_dim,
            ff_hidden_dim=ff_hidden_dim,
            detach_encoder_nodes=detach_encoder_nodes,
            use_checkpointing=use_checkpointing,
        )
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            temperature=1.0,
            tanh_clipping=0.0,
            mask_logits=True,
            **kwargs,
        )

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Toggle per-step decoder checkpointing."""
        self.decoder.use_checkpointing = bool(enabled)

    def policy_signature(self) -> str:
        """A stable fingerprint of the architecture for label metadata."""
        return (
            f"lehd-e{self.num_encoder_layers}-d{self.num_decoder_layers}"
            f"-h{self.num_heads}-q{self.qkv_dim}-ff{self.ff_hidden_dim}"
        )

    @torch.no_grad()
    def reference_probs(self, td: TensorDict) -> dict:
        """The reference's exact 2N probabilities for a reset ``td``."""
        hidden, _ = self.encoder(td)
        return self.decoder.reference_probs(td, hidden)

    @torch.no_grad()
    def reference_rollout(
        self, td: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The reference's greedy 2N rollout (argmax, no multistart/augmentation).

        Every step serves one customer; the flag says whether the vehicle
        returns to the depot first. Step 0 forces the flag to 1, as
        ``VRPTester.py:188-190`` does, and a customer whose demand exceeds the
        remaining capacity is served via the depot even if the model's argmax
        said direct -- the reference environment's effective flag. Unlike the
        reference's reported student flag, this returns the effective flags, so
        the tour is feasible.

        Args:
            td: A reset ``TensorDict`` with ``locs [B, N+1, 2]``, ``demand
                [B, N]``, ``depot [B, 2]``, ``visited [B, N+1]``,
                ``current_node``, ``used_capacity`` and ``vehicle_capacity``.

        Returns:
            ``(costs [B], nodes [B, N], flags [B, N])`` with 1-based customer
            ids and ``1`` at each route start.
        """
        from src.rrc.runner import cost_from_flags_torch

        hidden, _ = self.encoder(td)
        size = int(td.batch_size[0])
        n = int(td["demand"].shape[-1])
        device = td.device

        visited = td["visited"].clone()
        current = torch.zeros(size, dtype=torch.long, device=device)
        used = torch.zeros(size, dtype=torch.float32, device=device)
        capacity = td["vehicle_capacity"].reshape(size)
        demand = td["demand"].reshape(size, n)

        nodes = torch.zeros(size, n, dtype=torch.long, device=device)
        flags = torch.zeros(size, n, dtype=torch.long, device=device)

        for step in range(n):
            state = TensorDict(
                {
                    "visited": visited,
                    "current_node": current[:, None],
                    "used_capacity": used[:, None],
                    "vehicle_capacity": capacity[:, None],
                },
                batch_size=[size],
                device=device,
            )
            direct, via, unselected, _ = self.decoder._step_logits(
                state, hidden, capture=False
            )
            neg_inf = float("-inf")
            flat = torch.cat(
                [
                    direct.masked_fill(~unselected, neg_inf),
                    via.masked_fill(~unselected, neg_inf),
                ],
                dim=-1,
            )
            index = flat.argmax(dim=-1)
            flag = (index >= n).long()
            node = index % n + 1
            if step == 0:
                # `VRPTester.py:188-190`: the first node always starts a route.
                flag = torch.ones_like(flag)
            selected_demand = demand.gather(1, (node - 1)[:, None]).squeeze(1)
            over_capacity = used + selected_demand > capacity + 1e-5
            flag = torch.where(over_capacity, torch.ones_like(flag), flag)
            used = torch.where(flag.bool(), torch.zeros_like(used), used) + selected_demand
            current = node
            visited.scatter_(1, node[:, None], 1)
            nodes[:, step] = node
            flags[:, step] = flag

        costs = cost_from_flags_torch(
            nodes, flags.bool(), td["depot"], td["locs"][:, 1:]
        )
        return costs, nodes, flags
