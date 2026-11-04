"""A CVRP environment that trains POMO by REINFORCE on LEHD partial solutions.

The dataset and the post-start state are LEHD's, unchanged: :class:`~src.envs.LEHDCVRPEnv`
serves :class:`~src.envs.lehd_dataset.LEHDPartialDataset` for training, and
:meth:`~src.envs.cvrp_sl_lehd.PartialResetMixin.reset_partial` builds the state in which the
window's start is already served and the residual load is carried. Only two things are added
here, both about how a reconstruction is scored:

* :meth:`reset_partial` keeps ``start_node`` on the state, because the reward needs it;
* :meth:`_get_reward` prepends the start to the action sequence. Without that, rl4co's CVRP
  reward would score ``depot -> first remaining -> ... -> depot`` and would omit both the edge
  into the start and the start's own contribution to the window cost.

With the start prepended, the reward is exactly the negative reduced closed-tour cost of the
window (``depot -> start -> ... -> last -> depot``). That cost differs from the true repaired
full-tour cost by a constant that depends only on the window and its prefix, never on the
reconstruction; the derivation is in ``docs/LEHD_RL.md``. :func:`reduced_window_cost` is the
same computation in one place, used by the reward, by the teacher baseline in
:class:`~src.models.LEHDPOMORL`, and by the verifier.

There is deliberately no multi-start override. Rollouts are sampled from the single
pre-served start, which is what the LEHD partial defines and what RRC uses at inference.
:meth:`~src.envs.cvrp_sl_lehd.PartialResetMixin.reset_partial` is the only training entry
point; the inherited multi-start helpers remain available for the full-instance validation
batches, where they must keep POMO's ``N`` starts.
"""

from __future__ import annotations

import torch

from rl4co.utils.ops import get_tour_length

from src.envs.cvrp_sl_lehd import LEHDCVRPEnv

__all__ = ["LEHDRLCVRPEnv", "reduced_window_cost"]


def reduced_window_cost(
    depot: torch.Tensor, locs: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    """Closed-tour cost of a partial reconstruction, in the reduced problem's indexing.

    The sequence is ``depot -> first action -> ... -> last action -> depot``. The leading
    depot is supplied here, so callers pass only the reconstruction's local actions (with the
    forced start prepended when it is not in ``actions``). A depot action ``0`` in the middle
    of the sequence is a route boundary and contributes the two edges through the depot, which
    is exactly what the full solution does.

    Args:
        depot: ``[B, 2]`` depot coordinates.
        locs: ``[B, w, 2]`` window customer coordinates, sorted by original id (local id ``i``
            is row ``i - 1``).
        actions: ``[B, T]`` local ids: ``0`` is the depot, ``1..w`` are the window customers.
            Padding with ``0`` adds zero-length depot edges and does not change the cost.

    Returns:
        ``[B]`` positive tour lengths, in the coordinates' units.

    Raises:
        ValueError: If the shapes do not agree.
    """
    if depot.ndim != 2 or depot.shape[-1] != 2:
        raise ValueError(f"depot must be [B, 2], got {tuple(depot.shape)}")
    if locs.ndim != 3 or locs.shape[-1] != 2:
        raise ValueError(f"locs must be [B, w, 2], got {tuple(locs.shape)}")
    if actions.ndim != 2:
        raise ValueError(f"actions must be [B, T], got {tuple(actions.shape)}")
    if depot.shape[0] != locs.shape[0] or depot.shape[0] != actions.shape[0]:
        raise ValueError(
            f"batch sizes disagree: depot {tuple(depot.shape)}, locs {tuple(locs.shape)}, "
            f"actions {tuple(actions.shape)}"
        )

    actions = actions.to(torch.long)
    coords = torch.cat([depot[:, None, :], locs], dim=1)  # [B, w + 1, 2]
    # Prepend the depot index so the cyclic tour_length closes last -> depot and includes
    # depot -> first. Trailing padding 0s gather the depot and add no distance.
    index = torch.cat([torch.zeros_like(actions[:, :1]), actions], dim=1)
    ordered = coords.gather(1, index[..., None].expand(-1, -1, 2))
    return get_tour_length(ordered)


class LEHDRLCVRPEnv(LEHDCVRPEnv):
    """LEHD partial solutions, scored for one sampled reconstruction from the fixed start.

    Args:
        data_path: Path to the reference-solution file, forwarded to
            :class:`~src.envs.LEHDCVRPEnv`.
        val_data_path: Optional separate validation file, forwarded unchanged.
        num_loc: Number of customers in the **full** instance; inferred from the file when
            omitted. Reduced partial batches carry their own width ``w``.
        capacity: Vehicle capacity in the file's demand units; inferred when omitted.
        val_instances: Held-out instances carved from the tail.
        seed: Base seed for the partial draw.
        check_solution: Forwarded, but keep ``False``: the reward is called on a prepared
            partial state whose action sequence starts after the pre-served start.
        effective_batch_size: Optimizer batch size used to key the one-``w``-per-batch draw.
        **kwargs: Forwarded to :class:`~src.envs.LEHDCVRPEnv`.

    The environment has no multi-start behaviour of its own. Training calls
    :meth:`reset_partial` and samples one (or K) rollout(s) from the state; validation goes
    through :class:`~src.models.components.pomo_inference.POMOInferenceMixin` on full
    instances, where the inherited ``get_num_starts`` and ``select_start_nodes`` still apply.
    """

    def reset_partial(self, batch):
        """Build the post-start state and keep ``start_node`` for the reward.

        :class:`~src.envs.cvrp_sl_lehd.PartialResetMixin.reset_partial` builds the state from
        the reduced batch but drops everything the environment does not need; the reward needs
        the start, so it is re-attached here. The key survives the decoder's batchification,
        which is why :meth:`_get_reward` sees one start per rollout.
        """
        td = super().reset_partial(batch)
        td["start_node"] = batch["start_node"].to(torch.long)
        return td

    def _get_reward(self, td, actions):
        """Return ``-C_red`` of the reconstruction, with the pre-served start included.

        For a full-instance batch (validation, or any caller that did not go through
        :meth:`reset_partial`) there is no ``start_node`` and the parent's reward is correct
        unchanged. For a partial, the decoder's actions begin after the start, so the start is
        prepended before scoring; otherwise the cost would omit the start's edges and would not
        be comparable with the teacher baseline.
        """
        if "start_node" not in td.keys():
            return super()._get_reward(td, actions)

        start = td["start_node"].to(actions.dtype).reshape(-1, 1)
        if start.shape[0] != actions.shape[0]:
            raise ValueError(
                f"start_node has {start.shape[0]} rows but actions has {actions.shape[0]}; "
                "the decoder's batchification did not carry the start through"
            )
        actions = torch.cat([start, actions], dim=1)
        return -reduced_window_cost(td["locs"][:, 0, :], td["locs"][:, 1:, :], actions)
