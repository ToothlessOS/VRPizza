"""Per-batch POMO start-node selection shared by the file-backed and generated step arms.

POMO's multi-start machinery normally asks the environment for a scalar start count and a
fixed ``arange(k) % num_loc + 1`` set of start nodes. The step-matched arms need the *batch*
to carry its own starts instead: the labeled SL draw uses label-feasible starts, and the
generated RL draw uses a seeded sample of distinct customers. This mixin makes that work.

The mechanics are two overrides that change together:

* :meth:`get_num_starts` reports the batch's own start count when ``start_nodes`` is present,
  and falls back to the environment's default (``N``) otherwise. Validation and testing go
  through ``make_input_td``, which emits no ``start_nodes``, so they keep POMO's N starts.
* :meth:`select_start_nodes` returns the batch's own start nodes as a flat ``[num_starts * B]``
  action tensor in start-major order.

Both read the *post-reset* state, and rl4co's ``CVRPEnv._reset`` rebuilds a fresh
``TensorDict`` from an explicit key list, so :meth:`_reset` re-attaches ``start_nodes`` before
either can see it.
"""

from __future__ import annotations

__all__ = ["StartNodeMixin"]


class StartNodeMixin:
    """Carry per-batch ``start_nodes`` through reset and into POMO's multi-start.

    The host class must be a POMO-style CVRP environment whose ``reset``, ``get_num_starts``
    and ``select_start_nodes`` come from rl4co. The training batch must carry
    ``start_nodes`` ``[B, k]``; validation batches must not.
    """

    def _reset(self, td=None, batch_size=None):
        """Reset the environment, carrying ``start_nodes`` across.

        rl4co's ``CVRPEnv._reset`` builds a fresh TensorDict listing its own keys, so
        anything else on the input -- including the start nodes -- is dropped. Because
        :meth:`get_num_starts` and :meth:`select_start_nodes` only ever see the post-reset
        state, the key has to be re-attached here.
        """
        td_reset = super()._reset(td, batch_size)
        if td is not None and "start_nodes" in td.keys():
            td_reset.set("start_nodes", td["start_nodes"])
        return td_reset

    def get_num_starts(self, td) -> int:
        """Return the batch's start count, or the environment default when it carries none."""
        if "start_nodes" in td.keys():
            return int(td["start_nodes"].size(-1))
        return super().get_num_starts(td)

    def select_start_nodes(self, td, num_starts: int):
        """Return the batch's own start nodes as a flat ``[num_starts * B]`` action tensor.

        The layout must be *start-major* -- flat index ``k * B + b``. ``batchify`` expands a
        batch of ``B`` by ``repeats`` as ``x.expand(k, B, ...).contiguous().view(k * B, ...)``
        (``rl4co/utils/ops.py:11-14``) and ``unbatchify`` inverts it as
        ``view(k, B).permute(1, 0)``, so any other ordering would silently pair each instance
        with another instance's starts.

        Args:
            td: The post-reset state, carrying ``start_nodes`` ``[B, k]`` for training.
            num_starts: The count the decoder is expanding by.

        Returns:
            A ``[num_starts * B]`` int tensor, or rl4co's fixed selection when the batch has
            no start nodes, which is the case for validation and testing.
        """
        if "start_nodes" not in td.keys():
            return super().select_start_nodes(td, num_starts)

        start_nodes = td["start_nodes"]
        if int(start_nodes.size(-1)) != int(num_starts):
            raise ValueError(
                f"batch carries {start_nodes.size(-1)} start nodes but the decoder is "
                f"expanding by num_starts={num_starts}; the reward would be unbatchified "
                "incorrectly"
            )
        return start_nodes.transpose(0, 1).reshape(-1)
