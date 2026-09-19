"""POMO's validation rollout, shared by the supervised and reinforcement-learning models.

:class:`~src.models.POMOSL` and :class:`~src.models.POMORL` train on completely different
objectives -- teacher forcing on reference tours versus REINFORCE on generated instances -- but
the whole point of having both is to compare them. A comparison is only meaningful if the
validation number means the same thing on each side, and it does not by default: rl4co's own POMO
validation leaves ``out["reward"]`` as the raw ``[B, num_augment, num_starts]`` tensor, so a
metric declared as ``"reward"`` is the *mean* over every augmentation and every start, while the
rollout POMO is actually reported with takes the *max* over both.

This module holds that rollout exactly once. It is deliberately narrow: the mixin carries the
inference step and nothing else. In particular it must never grow a ``shared_step`` or a
``calculate_loss``, because ``POMORL`` resolves those along its MRO and would silently pick up
the supervised teacher-forcing loss instead of REINFORCE's.
"""

from __future__ import annotations

from rl4co.data.transforms import StateAugmentation
from rl4co.utils.ops import unbatchify

__all__ = ["POMOInferenceMixin", "validation_augmentation"]

#: The augmentation both models validate with. Pinned here rather than left to a library
#: default: it is an input to the comparison, not a hyperparameter either model is free to
#: vary independently.
AUGMENT_FN = "dihedral8"


def validation_augmentation(
    num_augment: int, augment_fn: str = AUGMENT_FN
) -> StateAugmentation | None:
    """Build the dihedral augmentation used at validation time, or ``None`` when unaugmented.

    Both models go through this one function so they cannot end up with different augmentation
    settings, which would move ``val/reward`` and ``val/gap_ref`` on one side only.

    Args:
        num_augment: Number of dihedral augmentations. ``1`` disables augmentation, which is
            useful for quick runs; ``dihedral8`` requires exactly 8.
        augment_fn: Augmentation function name, passed to ``StateAugmentation``.

    Returns:
        The augmentation, or ``None`` if ``num_augment <= 1``.
    """
    if num_augment <= 1:
        return None
    return StateAugmentation(num_augment=num_augment, augment_fn=augment_fn)


class POMOInferenceMixin:
    """POMO's multi-start, dihedral-augmented validation rollout and its three metrics.

    Mixed into :class:`~src.models.POMOSL` and :class:`~src.models.POMORL` ahead of whichever
    base class each of them really uses, so both evaluate with the same code.

    The host class must provide, before this is used:

    * ``env`` -- a :class:`~src.envs.CVRPEnv`, for ``make_input_td`` and ``reference_cost``;
    * ``policy`` -- the shared network;
    * ``num_augment`` and ``augment`` -- built by :func:`validation_augmentation`;
    * ``eval_greedy`` -- whether to also report an unaugmented greedy rollout.
    """

    def _inference_step(self, batch, phase: str) -> dict:
        """Evaluate with POMO's multi-start, dihedral-augmented rollout.

        Args:
            batch: A batch from the environment's dataset.
            phase: ``"val"`` or ``"test"``, which selects the policy decode type.

        Returns:
            A dict of rewards under the keys the declared metrics expect.
        """
        augment = self.augment
        if augment is not None and augment.num_augment != self.num_augment:
            # `unbatchify` below unpacks the reward with `self.num_augment`, so a mismatch
            # would silently reshape it into the wrong axes rather than fail.
            raise ValueError(
                f"augmentation applies {augment.num_augment} augmentations but the model "
                f"counts {self.num_augment}; the reward would be unbatchified incorrectly"
            )

        td = self.env.reset(self.env.make_input_td(batch))
        if augment is not None:
            td = augment(td)

        num_starts = self.env.get_num_starts(td)
        out = self.policy(td, self.env, phase=phase, num_starts=num_starts)

        # [B, num_augment, num_starts] -> best over both, matching POMO's inference.
        reward = unbatchify(out["reward"], (self.num_augment, num_starts))
        out["reward"] = reward.max(dim=-1).values.max(dim=-1).values

        if self.eval_greedy:
            # A second, fresh input: the first `td` was consumed by `reset` and then
            # expanded by the augmentation.
            td_greedy = self.env.reset(self.env.make_input_td(batch))
            greedy = self.policy(td_greedy, self.env, phase=phase, decode_type="greedy")
            out["reward_greedy"] = greedy["reward"]

        # Gap against the file's reference objective, measured on the headline rollout so
        # it moves together with the checkpoint metric. `reward_greedy` is the
        # unaugmented single rollout and will read considerably worse, especially early.
        reference = self.env.reference_cost(batch)
        out["gap_ref"] = (-out["reward"] - reference) / reference
        out["loss"] = None
        return out
