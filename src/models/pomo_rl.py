"""POMO trained by REINFORCE, validated the way the supervised model validates.

Training here is rl4co's POMO unchanged: REINFORCE with the shared baseline on instances drawn
from the environment's generator, which is what :class:`~src.envs.RLCVRPEnv` supplies for the
training phase.

Only the validation differs. rl4co's :class:`~rl4co.models.zoo.POMO` reports ``val/reward`` as
the mean over its 8 augmentations and 100 multi-starts, which is not the quantity
:class:`~src.models.POMOSL` reports and not a number the two can be plotted against each other.
Both models therefore take their validation from
:class:`~src.models.components.pomo_inference.POMOInferenceMixin`, so that ``val/reward``,
``val/reward_greedy`` and ``val/gap_ref`` mean the same thing on either training objective.

The REINFORCE training step is left entirely to :class:`~rl4co.models.zoo.POMO`, so the only
thing this class adds is the metric declaration and the phase split.
"""

from __future__ import annotations

from rl4co.models.zoo import POMO

from src.models.components.pomo_inference import POMOInferenceMixin, validation_augmentation

__all__ = ["POMORL"]


class POMORL(POMOInferenceMixin, POMO):
    """POMO with REINFORCE training and the supervised model's validation metrics.

    Args:
        env: The environment, normally :class:`~src.envs.RLCVRPEnv`.
        num_augment: Number of dihedral augmentations used for inference. Training always uses
            the unaugmented instance. Must be ``1`` or ``8``: rl4co asserts that ``dihedral8``
            is paired with exactly 8 augmentations.
        augment_fn: Augmentation function name, passed to the shared
            :func:`~src.models.components.pomo_inference.validation_augmentation`.
        eval_greedy: Whether to also report an unaugmented greedy rollout. This is the honest
            early-training signal, since the augmented multi-start reward is large from the
            start.
        **kwargs: Forwarded to :class:`~rl4co.models.zoo.POMO`, which is where ``policy``,
            ``policy_kwargs``, ``num_starts`` and the
            :class:`~rl4co.models.rl.common.base.RL4COLitModule` arguments go.
    """

    def __init__(
        self,
        env,
        num_augment: int = 8,
        augment_fn: str = "dihedral8",
        eval_greedy: bool = True,
        **kwargs,
    ) -> None:
        # Set before `super().__init__`, which is what records hyperparameters: POMO calls
        # `save_hyperparameters` first thing, and it walks the frame stack, so this lands in
        # the checkpoint alongside `num_augment` and `augment_fn` as it does for POMOSL.
        self.eval_greedy = eval_greedy

        # log_metrics silently drops any key that is not declared here, so the two new
        # validation keys need declaring or they never reach the logger. Training keeps
        # rl4co's defaults: the supervised model's train metrics are teacher-forcing
        # quantities (ce_loss, ppl, accuracy) with no counterpart here.
        kwargs.setdefault(
            "metrics",
            {
                "train": ["loss", "reward"],
                "val": ["reward", "reward_greedy", "gap_ref"],
                "test": ["reward", "reward_greedy", "gap_ref"],
            },
        )

        super().__init__(
            env, num_augment=num_augment, augment_fn=augment_fn, **kwargs
        )

        # POMO builds its own state augmentation above, from its own defaults. Rebuild it
        # through the shared helper so this model and POMOSL cannot end up augmenting
        # differently -- which would move `val/reward` and `val/gap_ref` on one side only and
        # quietly invalidate the comparison this class exists for. POMO's extra
        # `first_aug_identity` and `feats` knobs are deliberately left unused for that reason.
        self.augment = validation_augmentation(num_augment, augment_fn)

    def shared_step(self, batch, batch_idx, phase: str, dataloader_idx=None) -> dict:
        """Run one REINFORCE training step, or one shared POMO inference step when evaluating.

        Args:
            batch: A batch from the environment's dataset.
            batch_idx: Unused, required positionally by Lightning.
            phase: One of ``"train"``, ``"val"`` or ``"test"``.
            dataloader_idx: Unused, present for multi-dataloader evaluation.

        Returns:
            A dict with ``loss`` plus whatever :meth:`log_metrics` accepted.
        """
        if phase == "train":
            # Named explicitly rather than via `super()`. POMO's training branch is the point
            # of this class, and resolving it through the MRO would let a later addition to
            # `POMOInferenceMixin` -- a `calculate_loss`, say -- silently shadow REINFORCE's
            # with the supervised teacher-forcing one.
            return POMO.shared_step(self, batch, batch_idx, phase, dataloader_idx)

        out = self._inference_step(batch, phase)
        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}
