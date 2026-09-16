"""POMO trained by supervised learning on reference CVRP solutions.

POMO's usual training signal is REINFORCE with a shared baseline. Here it is replaced by
teacher forcing: the reference trajectory is replayed through the environment, and the
model is trained with cross-entropy on the action it should have taken at every step.

rl4co already contains the decoding machinery for this. Passing ``actions`` to
``ConstructivePolicy.forward`` switches the decode strategy to ``"evaluate"``, which
consumes the supplied actions verbatim instead of sampling, and returns their
log-likelihood. Because the environment is stepped with the reference action regardless
of what the model predicts, no multi-start or start-node selection machinery is needed:
forcing a start and letting the model choose one produce an identical trajectory, and
differ only in whether the first step enters the loss.

The training data supplies more than one reference trajectory per instance --
``2R`` of them, all exactly optimal -- so the model sees the same instance from every
legal starting point. See :mod:`src.envs.dataset`.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

import rl4co.models.common.constructive.base as _constructive
from rl4co.data.transforms import StateAugmentation
from rl4co.models.rl import RL4COLitModule
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.utils.ops import unbatchify

__all__ = ["POMOSL"]

# Grabbed before anything can replace it, so the context manager below is re-entrant with
# respect to repeated use and always restores the genuine function.
_ORIGINAL_GET_LOG_LIKELIHOOD = _constructive.get_log_likelihood


@contextmanager
def _capture_logprobs(sink: dict):
    """Expose the decoder's full action distribution for the duration of the block.

    ``ConstructivePolicy.forward`` keeps only the per-step log-likelihood of the action
    that was actually taken, obtained by gathering from a ``[B, T, num_actions]``
    log-probability tensor that is then dropped. Token-level accuracy needs the argmax
    over all actions, so the one call that receives that tensor is intercepted and its
    argument stashed in ``sink`` under the key ``"logprobs"``.

    ``store_all_logp=True`` must be passed to the policy for the tensor to be the full
    distribution rather than an already-gathered ``[B, T]``; ``shared_step`` does this.
    The tensor is left in the autograd graph, so it is only kept alive by the caller's
    reference to ``sink`` and is freed as soon as ``shared_step`` returns.

    Note:
        This patches a module global, so it is not thread-safe. Training here is a single
        process, and the block covers exactly one policy call.

    Args:
        sink: A dict the caller owns, populated with ``"logprobs"`` when the policy
            performs a teacher-forced decode.

    Yields:
        The ``sink`` dict, for convenience.
    """

    def _patched(logprobs, actions=None, mask=None, return_sum=True):
        sink["logprobs"] = logprobs
        return _ORIGINAL_GET_LOG_LIKELIHOOD(logprobs, actions, mask, return_sum)

    _constructive.get_log_likelihood = _patched
    try:
        yield sink
    finally:
        _constructive.get_log_likelihood = _ORIGINAL_GET_LOG_LIKELIHOOD


class POMOSL(RL4COLitModule):
    """POMO with a supervised (teacher-forced) objective.

    Subclasses :class:`~rl4co.models.RL4COLitModule` rather than
    :class:`~rl4co.models.zoo.POMO`: the latter asserts a shared baseline, requires
    ``num_starts > 1`` while training and rewrites the decode types to their multi-start
    variants, none of which apply to a supervised objective. The network is unchanged --
    the same :class:`~rl4co.models.zoo.am.AttentionModelPolicy` with POMO's
    hyperparameters -- so multi-start inference still works at validation and test time.

    Args:
        env: The environment, normally :class:`~src.envs.CVRPEnv`.
        policy: Optional pre-built policy. Defaults to
            :class:`~rl4co.models.zoo.am.AttentionModelPolicy` with POMO's settings.
        policy_kwargs: Extra arguments merged into the default policy configuration.
        num_augment: Number of dihedral augmentations used for inference. Training always
            uses the unaugmented instance. Must be ``1`` or ``8``: rl4co asserts that
            ``dihedral8`` is paired with exactly 8 augmentations.
        augment_fn: Augmentation function name, passed to ``StateAugmentation``.
        eval_greedy: Whether to also report an unaugmented greedy rollout. This is the
            honest early-training signal, since the augmented multi-start reward is
            large from the start.
        **kwargs: Forwarded to :class:`~rl4co.models.RL4COLitModule`.
    """

    def __init__(
        self,
        env,
        policy=None,
        policy_kwargs: dict | None = None,
        num_augment: int = 8,
        augment_fn: str = "dihedral8",
        eval_greedy: bool = True,
        **kwargs,
    ) -> None:
        if policy is None:
            defaults = {
                "num_encoder_layers": 6,
                "normalization": "instance",
                "use_graph_context": False,
            }
            defaults.update(policy_kwargs or {})
            policy = AttentionModelPolicy(env_name=env.name, **defaults)

        self.num_augment = num_augment
        self.eval_greedy = eval_greedy
        self.augment = (
            StateAugmentation(num_augment=num_augment, augment_fn=augment_fn)
            if num_augment > 1
            else None
        )

        # log_metrics silently drops any key that is not declared here, so an
        # undeclared supervised loss would vanish from the logs without warning.
        kwargs.setdefault(
            "metrics",
            {
                "train": ["loss", "ce_loss", "ppl", "accuracy"],
                "val": ["reward", "reward_greedy", "gap_ref"],
                "test": ["reward", "reward_greedy", "gap_ref"],
            },
        )
        super().__init__(env, policy, **kwargs)

    # -- steps -------------------------------------------------------------------

    def shared_step(self, batch, _batch_idx, phase: str, dataloader_idx=None) -> dict:
        """Run one supervised training step, or one POMO inference step when evaluating.

        Args:
            batch: A batch from :class:`~src.envs.dataset.CVRPSolutionDataset`.
            _batch_idx: Unused, required positionally by Lightning.
            phase: One of ``"train"``, ``"val"`` or ``"test"``.
            dataloader_idx: Unused, present for multi-dataloader evaluation.

        Returns:
            A dict with ``loss`` plus whatever :meth:`log_metrics` accepted.
        """
        if phase == "train":
            # Read the reference trajectories before resetting: `CVRPEnv.reset` mutates
            # its input in place, so anything read off `batch` afterwards is widened.
            actions, _, valid = self.env.build_expert_actions(batch)
            td = self.env.reset(self.env.make_input_td(batch))
            # `store_all_logp=True` makes the decoder retain the full action
            # distribution; the capture hands it to `calculate_loss` for the accuracy.
            captured: dict = {}
            with _capture_logprobs(captured):
                out = self.policy(
                    td,
                    self.env,
                    phase="train",
                    actions=actions,
                    store_all_logp=True,
                    return_sum_log_likelihood=False,
                )
            out = self.calculate_loss(
                td, batch, out, valid=valid, logprobs=captured.get("logprobs")
            )
        else:
            out = self._inference_step(batch, phase)

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def _inference_step(self, batch, phase: str) -> dict:
        """Evaluate with POMO's multi-start, dihedral-augmented rollout.

        Args:
            batch: A batch from the environment's dataset.
            phase: ``"val"`` or ``"test"``, which selects the policy decode type.

        Returns:
            A dict of rewards under the keys the declared metrics expect.
        """
        td = self.env.reset(self.env.make_input_td(batch))
        if self.augment is not None:
            td = self.augment(td)

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

    def calculate_loss(
        self, _td, _batch, policy_out: dict, valid=None, logprobs=None, **_kwargs
    ) -> dict:
        """Compute per-step cross-entropy against the reference actions.

        Args:
            _td: The final environment state; unused, but kept so the signature mirrors
                :meth:`~rl4co.models.REINFORCE.calculate_loss`.
            _batch: The input batch; unused for the same reason.
            policy_out: Output of the policy, updated in place.
            valid: ``[B, T]`` mask marking real steps. Padded steps are excluded, as are
                steps past the point where the decoder stopped consuming actions.
            logprobs: Optional ``[B, T, num_actions]`` action distribution captured from
                the decoder. When given, a token-level accuracy is added.

        Returns:
            The updated ``policy_out``, carrying ``loss``, ``ce_loss``, ``ppl`` and, when
            ``logprobs`` is supplied, ``accuracy``.
        """
        log_likelihood = policy_out["log_likelihood"]
        mask = valid[:, : log_likelihood.shape[1]].to(log_likelihood.dtype)
        ce_loss = -((log_likelihood * mask).sum() / mask.sum())

        policy_out["ce_loss"] = ce_loss
        policy_out["ppl"] = ce_loss.detach().exp()
        policy_out["loss"] = ce_loss
        if logprobs is not None:
            policy_out["accuracy"] = self._token_accuracy(
                logprobs, policy_out["actions"], mask
            )
        return policy_out

    @staticmethod
    def _token_accuracy(
        logprobs: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Fraction of scored steps where the model's greedy action is the reference one.

        The actions returned by the decoder under teacher forcing are the reference
        actions, so this compares them against ``argmax`` of the masked action
        distribution -- the action a greedy rollout would take. Unlike the cross-entropy
        it is not a smooth quantity, which makes it the more legible of the two when
        judging whether the model has learned the reference or is only getting the
        high-confidence steps right.

        Args:
            logprobs: ``[B, T, num_actions]`` log-probabilities, already action-masked by
                :func:`~rl4co.utils.decoding.process_logits`.
            actions: ``[B, T]`` reference actions, aligned step-for-step with ``logprobs``.
            mask: ``[B, T]`` weight, ``1`` on scored steps and ``0`` elsewhere.

        Returns:
            A scalar tensor: the masked mean over steps and batch elements.
        """
        if logprobs.shape[:2] != actions.shape:
            raise ValueError(
                f"logprobs {tuple(logprobs.shape)} and actions {tuple(actions.shape)} "
                "are not aligned; the capture expects a full [B, T, num_actions] tensor"
            )
        correct = (logprobs.argmax(dim=-1) == actions).to(mask.dtype)
        return (correct * mask).sum() / mask.sum()

    # -- inference helpers --------------------------------------------------------

    @torch.no_grad()
    def greedy_rollout(self, batch, decode_type: str = "greedy") -> dict:
        """Roll out the learned policy on a batch, without augmentation or multi-start.

        Args:
            batch: A batch from the environment's dataset.
            decode_type: Decoding strategy, ``"greedy"`` or ``"sampling"``.

        Returns:
            The policy output dict, including ``reward``, ``actions`` and
            ``log_likelihood``.
        """
        td = self.env.reset(self.env.make_input_td(batch))
        return self.policy(td, self.env, phase="val", decode_type=decode_type)
