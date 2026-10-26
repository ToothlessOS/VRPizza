"""POMO trained LEHD-style: reconstruct a sampled partial solution from its forced start.

The difference from :class:`~src.models.POMOSL` is the training example and the state it is
replayed from, not the network. Each batch item is a reduced problem (depot + the window's
``w`` customers) whose start has already been served, and the target is the remaining
``w-1`` customer actions plus a depot action at every interior route boundary. The forward
call, loss machinery, metrics and validation are the supervised POMO ones; only the train
branch changes.

The loss is a **per-row** mean cross-entropy. ``POMOSL`` averages over all scored tokens,
which would weight a row with more route boundaries (and therefore more depot steps) more
heavily. LEHD's batches give every window the same ``w-1`` decisions, so rows are equally
weighted; with POMO's explicit depot actions the row length varies, and the per-row mean
restores that balance. The global-token mean is still reported as ``token_ce`` for
comparability with the existing supervised runs.
"""

from __future__ import annotations

import torch

from src.models.pomo_sl import POMOSL, _capture_logprobs

__all__ = ["LEHDPOMOSL"]


class LEHDPOMOSL(POMOSL):
    """POMO with LEHD's partial-solution reconstruction objective.

    The architecture is exactly :class:`~src.models.POMOSL`'s -- same encoder, decoder and
    ``state_dict`` -- so the checkpoint can be evaluated by the plain POMO driver and loaded
    into a ``POMOSL`` instance. Only :meth:`shared_step`'s train branch and the loss weighting
    differ.

    Args:
        env: A :class:`~src.envs.LEHDCVRPEnv`, which supplies ``reset_partial``.
        **kwargs: Forwarded to :class:`~src.models.POMOSL`.
    """

    def __init__(self, env, **kwargs) -> None:
        # `POMOSL` declares the metrics it logs; add the global-token diagnostic. Validation
        # keys stay identical so the checkpoints remain comparable with the other runs.
        kwargs.setdefault(
            "metrics",
            {
                "train": ["loss", "ce_loss", "token_ce", "ppl", "accuracy"],
                "val": ["reward", "reward_greedy", "gap_ref"],
                "test": ["reward", "reward_greedy", "gap_ref"],
            },
        )
        super().__init__(env, **kwargs)

    def shared_step(self, batch, _batch_idx, phase: str, dataloader_idx=None) -> dict:
        """Run one partial-reconstruction step, or one POMO inference step when evaluating.

        Args:
            batch: A batch from :class:`~src.envs.lehd_dataset.LEHDPartialDataset`.
            _batch_idx: Unused, required positionally by Lightning.
            phase: One of ``"train"``, ``"val"`` or ``"test"``.
            dataloader_idx: Unused, present for multi-dataloader evaluation.

        Returns:
            A dict with ``loss`` plus whatever :meth:`log_metrics` accepted.
        """
        if phase == "train":
            # Read the target before resetting: `make_input_td` and `reset` are free to
            # rewrite the batch's instance fields, and the training target must not move.
            actions = batch["actions"]
            valid = batch["valid"]
            td = self.env.reset_partial(batch)

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

    def calculate_loss(
        self, _td, _batch, policy_out: dict, valid=None, logprobs=None, **_kwargs
    ) -> dict:
        """Compute the per-row reconstruction cross-entropy.

        Args:
            _td: The final environment state; unused.
            _batch: The input batch; unused.
            policy_out: Output of the policy, updated in place.
            valid: ``[B, T]`` mask marking target steps. Rows may have different target
                lengths because a route boundary costs an extra depot action.
            logprobs: Optional ``[B, T, num_actions]`` distribution captured from the
                decoder, used only for the token-level accuracy.

        Returns:
            The updated ``policy_out`` with ``loss``, ``ce_loss`` (per-row mean, the training
            objective), ``token_ce`` (global token mean, for comparison with ``POMOSL``),
            ``ppl`` and, when ``logprobs`` is supplied, ``accuracy``.
        """
        log_likelihood = policy_out["log_likelihood"]
        mask = valid[:, : log_likelihood.shape[1]].to(log_likelihood.dtype)
        if log_likelihood.shape[1] != mask.shape[1]:
            raise ValueError(
                f"log-likelihood {tuple(log_likelihood.shape)} and valid "
                f"{tuple(valid.shape)} do not share a step axis"
            )

        row_sums = (log_likelihood * mask).sum(dim=1)
        row_counts = mask.sum(dim=1).clamp_min(1.0)
        per_row = -row_sums / row_counts
        ce_loss = per_row.mean()

        policy_out["ce_loss"] = ce_loss
        policy_out["token_ce"] = -((log_likelihood * mask).sum() / mask.sum().clamp_min(1.0))
        policy_out["ppl"] = ce_loss.detach().exp()
        policy_out["loss"] = ce_loss
        if logprobs is not None:
            policy_out["accuracy"] = self._token_accuracy(
                logprobs, policy_out["actions"], mask
            )
        return policy_out
