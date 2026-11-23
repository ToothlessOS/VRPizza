"""LEHD-backbone local-reconstruction SIL.

The pipeline is :class:`~src.models.SILRCPOMOLitModule` unchanged: an RL warm-up
on generated CVRP20, then label cycles that alternate a full RRC repair pass with
teacher-forced CE on sampled sub-paths. Only the policy is different --
:class:`~src.models.components.lehd_policy.LEHDPolicy`, the reference's light
encoder / heavy decoder.

The reference scores a ``2N`` ``(customer, via_depot)`` action per step, while the
repo's partial labels are POMO's ``0 = depot, 1..N`` sequences. The policy exposes
the per-step marginal as ``N+1`` logits (see ``lehd_policy.py``), which is enough
for rollouts, RRC and the coarse N+1 CE. For training, this module replaces that
coarse CE with the reference's exact one:

* a customer token that does not follow a depot is scored as ``direct`` at its
  own state;
* a depot token is scored as ``via`` for the **next** customer at the pre-depot
  state, and the structural customer token that follows is not scored again.

Each reference action is one of the ``w-1`` customers in the window, so the
per-row denominator is the reference's, not the POMO token count. This needs the
decoder's raw direct/via logits, captured through the side channel on
``LEHDDecoder``; the N+1 log-likelihood that ``ConstructivePolicy`` returns is
ignored by the exact loss.

Set ``exact_ce=False`` to fall back to the repo's N+1 partial CE (useful for
debugging the adapter, but it is not the reference objective).
"""

from __future__ import annotations

import torch

from src.models.components.lehd_policy import LEHDPolicy
from src.models.pomo_sil_rc import SILRCPOMOLitModule
from src.models.pomo_sl_lehd import partial_teacher_forced_train_step

__all__ = ["SILRCLEHDLitModule", "lehd_exact_ce_step"]


def _gather_step(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather one entry per row from ``[B, n]``."""
    return values.gather(1, index[:, None]).squeeze(1)


def _calculate_exact_ce(
    module,
    batch,
    policy_out: dict,
    captures: list[dict[str, torch.Tensor]],
) -> dict:
    """Reference 2N teacher-forced CE from the decoder's per-step captures.

    Args:
        module: The LEHD Lightning module (only ``env`` is read here).
        batch: The partial batch; ``actions [B, T]`` and ``valid [B, T]``.
        policy_out: Output of the policy call, updated in place.
        captures: One dict per decode step with ``direct [B, w]``, ``via
            [B, w]``, ``unselected [B, w]`` and ``logz [B]``.

    Returns:
        ``policy_out`` with ``loss``, ``ce_loss``, ``token_ce``, ``ppl`` and
        ``accuracy``. The accuracy is over the reference's 2N argmax.
    """
    actions = batch["actions"]
    valid = batch["valid"]
    size, width = actions.shape
    device = actions.device
    zero = torch.zeros(size, device=device)

    if captures:
        n_customers = int(captures[0]["direct"].shape[1])
    else:
        n_customers = int(batch["num_loc"].reshape(-1)[0])

    logp_sum = torch.zeros(size, device=device)
    count = torch.zeros(size, device=device)
    correct = torch.zeros(size, device=device)
    skip = torch.zeros(size, dtype=torch.bool, device=device)

    for step in range(min(width, len(captures))):
        capture = captures[step]
        direct = capture["direct"]
        via = capture["via"]
        unselected = capture["unselected"]
        log_z = capture["logz"]

        action = actions[:, step]
        # A row whose pre-action state has no unselected customer (the window
        # reached the route end) has log_z = -inf. Its remaining tokens are
        # structural; scoring them would turn the per-row CE into +/-inf.
        active = valid[:, step] & ~skip & unselected.any(dim=-1)
        action_index = (action.clamp_min(1) - 1).clamp(0, n_customers - 1)

        if step + 1 < width:
            next_action = actions[:, step + 1]
            next_valid = valid[:, step + 1]
        else:
            next_action = torch.zeros_like(action)
            next_valid = torch.zeros_like(valid[:, step])
        next_index = (next_action.clamp_min(1) - 1).clamp(0, n_customers - 1)

        logp_direct = _gather_step(direct, action_index) - log_z
        logp_via_next = _gather_step(via, next_index) - log_z

        is_customer = active & (action > 0)
        is_depot = active & (action == 0) & next_valid
        logp = torch.where(
            is_customer, logp_direct, torch.where(is_depot, logp_via_next, zero)
        )
        target_mask = (is_customer | is_depot).to(logp.dtype)
        logp_sum = logp_sum + logp * target_mask
        count = count + target_mask

        flat = torch.cat(
            [
                direct.masked_fill(~unselected, float("-inf")),
                via.masked_fill(~unselected, float("-inf")),
            ],
            dim=-1,
        )
        predicted = flat.argmax(dim=-1)
        target = torch.where(is_depot, next_index + n_customers, action_index)
        correct = correct + ((predicted == target) & (target_mask > 0)).to(logp.dtype)

        # The customer token after a depot implements the via action already
        # scored above; it is structural and must not be scored a second time.
        skip = is_depot

    count = count.clamp_min(1.0)
    per_row = -logp_sum / count
    ce_loss = per_row.mean()

    policy_out["loss"] = ce_loss
    policy_out["ce_loss"] = ce_loss
    policy_out["token_ce"] = -(logp_sum.sum() / count.sum().clamp_min(1.0))
    policy_out["ppl"] = ce_loss.detach().exp()
    policy_out["accuracy"] = correct.sum() / count.sum().clamp_min(1.0)
    return policy_out


def lehd_exact_ce_step(module, batch) -> dict:
    """One reference-style teacher-forced step for the LEHD backbone.

    Runs the policy in ``evaluate`` mode (so the target actions drive the
    environment), captures the decoder's raw 2N logits at every step, and scores
    the reference action sequence. The policy's own N+1 log-likelihood is not
    used.

    Args:
        module: A :class:`SILRCLEHDLitModule` with ``env`` and ``policy``.
        batch: A partial batch carrying ``actions``, ``valid``, ``start_node``,
            ``used_capacity0`` and the reduced instance fields.

    Returns:
        The policy output dict updated by :func:`_calculate_exact_ce`.
    """
    actions = batch["actions"]
    valid = batch["valid"]
    td = module.env.reset_partial(batch)

    decoder = module.policy.decoder
    decoder.set_capture(True)
    try:
        out = module.policy(
            td,
            module.env,
            phase="train",
            actions=actions,
            return_sum_log_likelihood=False,
        )
        captures = decoder.pop_captures()
    finally:
        decoder.set_capture(False)

    if valid.shape[1] != actions.shape[1]:
        raise ValueError("actions and valid do not share the step axis")
    return _calculate_exact_ce(module, batch, out, captures)


class SILRCLEHDLitModule(SILRCPOMOLitModule):
    """SIL-RC with the reference LEHD encoder/decoder.

    Args:
        env: A :class:`~src.envs.SILRCCVRPEnv` or
            :class:`~src.envs.SILRCStepCVRPEnv`.
        policy: A pre-built :class:`LEHDPolicy`, or ``None`` to build one with
            the reference configuration.
        policy_kwargs: Extra ``LEHDPolicy`` arguments when ``policy`` is ``None``.
        exact_ce: Use the reference 2N teacher-forced CE (default). ``False``
            falls back to the repo's N+1 partial CE.
        inference_num_starts: Default start count for validation/test when the
            caller does not pass one. ``1`` is the reference's greedy rollout;
            the heavy decoder makes POMO's ``N`` starts intractable.
        gradient_checkpointing: Checkpoint the decoder stack per decode step
            while training.
        **kwargs: Forwarded to :class:`~src.models.SILRCPOMOLitModule`.
    """

    BACKBONE = "lehd"

    def __init__(
        self,
        env,
        policy=None,
        policy_kwargs: dict | None = None,
        exact_ce: bool = True,
        inference_num_starts: int = 1,
        gradient_checkpointing: bool = True,
        **kwargs,
    ) -> None:
        if policy is None:
            options = {
                "env_name": env.name,
                "use_checkpointing": bool(gradient_checkpointing),
            }
            options.update(policy_kwargs or {})
            policy = LEHDPolicy(**options)
        else:
            setter = getattr(policy, "set_gradient_checkpointing", None)
            if callable(setter):
                setter(bool(gradient_checkpointing))

        self.exact_ce = bool(exact_ce)
        self.inference_num_starts = int(inference_num_starts)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        super().__init__(env, policy=policy, **kwargs)

    def policy_signature(self) -> str:
        """Architecture fingerprint for label/checkpoint metadata."""
        signature = getattr(self.policy, "policy_signature", None)
        if callable(signature):
            return signature()
        return f"lehd-{self.BACKBONE}"

    def _sil_step(self, batch) -> dict:
        """Reference exact 2N CE, or the repo's N+1 CE when disabled."""
        if self.exact_ce:
            return lehd_exact_ce_step(self, batch)
        return partial_teacher_forced_train_step(self, batch)
