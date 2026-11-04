"""POMO trained by REINFORCE on LEHD partial solutions.

The examples and the state are LEHD's (see :mod:`src.envs.cvrp_rl_lehd` and
``docs/LEHD_RL.md``); the network, validation and checkpoint interface are POMO's. This module
replaces only the training branch: one sampled reconstruction per partial from the fixed
spliced start, scored by its reduced tour cost, with one of two baselines.

* ``advantage_baseline="teacher"`` (default): the baseline is the HGS window's reduced cost,
  ``C_teacher``. It is a detached scalar computed from the label, never a target, so the policy
  can beat it. One rollout per partial is enough; K > 1 is allowed and simply averages the
  gradient over samples that share the baseline.
* ``advantage_baseline="shared"``: the baseline is the mean cost of K sampled reconstructions
  of the same window. This arm never reads the teacher actions or the reference cost, so its
  training signal is label- and ground-truth-free. It needs K >= 2; K = 1 gives zero advantage.

Both arms form the advantage relative to their baseline's cost and divide the score-function
term by the window length ``w``. The derivation, the fixed-start invariant and the data-use
boundary are in ``docs/LEHD_RL.md``; the constants are pinned by
``scripts/verify_lehd_rl.py``.
"""

from __future__ import annotations

import torch

from rl4co.utils.ops import unbatchify

from src.envs.cvrp_rl_lehd import reduced_window_cost
from src.models.pomo_rl import POMORL

__all__ = ["LEHDPOMORL"]


class LEHDPOMORL(POMORL):
    """POMO with a REINFORCE objective on LEHD partial solutions.

    The validation branch is inherited from :class:`~src.models.POMORL` (and therefore from
    :class:`~src.models.components.pomo_inference.POMOInferenceMixin`), so ``val/reward``,
    ``val/reward_greedy`` and ``val/gap_ref`` mean exactly what they mean for the ``rl-``,
    ``sl-`` and ``lehd-`` runs. The architecture is also unchanged, so a checkpoint's state
    dict loads into any of them.

    Args:
        env: A :class:`~src.envs.LEHDRLCVRPEnv`.
        advantage_baseline: ``"teacher"`` or ``"shared"``; see the module docstring.
        num_samples: Rollouts per partial. Defaults to 1 for ``"teacher"`` and 4 for
            ``"shared"``. Must be at least 2 for ``"shared"``.
        **kwargs: Forwarded to :class:`~src.models.POMORL` (``num_augment``, ``eval_greedy``,
            the Lightning-module arguments and so on).
    """

    def __init__(
        self,
        env,
        advantage_baseline: str = "teacher",
        num_samples: int | None = None,
        **kwargs,
    ) -> None:
        if advantage_baseline not in {"teacher", "shared"}:
            raise ValueError(
                f"advantage_baseline must be 'teacher' or 'shared', got {advantage_baseline!r}"
            )
        if num_samples is None:
            num_samples = 1 if advantage_baseline == "teacher" else 4
        self.advantage_baseline = advantage_baseline
        self.num_samples = int(num_samples)
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        if advantage_baseline == "shared" and self.num_samples < 2:
            raise ValueError(
                "the shared baseline needs at least two rollouts: with one sample its "
                "advantage is identically zero"
            )

        # Declared here rather than after `super().__init__`: `POMORL` sets its own default
        # with `setdefault`, so this must already be in `kwargs` for the train keys to reach
        # `log_metrics`. Validation keys stay identical to the other runs.
        kwargs.setdefault(
            "metrics",
            {
                "train": ["loss", "reward", "regret", "baseline_cost", "advantage_abs"],
                "val": ["reward", "reward_greedy", "gap_ref"],
                "test": ["reward", "reward_greedy", "gap_ref"],
            },
        )
        super().__init__(env, **kwargs)

    def shared_step(self, batch, batch_idx, phase: str, dataloader_idx=None) -> dict:
        """One sampled-partial REINFORCE step, or the shared POMO inference step.

        Args:
            batch: A batch from :class:`~src.envs.lehd_dataset.LEHDPartialDataset` during
                training, or the full-instance validation batch otherwise.
            batch_idx: Unused, required positionally by Lightning.
            phase: ``"train"``, ``"val"`` or ``"test"``.
            dataloader_idx: Present for multi-dataloader evaluation.

        Returns:
            A dict with ``loss`` plus whatever :meth:`log_metrics` accepted.
        """
        if phase != "train":
            return super().shared_step(batch, batch_idx, phase, dataloader_idx)

        # Read the label before the reset. It is the only thing the teacher arm takes from
        # `actions`; the shared arm does not read `actions` or `cost` at all.
        if self.advantage_baseline == "teacher":
            teacher_cost = reduced_window_cost(
                batch["depot"],
                batch["locs"],
                torch.cat([batch["start_node"][:, None], batch["actions"]], dim=1),
            )
        else:
            teacher_cost = None

        # One sample is plain sampling; passing num_samples=1 would only trigger rl4co's
        # "num_starts=1 is ignored" warning. K > 1 expands the batch into K samples.
        decoding = {"decode_type": "sampling"}
        if self.num_samples > 1:
            decoding["num_samples"] = self.num_samples

        td = self.env.reset_partial(batch)
        out = self.policy(td, self.env, phase="train", **decoding)

        if self.num_samples > 1:
            reward = unbatchify(out["reward"], self.num_samples)  # [B, K]
            logp = unbatchify(out["log_likelihood"], self.num_samples)  # [B, K]
        else:
            reward = out["reward"][:, None]
            logp = out["log_likelihood"][:, None]

        out = self.calculate_loss(td, batch, out, reward, logp, teacher_cost=teacher_cost)
        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def calculate_loss(
        self,
        _td,
        batch,
        policy_out: dict,
        reward: torch.Tensor,
        log_likelihood: torch.Tensor,
        teacher_cost: torch.Tensor | None = None,
    ) -> dict:
        """Form the relative advantage and the horizon-normalised REINFORCE loss.

        Args:
            _td: The post-start environment state; unused by the loss.
            batch: The partial batch; only ``num_loc`` (``w``) is read here.
            policy_out: The policy output, updated in place with the logged quantities.
            reward: ``[B, K]`` terminal rewards, i.e. minus the reduced closed-tour cost.
            log_likelihood: ``[B, K]`` summed log-probabilities of the sampled actions.
            teacher_cost: ``[B]`` positive teacher reduced costs; required by the teacher arm
                and ignored by the shared arm.

        Returns:
            ``policy_out`` with ``loss``, ``baseline_cost``, ``advantage_abs`` and, for the
            teacher arm, ``regret``.
        """
        reward = reward.reshape(-1, self.num_samples)
        log_likelihood = log_likelihood.reshape(-1, self.num_samples)

        if self.advantage_baseline == "teacher":
            if teacher_cost is None:
                raise ValueError("the teacher baseline requires teacher_cost")
            scale = teacher_cost[:, None].clamp_min(1e-6)  # C_teacher
            baseline = -scale
        else:
            # reward = -C, so the mean reward is -mean(C); the normaliser is +mean(C).
            baseline = reward.mean(dim=1, keepdim=True)
            scale = (-baseline).clamp_min(1e-6)

        advantage = (reward - baseline) / scale  # dimensionless; > 0 means better than baseline
        horizon = batch["num_loc"].to(advantage.dtype)[:, None]  # w
        loss = -(advantage * log_likelihood / horizon).mean()

        policy_out.update(
            {
                "loss": loss,
                "baseline_cost": (-baseline).squeeze(1).detach(),
                "advantage_abs": advantage.detach().abs().mean(),
            }
        )
        if self.advantage_baseline == "teacher":
            # Positive when the sample is worse than the expert; 0 when it matches.
            policy_out["regret"] = ((-reward - scale) / scale).mean().detach()
        return policy_out
