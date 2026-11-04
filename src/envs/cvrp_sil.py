"""A CVRP environment that serves SIL pseudo-experts for training.

The warm-up phase is exactly :class:`~src.envs.RLCVRPEnv`: training instances come from
rl4co's ``CVRPGenerator``, and validation/testing come from the tagged HGS file. Once SIL
generation has produced records, :meth:`dataset` serves them as supervised trajectories while
validation stays on the same held-out subset as every other run in the repository.

Generation is deliberately *not* triggered here. ``RL4COLitModule.setup`` and
``on_train_epoch_end`` call ``env.dataset`` without a policy in hand, and the search needs the
model; :class:`~src.models.POMOSIL` owns that call and installs the records through
:meth:`set_sil_records`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from src.envs.cvrp import _resolve_size
from src.envs.cvrp_rl import RLCVRPEnv

if TYPE_CHECKING:  # Import-only for annotations: importing `src.sil` eagerly would cycle
    from src.sil.dataset import SILSolutionDataset
    from src.sil.search import SILRecord

__all__ = ["SILCVRPEnv"]


class SILCVRPEnv(RLCVRPEnv):
    """CVRP environment for the RL warm-up and the SIL pseudo-expert training phases.

    Args:
        data_path: Tagged solution file. Training ignores it during both phases (the warm-up
            uses the generator, SIL uses the records), but validation and testing draw their
            held-out range from it, so pass the same file as the ``rl-``/``sl-`` runs.
        val_data_path: Optional separate validation file, forwarded to
            :class:`~src.envs.CVRPEnv`.
        num_loc: Number of customers.
        capacity: Vehicle capacity.
        val_instances: Held-out instances carved from the tail of ``data_path``.
        seed: Base seed for the split and the SIL slot draws.
        **kwargs: Forwarded to :class:`~src.envs.RLCVRPEnv`.
    """

    def __init__(
        self,
        data_path: str,
        val_data_path: str | None = None,
        num_loc: int | None = None,
        capacity: float | None = None,
        val_instances: int = 10_000,
        seed: int = 0,
        **kwargs,
    ) -> None:
        self._sil_records: list[SILRecord] | None = None
        super().__init__(
            data_path=data_path,
            val_data_path=val_data_path,
            num_loc=num_loc,
            capacity=capacity,
            val_instances=val_instances,
            seed=seed,
            **kwargs,
        )

    # -- SIL records ---------------------------------------------------------------

    @property
    def sil_records(self) -> list[SILRecord] | None:
        """The installed pseudo-expert records, or ``None`` during the warm-up phase."""
        return self._sil_records

    def set_sil_records(self, records: Sequence[SILRecord]) -> None:
        """Install the pseudo-experts used by the next training epoch."""
        self._sil_records = list(records)

    def clear_sil_records(self) -> None:
        """Forget the pseudo-experts, falling back to the generator dataset."""
        self._sil_records = None

    # -- data ----------------------------------------------------------------------

    def dataset(
        self, batch_size: int | Sequence = (), phase: str = "train", filename: str | None = None
    ):
        """Serve SIL trajectories once records exist; otherwise defer to ``RLCVRPEnv``.

        Args:
            batch_size: For the SIL phase, the number of trajectory slots to expose. For the
                warm-up phase, the number of generated instances, as in POMO RL.
            phase: ``"train"``, ``"val"`` or ``"test"``.
            filename: Optional path overriding the configured file.
        """
        if phase == "train" and filename is None and self._sil_records:
            # Imported here rather than at module scope: `src.envs` imports this module, so a
            # top-level `src.sil` import would close the cycle
            # `src.sil -> dataset -> src.envs -> cvrp_sil -> src.sil`.
            from src.sil.dataset import SILSolutionDataset

            self._epoch += 1
            return SILSolutionDataset(
                self._sil_records,
                length=_resolve_size(batch_size),
                seed=self.seed + self._epoch,
            )
        return super().dataset(batch_size, phase=phase, filename=filename)
