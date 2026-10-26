"""Score checkpoints with Random Re-Construct and its matched multistart baseline.

The plain evaluator (``notebooks/eval_pomo.py``) reports POMO's ``best of starts x
augmentations``. This driver adds LEHD's inference-time improvement on top of that incumbent:

* it runs the plain rollout once and records it as ``{model}_ms{A}x{S}`` (or
  ``{model}_greedy`` for LEHD's single-start protocol);
* it then runs RRC from that exact solution to each requested budget and records
  ``{model}_..._rrc{budget}``.

Both rows come from the same batch, with the same seeds, so the paired difference measures
RRC rather than a different initial solution. The initial POMO/8x flags are the same ones the
plain evaluator uses, which lets the final table separate three effects:

1. training -- ``lehd`` multistart+8x vs ``rl``/``sl`` multistart+8x;
2. RRC -- ``lehd`` RRC vs ``lehd`` multistart+8x;
3. combined -- ``lehd`` RRC vs the baseline training methods.

Run it with the project's virtualenv::

    .venv/bin/python notebooks/eval_rrc.py --limit 8 --rrc-budgets 1 2 5
    .venv/bin/python notebooks/eval_rrc.py --datasets vrp100_test_lkh
    .venv/bin/python notebooks/eval_rrc.py --num-augment 1 --num-starts 1
    .venv/bin/python notebooks/eval_rrc.py --report-from logs/eval_rrc/results.npz

The reconstruction always runs a single greedy trajectory: the window's start is fixed by the
splice, so POMO multistart cannot be used inside a reconstruction without changing the
partial problem. Multistart applies to the initial incumbent, which is where POMO's inference
multiplier lives.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.envs import InstanceSet, LEHDEvalCVRPEnv, load_instances, project_root
from src.rrc import RRCConfig, RRCStats, run_rrc

from notebooks.eval_pomo import (
    batch_size_for,
    build_batch,
    discover_checkpoints,
    discover_datasets,
    load_model,
    should_normalize,
    summarize,
)

DEFAULT_OUT = "logs/eval_rrc/results.npz"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the RRC evaluation flags."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--datasets", nargs="*", default=None, help="file stems to evaluate (default: all)"
    )
    parser.add_argument("--checkpoints", nargs="*", default=None, help="labels, e.g. lehd rl")
    parser.add_argument("--limit", type=int, default=None, help="instances per dataset")
    parser.add_argument("--batch-size", type=int, default=None, help="override the per-N batch")
    parser.add_argument(
        "--num-augment",
        type=int,
        default=8,
        help="dihedral augmentations for the initial incumbent / plain baseline",
    )
    parser.add_argument(
        "--num-starts",
        type=int,
        default=None,
        help="POMO starts for the initial incumbent (default: N; 1 disables multistart)",
    )
    parser.add_argument("--rrc-budgets", nargs="+", type=int, default=[50, 100, 300, 500, 1000])
    parser.add_argument("--rrc-w-min", type=int, default=4)
    parser.add_argument("--rrc-w-max", type=int, default=None)
    parser.add_argument("--rrc-seed", type=int, default=12)
    parser.add_argument(
        "--rrc-init",
        choices=("multistart", "greedy"),
        default="multistart",
        help="multistart (POMO best of A x S) or greedy (LEHD's single-start protocol)",
    )
    parser.add_argument(
        "--rrc-num-augment",
        type=int,
        default=1,
        help="dihedral augmentations per reconstruction (1 is faithful to LEHD)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--report-from", default=None, help="re-print a saved .npz, no inference")
    return parser.parse_args(argv)


def _starts_label(num_starts: int | None) -> str:
    """Human-readable start count for result labels."""
    return "N" if num_starts is None else str(int(num_starts))


def _label_parts(model_label: str, args: argparse.Namespace, budgets: tuple[int, ...]):
    """Return the baseline label and the per-budget RRC labels for one checkpoint."""
    if args.rrc_init == "multistart":
        baseline = f"{model_label}_ms{args.num_augment}x{_starts_label(args.num_starts)}"
    else:
        baseline = f"{model_label}_greedy"
    return baseline, [f"{baseline}_rrc{budget}" for budget in budgets]


def _looks_like_rrc(label: str) -> bool:
    """Whether a result label is an RRC budget row."""
    return "_rrc" in label


def _budget_of(label: str) -> int:
    """Trailing budget of an RRC result label."""
    try:
        return int(label.rsplit("_rrc", 1)[1])
    except (IndexError, ValueError):
        return 0


def print_report(
    arrays: dict[str, np.ndarray],
    stats: dict[str, RRCStats],
    dataset_order: list[str],
) -> None:
    """Print one block per dataset: baselines, RRC rows, paired differences and cost stats."""
    for name in dataset_order:
        labels = sorted(
            {
                key.split("|")[0]
                for key in arrays
                if key.count("|") == 2 and key.split("|")[2] == "gap"
                and key.split("|")[1] == name
            }
        )
        if not labels:
            continue

        baselines = [label for label in labels if not _looks_like_rrc(label)]
        rrc_labels = [label for label in labels if _looks_like_rrc(label)]
        header = (
            f"\n=== {name} ===\n"
            f"{'label':<44} {'n':>6} {'ref':>10} {'tour':>10} "
            f"{'gap %':>10} {'SEM':>7} {'gap of means':>13}"
        )
        print(header)
        print("-" * len(header))

        ordered = baselines + sorted(
            rrc_labels, key=lambda value: (_base_of(value), _budget_of(value))
        )
        for label in ordered:
            tour = arrays[f"{label}|{name}|tour"]
            gap = arrays[f"{label}|{name}|gap"]
            ref = arrays[f"{label}|{name}|ref"]
            scored = ~np.isnan(gap)
            summary = summarize(tour[scored], gap[scored], ref[scored])
            print(
                f"{label:<44} {summary.n:>6d} {summary.ref_mean:>10.4f} "
                f"{summary.tour_mean:>10.4f} {summary.gap_mean:>10.3f} "
                f"{summary.gap_sem:>7.3f} {summary.gap_of_means:>12.4f}%"
            )

        # Paired comparison against the exact incumbent each RRC run started from.
        for rrc_label in sorted(rrc_labels, key=_budget_of):
            base = _base_of(rrc_label)
            if base not in arrays:
                continue
            base_gap = arrays[f"{base}|{name}|gap"]
            rrc_gap = arrays[f"{rrc_label}|{name}|gap"]
            paired = ~(np.isnan(base_gap) | np.isnan(rrc_gap))
            if not paired.any():
                continue
            improvement = base_gap[paired] - rrc_gap[paired]
            sem = (
                float(improvement.std(ddof=1)) / np.sqrt(improvement.size)
                if improvement.size > 1
                else 0.0
            )
            wins = float((improvement > 0).mean()) * 100.0
            print(
                f"  paired {base} - {rrc_label}: "
                f"{improvement.mean():+7.3f} pp ± {sem:.3f} SEM, "
                f"better on {wins:5.1f}% of instances"
            )

        for base in baselines:
            run_stats = stats.get(f"{name}|{base}")
            if run_stats is None:
                continue
            print(
                f"  {base}: accepted {run_stats.accepted:,}, decoder rows "
                f"{run_stats.decoder_rows:,}, {run_stats.seconds:.1f}s"
            )


def _base_of(rrc_label: str) -> str:
    """Strip the ``_rrc{budget}`` suffix to recover the baseline label."""
    head, _, tail = rrc_label.rpartition("_rrc")
    return head if tail.isdigit() else rrc_label


def evaluate_dataset(
    name: str,
    inst: InstanceSet,
    models: dict[str, torch.nn.Module],
    args: argparse.Namespace,
    cfg: RRCConfig,
) -> tuple[dict[str, np.ndarray], dict[str, RRCStats]]:
    """Run the initial rollout and RRC for every checkpoint on one dataset."""
    total = len(inst)
    budgets = cfg.budgets
    normalize = should_normalize(name)

    arrays: dict[str, np.ndarray] = {}
    stats: dict[str, RRCStats] = {}

    baseline_labels: dict[str, str] = {}
    rrc_labels: dict[str, tuple[str, ...]] = {}
    for label in models:
        baseline, rrc = _label_parts(label, args, budgets)
        baseline_labels[label] = baseline
        rrc_labels[label] = tuple(rrc)
        arrays[f"{baseline}|{name}|tour"] = np.full(total, np.nan)
        arrays[f"{baseline}|{name}|gap"] = np.full(total, np.nan)
        for rrc_label in rrc:
            arrays[f"{rrc_label}|{name}|tour"] = np.full(total, np.nan)
            arrays[f"{rrc_label}|{name}|gap"] = np.full(total, np.nan)
    arrays[f"__ref|{name}"] = np.full(total, np.nan)

    for num_loc, positions in inst.groups_by_n().items():
        # One environment per size; `num_loc` is used by the full-instance start selection.
        env = LEHDEvalCVRPEnv(
            num_loc=int(num_loc), capacity=float(inst.capacity[positions].max())
        )
        for model in models.values():
            model.env = env

        size = args.batch_size or batch_size_for(int(num_loc))
        done = 0
        while done < positions.size:
            chunk = positions[done : done + size]
            records, scale = inst.model_records(chunk, normalize=normalize)
            batch = build_batch(records).to(args.device)
            reference = batch["cost"].double().cpu().numpy()
            arrays[f"__ref|{name}"][chunk] = reference * scale

            for label, model in models.items():
                result = run_rrc(model, env, batch, cfg, positions=chunk)
                baseline = baseline_labels[label]
                stats.setdefault(baseline, RRCStats()).merge(result.stats)
                initial = result.initial_cost
                arrays[f"{baseline}|{name}|tour"][chunk] = initial * scale
                arrays[f"{baseline}|{name}|gap"][chunk] = (
                    (initial - reference) / reference * 100.0
                )
                for index, rrc_label in enumerate(rrc_labels[label]):
                    cost = result.budget_costs[:, index]
                    arrays[f"{rrc_label}|{name}|tour"][chunk] = cost * scale
                    arrays[f"{rrc_label}|{name}|gap"][chunk] = (
                        (cost - reference) / reference * 100.0
                    )
            done += size

    # Every result row shares one reference array, indexed by original position.
    for label in list(baseline_labels.values()) + [
        rrc for labels in rrc_labels.values() for rrc in labels
    ]:
        arrays[f"{label}|{name}|ref"] = arrays[f"__ref|{name}"]
    arrays.pop(f"__ref|{name}")
    return arrays, stats


def main(argv: list[str] | None = None) -> int:
    """Load checkpoints, evaluate, report and save."""
    args = parse_args(argv)
    if args.report_from:
        saved = np.load(args.report_from)
        arrays = {key: saved[key] for key in saved.files}
        datasets = sorted(
            {
                key.split("|")[1]
                for key in arrays
                if key.count("|") == 2 and key.split("|")[2] == "gap"
            }
        )
        # Stats keys are `stats|label|dataset|field`; the report keys them by dataset+label.
        merged: dict[str, RRCStats] = {}
        for key in arrays:
            if not key.startswith("stats|") or key.count("|") != 3:
                continue
            _, label, dataset, field = key.split("|")
            entry = merged.setdefault(f"{dataset}|{label}", RRCStats())
            value = arrays[key].item()
            if field == "accepted":
                entry.accepted = int(value)
            elif field == "decoder_rows":
                entry.decoder_rows = int(value)
            elif field == "seconds":
                entry.seconds = float(value)
        print_report(arrays, merged, datasets)
        return 0

    torch.set_float32_matmul_precision("medium")
    checkpoints = discover_checkpoints()
    if args.checkpoints:
        unknown = [label for label in args.checkpoints if label not in checkpoints]
        if unknown:
            raise SystemExit(f"no checkpoint for {unknown}; found {sorted(checkpoints)}")
        checkpoints = {
            label: path for label, path in checkpoints.items() if label in args.checkpoints
        }
    if not checkpoints:
        raise SystemExit("no checkpoints to evaluate")

    datasets = discover_datasets(args.datasets)
    cfg = RRCConfig(
        budgets=tuple(args.rrc_budgets),
        w_min=args.rrc_w_min,
        w_max=args.rrc_w_max,
        seed=args.rrc_seed,
        init=args.rrc_init,
        num_starts=args.num_starts,
        num_augment=args.num_augment,
        rrc_num_augment=args.rrc_num_augment,
    )

    print(f"checkpoints: {', '.join(sorted(checkpoints))}")
    print(f"datasets:    {', '.join(name for name, _ in datasets)}")
    print(
        f"device:      {args.device}   init {cfg.init} with "
        f"{cfg.num_augment}x{_starts_label(cfg.num_starts)}   "
        f"RRC budgets {list(cfg.budgets)}   reconstruction {cfg.rrc_num_augment}x"
    )

    results: dict[str, np.ndarray] = {}
    stats_by_dataset: dict[str, dict[str, RRCStats]] = {}
    dataset_order: list[str] = []

    for name, path in datasets:
        inst = load_instances(path, limit=args.limit)
        dataset_order.append(name)
        # A throwaway full-instance environment just to construct the modules; each group
        # replaces `model.env` with one whose `num_loc` matches the batch.
        probe = LEHDEvalCVRPEnv(num_loc=int(inst.num_loc[0]), capacity=float(inst.capacity[0]))
        models = {
            label: load_model(label, checkpoint, probe, args.device, args.num_augment)
            for label, checkpoint in checkpoints.items()
        }
        started = time.time()
        arrays, stats = evaluate_dataset(name, inst, models, args, cfg)
        print(f"\n{name}: {len(inst):,} instances in {time.time() - started:.1f}s")
        results.update(arrays)
        stats_by_dataset[name] = stats

    # Save the per-instance arrays and the compute counters, then report from the same dict.
    for name, stats in stats_by_dataset.items():
        for label, run_stats in stats.items():
            for field in ("accepted", "decoder_rows", "seconds"):
                key = f"stats|{label}|{name}|{field}"
                if f"{label}|{name}|gap" in results:
                    results[key] = np.asarray(getattr(run_stats, field))

    flat_stats = {
        f"{name}|{label}": run_stats
        for name, stats in stats_by_dataset.items()
        for label, run_stats in stats.items()
    }
    print_report(results, flat_stats, dataset_order)

    if not args.no_save:
        out = Path(args.out)
        if not out.is_absolute():
            out = project_root() / out
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **results)
        print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
