"""Benchmark SIL pseudo-expert generation (the SIL training bottleneck).

Measures ``src.sil.search.generate_best_solutions`` end to end: wall time and seconds per
instance, how many decoder and environment-step calls (and rows) the search issues, and the GPU
utilization sampled while it runs. This is the harness used to size the search changes; it is not
part of training.

Run it with the project's virtualenv::

    .venv/bin/python scripts/benchmark_sil_generation.py --search-type beam_search
    .venv/bin/python scripts/benchmark_sil_generation.py --search-type wor --instances 64
    .venv/bin/python scripts/benchmark_sil_generation.py --search-type beam_search --profile

The default sizes mirror ``notebooks/POMO_sil.py`` (N=100, capacity 50, beam 128x1 or WOR 32x4,
chunk 32). Generation is seeded, so repeated runs on the same machine are directly comparable;
the model is randomly initialised unless ``--checkpoint`` is passed, because the search cost is
independent of the weights.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import cProfile
import pstats

import torch

from src.envs import SILCVRPEnv
from src.models import POMOSIL
from src.sil import SILSearchConfig
from src.sil.search import generate_best_solutions
from src.sil.trajectory import SILSearchAdapter

DATA_PATH = str(Path(ROOT) / "data/CVRP/vrp100_hgs_train_100w-001.txt")

#: Defaults that mirror the training recipe. `chunk_size` is the adapter's default; 32 is the
#: notebook value today.
DEFAULT_INSTANCES = 32
DEFAULT_CHUNK_SIZE = 32
DEFAULT_BEAM_WIDTH = 128
DEFAULT_NUM_ROUNDS = 1
DEFAULT_SEED = 0

#: Fraction of a second between GPU utilization samples while generation runs.
GPU_SAMPLE_SECONDS = 0.05


class GpuSampler:
    """Sample GPU-SM utilization on a background thread (no-op if NVML is unavailable)."""

    def __init__(self) -> None:
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
        except Exception:  # pragma: no cover - NVML missing on some hosts
            self._handle = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self.samples.append(int(util.gpu))
            except Exception:
                pass
            self._stop.wait(GPU_SAMPLE_SECONDS)

    def __enter__(self) -> "GpuSampler":
        if self._handle is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def summary(self) -> dict:
        if not self.samples:
            return {"gpu_samples": 0}
        values = torch.tensor(self.samples, dtype=torch.float32)
        return {
            "gpu_samples": len(self.samples),
            "gpu_util_mean": float(values.mean()),
            "gpu_util_p50": float(values.median()),
            "gpu_util_max": float(values.max()),
        }


class Counters:
    """Count search calls and rows by wrapping the adapter callbacks and ``env.step``."""

    def __init__(self) -> None:
        self.decoder_calls = 0
        self.decoder_rows = 0
        self.transition_calls = 0
        self.transition_pairs = 0
        self.env_step_calls = 0
        self.env_step_rows = 0

    def patch(self, adapter_cls, env) -> None:
        counters = self
        cls = adapter_cls

        logprob = cls.child_log_probability_fn
        transition = cls.child_transition_fn
        step = env.step

        def counted_logprob(self, states):
            counters.decoder_calls += 1
            counters.decoder_rows += len(states)
            return logprob(self, states)

        def counted_transition(self, pairs):
            counters.transition_calls += 1
            counters.transition_pairs += len(pairs)
            return transition(self, pairs)

        def counted_step(td):
            counters.env_step_calls += 1
            counters.env_step_rows += int(td.shape[0])
            return step(td)

        cls.child_log_probability_fn = counted_logprob
        cls.child_transition_fn = counted_transition
        env.step = counted_step
        self._restore = lambda: (
            setattr(cls, "child_log_probability_fn", logprob),
            setattr(cls, "child_transition_fn", transition),
            setattr(env, "step", step),
        )

    def restore(self) -> None:
        self._restore()

    def summary(self) -> dict:
        def per_call(total: int, calls: int) -> float:
            return round(total / calls, 1) if calls else 0.0

        return {
            "decoder_calls": self.decoder_calls,
            "decoder_rows": self.decoder_rows,
            "decoder_rows_per_call": per_call(self.decoder_rows, self.decoder_calls),
            "transition_calls": self.transition_calls,
            "transition_pairs": self.transition_pairs,
            "env_step_calls": self.env_step_calls,
            "env_step_rows": self.env_step_rows,
            "env_step_rows_per_call": per_call(self.env_step_rows, self.env_step_calls),
        }


def build_model(args, env, device):
    """A POMOSIL whose policy the benchmark searches with."""
    model = POMOSIL(
        env,
        warmup_epochs=0,
        batch_size=4,
        train_data_size=8,
        val_batch_size=4,
        val_data_size=8,
        num_augment=1,
        sil_batch_size=4,
        sil_batches_per_epoch=1,
        num_instances_to_generate=args.instances,
        search_config=SILSearchConfig(
            search_type=args.search_type,
            beam_width=args.beam_width,
            num_rounds=args.num_rounds,
            chunk_size=args.chunk_size,
            seed=args.seed,
        ),
    )
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"checkpoint: {args.checkpoint} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    model.policy.eval().to(device)
    return model


def run_once(model, env, args, device):
    """Run one generation round and return (metrics, wall seconds, counters)."""
    beam_width = args.beam_width
    num_rounds = args.num_rounds
    if args.search_type == "beam_search":
        # Mirrors the notebook: the beam arm gets the same candidate budget in one pass.
        beam_width = args.beam_width * args.num_rounds
        num_rounds = 1

    cfg = SILSearchConfig(
        search_type=args.search_type,
        beam_width=beam_width,
        num_rounds=num_rounds,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )

    counters = Counters()
    counters.patch(SILSearchAdapter, env)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        records, metrics = generate_best_solutions(
            model.policy, env, args.instances, cfg, device=device
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    counters.restore()

    return records, metrics, elapsed, counters.summary()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--search-type", choices=("wor", "beam_search"), default="beam_search")
    parser.add_argument("--instances", type=int, default=DEFAULT_INSTANCES)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--beam-width",
        type=int,
        default=DEFAULT_BEAM_WIDTH,
        help="beam width per round; for beam_search the total budget is this times --num-rounds",
    )
    parser.add_argument("--num-rounds", type=int, default=DEFAULT_NUM_ROUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-loc", type=int, default=100)
    parser.add_argument("--capacity", type=float, default=50.0)
    parser.add_argument("--repeat", type=int, default=1, help="number of timed rounds")
    parser.add_argument("--checkpoint", default=None, help="optional checkpoint for the policy")
    parser.add_argument("--device", default=None, help="cuda/cpu; defaults to cuda if available")
    parser.add_argument("--profile", action="store_true", help="also print a cProfile top list")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)

    env = SILCVRPEnv(
        data_path=DATA_PATH,
        num_loc=args.num_loc,
        capacity=args.capacity,
        val_instances=100,
    )
    model = build_model(args, env, device)

    # Warm up so the timed rounds do not include CUDA context / allocator setup.
    run_once(model, env, args, device)

    results = []
    sampler = GpuSampler()
    with sampler:
        for _ in range(max(1, args.repeat)):
            records, metrics, elapsed, counters = run_once(model, env, args, device)
            results.append(
                {
                    "seconds": round(elapsed, 3),
                    "seconds_per_instance": round(elapsed / args.instances, 4),
                    "gen_mean_cost": metrics["gen_mean_cost"],
                    "gen_best_cost": metrics["gen_best_cost"],
                    **counters,
                }
            )

    best = min(results, key=lambda item: item["seconds"])
    summary = {
        "search_type": args.search_type,
        "instances": args.instances,
        "chunk_size": args.chunk_size,
        "beam_width": args.beam_width,
        "num_rounds": args.num_rounds,
        "device": str(device),
        "runs": results,
        "best_seconds": best["seconds"],
        "best_seconds_per_instance": best["seconds_per_instance"],
        "projected_minutes_per_1000": round(best["seconds_per_instance"] * 1000 / 60, 1),
        **sampler.summary(),
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"{args.search_type}: {args.instances} instances, chunk {args.chunk_size}, "
            f"beam {args.beam_width}x{args.num_rounds}, {device}"
        )
        print(
            f"  {best['seconds']:.2f} s  ({best['seconds_per_instance']:.3f} s/instance, "
            f"~{summary['projected_minutes_per_1000']:.1f} min per 1000)"
        )
        print(
            f"  decoder {best['decoder_calls']} calls / {best['decoder_rows']} rows "
            f"({best['decoder_rows_per_call']}/call); "
            f"transition {best['transition_calls']} calls / {best['transition_pairs']} pairs"
        )
        print(
            f"  env.step {best['env_step_calls']} calls / {best['env_step_rows']} rows "
            f"({best['env_step_rows_per_call']}/call)"
        )
        if "gpu_util_mean" in summary:
            print(
                f"  GPU util mean {summary['gpu_util_mean']:.1f}% "
                f"(p50 {summary['gpu_util_p50']:.0f}%, max {summary['gpu_util_max']:.0f}%, "
                f"{summary['gpu_samples']} samples)"
            )

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        run_once(model, env, args, device)
        profiler.disable()
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(20)


if __name__ == "__main__":
    main()
