#!/bin/bash

# submit_sil.sh - run both SIL sampling arms side by side, one GPU each.
# Usage: sbatch scripts/submit_sil.sh
#
# Both arms share the same warm-up recipe, generator distribution and SIL training budget; they
# differ only in how the pseudo-expert trajectories are sampled (deterministic beam search vs
# WOR-SBS). Running them concurrently keeps their epoch counts aligned.
#
# Generation runs inside the training process (no Ray, no extra workers), so the host-memory
# problem in submit_sync.sh does not apply: the SIL DataLoader uses zero workers and the record
# pool is a few thousand small NumPy arrays. The GPU is the constraint, and the search is the
# expensive part -- `NUM_INSTANCES_TO_GENERATE`, `BEAM_WIDTH` and `NUM_ROUNDS` in the notebook
# are the knobs to lower first if an epoch takes too long.

#------------------------
# SLURM resource requests
#------------------------
#SBATCH -c 8
#SBATCH --mem-per-cpu=4G        # -> 32 GB; the warm-up generator batches and search states
#SBATCH --job-name=POMO-SIL
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:2            # one card per sampling arm
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Runtime note: at the measured generation rate on the development GPU (~6 min per 1000
# instances for WOR, ~2 min for beam-128) the default 150 SIL epochs are ~15 h for WOR and ~5 h
# for beam. Measure the training GPU first with
#     .venv/bin/python scripts/benchmark_sil_generation.py --search-type wor
# If the partition caps the job below that, lower NUM_INSTANCES_TO_GENERATE to 500 in the
# notebook (both arms together, to keep the comparison matched) rather than raising the wall
# clock.

#------------------------
# Environment setup
#------------------------
cd /dkucc/home/sc927/VRPizza
uv sync

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p logs/sil
SIL_LOG_DIR=logs/sil

# `.venv/bin/python` rather than `uv run`: concurrent `uv run` invocations contend on the
# environment lock, and `uv sync` above has already built everything they need.
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    .venv/bin/python notebooks/POMO_sil.py --search-type beam_search > "$SIL_LOG_DIR/beam.log" 2>&1 &
BEAM_PID=$!

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
    .venv/bin/python notebooks/POMO_sil.py --search-type wor > "$SIL_LOG_DIR/wor.log" 2>&1 &
WOR_PID=$!

echo "beam (GPU 0) pid $BEAM_PID -> $SIL_LOG_DIR/beam.log"
echo "wor  (GPU 1) pid $WOR_PID -> $SIL_LOG_DIR/wor.log"

# Wait for both, then report: a failure in one must not be masked by the other succeeding.
wait "$BEAM_PID"
BEAM_STATUS=$?
wait "$WOR_PID"
WOR_STATUS=$?

echo "beam exit $BEAM_STATUS | wor exit $WOR_STATUS"
if [ "$BEAM_STATUS" -ne 0 ] || [ "$WOR_STATUS" -ne 0 ]; then
    echo "At least one training failed; see $SIL_LOG_DIR/."
    exit 1
fi

#------------------------
# Usage
#------------------------
# 1. Smoke-test first (runs on the login node's GPU or CPU):
#        .venv/bin/python notebooks/POMO_sil.py --search-type wor --smoke
# 2. Run this script with: sbatch scripts/submit_sil.sh
# 3. Watch an arm:         tail -f logs/sil/wor.log   (or beam.log)
# 4. Check job state:      sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS,ReqMem,Elapsed
#
# To compare the arms on identical warm-up weights, run one arm to completion, then start both
# from its `last.ckpt` with `--warmup-checkpoint checkpoints/<run>/last.ckpt` (and point each
# command above at a different RUN_NAME). Otherwise each arm pays for its own 50-epoch RL
# warm-up, which is the main confound between them.
