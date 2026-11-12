#!/bin/bash

# submit_comparison.sh - run the four step-matched POMO arms, one per GPU.
# Usage: sbatch scripts/submit_comparison.sh
#
# The arms decode 2,000 sequences per optimizer step, run 100 steps per epoch and 200 epochs,
# and draw from their natural data source (RL generated; SL/LEHD labeled). See
# docs/ComparisonProtocol.md. The shared-baseline LEHD arm costs 4x and is a fifth run; it is
# left commented out because this job already uses all four cards.

#------------------------
# SLURM resource requests
#------------------------
#SBATCH -c 32                   # 4 DataLoader workers per training, plus four main processes
#SBATCH --mem-per-cpu=4G        # -> 128 GB. 1G is not enough: spawn costs ~790 MB/worker.
#SBATCH --job-name=POMO-STEP
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

#------------------------
# Environment setup
#------------------------
cd /dkucc/home/sc927/VRPizza
uv sync

# Cap the per-process thread pools. Each of the 16 workers would otherwise start a full
# OpenMP/MKL pool, and 20 processes x 8 threads is ~160 threads competing for 32 cores.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p logs/comparison
LOG_DIR=logs/comparison

# The four arms are independent; `CUDA_VISIBLE_DEVICES` pins one card each and both scripts ask
# Lightning for `devices=1`. `.venv/bin/python` rather than `uv run`: concurrent `uv run`
# invocations contend on the environment lock, and `uv sync` above has already built it.
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    .venv/bin/python notebooks/POMO_rl_step.py > "$LOG_DIR/rl.log" 2>&1 &
RL_PID=$!

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
    .venv/bin/python notebooks/POMO_sl_step.py > "$LOG_DIR/sl.log" 2>&1 &
SL_PID=$!

CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 \
    .venv/bin/python notebooks/POMO_sl_lehd_step.py > "$LOG_DIR/lehd.log" 2>&1 &
LEHD_PID=$!

CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 \
    .venv/bin/python notebooks/POMO_rl_lehd_step.py --baseline teacher \
    > "$LOG_DIR/lehd_rl_teacher.log" 2>&1 &
LEHD_RL_PID=$!

echo "RL (GPU 0) pid $RL_PID -> $LOG_DIR/rl.log"
echo "SL (GPU 1) pid $SL_PID -> $LOG_DIR/sl.log"
echo "LEHD (GPU 2) pid $LEHD_PID -> $LOG_DIR/lehd.log"
echo "LEHD-RL teacher (GPU 3) pid $LEHD_RL_PID -> $LOG_DIR/lehd_rl_teacher.log"

# Wait for all four; a failure in one must not be masked by the others succeeding.
wait "$RL_PID"; RL_STATUS=$?
wait "$SL_PID"; SL_STATUS=$?
wait "$LEHD_PID"; LEHD_STATUS=$?
wait "$LEHD_RL_PID"; LEHD_RL_STATUS=$?

echo "RL $RL_STATUS | SL $SL_STATUS | LEHD $LEHD_STATUS | LEHD-RL $LEHD_RL_STATUS"
if [ "$RL_STATUS" -ne 0 ] || [ "$SL_STATUS" -ne 0 ] || \
   [ "$LEHD_STATUS" -ne 0 ] || [ "$LEHD_RL_STATUS" -ne 0 ]; then
    echo "At least one training failed; see $LOG_DIR/."
    exit 1
fi

#------------------------
# The shared-baseline arm (4x cost), second job
#------------------------
# Run this after the teacher job, on any free card:
#
#   CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
#       .venv/bin/python notebooks/POMO_rl_lehd_step.py --baseline shared \
#       > logs/comparison/lehd_rl_shared.log 2>&1
#
# It uses 500 partials x K=4 = 2,000 rollouts per step, not the natural 16 x K=4 = 8,000.

#------------------------
# Usage
#------------------------
# 1. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Submit: sbatch scripts/submit_comparison.sh
# 3. Watch: tail -f logs/comparison/rl.log (or sl.log, lehd.log, lehd_rl_teacher.log)
# 4. Queue: squeue --me
# 5. Account: sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS,ReqMem,Elapsed
# 6. Preflight: .venv/bin/python scripts/verify_comparison.py
#
# HOST MEMORY: spawn re-imports torch and rl4co (~790 MB) per worker. Four trainings x four
# workers is ~12.6 GB of workers; the request above is deliberately comfortable. If a worker
# dies with `_pickle.UnpicklingError: pickle data was truncated`, it is an OOM kill, not a
# pickling bug -- lower DATALOADER_WORKERS or raise --mem-per-cpu.
#
# GPU MEMORY: SL/LEHD run 500-row micro-batches x 4 accumulation; 2,000 rows in one forward is
# ~41 GiB at N=100. If an LEHD chunk near w=100 OOMs, use 250 x 8 (same effective step) rather
# than changing the step budget.
