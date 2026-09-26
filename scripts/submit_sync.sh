#!/bin/bash

# submit_sync.sh - run the synced SL and RL trainings side by side, one GPU each.
# Usage: sbatch scripts/submit_sync.sh

#------------------------
# SLURM resource requests
#------------------------
#SBATCH -c 16                   # 8 DataLoader workers per training, plus the two main processes
#SBATCH --mem-per-cpu=1G        # (1G) for each CPU core
#SBATCH --job-name=POMO-SYNC
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:2            # one card per training; DKUCC allows max 4 per job
#SBATCH --time=1-12:00:00
#SBATCH --output=%x-%j.out      # e.g., (location of the script)/POMO-SYNC-<JOBID>.out
#SBATCH --error=%x-%j.err       # separate error file

#------------------------
# Environment setup
#------------------------
cd /dkucc/home/sc927/VRPizza
uv sync

#------------------------
# Run both trainings at once, one per GPU
#------------------------
# The two are meant to be read against each other -- they draw the identical (instance, start)
# pairs every epoch and differ only in the objective -- so running them concurrently keeps
# their epoch counts aligned and halves the wall clock.
#
# `CUDA_VISIBLE_DEVICES` is what does the pinning. Both scripts ask Lightning for
# `devices=1`, so each sees exactly one card and neither needs to know about the other.
#
# `.venv/bin/python` rather than `uv run`: two concurrent `uv run` invocations contend on the
# environment lock, and the sync above has already built everything they need.
#
# Per-training logs rather than the SLURM stdout, which would interleave the two. They land in
# `logs/`, which is git-ignored.
mkdir -p logs/sync
SYNC_LOG_DIR=logs/sync

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    .venv/bin/python notebooks/POMO_sl_synced.py > "$SYNC_LOG_DIR/sl-synced.log" 2>&1 &
SL_PID=$!

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
    .venv/bin/python notebooks/POMO_rl_synced.py > "$SYNC_LOG_DIR/rl-synced.log" 2>&1 &
RL_PID=$!

echo "SL (GPU 0) pid $SL_PID -> $SYNC_LOG_DIR/sl-synced.log"
echo "RL (GPU 1) pid $RL_PID -> $SYNC_LOG_DIR/rl-synced.log"

# Wait for both, then report: a failure in one must not be masked by the other succeeding.
wait "$SL_PID"
SL_STATUS=$?
wait "$RL_PID"
RL_STATUS=$?

echo "SL exit $SL_STATUS | RL exit $RL_STATUS"
if [ "$SL_STATUS" -ne 0 ] || [ "$RL_STATUS" -ne 0 ]; then
    echo "At least one training failed; see $SYNC_LOG_DIR/."
    exit 1
fi

#------------------------
# Usage
#------------------------
# 1. Install uv if you haven't already: curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run this script with: sbatch scripts/submit_sync.sh
# 3. Watch either training: tail -f logs/sync/sl-synced.log (or rl-synced.log)
# 4. View queue status with: squeue --me
# 5. Show job details with: scontrol show job <JOBID>
# 6. Show detailed status with: sacct -j <JOBID>
# 7. Cancel job if needed: scancel <JOBID>   # kills both trainings
#
#------------------------
# GPU-related Usage
#------------------------
# SPECS: A40 48G; CUDA 13.1; MAX 4 CARDS PER JOB
# Can install the latest PyTorch 2.7.0 with CUDA 12.8
# Add the following lines to pyproject.toml for installation:
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu128" }]
# torchvision = [{ index = "pytorch-cu128" }]
# torchaudio = [{ index = "pytorch-cu128" }]  # Include if needed
# Then:
# uv add torch torchvision torchaudio
#
# NOTE ON MEMORY: the supervised run batches 1024 rows per step, roughly 4x what
# notebooks/POMO_sl.py was tuned around (2.65 GiB at batch 128, measured on a 7.62 GiB card).
# On an A40's 48 GiB that is comfortable, but it is the first thing to check if this job dies
# with a CUDA OOM. The fallback is to lower INSTANCES_PER_BATCH from 64 to 16 in *both*
# scripts -- which keeps the supervised batch at its original 256 rows and shrinks only the
# RL batch -- and raise MAX_EPOCHS to compensate. The two must be changed together; see
# docs/RL_SL_data_pipeline.md.
