#!/bin/bash

# submit_lehd.sh - train POMO on LEHD's partial-solution objective.
# Usage: sbatch scripts/submit_lehd.sh
#
# The run is 40 epochs x 999k partials at an effective batch of 1024 (128 x 8 accumulation).
# A partial averages ~52 customers, roughly half a full-trajectory row, so the compute is
# comparable to the existing 20M-row supervised run. Host memory is the first thing that
# breaks, not the GPU: rl4co forces spawn, so each of the 8 DataLoader workers re-imports
# torch and rl4co (~790 MB) before reading anything. 8 x 790 MB + the main process + CUDA
# host context is ~7-8 GB; --mem-per-cpu=4G x 16 gives 64 GB, deliberately comfortable.

#------------------------
# SLURM resource requests
#------------------------
#SBATCH -c 16
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=POMO-LEHD
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

#------------------------
# Environment setup
#------------------------
cd /dkucc/home/sc927/VRPizza
uv sync

# Cap the per-process thread pools; the DataLoader workers would otherwise each start a full
# OpenMP/MKL pool and compete for the 16 cores.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p logs/lehd
PYTHONUNBUFFERED=1 .venv/bin/python notebooks/POMO_sl_lehd.py > logs/lehd/train.log 2>&1

#------------------------
# Usage
#------------------------
# 1. Run this script with: sbatch scripts/submit_lehd.sh
# 2. Watch it:            tail -f logs/lehd/train.log
# 3. After it finishes:   sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS,ReqMem,Elapsed
# 4. Smoke test first:    .venv/bin/python notebooks/POMO_sl_lehd.py --smoke
#
# If the job dies with a CUDA OOM, lower MICRO_BATCH_SIZE in the training script (and keep
# EFFECTIVE_BATCH_SIZE / MICRO_BATCH_SIZE an integer); the one-w-per-effective-batch draw and
# the epoch arithmetic are unaffected. If it dies with a truncated pickle in a worker or an
# OOM-kill at ReqMem, lower DATALOADER_WORKERS, not the model batch.
