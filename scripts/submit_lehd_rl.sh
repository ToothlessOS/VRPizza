#!/bin/bash

# submit_lehd_rl.sh - train POMO by REINFORCE on LEHD partial solutions.
# Usage: sbatch scripts/submit_lehd_rl.sh
#
# The default is the teacher-baseline arm: one sampled rollout per LEHD partial, baseline =
# the HGS window's reduced cost. It is compute-comparable to notebooks/POMO_sl_lehd.py
# (40 epochs x 999k partials at effective batch 1024 = 128 x 8 accumulation).
#
# The label-free arm needs K >= 2 sampled rollouts per partial and costs K times as much.
# For a pilot, use K=2; for the full run, K=4 is the default in the script:
#
#   .venv/bin/python notebooks/POMO_rl_lehd.py --baseline shared --num-samples 4
#
# Host memory is the first thing that breaks, not the GPU: rl4co forces spawn, so each of the
# 8 DataLoader workers re-imports torch and rl4co (~790 MB) before reading anything. If the
# job dies with a truncated pickle in a worker or an OOM-kill at ReqMem, lower
# DATALOADER_WORKERS in the training script, not the model batch.

#------------------------
# SLURM resource requests
#------------------------
#SBATCH -c 16
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=POMO-LEHD-RL
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
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

mkdir -p logs/lehd_rl
PYTHONUNBUFFERED=1 .venv/bin/python notebooks/POMO_rl_lehd.py > logs/lehd_rl/teacher.log 2>&1

# Shared-baseline arm (label-free signal), K=4. Uncomment to run it in the same job after the
# teacher arm; budget the wall clock for both.
# PYTHONUNBUFFERED=1 .venv/bin/python notebooks/POMO_rl_lehd.py \
#     --baseline shared --num-samples 4 > logs/lehd_rl/shared.log 2>&1

#------------------------
# Usage
#------------------------
# 1. Run this script with: sbatch scripts/submit_lehd_rl.sh
# 2. Watch it:            tail -f logs/lehd_rl/teacher.log
# 3. After it finishes:   sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS,ReqMem,Elapsed
# 4. Smoke test first:    .venv/bin/python notebooks/POMO_rl_lehd.py --smoke
#                         .venv/bin/python notebooks/POMO_rl_lehd.py --smoke --baseline shared
#
# Checkpoints land under checkpoints/lehd_rl-... or checkpoints/lehd_rl_shared-...; the prefix
# before the first "-" is the label eval_pomo.py/eval_rrc.py use.
