#!/bin/bash

# submit_rrc.sh - score checkpoints with Random Re-Construct on the CVRP test sets.
# Usage: sbatch scripts/submit_rrc.sh
#
# This is a template: edit CHECKPOINTS / DATASETS / LIMIT / RRC_ARGS below. The default is the
# primary experiment -- every discovered checkpoint on vrp100, RRC budgets 50/100/300/500/1000,
# starting from the matched multistart+8x incumbent. Run with a small --limit first; budget
# 1000 on 10,000 instances is the expensive part (the paper reports hours for it).
#
# The budget curve is nested: one pass to the largest budget snapshots the incumbent at each
# requested budget, so 50/100/300/500/1000 cost one 1000-iteration run rather than five.

#------------------------
# SLURM resource requests
#------------------------
#SBATCH -c 8
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=POMO-RRC
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

#------------------------
# Environment setup
#------------------------
cd /dkucc/home/sc927/VRPizza
uv sync

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p logs/rrc

CHECKPOINTS=""                 # e.g. "--checkpoints lehd" ; empty = every discovered checkpoint
DATASETS="--datasets vrp100_test_lkh"
LIMIT="--limit 128"            # remove for the full 10k; keep for the first pass

# The primary arm: POMO multistart (N starts, the default) x 8 dihedral, RRC reconstruction 1x.
# Omit --num-starts so every size group uses its own N (100 here, 200/500/1000 if added).
RRC_ARGS="--num-augment 8 --rrc-num-augment 1 --rrc-budgets 50 100 300 500 1000"
# Other arms to run into separate files, e.g.:
#   --num-augment 1                                  # no dihedral
#   --num-starts 1   --num-augment 8                 # no multistart
#   --rrc-init greedy                                # LEHD protocol
#   --rrc-num-augment 8                              # augmentation inside reconstruction

PYTHONUNBUFFERED=1 .venv/bin/python notebooks/eval_rrc.py \
    $CHECKPOINTS $DATASETS $LIMIT $RRC_ARGS \
    --out logs/rrc/results.npz > logs/rrc/run.log 2>&1

#------------------------
# Usage
#------------------------
# 1. sbatch scripts/submit_rrc.sh
# 2. tail -f logs/rrc/run.log
# 3. Re-print a saved result without inference:
#      .venv/bin/python notebooks/eval_rrc.py --report-from logs/rrc/results.npz
#
# The report pairs every RRC budget row with the exact incumbent it started from, so read the
# "paired ... better on N%" lines rather than the unpaired gap columns when comparing arms.
