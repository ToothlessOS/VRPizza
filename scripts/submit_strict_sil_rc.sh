#!/bin/bash

# submit_strict_sil_rc.sh - run the strict POMO and strict LEHD SIL-RC arms side by side.
# Usage: sbatch scripts/submit_strict_sil_rc.sh
#
# This is the head-to-head version of the two strict entry points:
#
#   GPU 0: notebooks/POMO_sil_rc.py   (POMO AttentionModelPolicy)
#   GPU 1: notebooks/LEHD_sil_rc.py   (reference LEHD light encoder / heavy decoder)
#
# Both use the reference/SIT strict schedule: 100 CVRP20 warm-up epochs x 50,000
# instances, then 5 x 20 SIL epochs over a fixed 1M generated CVRP100 label set,
# 100 RRC rounds per cycle, Adam 1e-4, no weight decay, clip 10, LR 0.97/epoch.
# The data, labels, RRC pass and evaluation set are the same, so the checkpoints
# are paired. Only the backbone and its memory knobs differ:
#
#   POMO: 1,024 micro-batch x 1 accumulation, 1,024 repair batch
#   LEHD: 128 micro-batch x 8 accumulation (still 1,024), 256 repair batch,
#         per-step decoder gradient checkpointing, validation 8 x 1 start
#
# Cost notes:
# * Both strict arms hold a 1M-label store: ~1.2 GB of CPU RAM plus the ~1.2 GB
#   `sil_rc_labels.npz` sidecar each, and 100 RRC rounds over every label at each
#   of five cycle boundaries. The LEHD repair is much slower per window because
#   its decoder re-runs six self-attention layers at every step.
# * For a first run, reduce both label sets, e.g.:
#       POMO_ARGS="--labels 100000" LEHD_ARGS="--labels 100000" \
#           sbatch scripts/submit_strict_sil_rc.sh
#   The run name and checkpoint directory change with `--labels`, so resume with
#   the same value.
# * The LEHD arm can warm-start from the reference supervised checkpoint instead
#   of the RL warm-up:
#       LEHD_ARGS="--policy-checkpoint /path/checkpoint-40.pt"
#   That skips the 100 warm-up epochs, so it is no longer schedule-matched to the
#   POMO arm; use it as a separate diagnostic.
#
# Smoke-test first (CPU-capable, seconds):
#     .venv/bin/python notebooks/POMO_sil_rc.py --smoke
#     .venv/bin/python notebooks/LEHD_sil_rc.py --smoke
#     .venv/bin/python scripts/verify_sil_rc.py
#     .venv/bin/python scripts/verify_lehd_sil_rc.py

#------------------------
# SLURM resource requests
#------------------------
#SBATCH -c 8
#SBATCH --mem-per-cpu=8G        # -> 64 GB; two ~1.2 GB label stores plus copies
#SBATCH --job-name=STRICT-SIL-RC
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:2            # one card per backbone
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

#------------------------
# Environment setup
#------------------------
cd /dkucc/home/sc927/VRPizza
uv sync

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p logs/sil_rc_strict
STRICT_LOG_DIR=logs/sil_rc_strict

# Optional per-arm arguments. Both default to the full reference configuration
# (1M labels); the run name/directory change with the arguments, so resume with
# the same values.
POMO_ARGS="${POMO_ARGS:-}"
LEHD_ARGS="${LEHD_ARGS:-}"

# `.venv/bin/python` rather than `uv run`: concurrent `uv run` invocations contend
# on the environment lock, and `uv sync` above has already built everything.
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    .venv/bin/python notebooks/POMO_sil_rc.py $POMO_ARGS \
    > "$STRICT_LOG_DIR/pomo.log" 2>&1 &
POMO_PID=$!

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
    .venv/bin/python notebooks/LEHD_sil_rc.py $LEHD_ARGS \
    > "$STRICT_LOG_DIR/lehd.log" 2>&1 &
LEHD_PID=$!

echo "POMO strict (GPU 0) pid $POMO_PID -> $STRICT_LOG_DIR/pomo.log"
echo "LEHD strict (GPU 1) pid $LEHD_PID -> $STRICT_LOG_DIR/lehd.log"

# Wait for both, then report: a failure in one must not be masked by the other.
wait "$POMO_PID"
POMO_STATUS=$?
wait "$LEHD_PID"
LEHD_STATUS=$?

echo "POMO strict exit $POMO_STATUS | LEHD strict exit $LEHD_STATUS"
if [ "$POMO_STATUS" -ne 0 ] || [ "$LEHD_STATUS" -ne 0 ]; then
    echo "At least one training failed; see $STRICT_LOG_DIR/."
    exit 1
fi

#------------------------
# Usage
#------------------------
# 1. Smoke-test first (runs on the login node's GPU or CPU):
#        .venv/bin/python notebooks/POMO_sil_rc.py --smoke
#        .venv/bin/python notebooks/LEHD_sil_rc.py --smoke
# 2. Run this script with: sbatch scripts/submit_strict_sil_rc.sh
# 3. Watch an arm:         tail -f logs/sil_rc_strict/pomo.log   (or lehd.log)
# 4. Check job state:      sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS,ReqMem,Elapsed
# 5. Resume:               pass --resume to either notebook with the same CLI options;
#                          the label sidecar is reloaded, not redrawn.
# 6. OOM (LEHD):           lower --sil-micro-batch and raise --sil-accum so the
#                          effective batch stays 1,024; or pass --precision 32-true.
#
# Evaluate the pair on the same instances (the POMO strict run directory starts
# with `sil-rc`, whose discovery label is `sil`; give it an explicit `sil_rc`
# label so the table names it correctly):
#
#   .venv/bin/python notebooks/eval_pomo.py \
#       --checkpoint sil_rc_strict=checkpoints/<pomo-run>/last.ckpt \
#       --checkpoint lehd_sil_rc=checkpoints/<lehd-run>/last.ckpt \
#       --num-starts 1 --num-augment 1 --batch-size 8
