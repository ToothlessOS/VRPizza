#!/bin/bash

# submit_lehd_sil_rc.sh - run the strict and step-matched LEHD-backbone SIL-RC arms.
# Usage: sbatch scripts/submit_lehd_sil_rc.sh
#
# Both arms are the SIL-RC pipeline with the reference LEHD light encoder / heavy
# decoder (`src/models/components/lehd_policy.py`, `src/models/sil_rc_lehd.py`).
# The strict arm follows the reference's CVRP100 comment (1M fixed labels, 1,024
# effective batch, 5 x 20 epochs, LR 0.97/epoch); the step-matched arm fits the
# 2,000-decoded-sequences-per-step protocol (12,500 fresh labels/cycle, 5 x 30
# epochs, constant LR).
#
# Cost notes:
# * The heavy decoder re-runs six self-attention layers at every decode step. The
#   effective batch is unchanged, but the micro-batch is 128 x 8 (strict) and
#   125 x 16 (step), and the decoder stack is gradient-checkpointed by default.
# * The strict arm's 1M labels x 100 RRC rounds x 5 cycles is roughly 80x the step
#   arm's repair and is likely weeks at this decoder. Run the step arm first; pass
#   STRICT_ARGS="--labels 100000" (or lower) for a first strict run.
# * The step arm is 20,000 training steps plus 5 x 12,500 x 100 window
#   reconstructions, and is the recommended first full run.
# * Both arms write their label sidecar into their checkpoint directory.
#
# Smoke-test first (CPU-capable, seconds):
#     .venv/bin/python notebooks/LEHD_sil_rc.py --smoke
#     .venv/bin/python notebooks/LEHD_sil_rc_step.py --smoke
#     .venv/bin/python scripts/verify_lehd_sil_rc.py

#------------------------
# SLURM resource requests
#------------------------
#SBATCH -c 8
#SBATCH --mem-per-cpu=8G        # -> 64 GB; the strict label store is ~1.2 GB plus copies
#SBATCH --job-name=LEHD-SIL-RC
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:2            # one card per arm
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

mkdir -p logs/lehd_sil_rc
LEHD_SIL_RC_LOG_DIR=logs/lehd_sil_rc

# The strict arm's default is the reference's 1M-instance configuration. Override
# with e.g. STRICT_ARGS="--labels 100000" for a lighter first run; the run name and
# checkpoint directory change with the argument, so resume with the same value.
STRICT_ARGS="${STRICT_ARGS:-}"
STEP_ARGS="${STEP_ARGS:-}"

# `.venv/bin/python` rather than `uv run`: concurrent `uv run` invocations contend
# on the environment lock, and `uv sync` above has already built everything.
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    .venv/bin/python notebooks/LEHD_sil_rc.py $STRICT_ARGS \
    > "$LEHD_SIL_RC_LOG_DIR/strict.log" 2>&1 &
STRICT_PID=$!

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
    .venv/bin/python notebooks/LEHD_sil_rc_step.py $STEP_ARGS \
    > "$LEHD_SIL_RC_LOG_DIR/step.log" 2>&1 &
STEP_PID=$!

echo "strict (GPU 0) pid $STRICT_PID -> $LEHD_SIL_RC_LOG_DIR/strict.log"
echo "step   (GPU 1) pid $STEP_PID   -> $LEHD_SIL_RC_LOG_DIR/step.log"

# Wait for both, then report: a failure in one must not be masked by the other.
wait "$STRICT_PID"
STRICT_STATUS=$?
wait "$STEP_PID"
STEP_STATUS=$?

echo "strict exit $STRICT_STATUS | step exit $STEP_STATUS"
if [ "$STRICT_STATUS" -ne 0 ] || [ "$STEP_STATUS" -ne 0 ]; then
    echo "At least one training failed; see $LEHD_SIL_RC_LOG_DIR/."
    exit 1
fi

#------------------------
# Usage
#------------------------
# 1. Smoke-test first (runs on the login node's GPU or CPU):
#        .venv/bin/python notebooks/LEHD_sil_rc.py --smoke
#        .venv/bin/python notebooks/LEHD_sil_rc_step.py --smoke
# 2. Run this script with: sbatch scripts/submit_lehd_sil_rc.sh
# 3. Watch an arm:         tail -f logs/lehd_sil_rc/strict.log   (or step.log)
# 4. Check job state:      sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS,ReqMem,Elapsed
# 5. Resume:               pass --resume to either notebook with the same CLI options;
#                          the label sidecar is reloaded, not redrawn.
# 6. OOM:                  lower --sil-micro-batch and raise --sil-accum so the
#                          effective batch stays 1,024 (strict) or 2,000 (step).
