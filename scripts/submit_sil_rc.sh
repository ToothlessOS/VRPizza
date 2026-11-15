#!/bin/bash

# submit_sil_rc.sh - run the strict and step-matched SIL-RC arms side by side.
# Usage: sbatch scripts/submit_sil_rc.sh
#
# Both arms share the same core (`src/sil_rc/`, `src/models/pomo_sil_rc.py`); they
# differ in schedule. The strict arm follows the reference's commented CVRP100
# entry (1M fixed labels, 1,024 effective batch, 5 x 20 epochs, LR 0.97/epoch).
# The step-matched arm fits the 2,000-decoded-sequences-per-step protocol
# (12,500 fresh labels/cycle, 5 x 30 epochs, constant LR).
#
# Cost notes:
# * The strict arm generates and keeps 1M CVRP100 labels: ~1.2 GB of CPU RAM plus
#   the ~1.2 GB `sil_rc_labels.npz` sidecar, and 100 RRC rounds over every label at
#   each of five cycle boundaries. Its first epoch after warm-up can take hours.
#   For a first run, pass `--labels 100000` (or lower); set STRICT_ARGS below.
# * The step arm's labels are 12,500 per cycle; its cost is the 20,000 training
#   steps plus 5 x 12,500 x 100 window reconstructions.
# * Both arms write their label sidecar into their checkpoint directory; the strict
#   sidecar is the big one. Keep the job's scratch quota in mind.
#
# Smoke-test first (CPU-capable, seconds):
#     .venv/bin/python notebooks/POMO_sil_rc.py --smoke
#     .venv/bin/python notebooks/POMO_sil_rc_step.py --smoke
#     .venv/bin/python scripts/verify_sil_rc.py

#------------------------
# SLURM resource requests
#------------------------
#SBATCH -c 8
#SBATCH --mem-per-cpu=8G        # -> 64 GB; the strict label store is ~1.2 GB plus copies
#SBATCH --job-name=POMO-SIL-RC
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

mkdir -p logs/sil_rc
SIL_RC_LOG_DIR=logs/sil_rc

# The strict arm's default is the reference's 1M-instance configuration. Override
# with e.g. STRICT_ARGS="--labels 100000" for a lighter first run; the run name and
# checkpoint directory change with the argument, so resume with the same value.
STRICT_ARGS="${STRICT_ARGS:-}"

# `.venv/bin/python` rather than `uv run`: concurrent `uv run` invocations contend
# on the environment lock, and `uv sync` above has already built everything.
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    .venv/bin/python notebooks/POMO_sil_rc.py $STRICT_ARGS \
    > "$SIL_RC_LOG_DIR/strict.log" 2>&1 &
STRICT_PID=$!

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
    .venv/bin/python notebooks/POMO_sil_rc_step.py \
    > "$SIL_RC_LOG_DIR/step.log" 2>&1 &
STEP_PID=$!

echo "strict (GPU 0) pid $STRICT_PID -> $SIL_RC_LOG_DIR/strict.log"
echo "step   (GPU 1) pid $STEP_PID   -> $SIL_RC_LOG_DIR/step.log"

# Wait for both, then report: a failure in one must not be masked by the other.
wait "$STRICT_PID"
STRICT_STATUS=$?
wait "$STEP_PID"
STEP_STATUS=$?

echo "strict exit $STRICT_STATUS | step exit $STEP_STATUS"
if [ "$STRICT_STATUS" -ne 0 ] || [ "$STEP_STATUS" -ne 0 ]; then
    echo "At least one training failed; see $SIL_RC_LOG_DIR/."
    exit 1
fi

#------------------------
# Usage
#------------------------
# 1. Smoke-test first (runs on the login node's GPU or CPU):
#        .venv/bin/python notebooks/POMO_sil_rc.py --smoke
#        .venv/bin/python notebooks/POMO_sil_rc_step.py --smoke
# 2. Run this script with: sbatch scripts/submit_sil_rc.sh
# 3. Watch an arm:         tail -f logs/sil_rc/strict.log   (or step.log)
# 4. Check job state:      sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS,ReqMem,Elapsed
# 5. Resume:               pass --resume to either notebook with the same CLI options;
#                          the label sidecar is reloaded, not redrawn.
