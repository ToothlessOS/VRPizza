#!/bin/bash 

# submit_gpu.sh - submit GPU workload to SLURM cluster (DKUCC)
# Usage: sbatch submit.sh

#------------------------
# SLURM resource requests
#------------------------
#SBATCH -c 8                    # number of CPU cores (threads) to use
#SBATCH --mem-per-cpu=1G        # (1G) for each CPU core
#SBATCH --job-name=POMO-SL
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:1            # request 1 GPU (adjust as needed, max 4 per job on DKUCC)
#SBATCH --time=20:00:00            
#SBATCH --output=%x-%j.out      # e.g., (location of the script)/sc927-lkh3-<JOBID>.out
#SBATCH --error=%x-%j.err       # separate error file: (location of the script)/sc927-lkh3-<JOBID>.out

#------------------------
# Environment setup
#------------------------
# conda needs to 

cd /dkucc/home/sc927/VRPizza
uv sync

#------------------------
# Run code
#------------------------
 uv run notebooks/POMO_sl.py

#------------------------
# Usage
#------------------------
# 1. Install uv if you haven't already: curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run this script with: sbatch submit_gpu.sh
# 3. View queue status with: squeue --me
# 4. Show job details with: scontrol show job <JOBID>
# 5. Show detailed status with: sacct -j <JOBID>
# 6. View output and error files: cat sc927-lkh3-<JOBID>.out, cat sc927-lkh3-<JOBID>.err
# 7. Cancel job if needed: scancel <JOBID>

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