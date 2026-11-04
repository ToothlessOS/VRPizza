# New experiment - LEHD-style RL

The current implementation of RL training for POMO on CVRP uses generated CVRP100 dataset with `rl4co` and gives best greedy rollout performance in all of the different training methods evaluated in this repo (Entry point: `notebooks/POMO_og.py`).

While POMO trained with LEHD-style SL are given sampled subproblems of size [4, |V|] in training. During the inference, it uses randomly re-construction (RRC) to iteratively rebuild the intermediate solution for better performance. With RRC 500, it becomes the best performing model (Entry point: `notebooks/POMO_sl_lehd.py`; Documentation: `docs/LEHD_SL.md`; Original LEHD repo in `/home/toothlessos/Projects/nrp/CIAM-Group NCO_code main single_objective-LEHD`).

Therefore, a natural option to try is keep RL training, but use the sampled subproblem training data in LEHD-style SL - I hope combining the two will give us better performance.

# Tasks

1. Go through the codebase and understand the implementation of the two training methods
2. Theoritically, do we need to re-engineer the rewards? Since subproblems of different length will generate very different reward signals, which can be a big problem.
3. Design the implementation