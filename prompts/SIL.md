# New experiment (2) - Self-improvement learning

A self-improvement training algorithm with Stochastic Beam Search sampling techinique is provided in `/home/toothlessos/Projects/nrp/gumbeldore`. Your task is to review the repo and faithfully port it into the current repo for comparison with other training methods (RL, SL)

Importantly, we will only port the WOR SBS for stage 1 (See `/home/toothlessos/Projects/nrp/gumbeldore/README.md`); We will compare two different sampling methods to generate samples for self-improvement learning - 1. Simple bean search 2. WOR SBS

Based on the results from another paper, it is important to additional include a warm-up stage in order for SIL to reach comparable performance with RL and SL. The `gumbeldore` repo does not give a warm-up stage and we will need to implement one ourselves - first do N epochs on CVRP100 with RL (See: notebooks/POMO_og.py), than switch to SIL.

# Tasks

1. Study the reference SIL repo to faithfully port the SIL pipeline and WOR SBS algorithm
2. Design the implementation, include an additional warm-up stage with RL; SIL should use the same generated dataset with RL in `rl4co`