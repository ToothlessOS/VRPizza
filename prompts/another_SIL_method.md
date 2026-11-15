# Another SIL method:

We will implement a different SIL training pipeline from the following paper on POMO backbone:

https://arxiv.org/abs/2403.19561 - Self-Improved Learning for Scalable Neural Combinatorial Optimization

It's quite clever to use a local reconstruction process like LEHD's RRC to improve the quality of sampled solutions - compared with the SBS in the previous work, this remove the bottleneck to sample / improve the solutions on the CPU. Also based on the experiment results with LEHD's RRC, it can be more effective in constructing better solutions.

## Keypoints

1. Self-improvement learning
2. A warm-up training stage with RL and on CVRP20
3. Initial solution with random insertion or the model's greedy rollout
4. A process similar to LEHD's RRC for - local reconstruction to generate better samples for self-improvement learning

## Code reference

I have cloned the original repo, available at `/home/toothlessos/Projects/nrp/SIL/`

## Implementation plans

1. Review the official implementation provided.
2. Plan the integration into the current evaluation repo. Two versions - one that strictly port the original implementation, one with synced training (see `docs/ComparisonProtocol.md`). The local reconstruction may reuse / reference components from the LEHD implementation, check that out.
3. Confirm the training setup & parameters from the official implementation and created a table (both the warm-up stage and SIL stage) - I will cross check that with the appendix of the paper.
4. Do the port - note that we are training on CVRP100 currently (the original paper work on much bigger instances - but the POMO backbone may not support that properly)