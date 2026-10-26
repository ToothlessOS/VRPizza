The `POMO_sl` implemented in this repo performs worse than the original RL. However, the training data sampling in my version of SL here is different from the widely adopted version of SL from the LEHD paper. Therefore, we need to replicate LEHD's SL faithfully here (with POMO model and CVRP tasks).

The original LEHD code can be found at `/home/toothlessos/Projects/nrp/CIAM-Group NCO_code main single_objective-LEHD`. The most important difference here, instead of directly training on, say CVRP100 instances. LEHD samples partial solutions of length $w$ from the discrete uniform distribution Unif([4, |V|]) (here, |V| = 100).

LEHD also have an Random Re-Construct for inference - where it sample a partial solution and re-optimize it interactively. Please also implement that - we will need it to compare the results on eval datasets after training (the original paper uses 50, 100, 300, 500, 1000 runs of RRC in their tables).

Basically, we are keeping the POMO backbone but using the training and inference strategies of LEHD, one experiment that the authors of LEHD didn't try (they simply train POMO on TSP100). If possible, keep the 8x dihedral for POMO - I would be interested to see the results with / without it.

