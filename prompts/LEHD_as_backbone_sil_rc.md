# Use the LEHD backbone instead of POMO

We have implemented `notebooks/POMO_sil_rc.py` and `notebooks/POMO_sil_rc_step.py`, which replicates the SIL paper based on POMO backbone. One thing to try next step is to use the LEHD backbone instead of the POMO backbone, so we can also compare with the numbers in the original LEHD paper and see whether SIL without labels can reach similar performance with supervised learning (The good news is that we are using exactly the same test dataset downloaded from the LEHD repo, which makes the results directly comparable). => `notebooks/LEHD_sil_rc.py` and `notebooks/LEHD_sil_rc_step.py`, change the backbone model only, the rest stays the same.

## Instruction

1. Investigate how to swap the backbone model (based on my quick check, the call stack is `.venv/lib/python3.13/site-packages/rl4co/models/zoo/pomo/model.py`, which then referenced the `AttentionModelPolicy` in `.venv/lib/python3.13/site-packages/rl4co/models/zoo/am/policy.py`). What we will do is the build the LEHD module that can be swapped into.
2. The LEHD backbone is available in `/home/toothlessos/Projects/nrp/CIAM-Group NCO_code main single_objective-LEHD/CVRP/VRPModel.py`
3. Investigate the dimensions and faithfully port the LEHD model backbone.
4. Create new versions of `notebooks/eval_pomo.py` and `notebooks/eval_rrc.py` for LEHD backbone, if required.

Importantly, check the seed and reproducibility settings.