The RL-trained POMO has better performance that SL-trained POMO based on current results. However, the two training pipelines use different dataloader and data augmentation strategies, so I cannot determine whether the gap comes from the data or the training.

Therefore, I need you to create a NEW pipeline for RL that syncs data pipeline with SL, specifically, it should:

1. Use the same dataset with SL (but of course without labels)
2. Instead of the default N multi-start (where N is problem size), only use the same starting starting nodes that are feasible based on the supervised label.

These are not standard practices, but they control the data variables so the SL and RL results can be comparable.

If you have any question, make sure you ask me for clarification.

Design choices:
K = 16 starts, resampled per batch