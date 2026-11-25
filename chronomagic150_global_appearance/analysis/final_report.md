# ChronoMagic-150 LoT-Pass paired I2V analysis

Generated cases: 32 (16 source images × 2 conditions); completed decode cases: 32.
The independent unit is the source image/pair (n ≤ 16); frames are not treated as independent observations.
All bit-accuracy figures are raw decoder results with no optical-flow alignment.

## Paired raw bit accuracy

Completed pairs: 16. Mean transition − control video BA: -0.0829; median: -0.0517; transition lower in 16/16 pairs.
See `condition_summary.csv`, `paired_comparison.csv`, `paired_frame_trajectories.csv`, and the matching plots.

## Source-anchored stability and attribution

Stability compares each generated frame to the same-coordinate watermarked source using the inherited 16×16 MAE/SSIM patch implementation; temporal stability was not computed.
Encoder maps use the repository's existing residual-magnitude method. Decoder maps use the existing signed-message input-gradient attribution method, evaluated for each frame.
Pairwise association between change in raw BA and changes in SRR/largest stable region is in `paired_change_correlations.csv`. Per-frame overlap and enrichment are in `heatmaps/decoder/per_frame_overlap.csv`.

## Transition compliance

Manual validation status values currently present: `pending_manual_visual_review`. Review contact sheets under `transition_review/` and update `transition_validation.csv` before treating prompt compliance as confirmed.
Luminance and changed-area measures are descriptive diagnostics; they do not alone establish semantic transition success.

## Interpretation

This is a paired 16-source pilot adapted from ChronoMagic-150 to single-first-frame CogVideoX I2V. Transition and control prompts are matched but remain stochastic generation branches. Results support descriptive within-source comparisons, not strong causal attribution.
