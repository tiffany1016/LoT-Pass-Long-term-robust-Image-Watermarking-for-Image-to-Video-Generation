# CLAW LoT-Pass × CogVideoX analysis

Scheduled cases: 30 (15 images × 2 prompts). Raw-decoded: 30.
All clip-level summaries use 49 generated frames when available. A frame-zero copy-like frame is excluded from mean ACC only when detected using the recorded MAE/SSIM rule.

## Prompt-condition summary

| Condition | Completed | Raw mean ACC | Aligned mean ACC | Stable-region ratio | Largest stable region | Decoder stable overlap |
|---|---:|---:|---:|---:|---:|---:|
| static_control | 15 | 0.9186 | 0.9290 | 0.1548 | 0.1025 | 0.2078 |
| broad_fluid_motion | 15 | 0.9241 | 0.9326 | 0.2236 | 0.1614 | 0.2181 |

## Estimated coverage relationships

Coverage correlations average the two prompt-condition outcomes within each image first; n is therefore at most 15 source images.
| Outcome | n | Pearson r (p) | Spearman ρ (p) |
|---|---:|---:|---:|
| mean_raw_ba | 15 | -0.430 (0.109) | -0.464 (0.0817) |
| mean_aligned_ba | 15 | -0.366 (0.18) | -0.517 (0.0482) |
| stable_region_ratio | 15 | -0.534 (0.0402) | -0.474 (0.0742) |
| largest_stable_region_ratio | 15 | -0.508 (0.0532) | -0.496 (0.0603) |
| decoder_stable_overlap | 15 | -0.321 (0.243) | -0.318 (0.247) |
| encoder_stable_overlap | 15 | -0.522 (0.0461) | -0.546 (0.0354) |

## Interpretation notes

This is an exploratory, paired 15-image pilot; coverage estimates are visual pre-generation judgments, not measured motion ground truth. CLAW v1 is curated for transparent/semi-transparent liquid scenes. Interpret correlations descriptively, not as causal evidence.

Raw and RAFT-aligned ACC are reported separately. RAFT alignment can resample or miswarp frames; it is a diagnostic view, not a replacement for raw decoding.

See `per_case_metrics.csv`, `paired_condition_deltas.csv`, `dynamic_coverage_correlations.csv`, and `optical_flow/` for case-level data and full-frame aligned/heatmap videos.
