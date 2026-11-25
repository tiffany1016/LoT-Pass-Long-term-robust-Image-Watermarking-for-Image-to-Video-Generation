# VBENCH supplemental LoT-Pass RAFT alignment

- Completed videos: 40/40; aligned frame decodes: 1960.
- Mean per-video BA excluding detected copy frames: raw=0.8304, aligned=0.9579, mean paired change=+0.1275 across 40 videos.
- Alignment uses generated frame 0 as the fixed reference for every frame; it does not use the source image as the registration reference.
- The raw videos and raw BA analyses remain unchanged. Aligned BA is supplemental and does not replace same-coordinate/raw results.
- Each completed case has `aligned_video.mp4`, `decoder_heatmap_overlay.mp4`, per-frame raw gradient arrays, and an `alignment_record.json`.
- RAFT displacement can be confounded by scene cuts, object movement, illumination changes, occlusion, or flow failure; alignment may introduce resampling artifacts.

## Per-condition mean BA

| Condition | Videos | Raw mean BA | Aligned mean BA | Change |
|---|---:|---:|---:|---:|
| pan | 10 | 0.8179 | 0.9550 | +0.1371 |
| static | 10 | 0.9173 | 0.9588 | +0.0415 |
| tilt | 10 | 0.7953 | 0.9575 | +0.1622 |
| zoom | 10 | 0.7911 | 0.9603 | +0.1692 |

Per-frame plots show descriptive group means; individual frames are repeated measurements within videos.
