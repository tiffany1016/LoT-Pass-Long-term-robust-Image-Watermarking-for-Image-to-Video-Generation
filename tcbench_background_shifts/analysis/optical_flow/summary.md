# TCBENCH supplemental LoT-Pass RAFT alignment

- Completed videos: 36/36; aligned frame decodes: 1764.
- Mean per-video BA excluding detected copy frames: raw=0.9320, aligned=0.9522, mean paired change=+0.0202 across 36 videos.
- Alignment uses generated frame 0 as the fixed reference for every frame; it does not use the source image as the registration reference.
- The raw videos and raw BA analyses remain unchanged. Aligned BA is supplemental and does not replace same-coordinate/raw results.
- Each completed case has `aligned_video.mp4`, `decoder_heatmap_overlay.mp4`, per-frame raw gradient arrays, and an `alignment_record.json`.
- RAFT displacement can be confounded by scene cuts, object movement, illumination changes, occlusion, or flow failure; alignment may introduce resampling artifacts.

## Per-condition mean BA

| Condition | Videos | Raw mean BA | Aligned mean BA | Change |
|---|---:|---:|---:|---:|
| control | 18 | 0.9341 | 0.9524 | +0.0183 |
| transition | 18 | 0.9298 | 0.9520 | +0.0222 |

Per-frame plots show descriptive group means; individual frames are repeated measurements within videos.
