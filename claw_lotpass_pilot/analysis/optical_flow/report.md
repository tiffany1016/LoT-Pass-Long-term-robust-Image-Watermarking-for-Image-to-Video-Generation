# CLAW optical-flow alignment analysis

Completed videos: 30/30.

Each generated frame was aligned to generated frame 0 with the existing LoT-Pass Torchvision RAFT flow and remap. The decoder heatmap overlay video includes every aligned frame.

| Condition | Clips | Raw mean ACC | RAFT-aligned mean ACC | Mean change |
|---|---:|---:|---:|---:|
| static_control | 15 | 0.9186224489795919 | 0.9290391156462584 | 0.010416666666666652 |
| broad_fluid_motion | 15 | 0.9241496598639457 | 0.9325680272108844 | 0.008418367346938793 |

`per_frame_aligned_ba.csv` includes raw/aligned ACC, flow displacement, source-anchored aligned stability, and encoder/decoder stable-overlap metrics.
