# ChronoMagic optical-flow alignment analysis

Completed videos: 32/32.

Each generated frame was aligned to generated frame 0 with the existing LoT-Pass Torchvision RAFT flow and remap. The decoder heatmap overlay video includes every aligned frame.

| Condition | Clips | Raw mean ACC | RAFT-aligned mean ACC | Mean change |
|---|---:|---:|---:|---:|
| control | 16 | 0.8430325255102041 | 0.8624441964285714 | 0.01941167091836734 |
| transition | 16 | 0.7601243622448979 | 0.785514987244898 | 0.025390624999999993 |

`per_frame_aligned_ba.csv` includes raw/aligned ACC, flow displacement, source-anchored aligned stability, and encoder/decoder stable-overlap metrics.
