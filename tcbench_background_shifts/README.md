# TC-Bench-derived background-shifts experiment

This study covers every official TC-Bench `background shifts` entry (18 IDs), using first-frame CogVideoX-I2V with the existing LoT-Pass pipeline. It is not a reproduction of TC-Bench’s endpoint-conditioned protocol.

Stage A pinned the official TC-Bench repository and metadata, retrieved all 18 first frames, recorded their provenance and hashes, and prepared 18 custom controls alongside the verbatim official transition prompts. The full preview was approved by the user before Stage B.

Stage B is complete: LoT-Pass encoded each unique source once; 36 paired I2V clips were generated with seed 42; all 1,764 frames were raw-decoded and compared against their watermarked source; source-anchored stability metrics/overlays were produced; decoder attribution was recomputed for every frame with one `decoder_heatmap_overlay.mp4` per clip; and all 36 clips were visually screened using five sampled frames per clip. Detailed coverage and limitations are in `reports/all_18_background_shifts_summary.md`.

A user-authorized supplemental RAFT analysis is in `analysis/optical_flow/`: all 36 videos are aligned to generated frame 0, re-decoded frame by frame, and have aligned videos plus post-alignment decoder heatmap overlay videos. The original raw results are preserved.

The existing VBench++ uv environment is reused at `../vbench_lotpass_pilot/.venv`. Results, downloaded references, generation files, and analysis outputs are ignored by git and remain local. No source or results data were pushed.
