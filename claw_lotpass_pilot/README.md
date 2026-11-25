# CLAW LoT-Pass pilot

This experiment uses the official CLAW v1 test set. It draws 15 source images uniformly without replacement from the 126 `*_input.jpg` test images using seed 42. Each selected image has a frozen-scene control prompt and a broad-fluid-motion prompt. The second condition keeps solid structures and the camera fixed.

## Results

The approved pilot is complete: 15 source images, two paired prompts per image, 30 generated videos, raw and RAFT-aligned decoding, stable-region analysis, and full-frame decoder-heatmap overlays. The combined report is in `analysis/summary/final_report.md`.

Open `claw_lotpass_dashboard.html` from the repository root for the interactive results. It includes the condition summary, ACC curves, coverage analyses, and a per-image explorer. Each clip can be viewed as raw video, RAFT-aligned video, or aligned video with the decoder heatmap overlaid. The MP4 previews are embedded so the dashboard works offline. To rebuild it after changing results, run:

```bash
uv run --python vbench_lotpass_pilot/.venv/bin/python python claw_lotpass_pilot/scripts/build_dashboard.py
```

The builder writes the standalone dashboard to the repository root and a copy to `analysis/dashboard.html`.

## Stage A review package

The original approval package is in `selection/approval_request.md`. It includes the sample contact sheet, image-specific visual estimates, and all 30 prompts. The sample and prompts were approved before encoding and video generation.

To reproduce Stage A from the downloaded archive, run from the repository root:

```bash
uv run --python vbench_lotpass_pilot/.venv/bin/python python claw_lotpass_pilot/scripts/prepare_stage_a.py
uv run --python vbench_lotpass_pilot/.venv/bin/python python claw_lotpass_pilot/scripts/draft_stage_a_review.py
```

The data archive, extracted source images, approval package, runs, and analysis are excluded from Git. Keep the official archive at `downloads/CLAW.zip`.

## Approval gate

Do not encode images, launch CogVideoX, or run GPU analysis until the user approves the sample and prompts in `selection/approval_request.md`. After approval, use one LoT-Pass encoding per image for both prompt conditions and record all generation and analysis settings.

## Experiment stages

The user approved the Stage A package. The completed pipeline is:

```bash
uv run --python vbench_lotpass_pilot/.venv/bin/python python claw_lotpass_pilot/scripts/prepare_generation.py --user-approved
CUDA_VISIBLE_DEVICES=0 uv run --python vbench_lotpass_pilot/.venv/bin/python python claw_lotpass_pilot/scripts/encode_sources.py
CUDA_VISIBLE_DEVICES=0 HF_HOME=vbench_lotpass_pilot/model_cache/huggingface HF_HUB_OFFLINE=1 uv run --python vbench_lotpass_pilot/.venv/bin/python python claw_lotpass_pilot/scripts/generate_videos.py
CUDA_VISIBLE_DEVICES=0 uv run --python vbench_lotpass_pilot/.venv/bin/python python claw_lotpass_pilot/scripts/decode_raw.py
uv run --python vbench_lotpass_pilot/.venv/bin/python python claw_lotpass_pilot/scripts/stable_regions.py
uv run --python vbench_lotpass_pilot/.venv/bin/python python encoder_heatmap.py --pair claw_lotpass_pilot/images/canonical_clean=claw_lotpass_pilot/images/watermarked --out-dir claw_lotpass_pilot/analysis/heatmaps/encoder
CUDA_VISIBLE_DEVICES=0 TORCH_HOME=vbench_lotpass_pilot/model_cache/torch uv run --python vbench_lotpass_pilot/.venv/bin/python python claw_lotpass_pilot/scripts/align_optical_flow.py
uv run --python vbench_lotpass_pilot/.venv/bin/python python claw_lotpass_pilot/scripts/summarize_analysis.py
```

Before each CUDA command, check `nvidia-smi`, choose a GPU with no running processes, and set `CUDA_VISIBLE_DEVICES` to its physical index. The recorded run used GPU 0. Full-frame RAFT-aligned videos and decoder-heatmap overlays are in `analysis/optical_flow/<case_id>/`; raw ACC, stability videos, and the combined report are in `analysis/`.
