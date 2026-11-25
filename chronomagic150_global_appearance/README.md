# ChronoMagic-150 meteorological paired I2V experiment

This experiment adapts the official 150-pair ChronoMagic-Bench metadata into a single-first-frame CogVideoX study of global illumination and weather/visibility transitions. It selects 16 meteorological sources before watermark analysis, then generates a matched Transition and No-transition Control for each source. The analysis reports raw decoding only; it does not run optical-flow alignment or temporal stability.

The official source metadata and the selected 56 reference-video assets are retained under the ignored `dataset/`, `reference_videos/`, and `source_frames/` directories. Outputs are excluded from Git. The Stage A selection report, prompts, approval record, paired runs, plots, and offline HTML dashboard are stored in this experiment directory.

## Environment and device policy

Use the existing project environment through `uv`:

```bash
uv run --python vbench_lotpass_pilot/.venv/bin/python python <script>
```

Before every GPU command, inspect `nvidia-smi`, select a GPU with zero running processes, and pin it with `CUDA_VISIBLE_DEVICES=<physical index>`. Never share or terminate another process. Generation uses the local CogVideoX cache under `vbench_lotpass_pilot/model_cache/huggingface`.

## Current approved workflow

Stage A is complete and approved. The 16 approved inputs and 32 prompt rows are in `dataset/chronomagic150_selected16.csv` and `dataset/chronomagic150_generation_manifest.csv`.

1. `scripts/record_approval.py` records approval against the manifest hash.
2. `scripts/encode_sources.py` encodes each of the 16 sources once and records one 32-bit message reused by both conditions.
3. `scripts/generate_videos.py` resumes valid completed videos and generates the paired 49-frame clips with seed 42, 720×480, 8 FPS, 50 steps, and guidance 5.
4. Once all 32 generations are complete, run `scripts/run_analysis_pipeline.py`. It checks for a fully free GPU before each GPU stage, then runs raw decoding, CPU source-anchored stability and encoder heatmaps, GPU decoder heatmaps, and quantitative summaries.
5. Inspect `analysis/transition_review/*.jpg`, update `analysis/transition_validation.csv` with valid/partial/invalid transition and control labels plus camera-motion notes, then rerun `scripts/summarize_analysis.py` and `scripts/build_dashboard.py`.

The HTML dashboard is written as `chronomagic150_dashboard.html` at repository root and `analysis/dashboard.html`. It embeds raw clips plus full-frame source-anchored similarity, stable-mask, and per-frame decoder-heatmap overlays.

## Analysis definitions

- LoT-Pass bit accuracy is raw per-frame decoder accuracy; no optical flow is applied.
- Stability compares each generated frame to its own watermarked source at identical coordinates, using the inherited 16×16 MAE/SSIM implementation and its fixed thresholds.
- Stable-region overlays use alpha 0.35. Decoder attribution is the existing gradient method and should not be interpreted as the physical location of watermark bits.
- Statistical summaries treat each source image as one independent pair. Frame-level curves describe within-video trajectories, not independent observations.
- Transition/control semantic validation includes luminance and changed-area diagnostics plus manual visual review. Report partial or invalid prompt compliance rather than dropping cases.
