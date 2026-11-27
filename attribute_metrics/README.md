# Raw Video Attribute Metrics

This standalone evaluator measures six appearance metric families on raw generated videos. Each clip uses decoded frame 0 as its temporal reference, except mean-reference IHD. It does not use aligned videos, masks, optical flow, watermark decoding, or benchmark-specific metadata.

## Usage

Run the commands from the repository root. Use the existing repository environment, or install just the evaluator dependencies in a Python 3.11 environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r attribute_metrics/requirements.txt
```

```bash
python -m attribute_metrics --videos path/to/raw.mp4 --output-dir /tmp/attribute_metrics
```

For multiple clips, prepare a CSV manifest. Paths are resolved relative to the manifest, and clip IDs must be unique:

```csv
clip_id,video_path
case_a/condition_1,../results/case_a/condition_1.mp4
case_b/condition_2,../results/case_b/condition_2.mp4
```

```bash
python -m attribute_metrics --manifest videos.csv --output-dir /tmp/attribute_metrics \
  --metrics ihd color_w1 glcm_texture patch_de00 appearance_coverage lpips \
  --device cpu
```

LPIPS uses the official pretrained AlexNet v0.1 weights. Its first use may need access to the model cache or network to fetch weights. --weights-dir sets the PyTorch cache location. CPU is the default. For GPU execution, first confirm with nvidia-smi that the selected GPU has zero running processes, then set CUDA_VISIBLE_DEVICES to that single GPU index before choosing --device cuda. The CLI also checks for running compute processes before loading the model.

The CLI decodes videos in presentation order with PyAV and converts frames to RGB uint8. It checks fixed frame geometry, records frame indices from zero, and records metric-specific errors without replacing missing results with zero. Per-frame values for frame 0 are identity values where the metric is frame-0-referenced; its video mean excludes frame 0. A single-frame clip has unavailable temporal means.

Outputs are written only to the explicit output directory:

- per_frame_metrics.csv and per_video_metrics.csv
- run_config.json, run_metadata.json, and errors.jsonl
- .clips/ per-clip checkpoints used to resume an identical run

Resume requires the same input content hashes, selected metrics, thresholds, parameters, and dependency versions. Use a new output directory for changed inputs or settings. The evaluator does not discover or run experiment videos implicitly.

## Metrics

The six selectable families are ihd, color_w1, glcm_texture, patch_de00, appearance_coverage, and lpips. See the [metric specification](metric_spec.md) for formulas, parameters, references, and limitations.
