"""Command-line evaluator for raw generated videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import av
import numpy as np

from .metrics import (
    DEFAULT_COVERAGE_THRESHOLDS,
    GLCM_ANGLES,
    GLCM_PROPERTIES,
    METRIC_FAMILIES,
    appearance_coverage,
    color_histogram_wasserstein,
    cosine_similarity,
    glcm_features,
    ihd_scores,
    illumination_histogram,
    ihd_working_size,
    lab_patch_means,
    patch_delta_e,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, value: Any) -> None:
    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, np.ndarray):
            return clean(item.tolist())
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _input_items(args: argparse.Namespace) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if args.manifest:
        manifest = args.manifest.resolve()
        with manifest.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"clip_id", "video_path"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("manifest CSV must contain clip_id and video_path columns")
            for row in reader:
                clip_id, raw_path = (row.get("clip_id") or "").strip(), (row.get("video_path") or "").strip()
                if not clip_id or not raw_path:
                    raise ValueError("manifest contains an empty clip_id or video_path")
                video_path = Path(raw_path).expanduser()
                if not video_path.is_absolute():
                    video_path = manifest.parent / video_path
                items.append({"clip_id": clip_id, "video_path": str(video_path.resolve())})
    else:
        for raw_path in args.videos:
            video_path = Path(raw_path).expanduser().resolve()
            items.append({"clip_id": str(video_path), "video_path": str(video_path)})
    if not items:
        raise ValueError("no input videos were provided")
    ids = [item["clip_id"] for item in items]
    paths = [item["video_path"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("clip_id values must be unique")
    if len(paths) != len(set(paths)):
        raise ValueError("the same video path was supplied more than once")
    for item in items:
        path = Path(item["video_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
    return items


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("av", "numpy", "opencv-python", "scipy", "scikit-image", "torch", "lpips"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _assert_idle_gpu() -> None:
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not selected.isdigit():
        raise RuntimeError("GPU policy: set CUDA_VISIBLE_DEVICES to one physical GPU index")
    try:
        devices = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"GPU policy: could not verify GPU process state with nvidia-smi: {exc}") from exc
    uuid_by_index = {}
    for line in devices:
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) == 2:
            uuid_by_index[fields[0]] = fields[1]
    if selected not in uuid_by_index:
        raise RuntimeError(f"GPU policy: nvidia-smi did not report GPU index {selected}")
    selected_uuid = uuid_by_index[selected]
    if any(line.strip().startswith(selected_uuid + ",") for line in processes):
        raise RuntimeError(f"GPU policy: GPU {selected} has running compute processes; choose a fully free GPU")


def _lpips_model(device: str, weights_dir: Path | None):
    if weights_dir is not None:
        weights_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(weights_dir))
    import lpips
    import torch

    model = lpips.LPIPS(net="alex", version="0.1", pretrained=True)
    model.eval().to(device)
    return model, torch


def _metric_columns(metrics: set[str], thresholds: tuple[float, ...]) -> list[str]:
    columns: list[str] = []
    if "ihd" in metrics:
        columns.extend(("ihd_mean_ref_norm", "ihd_ref0_norm"))
    if "color_w1" in metrics:
        columns.extend(("color_w1_r", "color_w1_g", "color_w1_b", "color_w1_mean"))
    if "glcm_texture" in metrics:
        for prop in GLCM_PROPERTIES:
            columns.extend(f"glcm_{prop}_{angle}" for angle in (0, 45, 90, 135))
        columns.extend(("glcm_texture_similarity", "glcm_texture_distance"))
    if "patch_de00" in metrics:
        columns.extend(("patch_de00_mean", "patch_de00_median"))
    if "appearance_coverage" in metrics:
        columns.extend(f"appearance_coverage_de00_gt_{threshold:g}" for threshold in thresholds)
    if "lpips" in metrics:
        columns.append("lpips_alex_v01")
    return columns


def _clip_key(clip_id: str) -> str:
    return hashlib.sha256(clip_id.encode("utf-8")).hexdigest()[:24]


def _evaluate_video(
    item: dict[str, str],
    metrics: set[str],
    thresholds: tuple[float, ...],
    patch_size: int,
    ihd_working_side: int,
    device: str,
    lpips_batch_size: int,
    model_and_torch: tuple[Any, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    path = Path(item["video_path"])
    errors: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    ihd_histograms: list[np.ndarray | None] = []
    ref_rgb: np.ndarray | None = None
    ref_glcm: np.ndarray | None = None
    ref_patch_means: np.ndarray | None = None
    patch_weights: np.ndarray | None = None
    lpips_model = model_and_torch[0] if model_and_torch else None
    torch = model_and_torch[1] if model_and_torch else None
    lpips_pending: list[tuple[int, np.ndarray]] = []
    geometry: tuple[int, int] | None = None
    decode_error: str | None = None

    try:
        with av.open(str(path), mode="r") as container:
            stream = next((stream for stream in container.streams if stream.type == "video"), None)
            if stream is None:
                raise ValueError("input contains no video stream")
            for frame_idx, decoded in enumerate(container.decode(stream)):
                rgb = decoded.to_ndarray(format="rgb24")
                if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                    raise ValueError(f"decoder returned unexpected frame format at frame {frame_idx}")
                current_geometry = (int(rgb.shape[1]), int(rgb.shape[0]))
                if geometry is None:
                    geometry = current_geometry
                    ref_rgb = rgb.copy()
                elif current_geometry != geometry:
                    raise ValueError(
                        f"frame geometry changed at frame {frame_idx}: {current_geometry} != {geometry}"
                    )

                row: dict[str, Any] = {
                    "clip_id": item["clip_id"],
                    "video_path": item["video_path"],
                    "frame_idx": frame_idx,
                    "reference_frame_idx": 0,
                    "original_width": current_geometry[0],
                    "original_height": current_geometry[1],
                }
                if "ihd" in metrics:
                    try:
                        working_width, working_height = ihd_working_size(
                            current_geometry[0], current_geometry[1], ihd_working_side
                        )
                        row["ihd_working_width"] = working_width
                        row["ihd_working_height"] = working_height
                        ihd_histograms.append(illumination_histogram(rgb, working_side=ihd_working_side))
                    except Exception as exc:
                        ihd_histograms.append(None)
                        errors.append({"clip_id": item["clip_id"], "frame_idx": frame_idx, "metric": "ihd",
                                       "error": f"{type(exc).__name__}: {exc}"})

                if frame_idx > 0 and "color_w1" in metrics:
                    try:
                        row.update(color_histogram_wasserstein(ref_rgb, rgb))
                    except Exception as exc:
                        errors.append({"clip_id": item["clip_id"], "frame_idx": frame_idx, "metric": "color_w1",
                                       "error": f"{type(exc).__name__}: {exc}"})
                elif frame_idx == 0 and "color_w1" in metrics:
                    row.update({"color_w1_r": 0.0, "color_w1_g": 0.0,
                                "color_w1_b": 0.0, "color_w1_mean": 0.0})

                if "glcm_texture" in metrics:
                    try:
                        current_glcm = glcm_features(rgb)
                        if frame_idx == 0:
                            ref_glcm = current_glcm
                        for prop_idx, prop in enumerate(GLCM_PROPERTIES):
                            for angle_idx, angle in enumerate((0, 45, 90, 135)):
                                row[f"glcm_{prop}_{angle}"] = float(current_glcm[prop_idx * 4 + angle_idx])
                        if frame_idx > 0:
                            similarity = cosine_similarity(ref_glcm, current_glcm)
                            if abs(similarity - 1.0) <= 1e-12:
                                similarity = 1.0
                            row["glcm_texture_similarity"] = similarity
                            row["glcm_texture_distance"] = 1.0 - similarity
                        else:
                            row["glcm_texture_similarity"] = 1.0
                            row["glcm_texture_distance"] = 0.0
                    except Exception as exc:
                        errors.append({"clip_id": item["clip_id"], "frame_idx": frame_idx, "metric": "glcm_texture",
                                       "error": f"{type(exc).__name__}: {exc}"})

                if "patch_de00" in metrics or "appearance_coverage" in metrics:
                    try:
                        current_means, current_weights = lab_patch_means(rgb, patch_size=patch_size)
                        if frame_idx == 0:
                            ref_patch_means, patch_weights = current_means, current_weights
                            patch_scores = np.zeros(len(current_weights), dtype=np.float64)
                        else:
                            patch_values, patch_scores = patch_delta_e(ref_patch_means, current_means, patch_weights)
                            if "patch_de00" in metrics:
                                row.update(patch_values)
                        if frame_idx == 0 and "patch_de00" in metrics:
                            row["patch_de00_mean"] = 0.0
                            row["patch_de00_median"] = 0.0
                        if "appearance_coverage" in metrics:
                            row.update(appearance_coverage(patch_scores, patch_weights, thresholds))
                    except Exception as exc:
                        errors.append({"clip_id": item["clip_id"], "frame_idx": frame_idx,
                                       "metric": "patch_de00/appearance_coverage",
                                       "error": f"{type(exc).__name__}: {exc}"})

                if "lpips" in metrics:
                    if frame_idx == 0:
                        row["lpips_alex_v01"] = 0.0
                    else:
                        lpips_pending.append((frame_idx, rgb))

                frame_rows.append(row)

                if lpips_pending and (len(lpips_pending) >= lpips_batch_size):
                    _fill_lpips_rows(
                        lpips_pending, frame_rows, ref_rgb, lpips_model, torch, device,
                        item["clip_id"], errors,
                    )
                    lpips_pending.clear()
    except Exception as exc:
        decode_error = f"{type(exc).__name__}: {exc}"
        errors.append({"clip_id": item["clip_id"], "metric": "decode",
                       "error": decode_error, "traceback": traceback.format_exc()})

    if lpips_pending:
        _fill_lpips_rows(lpips_pending, frame_rows, ref_rgb, lpips_model, torch, device, item["clip_id"], errors)
    if not frame_rows and not decode_error:
        decode_error = "video decoder returned no frames"
        errors.append({"clip_id": item["clip_id"], "metric": "decode", "error": decode_error})

    if "ihd" in metrics:
        if ihd_histograms and all(hist is not None for hist in ihd_histograms):
            scores = ihd_scores(np.stack(ihd_histograms))
            for row, mean_value, ref0_value in zip(
                frame_rows, scores["ihd_mean_ref_norm"], scores["ihd_ref0_norm"]
            ):
                row["ihd_mean_ref_norm"] = float(mean_value)
                row["ihd_ref0_norm"] = float(ref0_value)
            ihd_video = scores["ihd_video_norm"]
            ihd_ref0_mean = scores["ihd_ref0_norm_mean"]
        else:
            for row in frame_rows:
                row["ihd_mean_ref_norm"] = math.nan
                row["ihd_ref0_norm"] = math.nan
            ihd_video = math.nan
            ihd_ref0_mean = math.nan
    else:
        ihd_video = ihd_ref0_mean = math.nan

    frame_count = len(frame_rows)
    summary: dict[str, Any] = {
        **item,
        "frame_count": frame_count,
        "original_width": geometry[0] if geometry else None,
        "original_height": geometry[1] if geometry else None,
        "decode_status": "failed" if decode_error else ("completed_with_metric_errors" if errors else "completed"),
        "error": decode_error or "",
    }
    if "ihd" in metrics:
        summary["ihd_video_norm"] = ihd_video
        summary["ihd_ref0_norm_mean"] = ihd_ref0_mean
    metric_value_columns = [column for column in _metric_columns(metrics, thresholds)
                            if column not in ("ihd_mean_ref_norm", "ihd_ref0_norm")]
    for column in metric_value_columns:
        values = [row[column] for row in frame_rows[1:] if column in row and _is_number(row[column])]
        summary[f"{column}_mean_frames_1_to_end"] = float(np.mean(values)) if values else math.nan
    for row in frame_rows:
        for column in _metric_columns(metrics, thresholds):
            row.setdefault(column, math.nan)
    return frame_rows, summary, errors


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def _fill_lpips_rows(
    pending: list[tuple[int, np.ndarray]],
    frame_rows: list[dict[str, Any]],
    reference: np.ndarray,
    model: Any,
    torch: Any,
    device: str,
    clip_id: str,
    errors: list[dict[str, Any]],
) -> None:
    try:
        if model is None or torch is None or reference is None:
            raise RuntimeError("LPIPS model is unavailable")
        reference_tensor = torch.from_numpy(reference.copy()).permute(2, 0, 1).unsqueeze(0)
        reference_tensor = reference_tensor.to(device=device, dtype=torch.float32).div_(127.5).sub_(1.0)
        images = np.stack([image for _, image in pending])
        current_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
        current_tensor = current_tensor.to(device=device, dtype=torch.float32).div_(127.5).sub_(1.0)
        references = reference_tensor.expand(len(pending), -1, -1, -1)
        with torch.inference_mode():
            values = model(references, current_tensor).reshape(-1).detach().cpu().numpy()
        row_by_index = {int(row["frame_idx"]): row for row in frame_rows}
        for (frame_idx, _), value in zip(pending, values):
            row_by_index[frame_idx]["lpips_alex_v01"] = float(value)
    except Exception as exc:
        for frame_idx, _ in pending:
            row_by_index = {int(row["frame_idx"]): row for row in frame_rows}
            row_by_index[frame_idx]["lpips_alex_v01"] = math.nan
            errors.append({"clip_id": clip_id, "frame_idx": frame_idx, "metric": "lpips",
                           "error": f"{type(exc).__name__}: {exc}"})


def _aggregate(output_dir: Path, items: list[dict[str, str]], thresholds: tuple[float, ...]) -> None:
    frame_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    clips_dir = output_dir / ".clips"
    for item in items:
        checkpoint = clips_dir / f"{_clip_key(item['clip_id'])}.json"
        if not checkpoint.is_file():
            continue
        result = json.loads(checkpoint.read_text(encoding="utf-8"))
        frame_rows.extend(result["per_frame"])
        video_rows.append(result["per_video"])
        error_rows.extend(result.get("errors", []))
    _csv_write(output_dir / "per_frame_metrics.csv", frame_rows)
    _csv_write(output_dir / "per_video_metrics.csv", video_rows)
    error_path = output_dir / "errors.jsonl"
    temporary = error_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in error_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, error_path)


def run(args: argparse.Namespace) -> int:
    items = _input_items(args)
    metrics = set(args.metrics)
    thresholds = tuple(args.coverage_thresholds)
    if not thresholds or any(not math.isfinite(x) or x < 0 for x in thresholds):
        raise ValueError("coverage thresholds must be a nonempty list of finite nonnegative values")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("coverage thresholds must be unique")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / ".clips"
    clips_dir.mkdir(exist_ok=True)

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    if device.startswith("cuda"):
        _assert_idle_gpu()

    manifest = {
        "schema_version": 1,
        "metrics": sorted(metrics),
        "coverage_thresholds": list(thresholds),
        "parameters": {
            "ihd": {
                "red_channel": 0,
                "gaussian_sigma": 80.0,
                "kernel_size": "working_image_height_by_width",
                "padding": "replicate",
                "uint8_rounding": "nearest_positive_ties_up",
                "zero_blur_replacement": "minimum_positive_value",
                "working_max_side": args.ihd_working_side,
                "resize_interpolation": {"down": "INTER_AREA", "up": "INTER_CUBIC"},
                "histogram_bins": 255,
                "histogram_range": [0, 255],
                "histogram_normalization": "actual_pixel_count",
            },
            "color_w1": {"channels": "RGB", "bins": 256, "support": [0.0, 1.0]},
            "glcm_texture": {
                "grayscale": "OpenCV COLOR_RGB2GRAY",
                "levels": 256,
                "distance": 1,
                "angles_radians": [float(angle) for angle in GLCM_ANGLES],
                "properties": list(GLCM_PROPERTIES),
                "symmetric": False,
                "normed": True,
            },
            "patch_de00": {
                "patch_size": args.patch_size,
                "color_space": "CIELAB, D65, 2-degree observer",
                "delta_e_weights": {"kL": 1, "kC": 1, "kH": 1},
                "edge_patches": "partial_without_padding",
            },
            "lpips": {"package": "lpips", "backbone": "alex", "version": "0.1",
                      "pretrained_calibration": True, "batch_size": args.lpips_batch_size},
        },
        "decoder": "PyAV/libav; decoded in presentation order as rgb24",
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "") if device == "cuda" else None,
        "versions": _versions(),
        "inputs": [{**item, "sha256": _sha256(Path(item["video_path"]))} for item in items],
    }
    # Compare the exact JSON representation so tuples and NumPy scalar types
    # cannot make an otherwise identical resume configuration appear different.
    manifest = json.loads(json.dumps(manifest, sort_keys=True))
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != manifest:
            raise ValueError(
                f"{config_path} exists with a different configuration/input fingerprint; "
                "choose a new output directory"
            )
    else:
        occupied = any((output_dir / name).exists() for name in (
            "per_frame_metrics.csv", "per_video_metrics.csv", "errors.jsonl"
        ))
        if occupied:
            raise ValueError("output directory contains prior results but no run_config.json")
        _json_write(config_path, manifest)

    model_and_torch = None
    model_load_error = None
    if "lpips" in metrics:
        try:
            model_and_torch = _lpips_model(device, args.weights_dir)
        except Exception as exc:
            model_and_torch = (None, None)
            model_load_error = f"{type(exc).__name__}: {exc}"

    for item in items:
        checkpoint = clips_dir / f"{_clip_key(item['clip_id'])}.json"
        if checkpoint.exists():
            old_result = json.loads(checkpoint.read_text(encoding="utf-8"))
            if old_result["per_video"].get("decode_status") == "completed":
                print(f"[SKIP] {item['clip_id']} (completed checkpoint)", flush=True)
                continue
            print(f"[RETRY] {item['clip_id']} (previous run was incomplete)", flush=True)
        else:
            print(f"[EVALUATE] {item['clip_id']}", flush=True)
        frames, summary, errors = _evaluate_video(
            item, metrics, thresholds, args.patch_size, args.ihd_working_side,
            device, args.lpips_batch_size, model_and_torch,
        )
        if model_load_error:
            errors.insert(0, {"clip_id": item["clip_id"], "metric": "lpips_model",
                              "error": model_load_error})
            summary["decode_status"] = "completed_with_metric_errors"
            summary["lpips_model_error"] = model_load_error
        _json_write(checkpoint, {"per_frame": frames, "per_video": summary, "errors": errors})
        _aggregate(output_dir, items, thresholds)
    _aggregate(output_dir, items, thresholds)
    _json_write(output_dir / "run_metadata.json", {
        **manifest,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status_counts": {
            status: sum(row.get("decode_status") == status for row in
                        (json.loads((clips_dir / f"{_clip_key(item['clip_id'])}.json").read_text())["per_video"]
                         for item in items if (clips_dir / f"{_clip_key(item['clip_id'])}.json").exists()))
            for status in ("completed", "completed_with_metric_errors", "failed")
        },
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path, help="CSV with unique clip_id,video_path columns")
    inputs.add_argument("--videos", nargs="+", help="Explicit raw video paths")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", choices=METRIC_FAMILIES, default=list(METRIC_FAMILIES))
    parser.add_argument("--coverage-thresholds", nargs="+", type=float, default=list(DEFAULT_COVERAGE_THRESHOLDS))
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--ihd-working-side", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--lpips-batch-size", type=int, default=1)
    parser.add_argument("--weights-dir", type=Path, help="Optional torch model-cache directory")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.metrics or len(set(args.metrics)) != len(args.metrics):
        raise SystemExit("--metrics must list one or more distinct metric families")
    if args.patch_size <= 0 or args.ihd_working_side <= 0 or args.lpips_batch_size <= 0:
        raise SystemExit("patch size, IHD working side, and LPIPS batch size must be positive")
    try:
        return run(args)
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
