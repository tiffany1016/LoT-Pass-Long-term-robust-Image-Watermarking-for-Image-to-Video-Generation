#!/usr/bin/env python3
"""Compute source-anchored patch stability and clear-video overlays (CPU only)."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "vbench_lotpass_pilot"
SELECTION = EXP / "selection"
ANALYSIS = EXP / "analysis/stability"
RUNS = EXP / "runs"
SEED = 42
PATCH = 16
SSIM_WIN = 7
DATA_RANGE = 1.0
DEFAULT_MAE = 0.05
DEFAULT_SSIM = 0.90
SIM_MAE_SCALE = 0.15
ALPHA = 0.35
CALIBRATION_CASES = {"VB001_static", "VB004_static", "VB007_static", "VB010_static"}
CALIBRATION_FRAMES = {1, 24, 48}
CAL_MAE_GRID = (0.03, 0.05, 0.075, 0.10, 0.15)
CAL_SSIM_GRID = (0.80, 0.85, 0.90, 0.95)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def case_dir(row: dict) -> Path:
    return RUNS / row["image_id"] / row["motion_group"] / f"seed_{SEED}"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def patch_grids(frame_rgb: np.ndarray, source_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return patch MAE, 7x7-window RGB SSIM means, and actual patch areas."""
    frame = frame_rgb.astype(np.float32) / 255.0
    source = source_rgb.astype(np.float32) / 255.0
    if frame.shape != source.shape or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Source/frame geometry mismatch: {source.shape} vs {frame.shape}")
    height, width = frame.shape[:2]
    y_edges = list(range(0, height, PATCH)) + [height]
    x_edges = list(range(0, width, PATCH)) + [width]
    frame_patches, source_patches, areas = [], [], []
    for yi in range(len(y_edges) - 1):
        for xi in range(len(x_edges) - 1):
            y0, y1, x0, x1 = y_edges[yi], y_edges[yi + 1], x_edges[xi], x_edges[xi + 1]
            frame_patches.append(frame[y0:y1, x0:x1])
            source_patches.append(source[y0:y1, x0:x1])
            areas.append((y1-y0) * (x1-x0))
    maes = np.asarray([np.mean(np.abs(a - b)) for a, b in zip(frame_patches, source_patches)], dtype=np.float32)
    ssims = np.empty(len(frame_patches), dtype=np.float32)
    c1, c2 = (0.01 * DATA_RANGE) ** 2, (0.03 * DATA_RANGE) ** 2
    for start in range(0, len(frame_patches), 64):
        a = np.stack(frame_patches[start:start+64], axis=0)
        b = np.stack(source_patches[start:start+64], axis=0)
        if a.shape[1] < SSIM_WIN or a.shape[2] < SSIM_WIN:
            raise ValueError(f"Patch too small for SSIM window: {a.shape[1:3]}")
        wa = np.lib.stride_tricks.sliding_window_view(a, (SSIM_WIN, SSIM_WIN), axis=(1, 2))
        wb = np.lib.stride_tricks.sliding_window_view(b, (SSIM_WIN, SSIM_WIN), axis=(1, 2))
        mu_a, mu_b = wa.mean(axis=(-1, -2)), wb.mean(axis=(-1, -2))
        var_a = np.maximum(0.0, (wa * wa).mean(axis=(-1, -2)) - mu_a * mu_a)
        var_b = np.maximum(0.0, (wb * wb).mean(axis=(-1, -2)) - mu_b * mu_b)
        cov = (wa * wb).mean(axis=(-1, -2)) - mu_a * mu_b
        score = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2))
        ssims[start:start+len(a)] = score.mean(axis=(1, 2, 3))
    shape = (len(y_edges) - 1, len(x_edges) - 1)
    return maes.reshape(shape), ssims.reshape(shape), np.asarray(areas, dtype=np.float32).reshape(shape)


def stable_metrics(mae: np.ndarray, ssim: np.ndarray, areas: np.ndarray, mae_threshold: float, ssim_threshold: float) -> dict:
    from scipy import ndimage
    mask = (mae < mae_threshold) & (ssim > ssim_threshold)
    total = float(areas.sum())
    stable_area = float(areas[mask].sum())
    labels, count = ndimage.label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))
    largest = 0.0
    for label in range(1, count + 1):
        largest = max(largest, float(areas[labels == label].sum()))
    similarity = 0.5 * np.clip(1.0 - mae / SIM_MAE_SCALE, 0.0, 1.0) + 0.5 * np.clip(ssim, 0.0, 1.0)
    mean_similarity = float(np.sum(similarity * areas) / total)
    return {"mask": mask, "similarity": similarity, "srr": stable_area / total,
            "lsrr": largest / total, "mean_similarity": mean_similarity}


def load_frame(reader, frame_idx: int) -> np.ndarray:
    return reader[frame_idx].asnumpy()


def calibration_thresholds(rows: list[dict], readers, source_by_image) -> tuple[float, float, dict]:
    samples = []
    for row in rows:
        if row["case_id"] not in CALIBRATION_CASES or row["case_id"] not in readers:
            continue
        reader = readers[row["case_id"]]
        for frame_idx in sorted(CALIBRATION_FRAMES):
            if frame_idx >= len(reader):
                continue
            mae, ssim, areas = patch_grids(load_frame(reader, frame_idx), source_by_image[row["image_id"]])
            samples.append((row["case_id"], frame_idx, mae, ssim, areas))
    if not samples:
        return DEFAULT_MAE, DEFAULT_SSIM, {"status": "calibration_unavailable; retained preregistered defaults", "samples": []}

    def median_srr(mae_threshold: float, ssim_threshold: float) -> float:
        vals = [stable_metrics(mae, ssim, areas, mae_threshold, ssim_threshold)["srr"] for _, _, mae, ssim, areas in samples]
        return float(np.median(vals))

    default_fraction = median_srr(DEFAULT_MAE, DEFAULT_SSIM)
    chosen_mae, chosen_ssim = DEFAULT_MAE, DEFAULT_SSIM
    candidate_results = []
    if default_fraction <= 0.01 or default_fraction >= 0.95:
        for mae_threshold in CAL_MAE_GRID:
            for ssim_threshold in CAL_SSIM_GRID:
                fraction = median_srr(mae_threshold, ssim_threshold)
                candidate_results.append({"mae_threshold": mae_threshold, "ssim_threshold": ssim_threshold,
                                          "median_stable_fraction": fraction})
        nondegenerate = [item for item in candidate_results if 0.01 < item["median_stable_fraction"] < 0.95]
        if nondegenerate:
            target = 0.10 if default_fraction <= 0.01 else 0.90
            selected = min(nondegenerate, key=lambda item: (abs(item["median_stable_fraction"]-target),
                                                              abs(item["mae_threshold"]-DEFAULT_MAE)+abs(item["ssim_threshold"]-DEFAULT_SSIM)))
            chosen_mae, chosen_ssim = selected["mae_threshold"], selected["ssim_threshold"]
    report = {
        "status": "completed; thresholds frozen before examining BA",
        "calibration_cases": sorted({case for case, _, *_ in samples}),
        "calibration_frame_indices": sorted({frame for _, frame, *_ in samples}),
        "sample_count": len(samples), "initial_thresholds": {"mae": DEFAULT_MAE, "ssim": DEFAULT_SSIM},
        "initial_median_stable_fraction": default_fraction,
        "candidate_grid_results": candidate_results,
        "chosen_thresholds": {"mae": chosen_mae, "ssim": chosen_ssim},
        "selection_rule": "Retain defaults unless median SRR on the predeclared static calibration set is <=1% or >=95%; if degenerate, choose grid pair with median SRR nearest 10% (low case) or 90% (high case), independent of BA.",
    }
    return float(chosen_mae), float(chosen_ssim), report


def colorize_similarity(similarity: np.ndarray, patch: int, height: int, width: int) -> np.ndarray:
    import matplotlib
    cmap = matplotlib.colormaps["RdYlBu"]
    colors = (cmap(np.clip(similarity, 0, 1))[..., :3] * 255).astype(np.uint8)
    return np.repeat(np.repeat(colors, patch, axis=0), patch, axis=1)[:height, :width]


def expand_grid(grid: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.repeat(np.repeat(grid, PATCH, axis=0), PATCH, axis=1)[:height, :width]


def main() -> None:
    import decord
    import imageio.v2 as imageio
    from PIL import Image

    approval = json.loads((SELECTION / "approval.json").read_text())
    manifest_path = SELECTION / "selection_manifest.json"
    manifest_sha = sha256(manifest_path)
    if not approval.get("approved") or approval.get("selection_manifest_sha256") != manifest_sha:
        raise RuntimeError("Selection approval is missing or its manifest hash no longer matches")
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((SELECTION / "vbench_40pairs.csv").open(encoding="utf-8")))
    completed_rows = []
    readers, source_by_image = {}, {}
    for row in rows:
        directory = case_dir(row)
        video = directory / "video.mp4"
        status_file = directory / "run_status.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        if video.is_file() and status.get("status") == "completed":
            readers[row["case_id"]] = decord.VideoReader(str(video), ctx=decord.cpu(0))
            completed_rows.append(row)
            source_by_image[row["image_id"]] = np.asarray(Image.open(directory / "effective_input.png").convert("RGB"), dtype=np.uint8)

    mae_threshold, ssim_threshold, calibration = calibration_thresholds(completed_rows, readers, source_by_image)
    calibration["selection_manifest_sha256"] = manifest_sha
    calibration["computed_at"] = datetime.now(timezone.utc).isoformat()
    calibration["patch_size"] = PATCH
    calibration["ssim_window"] = SSIM_WIN
    calibration["ssim_window_type"] = "uniform 7x7 valid windows, averaged within each 16x16 patch and over RGB"
    calibration["ssim_data_range"] = DATA_RANGE
    atomic_json(ANALYSIS / "calibration.json", calibration)
    fig, ax = plt.subplots(figsize=(6.5, 1.2))
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    colorbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="RdYlBu"), cax=ax, orientation="horizontal")
    colorbar.set_label("Source-anchored patch similarity (red low → yellow → blue high)")
    fig.tight_layout()
    fig.savefig(ANALYSIS / "similarity_legend.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    ba_by_case_frame = {}
    ba_csv = EXP / "analysis/raw_ba/per_frame_accuracy.csv"
    if ba_csv.is_file():
        for entry in csv.DictReader(ba_csv.open(encoding="utf-8")):
            ba_by_case_frame[(entry["case_id"], int(entry["frame_idx"]))] = float(entry["ba"])
    stability_rows = []
    all_metrics = []
    for case_index, row in enumerate(completed_rows, start=1):
        case_id = row["case_id"]
        directory = case_dir(row)
        out = directory / "stability"
        for subdir in ("similarity_maps", "mae_maps", "ssim_maps", "binary_masks", "overlays"):
            (out / subdir).mkdir(parents=True, exist_ok=True)
        reader = readers[case_id]
        source = source_by_image[row["image_id"]]
        overlay_tmp = out / ".similarity_overlay.partial.mp4"
        stable_tmp = out / ".stable_mask_overlay.partial.mp4"
        similarity_path, stable_path = out / "similarity_overlay.mp4", out / "stable_mask_overlay.mp4"
        sim_writer = imageio.get_writer(str(overlay_tmp), fps=8, codec="libx264", quality=8, macro_block_size=1)
        mask_writer = imageio.get_writer(str(stable_tmp), fps=8, codec="libx264", quality=8, macro_block_size=1)
        case_stats = []
        try:
            for frame_idx in range(len(reader)):
                frame = load_frame(reader, frame_idx)
                mae, ssim, areas = patch_grids(frame, source)
                metrics = stable_metrics(mae, ssim, areas, mae_threshold, ssim_threshold)
                similarity, mask = metrics["similarity"], metrics["mask"]
                np.save(out / "mae_maps" / f"{frame_idx:04d}.npy", mae)
                np.save(out / "ssim_maps" / f"{frame_idx:04d}.npy", ssim)
                np.save(out / "similarity_maps" / f"{frame_idx:04d}.npy", similarity)
                mask_pixels = expand_grid(mask.astype(np.uint8), frame.shape[0], frame.shape[1])
                Image.fromarray(mask_pixels * 255, mode="L").save(out / "binary_masks" / f"{frame_idx:04d}.png")
                color = colorize_similarity(similarity, PATCH, frame.shape[0], frame.shape[1])
                sim_overlay = np.clip((1-ALPHA)*frame.astype(np.float32) + ALPHA*color.astype(np.float32), 0, 255).astype(np.uint8)
                green = np.zeros_like(frame, dtype=np.uint8)
                green[..., 1] = 255
                stable_overlay = frame.copy()
                stable_overlay[mask_pixels.astype(bool)] = np.clip(
                    (1-ALPHA)*frame[mask_pixels.astype(bool)].astype(np.float32) + ALPHA*green[mask_pixels.astype(bool)].astype(np.float32), 0, 255
                ).astype(np.uint8)
                sim_writer.append_data(sim_overlay)
                mask_writer.append_data(stable_overlay)
                if frame_idx in (0, 24, 48):
                    Image.fromarray(sim_overlay).save(out / "overlays" / f"similarity_{frame_idx:04d}.png")
                    Image.fromarray(stable_overlay).save(out / "overlays" / f"stable_{frame_idx:04d}.png")
                ba = ba_by_case_frame.get((case_id, frame_idx), float("nan"))
                entry = {
                    "image_id": row["image_id"], "case_id": case_id, "motion_full": row["motion_full"],
                    "seed": 42, "frame_idx": frame_idx, "status": "completed", "ba": ba,
                    "srr": metrics["srr"], "lsrr": metrics["lsrr"], "mean_similarity": metrics["mean_similarity"],
                    "stable_area_pixels": float(areas[mask].sum()), "total_area_pixels": float(areas.sum()),
                    "mae_threshold": mae_threshold, "ssim_threshold": ssim_threshold,
                }
                stability_rows.append(entry)
                case_stats.append(entry)
            sim_writer.close()
            mask_writer.close()
            os.replace(overlay_tmp, similarity_path)
            os.replace(stable_tmp, stable_path)
        except Exception:
            sim_writer.close()
            mask_writer.close()
            overlay_tmp.unlink(missing_ok=True)
            stable_tmp.unlink(missing_ok=True)
            raise
        for entry in case_stats:
            all_metrics.append(entry)
        atomic_json(out / "stability_metadata.json", {
            "case_id": case_id, "seed": 42, "source_reference": str((directory / "effective_input.png").relative_to(ROOT)),
            "source_reference_sha256": sha256(directory / "effective_input.png"), "comparison": "same-coordinate RGB; no registration or optical flow",
            "patch_size": PATCH, "ssim_window": SSIM_WIN, "ssim_data_range": DATA_RANGE,
            "mae_threshold": mae_threshold, "ssim_threshold": ssim_threshold,
            "similarity_mapping": "0.5*clip(1-MAE/0.15,0,1)+0.5*clip(SSIM,0,1)",
            "overlay_alpha": ALPHA, "overlay_color_map": "RdYlBu (red low, yellow middle, blue high)",
            "frame_count": len(reader),
        })
        print(f"[STABILITY {case_index}/{len(completed_rows)}] {case_id}: {len(case_stats)} frames", flush=True)

    completed_cases = {row["case_id"] for row in completed_rows}
    for row in rows:
        if row["case_id"] not in completed_cases:
            status_file = case_dir(row) / "run_status.json"
            status = json.loads(status_file.read_text()).get("status", "missing") if status_file.exists() else "missing"
            stability_rows.append({"image_id": row["image_id"], "case_id": row["case_id"], "motion_full": row["motion_full"],
                                   "seed": 42, "frame_idx": "", "status": f"generation_{status}", "ba": float("nan"),
                                   "srr": float("nan"), "lsrr": float("nan"), "mean_similarity": float("nan"),
                                   "stable_area_pixels": float("nan"), "total_area_pixels": float("nan"),
                                   "mae_threshold": mae_threshold, "ssim_threshold": ssim_threshold})
    write_csv(ANALYSIS / "per_frame_stability.csv", stability_rows,
              ["image_id", "case_id", "motion_full", "seed", "frame_idx", "status", "ba", "srr", "lsrr", "mean_similarity",
               "stable_area_pixels", "total_area_pixels", "mae_threshold", "ssim_threshold"])
    for metric in ("srr", "lsrr"):
        fig, ax = plt.subplots(figsize=(7, 5))
        points = [r for r in stability_rows if np.isfinite(r["ba"])]
        ax.scatter([r[metric] for r in points], [r["ba"] for r in points], alpha=.5, s=16)
        ax.set(xlabel=metric.upper(), ylabel="Raw bit accuracy", title=f"{metric.upper()} vs raw BA (frames are correlated)")
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(ANALYSIS / f"{metric}_vs_ba.png", dpi=160)
        plt.close(fig)
    print(f"Computed stability for {len(completed_rows)} videos / {len(stability_rows)} frames; thresholds MAE<{mae_threshold}, SSIM>{ssim_threshold}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
