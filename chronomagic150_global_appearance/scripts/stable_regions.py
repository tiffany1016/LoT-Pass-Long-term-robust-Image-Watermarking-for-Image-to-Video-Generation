#!/usr/bin/env python3
"""Source-anchored stability and full-frame overlay videos for ChronoMagic."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "chronomagic150_global_appearance"
DATA = EXP / "dataset"
VIDEOS = EXP / "generated_videos"
WATERMARKED = EXP / "watermarked_sources"
OUT = EXP / "stable_region"
PATCH, MAE_THRESHOLD, SSIM_THRESHOLD, ALPHA, FPS = 16, 0.05, 0.90, 0.35, 8


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def load_reused_stability_code():
    path = ROOT / "claw_lotpass_pilot/scripts/stable_regions.py"
    spec = importlib.util.spec_from_file_location("lotpass_reused_stability", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.patch_grids, module.expand


def main() -> None:
    import decord
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from PIL import Image

    patch_grids, expand = load_reused_stability_code()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 1.1))
    fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="RdYlBu"), cax=ax, orientation="horizontal")
    ax.set_xlabel("Source-anchored patch similarity (red low → yellow → blue high)")
    fig.tight_layout(); fig.savefig(OUT / "similarity_legend.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    ba = {}
    raw_csv = EXP / "decoding/per_frame_raw_ba.csv"
    if raw_csv.is_file():
        for row in csv.DictReader(raw_csv.open(encoding="utf-8")):
            try:
                ba[(row["case_id"], int(row["frame_idx"]))] = float(row["raw_bit_accuracy"])
            except (KeyError, ValueError):
                continue
    cases = list(csv.DictReader((DATA / "chronomagic150_generation_manifest.csv").open(encoding="utf-8")))
    frame_rows, video_rows = [], []
    for index, row in enumerate(cases, 1):
        sample_id, condition = row["sample_id"], row["condition"]
        case_id = row["case_id"]
        video = VIDEOS / sample_id / f"{condition}.mp4"
        run_status = EXP / "runs" / sample_id / condition / "run_status.json"
        status = json.loads(run_status.read_text()) if run_status.is_file() else {}
        common = {"case_id": case_id, "sample_id": sample_id, "selection_index": row["selection_index"],
                  "analysis_group": row["analysis_group"], "condition": condition, "seed": int(row["seed"])}
        if status.get("status") != "completed" or not video.is_file():
            video_rows.append({**common, "status": "generation_" + str(status.get("status", "missing"))})
            continue
        case_out = OUT / sample_id / condition
        folders = ["mae_maps", "ssim_maps", "similarity_maps", "binary_masks", "overlays"]
        for folder in folders:
            (case_out / folder).mkdir(parents=True, exist_ok=True)
        source_path = WATERMARKED / sample_id / "watermarked.png"
        source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.uint8)
        reader = decord.VideoReader(str(video), ctx=decord.cpu(0))
        fps = float(reader.get_avg_fps())
        if abs(fps - FPS) > 0.05:
            raise ValueError(f"Unexpected FPS for {case_id}: {fps}")
        sim_tmp, mask_tmp = case_out / ".similarity.partial.mp4", case_out / ".stable_mask.partial.mp4"
        sim_writer = imageio.get_writer(str(sim_tmp), fps=fps, codec="libx264", quality=8,
                                        macro_block_size=1, ffmpeg_params=["-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        mask_writer = imageio.get_writer(str(mask_tmp), fps=fps, codec="libx264", quality=8,
                                         macro_block_size=1, ffmpeg_params=["-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        this_video_rows = []
        try:
            for frame_idx in range(len(reader)):
                frame = reader[frame_idx].asnumpy()
                mae, ssim, areas, stable, similarity, stats = patch_grids(frame, source)
                np.save(case_out / "mae_maps" / f"{frame_idx:04d}.npy", mae)
                np.save(case_out / "ssim_maps" / f"{frame_idx:04d}.npy", ssim)
                np.save(case_out / "similarity_maps" / f"{frame_idx:04d}.npy", similarity)
                mask_grid = stable.astype(np.uint8)
                mask = expand(mask_grid, frame.shape[0], frame.shape[1])
                Image.fromarray(mask * 255, mode="L").save(case_out / "binary_masks" / f"{frame_idx:04d}.png")
                colored = (matplotlib.colormaps["RdYlBu"](np.clip(expand(similarity, frame.shape[0], frame.shape[1]), 0, 1))[..., :3] * 255).astype(np.uint8)
                sim_overlay = np.clip((1-ALPHA)*frame.astype(np.float32) + ALPHA*colored.astype(np.float32), 0, 255).astype(np.uint8)
                stable_overlay = frame.copy()
                selected = mask.astype(bool)
                green = np.zeros_like(frame); green[..., 1] = 255
                stable_overlay[selected] = np.clip((1-ALPHA)*frame[selected].astype(np.float32) + ALPHA*green[selected].astype(np.float32), 0, 255).astype(np.uint8)
                sim_writer.append_data(sim_overlay); mask_writer.append_data(stable_overlay)
                if frame_idx in (0, 12, 24, 36, 48):
                    Image.fromarray(sim_overlay).save(case_out / "overlays" / f"similarity_{frame_idx:04d}.png")
                    Image.fromarray(stable_overlay).save(case_out / "overlays" / f"stable_{frame_idx:04d}.png")
                item = {**common, "frame_idx": frame_idx, "raw_bit_accuracy": ba.get((case_id, frame_idx), float("nan")),
                        **stats, "status": "completed", "patch_size": PATCH,
                        "mae_threshold": MAE_THRESHOLD, "ssim_threshold": SSIM_THRESHOLD}
                this_video_rows.append(item); frame_rows.append(item)
        finally:
            sim_writer.close(); mask_writer.close()
        sim_path, mask_path = case_out / "similarity_overlay.mp4", case_out / "stable_mask_overlay.mp4"
        os.replace(sim_tmp, sim_path); os.replace(mask_tmp, mask_path)
        summary = {**common, "status": "completed", "frame_count": len(reader), "fps": fps,
                   "mean_stable_region_ratio": float(np.mean([r["stable_region_ratio"] for r in this_video_rows])),
                   "mean_largest_stable_region_ratio": float(np.mean([r["largest_stable_region_ratio"] for r in this_video_rows])),
                   "mean_similarity": float(np.mean([r["mean_similarity"] for r in this_video_rows])),
                   "similarity_overlay": str(sim_path.relative_to(ROOT)), "stable_mask_overlay": str(mask_path.relative_to(ROOT))}
        video_rows.append(summary)
        (case_out / "stability_metadata.json").write_text(json.dumps({
            **common, "reference": "watermarked source image; same-coordinate pixel/patch comparison",
            "optical_flow_alignment": False, "temporal_stability": False, "patch_size": PATCH,
            "mae_threshold": MAE_THRESHOLD, "ssim_threshold": SSIM_THRESHOLD,
            "similarity_mapping": "0.5*clip(1-MAE/0.15,0,1)+0.5*clip(SSIM,0,1)",
            "overlay_alpha": ALPHA, "fps": fps, "frame_count": len(reader),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_csv(OUT / "per_frame_metrics.csv", frame_rows, list(dict.fromkeys(k for r in frame_rows for k in r)))
    write_csv(OUT / "per_video_metrics.csv", video_rows, list(dict.fromkeys(k for r in video_rows for k in r)))
    print(f"Source-anchored stable-region analysis complete: {len(video_rows)} videos / {len(frame_rows)} frames.", flush=True)


if __name__ == "__main__":
    main()
