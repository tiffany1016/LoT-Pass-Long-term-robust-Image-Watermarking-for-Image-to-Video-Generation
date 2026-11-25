#!/usr/bin/env python3
"""Estimate unaligned global displacement with the repository's RAFT helper."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "vbench_lotpass_pilot"
SELECTION = EXP / "selection"
RUNS = EXP / "runs"
OUT = EXP / "analysis/motion"
SEED = 42
SAMPLE_FRAMES = (12, 24, 36, 48)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def case_dir(row: dict) -> Path:
    return RUNS / row["image_id"] / row["motion_group"] / f"seed_{SEED}"


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU policy: launch with CUDA_VISIBLE_DEVICES=0")
    approval = json.loads((SELECTION / "approval.json").read_text())
    manifest_path = SELECTION / "selection_manifest.json"
    manifest_sha = sha256(manifest_path)
    if not approval.get("approved") or approval.get("selection_manifest_sha256") != manifest_sha:
        raise RuntimeError("Selection approval is missing or its manifest hash no longer matches")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    # Initialize CUDA before loading Decord's bundled multimedia libraries.
    torch.cuda.init()
    import decord

    from utils.utils import TorchvisionRAFT

    raft = TorchvisionRAFT(device="cuda:0")
    rows = list(csv.DictReader((SELECTION / "vbench_40pairs.csv").open(encoding="utf-8")))
    displacement_rows, summaries = [], {}
    base_validation = {}
    validation_path = OUT / "motion_validation.csv"
    if validation_path.is_file():
        base_validation = {r["case_id"]: r for r in csv.DictReader(validation_path.open(encoding="utf-8"))}

    for index, row in enumerate(rows, start=1):
        directory = case_dir(row)
        video = directory / "video.mp4"
        status_file = directory / "run_status.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        if not video.is_file() or status.get("status") != "completed":
            summaries[row["case_id"]] = {"automated_motion_summary": "unavailable: generation not completed"}
            continue
        reader = decord.VideoReader(str(video), ctx=decord.cpu(0))
        first = reader[0].asnumpy()
        vectors = []
        for frame_idx in SAMPLE_FRAMES:
            if frame_idx >= len(reader):
                continue
            target = reader[frame_idx].asnumpy()
            flow = raft.compute_flow(first, target)
            dx = float(np.median(flow[..., 0]))
            dy = float(np.median(flow[..., 1]))
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            moving_fraction = float(np.mean(magnitude > 0.5))
            item = {"image_id": row["image_id"], "case_id": row["case_id"], "motion_full": row["motion_full"],
                    "seed": SEED, "reference_frame": 0, "target_frame": frame_idx,
                    "median_flow_dx_px": dx, "median_flow_dy_px": dy,
                    "median_flow_magnitude_px": float(np.median(magnitude)),
                    "p90_flow_magnitude_px": float(np.quantile(magnitude, .90)),
                    "fraction_pixels_flow_gt_0_5px": moving_fraction,
                    "method": "existing TorchvisionRAFT.compute_flow; no warping or video modification"}
            displacement_rows.append(item)
            vectors.append((dx, dy, item["median_flow_magnitude_px"], moving_fraction))
        if vectors:
            matrix = np.asarray(vectors, dtype=float)
            summaries[row["case_id"]] = {
                "automated_motion_summary": "RAFT frame0-to-sample displacement; unaligned, descriptive only",
                "median_global_dx_px": float(np.median(matrix[:, 0])),
                "median_global_dy_px": float(np.median(matrix[:, 1])),
                "median_flow_magnitude_px": float(np.median(matrix[:, 2])),
                "mean_fraction_pixels_flow_gt_0_5px": float(np.mean(matrix[:, 3])),
                "sampled_frame_indices": list(SAMPLE_FRAMES),
            }
        print(f"[RAFT {index}/40] {row['case_id']}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "motion_displacement.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["image_id", "case_id", "motion_full", "seed", "reference_frame", "target_frame",
                  "median_flow_dx_px", "median_flow_dy_px", "median_flow_magnitude_px", "p90_flow_magnitude_px",
                  "fraction_pixels_flow_gt_0_5px", "method"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(displacement_rows)
    validation_rows = []
    for row in rows:
        old = base_validation.get(row["case_id"], {})
        validation_rows.append({**old, **summaries.get(row["case_id"], {}),
                                "image_id": row["image_id"], "case_id": row["case_id"],
                                "motion_group": row["motion_group"], "motion_full": row["motion_full"], "seed": SEED,
                                "motion_validation_status": old.get("motion_validation_status", "not_reviewed"),
                                "validation_method": "automatic RAFT displacement summary; human review still pending"})
    with validation_path.open("w", newline="", encoding="utf-8") as stream:
        fields = sorted({key for item in validation_rows for key in item})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(validation_rows)
    (OUT / "raft_method.json").write_text(json.dumps({
        "method": "repository utils.utils.TorchvisionRAFT.compute_flow",
        "reference_frame": 0, "target_frames": list(SAMPLE_FRAMES), "alignment_or_warping": False,
        "video_modified": False, "motion_status_labels": "not_reviewed; RAFT numbers do not automatically validate prompts",
        "selection_manifest_sha256": manifest_sha,
    }, indent=2) + "\n")
    print(f"Wrote unaligned RAFT summaries for {len(displacement_rows)} frame pairs; no human validity labels assigned")


if __name__ == "__main__":
    main()
