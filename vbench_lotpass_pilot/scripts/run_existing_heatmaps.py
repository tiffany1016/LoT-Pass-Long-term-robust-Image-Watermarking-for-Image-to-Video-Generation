#!/usr/bin/env python3
"""Reuse repository encoder residual and decoder gradient-saliency methods."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "vbench_lotpass_pilot"
SELECTION = EXP / "selection"
RUNS = EXP / "runs"
HEAT = EXP / "analysis/heatmaps"
SEED = 42
INPUT_RES = 256
ALPHA = 0.35
PANEL_SAMPLES = [1, 12, 24, 36, 48]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def case_dir(row: dict) -> Path:
    return RUNS / row["image_id"] / row["motion_group"] / f"seed_{SEED}"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def overlap(heat: np.ndarray, mask: np.ndarray) -> float:
    denom = float(np.sum(heat))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(heat * mask) / denom)


def main() -> None:
    approval = json.loads((SELECTION / "approval.json").read_text())
    manifest_path = SELECTION / "selection_manifest.json"
    manifest_sha = sha256(manifest_path)
    if not approval.get("approved") or approval.get("selection_manifest_sha256") != manifest_sha:
        raise RuntimeError("Selection approval is missing or its manifest hash no longer matches")

    encoder_out = HEAT / "encoder"
    encoder_out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(ROOT / "encoder_heatmap.py"),
        "--pair", f"{EXP / 'images/canonical_clean'}={EXP / 'images/watermarked'}",
        "--out-dir", str(encoder_out),
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Decoder gradient saliency requires the approved CUDA device")
    # Initialize CUDA before loading Decord/OpenCV's bundled multimedia libs.
    torch.cuda.init()
    import cv2
    import decord

    from PIL import Image
    from torchvision import transforms
    from gradient_saliency import compute_saliency, load_decoder, make_visualization

    device = torch.device("cuda:0")
    train_param_path = ROOT / "checkpoints/train_param.json"
    decoder = load_decoder(ROOT / "checkpoints/model-99.pth", train_param_path.parent, device)
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Resize((INPUT_RES, INPUT_RES)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    source_messages = np.load(RUNS / "source_message_map.npz", allow_pickle=True)
    messages = {key: np.asarray(source_messages[key]).reshape(-1).astype(np.float32)
                for key in source_messages.files if key != "ckpt"}
    raw_dir = encoder_out / "raw_maps/canonical_clean"
    per_frame_ba = {}
    with (EXP / "analysis/raw_ba/per_frame_accuracy.csv").open(encoding="utf-8") as stream:
        for item in csv.DictReader(stream):
            per_frame_ba[(item["case_id"], int(item["frame_idx"]))] = float(item["ba"])
    decode_summary = {}
    with (EXP / "analysis/raw_ba/case_summary.csv").open(encoding="utf-8") as stream:
        for item in csv.DictReader(stream):
            decode_summary[item["case_id"]] = item

    pair_rows = list(csv.DictReader((SELECTION / "vbench_40pairs.csv").open(encoding="utf-8")))
    metric_rows = []
    decoder_root = HEAT / "decoder"
    comparison_root = HEAT / "comparison_figures"
    decoder_root.mkdir(parents=True, exist_ok=True)
    comparison_root.mkdir(parents=True, exist_ok=True)
    for case_index, row in enumerate(pair_rows, start=1):
        directory = case_dir(row)
        video_path = directory / "video.mp4"
        status = json.loads((directory / "run_status.json").read_text()) if (directory / "run_status.json").exists() else {}
        if not video_path.is_file() or status.get("status") != "completed":
            continue
        source_id = row["image_id"]
        if source_id not in messages:
            raise KeyError(f"No source message for {source_id}")
        raw_encoder_path = raw_dir / f"{source_id}.npy"
        if not raw_encoder_path.is_file():
            raise FileNotFoundError(raw_encoder_path)
        encoder_map = np.load(raw_encoder_path).astype(np.float32)
        with Image.open(directory / "effective_input.png") as source_image:
            source_rgb = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
        enc_scale = float(np.quantile(encoder_map, 0.99)) or 1.0
        enc_display = np.clip(encoder_map / enc_scale, 0, 1)
        enc_colors = (matplotlib.colormaps["magma"](enc_display)[..., :3] * 255).astype(np.uint8)
        encoder_overlay = np.clip((1-ALPHA)*source_rgb.astype(np.float32) + ALPHA*enc_colors.astype(np.float32), 0, 255).astype(np.uint8)

        reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        frame_count = len(reader)
        # Attribution is computed independently on every frame. The fixed
        # representative subset is used only for composite comparison panels.
        if frame_count == 49:
            panel_indices = PANEL_SAMPLES if decode_summary.get(row["case_id"], {}).get("frame0_copy_like", "False").lower() == "true" else [0, 12, 24, 36, 48]
        else:
            panel_indices = sorted(set(int(x) for x in np.linspace(0, frame_count - 1, min(5, frame_count))))
        panel_indices = set(panel_indices)
        message = torch.as_tensor(messages[source_id], dtype=torch.float32, device=device).unsqueeze(0)
        case_out = decoder_root / row["case_id"]
        (case_out / "raw_gradients").mkdir(parents=True, exist_ok=True)
        (case_out / "overlays").mkdir(parents=True, exist_ok=True)
        fps = float(reader.get_avg_fps())
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid source FPS for {video_path}: {fps}")
        height, width = reader[0].asnumpy().shape[:2]
        overlay_path = case_out / "decoder_heatmap_overlay.mp4"
        writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open heatmap video writer: {overlay_path}")
        try:
            for frame_idx in range(frame_count):
                frame = reader[frame_idx].asnumpy()
                image_tensor = transform(Image.fromarray(frame).convert("RGB")).unsqueeze(0).to(device)
                _, saliency = compute_saliency(decoder, image_tensor, message, target_bit=-1)
                saliency = np.asarray(saliency, dtype=np.float32)
                np.save(case_out / "raw_gradients" / f"frame_{frame_idx:03d}.npy", saliency)
                saliency_up = cv2.resize(saliency, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                mask_path = directory / "stability/binary_masks" / f"{frame_idx:04d}.png"
                mask = (np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0).astype(np.float32)
                if mask.shape != frame.shape[:2]:
                    raise ValueError(f"Stable mask geometry mismatch at {row['case_id']} frame {frame_idx}")
                decoder_overlap = overlap(saliency_up, mask)
                srr = float(mask.mean())
                enrichment = decoder_overlap / srr if srr > 1e-12 and np.isfinite(decoder_overlap) else float("nan")
                encoder_overlap = overlap(encoder_map, mask)
                _, decoder_overlay = make_visualization(Image.fromarray(frame), saliency, alpha=ALPHA)
                overlay_rgb = np.asarray(decoder_overlay, dtype=np.uint8)
                writer.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
                decoder_overlay.save(case_out / "overlays" / f"frame_{frame_idx:03d}_gradient.png")
                if frame_idx in panel_indices:
                    stable_overlay = frame.copy()
                    green = np.zeros_like(frame)
                    green[..., 1] = 255
                    stable_overlay[mask.astype(bool)] = np.clip(
                        (1-ALPHA)*frame[mask.astype(bool)].astype(np.float32) + ALPHA*green[mask.astype(bool)].astype(np.float32), 0, 255
                    ).astype(np.uint8)
                    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
                    panels = [source_rgb, frame, stable_overlay, encoder_overlay, overlay_rgb]
                    titles = ["Watermarked input", f"Generated {row['case_id']} f{frame_idx}", "Stable patches (green)",
                              "Encoder residual overlay", "Decoder gradient overlay"]
                    for ax, panel, title in zip(axes, panels, titles):
                        ax.imshow(panel)
                        ax.set_title(title, fontsize=9)
                        ax.axis("off")
                    fig.suptitle(f"{row['case_id']} · frame {frame_idx} · seed {SEED}")
                    fig.tight_layout()
                    fig.savefig(comparison_root / f"{row['case_id']}_frame_{frame_idx:03d}.png", dpi=130)
                    plt.close(fig)
                metric_rows.append({
                    "image_id": source_id, "case_id": row["case_id"], "motion_full": row["motion_full"],
                    "seed": SEED, "frame_idx": frame_idx, "ba": per_frame_ba.get((row["case_id"], frame_idx), float("nan")),
                    "encoder_stable_overlap": encoder_overlap, "decoder_stable_overlap": decoder_overlap,
                    "srr": srr, "decoder_stable_enrichment": enrichment,
                    "decoder_objective": "mean ground-truth signed bit score",
                    "attribution_method": "input gradient; mean absolute gradient across RGB channels",
                    "attribution_normalization": "raw gradient magnitude saved; percentile 1-99 normalization used only for display",
                    "attribution_alpha": ALPHA,
                })
        finally:
            writer.release()
        print(f"[HEATMAP {case_index}/40] {row['case_id']}: rendered all {frame_count} frames to {overlay_path}", flush=True)

    metric_path = HEAT / "heatmap_metrics.csv"
    write_csv(metric_path, metric_rows,
              ["image_id", "case_id", "motion_full", "seed", "frame_idx", "ba", "encoder_stable_overlap",
               "decoder_stable_overlap", "srr", "decoder_stable_enrichment", "decoder_objective",
               "attribution_method", "attribution_normalization", "attribution_alpha"])
    for metric, filename, label in (("decoder_stable_overlap", "ba_vs_decoder_stable_overlap.png", "Decoder stable overlap"),
                                    ("decoder_stable_enrichment", "ba_vs_decoder_stable_enrichment.png", "Decoder stable enrichment")):
        values = [r for r in metric_rows if np.isfinite(r["ba"]) and np.isfinite(r[metric])]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter([r[metric] for r in values], [r["ba"] for r in values], alpha=.6, s=18)
        ax.set(xlabel=label, ylabel="Raw bit accuracy", title=f"Raw BA vs {label.lower()} (all frames)")
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(HEAT / filename, dpi=160)
        plt.close(fig)
    (HEAT / "heatmap_methods.json").write_text(json.dumps({
        "encoder": "Existing encoder_heatmap.py residual magnitude sqrt(mean((watermarked-clean)^2 over RGB)); raw magnitude map saved before display normalization.",
        "decoder": "Existing gradient_saliency.compute_saliency; mean ground-truth signed bit score; absolute input gradient averaged over RGB.",
        "decoder_frame_rule": "Every frame in each completed generated video; no temporal interpolation of attribution maps.",
        "display_alpha": ALPHA, "selection_manifest_sha256": manifest_sha,
        "normalization": "Raw numeric magnitudes are separate from percentile-normalized display maps.",
    }, indent=2) + "\n")
    print(f"Created per-frame decoder heatmaps and overlay videos for {len(metric_rows)} frames")


if __name__ == "__main__":
    main()
