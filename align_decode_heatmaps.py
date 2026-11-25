#!/usr/bin/env python3
"""Supplemental LoT-Pass RAFT alignment, aligned BA, and decoder heatmap videos."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
VENV_CACHE = ROOT / "vbench_lotpass_pilot/model_cache/torch"
os.environ.setdefault("TORCH_HOME", str(VENV_CACHE))
sys.path.insert(0, str(ROOT))

CHECKPOINT = ROOT / "checkpoints/model-99.pth"
INPUT_RES = 256
ALPHA = 0.35
SEED = 42


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def warp_to_ref(ref: np.ndarray, target: np.ndarray, flow: np.ndarray, cv2) -> np.ndarray:
    """Match test_I2V.decode_Videos_opticflow's target-to-reference remap."""
    height, width = ref.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    map_x = grid_x + flow[..., 0].astype(np.float32)
    map_y = grid_y + flow[..., 1].astype(np.float32)
    return cv2.remap(
        target, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def load_experiment(name: str):
    if name == "vbench":
        exp = ROOT / "vbench_lotpass_pilot"
        rows = read_csv(exp / "selection/vbench_40pairs.csv")
        messages_file = exp / "runs/source_message_map.npz"
        raw_file = exp / "analysis/raw_ba/per_frame_accuracy.csv"
        summary_file = exp / "analysis/raw_ba/case_summary.csv"
        messages_archive = np.load(messages_file, allow_pickle=True)
        messages = {key: np.asarray(messages_archive[key]).reshape(-1).astype(np.float32)
                    for key in messages_archive.files if key != "ckpt"}
        raw = {(r["case_id"], r["case_id"].rsplit("_", 1)[-1], int(r["frame_idx"])): float(r["ba"])
               for r in read_csv(raw_file)}
        summaries = {r["case_id"]: r for r in read_csv(summary_file)}
        cases = []
        for row in rows:
            cases.append({
                "experiment": name, "case_id": row["case_id"], "condition": row["motion_group"],
                "group": row["motion_group"], "source_id": row["image_id"], "seed": SEED,
                "video": exp / "runs" / row["image_id"] / row["motion_group"] / f"seed_{SEED}/video.mp4",
                "run_status": exp / "runs" / row["image_id"] / row["motion_group"] / f"seed_{SEED}/run_status.json",
                "output": exp / "analysis/optical_flow" / row["case_id"],
            })
    elif name == "tcbench":
        exp = ROOT / "tcbench_background_shifts"
        rows = read_csv(exp / "approval/tcbench_all_36pairs.csv")
        messages_file = exp / "runs/source_message_map.npz"
        raw_file = exp / "analysis/per_frame_raw_ba.csv"
        summary_file = exp / "analysis/per_video_raw_ba.csv"
        messages_archive = np.load(messages_file, allow_pickle=True)
        messages = {key: np.asarray(messages_archive[key]).reshape(-1).astype(np.float32)
                    for key in messages_archive.files if key != "ckpt"}
        raw = {(r["case_id"], r["condition"], int(r["frame_idx"])): float(r["ba"])
               for r in read_csv(raw_file)}
        summaries = {r["pair_id"]: r for r in read_csv(summary_file)}
        cases = []
        for row in rows:
            condition = row["condition"]
            cases.append({
                "experiment": name, "case_id": row["case_id"], "condition": condition,
                "group": condition, "source_id": row["case_id"], "seed": SEED,
                "video": exp / "generations" / row["case_id"] / condition / "video.mp4",
                "run_status": exp / "generations" / row["case_id"] / condition / "run_status.json",
                "output": exp / "analysis/optical_flow" / row["case_id"] / condition,
            })
    else:
        raise ValueError(name)
    for case in cases:
        case["message"] = messages[case["source_id"]]
        case["raw_lookup"] = raw
        if name == "vbench":
            case["raw_group"] = case["condition"]
            case["raw_summary"] = summaries.get(case["case_id"], {})
        else:
            case["raw_group"] = case["condition"]
            case["raw_summary"] = summaries.get(f"{case['case_id']}_{case['condition']}", {})
    return exp, cases


def render_dataset(name: str, raft, decoder, transform, torch, cv2, decord, imageio,
                   compute_saliency, make_visualization) -> tuple[list[dict], list[dict]]:
    exp, cases = load_experiment(name)
    frame_rows: list[dict] = []
    video_rows: list[dict] = []
    for case_index, case in enumerate(cases, start=1):
        video_path = case["video"]
        status = json.loads(case["run_status"].read_text()) if case["run_status"].is_file() else {}
        if not video_path.is_file() or status.get("status") != "completed":
            video_rows.append({"experiment": name, "case_id": case["case_id"], "condition": case["condition"],
                               "status": "skipped", "reason": "generated video missing or incomplete"})
            continue

        output = case["output"]
        gradients_dir = output / "raw_gradients"
        gradients_dir.mkdir(parents=True, exist_ok=True)
        aligned_final = output / "aligned_video.mp4"
        overlay_final = output / "decoder_heatmap_overlay.mp4"
        aligned_tmp = output / ".aligned_video.partial.mp4"
        overlay_tmp = output / ".decoder_heatmap_overlay.partial.mp4"
        reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        frame_count = len(reader)
        fps = float(reader.get_avg_fps())
        if frame_count == 0 or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid frame count/FPS for {video_path}: {frame_count}, {fps}")
        reference = reader[0].asnumpy()
        height, width = reference.shape[:2]
        message = torch.as_tensor(case["message"], dtype=torch.float32, device="cuda:0").unsqueeze(0)
        aligned_ba_values = []
        flow_dxs, flow_dys, flow_mags = [], [], []

        aligned_writer = imageio.get_writer(
            str(aligned_tmp), fps=fps, codec="libx264", quality=8, macro_block_size=1,
            ffmpeg_params=["-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        overlay_writer = imageio.get_writer(
            str(overlay_tmp), fps=fps, codec="libx264", quality=8, macro_block_size=1,
            ffmpeg_params=["-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        try:
            for frame_idx in range(frame_count):
                original = reader[frame_idx].asnumpy()
                if original.shape != reference.shape:
                    raise ValueError(f"Frame geometry changed in {video_path} at frame {frame_idx}")
                if frame_idx == 0:
                    aligned = original
                    dx = dy = magnitude = 0.0
                else:
                    flow = raft.compute_flow(reference, original)
                    aligned = warp_to_ref(reference, original, flow, cv2)
                    dx = float(np.median(flow[..., 0]))
                    dy = float(np.median(flow[..., 1]))
                    magnitude = float(np.median(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
                aligned_writer.append_data(aligned)

                image_tensor = transform(Image.fromarray(aligned).convert("RGB")).unsqueeze(0).to("cuda:0")
                decoded, saliency = compute_saliency(decoder, image_tensor, message, target_bit=-1)
                saliency = np.asarray(saliency, dtype=np.float32)
                np.save(gradients_dir / f"frame_{frame_idx:04d}.npy", saliency)
                prediction = decoded.detach().cpu().numpy().round().clip(0, 1)
                truth = message.detach().cpu().numpy()
                aligned_ba = float(1.0 - np.mean(np.abs(prediction - truth)))
                raw_ba = case["raw_lookup"].get((case["case_id"], case["raw_group"], frame_idx), float("nan"))

                _, overlay = make_visualization(Image.fromarray(aligned), saliency, alpha=ALPHA)
                overlay_writer.append_data(np.asarray(overlay, dtype=np.uint8))
                record = {
                    "experiment": name, "case_id": case["case_id"], "condition": case["condition"],
                    "group": case["group"], "seed": case["seed"], "frame_idx": frame_idx,
                    "fps": fps, "raw_BA": raw_ba, "aligned_BA": aligned_ba,
                    "aligned_minus_raw_BA": aligned_ba - raw_ba if np.isfinite(raw_ba) else float("nan"),
                    "median_flow_dx_px": dx, "median_flow_dy_px": dy,
                    "median_flow_magnitude_px": magnitude,
                    "alignment_reference_frame": 0, "alignment_method": "LoT-Pass TorchvisionRAFT; warp target with reference-to-target flow; cv2.remap INTER_LINEAR/BORDER_REPLICATE",
                }
                frame_rows.append(record)
                aligned_ba_values.append(aligned_ba)
                flow_dxs.append(dx); flow_dys.append(dy); flow_mags.append(magnitude)
        finally:
            aligned_writer.close()
            overlay_writer.close()
        os.replace(aligned_tmp, aligned_final)
        os.replace(overlay_tmp, overlay_final)

        raw_values = [r["raw_BA"] for r in frame_rows
                      if r["case_id"] == case["case_id"] and r["condition"] == case["condition"]]
        raw_mean = float(np.nanmean(raw_values)) if raw_values else float("nan")
        aligned_mean = float(np.mean(aligned_ba_values))
        copy_like = str(case["raw_summary"].get("frame0_copy_like", "false")).lower() == "true"
        raw_excl = raw_values[1:] if copy_like and len(raw_values) > 1 else raw_values
        aligned_excl = aligned_ba_values[1:] if copy_like and len(aligned_ba_values) > 1 else aligned_ba_values
        video_rows.append({
            "experiment": name, "case_id": case["case_id"], "condition": case["condition"],
            "group": case["group"], "seed": case["seed"], "status": "completed",
            "frame_count": frame_count, "fps": fps, "frame0_copy_like": copy_like,
            "raw_mean_BA_all_frames": raw_mean,
            "raw_mean_BA_excluding_copy_frame": float(np.nanmean(raw_excl)) if raw_excl else float("nan"),
            "aligned_mean_BA_all_frames": aligned_mean,
            "aligned_mean_BA_excluding_copy_frame": float(np.mean(aligned_excl)) if aligned_excl else float("nan"),
            "aligned_minus_raw_BA_all_frames": aligned_mean - raw_mean,
            "aligned_minus_raw_BA_excluding_copy_frame": (float(np.mean(aligned_excl)) - float(np.nanmean(raw_excl))) if raw_excl and np.isfinite(np.nanmean(raw_excl)) else float("nan"),
            "median_flow_dx_px": float(np.median(flow_dxs)), "median_flow_dy_px": float(np.median(flow_dys)),
            "median_flow_magnitude_px": float(np.median(flow_mags)),
            "aligned_video": str(aligned_final.relative_to(ROOT)),
            "decoder_heatmap_overlay_video": str(overlay_final.relative_to(ROOT)),
            "raw_gradients_dir": str(gradients_dir.relative_to(ROOT)),
        })
        (output / "alignment_record.json").write_text(json.dumps({
            "experiment": name, "case_id": case["case_id"], "condition": case["condition"],
            "source_video": str(video_path.relative_to(ROOT)), "source_video_frame_count": frame_count,
            "fps": fps, "resolution": [width, height], "reference_frame": 0,
            "method": "Existing test_I2V.decode_Videos_opticflow RAFT path: utils.utils.TorchvisionRAFT.compute_flow(reference, target); sample target at grid + flow; cv2.remap with INTER_LINEAR and BORDER_REPLICATE.",
            "decoder_checkpoint": str(CHECKPOINT.relative_to(ROOT)), "decoder_heatmap": "gradient_saliency.compute_saliency on every aligned frame; mean ground-truth signed bit score; mean absolute input gradient across RGB; alpha 0.35",
            "frame_order_preserved": True, "raw_source_left_unchanged": True,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"[{name} {case_index}/{len(cases)}] {case['case_id']}/{case['condition']}: aligned and heatmapped {frame_count} frames", flush=True)
    return frame_rows, video_rows


def summarize(name: str, frame_rows: list[dict], video_rows: list[dict], exp: Path) -> None:
    out = exp / "analysis/optical_flow"
    write_csv(out / "per_frame_aligned_ba.csv", frame_rows, list(frame_rows[0].keys()) if frame_rows else ["experiment"])
    write_csv(out / "per_video_aligned_ba.csv", video_rows, list(dict.fromkeys(k for r in video_rows for k in r)))

    completed = [r for r in video_rows if r.get("status") == "completed"]
    paired_values = [r for r in completed if np.isfinite(float(r.get("aligned_minus_raw_BA_excluding_copy_frame", "nan")))]
    overall_raw = float(np.mean([float(r["raw_mean_BA_excluding_copy_frame"]) for r in paired_values])) if paired_values else float("nan")
    overall_aligned = float(np.mean([float(r["aligned_mean_BA_excluding_copy_frame"]) for r in paired_values])) if paired_values else float("nan")
    overall_delta = float(np.mean([float(r["aligned_minus_raw_BA_excluding_copy_frame"]) for r in paired_values])) if paired_values else float("nan")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    for group in sorted({str(r["group"]) for r in paired_values}):
        subset = [r for r in paired_values if str(r["group"]) == group]
        ax.scatter([r["raw_mean_BA_excluding_copy_frame"] for r in subset],
                   [r["aligned_mean_BA_excluding_copy_frame"] for r in subset], label=group, alpha=.8)
    ax.plot([0, 1], [0, 1], color="0.4", linestyle="--", linewidth=1)
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="Raw mean BA (excluding copy frame)",
           ylabel="RAFT-aligned mean BA (excluding copy frame)",
           title=f"{name}: raw vs LoT-Pass RAFT-aligned per-video BA")
    ax.grid(alpha=.25); ax.legend(title="Condition", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "raw_vs_aligned_ba.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.get_cmap("tab10")
    for group_index, group in enumerate(sorted({str(r["group"]) for r in frame_rows})):
        subset = [r for r in frame_rows if str(r["group"]) == group]
        for metric, linestyle, label in (("raw_BA", "-", "raw"), ("aligned_BA", "--", "RAFT-aligned")):
            by_frame: dict[int, list[float]] = {}
            for row in subset:
                value = float(row[metric])
                if np.isfinite(value):
                    by_frame.setdefault(int(row["frame_idx"]), []).append(value)
            frames = sorted(by_frame)
            means = [float(np.mean(by_frame[index])) for index in frames]
            ax.plot(frames, means, color=colors(group_index % 10), linestyle=linestyle,
                    linewidth=2, label=f"{group}: {label}")
    ax.set(xlabel="Video frame", ylabel="Mean per-frame bit accuracy", ylim=(0, 1.02),
           title=f"{name}: raw and RAFT-aligned BA by frame (descriptive means)")
    ax.grid(alpha=.25); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out / "per_frame_raw_vs_aligned_ba.png", dpi=170); plt.close(fig)

    report = [
        f"# {name.upper()} supplemental LoT-Pass RAFT alignment", "",
        f"- Completed videos: {len(completed)}/{len(video_rows)}; aligned frame decodes: {len(frame_rows)}.",
        f"- Mean per-video BA excluding detected copy frames: raw={overall_raw:.4f}, aligned={overall_aligned:.4f}, mean paired change={overall_delta:+.4f} across {len(paired_values)} videos.",
        "- Alignment uses generated frame 0 as the fixed reference for every frame; it does not use the source image as the registration reference.",
        "- The raw videos and raw BA analyses remain unchanged. Aligned BA is supplemental and does not replace same-coordinate/raw results.",
        "- Each completed case has `aligned_video.mp4`, `decoder_heatmap_overlay.mp4`, per-frame raw gradient arrays, and an `alignment_record.json`.",
        "- RAFT displacement can be confounded by scene cuts, object movement, illumination changes, occlusion, or flow failure; alignment may introduce resampling artifacts.",
        "", "## Per-condition mean BA", "",
        "| Condition | Videos | Raw mean BA | Aligned mean BA | Change |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in sorted({str(r["group"]) for r in paired_values}):
        subset = [r for r in paired_values if str(r["group"]) == group]
        raw_group = float(np.mean([float(r["raw_mean_BA_excluding_copy_frame"]) for r in subset]))
        aligned_group = float(np.mean([float(r["aligned_mean_BA_excluding_copy_frame"]) for r in subset]))
        report.append(f"| {group} | {len(subset)} | {raw_group:.4f} | {aligned_group:.4f} | {aligned_group-raw_group:+.4f} |")
    report += ["", "Per-frame plots show descriptive group means; individual frames are repeated measurements within videos."]
    (out / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    metadata = {
        "alignment": "Existing LoT-Pass TorchvisionRAFT algorithm, flow computed from reference frame 0 to each target; target sampled at x+flow_x,y+flow_y using cv2.remap INTER_LINEAR/BORDER_REPLICATE.",
        "decoder": "Existing gradient_saliency.compute_saliency on every RAFT-aligned frame.",
        "display_alpha": ALPHA, "completed_videos": len(completed), "per_frame_rows": len(frame_rows),
        "raw_results_preserved": True,
    }
    (out / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[{name}] completed {len(completed)}/{len(video_rows)} videos; mean BA delta={overall_delta:+.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("all", "vbench", "tcbench"), default="all")
    parser.add_argument("--summarize-only", action="store_true", help="Rebuild summaries and plots from existing alignment CSVs without loading GPU models")
    args = parser.parse_args()
    names = ("vbench", "tcbench") if args.experiment == "all" else (args.experiment,)
    if args.summarize_only:
        for name in names:
            exp = ROOT / ("vbench_lotpass_pilot" if name == "vbench" else "tcbench_background_shifts")
            out = exp / "analysis/optical_flow"
            summarize(name, read_csv(out / "per_frame_aligned_ba.csv"),
                      read_csv(out / "per_video_aligned_ba.csv"), exp)
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU policy requires CUDA_VISIBLE_DEVICES=0")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.init()
    import cv2
    import decord
    import imageio.v2 as imageio
    from PIL import Image
    from torchvision import transforms
    from utils.utils import TorchvisionRAFT
    from gradient_saliency import compute_saliency, load_decoder, make_visualization

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    device = torch.device("cuda:0")
    decoder = load_decoder(CHECKPOINT, CHECKPOINT.parent, device)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    raft = TorchvisionRAFT(device="cuda:0")
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Resize((INPUT_RES, INPUT_RES)),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    for name in names:
        exp, _ = load_experiment(name)
        frames, videos = render_dataset(name, raft, decoder, transform, torch, cv2, decord, imageio,
                                        compute_saliency, make_visualization)
        summarize(name, frames, videos, exp)


if __name__ == "__main__":
    main()
