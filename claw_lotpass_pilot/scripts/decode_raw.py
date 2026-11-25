#!/usr/bin/env python3
"""Decode every generated CLAW frame without optical-flow alignment."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "claw_lotpass_pilot"
RUNS = EXP / "runs"
ANALYSIS = EXP / "analysis/raw_ba"
CHECKPOINT = ROOT / "checkpoints/model-99.pth"
INPUT_RES = 256
COPY_MAE_MAX, COPY_SSIM_MIN = 1 / 255, 0.999
CONDITIONS = ("static_control", "broad_fluid_motion")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU policy requires CUDA_VISIBLE_DEVICES=0")
    manifest = json.loads((EXP / "selection/generation_manifest.json").read_text())
    if not json.loads((EXP / "selection/approval.json").read_text()).get("approved"):
        raise RuntimeError("The Stage A package is not approved")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.init()
    import decord
    from PIL import Image
    from skimage.metrics import structural_similarity
    from torchvision import transforms
    from models import convnext
    from utils.train_param import TrainParam

    device = torch.device("cuda:0")
    train_param = json.loads((CHECKPOINT.parent / "train_param.json").read_text())
    decoder = convnext.ConvNextDecoder(TrainParam(**train_param))
    state = torch.load(CHECKPOINT, map_location="cpu")
    decoder.load_state_dict(state["decoder"]); del state
    decoder.eval().to(device)
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Resize((INPUT_RES, INPUT_RES)),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    archive = np.load(RUNS / "source_message_map.npz", allow_pickle=True)
    messages = {key: np.asarray(archive[key]).reshape(-1).astype(np.uint8)
                for key in archive.files if key != "ckpt"}
    frame_rows, summary_rows, curves = [], [], {}
    for case_index, row in enumerate(manifest["cases"], start=1):
        directory = RUNS / row["case_id"] / f"seed_{row['seed']}"
        video = directory / "video.mp4"
        status_path = directory / "run_status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        common = {"case_id": row["case_id"], "image_id": row["image_id"], "sample_index": row["sample_index"],
                  "condition": row["condition"], "seed": row["seed"]}
        if status.get("status") != "completed" or not video.is_file():
            summary_rows.append({**common, "status": f"generation_{status.get('status', 'missing')}"})
            continue
        if row["image_id"] not in messages:
            raise KeyError(f"No source watermark message for {row['image_id']}")
        truth = messages[row["image_id"]]
        gt = torch.as_tensor(truth, dtype=torch.float32, device=device).unsqueeze(0)
        reference = np.asarray(Image.open(directory / "effective_input.png").convert("RGB"), dtype=np.uint8)
        reader = decord.VideoReader(str(video), ctx=decord.cpu(0))
        fps = float(reader.get_avg_fps())
        case_rows, values, copy_flags = [], [], []
        for start in range(0, len(reader), 16):
            indices = list(range(start, min(start + 16, len(reader))))
            frames = [reader[i].asnumpy() for i in indices]
            batch = torch.stack([transform(Image.fromarray(frame).convert("RGB")) for frame in frames]).to(device)
            with torch.inference_mode():
                decoded = decoder(batch).detach().cpu().numpy().round().clip(0, 1).astype(np.uint8)
            for idx, frame, bits in zip(indices, frames, decoded):
                ba = float(1 - np.mean(np.abs(bits.reshape(-1).astype(np.float32) - truth.astype(np.float32))))
                if idx == 0 and frame.shape == reference.shape:
                    mae = float(np.mean(np.abs(frame.astype(np.float32) - reference.astype(np.float32))) / 255.0)
                    ssim = float(structural_similarity(reference, frame, channel_axis=2, data_range=255))
                    copy_like = mae < COPY_MAE_MAX and ssim > COPY_SSIM_MIN
                else:
                    mae = ssim = float("nan"); copy_like = False
                item = {**common, "frame_idx": idx, "fps": fps, "ba": ba,
                        "decoded_bits": "".join(str(int(b)) for b in bits.reshape(-1)),
                        "frame0_copy_mae": mae, "frame0_copy_ssim": ssim, "frame0_copy_like": copy_like}
                values.append(ba); copy_flags.append(copy_like); frame_rows.append(item); case_rows.append(item)
        if not values:
            summary_rows.append({**common, "status": "empty_video"})
            continue
        arr = np.asarray(values, dtype=np.float64)
        split = math.ceil(2 * len(arr) / 3)
        first_end = max(1, math.ceil(len(arr) / 3))
        excluded = arr[1:] if copy_flags[0] and len(arr) > 1 else arr
        last_all = arr[split:]
        last_excluded = excluded[max(0, split - (1 if copy_flags[0] else 0)):]
        summary_rows.append({
            **common, "status": "completed", "frame_count": len(arr), "fps": fps,
            "frame0_copy_like": copy_flags[0], "mean_raw_ba": float(arr.mean()),
            "mean_raw_ba_excluding_copy_frame": float(excluded.mean()),
            "minimum_raw_ba": float(arr.min()),
            "last_third_raw_ba": float(last_all.mean()) if len(last_all) else float("nan"),
            "last_third_raw_ba_excluding_copy_frame": float(last_excluded.mean()) if len(last_excluded) else float("nan"),
            "first_third_raw_ba": float(arr[:first_end].mean()),
            "last_minus_first_third_ba": float(last_all.mean() - arr[:first_end].mean()) if len(last_all) else float("nan"),
            "video_sha256": sha256(video),
        })
        curves[row["case_id"]] = arr
        write_csv(directory / "raw_ba.csv", case_rows, list(case_rows[0]))
        print(f"Decoded {row['case_id']}: {len(arr)} frames; mean raw BA={arr.mean():.4f}", flush=True)

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    write_csv(ANALYSIS / "per_frame_accuracy.csv", frame_rows,
              ["case_id", "image_id", "sample_index", "condition", "seed", "frame_idx", "fps", "ba",
               "decoded_bits", "frame0_copy_mae", "frame0_copy_ssim", "frame0_copy_like"])
    write_csv(ANALYSIS / "case_summary.csv", summary_rows,
              list(dict.fromkeys(key for row in summary_rows for key in row)))
    valid = [r for r in summary_rows if r.get("status") == "completed"]
    by_condition = {}
    for condition in CONDITIONS:
        vals = [float(r["mean_raw_ba_excluding_copy_frame"]) for r in valid if r["condition"] == condition]
        by_condition[condition] = {"video_count": len(vals), "mean_ba": float(np.mean(vals)) if vals else None}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"static_control": "#315b7d", "broad_fluid_motion": "#d77732"}
    plot_dir = ANALYSIS / "per_image"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for row in manifest["selected_images"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        for case in manifest["cases"]:
            if case["image_id"] != row["image_id"] or case["case_id"] not in curves:
                continue
            ax.plot(np.arange(len(curves[case["case_id"]])), curves[case["case_id"]],
                    color=colors[case["condition"]], label=case["condition"])
        ax.set(xlabel="Frame index", ylabel="Raw bit accuracy", ylim=(0, 1.02), title=f"{row['image_id']} · raw ACC")
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(plot_dir / f"{row['sample_index']:02d}_raw_acc.png", dpi=140)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for condition in CONDITIONS:
        rows = [r for r in valid if r["condition"] == condition and r["case_id"] in curves]
        if not rows:
            continue
        matrix = np.stack([curves[r["case_id"]] for r in rows])
        ax.plot(np.arange(matrix.shape[1]), matrix.mean(axis=0), color=colors[condition], label=condition)
        ax.fill_between(np.arange(matrix.shape[1]), matrix.mean(axis=0)-matrix.std(axis=0),
                        matrix.mean(axis=0)+matrix.std(axis=0), color=colors[condition], alpha=.18)
    ax.set(xlabel="Frame index", ylabel="Raw bit accuracy", ylim=(0, 1.02), title="Raw ACC by prompt condition")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(ANALYSIS / "raw_acc_by_condition.png", dpi=160); plt.close(fig)
    (ANALYSIS / "summary.json").write_text(json.dumps({
        "generation_manifest_sha256": hashlib.sha256((EXP / "selection/generation_manifest.json").read_bytes()).hexdigest(),
        "checkpoint_sha256": sha256(CHECKPOINT), "decoded_case_count": len(valid),
        "frame_row_count": len(frame_rows), "condition_means_excluding_copy_frame": by_condition,
        "copy_frame_rule": {"frame_index": 0, "rgb_mae_div_255_lt": COPY_MAE_MAX, "ssim_gt": COPY_SSIM_MIN},
    }, indent=2) + "\n")
    print(f"Raw decoding completed for {len(valid)} videos and {len(frame_rows)} frames.", flush=True)


if __name__ == "__main__":
    main()
