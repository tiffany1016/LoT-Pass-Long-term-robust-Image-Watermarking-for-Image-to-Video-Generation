#!/usr/bin/env python3
"""Decode every generated frame without optical-flow alignment."""
from __future__ import annotations

import csv
import hashlib
import json
import math
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
ANALYSIS = EXP / "analysis/raw_ba"
RUNS = EXP / "runs"
CHECKPOINT = ROOT / "checkpoints/model-99.pth"
SEED = 42
INPUT_RES = 256
COPY_MAE_MAX = 1.0 / 255.0
COPY_SSIM_MIN = 0.999


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
    # Decord's bundled FFmpeg libraries can conflict with CUDA driver
    # initialization if they are loaded first in this environment.
    torch.cuda.init()
    import decord

    from PIL import Image
    from torchvision import transforms
    from models import convnext
    from utils.train_param import TrainParam
    from skimage.metrics import structural_similarity

    device = torch.device("cuda:0")
    train_param = json.loads((CHECKPOINT.parent / "train_param.json").read_text())
    decoder = convnext.ConvNextDecoder(TrainParam(**train_param))
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    decoder.load_state_dict(checkpoint["decoder"])
    del checkpoint
    decoder.eval().to(device)
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Resize((INPUT_RES, INPUT_RES)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    messages_file = np.load(RUNS / "source_message_map.npz", allow_pickle=True)
    messages = {key: np.array(messages_file[key]).reshape(-1).astype(np.uint8)
                for key in messages_file.files if key != "ckpt"}

    pair_rows = list(csv.DictReader((SELECTION / "vbench_40pairs.csv").open(encoding="utf-8")))
    frame_rows, summary_rows = [], []
    output_by_image = defaultdict(dict)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    for case_index, row in enumerate(pair_rows, start=1):
        directory = case_dir(row)
        if row["image_id"] in messages:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "message.json").write_text(json.dumps({
                "case_id": row["case_id"], "image_id": row["image_id"],
                "message_bits": messages[row["image_id"]].astype(int).tolist(), "message_length": len(messages[row["image_id"]]),
                "reused_across_image_conditions": True,
            }, indent=2) + "\n")
        video_path = directory / "video.mp4"
        status_path = directory / "run_status.json"
        if not video_path.is_file():
            summary_rows.append({"image_id": row["image_id"], "case_id": row["case_id"],
                                 "motion_full": row["motion_full"], "seed": SEED,
                                 "status": "generation_failed_or_missing"})
            continue
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get("status") != "completed":
            summary_rows.append({"image_id": row["image_id"], "case_id": row["case_id"],
                                 "motion_full": row["motion_full"], "seed": SEED,
                                 "status": "generation_not_completed"})
            continue
        if row["image_id"] not in messages:
            raise KeyError(f"No source message for {row['image_id']}")
        message = torch.as_tensor(messages[row["image_id"]], dtype=torch.float32, device=device).unsqueeze(0)
        reference = np.asarray(Image.open(directory / "effective_input.png").convert("RGB"), dtype=np.uint8)
        reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        video_fps = float(reader.get_avg_fps())
        raw_ba, copied_flags, decoded_bits = [], [], []
        case_frame_rows = []
        for start in range(0, len(reader), 16):
            indices = list(range(start, min(start + 16, len(reader))))
            frames = [reader[i].asnumpy() for i in indices]
            batch = torch.stack([transform(Image.fromarray(frame).convert("RGB")) for frame in frames]).to(device)
            with torch.inference_mode():
                decoded = decoder(batch)
            bits = decoded.detach().cpu().numpy().round().clip(0, 1).astype(np.uint8)
            for frame_idx, frame, decoded_bit in zip(indices, frames, bits):
                gt = messages[row["image_id"]]
                ba = float(1.0 - np.mean(np.abs(decoded_bit.reshape(-1).astype(np.float32) - gt)))
                raw_ba.append(ba)
                decoded_bits.append("".join(map(str, decoded_bit.reshape(-1).astype(int).tolist())))
                if frame_idx == 0 and frame.shape == reference.shape:
                    mae = float(np.mean(np.abs(frame.astype(np.float32) - reference.astype(np.float32))) / 255.0)
                    ssim = float(structural_similarity(reference, frame, channel_axis=2, data_range=255))
                    copied = mae < COPY_MAE_MAX and ssim > COPY_SSIM_MIN
                else:
                    mae, ssim, copied = float("nan"), float("nan"), False
                copied_flags.append(copied)
                frame_record = {
                    "image_id": row["image_id"], "case_id": row["case_id"], "motion_full": row["motion_full"],
                    "seed": SEED, "frame_idx": frame_idx, "fps": video_fps, "ba": ba,
                    "decoded_bits": decoded_bits[-1], "frame0_copy_mae": mae,
                    "frame0_copy_ssim": ssim, "frame0_copy_like": copied,
                }
                frame_rows.append(frame_record)
                case_frame_rows.append(frame_record)
        n = len(raw_ba)
        split = math.ceil(2 * n / 3)
        first_end = max(1, math.ceil(n / 3))
        copy_like = bool(copied_flags[0]) if copied_flags else False
        included = np.asarray(raw_ba, dtype=np.float64)
        excluded = included[1:] if copy_like and n > 1 else included
        last_all = included[split:]
        last_excluded = excluded[max(0, split - (1 if copy_like else 0)):]
        stats = {
            "image_id": row["image_id"], "case_id": row["case_id"], "motion_full": row["motion_full"],
            "seed": SEED, "status": "completed", "frame_count": n, "fps": video_fps,
            "frame0_copy_like": copy_like,
            "mean_ba_all_frames": float(np.mean(included)),
            "mean_ba_excluding_copy_frame": float(np.mean(excluded)),
            "last_third_mean_ba_all_frames": float(np.mean(last_all)) if len(last_all) else float("nan"),
            "last_third_mean_ba_excluding_copy_frame": float(np.mean(last_excluded)) if len(last_excluded) else float("nan"),
            "minimum_ba_all_frames": float(np.min(included)),
            "first_third_mean_ba": float(np.mean(included[:first_end])),
            "last_third_minus_first_third_ba": float(np.mean(last_all) - np.mean(included[:first_end])) if len(last_all) else float("nan"),
            "video_sha256": sha256(video_path),
        }
        summary_rows.append(stats)
        output_by_image[row["image_id"]][row["motion_group"]] = (row, np.asarray(raw_ba, dtype=np.float32))
        write_csv(directory / "raw_ba.csv", case_frame_rows,
                  ["image_id", "case_id", "motion_full", "seed", "frame_idx", "fps", "ba", "decoded_bits",
                   "frame0_copy_mae", "frame0_copy_ssim", "frame0_copy_like"])
        np.savez_compressed(directory / "decoded_bits.npz", frame_indices=np.arange(n), bits=np.asarray(decoded_bits, dtype="U64"))
        atomic_json(directory / "decode_status.json", {"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
                                                        "frames": n, "raw_alignment": "none", "checkpoint_sha256": sha256(CHECKPOINT)})
        print(f"[{case_index}/40] {row['case_id']}: frames={n}; mean BA={stats['mean_ba_all_frames']:.4f}; copy0={copy_like}", flush=True)

    write_csv(ANALYSIS / "per_frame_accuracy.csv", frame_rows,
              ["image_id", "case_id", "motion_full", "seed", "frame_idx", "fps", "ba", "decoded_bits",
               "frame0_copy_mae", "frame0_copy_ssim", "frame0_copy_like"])
    write_csv(ANALYSIS / "case_summary.csv", summary_rows,
              ["image_id", "case_id", "motion_full", "seed", "status", "frame_count", "fps", "frame0_copy_like",
               "mean_ba_all_frames", "mean_ba_excluding_copy_frame", "last_third_mean_ba_all_frames",
               "last_third_mean_ba_excluding_copy_frame", "minimum_ba_all_frames", "first_third_mean_ba",
               "last_third_minus_first_third_ba", "video_sha256"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = ANALYSIS / "per_image"
    plot_dir.mkdir(parents=True, exist_ok=True)
    color_map = {"static": "black", "pan": "tab:blue", "tilt": "tab:orange", "zoom": "tab:green"}
    for image_id, conditions in output_by_image.items():
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for condition in ("static", "pan", "tilt", "zoom"):
            if condition not in conditions:
                continue
            row, values = conditions[condition]
            ax.plot(np.arange(len(values)), values, label=f"{condition} ({row['motion_full']})", color=color_map[condition])
        ax.set(xlabel="Actual frame index", ylabel="Raw bit accuracy", ylim=(0, 1.02), title=f"{image_id}: per-frame raw BA, seed {SEED}")
        ax.grid(alpha=.25)
        ax.legend(loc="lower left", ncol=2)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{image_id}_ba.png", dpi=160)
        plt.close(fig)
    (ANALYSIS / "decode_metadata.json").write_text(json.dumps({
        "selection_manifest_sha256": manifest_sha, "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(CHECKPOINT), "raw_frames_only": True,
        "optical_flow_alignment": False, "copy_like_rule": {"frame": 0, "mean_rgb_mae_div_255_lt": COPY_MAE_MAX,
                                                                 "ssim_gt": COPY_SSIM_MIN},
        "n_cases_with_raw_decode": len(output_by_image), "n_per_frame_rows": len(frame_rows),
    }, indent=2) + "\n")
    print(f"Decoded {len(output_by_image)} cases / {len(frame_rows)} frames; output: {ANALYSIS}")


if __name__ == "__main__":
    main()
