#!/usr/bin/env python3
"""Decode every generated frame with the existing LoT-Pass raw decoder."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "tcbench_background_shifts"
GENERATIONS = EXP / "generations"
ANALYSIS = EXP / "analysis"
CHECKPOINT = ROOT / "checkpoints/model-99.pth"
INPUT_RES = 256
COPY_MAE_MAX = 1.0 / 255.0
COPY_SSIM_MIN = 0.999


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU policy: launch with CUDA_VISIBLE_DEVICES=0")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.init()
    import decord
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity
    from torchvision import transforms
    from models import convnext
    from utils.train_param import TrainParam

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    encoding = json.loads((EXP / "runs/encoding_summary.json").read_text())
    message_path = EXP / "runs/source_message_map.npz"
    archive = np.load(message_path, allow_pickle=True)
    messages = {key: np.asarray(archive[key]).reshape(-1).astype(np.uint8)
                for key in archive.files if key != "ckpt"}
    with (EXP / "approval/tcbench_all_36pairs.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    device = torch.device("cuda:0")
    params = TrainParam(**json.loads((CHECKPOINT.parent / "train_param.json").read_text()))
    decoder = convnext.ConvNextDecoder(params)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    decoder.load_state_dict(checkpoint["decoder"])
    del checkpoint
    decoder.eval().to(device)
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Resize((INPUT_RES, INPUT_RES)),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    per_frame, per_video = [], []
    for index, row in enumerate(rows, 1):
        case_id, condition = row["case_id"], row["condition"]
        directory = GENERATIONS / case_id / condition
        video_path = directory / "video.mp4"
        status_path = directory / "run_status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        common = {"pair_id": row["pair_id"], "case_id": case_id, "official_id": int(row["official_id"]),
                  "condition": condition, "seed": 42, "prompt_used": row["prompt_used"],
                  "generation_status": status.get("status", "missing"),
                  "video_path": str(video_path.relative_to(ROOT)) if video_path.is_file() else ""}
        if not video_path.is_file() or status.get("status") != "completed":
            per_video.append({**common, "decode_status": "generation_failed_or_missing"})
            continue
        decode_status_path = directory / "decode_status.json"
        if decode_status_path.is_file():
            old = json.loads(decode_status_path.read_text())
            if old.get("status") == "completed" and old.get("video_sha256") == sha256(video_path):
                # Still read the CSV records below so a resumed run can rebuild aggregate outputs.
                per_frame_file = directory / "per_frame_raw_ba.csv"
                per_video_file = directory / "raw_ba_summary.json"
                if per_frame_file.is_file() and per_video_file.is_file():
                    per_frame.extend(list(csv.DictReader(per_frame_file.open(encoding="utf-8"))))
                    per_video.append({**common, **json.loads(per_video_file.read_text())})
                    continue
        try:
            reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
            source_path = EXP / "encoded" / case_id / "watermarked.png"
            source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.uint8)
            raw_ba, frame_rows = [], []
            for start in range(0, len(reader), 16):
                indices = list(range(start, min(start + 16, len(reader))))
                frames = [reader[i].asnumpy() for i in indices]
                batch = torch.stack([transform(Image.fromarray(frame).convert("RGB")) for frame in frames]).to(device)
                with torch.inference_mode():
                    decoded = decoder(batch).detach().cpu().numpy().round().clip(0, 1).astype(np.uint8)
                for frame_idx, frame, bits in zip(indices, frames, decoded):
                    target = messages[case_id]
                    ba = float(1.0 - np.mean(np.abs(bits.reshape(-1).astype(np.float32) - target)))
                    if frame_idx == 0 and frame.shape == source.shape:
                        mae = float(np.mean(np.abs(frame.astype(np.float32)-source.astype(np.float32))) / 255.0)
                        ssim = float(structural_similarity(source, frame, channel_axis=2, data_range=255))
                        copied = mae < COPY_MAE_MAX and ssim > COPY_SSIM_MIN
                    else:
                        mae, ssim, copied = float("nan"), float("nan"), False
                    item = {**common, "frame_idx": frame_idx, "fps": float(reader.get_avg_fps()),
                            "ba": ba, "decoded_bits": "".join(map(str, bits.reshape(-1).astype(int).tolist())),
                            "frame0_copy_mae": mae, "frame0_copy_ssim": ssim, "frame0_copy_like": copied}
                    frame_rows.append(item)
                    per_frame.append(item)
                    raw_ba.append(ba)
            if not raw_ba:
                raise RuntimeError("Decoder returned no video frames")
            n = len(raw_ba)
            split = math.ceil(2 * n / 3)
            first_end = max(1, math.ceil(n / 3))
            copy_like = bool(frame_rows[0]["frame0_copy_like"])
            included = np.asarray(raw_ba, dtype=np.float64)
            excluded = included[1:] if copy_like and n > 1 else included
            last = included[split:]
            last_excluded = excluded[max(0, split - (1 if copy_like else 0)):]
            summary = {
                "decode_status": "completed", "frame_count": n, "fps": float(reader.get_avg_fps()),
                "frame0_copy_like": copy_like,
                "mean_ba_all_frames": float(np.mean(included)),
                "mean_ba_excluding_copy_frame": float(np.mean(excluded)),
                "last_third_mean_ba_all_frames": float(np.mean(last)) if len(last) else float("nan"),
                "last_third_mean_ba_excluding_copy_frame": float(np.mean(last_excluded)) if len(last_excluded) else float("nan"),
                "minimum_ba_all_frames": float(np.min(included)),
                "first_third_mean_ba": float(np.mean(included[:first_end])),
                "early_to_late_ba_drop": float(np.mean(included[:first_end])-np.mean(last)) if len(last) else float("nan"),
                "video_sha256": sha256(video_path), "source_message_sha256": encoding["source_message_map_sha256"],
                "checkpoint_sha256": encoding["checkpoint_sha256"], "raw_alignment": "none; no optical flow",
            }
            write_csv(directory / "per_frame_raw_ba.csv", frame_rows,
                      list(frame_rows[0].keys()))
            atomic_json(directory / "raw_ba_summary.json", summary)
            atomic_json(decode_status_path, {"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
                                             "video_sha256": summary["video_sha256"], "frame_count": n,
                                             "checkpoint_sha256": encoding["checkpoint_sha256"],
                                             "raw_alignment": "none; no optical flow"})
            per_video.append({**common, **summary})
        except Exception as exc:
            failure = {**common, "decode_status": "failed", "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()}
            per_video.append(failure)
            atomic_json(decode_status_path, failure)
            print(f"[DECODE_ERROR] {row['pair_id']}: {type(exc).__name__}: {exc}", flush=True)

    frame_fields = list(dict.fromkeys(key for row in per_frame for key in row))
    video_fields = list(dict.fromkeys(key for row in per_video for key in row))
    write_csv(ANALYSIS / "per_frame_raw_ba.csv", per_frame, frame_fields)
    write_csv(ANALYSIS / "per_video_raw_ba.csv", per_video, video_fields)
    print(f"Decoded {sum(r.get('decode_status')=='completed' for r in per_video)}/36 videos; {len(per_frame)} frames.", flush=True)


if __name__ == "__main__":
    main()
