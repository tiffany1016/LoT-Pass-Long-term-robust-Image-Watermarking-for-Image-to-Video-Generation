#!/usr/bin/env python3
"""Decode every frame of every completed ChronoMagic video without alignment."""
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

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "chronomagic150_global_appearance"
DATA = EXP / "dataset"
RUNS = EXP / "runs"
VIDEOS = EXP / "generated_videos"
WATERMARKED = EXP / "watermarked_sources"
OUT = EXP / "decoding"
CHECKPOINT = ROOT / "checkpoints/model-99.pth"
INPUT_RES = 256
COPY_MAE_MAX, COPY_SSIM_MIN = 1.0 / 255.0, 0.999


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


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU policy requires CUDA_VISIBLE_DEVICES=0")
    approval = json.loads((EXP / "stage_a_review/approval_record.json").read_text())
    if not approval.get("approved"):
        raise RuntimeError("ChronoMagic Stage A is not approved")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in configured uv environment")
    torch.cuda.init()
    import decord
    from PIL import Image
    from skimage.metrics import structural_similarity
    from torchvision import transforms
    from models import convnext
    from utils.train_param import TrainParam

    encoding = json.loads((WATERMARKED / "encoding_summary.json").read_text())
    with np.load(WATERMARKED / "source_message_map.npz", allow_pickle=True) as archive:
        messages = {key: np.asarray(archive[key]).reshape(-1).astype(np.uint8)
                    for key in archive.files if key != "ckpt"}
    with (DATA / "chronomagic150_generation_manifest.csv").open(encoding="utf-8", newline="") as stream:
        cases = list(csv.DictReader(stream))

    device = torch.device("cuda:0")
    train_param = json.loads((CHECKPOINT.parent / "train_param.json").read_text())
    decoder = convnext.ConvNextDecoder(TrainParam(**train_param))
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    decoder.load_state_dict(checkpoint["decoder"])
    del checkpoint
    decoder.eval().to(device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((INPUT_RES, INPUT_RES)),
                                    transforms.Normalize([0.5] * 3, [0.5] * 3)])
    OUT.mkdir(parents=True, exist_ok=True)
    all_frame_rows, all_video_rows = [], []
    for case_index, row in enumerate(cases, 1):
        sample_id, condition = row["sample_id"], row["condition"]
        video_path = VIDEOS / sample_id / f"{condition}.mp4"
        run_dir = RUNS / sample_id / condition
        run_dir.mkdir(parents=True, exist_ok=True)
        status_path = run_dir / "decode_status.json"
        run_status_path = run_dir / "run_status.json"
        run_status = json.loads(run_status_path.read_text()) if run_status_path.is_file() else {}
        common = {"case_id": row["case_id"], "sample_id": sample_id,
                  "selection_index": row["selection_index"], "analysis_group": row["analysis_group"],
                  "condition": condition, "seed": int(row["seed"]), "prompt": row["prompt"],
                  "generation_status": run_status.get("status", "missing"),
                  "video_path": str(video_path.relative_to(ROOT)) if video_path.is_file() else ""}
        if run_status.get("status") != "completed" or not video_path.is_file():
            all_video_rows.append({**common, "decode_status": "generation_failed_or_missing"})
            continue
        video_hash = sha256(video_path)
        per_frame_path, summary_path = run_dir / "per_frame_raw_ba.csv", run_dir / "raw_ba_summary.json"
        if status_path.is_file():
            old = json.loads(status_path.read_text())
            if old.get("status") == "completed" and old.get("video_sha256") == video_hash and per_frame_path.is_file() and summary_path.is_file():
                existing = list(csv.DictReader(per_frame_path.open(encoding="utf-8")))
                all_frame_rows.extend(existing)
                all_video_rows.append({**common, **json.loads(summary_path.read_text())})
                continue
        try:
            if sample_id not in messages:
                raise KeyError(f"No encoded payload for {sample_id}")
            reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
            fps = float(reader.get_avg_fps())
            source = np.asarray(Image.open(WATERMARKED / sample_id / "watermarked.png").convert("RGB"), dtype=np.uint8)
            target = messages[sample_id]
            gt = torch.as_tensor(target, dtype=torch.float32, device=device).unsqueeze(0)
            case_rows, values = [], []
            for start in range(0, len(reader), 16):
                indices = list(range(start, min(start + 16, len(reader))))
                frames = [reader[i].asnumpy() for i in indices]
                batch = torch.stack([transform(Image.fromarray(frame).convert("RGB")) for frame in frames]).to(device)
                with torch.inference_mode():
                    decoded = decoder(batch).detach().cpu().numpy().round().clip(0, 1).astype(np.uint8)
                for frame_idx, frame, bits in zip(indices, frames, decoded):
                    ba = float(1.0 - np.mean(np.abs(bits.reshape(-1).astype(np.float32) - target.astype(np.float32))))
                    if frame_idx == 0 and frame.shape == source.shape:
                        mae = float(np.mean(np.abs(frame.astype(np.float32) - source.astype(np.float32))) / 255.0)
                        ssim = float(structural_similarity(source, frame, channel_axis=2, data_range=255))
                        copied = mae < COPY_MAE_MAX and ssim > COPY_SSIM_MIN
                    else:
                        mae, ssim, copied = float("nan"), float("nan"), False
                    metric = {**common, "frame_idx": frame_idx, "fps": fps, "raw_bit_accuracy": ba,
                              "decoded_bits": "".join(str(int(b)) for b in bits.reshape(-1)),
                              "frame0_copy_mae": mae, "frame0_copy_ssim": ssim, "frame0_copy_like": copied}
                    case_rows.append(metric); values.append(ba); all_frame_rows.append(metric)
            if not values:
                raise RuntimeError("Decoder returned no frames")
            count = len(values)
            split = math.ceil(2 * count / 3)
            last = values[split:]
            frame0_copy = bool(case_rows[0]["frame0_copy_like"])
            included = values[1:] if frame0_copy and count > 1 else values
            if frame0_copy and split == 1 and len(values) > 1:
                last = values[split:]
            summary = {
                "decode_status": "completed", "frame_count": count, "fps": fps,
                "frame0_copy_like": frame0_copy,
                "mean_ba_all_frames": float(np.mean(values)),
                "mean_ba_excluding_copy_frame": float(np.mean(included)),
                "minimum_ba_all_frames": float(np.min(values)),
                "last_third_mean_ba_all_frames": float(np.mean(last)) if last else float("nan"),
                "last_third_mean_ba_excluding_copy_frame": float(np.mean(last)) if last else float("nan"),
                "video_sha256": video_hash, "checkpoint_sha256": encoding["checkpoint_sha256"],
                "raw_alignment": "none; no optical-flow alignment",
            }
            write_csv(per_frame_path, case_rows, list(case_rows[0]))
            atomic_json(summary_path, summary)
            atomic_json(status_path, {"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
                                      "video_sha256": video_hash, "frame_count": count,
                                      "raw_alignment": "none; no optical-flow alignment"})
            all_video_rows.append({**common, **summary})
        except Exception as exc:
            failure = {**common, "decode_status": "failed", "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()}
            all_video_rows.append(failure)
            atomic_json(status_path, failure)
            print(f"[DECODE_ERROR] {row['case_id']} | {type(exc).__name__}: {exc}", flush=True)

    write_csv(OUT / "per_frame_raw_ba.csv", all_frame_rows, list(dict.fromkeys(k for r in all_frame_rows for k in r)))
    write_csv(OUT / "per_video_raw_ba.csv", all_video_rows, list(dict.fromkeys(k for r in all_video_rows for k in r)))
    completed = sum(r.get("decode_status") == "completed" for r in all_video_rows)
    print(f"Decoded {completed}/{len(cases)} videos, {len(all_frame_rows)} frame records.", flush=True)


if __name__ == "__main__":
    main()
