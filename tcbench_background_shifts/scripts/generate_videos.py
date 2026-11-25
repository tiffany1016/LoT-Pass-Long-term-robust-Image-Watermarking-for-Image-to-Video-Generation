#!/usr/bin/env python3
"""Generate approved TC-Bench transition/control pairs with CogVideoX-I2V."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "tcbench_background_shifts"
PAIRS = EXP / "approval/tcbench_all_36pairs.csv"
ENCODED = EXP / "encoded"
GENERATIONS = EXP / "generations"
RUNS = EXP / "runs"
CHECKPOINT = ROOT / "checkpoints/model-99.pth"
MODEL_ID = "THUDM/CogVideoX-5b-I2V"
SEED, FRAMES, FPS = 42, 49, 8
WIDTH, HEIGHT = 720, 480
GUIDANCE, STEPS, QUALITY = 5.0, 50, 10
NEGATIVE_PROMPT = (
    "Vibrant colors, overexposed, static, blurry details, subtitles, style, artwork, painting, scene, static, "
    "overall grayish, poor quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, static scene, "
    "cluttered background, three legs, many people in the background, walking backward"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def inspect_video(path: Path) -> dict:
    import av
    count, width, height, fps = 0, None, None, None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        width, height = stream.width, stream.height
        fps = float(stream.average_rate) if stream.average_rate else None
        for _ in container.decode(stream):
            count += 1
    return {"frame_count": count, "width": width, "height": height, "fps": fps}


def case_dir(row: dict) -> Path:
    return GENERATIONS / row["case_id"] / row["condition"]


def config_for(row: dict, message: list[int], input_path: Path) -> dict:
    return {
        "study_name": "TC-Bench-derived first-frame I2V evaluation of all official background-shifts entries",
        "case_id": row["case_id"], "official_id": int(row["official_id"]),
        "condition": row["condition"], "is_official_prompt": row["is_official_prompt"].lower() == "true",
        "prompt": row["prompt_used"], "prompt_official": row["prompt_official"],
        "prompt_control": row["prompt_control"], "model_id": MODEL_ID,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "checkpoint_sha256": sha256(CHECKPOINT),
        "seed": SEED, "message_bits": message, "message_length": len(message),
        "guidance_scale": GUIDANCE, "inference_steps": STEPS, "num_frames": FRAMES,
        "fps": FPS, "width": WIDTH, "height": HEIGHT, "video_quality": QUALITY,
        "negative_prompt": NEGATIVE_PROMPT,
        "watermarked_input": str(input_path.relative_to(ROOT)), "input_sha256": sha256(input_path),
        "input_preprocessing": "approved 720x480 center-cropped first frame; LoT-Pass encoded once per official source; no further geometry transform",
        "reference_video_url": row["reference_video_url"],
        "reference_video_id": row["reference_video_id"],
        "reference_interval_seconds": [float(row["reference_start_sec"]), float(row["reference_end_sec"])],
    }


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU policy: launch with CUDA_VISIBLE_DEVICES=0")
    if not (EXP / "approval/approval_record.md").is_file():
        raise RuntimeError("No explicit Stage A approval record found")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    encoded_summary_path = RUNS / "encoding_summary.json"
    message_path = RUNS / "source_message_map.npz"
    if not encoded_summary_path.is_file() or not message_path.is_file():
        raise RuntimeError("Run and validate source encoding before generation")
    with message_path.open("rb") as stream:
        import numpy as np
        archive = np.load(stream, allow_pickle=True)
        messages = {key: np.asarray(archive[key]).reshape(-1).astype(int).tolist()
                    for key in archive.files if key != "ckpt"}
    with PAIRS.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 36 or len({r["pair_id"] for r in rows}) != 36:
        raise RuntimeError(f"Expected 36 unique approved prompt pairs; found {len(rows)}")
    if any(r["source_runnable"].lower() != "true" or r["approval_status"] != "approved" for r in rows):
        raise RuntimeError("Every runnable pair must have explicit approval before generation")

    pending = []
    for row in rows:
        case_id = row["case_id"]
        if case_id not in messages:
            raise KeyError(f"No encoded message for {case_id}")
        input_path = ENCODED / case_id / "watermarked.png"
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        expected_config = config_for(row, messages[case_id], input_path)
        directory = case_dir(row)
        directory.mkdir(parents=True, exist_ok=True)
        config_path = directory / "run_config.json"
        status_path = directory / "run_status.json"
        output = directory / "video.mp4"
        previous_config = json.loads(config_path.read_text()) if config_path.exists() else None
        previous_status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if output.is_file():
            try:
                integrity = inspect_video(output)
                config_matches = previous_config == expected_config
                if (integrity["frame_count"] == FRAMES and integrity["width"] == WIDTH
                        and integrity["height"] == HEIGHT and config_matches
                        and previous_status.get("status") == "completed"):
                    continue
            except Exception:
                pass
            quarantine = directory / f"video.invalid.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.mp4"
            os.replace(output, quarantine)
        atomic_json(config_path, expected_config)
        status = {"pair_id": row["pair_id"], "case_id": case_id, "condition": row["condition"],
                  "status": "pending", "input_sha256": expected_config["input_sha256"],
                  "run_config_sha256": hashlib.sha256(json.dumps(expected_config, sort_keys=True).encode()).hexdigest()}
        if previous_status.get("status") == "running":
            status["previous_interruption"] = {"started_at": previous_status.get("started_at"),
                                                "recorded_at": now()}
        elif previous_status.get("status") == "failed":
            status["previous_failure"] = previous_status.get("error")
        partial = directory / ".video.partial.mp4"
        if partial.is_file():
            quarantine = directory / f"video.interrupted.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.mp4"
            os.replace(partial, quarantine)
        atomic_json(status_path, status)
        pending.append((row, input_path, directory, expected_config))

    if not pending:
        return

    import torch
    from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
    from diffusers.utils import export_to_video
    from PIL import Image
    from transformers import T5EncoderModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the uv environment")
    torch.cuda.init()
    device = torch.device("cuda:0")
    log_path = EXP / "logs/generation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            MODEL_ID, subfolder="transformer", torch_dtype=torch.float16, local_files_only=True
        )
        text_encoder = T5EncoderModel.from_pretrained(
            MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16, local_files_only=True
        )
        vae = AutoencoderKLCogVideoX.from_pretrained(
            MODEL_ID, subfolder="vae", torch_dtype=torch.float16, local_files_only=True
        )
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            MODEL_ID, text_encoder=text_encoder, transformer=transformer, vae=vae,
            torch_dtype=torch.float16, local_files_only=True,
        )
        pipe.to(device)
        pipe.enable_sequential_cpu_offload()
        pipe.set_progress_bar_config(disable=True)

        for row, input_path, directory, config in pending:
            pair_id = row["pair_id"]
            status_path = directory / "run_status.json"
            status = json.loads(status_path.read_text())
            status.update({"status": "running", "started_at": now()})
            atomic_json(status_path, status)
            temp_output = directory / ".video.partial.mp4"
            start_time = time.monotonic()
            try:
                with Image.open(input_path) as opened:
                    image = opened.convert("RGB")
                    if image.size != (WIDTH, HEIGHT):
                        raise ValueError(f"Unexpected input dimensions {image.size} for {pair_id}")
                generator = torch.Generator(device=device).manual_seed(SEED)
                frames = pipe(
                    image=image, prompt=row["prompt_used"], negative_prompt=NEGATIVE_PROMPT,
                    guidance_scale=GUIDANCE, use_dynamic_cfg=True,
                    num_inference_steps=STEPS, num_frames=FRAMES, generator=generator,
                ).frames[0]
                export_to_video(frames, str(temp_output), fps=FPS, quality=QUALITY)
                integrity = inspect_video(temp_output)
                if integrity["frame_count"] != FRAMES or (integrity["width"], integrity["height"]) != (WIDTH, HEIGHT):
                    raise RuntimeError(f"Video integrity check failed: {integrity}")
                output = directory / "video.mp4"
                os.replace(temp_output, output)
                status.update({"status": "completed", "completed_at": now(),
                               "elapsed_seconds": round(time.monotonic() - start_time, 2),
                               "output_path": str(output.relative_to(ROOT)),
                               "output_sha256": sha256(output), "video_integrity": integrity})
                atomic_json(status_path, status)
                sys.__stdout__.write(f"[VIDEO_FINISHED] {pair_id} — {status['elapsed_seconds']:.0f}s\n")
                sys.__stdout__.flush()
            except Exception as exc:
                temp_output.unlink(missing_ok=True)
                status.update({"status": "failed", "failed_at": now(),
                               "elapsed_seconds": round(time.monotonic() - start_time, 2),
                               "error": f"{type(exc).__name__}: {exc}",
                               "traceback": traceback.format_exc()})
                atomic_json(status_path, status)
                torch.cuda.empty_cache()
                sys.__stderr__.write(f"[ERROR] {pair_id}: {type(exc).__name__}: {exc}\n")
                sys.__stderr__.flush()
if __name__ == "__main__":
    main()
