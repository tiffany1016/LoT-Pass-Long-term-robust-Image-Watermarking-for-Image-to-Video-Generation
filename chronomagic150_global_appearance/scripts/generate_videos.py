#!/usr/bin/env python3
"""Generate the approved ChronoMagic paired I2V videos with resumable outputs."""
from __future__ import annotations

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
EXP = ROOT / "chronomagic150_global_appearance"
DATA = EXP / "dataset"
RUNS = EXP / "runs"
VIDEOS = EXP / "generated_videos"
WATERMARKED = EXP / "watermarked_sources"
CHECKPOINT = ROOT / "checkpoints/model-99.pth"
CACHE = ROOT / "vbench_lotpass_pilot/model_cache/huggingface"
os.environ.setdefault("HF_HOME", str(CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE / "hub"))
MODEL_ID = "THUDM/CogVideoX-5b-I2V"
SEED, FRAMES, FPS, WIDTH, HEIGHT = 42, 49, 8, 720, 480
GUIDANCE, STEPS, QUALITY = 5.0, 50, 10
NEGATIVE_PROMPT = (
    "Vibrant colors, overexposed, blurry details, subtitles, style, artwork, painting, overall grayish, "
    "poor quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, cluttered background, three legs, "
    "many people in the background, walking backward"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def inspect_video(path: Path) -> dict:
    import av
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for _ in container.decode(stream):
            count += 1
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        return {"frame_count": count, "width": int(stream.width), "height": int(stream.height), "fps": fps}


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU policy requires CUDA_VISIBLE_DEVICES=0")
    approval_path = EXP / "stage_a_review/approval_record.json"
    manifest_path = DATA / "chronomagic150_generation_manifest.csv"
    approval = json.loads(approval_path.read_text())
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if not approval.get("approved") or approval.get("generation_manifest_sha256") != manifest_hash:
        raise RuntimeError("Approval missing or Stage A manifest changed")
    if not (WATERMARKED / "encoding_summary.json").is_file():
        raise RuntimeError("Source encoding must be complete before video generation")
    encoding = json.loads((WATERMARKED / "encoding_summary.json").read_text())
    if encoding.get("generation_manifest_sha256") != manifest_hash:
        raise RuntimeError("Encoded sources were prepared against a different manifest")
    message_archive = __import__("numpy").load(WATERMARKED / "source_message_map.npz", allow_pickle=True)
    message_ids = set(message_archive.files)
    cases = read_rows(manifest_path)
    if len(cases) != 32 or len({row["case_id"] for row in cases}) != 32:
        raise RuntimeError(f"Expected 32 unique cases; found {len(cases)}")
    if message_ids != {row["sample_id"] for row in cases}:
        raise RuntimeError("Watermark message IDs do not match the approved sources")
    pending = []
    for row in cases:
        sample_id, condition = row["sample_id"], row["condition"]
        run_dir = RUNS / sample_id / condition
        run_dir.mkdir(parents=True, exist_ok=True)
        output = VIDEOS / sample_id / f"{condition}.mp4"
        status_path = run_dir / "run_status.json"
        config_path = run_dir / "generation_config.json"
        source = WATERMARKED / sample_id / "watermarked.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        config = {
            **row, "model_id": MODEL_ID, "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "checkpoint_sha256": sha256(CHECKPOINT), "seed": SEED,
            "guidance_scale": GUIDANCE, "inference_steps": STEPS, "num_frames": FRAMES,
            "fps": FPS, "width": WIDTH, "height": HEIGHT, "video_quality": QUALITY,
            "negative_prompt": NEGATIVE_PROMPT, "watermarked_input": str(source.relative_to(ROOT)),
            "watermarked_input_sha256": sha256(source), "generation_manifest_sha256": manifest_hash,
            "same_source_encoding_for_pair": True,
        }
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        previous = json.loads(status_path.read_text()) if status_path.is_file() else {}
        if output.is_file() and previous.get("status") == "completed" and previous.get("generation_config_sha256") == config_hash:
            try:
                integrity = inspect_video(output)
                if integrity["frame_count"] == FRAMES and integrity["width"] == WIDTH and integrity["height"] == HEIGHT and abs(integrity["fps"] - FPS) < 0.05:
                    continue
            except Exception:
                pass
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        partial = output.with_name(f".{condition}.partial.mp4")
        if partial.exists():
            partial.unlink()
        atomic_json(status_path, {"case_id": row["case_id"], "sample_id": sample_id,
                                  "condition": condition, "status": "pending",
                                  "generation_config_sha256": config_hash,
                                  "previous_failure": previous.get("error") if previous.get("status") == "failed" else None})
        pending.append((row, source, output, partial, status_path, config_hash))
    if not pending:
        print("All 32 videos already exist and pass integrity checks.", flush=True)
        return

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the configured uv environment")
    torch.cuda.init()
    from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
    from diffusers.utils import export_to_video
    from PIL import Image
    from transformers import T5EncoderModel

    model_args = {"torch_dtype": torch.float16, "local_files_only": True}
    device = torch.device("cuda:0")
    log_path = EXP / "logs/generation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    failed = []
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = log
        sys.stderr = log
        try:
            transformer = CogVideoXTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer", **model_args)
            text_encoder = T5EncoderModel.from_pretrained(MODEL_ID, subfolder="text_encoder", **model_args)
            vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_ID, subfolder="vae", **model_args)
            pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                MODEL_ID, transformer=transformer, text_encoder=text_encoder, vae=vae, **model_args)
            pipe.to(device)
            pipe.enable_sequential_cpu_offload()
            pipe.set_progress_bar_config(disable=True)
            for index, (row, source, output, partial, status_path, config_hash) in enumerate(pending, start=1):
                status = {"case_id": row["case_id"], "sample_id": row["sample_id"],
                          "condition": row["condition"], "status": "running", "started_at": now(),
                          "generation_config_sha256": config_hash}
                atomic_json(status_path, status)
                started = time.monotonic()
                try:
                    with Image.open(source) as opened:
                        image = opened.convert("RGB")
                    if image.size != (WIDTH, HEIGHT):
                        raise RuntimeError(f"Unexpected source image size {image.size}")
                    generator = torch.Generator(device=device).manual_seed(SEED)
                    frames = pipe(image=image, prompt=row["prompt"], negative_prompt=NEGATIVE_PROMPT,
                                  guidance_scale=GUIDANCE, use_dynamic_cfg=True,
                                  num_inference_steps=STEPS, num_frames=FRAMES,
                                  generator=generator).frames[0]
                    output.parent.mkdir(parents=True, exist_ok=True)
                    export_to_video(frames, str(partial), fps=FPS, quality=QUALITY)
                    integrity = inspect_video(partial)
                    if integrity["frame_count"] != FRAMES or (integrity["width"], integrity["height"]) != (WIDTH, HEIGHT) or abs(integrity["fps"] - FPS) > 0.05:
                        raise RuntimeError(f"Video integrity check failed: {integrity}")
                    os.replace(partial, output)
                    status.update({"status": "completed", "completed_at": now(),
                                   "elapsed_seconds": round(time.monotonic() - started, 2),
                                   "output_path": str(output.relative_to(ROOT)), "output_sha256": sha256(output),
                                   "video_integrity": integrity})
                    atomic_json(status_path, status)
                    # This line is the user-visible completion event consumed by the runner.
                    original_stdout.write(f"[VIDEO_FINISHED] {row['case_id']} | {status['elapsed_seconds']:.0f}s\n")
                    original_stdout.flush()
                except Exception as exc:
                    partial.unlink(missing_ok=True)
                    status.update({"status": "failed", "failed_at": now(),
                                   "elapsed_seconds": round(time.monotonic() - started, 2),
                                   "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                    atomic_json(status_path, status)
                    failed.append(row["case_id"])
                    original_stderr.write(f"[VIDEO_ERROR] {row['case_id']} | {type(exc).__name__}: {exc}\n")
                    original_stderr.flush()
                    torch.cuda.empty_cache()
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
    if failed:
        raise RuntimeError("Video generation failures: " + ", ".join(failed))
    print(f"[GENERATION_PASS_FINISHED] {len(cases)} paired videos generated and validated.", flush=True)


if __name__ == "__main__":
    main()
