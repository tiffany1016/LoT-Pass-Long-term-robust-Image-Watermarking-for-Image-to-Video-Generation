#!/usr/bin/env python3
"""Generate approved VBench++ pilot clips using the repository CogVideoX stack."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "vbench_lotpass_pilot"
SELECTION = EXP / "selection"
RUNS = EXP / "runs"
INPUTS = EXP / "images/watermarked"
CHECKPOINT = ROOT / "checkpoints/model-99.pth"
MODEL_ID = "THUDM/CogVideoX-5b-I2V"
SEED = 42
FRAMES = 49
FPS = 8
WIDTH = 720
HEIGHT = 480
GUIDANCE = 5.0
STEPS = 50
QUALITY = 10
NEGATIVE_PROMPT = (
    "Vibrant colors, overexposed, static, blurry details, subtitles, style, artwork, painting, scene, static, "
    "overall grayish, poor quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, static scene, "
    "cluttered background, three legs, many people in the background, walking backward"
)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def case_dir(row: dict) -> Path:
    return RUNS / row["image_id"] / row["motion_group"] / f"seed_{SEED}"


def inspect_video(path: Path) -> dict:
    import decord
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
    if len(reader) < 1:
        raise RuntimeError("video has no readable frames")
    frame = reader[0].asnumpy()
    fps = float(reader.get_avg_fps())
    width, height = int(frame.shape[1]), int(frame.shape[0])
    return {"frame_count": len(reader), "fps": fps, "width": width, "height": height}


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU policy: launch with CUDA_VISIBLE_DEVICES=0")
    approval = json.loads((SELECTION / "approval.json").read_text())
    manifest_path = SELECTION / "selection_manifest.json"
    manifest_sha = sha256(manifest_path)
    if not approval.get("approved") or approval.get("selection_manifest_sha256") != manifest_sha:
        raise RuntimeError("Selection approval is missing or its manifest hash no longer matches")
    rows = list(__import__("csv").DictReader((SELECTION / "vbench_40pairs.csv").open(encoding="utf-8")))
    if len(rows) != 40:
        raise RuntimeError(f"Expected exactly 40 approved pairs; found {len(rows)}")
    source_message_archive = __import__("numpy").load(RUNS / "source_message_map.npz", allow_pickle=True)

    import torch
    from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
    from diffusers.utils import export_to_video
    from PIL import Image
    from transformers import T5EncoderModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    for row in rows:
        directory = case_dir(row)
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / "video.mp4"
        input_path = INPUTS / f'{row["image_id"]}.png'
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        if not (directory / "effective_input.png").exists():
            import shutil
            shutil.copy2(input_path, directory / "effective_input.png")
        bits = source_message_archive[row["image_id"]].reshape(-1).astype(int).tolist()
        atomic_json(directory / "message.json", {"case_id": row["case_id"], "image_id": row["image_id"],
                                                   "message_bits": bits, "message_length": len(bits),
                                                   "reused_across_image_conditions": True})
        config = {
            "case_id": row["case_id"], "image_id": row["image_id"], "motion_full": row["motion_full"],
            "prompt": row["prompt"], "model_id": MODEL_ID, "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "seed": SEED, "guidance_scale": GUIDANCE, "inference_steps": STEPS,
            "num_frames": FRAMES, "fps": FPS, "width": WIDTH, "height": HEIGHT,
            "video_quality": QUALITY, "negative_prompt": NEGATIVE_PROMPT,
            "input_path": str(input_path.relative_to(ROOT)), "input_sha256": sha256(input_path),
            "input_preprocessing": "canonical 720x480 image, LoT-Pass encoded once per image; no further geometry transform",
            "selection_manifest_sha256": manifest_sha,
        }
        atomic_json(directory / "generation_config.json", config)
        status_path = directory / "run_status.json"
        if output.is_file():
            try:
                integrity = inspect_video(output)
                if integrity["frame_count"] == FRAMES and integrity["width"] == WIDTH and integrity["height"] == HEIGHT:
                    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
                    if previous.get("status") == "completed":
                        print(f"[SKIP_VALID] {row['case_id']}", flush=True)
                        continue
            except Exception:
                pass
            output.unlink()
        prior = json.loads(status_path.read_text()) if status_path.exists() else {}
        status = {"case_id": row["case_id"], "seed": SEED, "status": "pending",
                  "input_sha256": config["input_sha256"], "selection_manifest_sha256": manifest_sha}
        if prior.get("status") == "failed":
            status["previous_failure"] = prior.get("error")
        atomic_json(status_path, status)

    log_dir = EXP / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "generation.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log_stream:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(old_stdout, log_stream)
        sys.stderr = Tee(old_stderr, log_stream)
        try:
            print(f"Loading {MODEL_ID} for {len(rows)} approved cases; device={device}; local HF cache.", flush=True)
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                MODEL_ID, subfolder="transformer", torch_dtype=torch.float16
            )
            text_encoder = T5EncoderModel.from_pretrained(
                MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16
            )
            vae = AutoencoderKLCogVideoX.from_pretrained(
                MODEL_ID, subfolder="vae", torch_dtype=torch.float16
            )
            pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                MODEL_ID, text_encoder=text_encoder, transformer=transformer, vae=vae,
                torch_dtype=torch.float16,
            )
            pipe.to(device)
            pipe.enable_sequential_cpu_offload()
            for index, row in enumerate(rows, start=1):
                directory = case_dir(row)
                output = directory / "video.mp4"
                status_path = directory / "run_status.json"
                status = json.loads(status_path.read_text())
                if status.get("status") == "completed":
                    continue
                status.update({"status": "running", "started_at": now()})
                atomic_json(status_path, status)
                temp_output = directory / ".video.partial.mp4"
                try:
                    image_path = directory / "effective_input.png"
                    with Image.open(image_path) as opened:
                        image = opened.convert("RGB")
                        if image.size != (WIDTH, HEIGHT):
                            raise ValueError(f"Unexpected input dimensions {image.size} for {row['case_id']}")
                    generator = torch.Generator(device=device).manual_seed(SEED)
                    print(f"[START] {index}/{len(rows)} {row['case_id']} {row['motion_full']}", flush=True)
                    frames = pipe(
                        image=image, prompt=row["prompt"], negative_prompt=NEGATIVE_PROMPT,
                        guidance_scale=GUIDANCE, use_dynamic_cfg=True,
                        num_inference_steps=STEPS, num_frames=FRAMES, generator=generator,
                    ).frames[0]
                    export_to_video(frames, str(temp_output), fps=FPS, quality=QUALITY)
                    integrity = inspect_video(temp_output)
                    if integrity["frame_count"] != FRAMES or (integrity["width"], integrity["height"]) != (WIDTH, HEIGHT):
                        raise RuntimeError(f"Output integrity check failed: {integrity}")
                    os.replace(temp_output, output)
                    status.update({"status": "completed", "completed_at": now(), "output_path": str(output.relative_to(ROOT)),
                                   "output_sha256": sha256(output), "video_integrity": integrity})
                    atomic_json(status_path, status)
                    print(f"[COMPLETE] {index}/{len(rows)} {row['case_id']} {integrity}", flush=True)
                except Exception as exc:
                    temp_output.unlink(missing_ok=True)
                    status.update({"status": "failed", "failed_at": now(), "error": f"{type(exc).__name__}: {exc}",
                                   "traceback": traceback.format_exc()})
                    atomic_json(status_path, status)
                    print(f"[FAILED] {row['case_id']}: {type(exc).__name__}: {exc}", flush=True)
                    torch.cuda.empty_cache()
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
    print(f"Generation pass ended. Log: {log_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
