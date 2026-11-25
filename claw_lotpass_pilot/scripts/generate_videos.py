#!/usr/bin/env python3
"""Generate the approved 30 CLAW image/prompt cases with CogVideoX."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "claw_lotpass_pilot"
CACHE = ROOT / "vbench_lotpass_pilot/model_cache/huggingface"
os.environ.setdefault("HF_HOME", str(CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE / "hub"))
sys.path.insert(0, str(ROOT))
SELECTION = EXP / "selection"
RUNS = EXP / "runs"
INPUTS = EXP / "images/watermarked"
MODEL_ID = "THUDM/CogVideoX-5b-I2V"
FRAMES, FPS, WIDTH, HEIGHT = 49, 8, 720, 480
GUIDANCE, STEPS, QUALITY = 5.0, 50, 10


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def case_dir(row: dict) -> Path:
    return RUNS / row["case_id"] / f"seed_{int(row['seed'])}"


def inspect_video(path: Path) -> dict:
    import decord
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
    if not len(reader):
        raise RuntimeError("video has no decodable frames")
    frame = reader[0].asnumpy()
    return {"frame_count": len(reader), "fps": float(reader.get_avg_fps()),
            "width": int(frame.shape[1]), "height": int(frame.shape[0])}


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU policy requires CUDA_VISIBLE_DEVICES=0")
    manifest_path = SELECTION / "generation_manifest.json"
    approval_path = SELECTION / "approval.json"
    manifest = json.loads(manifest_path.read_text())
    approval = json.loads(approval_path.read_text())
    manifest_hash = sha256(manifest_path)
    if not approval.get("approved") or approval.get("generation_manifest_sha256") != manifest_hash:
        raise RuntimeError("Missing approval or generation manifest hash mismatch")
    cases = manifest["cases"]
    if len(cases) != 30:
        raise RuntimeError(f"Expected 30 cases, found {len(cases)}")
    encoding_summary = json.loads((RUNS / "encoding_summary.json").read_text())
    if encoding_summary.get("generation_manifest_sha256") != manifest_hash:
        raise RuntimeError("Encoded source messages do not match the approved generation manifest")
    messages_archive = __import__("numpy").load(RUNS / "source_message_map.npz", allow_pickle=True)
    messages = {key: __import__("numpy").asarray(messages_archive[key]).reshape(-1).astype(int).tolist()
                for key in messages_archive.files if key != "ckpt"}

    import torch
    from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
    from diffusers.utils import export_to_video
    from PIL import Image
    from transformers import T5EncoderModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    params = {"torch_dtype": torch.float16, "local_files_only": True}
    print(f"Loading cached {MODEL_ID} for {len(cases)} cases on cuda:0", flush=True)
    transformer = CogVideoXTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer", **params)
    text_encoder = T5EncoderModel.from_pretrained(MODEL_ID, subfolder="text_encoder", **params)
    vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_ID, subfolder="vae", **params)
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        MODEL_ID, transformer=transformer, text_encoder=text_encoder, vae=vae, **params,
    )
    pipe.to(device)
    pipe.enable_sequential_cpu_offload()
    RUNS.mkdir(parents=True, exist_ok=True)
    log_path = EXP / "logs/generation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    failed = []
    for index, row in enumerate(cases, start=1):
        directory = case_dir(row)
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / "video.mp4"
        status_path = directory / "run_status.json"
        input_path = INPUTS / f"{row['image_id']}.png"
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        if row["image_id"] not in messages:
            raise KeyError(f"No encoded message for source {row['image_id']}")
        effective = directory / "effective_input.png"
        if not effective.is_file():
            shutil.copy2(input_path, effective)
        atomic_json(directory / "message.json", {
            "case_id": row["case_id"], "image_id": row["image_id"],
            "condition": row["condition"], "message_bits": messages[row["image_id"]],
            "message_length": len(messages[row["image_id"]]), "reused_across_conditions": True,
        })
        settings = {
            **row, "model_id": MODEL_ID, "checkpoint": "checkpoints/model-99.pth",
            "seed": int(row["seed"]), "guidance_scale": GUIDANCE, "inference_steps": STEPS,
            "num_frames": FRAMES, "fps": FPS, "width": WIDTH, "height": HEIGHT,
            "video_quality": QUALITY, "negative_prompt": manifest["generation_settings"]["negative_prompt"],
            "input_path": str(input_path.relative_to(ROOT)), "input_sha256": sha256(input_path),
            "effective_input_sha256": sha256(effective),
            "source_encoding_reused_for_both_conditions": True,
            "generation_manifest_sha256": manifest_hash,
            "preprocessing": "center crop to 720x480; LoT-Pass encoded once per source image",
        }
        atomic_json(directory / "generation_config.json", settings)
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        if output.is_file() and previous.get("status") == "completed":
            try:
                integrity = inspect_video(output)
                if integrity == {"frame_count": FRAMES, "fps": float(FPS), "width": WIDTH, "height": HEIGHT}:
                    print(f"[SKIP_VALID] {row['case_id']}", flush=True)
                    continue
            except Exception:
                pass
        status = {"case_id": row["case_id"], "status": "running", "started_at": now(),
                  "seed": int(row["seed"]), "generation_manifest_sha256": manifest_hash}
        if previous.get("status") == "failed":
            status["previous_failure"] = previous.get("error")
        atomic_json(status_path, status)
        partial = directory / ".video.partial.mp4"
        try:
            with Image.open(effective) as opened:
                image = opened.convert("RGB")
                if image.size != (WIDTH, HEIGHT):
                    raise ValueError(f"Unexpected input dimensions {image.size}")
            generator = torch.Generator(device=device).manual_seed(int(row["seed"]))
            print(f"[START] {index}/30 {row['case_id']}", flush=True)
            frames = pipe(
                image=image, prompt=row["prompt"],
                negative_prompt=manifest["generation_settings"]["negative_prompt"],
                guidance_scale=GUIDANCE, use_dynamic_cfg=True,
                num_inference_steps=STEPS, num_frames=FRAMES, generator=generator,
            ).frames[0]
            export_to_video(frames, str(partial), fps=FPS, quality=QUALITY)
            integrity = inspect_video(partial)
            if integrity["frame_count"] != FRAMES or integrity["width"] != WIDTH or integrity["height"] != HEIGHT or abs(integrity["fps"] - FPS) > 0.05:
                raise RuntimeError(f"Output integrity check failed: {integrity}")
            os.replace(partial, output)
            atomic_json(status_path, {
                **status, "status": "completed", "completed_at": now(),
                "output_path": str(output.relative_to(ROOT)), "output_sha256": sha256(output),
                "video_integrity": integrity,
            })
            message = f"[COMPLETE] {row['case_id']} {integrity}"
            print(message, flush=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"{now()} {message}\n")
        except Exception as exc:
            partial.unlink(missing_ok=True)
            atomic_json(status_path, {
                **status, "status": "failed", "failed_at": now(),
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            })
            failed.append(row["case_id"])
            message = f"[FAILED] {row['case_id']}: {type(exc).__name__}: {exc}"
            print(message, flush=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"{now()} {message}\n")
            torch.cuda.empty_cache()
    print(f"Generation pass finished; completed={30-len(failed)}, failed={len(failed)}", flush=True)
    if failed:
        print("Failed case IDs: " + ", ".join(failed), flush=True)


if __name__ == "__main__":
    main()
