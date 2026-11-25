#!/usr/bin/env python3
"""LoT-Pass encode each approved CLAW source exactly once."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "claw_lotpass_pilot"
SELECTION = EXP / "selection"
CANONICAL = EXP / "images/canonical_clean"
WATERMARKED = EXP / "images/watermarked"
WORK = EXP / "runs/encoding_work"
CHECKPOINT = ROOT / "checkpoints/model-99.pth"
SEED, INPUT_RES, MESSAGE_BITS = 42, 256, 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU policy requires CUDA_VISIBLE_DEVICES=0")
    approval = json.loads((SELECTION / "approval.json").read_text())
    manifest_path = SELECTION / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not approval.get("approved") or approval.get("generation_manifest_sha256") != sha256(manifest_path):
        raise RuntimeError("Approval is missing or generation manifest changed after approval")
    if len(manifest["selected_images"]) != 15 or len(manifest["cases"]) != 30:
        raise RuntimeError("Expected 15 selected sources and 30 approved cases")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if WORK.exists() and any(WORK.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing work directory {WORK}")

    import torch
    from test_I2V import encode_unfix
    from utils.utils import setup_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    setup_seed(SEED)
    WORK.mkdir(parents=True, exist_ok=True)
    WATERMARKED.mkdir(parents=True, exist_ok=True)
    encode_unfix(str(CANONICAL), str(WORK), device, ckpt=str(CHECKPOINT), input_res=INPUT_RES, message=MESSAGE_BITS)

    encoded_dir = WORK / "Encoded"
    map_path = WORK / "message_map.npz"
    archive = np.load(map_path, allow_pickle=True)
    messages = {key: np.asarray(archive[key]).reshape(-1).astype(np.uint8)
                for key in archive.files if key != "ckpt"}
    if set(messages) != {item["image_id"] for item in manifest["selected_images"]}:
        raise RuntimeError("Encoder message map does not match the approved 15 sources")
    shutil.copy2(map_path, EXP / "runs/source_message_map.npz")
    records = []
    for item in manifest["selected_images"]:
        image_id = item["image_id"]
        source, encoded = CANONICAL / f"{image_id}.png", encoded_dir / f"{image_id}.png"
        if not encoded.is_file():
            raise FileNotFoundError(encoded)
        final = WATERMARKED / f"{image_id}.png"
        shutil.copy2(encoded, final)
        with np.load(map_path, allow_pickle=True) as archive:
            bits = np.asarray(archive[image_id]).reshape(-1).astype(int).tolist()
        records.append({
            "image_id": image_id, "sample_index": item["sample_index"],
            "canonical_clean_path": str(source.relative_to(ROOT)), "watermarked_path": str(final.relative_to(ROOT)),
            "message_bits": bits, "message_length": len(bits), "encoder_seed": SEED,
            "encoder_resolution": INPUT_RES, "source_sha256": sha256(source),
            "watermarked_sha256": sha256(final),
        })
    summary = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "checkpoint_sha256": sha256(CHECKPOINT),
        "generation_manifest_sha256": sha256(manifest_path), "message_map_sha256": sha256(EXP / "runs/source_message_map.npz"),
        "encoding_seed": SEED, "message_length": MESSAGE_BITS, "encoder_input_resolution": INPUT_RES,
        "device": "CUDA_VISIBLE_DEVICES=0 (physical GPU 0)", "source_encoding_count": len(records),
        "encoding_reused_for_both_conditions": True, "sources": records,
    }
    (EXP / "runs/encoding_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Encoded {len(records)} source images once; each message is shared by its two prompt conditions.", flush=True)


if __name__ == "__main__":
    main()
