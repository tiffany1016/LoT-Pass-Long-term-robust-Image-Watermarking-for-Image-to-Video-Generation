#!/usr/bin/env python3
"""Encode each approved canonical pilot image once and run pre-I2V decode checks."""
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
EXP = ROOT / "vbench_lotpass_pilot"
SELECTION = EXP / "selection"
SOURCE_DIR = EXP / "images/canonical_clean"
WATERMARKED_DIR = EXP / "images/watermarked"
WORK_DIR = EXP / "runs/encoding_work"
CHECKPOINT = ROOT / "checkpoints/model-99.pth"
SEED = 42
MESSAGE_BITS = 32
INPUT_RES = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU policy: launch with CUDA_VISIBLE_DEVICES=0")
    approval = json.loads((SELECTION / "approval.json").read_text())
    manifest_path = SELECTION / "selection_manifest.json"
    manifest_sha = sha256(manifest_path)
    if not approval.get("approved") or approval.get("selection_manifest_sha256") != manifest_sha:
        raise RuntimeError("Selection approval is missing or its manifest hash no longer matches")
    selected = json.loads(manifest_path.read_text())["selected_images"]
    if len(selected) != 10 or len({item["image_id"] for item in selected}) != 10:
        raise RuntimeError("Approved manifest must contain exactly 10 unique source images")
    for item in selected:
        image_path = ROOT / item["canonical_image_path"]
        if not image_path.is_file() or sha256(image_path) != item["canonical_sha256"]:
            raise RuntimeError(f"Canonical image missing or changed: {image_path}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if WORK_DIR.exists() and any(WORK_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing encoding work directory: {WORK_DIR}")

    import torch
    from PIL import Image
    from torchvision import transforms
    from models import convnext
    from test_I2V import encode_unfix
    from utils.train_param import TrainParam
    from utils.utils import calculate_acc, setup_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the uv environment")
    device = torch.device("cuda:0")
    setup_seed(SEED)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    WATERMARKED_DIR.mkdir(parents=True, exist_ok=True)

    encode_unfix(str(SOURCE_DIR), str(WORK_DIR), device, ckpt=str(CHECKPOINT),
                 input_res=INPUT_RES, message=MESSAGE_BITS)
    encoded_dir = WORK_DIR / "Encoded"
    message_map_path = WORK_DIR / "message_map.npz"
    with np.load(message_map_path, allow_pickle=True) as archive:
        messages = {name: np.array(archive[name]) for name in archive.files}
    if len([key for key in messages if key != "ckpt"]) != len(selected):
        raise RuntimeError("Encoder produced an unexpected number of source messages")

    train_param = json.loads((CHECKPOINT.parent / "train_param.json").read_text())
    params = TrainParam(**train_param)
    decoder = convnext.ConvNextDecoder(params)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.eval().to(device)
    del checkpoint
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Resize((INPUT_RES, INPUT_RES)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    records = []
    decoded_arrays = {}
    with torch.inference_mode():
        for item in selected:
            image_id = item["image_id"]
            key = image_id
            message = np.asarray(messages[key]).reshape(1, MESSAGE_BITS)
            clean_path = ROOT / item["canonical_image_path"]
            encoded_path = encoded_dir / f"{image_id}.png"
            if not encoded_path.is_file():
                raise FileNotFoundError(encoded_path)
            final_encoded = WATERMARKED_DIR / f"{image_id}.png"
            shutil.copy2(encoded_path, final_encoded)

            with Image.open(clean_path) as image:
                clean_image = transform(image.convert("RGB")).unsqueeze(0).to(device)
            with Image.open(final_encoded) as image:
                encoded_image = transform(image.convert("RGB")).unsqueeze(0).to(device)
            clean_decoded = decoder(clean_image)
            encoded_decoded = decoder(encoded_image)
            gt = torch.as_tensor(message, dtype=encoded_decoded.dtype, device=device)
            clean_ba = float(calculate_acc(gt, clean_decoded))
            encoded_ba = float(calculate_acc(gt, encoded_decoded))
            decoded_arrays[f"{image_id}_clean"] = clean_decoded.detach().cpu().numpy()
            decoded_arrays[f"{image_id}_encoded"] = encoded_decoded.detach().cpu().numpy()
            decoded_arrays[f"{image_id}_ground_truth"] = message
            records.append({
                "image_id": image_id,
                "source_image": item["image_name"],
                "canonical_clean_path": item["canonical_image_path"],
                "watermarked_generator_input": str(final_encoded.relative_to(ROOT)),
                "message_bits": message[0].astype(int).tolist(),
                "encoding_seed": SEED,
                "message_length": MESSAGE_BITS,
                "encoder_resolution": INPUT_RES,
                "pre_i2v_clean_decode_ba": clean_ba,
                "pre_i2v_encoded_decode_ba": encoded_ba,
                "watermarked_sha256": sha256(final_encoded),
                "status": "completed",
            })
            print(f"[{image_id}] clean BA={clean_ba:.4f}; encoded BA={encoded_ba:.4f}", flush=True)

    np.savez_compressed(EXP / "runs/encoding_sanity_decodes.npz", **decoded_arrays)
    shutil.copy2(message_map_path, EXP / "runs/source_message_map.npz")
    result = {
        "selection_manifest_sha256": manifest_sha,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "source_message_map_sha256": sha256(EXP / "runs/source_message_map.npz"),
        "encoding_seed": SEED,
        "message_length": MESSAGE_BITS,
        "encoder_input_resolution": INPUT_RES,
        "device": "cuda:0 (physical GPU 0 via CUDA_VISIBLE_DEVICES=0)",
        "source_encodings": records,
    }
    (EXP / "runs/encoding_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Encoded {len(records)} source images and validated pre-I2V decode mapping")


if __name__ == "__main__":
    main()
