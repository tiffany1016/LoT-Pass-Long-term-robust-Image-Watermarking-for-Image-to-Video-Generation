#!/usr/bin/env python3
"""Run LoT-Pass once per approved TC-Bench reference image."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "tcbench_background_shifts"
IMAGES = EXP / "references/source_frames_model_ready"
WORK = EXP / "encoding_work"
OUTPUT = EXP / "encoded"
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


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU policy: launch with CUDA_VISIBLE_DEVICES=0")
    if not CHECKPOINT.is_file() or not (CHECKPOINT.parent / "train_param.json").is_file():
        raise FileNotFoundError("LoT-Pass checkpoint or train_param.json is missing")
    if not (EXP / "approval/approval_record.md").is_file():
        raise RuntimeError("No explicit Stage A approval record found")
    source_paths = sorted(IMAGES.glob("TC*.png"))
    expected_ids = [101, 102, 103, 104, 105, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
    if [int(path.stem[2:]) for path in source_paths] != expected_ids:
        raise RuntimeError("Expected exactly the 18 approved model-ready TC-Bench images")
    if (EXP / "runs/encoding_summary.json").is_file():
        previous = json.loads((EXP / "runs/encoding_summary.json").read_text())
        if previous.get("checkpoint_sha256") == sha256(CHECKPOINT):
            print("Existing encoding summary found; refusing to re-encode and overwrite results.", flush=True)
            return
        raise FileExistsError("Existing encoding results use a different checkpoint")

    WORK.mkdir(parents=True, exist_ok=True)
    stage_sources = WORK / "source_images"
    stage_sources.mkdir(parents=True, exist_ok=True)
    for source in source_paths:
        shutil.copy2(source, stage_sources / source.name)

    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from models import convnext
    from test_I2V import encode_unfix
    from utils.train_param import TrainParam
    from utils.utils import calculate_acc, setup_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the uv environment")
    torch.cuda.init()
    device = torch.device("cuda:0")
    encoded_dir = WORK / "Encoded"
    message_map_path = WORK / "message_map.npz"
    reusable_encoder_pass = message_map_path.is_file() and all(
        (encoded_dir / source.name).is_file() for source in source_paths
    )
    if reusable_encoder_pass:
        print("Resuming from the completed encoder pass; preserving its encoded images and messages.", flush=True)
    else:
        if any(encoded_dir.glob("*.png")) or message_map_path.exists():
            raise RuntimeError("Partial encoder outputs found; refusing to overwrite or mix them")
        setup_seed(SEED)
        encode_unfix(str(stage_sources), str(WORK), device, ckpt=str(CHECKPOINT),
                     input_res=INPUT_RES, message=MESSAGE_BITS)

    with np.load(message_map_path, allow_pickle=True) as archive:
        messages = {name: np.asarray(archive[name]).reshape(-1).astype(np.uint8)
                    for name in archive.files if name != "ckpt"}
    if set(messages) != {path.stem for path in source_paths}:
        raise RuntimeError(f"Encoder message IDs do not match the 18 official IDs: {sorted(messages)}")

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

    records, decoded = [], {}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index, source in enumerate(source_paths, 1):
            case_id = source.stem
            encoded_path = encoded_dir / source.name
            if not encoded_path.is_file():
                raise FileNotFoundError(encoded_path)
            case_out = OUTPUT / case_id
            case_out.mkdir(parents=True, exist_ok=True)
            target = case_out / "watermarked.png"
            shutil.copy2(encoded_path, target)
            shutil.copy2(source, case_out / "source_model_ready.png")
            message = messages[case_id]
            gt = torch.as_tensor(message, dtype=torch.float32, device=device).unsqueeze(0)
            with Image.open(source) as image:
                clean = transform(image.convert("RGB")).unsqueeze(0).to(device)
            with Image.open(target) as image:
                marked = transform(image.convert("RGB")).unsqueeze(0).to(device)
            clean_decode, marked_decode = decoder(clean), decoder(marked)
            clean_ba = float(calculate_acc(gt, clean_decode))
            marked_ba = float(calculate_acc(gt, marked_decode))
            decoded[f"{case_id}_clean"] = clean_decode.cpu().numpy()
            decoded[f"{case_id}_watermarked"] = marked_decode.cpu().numpy()
            decoded[f"{case_id}_message"] = message.reshape(1, -1)
            record = {
                "case_id": case_id, "official_id": int(case_id[2:]),
                "source_image": str(source.relative_to(ROOT)),
                "source_image_sha256": sha256(source),
                "watermarked_image": str(target.relative_to(ROOT)),
                "watermarked_sha256": sha256(target),
                "message_bits": message.astype(int).tolist(), "message_length": len(message),
                "encoding_seed": SEED, "encoder_resolution": INPUT_RES,
                "pre_i2v_clean_decode_ba": clean_ba,
                "pre_i2v_watermarked_decode_ba": marked_ba,
                "status": "completed",
            }
            records.append(record)
            atomic_json(case_out / "encoding_record.json", record)
            print(f"[ENCODED {index}/18] {case_id}: clean BA={clean_ba:.4f}; watermarked BA={marked_ba:.4f}", flush=True)

    (EXP / "runs").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EXP / "runs/encoding_sanity_decodes.npz", **decoded)
    shutil.copy2(message_map_path, EXP / "runs/source_message_map.npz")
    result = {
        "study_name": "TC-Bench-derived first-frame I2V evaluation of all official background-shifts entries",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "checkpoint_sha256": sha256(CHECKPOINT),
        "source_message_map_sha256": sha256(EXP / "runs/source_message_map.npz"),
        "encoding_seed": SEED, "message_length": MESSAGE_BITS,
        "encoder_input_resolution": INPUT_RES,
        "device": "cuda:0 (physical GPU 0 via CUDA_VISIBLE_DEVICES=0)",
        "source_encodings": records,
    }
    (EXP / "runs").mkdir(exist_ok=True)
    atomic_json(EXP / "runs/encoding_summary.json", result)
    print(f"Encoded and pre-I2V decoded {len(records)} unique source images.", flush=True)


if __name__ == "__main__":
    main()
