#!/usr/bin/env python3
"""Encode each approved ChronoMagic source once and save its 32-bit payload."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "chronomagic150_global_appearance"
DATA = EXP / "dataset"
WORK = EXP / "watermarked_sources/encoding_work"
ALIAS_SOURCES = WORK / "source_images"
ENCODED = WORK / "Encoded"
WATERMARKED = EXP / "watermarked_sources"
CHECKPOINT = ROOT / "checkpoints/model-99.pth"
INPUT_RES, MESSAGE_BITS, ENCODER_SEED = 256, 32, 42


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU policy requires CUDA_VISIBLE_DEVICES=0")
    approval = json.loads((EXP / "stage_a_review/approval_record.json").read_text())
    manifest_path = DATA / "chronomagic150_generation_manifest.csv"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if not approval.get("approved") or approval.get("generation_manifest_sha256") != manifest_hash:
        raise RuntimeError("Approval missing or Stage A generation manifest changed after approval")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    selected = rows(DATA / "chronomagic150_selected16.csv")
    cases = rows(manifest_path)
    if len(selected) != 16 or len(cases) != 32:
        raise RuntimeError(f"Expected 16 sources and 32 cases; found {len(selected)} and {len(cases)}")
    existing = EXP / "watermarked_sources/encoding_summary.json"
    if existing.is_file():
        summary = json.loads(existing.read_text())
        if summary.get("generation_manifest_sha256") == manifest_hash and summary.get("source_encoding_count") == 16:
            print("Validated existing encoding outputs; skipping repeat encoding.", flush=True)
            return
        raise FileExistsError("Existing encoding outputs do not match this approved manifest")
    if WORK.exists() and any(WORK.iterdir()):
        raise FileExistsError(f"Partial encoder outputs need review before continuing: {WORK}")

    import torch
    from PIL import Image
    from torchvision import transforms
    from models import convnext
    from test_I2V import encode_unfix
    from utils.train_param import TrainParam
    from utils.utils import calculate_acc, setup_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected uv environment")
    torch.cuda.init()
    ALIAS_SOURCES.mkdir(parents=True, exist_ok=True)
    aliases = {}
    for row in selected:
        alias = f"CM{int(row['selection_index']):03d}"
        source = ROOT / row["source_frame_model_input"]
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, ALIAS_SOURCES / f"{alias}.png")
        aliases[alias] = {"sample_id": row["sample_id"], "selection_index": int(row["selection_index"]),
                          "source_path": row["source_frame_model_input"], "source_sha256": sha256(source)}
    device = torch.device("cuda:0")
    setup_seed(ENCODER_SEED)
    encode_unfix(str(ALIAS_SOURCES), str(WORK), device, ckpt=str(CHECKPOINT),
                 input_res=INPUT_RES, message=MESSAGE_BITS)
    source_archive = np.load(WORK / "message_map.npz", allow_pickle=True)
    alias_messages = {key: np.asarray(source_archive[key]).reshape(-1).astype(np.uint8)
                      for key in source_archive.files if key != "ckpt"}
    if set(alias_messages) != set(aliases):
        raise RuntimeError("Encoded message IDs do not match all 16 approved source aliases")

    params = TrainParam(**json.loads((CHECKPOINT.parent / "train_param.json").read_text()))
    decoder = convnext.ConvNextDecoder(params)
    state = torch.load(CHECKPOINT, map_location="cpu")
    decoder.load_state_dict(state["decoder"])
    del state
    decoder.eval().to(device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((INPUT_RES, INPUT_RES)),
                                    transforms.Normalize([0.5] * 3, [0.5] * 3)])
    WATERMARKED.mkdir(parents=True, exist_ok=True)
    message_map = {}
    records = []
    with torch.inference_mode():
        for alias, info in aliases.items():
            sample_id = info["sample_id"]
            clean = ALIAS_SOURCES / f"{alias}.png"
            encoded = ENCODED / f"{alias}.png"
            if not encoded.is_file():
                raise FileNotFoundError(encoded)
            final_dir = WATERMARKED / sample_id
            final_dir.mkdir(parents=True, exist_ok=True)
            final_clean = final_dir / "source_model_input.png"
            final_marked = final_dir / "watermarked.png"
            shutil.copy2(clean, final_clean)
            shutil.copy2(encoded, final_marked)
            bits = alias_messages[alias]
            message_map[sample_id] = bits
            gt = torch.as_tensor(bits, dtype=torch.float32, device=device).unsqueeze(0)
            with Image.open(clean) as im:
                clean_tensor = transform(im.convert("RGB")).unsqueeze(0).to(device)
            with Image.open(encoded) as im:
                marked_tensor = transform(im.convert("RGB")).unsqueeze(0).to(device)
            clean_ba = float(calculate_acc(gt, decoder(clean_tensor)))
            marked_ba = float(calculate_acc(gt, decoder(marked_tensor)))
            record = {**info, "watermarked_path": str(final_marked.relative_to(ROOT)),
                      "watermarked_sha256": sha256(final_marked), "message_bits": bits.astype(int).tolist(),
                      "message_length": int(bits.size), "encoder_seed": ENCODER_SEED,
                      "encoder_input_resolution": INPUT_RES, "clean_decode_ba": clean_ba,
                      "watermarked_decode_ba": marked_ba, "status": "completed"}
            records.append(record)
            print(f"[ENCODED] {sample_id} · input BA={clean_ba:.4f}, watermarked BA={marked_ba:.4f}", flush=True)
    np.savez_compressed(WATERMARKED / "source_message_map.npz", **message_map)
    summary = {
        "study_name": "ChronoMagic-150-derived meteorological paired I2V LoT-Pass evaluation",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "checkpoint_sha256": sha256(CHECKPOINT),
        "generation_manifest_sha256": manifest_hash, "source_message_map_sha256": sha256(WATERMARKED / "source_message_map.npz"),
        "encoding_seed": ENCODER_SEED, "message_length": MESSAGE_BITS, "encoder_input_resolution": INPUT_RES,
        "device": "cuda:0 (physical GPU 0 via CUDA_VISIBLE_DEVICES=0)", "source_encoding_count": len(records),
        "same_encoded_source_and_payload_reused_for_both_conditions": True, "sources": records,
    }
    (WATERMARKED / "encoding_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ENCODING_FINISHED] {len(records)} unique sources encoded once.", flush=True)


if __name__ == "__main__":
    main()
