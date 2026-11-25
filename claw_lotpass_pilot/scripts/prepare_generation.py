#!/usr/bin/env python3
"""Create approved canonical inputs and the 30-case generation manifest."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "claw_lotpass_pilot"
SELECTION = EXP / "selection"
CANONICAL = EXP / "images/canonical_clean"
TARGET_SIZE = (720, 480)
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-approved", action="store_true", help="Record the explicit user approval for Stage A")
    args = parser.parse_args()
    if not args.user_approved:
        raise RuntimeError("Refusing to prepare generation cases without --user-approved")
    sample = list(csv.DictReader((SELECTION / "claw_sample_15.csv").open(encoding="utf-8")))
    prompts = list(csv.DictReader((SELECTION / "claw_prompt_pairs.csv").open(encoding="utf-8")))
    prompt_by_image = {}
    for item in prompts:
        prompt_by_image.setdefault(item["image_id"], {})[item["condition"]] = item
    if len(sample) != 15 or len(prompts) != 30:
        raise RuntimeError(f"Expected 15 images/30 prompts; found {len(sample)}/{len(prompts)}")

    CANONICAL.mkdir(parents=True, exist_ok=True)
    selected = []
    cases = []
    for row in sorted(sample, key=lambda item: int(item["sample_index"])):
        image_id = row["image_id"]
        if set(prompt_by_image.get(image_id, {})) != {"static_control", "broad_fluid_motion"}:
            raise RuntimeError(f"Incomplete prompt pair for {image_id}")
        source = Path(row["absolute_path"])
        if sha256(source) != row["source_sha256"]:
            raise RuntimeError(f"Source changed since approval: {source}")
        canonical = CANONICAL / f"{image_id}.png"
        with Image.open(source) as opened:
            source_size = opened.size
            fitted = ImageOps.fit(
                opened.convert("RGB"), TARGET_SIZE, method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
            fitted.save(canonical, format="PNG", optimize=True)
        selected.append({
            "sample_index": int(row["sample_index"]), "image_id": image_id,
            "source_filename": row["filename"], "source_path": str(source.relative_to(ROOT)),
            "source_sha256": row["source_sha256"], "source_width": int(row["width"]),
            "source_height": int(row["height"]), "source_size": list(source_size),
            "canonical_image_path": str(canonical.relative_to(ROOT)),
            "canonical_sha256": sha256(canonical), "canonical_width": TARGET_SIZE[0],
            "canonical_height": TARGET_SIZE[1],
            "preprocessing": "PIL ImageOps.fit; center crop to 3:2 then LANCZOS resize to 720x480",
            "reference_video_member": row["reference_video_member"],
        })
        for condition in ("static_control", "broad_fluid_motion"):
            prompt = prompt_by_image[image_id][condition]
            case_id = f"CLAW{int(row['sample_index']):03d}_{condition}"
            cases.append({
                "case_id": case_id, "image_id": image_id,
                "sample_index": int(row["sample_index"]), "condition": condition,
                "prompt": prompt["generated_prompt"], "prompt_word_count": int(prompt["prompt_word_count"]),
                "seed": SEED, "reference_video_member": row["reference_video_member"],
                "canonical_image_path": str(canonical.relative_to(ROOT)),
            })

    manifest = {
        "dataset": "CLAW v1 official test set", "dataset_archive_sha256": sha256(EXP / "downloads/CLAW.zip"),
        "sample_seed": 42, "sample_size": 15, "generation_case_count": 30,
        "approval_request_sha256": sha256(SELECTION / "approval_request.md"),
        "selected_images": selected, "cases": cases,
        "generation_settings": {
            "model_id": "THUDM/CogVideoX-5b-I2V", "seed": 42, "frames": 49, "fps": 8,
            "width": 720, "height": 480, "guidance_scale": 5.0, "inference_steps": 50,
            "video_quality": 10, "message_bits": 32, "encoder_resolution": 256,
            "negative_prompt": "Vibrant colors, overexposed, blurry details, subtitles, style, artwork, painting, overall grayish, poor quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, cluttered background, three legs, many people in the background, walking backward",
        },
    }
    manifest_path = SELECTION / "generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (SELECTION / "generation_cases.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["case_id", "image_id", "sample_index", "condition", "prompt", "prompt_word_count", "seed", "reference_video_member", "canonical_image_path"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(cases)

    approval = {
        "approved": True, "approved_at": datetime.now(timezone.utc).isoformat(),
        "approved_by": "user in conversation", "generation_manifest_sha256": sha256(manifest_path),
        "approval_action": "User instructed to start the rest of the experiment after reviewing the Stage A package.",
        "approval_request_sha256": sha256(SELECTION / "approval_request.md"),
    }
    (SELECTION / "approval.json").write_text(json.dumps(approval, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(selected)} canonical images and {len(cases)} approved generation cases.")


if __name__ == "__main__":
    main()
