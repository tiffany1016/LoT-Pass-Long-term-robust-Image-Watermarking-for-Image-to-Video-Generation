#!/usr/bin/env python3
"""Inventory CLAW v1 and create a reproducible 15-image review package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "claw_lotpass_pilot"
ARCHIVE = EXP / "downloads/CLAW.zip"
SOURCE = EXP / "images/source"
SELECTION = EXP / "selection"
SEED = 42
SAMPLE_SIZE = 15
SOURCE_URL = "https://github.com/simon3dv/SLR-SFS"
DATASET_URL = "https://slr-sfs.github.io/dataset/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def contact_sheet(rows: list[dict], output: Path) -> None:
    columns, cell_w, cell_h, image_h = 3, 460, 310, 258
    rows_n = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows_n * cell_h), "#eceff1")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(rows):
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        with Image.open(row["absolute_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((cell_w - 20, image_h - 12), Image.Resampling.LANCZOS)
        image_x = x + (cell_w - image.width) // 2
        image_y = y + 8 + (image_h - image.height) // 2
        sheet.paste(image, (image_x, image_y))
        label = f'CLAW sample {row["sample_index"]:02d}'
        draw.text((x + 10, y + image_h + 12), label, fill="#111111", font=font)
        filename = row["filename"]
        if len(filename) > 62:
            filename = filename[:59] + "..."
        draw.text((x + 10, y + image_h + 32), filename, fill="#424242", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    args = parser.parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(f"CLAW archive not found: {args.archive}")

    with zipfile.ZipFile(args.archive) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"Archive integrity check failed at {bad_member}")
        image_members = sorted(
            name for name in archive.namelist()
            if name.lower().endswith("_input.jpg")
            and PurePosixPath(name).parent.name.lower() == "test"
        )
        if not image_members:
            raise RuntimeError("No CLAW test *_input.jpg images found in the archive")
        if args.sample_size < 1 or args.sample_size > len(image_members):
            raise ValueError(f"sample-size must be within 1..{len(image_members)}")
        rng = random.Random(args.seed)
        chosen = rng.sample(image_members, args.sample_size)
        selected_ids = {PurePosixPath(name).name[:-10] for name in chosen}

        rows = []
        member_by_id = {PurePosixPath(name).name[:-10]: name for name in image_members}
        for image_id, member in member_by_id.items():
            filename = PurePosixPath(member).name
            target = SOURCE / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file() or target.stat().st_size != archive.getinfo(member).file_size:
                target.write_bytes(archive.read(member))
            with Image.open(target) as image:
                width, height = image.size
                extension = image.format.lower() if image.format else target.suffix.lower().lstrip(".")
            rows.append({
                "image_id": image_id,
                "filename": filename,
                "absolute_path": str(target.resolve()),
                "file_extension": extension,
                "width": width,
                "height": height,
                "source_sha256": sha256(target),
                "reference_video_member": str(PurePosixPath(member).with_name(f"{image_id}.mp4")),
                "reference_video_in_archive": str(PurePosixPath(member).with_name(f"{image_id}.mp4")) in archive.namelist(),
                "selected": image_id in selected_ids,
                "sample_index": chosen.index(member) + 1 if image_id in selected_ids else "",
                "sample_seed": args.seed if image_id in selected_ids else "",
            })
        rows.sort(key=lambda row: row["image_id"])
        sample_rows = sorted((row.copy() for row in rows if row["selected"]), key=lambda row: row["sample_index"])

    manifest_fields = [
        "image_id", "filename", "absolute_path", "file_extension", "width", "height",
        "source_sha256", "reference_video_member", "reference_video_in_archive",
        "selected", "sample_index", "sample_seed",
    ]
    write_csv(SELECTION / "claw_test_manifest.csv", rows, manifest_fields)
    write_csv(SELECTION / "claw_sample_15.csv", sample_rows, manifest_fields)
    analysis_fields = [
        "image_id", "scene_category", "dynamic_elements", "estimated_dynamic_coverage",
        "estimated_static_coverage", "dense_dynamic_candidate", "camera_stationary_suitability", "notes",
    ]
    write_csv(SELECTION / "claw_sample_analysis.csv", [], analysis_fields)
    prompt_fields = ["image_id", "condition", "generated_prompt", "prompt_word_count"]
    write_csv(SELECTION / "claw_prompt_pairs.csv", [], prompt_fields)
    contact_sheet(sample_rows, SELECTION / "contact_sheet_sample_15.png")

    report = [
        "# CLAW v1 LoT-Pass pilot — Stage A review package", "",
        f"- Official source: {DATASET_URL}",
        f"- Code / dataset release: {SOURCE_URL}",
        f"- Archive SHA-256: `{sha256(args.archive)}`",
        f"- Test input images found: {len(rows)}", f"- Random sample: {args.sample_size} images", f"- Sampling seed: {args.seed}",
        "- Sampling method: uniform without replacement from sorted CLAW test input-image filenames.",
        "- Source resizing for generation: center crop to 720×480 after approval.",
        "- Conditions: frozen-scene control; broad fluid motion with fixed camera and solid background preserved.",
        "", "## Approval gate", "",
        "Review the 15 selected images and both prompts per image. Do not encode or generate until the user approves this package.",
        "", "## Selected images", "",
        "| Sample | Image ID | Dimensions | Reference clip |", "|---:|---|---:|---|",
    ]
    for row in sample_rows:
        report.append(
            f'| {row["sample_index"]} | `{row["image_id"]}` | {row["width"]}×{row["height"]} | '
            f'{"present" if row["reference_video_in_archive"] else "missing"} |'
        )
    (SELECTION / "selection_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Inventoried {len(rows)} CLAW test inputs; selected {len(sample_rows)} with seed {args.seed}.")
    print(f"Review contact sheet: {SELECTION / 'contact_sheet_sample_15.png'}")


if __name__ == "__main__":
    main()
