#!/usr/bin/env python3
"""Build the VBench++ LoT-Pass pilot's Stage A selection package."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import subprocess
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "vbench_lotpass_pilot"
DOWNLOADS = EXP / "downloads"
SELECTION = EXP / "selection"
ORIGINALS = EXP / "images/official/original"
OFFICIAL_CROPS = EXP / "images/official/crop_8-5"
CANONICAL = EXP / "images/canonical_clean"
FULL_INFO = DOWNLOADS / "vbench2_i2v_full_info.json"
SOURCE_INFO = DOWNLOADS / "i2v-bench-info.json"
SEED = 42
TARGET_SIZE = (720, 480)  # width, height
TARGET_RATIO = 3 / 2
CROP_RATIO = "8-5"

MOTIONS = {
    "static": "camera static",
    "pan_left": "camera pans left",
    "pan_right": "camera pans right",
    "tilt_up": "camera tilts up",
    "tilt_down": "camera tilts down",
    "zoom_in": "camera zooms in",
    "zoom_out": "camera zooms out",
}
MOTION_GROUP = {
    "static": "static", "pan_left": "pan", "pan_right": "pan",
    "tilt_up": "tilt", "tilt_down": "tilt", "zoom_in": "zoom", "zoom_out": "zoom",
}
CATEGORIES = ("architecture", "indoor", "scenery", "abstract")
QUOTAS = {"architecture": 3, "indoor": 3, "scenery": 3, "abstract": 1}
DRIVE = {
    "crop": ("1Y_JnYnyJ3a6QhiranoX0MQVZFcTDPekZ", "crop.zip"),
    "origin": ("1qhkLCSBkzll0dkKpwlDTwLL0nxdQ4nrY", "origin.zip"),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class HTTPRangeFile:
    """Minimal seekable file backed by HTTP byte-range requests."""

    def __init__(self, url: str):
        self.url = url
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            content_range = response.headers.get("Content-Range", "")
            if response.status != 206 or "/" not in content_range:
                raise RuntimeError(f"Drive did not return a byte range for {url}: {response.status}")
            self.size = int(content_range.rsplit("/", 1)[1])
            response.read()
        self.pos = 0

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            pos = offset
        elif whence == 1:
            pos = self.pos + offset
        elif whence == 2:
            pos = self.size + offset
        else:
            raise ValueError("invalid whence")
        if pos < 0:
            raise ValueError("negative seek position")
        self.pos = min(pos, self.size)
        return self.pos

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            amount = self.size - self.pos
        amount = min(amount, self.size - self.pos)
        if amount <= 0:
            return b""
        end = self.pos + amount - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(request, timeout=180) as response:
            data = response.read()
            if response.status != 206 or len(data) != amount:
                raise RuntimeError(f"Incomplete range response at {self.pos}-{end}")
        self.pos += len(data)
        return data


def drive_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def csv_write(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract_members(archive_kind: str, member_names: list[str], output_dir: Path) -> dict[str, Path]:
    file_id, _ = DRIVE[archive_kind]
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    with zipfile.ZipFile(HTTPRangeFile(drive_url(file_id))) as archive:
        for member in member_names:
            if member not in archive.namelist():
                raise FileNotFoundError(f"Official VBench archive lacks {member}")
            data = archive.read(member)
            out = output_dir / Path(member).name
            out.write_bytes(data)
            result[Path(member).name] = out
    return result


def select_images(eligible_by_category: dict[str, list[str]]) -> tuple[list[tuple[str, str]], dict]:
    rng = random.Random(SEED)
    chosen: list[tuple[str, str]] = []
    remaining_quota = {}
    for category in CATEGORIES:
        names = sorted(eligible_by_category.get(category, []))
        count = min(QUOTAS[category], len(names))
        sampled = rng.sample(names, count) if count else []
        chosen.extend((category, name) for name in sampled)
        remaining_quota[category] = QUOTAS[category] - count

    deficit = sum(remaining_quota.values())
    fallback = []
    if deficit:
        already = {name for _, name in chosen}
        pool = []
        for category in CATEGORIES:
            pool.extend((category, name) for name in sorted(eligible_by_category.get(category, [])) if name not in already)
        if len(pool) < deficit:
            raise RuntimeError(f"Only {len(chosen) + len(pool)} eligible images; cannot fill 10-image pilot")
        sampled = rng.sample(pool, deficit)
        chosen.extend(sampled)
        fallback = [{"image_name": name, "filled_category": category} for category, name in sampled]
    if len(chosen) != 10 or len({name for _, name in chosen}) != 10:
        raise RuntimeError("Selection did not produce 10 unique images")
    return chosen, {"fallback_used": bool(fallback), "fallback_assignments": fallback}


def make_contact_sheet(images: list[dict]) -> None:
    cell_w, cell_h = 480, 350
    cols = 2
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font, small = ImageFont.load_default(), ImageFont.load_default()
    for index, item in enumerate(images):
        x = (index % cols) * cell_w
        y = (index // cols) * cell_h
        with Image.open(ROOT / item["canonical_image_path"]) as source:
            preview = source.convert("RGB")
            preview.thumbnail((cell_w - 24, 270), Image.Resampling.LANCZOS)
        sheet.paste(preview, (x + (cell_w - preview.width) // 2, y + 8))
        short = item["image_name"]
        if len(short) > 52:
            short = short[:49] + "..."
        draw.text((x + 12, y + 285), f'{item["image_id"]} · {item["image_type"]}', fill="black", font=font)
        draw.text((x + 12, y + 313), short, fill="black", font=small)
    sheet.save(SELECTION / "contact_sheet.png", format="PNG", optimize=True)


def main() -> None:
    for path in (FULL_INFO, SOURCE_INFO):
        if not path.is_file():
            raise FileNotFoundError(path)
    info = load_json(FULL_INFO)
    source_info = load_json(SOURCE_INFO)
    camera_entries = [(index, item) for index, item in enumerate(info) if "camera_motion" in item.get("dimension", [])]

    camera_rows = []
    grouped: dict[str, dict[str, list[tuple[int, dict]]]] = defaultdict(lambda: defaultdict(list))
    unknown_suffixes = []
    for index, item in camera_entries:
        prompt = item.get("prompt_en", "")
        motion = next((label for label, suffix in MOTIONS.items() if prompt.endswith(suffix)), None)
        if motion is None:
            unknown_suffixes.append({"metadata_index": index, "prompt_en": prompt})
        else:
            grouped[item["image_name"]][motion].append((index, item))
        camera_rows.append({
            "metadata_index": index,
            "image_name": item.get("image_name", ""),
            "image_type": item.get("image_type", ""),
            "motion_full": motion or "UNKNOWN",
            "prompt_en": prompt,
            "dimension": json.dumps(item.get("dimension", []), ensure_ascii=False),
            "original_record_json": json.dumps(item, ensure_ascii=False, sort_keys=True),
        })
    if unknown_suffixes:
        raise RuntimeError(f"Unknown camera-motion prompt suffixes: {unknown_suffixes[:5]}")

    duplicate_report = []
    eligible_names = []
    rejected = []
    for name in sorted(grouped):
        variants = grouped[name]
        if set(variants) != set(MOTIONS):
            rejected.append({"image_name": name, "reason": "missing_motion_variants", "available": sorted(variants)})
            continue
        duplicate_conflict = False
        for motion, records in variants.items():
            distinct = {json.dumps(item, sort_keys=True, ensure_ascii=False) for _, item in records}
            if len(records) > 1:
                duplicate_report.append({"image_name": name, "motion_full": motion, "count": len(records), "identical": len(distinct) == 1})
            if len(distinct) > 1:
                duplicate_conflict = True
        if duplicate_conflict:
            rejected.append({"image_name": name, "reason": "conflicting_duplicate_motion_records"})
        else:
            eligible_names.append(name)
    csv_write(SELECTION / "official_camera_motion_all.csv", camera_rows,
              ["metadata_index", "image_name", "image_type", "motion_full", "prompt_en", "dimension", "original_record_json"])

    source_by_name = {item.get("file_name"): item for item in source_info}
    category_by_name = {}
    for name in eligible_names:
        records = [item for _, item in next(iter(grouped[name].values()))]
        image_type = records[0].get("image_type", "")
        src = source_by_name.get(name)
        if src and src.get("type") != image_type:
            raise RuntimeError(f"image_type mismatch for {name}: prompt metadata={image_type}, source metadata={src.get('type')}")
        category_by_name[name] = image_type
    eligible_by_category = {category: sorted(name for name in eligible_names if category_by_name[name] == category) for category in CATEGORIES}
    eligible_rows = []
    for name in eligible_names:
        variants = grouped[name]
        row = {"image_name": name, "image_type": category_by_name[name], "has_all_seven_variants": True,
               "variant_count": len(variants), "duplicate_records": sum(max(0, len(v) - 1) for v in variants.values()),
               "source_metadata_present": name in source_by_name}
        for motion in MOTIONS:
            row[f"{motion}_prompt"] = variants[motion][0][1]["prompt_en"]
        eligible_rows.append(row)
    csv_write(SELECTION / "eligible_images.csv", eligible_rows,
              ["image_name", "image_type", "has_all_seven_variants", "variant_count", "duplicate_records", "source_metadata_present"] + [f"{m}_prompt" for m in MOTIONS])

    selected, fallback = select_images(eligible_by_category)
    # Resolve archive members only for selected images and their official 8:5 crops.
    image_names = [name for _, name in selected]
    origin_members = [f"origin/{name}" for name in image_names]
    crop_members = [f"crop/{CROP_RATIO}/{name}" for name in image_names]
    origin_paths = extract_members("origin", origin_members, ORIGINALS)
    crop_paths = extract_members("crop", crop_members, OFFICIAL_CROPS)

    selected_rows = []
    selected_manifest = []
    for selected_index, (category, name) in enumerate(selected, start=1):
        image_id = f"VB{selected_index:03d}"
        original_path = origin_paths[name]
        crop_path = crop_paths[name]
        with Image.open(original_path) as image:
            image.verify()
        with Image.open(crop_path) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                raise RuntimeError(f"Invalid crop image: {crop_path}")
            crop_w, crop_h = image.size
            if crop_w / crop_h < TARGET_RATIO:
                new_h = int(crop_w / TARGET_RATIO)
                top = (crop_h - new_h) // 2
                transform = {"crop_box_xywh": [0, top, crop_w, new_h], "axis": "vertical"}
            else:
                new_w = int(crop_h * TARGET_RATIO)
                left = (crop_w - new_w) // 2
                transform = {"crop_box_xywh": [left, 0, new_w, crop_h], "axis": "horizontal"}
            clean = image.convert("RGB").crop((transform["crop_box_xywh"][0], transform["crop_box_xywh"][1],
                                                 transform["crop_box_xywh"][0] + transform["crop_box_xywh"][2],
                                                 transform["crop_box_xywh"][1] + transform["crop_box_xywh"][3]))
            clean = clean.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        canonical_path = CANONICAL / f"{image_id}.png"
        clean.save(canonical_path, format="PNG", optimize=True)
        row = {
            "image_id": image_id, "selected_index": selected_index, "image_name": name, "image_type": category,
            "official_image_path": relative(original_path), "official_crop_path": relative(crop_path),
            "canonical_image_path": relative(canonical_path), "source": "VBench++ official I2V image suite",
            "selection_seed": SEED,
        }
        selected_rows.append(row)
        selected_manifest.append({
            **row,
            "source_url": source_by_name.get(name, {}).get("url"),
            "original_sha256": sha256_file(original_path), "official_crop_sha256": sha256_file(crop_path),
            "canonical_sha256": sha256_file(canonical_path),
            "crop_variant": CROP_RATIO, "crop_transform": transform,
        })
    csv_write(SELECTION / "selected_10_images.csv", selected_rows,
              ["image_id", "selected_index", "image_name", "image_type", "official_image_path", "official_crop_path", "canonical_image_path", "source", "selection_seed"])

    pair_rows = []
    for selected_row in selected_rows:
        index = selected_row["selected_index"]
        chosen_motion = ["static", "pan_left" if index % 2 else "pan_right",
                         "tilt_up" if index % 2 else "tilt_down",
                         "zoom_in" if index % 2 else "zoom_out"]
        for motion in chosen_motion:
            record = grouped[selected_row["image_name"]][motion][0][1]
            pair_rows.append({
                "case_id": f'{selected_row["image_id"]}_{MOTION_GROUP[motion]}',
                "image_id": selected_row["image_id"], "selected_index": index,
                "image_name": selected_row["image_name"], "image_type": selected_row["image_type"],
                "motion_group": MOTION_GROUP[motion], "motion_full": motion,
                "prompt": record["prompt_en"], "official_image_path": selected_row["official_image_path"],
                "canonical_image_path": selected_row["canonical_image_path"], "source": selected_row["source"],
                "selection_seed": SEED,
            })
    csv_write(SELECTION / "vbench_40pairs.csv", pair_rows,
              ["case_id", "image_id", "selected_index", "image_name", "image_type", "motion_group", "motion_full", "prompt", "official_image_path", "canonical_image_path", "source", "selection_seed"])

    # Confirm the planned directional balance and exact prompt pairing.
    motion_counts = Counter(row["motion_full"] for row in pair_rows)
    if len(pair_rows) != 40 or motion_counts != Counter({"static": 10, "pan_left": 5, "pan_right": 5, "tilt_up": 5, "tilt_down": 5, "zoom_in": 5, "zoom_out": 5}):
        raise RuntimeError(f"Unexpected pilot pair balance: {motion_counts}")

    make_contact_sheet(selected_rows)
    metadata_hashes = {FULL_INFO.name: sha256_file(FULL_INFO), SOURCE_INFO.name: sha256_file(SOURCE_INFO)}
    category_counts = Counter(row["image_type"] for row in selected_rows)
    manifest = {
        "study": "VBench++ I2V x LoT-Pass camera-motion pilot",
        "selection_seed": SEED,
        "selection_method": "Python random.Random(42).sample over lexically sorted eligible filenames, stratified by category",
        "requested_category_quota": QUOTAS,
        "actual_category_distribution": dict(category_counts),
        "fallback": fallback,
        "source_counts": {
            "full_info_records": len(info), "camera_motion_records": len(camera_entries),
            "camera_motion_images": len(grouped), "eligible_images": len(eligible_names),
            "source_metadata_images": len(source_info), "unknown_suffixes": unknown_suffixes,
            "duplicate_records": duplicate_report, "rejected_images": rejected,
        },
        "metadata_sha256": metadata_hashes,
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "preprocessing": {
            "official_crop_variant": CROP_RATIO,
            "policy": "Center-crop official 8:5 crop to 3:2, then resize to 720x480 with Pillow LANCZOS; no stretching or padding.",
            "target_size_width_height": list(TARGET_SIZE),
            "implementation": "Pillow " + Image.__version__,
        },
        "selected_images": selected_manifest,
        "motion_counts": dict(motion_counts),
    }
    (SELECTION / "selection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    selection_hash = sha256_file(SELECTION / "selection_manifest.json")

    lines = [
        "# VBench++ LoT-Pass pilot: Stage A selection for approval", "",
        "This package contains a deterministic 10-image sample and 40 verbatim official camera-motion prompts. No watermark encoding, video generation, decoding, or GPU analysis has been run.", "",
        "## Source and selection", "",
        f'- Official VBench++ prompt metadata: {len(info)} entries; {len(camera_entries)} camera-motion entries across {len(grouped)} images.',
        f'- Eligible images: {len(eligible_names)} with all seven motion variants; unknown suffixes: {len(unknown_suffixes)}; conflicting duplicates: {sum(1 for x in rejected if x["reason"] == "conflicting_duplicate_motion_records")}.',
        f'- Selected categories: {dict(category_counts)}; selection seed: {SEED}.',
        f'- Official images are from the VBench++ image suite. Canonical inputs use the official {CROP_RATIO} crop, center-cropped to 3:2 and resized to 720x480 with LANCZOS.',
        f'- Contact sheet: `{relative(SELECTION / "contact_sheet.png")}`.',
        f'- Metadata checksums and per-image hashes are recorded in `{relative(SELECTION / "selection_manifest.json")}`.', "",
        f'- Selection manifest SHA-256 (approval reference): `{selection_hash}`.', "",
        "## Selected images", "",
        "| ID | Scene | Official image | Canonical input |", "|---|---|---|---|",
    ]
    for row in selected_rows:
        lines.append(f'| {row["image_id"]} | {row["image_type"]} | {row["image_name"]} | `{row["canonical_image_path"]}` |')
    lines += ["", "## Forty official prompts", "", "| Case | Motion | Prompt (copied verbatim) |", "|---|---|---|"]
    for row in pair_rows:
        prompt = row["prompt"].replace("|", "\\|").replace("\n", " ")
        lines.append(f'| {row["case_id"]} | {row["motion_full"]} | {prompt} |')
    lines += ["", "## Files", "", "- `official_camera_motion_all.csv`: all official camera-motion metadata records.",
              "- `eligible_images.csv`: per-image seven-variant eligibility audit.",
              "- `selected_10_images.csv`: selected image IDs, provenance, and paths.",
              "- `vbench_40pairs.csv`: selected image-prompt pairs.",
              "- `contact_sheet.png`: canonical generator-input previews.",
              "- `selection_manifest.json`: seeds, metadata checksums, transforms, and image hashes.", "",
              "**Approval gate:** Stage B must not begin until the user explicitly approves this exact selection manifest. Approval should reference the SHA-256 of `selection_manifest.json`.", ""]
    (SELECTION / "approval_request.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Stage A package created: {relative(SELECTION)}")
    print(f"Selected images: {len(selected_rows)}; pairs: {len(pair_rows)}; motion counts: {dict(motion_counts)}")


if __name__ == "__main__":
    main()
