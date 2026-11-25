#!/usr/bin/env python3
"""Make a contact sheet for manual review; never invent motion-validity labels."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "vbench_lotpass_pilot"
SELECTION = EXP / "selection"
RUNS = EXP / "runs"
OUT = EXP / "analysis/motion"
SEED = 42


def case_dir(row: dict) -> Path:
    return RUNS / row["image_id"] / row["motion_group"] / f"seed_{SEED}"


def main() -> None:
    import decord
    from PIL import Image, ImageDraw, ImageFont

    rows = list(csv.DictReader((SELECTION / "vbench_40pairs.csv").open(encoding="utf-8")))
    OUT.mkdir(parents=True, exist_ok=True)
    thumb_w, thumb_h = 192, 108
    label_w, row_h = 250, 132
    canvas = Image.new("RGB", (label_w + thumb_w * 5, row_h * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        small = ImageFont.truetype("DejaVuSans.ttf", 11)
    except OSError:
        font = small = ImageFont.load_default()
    validation_rows = []
    for index, row in enumerate(rows):
        y = index * row_h
        directory = case_dir(row)
        video_path = directory / "video.mp4"
        status = json.loads((directory / "run_status.json").read_text()) if (directory / "run_status.json").exists() else {"status": "pending"}
        draw.text((8, y + 6), row["case_id"], fill="black", font=font)
        draw.text((8, y + 30), row["motion_full"], fill="black", font=small)
        draw.text((8, y + 50), status.get("status", "unknown"), fill="darkgreen" if status.get("status") == "completed" else "red", font=small)
        frames_done = 0
        if video_path.is_file() and status.get("status") == "completed":
            reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
            indices = sorted(set(round(i * (len(reader)-1) / 4) for i in range(5)))
            frames_done = len(reader)
            for col, frame_idx in enumerate(indices):
                image = Image.fromarray(reader[frame_idx].asnumpy()).convert("RGB")
                image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                canvas.paste(image, (label_w + col * thumb_w, y + 4))
                draw.text((label_w + col * thumb_w + 4, y + thumb_h + 6), f"frame {frame_idx}", fill="black", font=small)
        validation_rows.append({
            "image_id": row["image_id"], "case_id": row["case_id"], "motion_group": row["motion_group"],
            "motion_full": row["motion_full"], "seed": SEED, "generation_status": status.get("status", "unknown"),
            "frame_count": frames_done or "", "motion_validation_status": "not_reviewed",
            "validation_method": "manual review pending; review_sheet.png shows sampled actual frames" if frames_done else "no video available for review",
        })
        draw.line((0, y + row_h - 1, canvas.width, y + row_h - 1), fill=(200, 200, 200), width=1)
    canvas.save(OUT / "review_sheet.png", optimize=True)
    with (OUT / "motion_validation.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["image_id", "case_id", "motion_group", "motion_full", "seed", "generation_status", "frame_count", "motion_validation_status", "validation_method"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(validation_rows)
    print(f"Created motion review sheet for {len(rows)} scheduled cases; all labels remain not_reviewed")


if __name__ == "__main__":
    main()
