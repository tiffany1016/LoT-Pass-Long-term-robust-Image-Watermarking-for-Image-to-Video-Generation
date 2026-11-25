#!/usr/bin/env python3
"""Create the deterministic 16-sample / 32-prompt Stage A review package."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "chronomagic150_global_appearance"
DATA = EXP / "dataset"
FRAMES = EXP / "source_frames/original"
MODEL_FRAMES = EXP / "source_frames/model_input"
REVIEW = EXP / "stage_a_review"
TARGET_SIZE = (720, 480)
FRAME_JSON = DATA / "meteorological_first_frames.json"
COUNTS_JSON = DATA / "metadata_count_summary.json"
INVENTORY = DATA / "chronomagic150_meteorological_all.csv"
SELECTED_CSV = DATA / "chronomagic150_selected16.csv"
MANIFEST_CSV = DATA / "chronomagic150_generation_manifest.csv"

# Deterministic, pre-BA selection. night2day_08.0 is retained as described in
# its caption, including the complete night/day/night cycle. The other night2day
# sample is excluded because its initial frame is nearly uniformly black.
SELECTED = {
    "day2night_13.0": ("illumination_decrease", "yes", "Clear daytime-to-night sky transition; starts with a bright daytime sky and affects most of the frame.", "illumination"),
    "day2night_17.0": ("illumination_decrease", "yes", "Dusk-to-night city transition with scene-wide darkening and increasing city lights.", "illumination"),
    "night2day_08.0": ("illumination_cycle", "yes", "Official caption describes a full night→day→night cycle; first frame visibly matches its nighttime start.", "illumination"),
    "night_02": ("illumination_decrease", "yes", "Post-sunset-to-night transition changes illumination across sky, mountains, and foreground; the first frame retains visible scene structure.", "illumination"),
    "sunrise_05": ("illumination_increase", "yes", "Predawn-to-morning transition is visible across the horizon and landscape.", "illumination"),
    "sunrise_07": ("illumination_increase", "yes", "Dawn progressively illuminates a cityscape; the source frame matches early dawn.", "illumination"),
    "sunset_19": ("illumination_decrease", "yes", "Daylight-to-night city transition; the source frame is daytime as described.", "illumination"),
    "day_17": ("illumination_decrease", "yes", "Broad sunset-to-night change begins with the sun high in a cloudy sky.", "illumination"),
    "day_07": ("illumination_decrease", "yes", "Daylight-to-evening transition changes light and shadows over a lake and forested shore.", "illumination"),
    "nature_01": ("illumination_increase", "yes", "Night-to-dawn transition changes illumination across sky, horizon, water, and land.", "illumination"),
    "nature_14": ("illumination_decrease", "yes", "Day-to-sunset transition affects the sky and broad mountain-valley scene.", "illumination"),
    "valley_06": ("illumination_decrease", "yes", "Daylight softens into dusk over a broad valley; first frame matches daytime partly cloudy conditions.", "illumination"),
    "forest_11": ("weather_visibility", "yes", "Thick fog obscures most of a mountain range and is expected to clear scene-wide.", "weather_visibility"),
    "valley_15": ("weather_visibility", "yes", "Broad cloud cover disperses and sunlight spreads over a large valley.", "weather_visibility"),
    "storm_03": ("weather_visibility", "yes", "Clear daytime city conditions develop into a dark storm and wet scene.", "weather_visibility"),
    "snowstorm_18": ("weather_visibility", "yes", "Bare driveway and yard provide a verified clear initial state before widespread snow accumulation.", "weather_visibility"),
}

CONTROL_STATES = {
    "day2night_13.0": "bright daytime with scattered clouds",
    "day2night_17.0": "warm dusk with orange sunset light and city illumination at its initial level",
    "night2day_08.0": "nighttime with a crescent moon and city lights",
    "night_02": "post-sunset twilight with a bright horizon, visible stars, and a dimly lit landscape",
    "sunrise_05": "predawn darkness with a faint glow along the horizon",
    "sunrise_07": "early-dawn light over the city",
    "sunset_19": "daylight over the city beneath the bright sky",
    "day_17": "daytime beneath the cloudy sky, before sunset",
    "day_07": "daylight over the lake and forested shore",
    "nature_01": "starry nighttime with only a faint glow at the horizon",
    "nature_14": "daytime with the sun high over the mountainous landscape",
    "valley_06": "daylight with partly cloudy skies over the valley",
    "forest_11": "thick mist and low visibility across the mountain range",
    "valley_15": "mixed cloud cover and sunshine over the valley",
    "storm_03": "clear daytime weather with blue sky and bright sunlight over the city",
    "snowstorm_18": "a bare, snow-free driveway and yard under overcast daylight",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def derive_label_and_reason(row: dict[str, str]) -> tuple[str, str, str]:
    sample_id, caption = row["videoid"], row["name"]
    if sample_id in SELECTED:
        label, _, reason, group = SELECTED[sample_id]
        return label, group, reason
    text = caption.casefold()
    if any(token in text for token in ("fog", "mist", "storm", "snow", "rain", "weather", "visibility", "cloud cover")):
        label = "weather_visibility"
        group = "weather_visibility"
    elif any(token in text for token in ("day", "night", "sunrise", "sunset", "dawn", "dusk", "bright", "dark", "light")):
        label = "illumination_change"
        group = "illumination"
    elif any(token in text for token in ("color", "colour", "season")):
        label = "color_temperature"
        group = "illumination"
    else:
        label, group = "other_or_local_change", ""
    if any(token in text for token in ("boat", "vehicle", "mower", "window", "moon", "eclipse", "aurora", "stars", "grass", "tide", "river flowing")):
        return label, group, "Not selected: caption is mainly localized motion, an object/astronomical event, or a non-target change rather than broad scene-wide appearance transition."
    return label, group, "Not selected by the deterministic 16-case quota; retained as a pre-BA candidate/exclusion in the full inventory."


def make_prompt(sample_id: str, caption: str, transition: bool) -> str:
    shared_start = "Use the provided image as the first frame. Create a realistic, gently evolving time-lapse of the same scene."
    shared_end = (
        " Preserve the original composition, viewpoint, and recognizable objects. Keep the camera stationary. "
        "Allow only plausible natural motion already consistent with the scene; do not introduce unrelated objects or camera movement."
    )
    if transition:
        return f"{shared_start} Follow this appearance change: {caption}{shared_end}"
    state = CONTROL_STATES[sample_id]
    return (
        f"{shared_start} Keep the overall scene in {state} throughout the entire video; do not progress into a different "
        f"lighting, weather, or visibility state. Allow the same restrained natural motion as the transition version, "
        f"such as gentle cloud drift or foliage movement where present.{shared_end}"
    )


def make_contact_sheet(selected: list[dict]) -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    cols, width, height = 4, 420, 300
    rows = (len(selected) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * height), "white")
    draw = ImageDraw.Draw(sheet)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    title_font = ImageFont.truetype(bold_path, 17)
    body_font = ImageFont.truetype(font_path, 12)
    for index, row in enumerate(selected, start=1):
        x, y = ((index - 1) % cols) * width, ((index - 1) // cols) * height
        original = Image.open(ROOT / row["source_frame_original"]).convert("RGB")
        original.thumbnail((width - 16, 190), Image.Resampling.LANCZOS)
        draw.text((x + 8, y + 6), f"{index:02d} · {row['sample_id']} · {row['project_label']}", fill="#102c3d", font=title_font)
        sheet.paste(original, (x + 8, y + 31))
        caption = " ".join(row["official_caption"].split())
        words, lines, line = caption.split(), [], ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if draw.textbbox((0, 0), candidate, font=body_font)[2] > width - 16 and line:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        for offset, text in enumerate(lines[:4]):
            draw.text((x + 8, y + 230 + offset * 15), text, fill="#273b49", font=body_font)
    sheet.save(REVIEW / "selected16_contact_sheet.png", optimize=True)


def main() -> None:
    metadata = read_csv(DATA / "ChronoMagic-Bench-150.csv")
    all_rows = [row for row in metadata if row["main_category"].strip().casefold() == "meteorological"]
    frame_meta = {row["sample_id"]: row for row in json.loads(FRAME_JSON.read_text(encoding="utf-8"))}
    selected_ids = list(SELECTED)
    if len(selected_ids) != 16 or len({v[3] for v in SELECTED.values()}) != 2:
        raise RuntimeError("Selection must contain exactly 16 distinct cases in the two required groups")
    missing = [sample_id for sample_id in selected_ids if sample_id not in frame_meta]
    if missing:
        raise RuntimeError(f"Selected IDs lack an extracted official first frame: {missing}")

    MODEL_FRAMES.mkdir(parents=True, exist_ok=True)
    selected_rows, manifest_rows = [], []
    for index, sample_id in enumerate(selected_ids, start=1):
        original = next(row for row in all_rows if row["videoid"] == sample_id)
        original_path = FRAMES / f"{sample_id}.png"
        model_path = MODEL_FRAMES / f"{sample_id}.png"
        with Image.open(original_path) as image:
            source_size = image.size
            # Fixed content-preserving center crop to the repository's 720x480 I2V input geometry.
            fitted = ImageOps.fit(image.convert("RGB"), TARGET_SIZE, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            fitted.save(model_path, format="PNG", optimize=True)
        label, _, reason, group = SELECTED[sample_id]
        match = "matches" if sample_id != "sunset_14" else "mismatch"
        selected = {
            "selection_index": index, "sample_id": sample_id, "main_category": original["main_category"],
            "sub_category": original["sub_category"], "project_label": label,
            "analysis_group": group, "global_appearance_candidate": "yes", "selection_reason": reason,
            "official_caption": original["name"],
            "reference_video_url": "https://huggingface.co/datasets/BestWishYsh/ChronoMagic-Bench/tree/main/Videos/Reference_Videos.zip",
            "reference_video_member": f"{sample_id}.mp4",
            "reference_video_path": f"chronomagic150_global_appearance/reference_videos/{sample_id}.mp4",
            "reference_asset_status": "available_and_extracted",
            "source_frame_original": f"chronomagic150_global_appearance/source_frames/original/{sample_id}.png",
            "source_frame_model_input": f"chronomagic150_global_appearance/source_frames/model_input/{sample_id}.png",
            "frame_timestamp_seconds": frame_meta[sample_id]["timestamp_seconds"],
            "initial_state_match": match,
            "original_frame_size_wh": json.dumps([source_size[0], source_size[1]]),
            "model_input_size_wh": json.dumps(list(TARGET_SIZE)),
            "preprocessing": "PIL ImageOps.fit center crop then LANCZOS to 720x480",
            "transition_prompt_type": "adapted from official caption",
            "control_prompt_type": "custom no-transition control",
            "transition_prompt": make_prompt(sample_id, original["name"], True),
            "control_prompt": make_prompt(sample_id, original["name"], False),
        }
        selected_rows.append(selected)
        for condition, prompt in (("transition", selected["transition_prompt"]), ("control", selected["control_prompt"])):
            manifest_rows.append({
                "case_id": f"{sample_id}__{condition}", "sample_id": sample_id,
                "selection_index": index, "analysis_group": group, "project_label": label,
                "condition": condition, "source_image_path": selected["source_frame_model_input"],
                "official_caption": original["name"], "prompt": prompt,
                "prompt_type": selected["transition_prompt_type"] if condition == "transition" else selected["control_prompt_type"],
                "seed": 42, "seed_pair_policy": "same fixed seed 42 for both conditions in pair",
                "model_id": "THUDM/CogVideoX-5b-I2V", "width": 720, "height": 480,
                "frames": 49, "fps": 8, "inference_steps": 50, "guidance_scale": 5.0,
                "status": "awaiting user approval",
            })

    fields = list(selected_rows[0])
    write_csv(SELECTED_CSV, selected_rows, fields)
    write_csv(MANIFEST_CSV, manifest_rows, list(manifest_rows[0]))

    inventory_rows = []
    for row in all_rows:
        sample_id = row["videoid"]
        label, group, reason = derive_label_and_reason(row)
        frame_path = FRAMES / f"{sample_id}.png"
        inventory_rows.append({
            "sample_id": sample_id, "main_category": row["main_category"], "sub_category": row["sub_category"],
            "official_caption": row["name"],
            "reference_video_url": "https://huggingface.co/datasets/BestWishYsh/ChronoMagic-Bench/tree/main/Videos/Reference_Videos.zip",
            "reference_video_member": f"{sample_id}.mp4", "reference_video_path": f"chronomagic150_global_appearance/reference_videos/{sample_id}.mp4",
            "reference_asset_status": "available_and_extracted" if frame_path.is_file() else "unavailable",
            "project_label": label, "global_appearance_candidate": "yes" if sample_id in SELECTED else ("uncertain" if group else "no"),
            "selection_reason": SELECTED[sample_id][2] if sample_id in SELECTED else "",
            "exclusion_reason": "" if sample_id in SELECTED else reason,
            "source_frame_original": f"chronomagic150_global_appearance/source_frames/original/{sample_id}.png" if frame_path.is_file() else "",
            "frame_timestamp_seconds": frame_meta[sample_id]["timestamp_seconds"] if sample_id in frame_meta else "",
            "initial_state_match": "matches" if sample_id in SELECTED else "not_assessed_for_selection",
            "candidate_group": SELECTED[sample_id][3] if sample_id in SELECTED else group,
        })
    inventory_fields = list(inventory_rows[0])
    write_csv(INVENTORY, inventory_rows, inventory_fields)

    counts = json.loads(COUNTS_JSON.read_text(encoding="utf-8"))
    selected_counts = Counter(row["analysis_group"] for row in selected_rows)
    source_assets = json.loads((DATA / "reference_retrieval_status.json").read_text(encoding="utf-8"))
    with (REVIEW / "selection_index.md").open("w", encoding="utf-8") as stream:
        stream.write("# Selected sample index\n\nContact-sheet index numbers correspond to rows in `../dataset/chronomagic150_selected16.csv`.\n\n")
        for row in selected_rows:
            stream.write(f"{row['selection_index']:02d}. **{row['sample_id']}** — {row['analysis_group']} — transition/control prompts are in the linked CSV row.\n")
    report = [
        "# ChronoMagic-150 Stage A selection review",
        "",
        "## Dataset and asset census",
        "",
        f"- Official source: `BestWishYsh/ChronoMagic-Bench`, file `Captions/ChronoMagic-Bench-150.csv` (SHA-256 `{counts['source_csv_sha256']}`).",
        f"- Metadata rows: **{counts['total_records']}**; normalized `meteorological` rows: **{counts['normalized_meteorological_count']}**.",
        f"- Main-category counts (original strings): `{json.dumps(counts['main_category_counts_preserving_blank'], ensure_ascii=False)}`.",
        f"- Meteorological subcategory counts: `{json.dumps(counts['meteorological_subcategory_counts'], ensure_ascii=False)}`.",
        f"- The official ZIP contains {source_assets['zip_member_count']} members; all {source_assets['expected_metadata_members_found']} of 56 meteorological IDs were present, extracted and decoded at frame zero.",
        "- The 12.45 GB archive was accessed using HTTP byte ranges; only the 56 meteorological MP4 members were extracted (about 422 MB total). ZIP CRCs, extracted sizes, and per-video SHA-256 values are recorded in `../dataset/reference_retrieval_status.json`.",
        "- Initial frames use the first decodable frame at timestamp 0.0 seconds. The original frames are preserved; selected model inputs use a center crop and LANCZOS resize to 720×480.",
        "",
        "## Selection rule and counts",
        "",
        "- Exactly 16 distinct cases were selected before any LoT-Pass encoding or bit-accuracy analysis: 12 illumination-related and 4 weather/visibility-related.",
        "- Deterministic ranking: include both `day2night` and `sunrise` samples, the interpretable-composition `night2day_08.0` cycle, then `night_02`, `sunset_19`, both `day` samples, both `nature` samples, and `valley_06`. The other `night2day_15.0` starts nearly uniformly black with little visible scene structure, so `night_02` provides the nightfall case with an interpretable source image. Weather/visibility selection covers fog clearing, cloud clearing, storm formation, and snow accumulation.",
        "- No selection used LoT-Pass BA. IDs, captions, category strings, and source assets remain traceable in the full inventory.",
        f"- Selected allocation: `{dict(selected_counts)}`. The `night2day_08.0` caption describes a full night→day→night cycle; it remains included because its initial frame matches the stated nighttime start, and the transition prompt preserves that complete captioned cycle.",
        "- `sunset_14` was not selected: the caption says the sun starts high in the sky, while its first reference frame already shows the sun at the horizon. `lake_04` was likewise not used for the sunset quota because its first frame is already near sunset despite the caption's high-sun starting state.",
        "",
        "## Review artifacts",
        "",
        "- `../dataset/chronomagic150_meteorological_all.csv`: all meteorological records, project labels, asset availability, and selection/exclusion rationale.",
        "- `../dataset/chronomagic150_selected16.csv`: selected source metadata, verified first frames, and both verbatim prompts.",
        "- `../dataset/chronomagic150_generation_manifest.csv`: 32 paired generation cases, fixed settings, and seed policy; status remains awaiting approval.",
        "- `selected16_contact_sheet.png` and `selection_index.md`: original-ID visual review and mapping to full prompts.",
        "",
        "## Approval gate",
        "",
        "This package is for review only. No LoT-Pass image encoding, CogVideoX generation, watermark decoding, or GPU analysis has been run. Stage B will start only after explicit user approval.",
    ]
    (REVIEW / "selection_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    make_contact_sheet(selected_rows)
    print(f"Wrote Stage A review: {len(selected_rows)} samples, {len(manifest_rows)} prompt cases; {dict(selected_counts)}")


if __name__ == "__main__":
    main()
