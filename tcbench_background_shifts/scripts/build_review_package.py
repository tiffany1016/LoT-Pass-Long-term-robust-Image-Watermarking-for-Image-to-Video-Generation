#!/usr/bin/env python3
"""Draft matched controls and create the all-entry approval package."""
from __future__ import annotations

import csv
import hashlib
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "tcbench_background_shifts"
METADATA = EXP / "metadata"
APPROVAL = EXP / "approval"
RETRIEVAL_STATUS = json.loads((METADATA / "reference_retrieval_status.json").read_text(encoding="utf-8"))
EXPECTED_IDS = [101, 102, 103, 104, 105, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]

# Candidate controls are grounded in the TC-Bench prompt and the extracted frame.
# Ambiguous or unavailable inputs are deliberately marked for user review.
DRAFTS = {
    101: dict(initial="A cityscape at sunset.", target="The scene changes from sunset lighting to night.",
              removed="The change from sunset to night.", preserved="The cityscape continues naturally under sunset lighting.",
              prompt="A cityscape at sunset. The scene continues naturally while the sunset lighting remains throughout the video.",
              note="The extracted frame shows the city at sunset, matching the prompt's stated initial state.", match="yes",
              mechanism="illumination", whole="likely", camera="no", review=False),
    102: dict(initial="A forest in summer greenery.", target="The forest changes from green summer conditions to winter snow.",
              removed="The seasonal change from summer greenery to winter snow.", preserved="The forest remains green while leaves and branches move naturally.",
              prompt="A forest in summer greenery. Leaves and branches move naturally while the trees remain green and no snow appears.",
              note="The extracted frame shows a green summer forest, matching the prompt's initial state.", match="yes",
              mechanism="environment-material", whole="likely", camera="no", review=False),
    103: dict(initial="A sunny afternoon transitioning to a serene sunset.", target="The sky and lighting become sunset-like.",
              removed="The transition from sunny afternoon to sunset.", preserved="A low sun and orange sunset sky remain steady as the scene continues naturally.",
              prompt="A low sun sets over a dark horizon beneath an orange sky. The scene continues naturally while the sunset lighting remains steady.",
              note="The extracted frame already shows an orange sunset, contradicting the prompt's sunny-afternoon initial state; control follows the visible frame and is flagged.", match="no",
              mechanism="illumination", whole="likely", camera="no", review=True),
    104: dict(initial="A desk lamp stands on the floor beside the PIXAR letters.", target="The lamp jumps from the ground onto the letter I.",
              removed="The lamp jumping from the ground onto the letter I.", preserved="The logo scene and lamp remain visible; the lamp may make small natural movements while staying on the floor.",
              prompt="The PIXAR letters stand across a blue floor. The desk lamp makes small natural movements beside the letters while staying on the floor, without jumping onto a letter.",
              note="The frame shows the lamp on the floor beside the letters, before the described jump.", match="yes",
              mechanism="local-object-event", whole="unlikely", camera="no", review=False),
    105: dict(initial="A star-filled night sky above a black mountain silhouette.", target="A meteor streaks from the upper left toward the lower right.",
              removed="The meteor crossing the sky.", preserved="The starry sky and mountain silhouette remain; stars and faint atmospheric light continue naturally.",
              prompt="A star-filled night sky above a black silhouette of a mountain range. The sky remains free of meteors as the scene continues naturally.",
              note="A bright streak is already visible in the extracted frame; its relation to the prompt's meteor event is uncertain, so the control needs review.", match="uncertain",
              mechanism="local-object-event", whole="unlikely", camera="no", review=True),
    106: dict(initial="A rainy sky over a distant horizon.", target="A rainbow appears as the rainy sky changes.",
              removed="The appearance of a rainbow and change from rainy conditions.", preserved="Rain and gray skies continue naturally without a rainbow.",
              prompt="Rain falls beneath a gray sky over a distant horizon. The rainy weather continues naturally and no rainbow appears.",
              note="The extracted frame shows raindrops against a gray sky, matching the prompt's rainy initial state.", match="yes",
              mechanism="weather", whole="uncertain", camera="no", review=False),
    108: dict(initial="A girl watches a bird at the right side of the sky above the ocean.", target="The bird flies from right to left across the sky.",
              removed="The bird's right-to-left flight across the sky.", preserved="The girl, ocean, and sails remain; the bird stays near the right side while water and sails move naturally.",
              prompt="A girl watches a bird in the sky above the ocean. The bird remains near the right side of the sky while the water and sails move naturally.",
              note="The extracted frame shows the girl and a bird near the upper-right sky, consistent with the prompt's starting configuration.", match="yes",
              mechanism="local-object-event", whole="unlikely", camera="no", review=False),
    109: dict(initial="The Rockefeller Center Christmas tree lies horizontally on the ground in the plaza.", target="The tree is hoisted upright.",
              removed="The tree being lifted from horizontal to upright.", preserved="The plaza remains active; workers and the crane can move around the tree while it stays on the ground.",
              prompt="A view of Rockefeller Center with its Christmas tree lying horizontally on the ground. Workers and the crane continue around it while the tree remains on the ground.",
              note="The extracted frame shows the tree horizontal beside the crane, matching the prompt's starting state.", match="yes",
              mechanism="local-object-event", whole="unlikely", camera="no", review=False),
    110: dict(initial="Two tall trees are covered with snow.", target="The snow melts from the trees.",
              removed="The snow melting from the trees.", preserved="The snow-covered trees remain snowy as the branches and surrounding scene move naturally.",
              prompt="Two tall trees remain covered with snow. The surrounding scene continues naturally without the snow melting.",
              note="The extracted frame shows snow-covered trees, matching the prompt's initial state.", match="yes",
              mechanism="environment-material", whole="unlikely", camera="no", review=False),
    111: dict(initial="A candelabra hangs at the upper right in front of a large golden chandelier.", target="The candelabra swings toward the camera in front of the chandelier.",
              removed="The candelabra's approach toward the camera.", preserved="The golden chandelier stays in view while the candelabra sways gently near the upper right.",
              prompt="A cartoon scene shows a large golden chandelier with a candelabra near the upper right. The candelabra sways gently in place without moving toward the camera.",
              note="The extracted frame shows the candelabra at the upper right and the chandelier in the background.", match="yes",
              mechanism="local-object-event", whole="unlikely", camera="no", review=False),
    112: dict(initial="A purple bunny watches a rainbow stream from a bottle on its right.", target="The rainbow stream travels from right to left and forms an arc above the bunny.",
              removed="The rainbow stream traveling across the frame and forming the arc.", preserved="The bunny and bottle remain in the scene; the rainbow stays near the bottle while the scene continues naturally.",
              prompt="A purple bunny beside a bottle. The rainbow stays in its initial position on the right while the scene continues naturally, without traveling across the frame.",
              note="The extracted frame already shows the bunny and a rainbow arc over the bottle; the transition may have begun before the official interval. Review the proposed held-state control.", match="uncertain",
              mechanism="local-object-event", whole="unlikely", camera="no", review=True),
    113: dict(initial="A lighthouse rotates against a dark night sky.", target="The background changes from dark night to sunrise.",
              removed="The transition from night to sunrise.", preserved="The lighthouse continues rotating and its beam sweeps while night lighting remains consistent.",
              prompt="A lighthouse rotates beneath a dark night sky. Its beam sweeps naturally while the night lighting remains consistent and no sunrise appears.",
              note="The extracted frame shows a lighthouse beam against a dark night sky, matching the prompt's initial state.", match="yes",
              mechanism="illumination", whole="likely", camera="no", review=False),
    114: dict(initial="A Disney castle at night with fireworks already visible in the sky.", target="A firework trail sweeps from right to left and forms an arc over the castle.",
              removed="The firework trail crossing and forming an arc above the castle.", preserved="The castle remains at night; the existing fireworks stay in their initial visible configuration without a new trail sweeping across the sky.",
              prompt="A Disney castle at night with the fireworks already visible in the sky. The existing fireworks remain in their current positions; no new trail sweeps across the sky or forms an arc.",
              note="The extracted frame already shows fireworks over the castle, so it does not match a pre-fireworks initial state. The custom control preserves the visible fireworks while removing the described sweep; review this adaptation.", match="no",
              mechanism="local-object-event", whole="unlikely", camera="no", review=True),
    115: dict(initial="A crowd watches an airplane while it is still distant in the sky.", target="The airplane grows larger as it flies toward the camera.",
              removed="The airplane's approach toward the camera and resulting size increase.", preserved="The crowd continues watching while the airplane remains distant and small in the sky.",
              prompt="A crowd watches a distant airplane in the sky. The crowd continues naturally while the airplane remains small and far away.",
              note="The extracted frame shows spectators and an airplane in the distance, consistent with the prompt's starting state.", match="yes",
              mechanism="local-object-event", whole="unlikely", camera="no", review=False),
    116: dict(initial="An empty street view.", target="The street becomes busy with people and activity.",
              removed="The transition from empty to busy.", preserved="The street remains quiet and empty while the scene continues naturally.",
              prompt="An empty street scene with buildings and road. The street remains quiet and empty while the scene continues naturally.",
              note="The extracted frame shows an empty street, matching the prompt's initial state.", match="yes",
              mechanism="local-object-event", whole="uncertain", camera="no", review=False),
    117: dict(initial="An In-N-Out Burger sign with an airplane at the left side of the background sky.", target="The airplane flies from left to right over the sign.",
              removed="The airplane's left-to-right flight over the sign.", preserved="The sign remains in view while the airplane stays in the background sky and the scene continues naturally.",
              prompt="An In-N-Out Burger sign beneath an airplane in the background sky. The airplane remains in the sky without flying over the sign, while the scene continues naturally.",
              note="The extracted frame shows the sign with a plane near the left side of the sky, consistent with the described initial setup.", match="yes",
              mechanism="local-object-event", whole="unlikely", camera="no", review=False),
    118: dict(initial="A space dog watches a robot arm above a stone on the ground.", target="The robot arm picks up the stone.",
              removed="The stone being picked up from the ground.", preserved="The dog watches as the arm moves subtly, while the stone stays on the ground.",
              prompt="A space dog watches a robot arm above a stone on the ground. The dog and arm move subtly while the stone remains on the ground and is not picked up.",
              note="The extracted frame shows the dog, arm, and stone before the pickup is complete.", match="yes",
              mechanism="local-object-event", whole="unlikely", camera="no", review=False),
    119: dict(initial="A crowd cheers around a huge Christmas tree at night.", target="Fireworks explode over the tree.",
              removed="The fireworks appearing and exploding over the tree.", preserved="The crowd and tree remain in the night scene; the sky stays dark without fireworks.",
              prompt="A crowd cheers around a huge Christmas tree at night. The crowd continues naturally while the sky stays dark and free of fireworks.",
              note="The extracted frame already shows fireworks over the tree, while the official prompt gives no explicit pre-fireworks state; control and initial-state match need review.", match="uncertain",
              mechanism="local-object-event", whole="unlikely", camera="no", review=True),
}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_contact_sheet(rows: list[dict]) -> None:
    font = ImageFont.load_default()
    cols, cell_w, cell_h, image_h = 3, 440, 320, 236
    sheet = Image.new("RGB", (cols * cell_w, 6 * cell_h), (246, 246, 246))
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        x, y = (index % cols) * cell_w, (index // cols) * cell_h
        image_path = ROOT / row.get("source_image_original_path", "")
        if row.get("download_status") == "downloaded" and image_path.is_file():
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((cell_w - 16, image_h - 8))
            sheet.paste(image, (x + 8, y + 4))
        else:
            draw.rectangle((x + 8, y + 4, x + cell_w - 8, y + image_h), fill=(220, 220, 220), outline=(110, 110, 110), width=2)
            draw.multiline_text((x + 22, y + 85), f"MISSING SOURCE\n{row.get('download_status', 'unknown')}",
                                fill=(140, 0, 0), font=font, spacing=6)
        draw.rectangle((x, y + image_h, x + cell_w, y + cell_h), fill="white")
        initial = DRAFTS[int(row["official_id"])]["initial"]
        prompt = " ".join(row["prompt_official"].split())
        status = row.get("download_status", "unknown")
        draw.text((x + 8, y + image_h + 5), f"{row['case_id']} | {status} | initial: {initial}", fill="black", font=font)
        draw.multiline_text((x + 8, y + image_h + 20), "\n".join(textwrap.wrap(prompt, width=68)[:2]),
                            fill=(45, 45, 45), font=font, spacing=2)
    APPROVAL.mkdir(parents=True, exist_ok=True)
    sheet.save(APPROVAL / "contact_sheet_all_18.png")


def main() -> None:
    source_rows = list(csv.DictReader((METADATA / "all_18_initial.csv").open(encoding="utf-8")))
    ids = [int(row["official_id"]) for row in source_rows]
    if ids != EXPECTED_IDS:
        raise RuntimeError(f"Unexpected official IDs: {ids}")
    if set(DRAFTS) != set(EXPECTED_IDS):
        raise RuntimeError("Control draft table does not cover the exact official ID set")

    image_rows, pair_rows = [], []
    missing_count = 0
    for row in source_rows:
        official_id = int(row["official_id"])
        draft = DRAFTS[official_id]
        runnable = row["download_status"] == "downloaded" and bool(row.get("source_image_model_ready_path"))
        missing_count += not runnable
        image_row = {**row, **{f"control_{key}": draft[key] for key in
                               ("initial", "target", "removed", "preserved", "prompt", "note")},
                     "input_matches_initial_state": draft["match"],
                     "expected_change_mechanism": draft["mechanism"],
                     "expected_whole_frame_change": draft["whole"],
                     "possible_camera_motion": draft["camera"],
                     "control_needs_review": draft["review"],
                     "source_runnable": runnable,
                     "approval_status": "pending_user_approval" if runnable else "not_runnable_missing_source"}
        image_rows.append(image_row)
        for condition, used_prompt, is_official in (
            ("transition", row["prompt_official"], True), ("control", draft["prompt"], False)
        ):
            pair_rows.append({
                "pair_id": f"{row['case_id']}_{condition}", "case_id": row["case_id"],
                "official_id": official_id, "condition": condition, "is_official_prompt": is_official,
                "prompt_official": row["prompt_official"], "prompt_control": draft["prompt"],
                "prompt_used": used_prompt, "reference_video_url": row["reference_video_url"],
                "reference_video_id": row["reference_video_id"],
                "reference_start_sec": row["reference_start_sec"], "reference_end_sec": row["reference_end_sec"],
                "source_frame_time_original_seconds": row.get("source_frame_time_original_seconds", ""),
                "source_image_original_path": row.get("source_image_original_path", ""),
                "source_image_model_ready_path": row.get("source_image_model_ready_path", ""),
                "reference_source_video_path": row.get("reference_source_video_path", ""),
                "keywords": row["keywords"], "expected_change_mechanism": draft["mechanism"],
                "expected_whole_frame_change": draft["whole"], "possible_camera_motion": draft["camera"],
                "input_matches_initial_state": draft["match"], "download_status": row["download_status"],
                "official_category_count": len(source_rows), "source_runnable": runnable,
                "approval_status": "pending_user_approval" if runnable else "not_runnable_missing_source",
                "control_initial_state": draft["initial"], "control_target_state": draft["target"],
                "control_removed_transition": draft["removed"], "control_preserved_dynamics": draft["preserved"],
                "control_needs_review": draft["review"], "control_authoring_note": draft["note"],
                "sample_status": "pending_approval" if runnable else "not_runnable_missing_source",
                "source_sha256": row.get("reference_source_video_sha256", ""),
                "source_frame_sha256": row.get("source_frame_sha256", ""),
                "model_ready_sha256": row.get("model_ready_sha256", ""),
            })

    metadata_fields = list(dict.fromkeys(key for row in image_rows for key in row))
    write_csv(METADATA / "all_18_initial.csv", image_rows, metadata_fields)
    write_csv(APPROVAL / "tcbench_all_18_images.csv", image_rows, metadata_fields)
    pair_fields = list(dict.fromkeys(key for row in pair_rows for key in row))
    write_csv(APPROVAL / "tcbench_all_36pairs.csv", pair_rows, pair_fields)
    make_contact_sheet(source_rows)

    review_lines = [
        "# TC-Bench background-shifts: all-entry Stage A review", "",
        "Study: TC-Bench-derived first-frame I2V evaluation of all official background-shifts entries. This is not the original endpoint-conditioned TC-Bench protocol.", "",
        f"Official entries: {len(source_rows)}. Source videos retrievable: {len(source_rows)-missing_count}. Planned prompt pairs: {len(pair_rows)}; runnable pairs pending approval: {2*(len(source_rows)-missing_count)}; missing-source pairs: {2*missing_count}.", "",
        "Review `contact_sheet_all_18.png`. Transition prompts below are copied verbatim from pinned official metadata; controls are custom study prompts.", "",
    ]
    for row in image_rows:
        official_id = int(row["official_id"])
        draft = DRAFTS[official_id]
        review_lines += [
            f"## {row['case_id']} (official ID {official_id})", "",
            f"- Source: `{row.get('download_status')}`; video `{row.get('reference_video_id')}`; interval [{row['reference_start_sec']}, {row['reference_end_sec']}] s.",
            f"- Source URL: {row['reference_video_url']}",
            f"- Initial-state compatibility: **{draft['match']}**. Expected change: `{draft['mechanism']}`; whole-frame change: `{draft['whole']}`; possible camera motion: `{draft['camera']}`.",
            f"- Initial state: {draft['initial']}", f"- Target state: {draft['target']}",
            f"- Removed transition in control: {draft['removed']}", f"- Preserved dynamics: {draft['preserved']}",
            f"- Control needs review: **{str(draft['review']).lower()}**. {draft['note']}",
            "- Official transition prompt (verbatim):", "", f"> {row['prompt_official']}", "",
            "- Custom no-transition control prompt:", "", f"> {draft['prompt']}", "",
            f"- Model-ready frame: `{row.get('source_image_model_ready_path') or 'missing'}`; original first frame: `{row.get('source_image_original_path') or 'missing'}`; source video: `{row.get('reference_source_video_path') or 'missing'}`.", "",
        ]
    review_lines += [
        "## Items requiring attention before approval", "",
        "- All 18 official reference videos are now retrievable, including TC102, TC112, and TC114 after using YouTube's Android player client. No substitute assets were used.",
        "- TC103's extracted frame is already sunset-like although its official prompt says sunny afternoon to sunset; compatibility is marked `no`.",
        "- TC105's frame already contains a bright sky streak; TC112 already shows a rainbow arc; TC114 already shows fireworks; and TC119 already shows fireworks. Review these initial-state/control pairs because the official prompts do not provide a clear pre-transition frame.",
        "- No encoding, generation, decoding, or GPU analysis has been run for TC-Bench. Stage B remains gated on explicit approval of this complete package.", "",
    ]
    (APPROVAL / "selection_review_all_18.md").write_text("\n".join(review_lines), encoding="utf-8")
    (EXP / "reports").mkdir(parents=True, exist_ok=True)
    source_status = RETRIEVAL_STATUS["unique_source_status"]
    unavailable = [
        (source_id, status) for source_id, status in sorted(source_status.items())
        if status.get("status") != "downloaded"
    ]
    unavailable_lines = [
        f"- TC{case_id}: {status.get('url', '')} — {status.get('error_type', 'Unavailable')}: {status.get('error', 'source unavailable')}"
        for source_id, status in unavailable
        for case_id in [next((row['official_id'] for row in source_rows if row['reference_video_id'] == source_id), 'unknown')]
    ]
    (EXP / "reports" / "exceptions_and_missing_assets.md").write_text(
        "# TC-Bench Stage A source exceptions\n\n"
        "The census keeps all 18 official entries. All 16 unique reference videos were retrieved, including the three sources that required yt-dlp's Android player client. No substitutes were downloaded.\n\n"
        + ("No source download failures remain.\n\n" if not unavailable_lines else "Remaining source failures:\n" + "\n".join(unavailable_lines) + "\n\n")
        + "Initial-state review flags: TC103 is inconsistent with its official sunny-afternoon initial state; TC105, TC112, TC114, and TC119 already show a target-like event or state in the extracted frame. These cases remain visible for review and are not silently excluded from the official census.\n",
        encoding="utf-8")

    manifest_path = METADATA / "dataset_census_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "reference_status": "retrieved_all_official_sources" if missing_count == 0 else "retrieved_with_unavailable_sources",
        "available_source_entries": len(source_rows) - missing_count,
        "unavailable_source_entries": missing_count,
        "available_official_ids": [int(r["official_id"]) for r in image_rows if r["source_runnable"]],
        "unavailable_official_ids": [int(r["official_id"]) for r in image_rows if not r["source_runnable"]],
        "planned_prompt_pairs": len(pair_rows), "runnable_prompt_pairs_pending_approval": 2 * (len(source_rows) - missing_count),
        "source_frame_extraction": "PyAV full-video seek backward to preceding keyframe, then first decoded frame PTS >= official start and < official end; timestamps preserved in source time; no FFmpeg trim or double offset",
        "control_draft_review_ids": [official_id for official_id, d in DRAFTS.items() if d["review"]],
        "initial_state_mismatch_ids": [official_id for official_id, d in DRAFTS.items() if d["match"] == "no"],
        "approval_package": [
            "tcbench_background_shifts/approval/contact_sheet_all_18.png",
            "tcbench_background_shifts/approval/selection_review_all_18.md",
            "tcbench_background_shifts/approval/tcbench_all_18_images.csv",
            "tcbench_background_shifts/approval/tcbench_all_36pairs.csv",
        ],
        "stage_b_approval_status": "pending explicit user approval",
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Built all-18 review package: {len(source_rows)-missing_count}/18 sources available; {len(pair_rows)} planned pairs.")


if __name__ == "__main__":
    main()
