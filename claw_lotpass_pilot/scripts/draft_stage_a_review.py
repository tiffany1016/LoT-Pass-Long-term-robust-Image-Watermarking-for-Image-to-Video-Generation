#!/usr/bin/env python3
"""Add human-vision scene estimates and paired prompts for the seed-42 sample."""
from __future__ import annotations

import csv
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / "claw_lotpass_pilot/selection"

# Estimates are intentionally pre-generation judgments from the input images.
SCENES = {
    "water-runs-down-a-rocky-wall---closeup-sb-xqawmdkdyavp9n": (
        "waterfall", "A broad sheet of water runs down a rocky wall beside small plants.", 65,
        "yes", "High", "The cascade occupies most of the frame; the rocks and plants appear stationary.",
        "a close view of water running down a rocky wall beside small plants",
        "a continuous sheet cascade, branching rivulets, splashes, and foam",
    ),
    "iao-valley-waterfall-2-e30g6ov": (
        "waterfall", "Water cascades through a narrow rocky channel between large boulders.", 42,
        "uncertain", "High", "Strong central waterfall motion, with large stationary boulders around it.",
        "a waterfall descending through a narrow rocky channel between large boulders",
        "the falling cascade, churning water, splashes, and whitewater",
    ),
    "aerial-view-of-turquoise-water-going-down-the-river-at-soca-river-slovenia-sproyj5ddkgfb7261": (
        "river", "A wide turquoise river surface with a narrow wooded bank at the top edge.", 88,
        "yes", "High", "Water covers nearly the full image; the distant shoreline is a small static region.",
        "a wide turquoise river viewed from above with a narrow wooded bank",
        "broad, continuous currents, ripples, and shifting patterns across the turquoise water",
    ),
    "waterfall-in-green-nature-11-h3siv10ydkhibie27": (
        "waterfall/stream", "A small cascade and stream amid green vegetation, rocks, and shallow water.", 55,
        "yes", "High", "Visible water crosses much of the frame; surrounding plants and rocks remain fixed.",
        "a small waterfall and stream amid green vegetation and rocks",
        "the cascade, stream currents, ripples, and splashes throughout the visible water",
    ),
    "rapid-water-flowing-through-rocks-with-audio-dwex0i5": (
        "river rapids", "Clear water moves around several large rocks in a shallow stream.", 55,
        "yes", "High", "Water and rocks share the frame; motion can be localized to the connected stream.",
        "a shallow clear stream winding around several large rocks",
        "strong currents, eddies, ripples, and small foamy splashes around the rocks",
    ),
    "picturesque-mountain-river-close-up-hbkgyflgnjpnd2va7": (
        "river/waterfall", "A mountain stream falls over a low ledge into a rocky pool.", 66,
        "yes", "High", "Flowing water spans the center and foreground, bordered by stationary rocks and plants.",
        "a mountain stream falling over a low rocky ledge into a pool",
        "the falling water, flowing pool, ripples, and foam across the connected stream",
    ),
    "nature-background-river-h__jskoiisqpej7y": (
        "river rapids", "Whitewater flows through a rocky channel with a few visible stones.", 72,
        "yes", "High", "Water occupies most of the view; the foreground rocks provide stationary structure.",
        "whitewater rushing through a rocky channel",
        "broad rapids, turbulent currents, splashes, and foam across the visible water",
    ),
    "in-the-frame-there-is-a-picturesque-shot-of-cascading-water-of-mountain-stream-rapidly-flowing-down-rp84qrksips8jas4": (
        "waterfall/stream", "A mountain stream cascades down a rocky slope beside moss and plants.", 62,
        "yes", "High", "The cascade is large and continuous; moss, rocks, and vegetation remain static.",
        "a mountain stream cascading down a rocky slope beside moss and plants",
        "the waterfall, branching streams, splashes, and foam over the visible cascade",
    ),
    "iao-valley-slowmotion-water-on-rocks-mw9cngy": (
        "stream", "A shallow stream threads between prominent boulders, with water in the lower frame.", 38,
        "uncertain", "High", "Large rocks dominate; the fluid motion is clear but occupies a smaller region.",
        "a shallow stream threading between large boulders",
        "flowing currents, ripples, and small eddies in the visible channels between the rocks",
    ),
    "water-stream-flow-among-stones-clear-water-quickly-fall-downhill-flowing-around-dark-wet-stones-close-up-of-waterfall-pouring-down-black-rocks-beautiful-mountain-waterfall-background-hspydvq9fjfccehsh": (
        "stream/waterfall", "A close view of clear water tumbling between dark wet stones.", 52,
        "yes", "High", "The water is distributed through several channels, with stationary stones throughout.",
        "clear water tumbling between dark wet stones in a close view",
        "continuous downhill flow, small cascades, ripples, splashes, and foam between the stones",
    ),
    "waterfall-view-in-nature-22-rtxa3amtbk1rct98d": (
        "waterfall/stream", "A narrow waterfall pours into a shallow rocky basin surrounded by foliage.", 64,
        "yes", "High", "The falling water and basin occupy much of the image; foliage and rocks remain static.",
        "a narrow waterfall pouring into a shallow rocky basin surrounded by foliage",
        "the falling cascade, churning basin, ripples, and splashes throughout the water",
    ),
    "transparent-spring-water-flows-down-the-stones-slow-mo-b7xf731z0wj9a65gc9": (
        "stream", "A thin transparent stream flows over textured stones in a shallow channel.", 43,
        "uncertain", "High", "The stream is visible but much of the frame is stationary stone surface.",
        "a thin transparent spring flowing over textured stones in a shallow channel",
        "the flowing water, ripples, and small eddies across the visible stream",
    ),
    "flowing-water-video-background-rgxrl0jsisr0rtix": (
        "river", "A shallow, broad river surface flows over submerged stones across the frame.", 82,
        "yes", "High", "Water fills most of the image; submerged rocks are visible beneath the moving surface.",
        "a shallow river flowing over submerged stones across the frame",
        "broad currents, ripples, and shifting water patterns over the submerged stones",
    ),
    "water-flowing-between-stones-creating-white-bubbling-foam-stones-in-sheer-stream-of-blue-water-flow-in-mountains-water-stream-on-stone-hr29fe55zjfcblgrf": (
        "stream/rapids", "Blue water flows between stones, forming conspicuous white bubbles and foam.", 63,
        "yes", "High", "The stream spans most of the image with stationary stones dividing its flow.",
        "blue mountain water flowing between stones in a rocky stream",
        "currents, bubbling foam, eddies, and splashes throughout the channels between the stones",
    ),
    "streaming-river-from-mountains-landscape-with-stream-flowing-between-rocks-power-of-nature-sdrrjbfdbk0zfuyxx": (
        "river rapids", "A wide foaming mountain stream rushes across a rocky riverbed.", 72,
        "yes", "High", "Water occupies most of the frame; exposed rocks form fixed islands and banks.",
        "a wide foaming mountain stream rushing across a rocky riverbed",
        "broad rapids, churning currents, foam, and splashes throughout the visible stream",
    ),
}


def prompt_pair(scene: str, movement: str) -> tuple[str, str]:
    static = (
        f"Recreate {scene} exactly as shown, preserving the composition, water, rocks, and plants. "
        "Keep the camera completely stationary and freeze every visible detail: the water, foam, reflections, "
        "and surroundings remain in place. Do not add objects or introduce cuts, transitions, or lighting changes."
    )
    dynamic = (
        f"Preserve {scene} and keep the camera stationary. Animate {movement} with continuous, realistic "
        "motion throughout the visible liquid. Maintain the composition; keep rocks, plants, banks, and other "
        "solid structures fixed. Add no objects, cuts, transitions, weather, or lighting changes."
    )
    return static, dynamic


def main() -> None:
    sample = list(csv.DictReader((SELECTION / "claw_sample_15.csv").open(encoding="utf-8")))
    if len(sample) != 15:
        raise RuntimeError(f"Expected 15 sampled images, found {len(sample)}")
    analysis_rows, prompt_rows = [], []
    for row in sample:
        image_id = row["image_id"]
        if image_id not in SCENES:
            raise KeyError(f"No visual review entry for sampled image {image_id}")
        category, elements, dynamic_pct, candidate, camera, notes, scene, movement = SCENES[image_id]
        analysis_rows.append({
            "image_id": image_id, "scene_category": category, "dynamic_elements": elements,
            "estimated_dynamic_coverage": dynamic_pct, "estimated_static_coverage": 100-dynamic_pct,
            "dense_dynamic_candidate": candidate, "camera_stationary_suitability": camera, "notes": notes,
        })
        static, dynamic = prompt_pair(scene, movement)
        for condition, prompt in (("static_control", static), ("broad_fluid_motion", dynamic)):
            count = len(prompt.split())
            if not 25 <= count <= 60:
                raise ValueError(f"Prompt length {count} outside 25-60 words for {image_id}/{condition}")
            prompt_rows.append({"image_id": image_id, "condition": condition,
                                "generated_prompt": prompt, "prompt_word_count": count})

    analysis_fields = [
        "image_id", "scene_category", "dynamic_elements", "estimated_dynamic_coverage",
        "estimated_static_coverage", "dense_dynamic_candidate", "camera_stationary_suitability", "notes",
    ]
    prompt_fields = ["image_id", "condition", "generated_prompt", "prompt_word_count"]
    with (SELECTION / "claw_sample_analysis.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=analysis_fields)
        writer.writeheader()
        writer.writerows(analysis_rows)
    with (SELECTION / "claw_prompt_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=prompt_fields)
        writer.writeheader()
        writer.writerows(prompt_rows)

    report_path = SELECTION / "selection_report.md"
    report = report_path.read_text(encoding="utf-8").split("\n## Image analysis and selection method\n")[0]
    report = report.split("\n## Selected images\n")[0]
    report += "\n## Image analysis and selection method\n\n"
    report += (
        "The sample is a uniform draw without replacement; no scene-category or expected-outcome filters were used. "
        "Coverage estimates are visual pre-generation estimates. The sample is river/stream/waterfall-heavy because "
        "the CLAW v1 test set itself is curated for liquid scenes. Motion prompts animate fluid broadly while leaving "
        "solid structures and the camera fixed.\n\n"
    )
    coverage_values = [int(row["estimated_dynamic_coverage"]) for row in analysis_rows]
    category_counts = Counter(row["scene_category"] for row in analysis_rows)
    report += f"Scene categories: {', '.join(f'{name} ({count})' for name, count in sorted(category_counts.items()))}.\n\n"
    report += (
        f"Estimated dynamic coverage: mean {statistics.mean(coverage_values):.1f}%, "
        f"median {statistics.median(coverage_values):.0f}%, range {min(coverage_values)}–{max(coverage_values)}%. "
        f"Coverage bins: <40% = {sum(v < 40 for v in coverage_values)}, "
        f"40–59% = {sum(40 <= v < 60 for v in coverage_values)}, "
        f"≥60% = {sum(v >= 60 for v in coverage_values)}.\n\n"
    )
    report += "## Selected images\n\n| Sample | Image ID | Scene | Dynamic coverage | Candidate |\n|---:|---|---|---:|---|\n"
    analyses = {row["image_id"]: row for row in analysis_rows}
    for row in sorted(sample, key=lambda item: int(item["sample_index"])):
        a = analyses[row["image_id"]]
        report += (
            f'| {row["sample_index"]} | `{row["image_id"]}` | {a["scene_category"]} | '
            f'{a["estimated_dynamic_coverage"]}% | {a["dense_dynamic_candidate"]} |\n'
        )
    report += "\n## Prompt conditions\n\n"
    report += "- `static_control`: camera fixed; freeze all visible scene and water motion.\n"
    report += "- `broad_fluid_motion`: camera fixed; encourage broad, continuous motion throughout visible liquid regions while preserving solid background structures.\n"
    report += "\nEach image has one prompt per condition in `claw_prompt_pairs.csv`; both prompt lengths are 25–60 words.\n"
    report += "\n## Approval gate\n\nReview the contact sheet, selected IDs, analysis estimates, and all 30 prompts. No encoding or video generation has been run.\n"
    report_path.write_text(report, encoding="utf-8")

    prompts_by_id = {}
    for row in prompt_rows:
        prompts_by_id.setdefault(row["image_id"], {})[row["condition"]] = row
    approval = [
        "# CLAW Stage A approval request", "",
        "Review the sample contact sheet and the selected source images, scene estimates, and prompts below. "
        "No source images have been watermarked and no videos have been generated.", "",
        "![15-image random sample contact sheet](contact_sheet_sample_15.png)", "",
        "The sample is a uniform random draw of 15 from 126 official CLAW v1 test inputs (seed 42). "
        "Motion prompts target broad fluid motion while preserving rocks, vegetation, and the fixed camera.", "",
    ]
    for row in sorted(sample, key=lambda item: int(item["sample_index"])):
        image_id = row["image_id"]
        a = analyses[image_id]
        pair = prompts_by_id[image_id]
        approval.extend([
            f'## Sample {int(row["sample_index"]):02d}: `{image_id}`', "",
            f'![{image_id}](../images/source/{row["filename"]})', "",
            f'- Scene: {a["scene_category"]}; estimated dynamic coverage: {a["estimated_dynamic_coverage"]}%; '
            f'dense-motion candidate: {a["dense_dynamic_candidate"]}.',
            f'- Notes: {a["notes"]}',
            f'- Static control ({pair["static_control"]["prompt_word_count"]} words): '
            f'{pair["static_control"]["generated_prompt"]}',
            f'- Broad-fluid-motion prompt ({pair["broad_fluid_motion"]["prompt_word_count"]} words): '
            f'{pair["broad_fluid_motion"]["generated_prompt"]}', "",
        ])
    approval.extend([
        "## Approval gate", "",
        "Please approve or identify image/prompt edits. Encoding, CogVideoX generation, decoding, and GPU analysis "
        "remain paused until this sample and prompt pair set is approved.", "",
    ])
    (SELECTION / "approval_request.md").write_text("\n".join(approval), encoding="utf-8")
    print(f"Wrote visual estimates for {len(analysis_rows)} images and {len(prompt_rows)} prompts.")


if __name__ == "__main__":
    main()
