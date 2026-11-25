#!/usr/bin/env python3
"""Create paired raw-BA, stability, attribution, and visual-transition summaries."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "chronomagic150_global_appearance"
DATA = EXP / "dataset"
VIDEOS = EXP / "generated_videos"
WATERMARKED = EXP / "watermarked_sources"
ANALYSIS = EXP / "analysis"
REVIEW = ANALYSIS / "transition_review"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if fields is None:
            fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def safe_float(value, default=float("nan")) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def format_num(value, digits=4):
    return f"{value:.{digits}f}" if value is not None and math.isfinite(float(value)) else "NA"


def main() -> None:
    import decord
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont
    from scipy.stats import pearsonr, spearmanr

    ANALYSIS.mkdir(parents=True, exist_ok=True); REVIEW.mkdir(parents=True, exist_ok=True)
    cases = read_csv(DATA / "chronomagic150_generation_manifest.csv")
    decoding = read_csv(EXP / "decoding/per_video_raw_ba.csv")
    frame_ba = read_csv(EXP / "decoding/per_frame_raw_ba.csv")
    stability = read_csv(EXP / "stable_region/per_video_metrics.csv")
    heatmaps = read_csv(EXP / "heatmaps/decoder/per_video_overlap.csv")
    decode_by_case = {r["case_id"]: r for r in decoding}
    stable_by_case = {r["case_id"]: r for r in stability}
    heat_by_case = {r["case_id"]: r for r in heatmaps}
    frame_map = defaultdict(dict)
    for row in frame_ba:
        frame_map[row["case_id"]][int(row["frame_idx"])] = safe_float(row["raw_bit_accuracy"])
    clip_rows = []
    appearance_frames, appearance_summary, transition_validation = [], [], []
    case_frames = {}
    for row in cases:
        case_id, sample_id, condition = row["case_id"], row["sample_id"], row["condition"]
        video = VIDEOS / sample_id / f"{condition}.mp4"
        decode = decode_by_case.get(case_id, {})
        stable = stable_by_case.get(case_id, {})
        heat = heat_by_case.get(case_id, {})
        record = {**row,
                  "mean_ba": safe_float(decode.get("mean_ba_excluding_copy_frame")),
                  "mean_ba_all_frames": safe_float(decode.get("mean_ba_all_frames")),
                  "minimum_ba": safe_float(decode.get("minimum_ba_all_frames")),
                  "last_third_mean_ba": safe_float(decode.get("last_third_mean_ba_excluding_copy_frame")),
                  "frame_count": safe_float(decode.get("frame_count")),
                  "mean_stable_region_ratio": safe_float(stable.get("mean_stable_region_ratio")),
                  "mean_largest_stable_region_ratio": safe_float(stable.get("mean_largest_stable_region_ratio")),
                  "mean_similarity": safe_float(stable.get("mean_similarity")),
                  "mean_encoder_stable_overlap": safe_float(heat.get("mean_encoder_stable_overlap")),
                  "mean_decoder_stable_overlap": safe_float(heat.get("mean_decoder_stable_overlap")),
                  "decode_status": decode.get("decode_status", "missing")}
        clip_rows.append(record)
        if decode.get("decode_status") != "completed" or not video.is_file():
            continue
        reader = decord.VideoReader(str(video), ctx=decord.cpu(0))
        first = reader[0].asnumpy().astype(np.float32) / 255.0
        luma0 = 0.2126 * first[..., 0] + 0.7152 * first[..., 1] + 0.0722 * first[..., 2]
        selected_indices = sorted(set(min(len(reader) - 1, i) for i in (0, 12, 24, 36, 48)))
        thumbs = []
        source = Image.open(WATERMARKED / sample_id / "watermarked.png").convert("RGB")
        source_thumb = source.copy(); source_thumb.thumbnail((260, 146), Image.Resampling.LANCZOS)
        thumbs.append(("Watermarked input", source_thumb))
        frame_metrics = []
        for frame_idx in range(len(reader)):
            arr = reader[frame_idx].asnumpy().astype(np.float32) / 255.0
            luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
            diff = np.abs(luma - luma0)
            item = {"case_id": case_id, "sample_id": sample_id, "analysis_group": row["analysis_group"],
                    "condition": condition, "frame_idx": frame_idx,
                    "mean_luminance": float(luma.mean()), "luminance_std": float(luma.std()),
                    "fraction_pixels_luma_change_gt_0_08": float(np.mean(diff > 0.08)),
                    "fraction_pixels_luma_change_gt_0_15": float(np.mean(diff > 0.15)),
                    "mean_abs_luma_change_from_frame0": float(diff.mean()),
                    "raw_bit_accuracy": frame_map[case_id].get(frame_idx, float("nan"))}
            frame_metrics.append(item); appearance_frames.append(item)
            if frame_idx in selected_indices:
                img = Image.fromarray(reader[frame_idx].asnumpy())
                img.thumbnail((260, 146), Image.Resampling.LANCZOS)
                thumbs.append((f"Generated f{frame_idx}", img))
        lumas = [r["mean_luminance"] for r in frame_metrics]
        appearance = {
            "case_id": case_id, "sample_id": sample_id, "analysis_group": row["analysis_group"],
            "condition": condition, "frame_count": len(frame_metrics),
            "frame0_mean_luminance": lumas[0], "final_mean_luminance": lumas[-1],
            "final_minus_initial_mean_luminance": lumas[-1] - lumas[0],
            "minimum_mean_luminance": min(lumas), "maximum_mean_luminance": max(lumas),
            "max_abs_luminance_delta_from_frame0": max(r["mean_abs_luma_change_from_frame0"] for r in frame_metrics),
            "final_fraction_pixels_luma_change_gt_0_08": frame_metrics[-1]["fraction_pixels_luma_change_gt_0_08"],
            "final_fraction_pixels_luma_change_gt_0_15": frame_metrics[-1]["fraction_pixels_luma_change_gt_0_15"],
        }
        appearance_summary.append(appearance)
        sheet = Image.new("RGB", (5 * 270, 2 * 185), "white")
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        except OSError:
            font = ImageFont.load_default()
        for j, (label, thumb) in enumerate(thumbs[:6]):
            x, y = (j % 5) * 270, (j // 5) * 185
            draw.text((x + 4, y + 3), label, fill="black", font=font)
            sheet.paste(thumb, (x + 4, y + 24))
        sheet.save(REVIEW / f"{sample_id}_{condition}.jpg", quality=88, optimize=True)
        transition_validation.append({
            "sample_id": sample_id, "case_id": case_id, "analysis_group": row["analysis_group"],
            "condition": condition, "target_transition": row["official_caption"] if condition == "transition" else "Maintain the source frame's initial appearance state.",
            "status": "pending_manual_visual_review", "camera_motion_contamination": "pending_manual_visual_review",
            "frame0_mean_luminance": lumas[0], "final_mean_luminance": lumas[-1],
            "final_minus_initial_mean_luminance": lumas[-1] - lumas[0],
            "final_fraction_pixels_luma_change_gt_0_08": appearance["final_fraction_pixels_luma_change_gt_0_08"],
            "review_sheet": str((REVIEW / f"{sample_id}_{condition}.jpg").relative_to(ROOT)),
            "review_notes": "",
        })

    write_csv(ANALYSIS / "per_video_metrics.csv", clip_rows)
    write_csv(ANALYSIS / "frame_appearance_metrics.csv", appearance_frames)
    write_csv(ANALYSIS / "appearance_summary.csv", appearance_summary)
    validation_path = ANALYSIS / "transition_validation.csv"
    if not validation_path.is_file():
        write_csv(validation_path, transition_validation)

    by_sample = defaultdict(dict)
    for row in clip_rows:
        by_sample[row["sample_id"]][row["condition"]] = row
    pair_rows = []
    for sample_id, pair in by_sample.items():
        if "transition" not in pair or "control" not in pair:
            continue
        t, c = pair["transition"], pair["control"]
        pair_rows.append({
            "sample_id": sample_id, "selection_index": t["selection_index"], "analysis_group": t["analysis_group"],
            "transition_mean_ba": t["mean_ba"], "control_mean_ba": c["mean_ba"],
            "delta_mean_ba_transition_minus_control": t["mean_ba"] - c["mean_ba"],
            "transition_last_third_ba": t["last_third_mean_ba"], "control_last_third_ba": c["last_third_mean_ba"],
            "delta_last_third_ba_transition_minus_control": t["last_third_mean_ba"] - c["last_third_mean_ba"],
            "transition_minimum_ba": t["minimum_ba"], "control_minimum_ba": c["minimum_ba"],
            "transition_srr": t["mean_stable_region_ratio"], "control_srr": c["mean_stable_region_ratio"],
            "delta_srr_transition_minus_control": t["mean_stable_region_ratio"] - c["mean_stable_region_ratio"],
            "transition_largest_stable_region": t["mean_largest_stable_region_ratio"], "control_largest_stable_region": c["mean_largest_stable_region_ratio"],
            "transition_decoder_overlap": t["mean_decoder_stable_overlap"], "control_decoder_overlap": c["mean_decoder_stable_overlap"],
            "delta_decoder_overlap_transition_minus_control": t["mean_decoder_stable_overlap"] - c["mean_decoder_stable_overlap"],
        })
    write_csv(ANALYSIS / "paired_comparison.csv", pair_rows)

    condition_summary = []
    for group in ("all", "illumination", "weather_visibility"):
        for condition in ("transition", "control"):
            values = [r for r in clip_rows if r["decode_status"] == "completed" and r["condition"] == condition and (group == "all" or r["analysis_group"] == group)]
            condition_summary.append({"analysis_group": group, "condition": condition, "n_independent_sources": len(values),
                "mean_video_ba": float(np.nanmean([r["mean_ba"] for r in values])) if values else float("nan"),
                "median_video_ba": float(np.nanmedian([r["mean_ba"] for r in values])) if values else float("nan"),
                "mean_last_third_ba": float(np.nanmean([r["last_third_mean_ba"] for r in values])) if values else float("nan"),
                "mean_minimum_ba": float(np.nanmean([r["minimum_ba"] for r in values])) if values else float("nan"),
                "mean_stable_region_ratio": float(np.nanmean([r["mean_stable_region_ratio"] for r in values])) if values else float("nan"),
                "mean_largest_stable_region_ratio": float(np.nanmean([r["mean_largest_stable_region_ratio"] for r in values])) if values else float("nan"),
                "mean_decoder_stable_overlap": float(np.nanmean([r["mean_decoder_stable_overlap"] for r in values])) if values else float("nan")})
    write_csv(ANALYSIS / "condition_summary.csv", condition_summary)

    def corr(x, y):
        valid = np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) < 3:
            return {"n": int(valid.sum()), "pearson_r": float("nan"), "pearson_p": float("nan"), "spearman_rho": float("nan"), "spearman_p": float("nan")}
        p = pearsonr(x[valid], y[valid]); s = spearmanr(x[valid], y[valid])
        return {"n": int(valid.sum()), "pearson_r": float(p.statistic), "pearson_p": float(p.pvalue),
                "spearman_rho": float(s.statistic), "spearman_p": float(s.pvalue)}
    pair_ba = np.asarray([r["delta_mean_ba_transition_minus_control"] for r in pair_rows], dtype=float)
    pair_srr = np.asarray([r["delta_srr_transition_minus_control"] for r in pair_rows], dtype=float)
    pair_largest = np.asarray([r["transition_largest_stable_region"] - r["control_largest_stable_region"] for r in pair_rows], dtype=float)
    correlations = []
    for name, x in (("paired_change_in_srr", pair_srr), ("paired_change_in_largest_stable_region", pair_largest)):
        correlations.append({"metric": name, **corr(x, pair_ba)})
    write_csv(ANALYSIS / "paired_change_correlations.csv", correlations)

    # Frame curves average clips first within condition; the independent n remains the number of sources.
    trajectory = []
    for group in ("all", "illumination", "weather_visibility"):
        for condition in ("transition", "control"):
            rows = [r for r in cases if r["condition"] == condition and (group == "all" or r["analysis_group"] == group)]
            curves = [frame_map.get(r["case_id"], {}) for r in rows]
            for frame_idx in range(49):
                vals = [curve[frame_idx] for curve in curves if frame_idx in curve and math.isfinite(curve[frame_idx])]
                trajectory.append({"analysis_group": group, "condition": condition, "frame_idx": frame_idx,
                                   "raw_bit_accuracy_mean": float(np.mean(vals)) if vals else float("nan"),
                                   "n_source_samples": len(vals)})
    write_csv(ANALYSIS / "paired_frame_trajectories.csv", trajectory)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for condition, color in (("transition", "#cf6c35"), ("control", "#287c9a")):
        rows = [r for r in trajectory if r["analysis_group"] == "all" and r["condition"] == condition]
        ax.plot([r["frame_idx"] for r in rows], [r["raw_bit_accuracy_mean"] for r in rows], label=condition.title(), color=color, linewidth=2)
    ax.set(xlabel="Frame index", ylabel="Raw bit accuracy", ylim=(0, 1), title="Mean raw LoT-Pass accuracy by frame")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(ANALYSIS / "raw_ba_trajectories.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ordered = sorted(pair_rows, key=lambda r: int(r["selection_index"]))
    colors = ["#cf6c35" if r["analysis_group"] == "illumination" else "#287c9a" for r in ordered]
    ax.bar([r["sample_id"] for r in ordered], [r["delta_mean_ba_transition_minus_control"] for r in ordered], color=colors)
    ax.axhline(0, color="#333", linewidth=1); ax.tick_params(axis="x", rotation=75)
    ax.set(ylabel="Transition − control mean raw BA", title="Paired per-source bit-accuracy change")
    ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(ANALYSIS / "paired_delta_mean_ba.png", dpi=160); plt.close(fig)
    completed_clips = [r for r in clip_rows if math.isfinite(r["mean_ba"]) and math.isfinite(r["mean_stable_region_ratio"])]
    fig, ax = plt.subplots(figsize=(6, 5))
    for condition, color, marker in (("transition", "#cf6c35", "o"), ("control", "#287c9a", "s")):
        vals = [r for r in completed_clips if r["condition"] == condition]
        ax.scatter([r["mean_stable_region_ratio"] for r in vals], [r["mean_ba"] for r in vals], c=color, marker=marker, label=condition, alpha=.75)
    ax.set(xlabel="Video mean stable-region ratio", ylabel="Video mean raw BA", title="Raw BA vs retained source-anchored regions")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(ANALYSIS / "ba_vs_stability.png", dpi=160); plt.close(fig)
    heat_clips = [r for r in completed_clips if math.isfinite(r["mean_decoder_stable_overlap"])]
    fig, ax = plt.subplots(figsize=(6, 5))
    for condition, color, marker in (("transition", "#cf6c35", "o"), ("control", "#287c9a", "s")):
        vals = [r for r in heat_clips if r["condition"] == condition]
        ax.scatter([r["mean_decoder_stable_overlap"] for r in vals], [r["mean_ba"] for r in vals], c=color, marker=marker, label=condition, alpha=.75)
    ax.set(xlabel="Video mean decoder-attribution overlap with stable mask", ylabel="Video mean raw BA", title="Raw BA vs decoder / stable-region overlap")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(ANALYSIS / "ba_vs_heatmap_overlap.png", dpi=160); plt.close(fig)

    complete_pairs = [r for r in pair_rows if math.isfinite(r["delta_mean_ba_transition_minus_control"])]
    mean_delta = float(np.mean([r["delta_mean_ba_transition_minus_control"] for r in complete_pairs])) if complete_pairs else float("nan")
    median_delta = float(np.median([r["delta_mean_ba_transition_minus_control"] for r in complete_pairs])) if complete_pairs else float("nan")
    negative_count = sum(r["delta_mean_ba_transition_minus_control"] < 0 for r in complete_pairs)
    statuses = {r["status"] for r in read_csv(validation_path)} if validation_path.is_file() else {"pending_manual_visual_review"}
    report = [
        "# ChronoMagic-150 LoT-Pass paired I2V analysis", "",
        f"Generated cases: {len(clip_rows)} (16 source images × 2 conditions); completed decode cases: {sum(r['decode_status']=='completed' for r in clip_rows)}.",
        "The independent unit is the source image/pair (n ≤ 16); frames are not treated as independent observations.",
        "All bit-accuracy figures are raw decoder results with no optical-flow alignment.", "",
        "## Paired raw bit accuracy", "",
        f"Completed pairs: {len(complete_pairs)}. Mean transition − control video BA: {format_num(mean_delta)}; median: {format_num(median_delta)}; transition lower in {negative_count}/{len(complete_pairs)} pairs.",
        "See `condition_summary.csv`, `paired_comparison.csv`, `paired_frame_trajectories.csv`, and the matching plots.", "",
        "## Source-anchored stability and attribution", "",
        "Stability compares each generated frame to the same-coordinate watermarked source using the inherited 16×16 MAE/SSIM patch implementation; temporal stability was not computed.",
        "Encoder maps use the repository's existing residual-magnitude method. Decoder maps use the existing signed-message input-gradient attribution method, evaluated for each frame.",
        "Pairwise association between change in raw BA and changes in SRR/largest stable region is in `paired_change_correlations.csv`. Per-frame overlap and enrichment are in `heatmaps/decoder/per_frame_overlap.csv`.", "",
        "## Transition compliance", "",
        f"Manual validation status values currently present: `{', '.join(sorted(statuses))}`. Review contact sheets under `transition_review/` and update `transition_validation.csv` before treating prompt compliance as confirmed.",
        "Luminance and changed-area measures are descriptive diagnostics; they do not alone establish semantic transition success.", "",
        "## Interpretation", "",
        "This is a paired 16-source pilot adapted from ChronoMagic-150 to single-first-frame CogVideoX I2V. Transition and control prompts are matched but remain stochastic generation branches. Results support descriptive within-source comparisons, not strong causal attribution.",
    ]
    (ANALYSIS / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    summary = {"n_source_pairs": len(pair_rows), "completed_pairs": len(complete_pairs),
               "mean_paired_delta_mean_ba": mean_delta, "median_paired_delta_mean_ba": median_delta,
               "transition_lower_pair_count": negative_count,
               "generation_status_counts": {status: sum(r.get("generation_status") == status for r in clip_rows) for status in sorted({r.get("generation_status", "missing") for r in clip_rows})},
               "decode_status_counts": {status: sum(r.get("decode_status") == status for r in clip_rows) for status in sorted({r.get("decode_status", "missing") for r in clip_rows})},
               "manual_transition_statuses": sorted(statuses), "optical_flow_alignment": False}
    (ANALYSIS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Summarized {len(pair_rows)} matched pairs; completed pairs={len(complete_pairs)}.", flush=True)


if __name__ == "__main__":
    main()
