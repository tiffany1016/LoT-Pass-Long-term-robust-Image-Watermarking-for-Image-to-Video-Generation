#!/usr/bin/env python3
"""Create paired, category-level summaries using source images as analysis units."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "vbench_lotpass_pilot"
SELECTION = EXP / "selection"
ANALYSIS = EXP / "analysis"
SEED = 42


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    pair_rows = read_csv(SELECTION / "vbench_40pairs.csv")
    raw = read_csv(ANALYSIS / "raw_ba/per_frame_accuracy.csv")
    decode_summary = read_csv(ANALYSIS / "raw_ba/case_summary.csv")
    stability = read_csv(ANALYSIS / "stability/per_frame_stability.csv")
    motion = read_csv(ANALYSIS / "motion/motion_validation.csv")
    heatmaps = read_csv(ANALYSIS / "heatmaps/heatmap_metrics.csv")
    aligned_rows = read_csv(ANALYSIS / "optical_flow/per_video_aligned_ba.csv")
    by_case_frames = defaultdict(dict)
    for row in raw:
        by_case_frames[row["case_id"]][int(row["frame_idx"])] = fnum(row["ba"])
    by_case_stability = defaultdict(list)
    for row in stability:
        if row.get("frame_idx") not in (None, ""):
            by_case_stability[row["case_id"]].append(row)
    summaries = {row["case_id"]: row for row in decode_summary}
    motion_status = {row["case_id"]: row.get("motion_validation_status", "not_reviewed") for row in motion}
    case_meta = {row["case_id"]: row for row in pair_rows}
    image_types = {row["image_id"]: row["image_type"] for row in pair_rows}

    def case_metrics(case_id: str) -> dict:
        frames = by_case_frames.get(case_id, {})
        generated_idx = sorted(i for i in frames if i >= 1)
        if not generated_idx:
            return {"generated_mean": float("nan"), "last_third": float("nan"), "min_ba": float("nan"),
                    "srr": float("nan"), "lsrr": float("nan"), "completed": False}
        n = max(frames) + 1
        last_start = int(np.ceil(2 * n / 3))
        vals = np.asarray([frames[i] for i in generated_idx], dtype=float)
        last_vals = np.asarray([frames[i] for i in generated_idx if i >= last_start], dtype=float)
        stab = by_case_stability.get(case_id, [])
        srr_vals = [fnum(r.get("srr")) for r in stab if int(r.get("frame_idx", 0)) >= 1]
        lsrr_vals = [fnum(r.get("lsrr")) for r in stab if int(r.get("frame_idx", 0)) >= 1]
        return {"generated_mean": float(np.nanmean(vals)), "last_third": float(np.nanmean(last_vals)) if len(last_vals) else float("nan"),
                "min_ba": float(np.nanmin(vals)), "srr": float(np.nanmean(srr_vals)) if srr_vals else float("nan"),
                "lsrr": float(np.nanmean(lsrr_vals)) if lsrr_vals else float("nan"), "completed": True}

    per_case = {case_id: case_metrics(case_id) for case_id in case_meta}
    # Paired comparisons use the same generated frame indices (1-48 for 49-frame clips).
    paired_rows = []
    paired_deltas = defaultdict(list)
    for image_id in sorted({row["image_id"] for row in pair_rows}):
        static_row = next((r for r in pair_rows if r["image_id"] == image_id and r["motion_group"] == "static"), None)
        if static_row is None:
            continue
        for group in ("pan", "tilt", "zoom"):
            motion_row = next((r for r in pair_rows if r["image_id"] == image_id and r["motion_group"] == group), None)
            if motion_row is None:
                continue
            st = by_case_frames.get(static_row["case_id"], {})
            mt = by_case_frames.get(motion_row["case_id"], {})
            shared = sorted(i for i in set(st) & set(mt) if i >= 1)
            static_mean = float(np.mean([st[i] for i in shared])) if shared else float("nan")
            moving_mean = float(np.mean([mt[i] for i in shared])) if shared else float("nan")
            delta = static_mean - moving_mean if shared else float("nan")
            if np.isfinite(delta):
                paired_deltas[group].append(delta)
            paired_rows.append({
                "image_id": image_id, "image_type": image_types.get(image_id, ""), "motion_group": group,
                "motion_case_id": motion_row["case_id"], "motion_full": motion_row["motion_full"],
                "static_case_id": static_row["case_id"], "shared_frame_indices": ",".join(map(str, shared)),
                "n_shared_generated_frames": len(shared), "static_mean_ba": static_mean,
                "motion_mean_ba": moving_mean, "delta_ba_static_minus_motion": delta,
                "static_low_accuracy_flag": bool(per_case[static_row["case_id"]]["last_third"] < .80) if np.isfinite(per_case[static_row["case_id"]]["last_third"]) else "",
                "motion_last_third_ba": per_case[motion_row["case_id"]]["last_third"],
                "motion_last_third_below_0_80": bool(per_case[motion_row["case_id"]]["last_third"] < .80) if np.isfinite(per_case[motion_row["case_id"]]["last_third"]) else "",
                "static_minus_motion_last_third_over_0_15": bool(per_case[static_row["case_id"]]["last_third"]-per_case[motion_row["case_id"]]["last_third"] > .15)
                    if np.isfinite(per_case[static_row["case_id"]]["last_third"]) and np.isfinite(per_case[motion_row["case_id"]]["last_third"]) else "",
                "motion_validation_status": motion_status.get(motion_row["case_id"], "not_reviewed"),
            })
    paired_dir = ANALYSIS / "paired"
    write_csv(paired_dir / "motion_vs_static.csv", paired_rows,
              ["image_id", "image_type", "motion_group", "motion_case_id", "motion_full", "static_case_id",
               "shared_frame_indices", "n_shared_generated_frames", "static_mean_ba", "motion_mean_ba",
               "delta_ba_static_minus_motion", "static_low_accuracy_flag", "motion_last_third_ba",
               "motion_last_third_below_0_80", "static_minus_motion_last_third_over_0_15", "motion_validation_status"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for group in ("pan", "tilt", "zoom"):
        values = [row["delta_ba_static_minus_motion"] for row in paired_rows if row["motion_group"] == group and np.isfinite(row["delta_ba_static_minus_motion"])]
        ax.scatter([group] * len(values), values, alpha=.75, label=group)
        if values:
            ax.plot([group], [np.mean(values)], marker="_", markersize=26, color="black")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set(ylabel="Static BA − motion BA", title="Within-image paired differences (frames 1–48)")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(paired_dir / "paired_ba_comparison.png", dpi=160)
    plt.close(fig)

    category_rows = []
    conditions = ("static", "pan", "tilt", "zoom")
    for condition in conditions:
        cond_rows = [r for r in pair_rows if r["motion_group"] == condition]
        completed = [r for r in cond_rows if per_case[r["case_id"]]["completed"]]
        means = [per_case[r["case_id"]]["generated_mean"] for r in completed]
        lasts = [per_case[r["case_id"]]["last_third"] for r in completed]
        srrs = [per_case[r["case_id"]]["srr"] for r in completed]
        lsrrs = [per_case[r["case_id"]]["lsrr"] for r in completed]
        paired = [r for r in paired_rows if r["motion_group"] == condition]
        realized = [r for r in cond_rows if motion_status.get(r["case_id"]) == "valid"]
        category_rows.append({
            "condition": condition, "scheduled": len(cond_rows), "completed": len(completed),
            "valid_motion_videos": len(realized), "generation_success_fraction": len(completed)/len(cond_rows) if cond_rows else float("nan"),
            "mean_ba_frames_1_48": float(np.mean(means)) if means else float("nan"),
            "last_third_mean_ba": float(np.mean(lasts)) if lasts else float("nan"),
            "mean_within_image_delta_ba_to_static": float(np.mean([r["delta_ba_static_minus_motion"] for r in paired if np.isfinite(r["delta_ba_static_minus_motion"])])) if condition != "static" and any(np.isfinite(r["delta_ba_static_minus_motion"]) for r in paired) else 0.0 if condition == "static" else float("nan"),
            "mean_srr": float(np.mean(srrs)) if srrs else float("nan"), "median_srr": float(np.median(srrs)) if srrs else float("nan"),
            "mean_lsrr": float(np.mean(lsrrs)) if lsrrs else float("nan"),
            "last_third_ba_below_0_80_rate": float(np.mean([v < .80 for v in lasts])) if lasts else float("nan"),
            "static_minus_motion_over_0_15_rate": float(np.mean([r["static_minus_motion_last_third_over_0_15"] is True for r in paired])) if paired and condition != "static" else float("nan"),
            "realized_motion_subset_n": len(realized), "realized_motion_failure_rate": float("nan"),
            "motion_label_status": "not reviewed; realized-motion subset unavailable",
        })
    category_dir = ANALYSIS / "category"
    write_csv(category_dir / "category_summary.csv", category_rows, list(category_rows[0].keys()))

    fig, ax = plt.subplots(figsize=(8, 5))
    for condition in conditions:
        values = [per_case[r["case_id"]]["generated_mean"] for r in pair_rows if r["motion_group"] == condition and per_case[r["case_id"]]["completed"]]
        if values:
            jitter = np.linspace(-.12, .12, len(values))
            x = conditions.index(condition)
            ax.scatter(np.full(len(values), x) + jitter, values, alpha=.75, label=condition)
    ax.set_xticks(range(len(conditions)), conditions)
    ax.set(ylabel="Per-video mean BA (frames 1–48)", ylim=(0, 1.02), title="Per-image raw BA by requested condition")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(category_dir / "category_ba_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(conditions))
    low = [row["last_third_ba_below_0_80_rate"] for row in category_rows]
    diff = [row["static_minus_motion_over_0_15_rate"] for row in category_rows]
    ax.bar(x - .18, np.nan_to_num(low, nan=0), width=.36, label="Last-third BA < 0.80")
    ax.bar(x + .18, np.nan_to_num(diff, nan=0), width=.36, label="Static − motion > 0.15")
    ax.set_xticks(x, conditions)
    ax.set(ylabel="Fraction of cases", ylim=(0, 1), title="Pilot screening flags (not established failure definitions)")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(category_dir / "category_failure_rate.png", dpi=160)
    plt.close(fig)

    stable_case_means = {case: per_case[case] for case in per_case if per_case[case]["completed"] and np.isfinite(per_case[case]["srr"])}
    valid_pairs = [(v["generated_mean"], v["srr"], case) for case, v in stable_case_means.items()]
    high_ba_low_srr, low_ba_high_srr = [], []
    if valid_pairs:
        ba_median = float(np.median([p[0] for p in valid_pairs]))
        srr_median = float(np.median([p[1] for p in valid_pairs]))
        high_ba_low_srr = [case for ba, srr, case in valid_pairs if ba >= ba_median and srr <= srr_median]
        low_ba_high_srr = [case for ba, srr, case in valid_pairs if ba <= ba_median and srr >= srr_median]
    correlations = {}
    for metric in ("srr", "lsrr"):
        pairs = [(v["generated_mean"], v[metric]) for v in stable_case_means.values() if np.isfinite(v[metric])]
        if len(pairs) >= 3:
            rho, pval = spearmanr([v[0] for v in pairs], [v[1] for v in pairs])
            correlations[metric] = {"spearman_rho": float(rho), "p_value_descriptive_only": float(pval), "n_cases": len(pairs)}
        else:
            correlations[metric] = {"spearman_rho": None, "p_value_descriptive_only": None, "n_cases": len(pairs)}

    scheduled = len(pair_rows)
    done = sum(per_case[r["case_id"]]["completed"] for r in pair_rows)
    failed = [r["case_id"] for r in pair_rows if not per_case[r["case_id"]]["completed"]]
    heatmap_cases = sorted({r["case_id"] for r in heatmaps})
    heatmap_videos = sum((ANALYSIS / "heatmaps/decoder" / case / "decoder_heatmap_overlay.mp4").is_file()
                         for case in heatmap_cases)
    aligned_cases = [r for r in aligned_rows if r.get("status") == "completed"]
    aligned_raw_mean = np.mean([fnum(r["raw_mean_BA_excluding_copy_frame"]) for r in aligned_cases]) if aligned_cases else float("nan")
    aligned_mean = np.mean([fnum(r["aligned_mean_BA_excluding_copy_frame"]) for r in aligned_cases]) if aligned_cases else float("nan")
    aligned_delta = np.mean([fnum(r["aligned_minus_raw_BA_excluding_copy_frame"]) for r in aligned_cases]) if aligned_cases else float("nan")
    aligned_video_count = sum((ANALYSIS / "optical_flow" / r["case_id"] / "aligned_video.mp4").is_file() for r in aligned_cases)
    aligned_overlay_count = sum((ANALYSIS / "optical_flow" / r["case_id"] / "decoder_heatmap_overlay.mp4").is_file() for r in aligned_cases)
    report = [
        "# VBench++ LoT-Pass camera-motion pilot: results summary", "",
        f"Scheduled {scheduled} image-prompt pairs; raw-decode/stability-ready videos: {done}; remaining/failed: {len(failed)}.",
        "", "## Category summary", "", "| Condition | Completed / scheduled | Mean BA (frames 1–48) | Last-third BA | Mean SRR | Mean LSRR |", "|---|---:|---:|---:|---:|---:|",
    ]
    for r in category_rows:
        report.append(f"| {r['condition']} | {r['completed']}/{r['scheduled']} | {r['mean_ba_frames_1_48']:.4f} | {r['last_third_mean_ba']:.4f} | {r['mean_srr']:.4f} | {r['mean_lsrr']:.4f} |" if r["completed"] else f"| {r['condition']} | 0/{r['scheduled']} | NA | NA | NA | NA |")
    report += ["", "## Paired motion versus static", "",
               "Each within-image comparison uses shared generated frames 1–48 (frame 0 excluded consistently). Positive delta means static BA exceeded motion BA.", ""]
    for group in ("pan", "tilt", "zoom"):
        vals = paired_deltas.get(group, [])
        report.append(f"- {group}: n={len(vals)} paired sources; mean static-minus-motion delta={np.mean(vals):.4f}" if vals else f"- {group}: no complete pairs")
    report += ["", "## Stability associations and counterexamples", "",
               f"Descriptive Spearman associations (per-video averages; frames are not independent): `{json.dumps(correlations)}`.",
               f"Median-split counterexamples (case IDs): high BA / low SRR: {', '.join(high_ba_low_srr) if high_ba_low_srr else 'none observed'}; low BA / high SRR: {', '.join(low_ba_high_srr) if low_ba_high_srr else 'none observed'}.",
               "", "## Decoder attribution heatmaps", "",
               f"Existing decoder gradient attribution was recomputed for every frame: {len(heatmaps)} maps across {len(heatmap_cases)} videos, with all-frame `decoder_heatmap_overlay.mp4` outputs present for {heatmap_videos} videos. Overlay alpha is 0.35; per-frame overlap metrics and raw gradients are saved under `analysis/heatmaps/`. Frames within a video are repeated observations, not independent samples.",
               "", "## Supplemental RAFT alignment and decoding", "",
               f"All-frame LoT-Pass RAFT alignment completed for {len(aligned_cases)}/{scheduled} videos. Mean per-video BA excluding detected copy frames: raw {aligned_raw_mean:.4f}, aligned {aligned_mean:.4f}, mean within-video change {aligned_delta:+.4f}. This supplemental path uses generated frame 0 as the reference; raw decoding and same-coordinate stability remain unchanged. Aligned video and after-alignment decoder heatmap overlays exist for {aligned_video_count} and {aligned_overlay_count} videos, respectively, under `analysis/optical_flow/`.",
               "", "## Motion validity and limits", "",
               "Motion-validation rows are marked `not_reviewed` until a human inspects `analysis/motion/review_sheet.png`; requested motion is not treated as realized motion. No realized-motion subset rate is claimed.",
               "This is a stratified 10-image, one-seed exploratory pilot. Its frequencies do not generalize to all VBench++ images. Same-coordinate stability can drop under geometric motion and is not proof of watermark survival. Decoder gradients are sensitivity attributions, not causal or physical watermark locations.",
               "", "## Incomplete cases", ""]
    report += [f"- {case}" for case in failed] if failed else ["- None."]
    report += [""]
    (category_dir / "summary_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Built paired/category summaries for {done}/{scheduled} completed cases")


if __name__ == "__main__":
    main()
