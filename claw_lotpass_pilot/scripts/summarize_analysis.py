#!/usr/bin/env python3
"""Join CLAW run metrics, compute paired condition effects, and write final report."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/"claw_lotpass_pilot"
ANALYSIS=EXP/"analysis"
OUT=ANALYSIS/"summary"


def rows(path: Path) -> list[dict[str,str]]:
    with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))


def write_csv(path: Path,items: list[dict]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for item in items for k in item)) or ["status"]
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(items)


def number(value):
    try:
        x=float(value)
        return x if np.isfinite(x) else float("nan")
    except (TypeError,ValueError):return float("nan")


def mean(rows_,key):
    values=[number(r.get(key)) for r in rows_]
    values=[v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr,spearmanr

    manifest=json.loads((EXP/"selection/generation_manifest.json").read_text())
    analysis={r["image_id"]:r for r in rows(EXP/"selection/claw_sample_analysis.csv")}
    raw={r["case_id"]:r for r in rows(ANALYSIS/"raw_ba/case_summary.csv")}
    stable={r["case_id"]:r for r in rows(ANALYSIS/"stability/per_video_stability.csv")}
    aligned={r["case_id"]:r for r in rows(ANALYSIS/"optical_flow/per_video_aligned_ba.csv")}
    by_case=[]
    for case in manifest["cases"]:
        aid=analysis[case["image_id"]]
        r,s,a=raw.get(case["case_id"],{}),stable.get(case["case_id"],{}),aligned.get(case["case_id"],{})
        by_case.append({
            "case_id":case["case_id"],"image_id":case["image_id"],"sample_index":case["sample_index"],
            "condition":case["condition"],"scene_category":aid["scene_category"],
            "estimated_dynamic_coverage":aid["estimated_dynamic_coverage"],
            "raw_status":r.get("status","missing"),"mean_raw_ba":r.get("mean_raw_ba_excluding_copy_frame",""),
            "minimum_raw_ba":r.get("minimum_raw_ba",""),"last_third_raw_ba":r.get("last_third_raw_ba_excluding_copy_frame",""),
            "stable_status":s.get("status","missing"),"stable_region_ratio":s.get("mean_stable_region_ratio",""),
            "largest_stable_region_ratio":s.get("mean_largest_stable_region_ratio",""),"mean_similarity":s.get("mean_similarity",""),
            "aligned_status":a.get("status","missing"),"mean_aligned_ba":a.get("aligned_mean_BA_excluding_copy_frame",""),
            "aligned_minus_raw_ba":a.get("aligned_minus_raw_BA_excluding_copy_frame",""),
            "aligned_stable_region_ratio":a.get("mean_stable_region_ratio_aligned",""),
            "decoder_stable_overlap":a.get("mean_decoder_stable_overlap",""),
            "encoder_stable_overlap":a.get("mean_encoder_stable_overlap",""),
        })
    OUT.mkdir(parents=True,exist_ok=True)
    write_csv(OUT/"per_case_metrics.csv",by_case)

    complete=[r for r in by_case if r["raw_status"]=="completed"]
    conditions=("static_control","broad_fluid_motion")
    condition_summary=[]
    for condition in conditions:
        subset=[r for r in complete if r["condition"]==condition]
        record={"condition":condition,"scheduled_cases":15,"completed_raw_cases":len(subset)}
        for metric in ("mean_raw_ba","minimum_raw_ba","last_third_raw_ba","stable_region_ratio",
                       "largest_stable_region_ratio","mean_similarity","mean_aligned_ba","aligned_minus_raw_ba",
                       "aligned_stable_region_ratio","decoder_stable_overlap","encoder_stable_overlap"):
            record["mean_"+metric]=mean(subset,metric)
        condition_summary.append(record)
    write_csv(OUT/"condition_summary.csv",condition_summary)

    paired=[]
    for image in manifest["selected_images"]:
        entries={r["condition"]:r for r in by_case if r["image_id"]==image["image_id"]}
        base={"image_id":image["image_id"],"sample_index":image["sample_index"],
              "estimated_dynamic_coverage":analysis[image["image_id"]]["estimated_dynamic_coverage"]}
        for metric in ("mean_raw_ba","mean_aligned_ba","stable_region_ratio","largest_stable_region_ratio",
                       "decoder_stable_overlap","aligned_minus_raw_ba"):
            left=number(entries.get("static_control",{}).get(metric))
            right=number(entries.get("broad_fluid_motion",{}).get(metric))
            base["static_"+metric]=left if np.isfinite(left) else ""
            base["motion_"+metric]=right if np.isfinite(right) else ""
            base["motion_minus_static_"+metric]=right-left if np.isfinite(left) and np.isfinite(right) else ""
        paired.append(base)
    write_csv(OUT/"paired_condition_deltas.csv",paired)

    # Average the paired conditions within each source image before correlating with its estimated motion coverage.
    image_means=[]
    for image in manifest["selected_images"]:
        subset=[r for r in by_case if r["image_id"]==image["image_id"]]
        row={"image_id":image["image_id"],"sample_index":image["sample_index"],
             "estimated_dynamic_coverage":int(analysis[image["image_id"]]["estimated_dynamic_coverage"])}
        for metric in ("mean_raw_ba","mean_aligned_ba","stable_region_ratio","largest_stable_region_ratio",
                       "decoder_stable_overlap","encoder_stable_overlap"):
            row[metric]=mean(subset,metric)
        image_means.append(row)
    write_csv(OUT/"per_image_condition_means.csv",image_means)

    correlation_rows=[]
    dynamic=np.asarray([r["estimated_dynamic_coverage"] for r in image_means],dtype=float)
    for metric in ("mean_raw_ba","mean_aligned_ba","stable_region_ratio","largest_stable_region_ratio","decoder_stable_overlap","encoder_stable_overlap"):
        values=np.asarray([r[metric] for r in image_means],dtype=float)
        valid=np.isfinite(dynamic)&np.isfinite(values)
        if int(valid.sum())>=3:
            pr=pearsonr(dynamic[valid],values[valid]);sr=spearmanr(dynamic[valid],values[valid])
            correlation_rows.append({"x":"estimated_dynamic_coverage","y":metric,"n_images":int(valid.sum()),
                                     "pearson_r":float(pr.statistic),"pearson_p":float(pr.pvalue),
                                     "spearman_rho":float(sr.statistic),"spearman_p":float(sr.pvalue),
                                     "unit":"source-image means across both prompt conditions"})
        else:
            correlation_rows.append({"x":"estimated_dynamic_coverage","y":metric,"n_images":int(valid.sum())})
    write_csv(OUT/"dynamic_coverage_correlations.csv",correlation_rows)

    colors={"static_control":"#315b7d","broad_fluid_motion":"#d77732"}
    for metric,ylabel,title,filename in (
        ("mean_raw_ba","Mean raw ACC","Estimated fluid coverage vs raw ACC","coverage_vs_raw_acc.png"),
        ("stable_region_ratio","Stable-region ratio","Estimated fluid coverage vs stable regions","coverage_vs_stability.png"),
        ("decoder_stable_overlap","Decoder heatmap overlap","Estimated fluid coverage vs aligned decoder/stable overlap","coverage_vs_decoder_overlap.png"),
    ):
        fig,ax=plt.subplots(figsize=(6,5))
        for cond in conditions:
            subset=[r for r in by_case if r["condition"]==cond and np.isfinite(number(r.get(metric)))]
            ax.scatter([number(r["estimated_dynamic_coverage"]) for r in subset],[number(r[metric]) for r in subset],
                       alpha=.75,s=28,color=colors[cond],label=cond)
        ax.set(xlabel="Estimated dynamic-fluid coverage (%)",ylabel=ylabel,title=title)
        ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(OUT/filename,dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    valid=[r for r in by_case if np.isfinite(number(r.get("mean_raw_ba"))) and np.isfinite(number(r.get("mean_aligned_ba")))]
    x=np.arange(len(valid));width=.38
    ax.bar(x-width/2,[number(r["mean_raw_ba"]) for r in valid],width,label="Raw")
    ax.bar(x+width/2,[number(r["mean_aligned_ba"]) for r in valid],width,label="RAFT aligned")
    ax.set(xlabel="Case index",ylabel="Mean ACC",ylim=(0,1),title="Raw and aligned mean ACC by case")
    ax.grid(axis="y",alpha=.25);ax.legend();fig.tight_layout();fig.savefig(OUT/"case_raw_aligned_acc.png",dpi=160);plt.close(fig)

    report=["# CLAW LoT-Pass × CogVideoX analysis","",
            f"Scheduled cases: {len(manifest['cases'])} (15 images × 2 prompts). Raw-decoded: {len(complete)}.",
            "All clip-level summaries use 49 generated frames when available. A frame-zero copy-like frame is excluded from mean ACC only when detected using the recorded MAE/SSIM rule.","",
            "## Prompt-condition summary","",
            "| Condition | Completed | Raw mean ACC | Aligned mean ACC | Stable-region ratio | Largest stable region | Decoder stable overlap |","|---|---:|---:|---:|---:|---:|---:|"]
    for row in condition_summary:
        def fmt(key):
            x=number(row.get("mean_"+key));return f"{x:.4f}" if np.isfinite(x) else "NA"
        report.append(f"| {row['condition']} | {row['completed_raw_cases']} | {fmt('mean_raw_ba')} | {fmt('mean_aligned_ba')} | {fmt('stable_region_ratio')} | {fmt('largest_stable_region_ratio')} | {fmt('decoder_stable_overlap')} |")
    report += ["","## Estimated coverage relationships","",
               "Coverage correlations average the two prompt-condition outcomes within each image first; n is therefore at most 15 source images.",
               "| Outcome | n | Pearson r (p) | Spearman ρ (p) |","|---|---:|---:|---:|"]
    for row in correlation_rows:
        pearson="NA" if row.get("pearson_r") is None else f"{float(row['pearson_r']):.3f} ({float(row['pearson_p']):.3g})"
        spearman="NA" if row.get("spearman_rho") is None else f"{float(row['spearman_rho']):.3f} ({float(row['spearman_p']):.3g})"
        report.append(f"| {row['y']} | {row['n_images']} | {pearson} | {spearman} |")
    report += ["","## Interpretation notes","",
               "This is an exploratory, paired 15-image pilot; coverage estimates are visual pre-generation judgments, not measured motion ground truth. CLAW v1 is curated for transparent/semi-transparent liquid scenes. Interpret correlations descriptively, not as causal evidence.",
               "","Raw and RAFT-aligned ACC are reported separately. RAFT alignment can resample or miswarp frames; it is a diagnostic view, not a replacement for raw decoding.",
               "","See `per_case_metrics.csv`, `paired_condition_deltas.csv`, `dynamic_coverage_correlations.csv`, and `optical_flow/` for case-level data and full-frame aligned/heatmap videos."]
    (OUT/"final_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(f"Wrote combined summaries for {len(manifest['cases'])} planned cases ({len(complete)} raw-decoded).",flush=True)


if __name__=="__main__":main()
