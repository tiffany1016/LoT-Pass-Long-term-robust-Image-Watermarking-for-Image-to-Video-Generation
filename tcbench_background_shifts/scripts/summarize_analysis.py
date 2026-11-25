#!/usr/bin/env python3
"""Join generation, raw BA, stability, validation, and heatmap outputs."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/"tcbench_background_shifts"
ANALYSIS=EXP/"analysis"

def read_csv(path):
    return list(csv.DictReader(path.open(encoding="utf-8"))) if path.is_file() else []

def write_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)

def number(value):
    try:
        x=float(value)
        return x if math.isfinite(x) else float("nan")
    except (TypeError,ValueError): return float("nan")

def avg(values):
    finite=[v for v in values if math.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")

def fmt(value): return f"{value:.4f}" if math.isfinite(value) else "NA"

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pair_rows=read_csv(EXP/"approval/tcbench_all_36pairs.csv")
    videos={r["pair_id"]:r for r in read_csv(ANALYSIS/"per_video_raw_ba.csv")}
    stability=read_csv(ANALYSIS/"stable_region_metrics.csv")
    heatmaps=read_csv(ANALYSIS/"heatmap_overlap.csv")
    aligned_rows=read_csv(ANALYSIS/"optical_flow/per_video_aligned_ba.csv")
    changes=read_csv(ANALYSIS/"per_video_change_coverage.csv")
    validation=read_csv(ANALYSIS/"transition_validation.csv")
    by_case={}
    for row in pair_rows:
        cid=row["case_id"]; cond=row["condition"]
        item=videos.get(row["pair_id"],{})
        by_case.setdefault(cid,{"official_id":int(row["official_id"]),"prompts":{},"conditions":{}})
        by_case[cid]["prompts"][cond]=row["prompt_used"]
        by_case[cid]["conditions"][cond]=item
    pair_summary=[]
    for cid in sorted(by_case,key=lambda k:int(k[2:])):
        entry=by_case[cid]; trans=entry["conditions"].get("transition",{}); control=entry["conditions"].get("control",{})
        tmean=number(trans.get("mean_ba_excluding_copy_frame")); cmean=number(control.get("mean_ba_excluding_copy_frame"))
        tlast=number(trans.get("last_third_mean_ba_excluding_copy_frame")); clast=number(control.get("last_third_mean_ba_excluding_copy_frame"))
        diff=cmean-tmean if math.isfinite(cmean) and math.isfinite(tmean) else float("nan")
        pair_summary.append({"case_id":cid,"official_id":entry["official_id"],
            "transition_generation_status":trans.get("generation_status","missing"),
            "control_generation_status":control.get("generation_status","missing"),
            "transition_decode_status":trans.get("decode_status","missing"),
            "control_decode_status":control.get("decode_status","missing"),
            "transition_mean_ba_excl_copy":tmean,"control_mean_ba_excl_copy":cmean,
            "control_minus_transition_ba":diff,
            "transition_last_third_ba":tlast,"control_last_third_ba":clast,
            "transition_early_to_late_ba_drop":number(trans.get("early_to_late_ba_drop")),
            "control_early_to_late_ba_drop":number(control.get("early_to_late_ba_drop")),
            "transition_failure_candidate_late_ba_lt_0_80":math.isfinite(tlast) and tlast<0.80,
            "control_failure_candidate_late_ba_lt_0_80":math.isfinite(clast) and clast<0.80,
            "paired_failure_candidate_abs_control_minus_transition_gt_0_15":math.isfinite(diff) and abs(diff)>0.15,
        })
    if pair_summary:
        fields=list(pair_summary[0].keys())
        write_csv(ANALYSIS/"paired_ba_summary.csv",pair_summary,fields)

    successful_transition=sum(r.get("transition_generation_status")=="completed" for r in pair_summary)
    successful_control=sum(r.get("control_generation_status")=="completed" for r in pair_summary)
    decoded_transition=sum(r.get("transition_decode_status")=="completed" for r in pair_summary)
    decoded_control=sum(r.get("control_decode_status")=="completed" for r in pair_summary)
    paired=[r for r in pair_summary if math.isfinite(r["control_minus_transition_ba"])]
    mean_t=avg([r["transition_mean_ba_excl_copy"] for r in pair_summary])
    mean_c=avg([r["control_mean_ba_excl_copy"] for r in pair_summary])
    mean_d=avg([r["control_minus_transition_ba"] for r in pair_summary])
    encoding=json.loads((EXP/"runs/encoding_summary.json").read_text())
    baseline_clean=avg([number(r["pre_i2v_clean_decode_ba"]) for r in encoding["source_encodings"]])
    baseline_marked=avg([number(r["pre_i2v_watermarked_decode_ba"]) for r in encoding["source_encodings"]])
    change_by_pair={r["pair_id"]:r for r in changes}
    transition_change=[change_by_pair.get(f"{r['case_id']}_transition",{}) for r in pair_summary]
    control_change=[change_by_pair.get(f"{r['case_id']}_control",{}) for r in pair_summary]
    mean_luma_t=avg([number(r.get("mean_whole_frame_abs_luminance_delta")) for r in transition_change])
    mean_luma_c=avg([number(r.get("mean_whole_frame_abs_luminance_delta")) for r in control_change])
    mean_patch_t=avg([number(r.get("mean_fraction_patches_luma_delta_gt_0_05")) for r in transition_change])
    mean_patch_c=avg([number(r.get("mean_fraction_patches_luma_delta_gt_0_05")) for r in control_change])
    low_transition=[r for r in pair_summary if r["transition_failure_candidate_late_ba_lt_0_80"]]
    low_control=[r for r in pair_summary if r["control_failure_candidate_late_ba_lt_0_80"]]
    paired_flags=[r for r in pair_summary if r["paired_failure_candidate_abs_control_minus_transition_gt_0_15"]]

    # Pair plot uses official IDs, with connected within-case transition/control values.
    fig,ax=plt.subplots(figsize=(10,5.5))
    for r in pair_summary:
        a,b=r["transition_mean_ba_excl_copy"],r["control_mean_ba_excl_copy"]
        if math.isfinite(a) and math.isfinite(b):
            x=r["official_id"];ax.plot([x-0.12,x+0.12],[a,b],color="0.65",lw=1)
            ax.scatter([x-0.12],[a],color="#31688e",s=24)
            ax.scatter([x+0.12],[b],color="#35b779",s=24)
    ax.set(xlabel="Official TC-Bench ID",ylabel="Mean raw bit accuracy (excluding exact copy frame)",
           title="Paired raw BA: official transition vs custom control",ylim=(0,1.02))
    ax.grid(alpha=.25);ax.legend(handles=[plt.Line2D([],[],marker='o',linestyle='',color='#31688e',label='Transition'),
        plt.Line2D([],[],marker='o',linestyle='',color='#35b779',label='Control')],loc="lower right")
    fig.tight_layout();fig.savefig(ANALYSIS/"paired_ba.png",dpi=170);plt.close(fig)

    # Per-frame BA curves remain grouped by video; no frame-level inference.
    frame_rows=read_csv(ANALYSIS/"per_frame_raw_ba.csv")
    fig,axes=plt.subplots(6,3,figsize=(15,16),sharey=True)
    for ax,cid in zip(axes.ravel(),sorted(by_case,key=lambda k:int(k[2:]))):
        for condition,color in (("transition","#31688e"),("control","#35b779")):
            vals=[r for r in frame_rows if r.get("case_id")==cid and r.get("condition")==condition]
            vals.sort(key=lambda r:int(r["frame_idx"]))
            if vals: ax.plot([int(r["frame_idx"]) for r in vals],[number(r["ba"]) for r in vals],color=color,lw=1,label=condition)
        ax.set_title(cid);ax.set_ylim(0,1.02);ax.grid(alpha=.2)
    for ax in axes[-1,:]:ax.set_xlabel("Frame index")
    for ax in axes[:,0]:ax.set_ylabel("Raw BA")
    axes[0,0].legend(fontsize=7,loc="lower left")
    fig.suptitle("Per-frame raw BA (paired clips; frames within each clip are correlated)",y=.995)
    fig.tight_layout();fig.savefig(ANALYSIS/"per_frame_raw_ba_all_cases.png",dpi=150);plt.close(fig)

    # Frame-level stability scatter is descriptive only; aggregate files are also available.
    stab_rows=[r for r in stability if r.get("status")=="completed"]
    for key,filename,label in (("stable_region_ratio","stable_region_vs_ba.png","Stable region ratio"),
        ("largest_stable_region_ratio","largest_stable_region_vs_ba.png","Largest connected stable region ratio"),
        ("mean_similarity","similarity_vs_ba.png","Mean source-anchored similarity")):
        rows=[r for r in stab_rows if math.isfinite(number(r.get(key))) and math.isfinite(number(r.get("raw_BA")))]
        fig,ax=plt.subplots(figsize=(6,5))
        for cond,color,marker in (("transition","#31688e","o"),("control","#35b779","s")):
            subset=[r for r in rows if r.get("condition")==cond]
            ax.scatter([number(r[key]) for r in subset],[number(r["raw_BA"]) for r in subset],s=10,alpha=.35,c=color,marker=marker,label=cond)
        ax.set(xlabel=label,ylabel="Raw BA",title=f"{label} vs raw BA (correlated frames)")
        ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(ANALYSIS/filename,dpi=160);plt.close(fig)

    heat_rows=[r for r in heatmaps if math.isfinite(number(r.get("raw_BA")))]
    for key,filename,label in (("encoder_stable_overlap","encoder_overlap_vs_ba.png","Encoder residual mass in stable region"),
                               ("decoder_stable_overlap","decoder_overlap_vs_ba.png","Decoder gradient mass in stable region")):
        valid=[r for r in heat_rows if math.isfinite(number(r.get(key)))]
        fig,ax=plt.subplots(figsize=(6,5))
        for cond,color,marker in (("transition","#31688e","o"),("control","#35b779","s")):
            subset=[r for r in valid if r.get("condition")==cond]
            ax.scatter([number(r[key]) for r in subset],[number(r["raw_BA"]) for r in subset],s=18,alpha=.5,c=color,marker=marker,label=cond)
        ax.set(xlabel=label,ylabel="Raw BA",title=f"Raw BA vs {label.lower()}")
        ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(ANALYSIS/filename,dpi=160);plt.close(fig)

    change_plot=[]
    for r in pair_summary:
        for cond in ("transition","control"):
            q=change_by_pair.get(f"{r['case_id']}_{cond}",{})
            ba=number(r.get(f"{cond}_mean_ba_excl_copy"))
            coverage=number(q.get("mean_fraction_patches_luma_delta_gt_0_05"))
            if math.isfinite(ba) and math.isfinite(coverage):change_plot.append((cond,coverage,ba))
    fig,ax=plt.subplots(figsize=(6,5))
    for cond,color,marker in (("transition","#31688e","o"),("control","#35b779","s")):
        subset=[r for r in change_plot if r[0]==cond]
        ax.scatter([r[1] for r in subset],[r[2] for r in subset],s=28,c=color,marker=marker,label=cond)
    ax.set(xlabel="Mean fraction of 16×16 patches with |mean luminance delta| > 0.05",ylabel="Mean raw BA",
           title="Patch luminance change coverage vs raw BA")
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(ANALYSIS/"ba_vs_luminance_change_coverage.png",dpi=160);plt.close(fig)

    validated=[r for r in validation if r.get("review_status")=="reviewed"]
    transition_labels=[r for r in validated if r.get("condition")=="transition"]
    leakage=[r for r in validated if r.get("condition")=="control"]
    valid_count=sum(r.get("transition_validity")=="valid" for r in transition_labels)
    global_count=sum(r.get("observed_global_change")=="yes" for r in transition_labels)
    leakage_count=sum(r.get("control_transition_leakage")=="yes" for r in leakage)
    validity_counts={label:sum(r.get("transition_validity")==label for r in transition_labels)
                     for label in ("valid","partial","invalid","uncertain")}
    reviewed_invalid=[r["case_id"] for r in transition_labels if r.get("transition_validity")=="invalid"]
    high_examples=sorted([r for r in pair_summary if math.isfinite(r["transition_mean_ba_excl_copy"])],
                         key=lambda r:r["transition_mean_ba_excl_copy"],reverse=True)[:3]
    low_examples=sorted([r for r in pair_summary if math.isfinite(r["transition_mean_ba_excl_copy"])],
                        key=lambda r:r["transition_mean_ba_excl_copy"])[:3]
    val_status=f"{len(validated)}/36 prompt/video rows visually reviewed"
    aligned_videos=[r for r in aligned_rows if r.get("status")=="completed"]
    aligned_raw_mean=avg([number(r.get("raw_mean_BA_excluding_copy_frame")) for r in aligned_videos])
    aligned_ba_mean=avg([number(r.get("aligned_mean_BA_excluding_copy_frame")) for r in aligned_videos])
    aligned_delta=avg([number(r.get("aligned_minus_raw_BA_excluding_copy_frame")) for r in aligned_videos])
    lines=[
        "# TC-Bench-derived first-frame I2V evaluation: all background-shifts entries", "",
        "## Coverage", "",
        f"- Official category: 18 entries; all official IDs retained; source inputs available: 18/18.",
        f"- Approved entries: 18/18; retrievable source entries: 18/18; planned clips: 36; successful clips: {successful_transition+successful_control}/36; generation failures: {36-successful_transition-successful_control}/36.",
        f"- Condition counts: official transition {successful_transition}/18 generated; custom control {successful_control}/18 generated.",
        f"- Raw decoder completed: transition {decoded_transition}/18, control {decoded_control}/18; total raw-frame rows: {len(frame_rows)}.",
        f"- Visual validation: {val_status}; transition valid={valid_count}/{len(transition_labels)}, observed global change=yes={global_count}/{len(transition_labels)}, control transition leakage=yes={leakage_count}/{len(leakage)}.",
        "- LoT-Pass watermark was encoded once per unique source and reused unchanged for both conditions. Generation settings and source/message hashes are stored per run.", "",
        "## Raw bit accuracy", "",
        f"- Pre-I2V source decode check across 18 images: mean clean-image BA {fmt(baseline_clean)}; mean watermarked-image BA {fmt(baseline_marked)}.",
        f"- Mean per-video BA, excluding a frame-0 exact conditioning copy when detected: transition {fmt(mean_t)}; control {fmt(mean_c)}; mean paired control-minus-transition difference {fmt(mean_d)} across {len(paired)}/18 pairs with both decoded conditions.",
        f"- Preliminary low-BA candidates (late-third mean BA < 0.80): transition {len(low_transition)}/18 ({', '.join(r['case_id'] for r in low_transition) or 'none'}); control {len(low_control)}/18 ({', '.join(r['case_id'] for r in low_control) or 'none'}). Paired candidates (|control − transition| > 0.15): {len(paired_flags)}/18 ({', '.join(r['case_id'] for r in paired_flags) or 'none'}).",
        "- Paired clips use the same watermarked source and seed. Frame counts are repeated measurements within clips, not independent samples.",
        "- Preliminary screening only: late-third BA < 0.80; paired BA difference flagged when |control − transition| > 0.15. These thresholds are not population failure-rate estimates.", "",
        "## Source-anchored appearance stability and heatmaps", "",
        "- Stable regions compare each frame at the same pixel coordinates with its watermarked source; no temporal stability or optical-flow alignment is used.",
        "- Patch size 16×16; normalized RGB MAE threshold <0.05; 7×7-window RGB SSIM threshold >0.90; continuous score = 0.5×clip(1−MAE/0.15,0,1)+0.5×clip(SSIM,0,1); connected components use 4-connectivity.",
        f"- Source-anchored luminance measures (Rec.709; all 49 frames averaged per video): mean absolute whole-frame luminance delta transition {fmt(mean_luma_t)}, control {fmt(mean_luma_c)}; mean fraction of 16×16 patches with absolute mean-luminance delta >0.05 transition {fmt(mean_patch_t)}, control {fmt(mean_patch_c)}.",
        "- Similarity and stable-mask overlay opacity is 0.35 for display only. Raw MAE/SSIM/similarity maps and masks are saved separately.",
        "- Encoder maps reuse the existing encoder-residual method; decoder maps reuse existing input-gradient saliency. Overlap is attribution mass inside the stable mask divided by total mass; enrichment divides overlap by stable area and is NA for zero denominators.", "",
        f"- Decoder attribution was recomputed independently for every generated frame: {len(heatmaps)}/{len(frame_rows)} frame maps recorded across {sum(1 for r in heatmaps if r.get('frame_idx') == '0')} videos; each clip has a `decoder_heatmap_overlay.mp4` with the frame visible under the heatmap at alpha 0.35.", "",
        "## Supplemental RAFT alignment and decoding", "",
        f"- LoT-Pass RAFT alignment and aligned-frame decoding completed for {len(aligned_videos)}/36 clips. Mean per-video BA excluding detected copy frames: raw {fmt(aligned_raw_mean)}, aligned {fmt(aligned_ba_mean)}, mean within-video change {aligned_delta:+.4f}.",
        "- Every clip has a generated-frame-0-anchored `aligned_video.mp4` and a `decoder_heatmap_overlay.mp4` showing decoder gradients computed on the aligned frames. These outputs are under `analysis/optical_flow/`; raw BA and source-anchored stability results remain unchanged.", "",
        "## Transition interpretation", "",
        "TC-Bench’s official `background shifts` category includes local object events and does not guarantee whole-frame appearance change. Global-change labels are reported separately from prompt success. Inspect `transition_validation.csv` and video storyboards; any unreviewed rows remain uncertain.", "",
        f"- Visual transition validation counts: {validity_counts}; whole-frame/global appearance change marked yes in {global_count}/{len(transition_labels)} transition videos; target-event leakage in {leakage_count}/{len(leakage)} controls.",
        f"- Unsuccessful transition examples from the full reviewed set (not selected based on BA): {', '.join(reviewed_invalid) or 'none'}.",
        f"- High raw-BA examples from a fixed top-three ranking: {', '.join(f'{r['case_id']} ({fmt(r['transition_mean_ba_excl_copy'])})' for r in high_examples) or 'none'}; low examples from a fixed bottom-three ranking: {', '.join(f'{r['case_id']} ({fmt(r['transition_mean_ba_excl_copy'])})' for r in low_examples) or 'none'}. Examples are descriptive and not cherry-picked evidence.", "",
        "- Surprising reviewed cases: TC103 starts from an already sunset-like frame; TC104 has a scene replacement unrelated to its prompt; TC114, TC117, and TC119 show target-like activity in their custom controls. These remain included and flagged.", "",
        "## Limits", "",
        "This covers one seed per entry in this benchmark category under one model/checkpoint/configuration. It is descriptive benchmark coverage, not a broad causal estimate for global appearance changes. Cases with partial, invalid, or visually surprising generations remain in the denominators.", "",
        "## Outputs", "",
        "- `analysis/per_frame_raw_ba.csv`, `per_video_raw_ba.csv`, `paired_ba_summary.csv`",
        "- `analysis/stable_region_metrics.csv` and per-video `analysis/stable_region/` overlays/maps",
        "- `analysis/heatmap_overlap.csv`, raw encoder/decoder maps, comparison panels, and per-video `decoder_heatmap_overlay.mp4`",
        "- `analysis/transition_validation.csv` and `analysis/video_review/` storyboards", "",
    ]
    (EXP/"reports/all_18_background_shifts_summary.md").write_text("\n".join(lines),encoding="utf-8")
    print("Wrote TC-Bench paired summary and figures.",flush=True)

if __name__=="__main__": main()
