#!/usr/bin/env python3
"""RAFT-align every generated frame, decode aligned ACC, and render heatmap videos."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "claw_lotpass_pilot"
TORCH_HOME = ROOT / "vbench_lotpass_pilot/model_cache/torch"
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP / "scripts"))
CHECKPOINT = ROOT / "checkpoints/model-99.pth"
INPUT_RES = 256
ALPHA = 0.35


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)


def warp_to_ref(ref: np.ndarray, target: np.ndarray, flow: np.ndarray, cv2) -> np.ndarray:
    h,w=ref.shape[:2]
    gx,gy=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
    return cv2.remap(target,gx+flow[...,0].astype(np.float32),gy+flow[...,1].astype(np.float32),
                     interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)


def overlap(heat: np.ndarray, mask: np.ndarray) -> float:
    denom=float(np.sum(heat))
    return float(np.sum(heat*mask)/denom) if denom>1e-12 else float("nan")


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise RuntimeError("GPU policy requires CUDA_VISIBLE_DEVICES=0")
    manifest=json.loads((EXP/"selection/generation_manifest.json").read_text())
    approval=json.loads((EXP/"selection/approval.json").read_text())
    if not approval.get("approved"):
        raise RuntimeError("The Stage A package is not approved")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.init()
    import cv2
    import decord
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision import transforms
    from utils.utils import TorchvisionRAFT
    from gradient_saliency import compute_saliency, load_decoder, make_visualization
    from stable_regions import patch_grids, expand

    device=torch.device("cuda:0")
    decoder=load_decoder(CHECKPOINT,CHECKPOINT.parent,device)
    for param in decoder.parameters(): param.requires_grad_(False)
    raft=TorchvisionRAFT(device="cuda:0")
    transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((INPUT_RES,INPUT_RES)),transforms.Normalize([.5]*3,[.5]*3)])
    archive=np.load(EXP/"runs/source_message_map.npz",allow_pickle=True)
    messages={k:np.asarray(archive[k]).reshape(-1).astype(np.float32) for k in archive.files if k!="ckpt"}
    raw_rows=read_csv(EXP/"analysis/raw_ba/per_frame_accuracy.csv")
    raw={(r["case_id"],int(r["frame_idx"])):float(r["ba"]) for r in raw_rows}
    raw_summary={r["case_id"]:r for r in read_csv(EXP/"analysis/raw_ba/case_summary.csv")}
    encoder_root=EXP/"analysis/heatmaps/encoder/raw_maps/canonical_clean"
    output_root=EXP/"analysis/optical_flow"
    frame_rows,video_rows=[],[]

    for case_index,case in enumerate(manifest["cases"],start=1):
        directory=EXP/"runs"/case["case_id"]/f"seed_{case['seed']}"
        video=directory/"video.mp4"
        status_path=directory/"run_status.json"
        status=json.loads(status_path.read_text()) if status_path.is_file() else {}
        common={"case_id":case["case_id"],"image_id":case["image_id"],"sample_index":case["sample_index"],"condition":case["condition"],"seed":case["seed"]}
        if status.get("status")!="completed" or not video.is_file():
            video_rows.append({**common,"status":f"generation_{status.get('status','missing')}"})
            continue
        if case["image_id"] not in messages: raise KeyError(f"No message for {case['image_id']}")
        encoder_map_path=encoder_root/f"{case['image_id']}.npy"
        encoder_map=np.load(encoder_map_path).astype(np.float32) if encoder_map_path.is_file() else None
        reader=decord.VideoReader(str(video),ctx=decord.cpu(0)); count=len(reader); fps=float(reader.get_avg_fps())
        if count==0 or fps<=0: raise ValueError(f"Invalid source video {video}")
        reference=reader[0].asnumpy(); height,width=reference.shape[:2]
        source=np.asarray(Image.open(directory/"effective_input.png").convert("RGB"),dtype=np.uint8)
        truth=torch.as_tensor(messages[case["image_id"]],dtype=torch.float32,device=device).unsqueeze(0)
        out=output_root/case["case_id"]
        grad_dir=out/"raw_gradients"; overlay_dir=out/"overlays"
        grad_dir.mkdir(parents=True,exist_ok=True);overlay_dir.mkdir(parents=True,exist_ok=True)
        aligned_final=out/"aligned_video.mp4"; overlay_final=out/"decoder_heatmap_overlay.mp4"
        aligned_tmp=out/".aligned_video.partial.mp4";overlay_tmp=out/".decoder_heatmap_overlay.partial.mp4"
        writer=imageio.get_writer(str(aligned_tmp),fps=fps,codec="libx264",quality=8,macro_block_size=1,
                                  ffmpeg_params=["-preset","veryfast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart"])
        heat_writer=imageio.get_writer(str(overlay_tmp),fps=fps,codec="libx264",quality=8,macro_block_size=1,
                                       ffmpeg_params=["-preset","veryfast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart"])
        case_frames=[];aligned_values=[];dxs=[];dys=[];magnitudes=[]
        try:
            for frame_idx in range(count):
                original=reader[frame_idx].asnumpy()
                if original.shape!=reference.shape: raise ValueError(f"Frame geometry changes in {video}")
                if frame_idx==0:
                    aligned=original;dx=dy=mag=0.0
                else:
                    flow=raft.compute_flow(reference,original)
                    aligned=warp_to_ref(reference,original,flow,cv2)
                    dx=float(np.median(flow[...,0]));dy=float(np.median(flow[...,1]))
                    mag=float(np.median(np.sqrt(flow[...,0]**2+flow[...,1]**2)))
                writer.append_data(aligned)
                tensor=transform(Image.fromarray(aligned).convert("RGB")).unsqueeze(0).to(device)
                decoded,saliency=compute_saliency(decoder,tensor,truth,target_bit=-1)
                saliency=np.asarray(saliency,dtype=np.float32)
                np.save(grad_dir/f"frame_{frame_idx:04d}.npy",saliency)
                bits=decoded.detach().cpu().numpy().round().clip(0,1)
                aligned_ba=float(1-np.mean(np.abs(bits-truth.detach().cpu().numpy())))
                raw_ba=raw.get((case["case_id"],frame_idx),float("nan"))
                _,overlay=make_visualization(Image.fromarray(aligned),saliency,alpha=ALPHA)
                overlay_np=np.asarray(overlay,dtype=np.uint8);heat_writer.append_data(overlay_np)
                if frame_idx in (0,12,24,36,48):
                    overlay.save(overlay_dir/f"frame_{frame_idx:04d}_aligned_decoder_heatmap.png")
                heat_up=cv2.resize(np.maximum(saliency,0),(width,height),interpolation=cv2.INTER_LINEAR)
                _,_,areas,stable,_,stable_metrics=patch_grids(aligned,source)
                mask=np.repeat(np.repeat(stable.astype(np.uint8),16,axis=0),16,axis=1)[:height,:width].astype(np.float32)
                dec_overlap=overlap(heat_up,mask)
                enc_overlap=overlap(encoder_map,mask) if encoder_map is not None else float("nan")
                srr=stable_metrics["stable_region_ratio"]
                entry={**common,"frame_idx":frame_idx,"fps":fps,"raw_BA":raw_ba,"aligned_BA":aligned_ba,
                       "aligned_minus_raw_BA":aligned_ba-raw_ba if np.isfinite(raw_ba) else float("nan"),
                       "median_flow_dx_px":dx,"median_flow_dy_px":dy,"median_flow_magnitude_px":mag,
                       "stable_region_ratio_aligned":srr,"largest_stable_region_ratio_aligned":stable_metrics["largest_stable_region_ratio"],
                       "mean_similarity_aligned":stable_metrics["mean_similarity"],"decoder_stable_overlap":dec_overlap,
                       "decoder_stable_enrichment":dec_overlap/srr if srr>1e-12 and np.isfinite(dec_overlap) else float("nan"),
                       "encoder_stable_overlap":enc_overlap,"alignment_reference_frame":0,
                       "alignment_method":"LoT-Pass TorchvisionRAFT; flow(reference frame 0,target); cv2.remap at grid+flow with INTER_LINEAR/BORDER_REPLICATE"}
                frame_rows.append(entry);case_frames.append(entry);aligned_values.append(aligned_ba)
                dxs.append(dx);dys.append(dy);magnitudes.append(mag)
        finally:
            writer.close();heat_writer.close()
        os.replace(aligned_tmp,aligned_final);os.replace(overlay_tmp,overlay_final)
        raw_values=[r["raw_BA"] for r in case_frames]
        copy_like=raw_summary.get(case["case_id"],{}).get("frame0_copy_like","False").lower()=="true"
        raw_ex=raw_values[1:] if copy_like and len(raw_values)>1 else raw_values
        aligned_ex=aligned_values[1:] if copy_like and len(aligned_values)>1 else aligned_values
        video_rows.append({**common,"status":"completed","frame_count":count,"fps":fps,"frame0_copy_like":copy_like,
                           "raw_mean_BA_all_frames":float(np.nanmean(raw_values)),
                           "raw_mean_BA_excluding_copy_frame":float(np.nanmean(raw_ex)),
                           "aligned_mean_BA_all_frames":float(np.mean(aligned_values)),
                           "aligned_mean_BA_excluding_copy_frame":float(np.mean(aligned_ex)),
                           "aligned_minus_raw_BA_excluding_copy_frame":float(np.mean(aligned_ex)-np.nanmean(raw_ex)),
                           "mean_stable_region_ratio_aligned":float(np.mean([r["stable_region_ratio_aligned"] for r in case_frames])),
                           "mean_largest_stable_region_ratio_aligned":float(np.mean([r["largest_stable_region_ratio_aligned"] for r in case_frames])),
                           "mean_similarity_aligned":float(np.mean([r["mean_similarity_aligned"] for r in case_frames])),
                           "mean_decoder_stable_overlap":float(np.nanmean([r["decoder_stable_overlap"] for r in case_frames])),
                           "mean_encoder_stable_overlap":float(np.nanmean([r["encoder_stable_overlap"] for r in case_frames])),
                           "median_flow_dx_px":float(np.median(dxs)),"median_flow_dy_px":float(np.median(dys)),
                           "median_flow_magnitude_px":float(np.median(magnitudes)),
                           "aligned_video":str(aligned_final.relative_to(ROOT)),
                           "decoder_heatmap_overlay_video":str(overlay_final.relative_to(ROOT)),
                           "raw_gradients_dir":str(grad_dir.relative_to(ROOT))})
        (out/"alignment_record.json").write_text(json.dumps({
            **common,"source_video":str(video.relative_to(ROOT)),"frame_count":count,"fps":fps,"resolution":[width,height],
            "reference_frame":0,"source_video_unchanged":True,"watermark_checkpoint":str(CHECKPOINT.relative_to(ROOT)),
            "decoder_heatmap":"input-gradient magnitude for mean signed ground-truth bit score; all aligned frames; overlay alpha 0.35",
            "heatmap_stability_overlap":"computed in aligned coordinates using source-anchored 16x16 patch masks",
        },indent=2)+"\n",encoding="utf-8")
        print(f"RAFT aligned and heatmapped {case['case_id']}: {count} frames",flush=True)

    out=EXP/"analysis/optical_flow"
    frame_fields=list(dict.fromkeys(k for row in frame_rows for k in row))
    video_fields=list(dict.fromkeys(k for row in video_rows for k in row))
    write_csv(out/"per_frame_aligned_ba.csv",frame_rows,frame_fields or ["case_id"])
    write_csv(out/"per_video_aligned_ba.csv",video_rows,video_fields or ["case_id"])
    complete=[r for r in video_rows if r.get("status")=="completed"]
    conditions=("static_control","broad_fluid_motion")
    summaries={}
    for condition in conditions:
        subset=[r for r in complete if r["condition"]==condition]
        summaries[condition]={
            "video_count":len(subset),
            "raw_mean_ba_excluding_copy_frame":float(np.mean([r["raw_mean_BA_excluding_copy_frame"] for r in subset])) if subset else None,
            "aligned_mean_ba_excluding_copy_frame":float(np.mean([r["aligned_mean_BA_excluding_copy_frame"] for r in subset])) if subset else None,
            "mean_alignment_delta":float(np.mean([r["aligned_minus_raw_BA_excluding_copy_frame"] for r in subset])) if subset else None,
        }
    (out/"summary.json").write_text(json.dumps({"completed_videos":len(complete),"condition_summary":summaries,
        "method":"Existing LoT-Pass RAFT implementation; reference is generated frame 0; target warped to reference.",
        "heatmap_frames_per_video":"all frames"},indent=2)+"\n",encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    color={"static_control":"#315b7d","broad_fluid_motion":"#d77732"}
    positions,labels,heights,bar_colors,bar_alpha=[],[],[],[],[]
    for group_index,cond in enumerate(conditions):
        vals=[r for r in complete if r["condition"]==cond]
        if vals:
            x=group_index*3
            positions.extend([x,x+1]);labels.extend([cond+" raw",cond+" aligned"])
            heights.extend([np.mean([r["raw_mean_BA_excluding_copy_frame"] for r in vals]),
                            np.mean([r["aligned_mean_BA_excluding_copy_frame"] for r in vals])])
            bar_colors.extend([color[cond],color[cond]]);bar_alpha.extend([.55,1.0])
    fig,ax=plt.subplots(figsize=(9,4.5))
    bars=ax.bar(positions,heights,color=bar_colors)
    for bar,alpha in zip(bars,bar_alpha): bar.set_alpha(alpha)
    ax.set_xticks(positions,labels,rotation=15,ha="right")
    ax.set(ylabel="Mean bit accuracy",ylim=(0,1),title="Raw vs RAFT-aligned ACC by condition");ax.grid(axis="y",alpha=.25)
    fig.tight_layout();fig.savefig(out/"raw_vs_aligned_acc.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    for cond in conditions:
        by_case={r["case_id"]:r for r in frame_rows if r["condition"]==cond}
        cases=[row["case_id"] for row in video_rows if row.get("status")=="completed" and row["condition"]==cond]
        if not cases: continue
        raw_matrix=np.asarray([[r["raw_BA"] for r in frame_rows if r["case_id"]==case_id] for case_id in cases],dtype=np.float32)
        aligned_matrix=np.asarray([[r["aligned_BA"] for r in frame_rows if r["case_id"]==case_id] for case_id in cases],dtype=np.float32)
        frame_idx=np.arange(min(raw_matrix.shape[1],aligned_matrix.shape[1]))
        raw_mean=np.nanmean(raw_matrix[:,:len(frame_idx)],axis=0);aligned_mean=np.nanmean(aligned_matrix[:,:len(frame_idx)],axis=0)
        ax.plot(frame_idx,raw_mean,color=color[cond],linestyle="-",label=f"{cond} raw")
        ax.plot(frame_idx,aligned_mean,color=color[cond],linestyle="--",label=f"{cond} RAFT-aligned")
    ax.set(xlabel="Frame index",ylabel="Bit accuracy",ylim=(0,1.02),title="Raw and RAFT-aligned ACC curves")
    ax.grid(alpha=.25);ax.legend(ncol=2);fig.tight_layout();fig.savefig(out/"acc_curve_raw_vs_aligned_by_condition.png",dpi=160);plt.close(fig)
    report=["# CLAW optical-flow alignment analysis","",f"Completed videos: {len(complete)}/{len(video_rows)}.","",
            "Each generated frame was aligned to generated frame 0 with the existing LoT-Pass Torchvision RAFT flow and remap. The decoder heatmap overlay video includes every aligned frame.","",
            "| Condition | Clips | Raw mean ACC | RAFT-aligned mean ACC | Mean change |","|---|---:|---:|---:|---:|"]
    for cond in conditions:
        s=summaries[cond]
        report.append(f"| {cond} | {s['video_count']} | {s['raw_mean_ba_excluding_copy_frame'] if s['raw_mean_ba_excluding_copy_frame'] is not None else 'NA'} | {s['aligned_mean_ba_excluding_copy_frame'] if s['aligned_mean_ba_excluding_copy_frame'] is not None else 'NA'} | {s['mean_alignment_delta'] if s['mean_alignment_delta'] is not None else 'NA'} |")
    report += ["","`per_frame_aligned_ba.csv` includes raw/aligned ACC, flow displacement, source-anchored aligned stability, and encoder/decoder stable-overlap metrics."]
    (out/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(f"Optical-flow analysis complete: {len(complete)}/{len(video_rows)} videos.",flush=True)


if __name__=="__main__":
    main()
