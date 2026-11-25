#!/usr/bin/env python3
"""Compute existing decoder gradient saliency on every frame and render videos."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
EXP=ROOT/"tcbench_background_shifts"
GENERATIONS=EXP/"generations"
ANALYSIS=EXP/"analysis"
OUT=ANALYSIS/"heatmap_decoder"
CHECKPOINT=ROOT/"checkpoints/model-99.pth"
INPUT_RES=256
ALPHA=0.35

def write_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)

def overlap(heat,mask):
    denom=float(np.sum(heat))
    return float(np.sum(heat*mask)/denom) if denom>1e-12 else float("nan")

def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise RuntimeError("GPU policy: launch with CUDA_VISIBLE_DEVICES=0")
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable")
    torch.cuda.init()
    import cv2
    import decord
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision import transforms
    from gradient_saliency import compute_saliency,load_decoder,make_visualization

    device=torch.device("cuda:0")
    decoder=load_decoder(CHECKPOINT,CHECKPOINT.parent,device)
    transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((INPUT_RES,INPUT_RES)),
                                  transforms.Normalize([0.5]*3,[0.5]*3)])
    archive=np.load(EXP/"runs/source_message_map.npz",allow_pickle=True)
    messages={k:np.asarray(archive[k]).reshape(-1).astype(np.float32) for k in archive.files if k!="ckpt"}
    per_frame_ba={}
    with (ANALYSIS/"per_frame_raw_ba.csv").open(encoding="utf-8") as stream:
        for r in csv.DictReader(stream):
            per_frame_ba[(r["case_id"],r["condition"],int(r["frame_idx"]))]=float(r["ba"])
    pair_rows=list(csv.DictReader((EXP/"approval/tcbench_all_36pairs.csv").open(encoding="utf-8")))
    records=[]
    for row in pair_rows:
        case_id,condition=row["case_id"],row["condition"]
        gen_dir=GENERATIONS/case_id/condition
        status=json.loads((gen_dir/"run_status.json").read_text()) if (gen_dir/"run_status.json").is_file() else {}
        video=gen_dir/"video.mp4"
        if status.get("status")!="completed" or not video.is_file(): continue
        if case_id not in messages: raise KeyError(f"No message for {case_id}")
        reader=decord.VideoReader(str(video),ctx=decord.cpu(0)); count=len(reader)
        decode_summary_path=gen_dir/"raw_ba_summary.json"
        copy_like=json.loads(decode_summary_path.read_text()).get("frame0_copy_like",False) if decode_summary_path.is_file() else False
        panel_samples=[1,12,24,36,48] if copy_like else [0,12,24,36,48]
        panel_samples=sorted(set(min(count-1,i) for i in panel_samples))
        panel_samples=set(panel_samples)
        case_out=OUT/case_id/condition
        (case_out/"raw_gradients").mkdir(parents=True,exist_ok=True)
        (case_out/"overlays").mkdir(parents=True,exist_ok=True)
        source_path=EXP/"encoded"/case_id/"watermarked.png"
        source_rgb=np.asarray(Image.open(source_path).convert("RGB"),dtype=np.uint8)
        encoder_map=np.load(EXP/"encoded"/case_id/"encoder_heatmap.npy").astype(np.float32)
        if encoder_map.shape!=source_rgb.shape[:2]:
            raise ValueError(f"Encoder map geometry mismatch for {case_id}: {encoder_map.shape}")
        scale=float(np.quantile(encoder_map,0.99)) or 1.0
        enc_colors=(matplotlib.colormaps["magma"](np.clip(encoder_map/scale,0,1))[...,:3]*255).astype(np.uint8)
        encoder_overlay=np.clip((1-ALPHA)*source_rgb.astype(np.float32)+ALPHA*enc_colors.astype(np.float32),0,255).astype(np.uint8)
        msg=torch.as_tensor(messages[case_id],dtype=torch.float32,device=device).unsqueeze(0)
        fps=float(reader.get_avg_fps())
        if not np.isfinite(fps) or fps<=0: raise ValueError(f"Invalid FPS for {video}: {fps}")
        height,width=reader[0].asnumpy().shape[:2]
        video_path=case_out/"decoder_heatmap_overlay.mp4"
        writer=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
        if not writer.isOpened(): raise RuntimeError(f"Could not open heatmap video writer: {video_path}")
        for frame_idx in range(count):
            frame=reader[frame_idx].asnumpy()
            x=transform(Image.fromarray(frame).convert("RGB")).unsqueeze(0).to(device)
            _,saliency=compute_saliency(decoder,x,msg,target_bit=-1)
            saliency=np.asarray(saliency,dtype=np.float32)
            np.save(case_out/"raw_gradients"/f"frame_{frame_idx:03d}.npy",saliency)
            up=cv2.resize(saliency,(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_LINEAR)
            stability_dir=ANALYSIS/"stable_region"/case_id/condition
            mask_path=stability_dir/"binary_masks"/f"{frame_idx:04d}.png"
            sim_path=stability_dir/"overlays"/f"similarity_{frame_idx:04d}.png"
            stable_path=stability_dir/"overlays"/f"stable_{frame_idx:04d}.png"
            mask=(np.asarray(Image.open(mask_path).convert("L"),dtype=np.uint8)>0).astype(np.float32)
            srr=float(mask.mean())
            enc_overlap=overlap(encoder_map,mask)
            dec_overlap=overlap(up,mask)
            enrichment=dec_overlap/srr if srr>1e-12 and np.isfinite(dec_overlap) else float("nan")
            _,dec_overlay=make_visualization(Image.fromarray(frame),saliency,alpha=ALPHA)
            dec_overlay_rgb=np.asarray(dec_overlay,dtype=np.uint8)
            writer.write(cv2.cvtColor(dec_overlay_rgb,cv2.COLOR_RGB2BGR))
            dec_overlay.save(case_out/"overlays"/f"frame_{frame_idx:03d}_gradient.png")
            ba=per_frame_ba.get((case_id,condition,frame_idx),float("nan"))
            patch_mask=mask.astype(bool)
            green=np.zeros_like(frame);green[...,1]=255
            stable_overlay=frame.copy()
            stable_overlay[patch_mask]=np.clip((1-ALPHA)*frame[patch_mask].astype(np.float32)
                +ALPHA*green[patch_mask].astype(np.float32),0,255).astype(np.uint8)
            similarity=np.load(stability_dir/"similarity_maps"/f"{frame_idx:04d}.npy").astype(np.float32)
            expanded=np.repeat(np.repeat(similarity,16,axis=0),16,axis=1)[:frame.shape[0],:frame.shape[1]]
            sim_colors=(matplotlib.colormaps["RdYlBu"](np.clip(expanded,0,1))[...,:3]*255).astype(np.uint8)
            similarity_overlay=np.clip((1-ALPHA)*frame.astype(np.float32)+ALPHA*sim_colors.astype(np.float32),0,255).astype(np.uint8)
            if frame_idx in panel_samples:
                fig,axes=plt.subplots(2,3,figsize=(16,8))
                panels=[source_rgb,frame,stable_overlay,similarity_overlay,encoder_overlay,dec_overlay_rgb]
                titles=["Watermarked source","Generated frame",f"Stable mask overlay (SRR={srr:.3f})",
                        "Source-anchored similarity overlay","Encoder residual heatmap overlay","Decoder gradient heatmap overlay"]
                for ax,panel,title in zip(axes.ravel(),panels,titles):
                    ax.imshow(panel);ax.set_title(title,fontsize=10);ax.axis("off")
                fig.suptitle(f"{case_id} {condition}, frame {frame_idx}, seed 42, raw BA={ba:.3f}")
                fig.tight_layout();fig.savefig(case_out/"overlays"/f"comparison_{frame_idx:03d}.png",dpi=120);plt.close(fig)
            records.append({"pair_id":row["pair_id"],"case_id":case_id,"official_id":row["official_id"],
                "condition":condition,"seed":42,"frame_idx":frame_idx,"raw_BA":ba,
                "encoder_stable_overlap":enc_overlap,"decoder_stable_overlap":dec_overlap,
                "stable_region_ratio":srr,"decoder_stable_enrichment":enrichment,
                "decoder_objective":"mean ground-truth signed bit score",
                "attribution_method":"existing gradient_saliency.compute_saliency; mean absolute input gradient over RGB",
                "attribution_normalization":"raw gradient saved; percentile 1–99 normalization only for display",
                "display_alpha":ALPHA})
        writer.release()
        print(f"[DECODER HEATMAP] {case_id}/{condition}: rendered all {count} frames to {video_path}",flush=True)
    fields=list(dict.fromkeys(k for row in records for k in row))
    write_csv(ANALYSIS/"heatmap_overlap.csv",records,fields)
    methods={"encoder":"Existing encoder_heatmap.py residual magnitude sqrt(mean((watermarked-clean)^2 over RGB)); raw magnitude maps retained.",
        "decoder":"Existing gradient_saliency.compute_saliency; target is mean ground-truth signed bit score; absolute input gradient averaged over RGB.",
        "decoder_frame_rule":"Every frame of each completed generated video; no temporal interpolation of attribution maps.",
        "display_alpha":ALPHA,"overlap":"sum(raw_map * stable_mask) / sum(raw_map); NA when map sum <= 1e-12",
        "enrichment":"overlap / stable_region_ratio; NA when stable_region_ratio <= 1e-12",
        "raw_maps_saved":True}
    (ANALYSIS/"heatmap_methods.json").write_text(json.dumps(methods,indent=2)+"\n")
    print(f"Computed existing decoder saliency for {len(records)} frames and rendered per-video overlay movies.",flush=True)

if __name__=="__main__": main()
