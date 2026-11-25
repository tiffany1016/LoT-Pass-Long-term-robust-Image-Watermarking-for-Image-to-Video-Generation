#!/usr/bin/env python3
"""Compute source-anchored 16x16 stability and full-video overlays (CPU only)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "claw_lotpass_pilot"
RUNS = EXP / "runs"
OUT = EXP / "analysis/stability"
PATCH, SSIM_WINDOW = 16, 7
MAE_THRESHOLD, SSIM_THRESHOLD = 0.05, 0.90
SIMILARITY_MAE_SCALE, ALPHA = 0.15, 0.35


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def patch_grids(frame_rgb: np.ndarray, source_rgb: np.ndarray):
    """RGB-normalized MAE and uniform-window RGB SSIM for every 16x16 patch."""
    frame = frame_rgb.astype(np.float32) / 255.0
    source = source_rgb.astype(np.float32) / 255.0
    if frame.shape != source.shape or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Source/frame geometry mismatch: {source.shape} vs {frame.shape}")
    h, w = frame.shape[:2]
    ye, xe = list(range(0, h, PATCH)) + [h], list(range(0, w, PATCH)) + [w]
    fp, sp, areas = [], [], []
    for yi in range(len(ye)-1):
        for xi in range(len(xe)-1):
            y0,y1,x0,x1=ye[yi],ye[yi+1],xe[xi],xe[xi+1]
            fp.append(frame[y0:y1,x0:x1]); sp.append(source[y0:y1,x0:x1]); areas.append((y1-y0)*(x1-x0))
    maes = np.asarray([np.mean(np.abs(a-b)) for a,b in zip(fp,sp)], dtype=np.float32)
    ssims = np.empty(len(fp), dtype=np.float32)
    c1, c2 = 0.01**2, 0.03**2
    for start in range(0,len(fp),64):
        a,b=np.stack(fp[start:start+64]),np.stack(sp[start:start+64])
        wa=np.lib.stride_tricks.sliding_window_view(a,(SSIM_WINDOW,SSIM_WINDOW),axis=(1,2))
        wb=np.lib.stride_tricks.sliding_window_view(b,(SSIM_WINDOW,SSIM_WINDOW),axis=(1,2))
        ma,mb=wa.mean(axis=(-1,-2)),wb.mean(axis=(-1,-2))
        va=np.maximum(0,(wa*wa).mean(axis=(-1,-2))-ma*ma)
        vb=np.maximum(0,(wb*wb).mean(axis=(-1,-2))-mb*mb)
        cov=(wa*wb).mean(axis=(-1,-2))-ma*mb
        score=((2*ma*mb+c1)*(2*cov+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2))
        ssims[start:start+len(a)]=score.mean(axis=(1,2,3))
    shape=(len(ye)-1,len(xe)-1)
    areas=np.asarray(areas,dtype=np.float32).reshape(shape)
    mae=maes.reshape(shape); ssim=ssims.reshape(shape)
    stable=(mae<MAE_THRESHOLD)&(ssim>SSIM_THRESHOLD)
    similarity=0.5*np.clip(1-mae/SIMILARITY_MAE_SCALE,0,1)+0.5*np.clip(ssim,0,1)
    from scipy import ndimage
    labels,count=ndimage.label(stable,structure=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8))
    largest=max((float(areas[labels==i].sum()) for i in range(1,count+1)),default=0.0)
    total=float(areas.sum())
    return mae,ssim,areas,stable,similarity,{
        "stable_region_ratio":float(areas[stable].sum()/total),
        "largest_stable_region_ratio":largest/total,
        "mean_similarity":float(np.sum(similarity*areas)/total),
    }


def expand(grid: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.repeat(np.repeat(grid,PATCH,axis=0),PATCH,axis=1)[:h,:w]


def main() -> None:
    import decord
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from PIL import Image

    manifest = json.loads((EXP / "selection/generation_manifest.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5,1.1))
    bar=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap="RdYlBu"),cax=ax,orientation="horizontal")
    bar.set_label("Source-anchored patch similarity (red low → yellow middle → blue high)")
    fig.tight_layout(); fig.savefig(OUT/"similarity_legend.png",dpi=150,bbox_inches="tight"); plt.close(fig)

    ba = {}
    ba_path = EXP / "analysis/raw_ba/per_frame_accuracy.csv"
    if ba_path.is_file():
        for row in csv.DictReader(ba_path.open(encoding="utf-8")):
            ba[(row["case_id"],int(row["frame_idx"]))]=float(row["ba"])
    frame_rows, video_rows = [], []
    for case in manifest["cases"]:
        directory=RUNS/case["case_id"]/f"seed_{case['seed']}"
        status=json.loads((directory/"run_status.json").read_text()) if (directory/"run_status.json").is_file() else {}
        video=directory/"video.mp4"
        common={"case_id":case["case_id"],"image_id":case["image_id"],"sample_index":case["sample_index"],"condition":case["condition"],"seed":case["seed"]}
        if status.get("status")!="completed" or not video.is_file():
            video_rows.append({**common,"status":f"generation_{status.get('status','missing')}"})
            continue
        case_out=OUT/"cases"/case["case_id"]
        for sub in ("mae_maps","ssim_maps","similarity_maps","binary_masks","overlays"):
            (case_out/sub).mkdir(parents=True,exist_ok=True)
        source=np.asarray(Image.open(directory/"effective_input.png").convert("RGB"),dtype=np.uint8)
        reader=decord.VideoReader(str(video),ctx=decord.cpu(0)); fps=float(reader.get_avg_fps())
        sim_path=case_out/"similarity_overlay.mp4"; stable_path=case_out/"stable_mask_overlay.mp4"
        sim_tmp=case_out/".similarity_overlay.partial.mp4"; stable_tmp=case_out/".stable_mask_overlay.partial.mp4"
        sim_writer=imageio.get_writer(str(sim_tmp),fps=fps,codec="libx264",quality=8,macro_block_size=1,
                                      ffmpeg_params=["-preset","veryfast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart"])
        mask_writer=imageio.get_writer(str(stable_tmp),fps=fps,codec="libx264",quality=8,macro_block_size=1,
                                       ffmpeg_params=["-preset","veryfast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart"])
        metrics=[]
        try:
            for idx in range(len(reader)):
                frame=reader[idx].asnumpy()
                mae,ssim,areas,stable,similarity,stats=patch_grids(frame,source)
                np.save(case_out/"mae_maps"/f"{idx:04d}.npy",mae)
                np.save(case_out/"ssim_maps"/f"{idx:04d}.npy",ssim)
                np.save(case_out/"similarity_maps"/f"{idx:04d}.npy",similarity)
                mask=expand(stable.astype(np.uint8),frame.shape[0],frame.shape[1])
                Image.fromarray(mask*255,mode="L").save(case_out/"binary_masks"/f"{idx:04d}.png")
                colors=(matplotlib.colormaps["RdYlBu"](np.clip(similarity,0,1))[...,:3]*255).astype(np.uint8)
                colors=expand(colors,frame.shape[0],frame.shape[1])
                sim_overlay=np.clip((1-ALPHA)*frame.astype(np.float32)+ALPHA*colors.astype(np.float32),0,255).astype(np.uint8)
                green=np.zeros_like(frame); green[...,1]=255
                stable_overlay=frame.copy(); sel=mask.astype(bool)
                stable_overlay[sel]=np.clip((1-ALPHA)*frame[sel].astype(np.float32)+ALPHA*green[sel],0,255).astype(np.uint8)
                sim_writer.append_data(sim_overlay); mask_writer.append_data(stable_overlay)
                if idx in (0,12,24,36,48):
                    Image.fromarray(sim_overlay).save(case_out/"overlays"/f"similarity_{idx:04d}.png")
                    Image.fromarray(stable_overlay).save(case_out/"overlays"/f"stable_{idx:04d}.png")
                entry={**common,"frame_idx":idx,"ba":ba.get((case["case_id"],idx),float("nan")),
                       **stats,"status":"completed","patch_size":PATCH,"mae_threshold":MAE_THRESHOLD,"ssim_threshold":SSIM_THRESHOLD}
                metrics.append(entry); frame_rows.append(entry)
        finally:
            sim_writer.close(); mask_writer.close()
        sim_tmp.replace(sim_path); stable_tmp.replace(stable_path)
        video_rows.append({**common,"status":"completed","frame_count":len(reader),"fps":fps,
                           "mean_stable_region_ratio":float(np.mean([m["stable_region_ratio"] for m in metrics])),
                           "mean_largest_stable_region_ratio":float(np.mean([m["largest_stable_region_ratio"] for m in metrics])),
                           "mean_similarity":float(np.mean([m["mean_similarity"] for m in metrics])),
                           "similarity_overlay":str(sim_path.relative_to(ROOT)),"stable_mask_overlay":str(stable_path.relative_to(ROOT))})
        (case_out/"stability_metadata.json").write_text(json.dumps({
            **common,"reference":"watermarked source image (effective_input.png)","temporal_stability":False,
            "patch_size":PATCH,"ssim_window":SSIM_WINDOW,"mae_threshold":MAE_THRESHOLD,"ssim_threshold":SSIM_THRESHOLD,
            "similarity_mapping":"0.5*clip(1-MAE/0.15,0,1)+0.5*SSIM","overlay_alpha":ALPHA,
            "source_video_frame_count":len(reader),"fps":fps,
        },indent=2)+"\n",encoding="utf-8")
        print(f"Stability complete for {case['case_id']}: {len(metrics)} frames",flush=True)

    fields=["case_id","image_id","sample_index","condition","seed","frame_idx","ba","stable_region_ratio",
            "largest_stable_region_ratio","mean_similarity","status","patch_size","mae_threshold","ssim_threshold"]
    write_csv(OUT/"per_frame_stability.csv",frame_rows,fields)
    write_csv(OUT/"per_video_stability.csv",video_rows,list(dict.fromkeys(k for row in video_rows for k in row)))
    valid=[row for row in frame_rows if np.isfinite(row["ba"])]
    for metric,filename in (("stable_region_ratio","ba_vs_stable_region_ratio.png"),
                            ("largest_stable_region_ratio","ba_vs_largest_stable_region_ratio.png")):
        fig,ax=plt.subplots(figsize=(6,5))
        for condition,color in (("static_control","#315b7d"),("broad_fluid_motion","#d77732")):
            vals=[r for r in valid if r["condition"]==condition]
            ax.scatter([r[metric] for r in vals],[r["ba"] for r in vals],s=14,alpha=.45,label=condition,color=color)
        ax.set(xlabel=metric.replace("_"," "),ylabel="Raw bit accuracy",title=f"Raw BA vs {metric.replace('_',' ')}")
        ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(OUT/filename,dpi=150);plt.close(fig)
    print(f"Computed source-anchored stability for {len(video_rows)} cases / {len(frame_rows)} frames.",flush=True)


if __name__ == "__main__":
    main()
