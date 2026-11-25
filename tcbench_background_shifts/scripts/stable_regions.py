#!/usr/bin/env python3
"""Compute source-anchored 16x16 appearance stability and overlays (CPU only)."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "tcbench_background_shifts"
GENERATIONS = EXP / "generations"
ANALYSIS = EXP / "analysis"
OUT = ANALYSIS / "stable_region"
PATCH, SSIM_WINDOW = 16, 7
MAE_THRESHOLD, SSIM_THRESHOLD = 0.05, 0.90
SIMILARITY_MAE_SCALE, ALPHA, FPS = 0.15, 0.35, 8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def patch_grids(frame_rgb: np.ndarray, source_rgb: np.ndarray):
    """RGB normalized MAE and mean valid-window 7x7 SSIM per 16x16 patch."""
    frame = frame_rgb.astype(np.float32) / 255.0
    source = source_rgb.astype(np.float32) / 255.0
    if frame.shape != source.shape or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Source/frame geometry mismatch: {source.shape} vs {frame.shape}")
    h, w = frame.shape[:2]
    y_edges, x_edges = list(range(0, h, PATCH)) + [h], list(range(0, w, PATCH)) + [w]
    fpatches, spatches, sizes = [], [], []
    for yi in range(len(y_edges)-1):
        for xi in range(len(x_edges)-1):
            y0,y1,x0,x1=y_edges[yi],y_edges[yi+1],x_edges[xi],x_edges[xi+1]
            fpatches.append(frame[y0:y1,x0:x1]); spatches.append(source[y0:y1,x0:x1])
            sizes.append((y1-y0)*(x1-x0))
    maes=np.asarray([np.mean(np.abs(a-b)) for a,b in zip(fpatches,spatches)],dtype=np.float32)
    ssims=np.empty(len(fpatches),dtype=np.float32)
    c1,c2=(0.01**2),(0.03**2)
    for start in range(0,len(fpatches),64):
        a=np.stack(fpatches[start:start+64]); b=np.stack(spatches[start:start+64])
        if a.shape[1]<SSIM_WINDOW or a.shape[2]<SSIM_WINDOW:
            raise ValueError(f"Patch too small for SSIM: {a.shape[1:3]}")
        wa=np.lib.stride_tricks.sliding_window_view(a,(SSIM_WINDOW,SSIM_WINDOW),axis=(1,2))
        wb=np.lib.stride_tricks.sliding_window_view(b,(SSIM_WINDOW,SSIM_WINDOW),axis=(1,2))
        ma,mb=wa.mean(axis=(-1,-2)),wb.mean(axis=(-1,-2))
        va=np.maximum(0,(wa*wa).mean(axis=(-1,-2))-ma*ma)
        vb=np.maximum(0,(wb*wb).mean(axis=(-1,-2))-mb*mb)
        cov=(wa*wb).mean(axis=(-1,-2))-ma*mb
        score=((2*ma*mb+c1)*(2*cov+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2))
        ssims[start:start+len(a)]=score.mean(axis=(1,2,3))
    shape=(len(y_edges)-1,len(x_edges)-1)
    areas=np.asarray(sizes,dtype=np.float32).reshape(shape)
    mae=maes.reshape(shape); ssim=ssims.reshape(shape)
    stable=(mae<MAE_THRESHOLD)&(ssim>SSIM_THRESHOLD)
    similarity=0.5*np.clip(1.0-mae/SIMILARITY_MAE_SCALE,0,1)+0.5*np.clip(ssim,0,1)
    from scipy import ndimage
    labels,count=ndimage.label(stable,structure=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8))
    largest=max((float(areas[labels==label].sum()) for label in range(1,count+1)),default=0.0)
    total=float(areas.sum())
    return mae,ssim,areas,stable,similarity,{
        "stable_region_ratio":float(areas[stable].sum()/total),
        "largest_stable_region_ratio":largest/total,
        "mean_similarity":float(np.sum(similarity*areas)/total),
        "stable_component_connectivity":4,
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

    ANALYSIS.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(6.5,1.1))
    cbar=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap="RdYlBu"),cax=ax,orientation="horizontal")
    cbar.set_label("Source-anchored patch similarity (red low → yellow → blue high)")
    fig.tight_layout(); fig.savefig(ANALYSIS/"similarity_legend.png",dpi=150,bbox_inches="tight"); plt.close(fig)

    ba={}
    ba_path=ANALYSIS/"per_frame_raw_ba.csv"
    if ba_path.is_file():
        for row in csv.DictReader(ba_path.open(encoding="utf-8")):
            try: ba[(row["case_id"],row["condition"],int(row["frame_idx"]))]=float(row["ba"])
            except (ValueError,KeyError): pass
    pair_rows=list(csv.DictReader((EXP/"approval/tcbench_all_36pairs.csv").open(encoding="utf-8")))
    records=[]
    for row in pair_rows:
        case_id,condition=row["case_id"],row["condition"]
        gen_dir=GENERATIONS/case_id/condition
        status=json.loads((gen_dir/"run_status.json").read_text()) if (gen_dir/"run_status.json").is_file() else {}
        video=gen_dir/"video.mp4"
        common={"pair_id":row["pair_id"],"case_id":case_id,"official_id":row["official_id"],
                "condition":condition,"seed":42}
        if status.get("status")!="completed" or not video.is_file():
            records.append({**common,"status":"generation_"+str(status.get("status","missing"))})
            continue
        case_out=OUT/case_id/condition
        for sub in ("mae_maps","ssim_maps","similarity_maps","binary_masks","overlays"):
            (case_out/sub).mkdir(parents=True,exist_ok=True)
        source=np.asarray(Image.open(EXP/"encoded"/case_id/"watermarked.png").convert("RGB"),dtype=np.uint8)
        reader=decord.VideoReader(str(video),ctx=decord.cpu(0))
        sim_tmp=case_out/".similarity_overlay.partial.mp4"
        mask_tmp=case_out/".stable_mask_overlay.partial.mp4"
        sim_writer=imageio.get_writer(str(sim_tmp),fps=FPS,codec="libx264",quality=8,macro_block_size=1)
        mask_writer=imageio.get_writer(str(mask_tmp),fps=FPS,codec="libx264",quality=8,macro_block_size=1)
        case_rows=[]
        try:
            for idx in range(len(reader)):
                frame=reader[idx].asnumpy()
                mae,ssim,areas,stable,similarity,metrics=patch_grids(frame,source)
                np.save(case_out/"mae_maps"/f"{idx:04d}.npy",mae)
                np.save(case_out/"ssim_maps"/f"{idx:04d}.npy",ssim)
                np.save(case_out/"similarity_maps"/f"{idx:04d}.npy",similarity)
                stable_px=expand(stable.astype(np.uint8),frame.shape[0],frame.shape[1])
                Image.fromarray(stable_px*255,mode="L").save(case_out/"binary_masks"/f"{idx:04d}.png")
                cmap=matplotlib.colormaps["RdYlBu"]
                colors=(cmap(np.clip(expand(similarity,frame.shape[0],frame.shape[1]),0,1))[...,:3]*255).astype(np.uint8)
                sim_overlay=np.clip((1-ALPHA)*frame.astype(np.float32)+ALPHA*colors.astype(np.float32),0,255).astype(np.uint8)
                stable_overlay=frame.copy(); green=np.zeros_like(frame); green[...,1]=255
                mask=stable_px.astype(bool)
                stable_overlay[mask]=np.clip((1-ALPHA)*frame[mask].astype(np.float32)+ALPHA*green[mask].astype(np.float32),0,255).astype(np.uint8)
                sim_writer.append_data(sim_overlay); mask_writer.append_data(stable_overlay)
                if idx in (0,24,48):
                    Image.fromarray(sim_overlay).save(case_out/"overlays"/f"similarity_{idx:04d}.png")
                    Image.fromarray(stable_overlay).save(case_out/"overlays"/f"stable_{idx:04d}.png")
                item={**common,"frame_idx":idx,"status":"completed","raw_BA":ba.get((case_id,condition,idx),float("nan")),
                      **metrics,"mae_threshold":MAE_THRESHOLD,"ssim_threshold":SSIM_THRESHOLD}
                case_rows.append(item); records.append(item)
            sim_writer.close(); mask_writer.close()
            os.replace(sim_tmp,case_out/"similarity_overlay.mp4")
            os.replace(mask_tmp,case_out/"stable_mask_overlay.mp4")
        except Exception:
            sim_writer.close(); mask_writer.close()
            sim_tmp.unlink(missing_ok=True); mask_tmp.unlink(missing_ok=True)
            raise
        atomic_json(case_out/"stability_metadata.json",{
            "case_id":case_id,"condition":condition,"seed":42,
            "source_image":str((EXP/"encoded"/case_id/"watermarked.png").relative_to(ROOT)),
            "source_image_sha256":sha256(EXP/"encoded"/case_id/"watermarked.png"),
            "comparison":"same-coordinate RGB against watermarked source; no registration or optical flow",
            "patch_size":PATCH,"ssim_window":SSIM_WINDOW,"ssim_window_type":"uniform valid windows, averaged per patch and over RGB",
            "mae_threshold":MAE_THRESHOLD,"ssim_threshold":SSIM_THRESHOLD,
            "similarity_mapping":"0.5*clip(1-MAE/0.15,0,1)+0.5*clip(SSIM,0,1)",
            "overlay_alpha":ALPHA,"overlay_colormap":"RdYlBu (red low, yellow middle, blue high)",
            "stable_mask_connectivity":4,"frame_count":len(reader),"video_sha256":sha256(video),
        })
        print(f"[STABILITY_DONE] {row['pair_id']}",flush=True)
    fields=list(dict.fromkeys(k for item in records for k in item))
    write_csv(ANALYSIS/"stable_region_metrics.csv",records,fields)
    print(f"Source-anchored stability complete for {sum(r.get('status')=='completed' for r in records)} frames.",flush=True)


if __name__=="__main__": main()
