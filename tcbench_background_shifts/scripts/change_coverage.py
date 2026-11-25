#!/usr/bin/env python3
"""Measure source-anchored whole-frame and patch luminance differences."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/"tcbench_background_shifts"
ANALYSIS=EXP/"analysis"
PATCH=16
LUMA_WEIGHTS=np.asarray([0.2126,0.7152,0.0722],dtype=np.float32)

def write_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)

def main():
    import decord
    from PIL import Image
    rows=list(csv.DictReader((EXP/"approval/tcbench_all_36pairs.csv").open(encoding="utf-8")))
    per_frame=[];per_video=[]
    for row in rows:
        cid,cond=row["case_id"],row["condition"]
        directory=EXP/"generations"/cid/cond
        status=json.loads((directory/"run_status.json").read_text()) if (directory/"run_status.json").is_file() else {}
        video=directory/"video.mp4"
        common={"pair_id":row["pair_id"],"case_id":cid,"official_id":row["official_id"],"condition":cond}
        if status.get("status")!="completed" or not video.is_file():
            per_video.append({**common,"status":"generation_"+str(status.get("status","missing"))});continue
        source=np.asarray(Image.open(EXP/"encoded"/cid/"watermarked.png").convert("RGB"),dtype=np.float32)/255.0
        src_luma=np.einsum("hwc,c->hw",source,LUMA_WEIGHTS)
        reader=decord.VideoReader(str(video),ctx=decord.cpu(0))
        clip=[]
        for idx in range(len(reader)):
            frame=reader[idx].asnumpy().astype(np.float32)/255.0
            if frame.shape!=source.shape: raise ValueError(f"Dimension mismatch: {cid}/{cond}/{idx}")
            luma=np.einsum("hwc,c->hw",frame,LUMA_WEIGHTS)
            delta=luma-src_luma
            patch_vals=[];patch_areas=[]
            for y in range(0,luma.shape[0],PATCH):
                for x in range(0,luma.shape[1],PATCH):
                    a=delta[y:y+PATCH,x:x+PATCH]
                    patch_vals.append(float(a.mean()));patch_areas.append(a.size)
            patch_vals=np.asarray(patch_vals);patch_areas=np.asarray(patch_areas,dtype=np.float64)
            changed=np.abs(patch_vals)>0.05
            result={**common,"frame_idx":idx,"status":"completed",
                "source_mean_luminance":float(src_luma.mean()),"frame_mean_luminance":float(luma.mean()),
                "whole_frame_mean_luminance_delta":float(delta.mean()),
                "whole_frame_mean_abs_luminance_delta":float(np.abs(delta).mean()),
                "patches_abs_mean_luminance_delta_gt_0_05_ratio":float(patch_areas[changed].sum()/patch_areas.sum()),
                "p95_abs_patch_mean_luminance_delta":float(np.quantile(np.abs(patch_vals),0.95)),
                "patch_size":PATCH,"patch_luma_change_threshold":0.05}
            per_frame.append(result);clip.append(result)
        per_video.append({**common,"status":"completed","frame_count":len(clip),
            "mean_whole_frame_abs_luminance_delta":float(np.mean([r["whole_frame_mean_abs_luminance_delta"] for r in clip])),
            "max_whole_frame_abs_luminance_delta":float(np.max([r["whole_frame_mean_abs_luminance_delta"] for r in clip])),
            "mean_fraction_patches_luma_delta_gt_0_05":float(np.mean([r["patches_abs_mean_luminance_delta_gt_0_05_ratio"] for r in clip])),
            "mean_p95_abs_patch_mean_luminance_delta":float(np.mean([r["p95_abs_patch_mean_luminance_delta"] for r in clip]))})
    write_csv(ANALYSIS/"per_frame_change_coverage.csv",per_frame,list(per_frame[0]) if per_frame else ["pair_id","case_id","condition","status"])
    write_csv(ANALYSIS/"per_video_change_coverage.csv",per_video,list(per_video[0]) if per_video else ["pair_id","case_id","condition","status"])
    print(f"Measured luminance change coverage for {sum(r.get('status')=='completed' for r in per_video)}/36 videos.",flush=True)
if __name__=="__main__":main()
