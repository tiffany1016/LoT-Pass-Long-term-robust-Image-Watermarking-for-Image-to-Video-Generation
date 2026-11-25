#!/usr/bin/env python3
"""Create paired transition/control storyboards for visual transition review."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/"tcbench_background_shifts"
GENERATIONS=EXP/"generations"
OUT=EXP/"analysis/video_review"
SAMPLES=[0,12,24,36,48]
THUMB=(160,107)
BG=(248,248,248)

def main():
    import decord
    OUT.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((EXP/"approval/tcbench_all_36pairs.csv").open(encoding="utf-8")))
    validation_path=EXP/"analysis/transition_validation.csv"
    if not validation_path.exists():
        validation_rows=[]
        for row in rows:
            validation_rows.append({
                "pair_id":row["pair_id"],"case_id":row["case_id"],"official_id":row["official_id"],
                "condition":row["condition"],"prompt_used":row["prompt_used"],
                "transition_validity":"pending_visual_review" if row["condition"]=="transition" else "not_applicable",
                "control_transition_leakage":"pending_visual_review" if row["condition"]=="control" else "not_applicable",
                "observed_change_type":"pending_visual_review","observed_global_change":"pending_visual_review",
                "unintended_camera_drift_or_scene_cuts":"pending_visual_review",
                "review_status":"pending_visual_review","review_notes":"",
            })
        with validation_path.open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(validation_rows[0]));writer.writeheader();writer.writerows(validation_rows)
    cases=[]
    for official_id in sorted({int(r["official_id"]) for r in rows}):
        case_rows=[r for r in rows if int(r["official_id"])==official_id]
        case={"case_id":f"TC{official_id}","conditions":{}}
        for row in case_rows:
            directory=GENERATIONS/row["case_id"]/row["condition"]
            status=json.loads((directory/"run_status.json").read_text()) if (directory/"run_status.json").is_file() else {}
            video=directory/"video.mp4"
            if status.get("status")!="completed" or not video.is_file():
                case["conditions"][row["condition"]]=None
                continue
            reader=decord.VideoReader(str(video),ctx=decord.cpu(0))
            indices=[min(len(reader)-1,x) for x in SAMPLES]
            case["conditions"][row["condition"]]=[(idx,Image.fromarray(reader[idx].asnumpy()).convert("RGB")) for idx in indices]
        cases.append(case)
    font=ImageFont.load_default()
    for sheet_idx,start in enumerate(range(0,len(cases),3),1):
        group=cases[start:start+3]
        label_w,thumb_w,thumb_h=100,THUMB[0],THUMB[1]
        cell_gap, row_gap, header_h=5,20,26
        width=label_w+len(SAMPLES)*(thumb_w+cell_gap)+20
        case_h=header_h+2*(thumb_h+row_gap)+12
        height=20+len(group)*case_h
        sheet=Image.new("RGB",(width,height),BG); draw=ImageDraw.Draw(sheet)
        y=10
        for case in group:
            draw.text((10,y),case["case_id"],fill=(0,0,0),font=font)
            y+=header_h
            for condition in ("transition","control"):
                draw.text((8,y+thumb_h//2-5),condition,fill=(0,0,0),font=font)
                data=case["conditions"].get(condition)
                for j,sample_idx in enumerate(SAMPLES):
                    x=label_w+j*(thumb_w+cell_gap)
                    if data is None:
                        tile=Image.new("RGB",THUMB,(175,175,175))
                        ImageDraw.Draw(tile).text((5,5),"missing",fill=(0,0,0),font=font)
                    else:
                        frame=data[j][1]
                        tile=frame.resize(THUMB,Image.Resampling.LANCZOS)
                    sheet.paste(tile,(x,y))
                    draw.rectangle((x,y,x+thumb_w-1,y+thumb_h-1),outline=(40,40,40))
                    draw.text((x+2,y+thumb_h+2),f"f{sample_idx}",fill=(0,0,0),font=font)
                y+=thumb_h+row_gap
            y+=12
        path=OUT/f"storyboard_{sheet_idx:02d}.png"
        sheet.save(path)
        print(f"Created {path.relative_to(ROOT)}")

if __name__=="__main__": main()
