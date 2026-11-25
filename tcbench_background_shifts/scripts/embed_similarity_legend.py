#!/usr/bin/env python3
"""Embed a compact low-to-high similarity legend in each overlay video."""
from __future__ import annotations
from pathlib import Path
import os
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/"tcbench_background_shifts"
OUT=EXP/"analysis/stable_region"
ALPHA=0.35

def main():
    import decord
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    from PIL import Image,ImageDraw,ImageFont
    videos=sorted(OUT.glob("TC*/transition/similarity_overlay.mp4"))+sorted(OUT.glob("TC*/control/similarity_overlay.mp4"))
    # An explicit legend panel: red/yellow = low similarity; blue = high.
    width,height=242,40
    panel=Image.new("RGB",(width,height),(255,255,255))
    draw=ImageDraw.Draw(panel);font=ImageFont.load_default()
    draw.text((4,2),"Low similarity",fill=(0,0,0),font=font)
    draw.text((width-76,2),"High similarity",fill=(0,0,0),font=font)
    gradient=np.linspace(0,1,220,dtype=np.float32)[None,:]
    colors=(matplotlib.colormaps["RdYlBu"](gradient)[0,:,:3]*255).astype(np.uint8)
    band=np.repeat(colors[None,:,:],12,axis=0)
    panel.paste(Image.fromarray(band),(4,17))
    (EXP/"analysis/similarity_legend.png").parent.mkdir(parents=True,exist_ok=True)
    panel.save(EXP/"analysis/similarity_legend.png")
    for video in videos:
        reader=decord.VideoReader(str(video),ctx=decord.cpu(0))
        tmp=video.with_name(".similarity_legend.partial.mp4")
        writer=imageio.get_writer(str(tmp),fps=8,codec="libx264",quality=8,macro_block_size=1)
        try:
            for idx in range(len(reader)):
                frame=Image.fromarray(reader[idx].asnumpy()).convert("RGB")
                # Small opaque key; it does not participate in any metric.
                frame.paste(panel,(8,8))
                writer.append_data(np.asarray(frame))
            writer.close();os.replace(tmp,video)
        except Exception:
            writer.close();tmp.unlink(missing_ok=True);raise
        print(f"Embedded legend: {video.relative_to(ROOT)}",flush=True)

if __name__=="__main__":main()
