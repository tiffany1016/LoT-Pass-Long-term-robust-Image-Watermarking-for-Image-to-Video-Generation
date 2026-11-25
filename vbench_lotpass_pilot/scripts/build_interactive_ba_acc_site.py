#!/usr/bin/env python3
"""Build a static, local browser dashboard for VBench++ per-source BA curves."""
from __future__ import annotations

import csv
import base64
import json
import math
import os
import subprocess
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "vbench_lotpass_pilot"
OUT = EXP / "analysis/interactive_ba_acc.html"
OUT_OUTER = ROOT / "vbench_lotpass_pilot_dashboard.html"
PREVIEW_DIR = EXP / "analysis/embedded_video_previews"
PREVIEW_WIDTH = 480
PREVIEW_HEIGHT = 320
PREVIEW_CRF = "28"


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def embedded_video_url(case_id: str, source_path: Path) -> str:
    """Return a browser-compatible embedded preview; leave original videos untouched."""
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    preview_path = PREVIEW_DIR / f"{case_id}.mp4"
    if not preview_path.is_file() or preview_path.stat().st_mtime < source_path.stat().st_mtime:
        partial_path = preview_path.with_name(f".{preview_path.stem}.partial.mp4")
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source_path), "-vf",
            f"fps=8,scale={PREVIEW_WIDTH}:{PREVIEW_HEIGHT}:flags=lanczos",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", PREVIEW_CRF,
            "-profile:v", "baseline", "-level", "3.0", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(partial_path),
        ]
        subprocess.run(command, check=True, cwd=ROOT)
        os.replace(partial_path, preview_path)
    payload = base64.b64encode(preview_path.read_bytes()).decode("ascii")
    return "data:video/mp4;base64," + payload


def main() -> None:
    manifest = json.loads((EXP / "selection/selection_manifest.json").read_text(encoding="utf-8"))
    case_summaries = load_csv(EXP / "analysis/raw_ba/case_summary.csv")
    summary_by_case = {row["case_id"]: row for row in case_summaries}
    frame_rows = load_csv(EXP / "analysis/raw_ba/per_frame_accuracy.csv")
    values_by_case: dict[str, list[float | None]] = {}
    for row in frame_rows:
        try:
            value = float(row["ba"])
            value = value if math.isfinite(value) else None
        except (TypeError, ValueError):
            value = None
        values_by_case.setdefault(row["case_id"], []).append(value)
    aligned_values_by_case: dict[str, list[float | None]] = {}
    aligned_rows_path = EXP / "analysis/optical_flow/per_frame_aligned_ba.csv"
    if aligned_rows_path.is_file():
        for row in load_csv(aligned_rows_path):
            try:
                value = float(row["aligned_BA"])
                value = value if math.isfinite(value) else None
            except (TypeError, ValueError):
                value = None
            aligned_values_by_case.setdefault(row["case_id"], []).append(value)
    motion_rows = load_csv(EXP / "selection/vbench_40pairs.csv")
    motions_by_image: dict[str, list[dict]] = {}
    for row in motion_rows:
        summary = summary_by_case.get(row["case_id"], {})
        video_path = EXP / "runs" / row["image_id"] / row["motion_group"] / "seed_42" / "video.mp4"
        aligned_video_path = EXP / "analysis/optical_flow" / row["case_id"] / "aligned_video.mp4"
        heatmap_video_path = EXP / "analysis/optical_flow" / row["case_id"] / "decoder_heatmap_overlay.mp4"
        motions_by_image.setdefault(row["image_id"], []).append({
            "case_id": row["case_id"],
            "motion": row["motion_group"],
            "camera_setting": row["motion_full"],
            "raw_video_url": embedded_video_url(f"{row['case_id']}_raw", video_path) if video_path.is_file() else "",
            "aligned_video_url": embedded_video_url(f"{row['case_id']}_aligned", aligned_video_path) if aligned_video_path.is_file() else "",
            "heatmap_video_url": embedded_video_url(f"{row['case_id']}_aligned_heatmap", heatmap_video_path) if heatmap_video_path.is_file() else "",
            "prompt": row["prompt"],
            "status": summary.get("status", "missing"),
            "mean_ba": summary.get("mean_ba_excluding_copy_frame", ""),
            "last_third_ba": summary.get("last_third_mean_ba_excluding_copy_frame", ""),
            "minimum_ba": summary.get("minimum_ba_all_frames", ""),
            "values": values_by_case.get(row["case_id"], []),
            "aligned_values": aligned_values_by_case.get(row["case_id"], []),
        })

    aggregate_rows = []
    for motion in ("static", "pan", "tilt", "zoom"):
        complete = [row for row in motion_rows
                    if row["motion_group"] == motion
                    and summary_by_case.get(row["case_id"], {}).get("status") == "completed"]
        video_values = [values_by_case.get(row["case_id"], []) for row in complete]
        video_values = [[value for value in values if value is not None] for values in video_values]
        aligned_video_values = [aligned_values_by_case.get(row["case_id"], []) for row in complete]
        aligned_video_values = [[value for value in values if value is not None] for values in aligned_video_values]
        frame_count = max((len(values) for values in video_values), default=0)
        frame_means = [
            sum(values[i] for values in video_values if i < len(values))
            / sum(i < len(values) for values in video_values)
            if any(i < len(values) for values in video_values) else None
            for i in range(frame_count)
        ]
        aligned_frame_count = max((len(values) for values in aligned_video_values), default=0)
        aligned_frame_means = [
            sum(values[i] for values in aligned_video_values if i < len(values))
            / sum(i < len(values) for values in aligned_video_values)
            if any(i < len(values) for values in aligned_video_values) else None
            for i in range(aligned_frame_count)
        ]
        pooled_values = [value for values in video_values for value in values]
        pooled_aligned_values = [value for values in aligned_video_values for value in values]
        video_means = [sum(values) / len(values) for values in video_values if values]
        aligned_video_means = [sum(values) / len(values) for values in aligned_video_values if values]
        aggregate_rows.append({
            "motion": motion,
            "video_count": len(video_means),
            "frame_count_per_video": frame_count,
            "total_video_frames": len(pooled_values),
            "mean_acc_all_frames": sum(pooled_values) / len(pooled_values) if pooled_values else None,
            "mean_aligned_acc_all_frames": sum(pooled_aligned_values) / len(pooled_aligned_values) if pooled_aligned_values else None,
            "aligned_minus_raw_acc": (sum(pooled_aligned_values) / len(pooled_aligned_values) - sum(pooled_values) / len(pooled_values)) if pooled_values and pooled_aligned_values else None,
            "video_mean_sd": (sum((value - sum(video_means) / len(video_means)) ** 2
                                   for value in video_means) / len(video_means)) ** 0.5 if video_means else None,
            "frame_means": frame_means,
            "aligned_frame_means": aligned_frame_means,
            "aligned_video_mean_sd": (sum((value - sum(aligned_video_means) / len(aligned_video_means)) ** 2
                                            for value in aligned_video_means) / len(aligned_video_means)) ** 0.5 if aligned_video_means else None,
        })

    samples = []
    for item in manifest["selected_images"]:
        image_id = item["image_id"]
        source_image = EXP / "images/canonical_clean" / f"{image_id}.png"
        samples.append({
            "id": image_id,
            "selected_index": item["selected_index"],
            "name": item["image_name"],
            "category": item["image_type"],
            "source_url": item.get("source_url", ""),
            "image_data": "data:image/png;base64," + base64.b64encode(source_image.read_bytes()).decode("ascii"),
            "motions": sorted(motions_by_image.get(image_id, []),
                               key=lambda row: ("static", "pan", "tilt", "zoom").index(row["motion"])),
        })

    data = {
        "title": "VBench++ · LoT-Pass per-source bit accuracy",
        "seed": 42,
        "frames": 49,
        "fps": 8,
        "samples": samples,
        "all_video_averages": aggregate_rows,
    }
    html_template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VBench++ BA–ACC Explorer</title>
<style>
:root {
  color-scheme: light;
  --ink: #162a3a; --muted: #637585; --line: #dce5ec;
  --paper: #f4f7fa; --card: #fff; --blue: #197b9b; --navy: #173c59;
  font: 14px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); background: var(--paper); }
header { padding: 30px clamp(18px, 4vw, 54px) 24px; color: #f7fbff; background: linear-gradient(120deg, #13334d, #1c6880); }
h1 { margin: 0; font-size: clamp(1.45rem, 2.7vw, 2.15rem); letter-spacing: -.025em; }
.subtitle { margin: 8px 0 18px; color: #d7e8f1; max-width: 850px; }
.stats { display: flex; flex-wrap: wrap; gap: 8px; }
.stat { border: 1px solid #ffffff3a; border-radius: 999px; padding: 5px 11px; background: #ffffff12; font-size: .84rem; }
.page-nav { display: flex; gap: 8px; max-width: 1600px; margin: 0 auto; padding: 18px clamp(14px, 3vw, 40px) 0; }
.page-nav button[aria-pressed="true"] { background: #173c59; border-color: #173c59; color: #fff; }
.average-page { max-width: 1600px; margin: 0 auto; padding: 18px clamp(14px, 3vw, 40px) 54px; }
.average-intro { margin: 0 0 14px; color: var(--muted); }
.average-table-wrap { overflow-x: auto; margin: 0 0 18px; border: 1px solid var(--line); border-radius: 10px; background: #fff; }
.average-table { width: 100%; border-collapse: collapse; min-width: 620px; }
.average-table th, .average-table td { padding: 11px 14px; text-align: left; border-bottom: 1px solid var(--line); }
.average-table th { background: #f6f9fb; color: #395468; font-size: .82rem; }
.average-table tr:last-child td { border-bottom: 0; }
.motion-swatch { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 7px; vertical-align: -1px; }
.average-chart-card { border: 1px solid var(--line); border-radius: 12px; background: #fff; padding: 14px 18px; }
.average-chart-card h2 { margin: 0 0 5px; font-size: 1.05rem; }
.average-chart { display: block; width: 100%; height: auto; }
.average-note { margin: 8px 0 0; color: var(--muted); font-size: .8rem; }
.is-hidden { display: none !important; }
.layout { display: grid; grid-template-columns: 280px minmax(0, 1fr); gap: 22px; max-width: 1600px; margin: 0 auto; padding: 22px clamp(14px, 3vw, 40px) 54px; align-items: start; }
aside { position: sticky; top: 14px; border: 1px solid var(--line); border-radius: 12px; background: var(--card); padding: 15px; box-shadow: 0 5px 20px #1638520a; }
aside h2 { margin: 0 0 4px; font-size: 1.05rem; }
.hint { margin: 0 0 13px; color: var(--muted); font-size: .83rem; }
button { border: 1px solid #c7d4df; border-radius: 7px; padding: 8px 9px; background: #f9fbfd; color: var(--ink); font: inherit; cursor: pointer; }
button:hover { border-color: #77aabd; background: #eff8fb; }
.search { width: 100%; margin: 0 0 9px; border: 1px solid #c7d4df; border-radius: 7px; padding: 9px 10px; font: inherit; }
.sample-list { display: grid; gap: 4px; max-height: min(66vh, 700px); overflow: auto; }
.sample-option { display: block; width: 100%; border: 1px solid transparent; border-radius: 7px; padding: 8px 9px; background: transparent; text-align: left; }
.sample-option:hover { border-color: #d7e5eb; background: #f1f7fa; }
.sample-option[aria-pressed="true"] { border-color: #9fc8d5; background: #eaf5f8; }
.sample-option strong { display: block; font-size: .85rem; }
.sample-option small { display: block; color: var(--muted); overflow-wrap: anywhere; font-size: .74rem; }
main { min-width: 0; }
.empty { border: 1px dashed #b8c8d4; border-radius: 12px; background: #fff; padding: 38px; color: var(--muted); text-align: center; }
.sample-card { margin: 0 0 20px; border: 1px solid var(--line); border-radius: 13px; background: var(--card); overflow: hidden; box-shadow: 0 5px 20px #1638520a; }
.card-head { display: grid; grid-template-columns: 112px minmax(0,1fr) minmax(260px, .85fr); gap: 16px; padding: 16px 18px; border-bottom: 1px solid var(--line); background: linear-gradient(120deg, #fff, #f8fbfd); align-items: start; }
.sample-image { display: block; width: 112px; aspect-ratio: 3/2; object-fit: cover; border: 1px solid var(--line); border-radius: 8px; background: #edf2f5; }
.sample-meta { min-width: 0; }
.sample-title { margin: 0; font-size: 1.04rem; }
.sample-name { margin: 4px 0 8px; color: #42596b; }
.badges { display: flex; flex-wrap: wrap; gap: 6px; }
.badge { border-radius: 999px; background: #eaf3f7; color: #22556b; padding: 3px 9px; font-size: .74rem; }
.source-link { color: var(--blue); text-decoration: none; font-size: .78rem; }
.source-link:hover { text-decoration: underline; }
.video-panel { min-width: 0; border: 1px solid var(--line); border-radius: 9px; padding: 10px; background: #fff; }
.video-panel-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 8px; }
.video-panel-title { margin: 0; color: #395468; font-size: .83rem; font-weight: 650; }
.video-select { min-width: 0; max-width: 58%; border: 1px solid #c7d4df; border-radius: 6px; padding: 6px 8px; background: #fff; color: var(--ink); font: inherit; font-size: .8rem; }
.sample-video { display: block; width: 100%; max-height: 230px; aspect-ratio: 16/9; object-fit: contain; border-radius: 6px; background: #101820; }
.video-actions { display: flex; justify-content: flex-end; margin-top: 7px; }
.video-actions button { padding: 5px 8px; font-size: .78rem; }
.graph-wrap { padding: 13px 18px 15px; }
.graph-title { margin: 0 0 7px; color: #395468; font-size: .83rem; font-weight: 650; }
.curve-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; margin: 0 18px 2px; padding: 11px 13px; border: 1px solid var(--line); border-radius: 9px; background: #f8fbfd; }
.curve-controls strong { color: #395468; font-size: .82rem; }
.curve-actions { display: flex; gap: 6px; }
.curve-actions button { padding: 5px 8px; font-size: .78rem; }
.curve-option { display: inline-flex; align-items: center; gap: 5px; font-size: .8rem; cursor: pointer; }
.curve-option input { accent-color: var(--blue); }
.chart { display: block; width: 100%; height: auto; overflow: visible; border-radius: 8px; background: #fff; }
.gridline { stroke: #e3e9ee; stroke-width: 1; }
.axis { stroke: #566b7b; stroke-width: 1.2; }
.axis-label { fill: #31495b; font-size: 13px; }
.tick-label { fill: #607484; font-size: 11px; }
.curve { fill: none; stroke-width: 2.6; stroke-linejoin: round; stroke-linecap: round; }
.curve-point { stroke: #fff; stroke-width: 1; cursor: crosshair; }
.legend { fill: #354e60; font-size: 12px; }
.status { color: #22724d; font-weight: 650; }
footer { max-width: 1500px; margin: -34px auto 0; padding: 0 28px 26px; color: var(--muted); font-size: .78rem; }
@media(max-width: 900px) { .layout { grid-template-columns: 1fr; } aside { position: static; } .sample-list { max-height: 240px; grid-template-columns: repeat(auto-fit,minmax(190px,1fr)); } }
@media(max-width: 1050px) { .card-head { grid-template-columns: 92px minmax(0,1fr); } .video-panel { grid-column: 1 / -1; } }
@media(max-width: 520px) { .card-head { grid-template-columns: 82px minmax(0,1fr); gap: 11px; padding: 12px; } .sample-image { width: 82px; } .graph-wrap { padding: 10px 9px 12px; } }
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <p class="subtitle">Compare raw and LoT-Pass RAFT-aligned ACC across all VBench++ videos, or inspect an individual source. The video selector switches between raw, aligned, and aligned-with-heatmap previews.</p>
  <div class="stats" id="stats"></div>
</header>
<nav class="page-nav" aria-label="Dashboard pages">
  <button id="average-tab" type="button" aria-pressed="true">Raw and aligned averages</button>
  <button id="explorer-tab" type="button" aria-pressed="false">Per-source explorer</button>
</nav>
<section class="average-page" id="average-page">
  <p class="average-intro">Solid curves and raw mean ACC show the original videos; dashed curves and aligned mean ACC show LoT-Pass RAFT alignment to each generated clip’s frame 0. The table averages all 49 frames from each completed video.</p>
  <div class="average-table-wrap">
    <table class="average-table">
      <thead><tr><th>Camera setting</th><th>Raw mean ACC</th><th>RAFT-aligned mean ACC</th><th>Aligned − raw</th><th>Videos</th><th>Frames</th><th>Raw between-video SD¹</th></tr></thead>
      <tbody id="average-table-body"></tbody>
    </table>
  </div>
  <section class="average-chart-card">
    <h2>Average raw and RAFT-aligned ACC by frame</h2>
    <svg class="average-chart" id="average-chart" viewBox="0 0 960 440" role="img" aria-label="Average per-frame ACC across all VBench++ videos by camera setting"></svg>
    <p class="average-note">¹ SD is calculated across each video’s 49-frame mean; it describes variation among the 10 videos per setting. Frames within videos are repeated measurements, not independent samples.</p>
  </section>
</section>
<div class="layout is-hidden" id="explorer-page">
  <aside aria-label="Sample selection">
    <h2>Source samples</h2>
  <p class="hint">Choose one source to compare raw and RAFT-aligned curves and videos.</p>
    <input class="search" id="search" type="search" placeholder="Filter by ID, name, or category" aria-label="Filter source samples">
    <div class="sample-list" id="sample-list"></div>
  </aside>
  <main id="results" aria-live="polite"></main>
</div>
<footer>ACC/BA is the fraction of watermark bits decoded correctly per frame. Solid curves are raw; dashed curves are LoT-Pass RAFT-aligned. Hover a point for its frame index and value. Alignment uses each generated video’s frame 0 as reference; frames in a clip are repeated measurements, not independent samples.</footer>
<script>
const DATA = __DATA__;
let selectedSampleId = DATA.samples[0]?.id || null;
const visibleCurves = new Map(DATA.samples.map(sample => [sample.id, new Set(sample.motions.flatMap(motion => [`${motion.motion}:raw`, ...(motion.aligned_values?.length ? [`${motion.motion}:aligned`] : [])]))]));
const title = document.getElementById('title');
title.textContent = DATA.title;
const stats = document.getElementById('stats');
[
  `${DATA.samples.length} source images`,
  `${DATA.samples.reduce((n, s) => n + s.motions.length, 0)} generated clips`,
  `Raw and RAFT-aligned BA`,
  `${DATA.frames} frames per clip`,
  `${DATA.fps} FPS`,
  `Seed ${DATA.seed}`
].forEach(label => { const el=document.createElement('span'); el.className='stat'; el.textContent=label; stats.appendChild(el); });

const averageTab=document.getElementById('average-tab');
const explorerTab=document.getElementById('explorer-tab');
const averagePage=document.getElementById('average-page');
const explorerPage=document.getElementById('explorer-page');
const averageColors={static:'#17212b',pan:'#0782a0',tilt:'#d97706',zoom:'#609542'};
function showPage(page) {
  const showAverage=page==='average';
  averagePage.classList.toggle('is-hidden',!showAverage);
  explorerPage.classList.toggle('is-hidden',showAverage);
  averageTab.setAttribute('aria-pressed',String(showAverage));
  explorerTab.setAttribute('aria-pressed',String(!showAverage));
}
averageTab.addEventListener('click',()=>showPage('average'));
explorerTab.addEventListener('click',()=>showPage('explorer'));

const averageTableBody=document.getElementById('average-table-body');
DATA.all_video_averages.forEach(row=>{
  const tr=document.createElement('tr');
  const setting=document.createElement('td');
  const swatch=document.createElement('span');swatch.className='motion-swatch';swatch.style.background=averageColors[row.motion];
  setting.append(swatch,document.createTextNode(row.motion));
  const acc=document.createElement('td');acc.textContent=row.mean_acc_all_frames==null?'NA':row.mean_acc_all_frames.toFixed(4);
  const aligned=document.createElement('td');aligned.textContent=row.mean_aligned_acc_all_frames==null?'NA':row.mean_aligned_acc_all_frames.toFixed(4);
  const delta=document.createElement('td');delta.textContent=row.aligned_minus_raw_acc==null?'NA':`${row.aligned_minus_raw_acc>=0?'+':''}${row.aligned_minus_raw_acc.toFixed(4)}`;
  const videos=document.createElement('td');videos.textContent=String(row.video_count);
  const frames=document.createElement('td');frames.textContent=String(row.total_video_frames);
  const sd=document.createElement('td');sd.textContent=row.video_mean_sd==null?'NA':row.video_mean_sd.toFixed(4);
  tr.append(setting,acc,aligned,delta,videos,frames,sd);averageTableBody.appendChild(tr);
});

function drawAggregateChart() {
  const svg=document.getElementById('average-chart');
  const ns='http://www.w3.org/2000/svg', W=960, H=470;
  const left=72,right=24,top=22,bottom=122,plotW=W-left-right,plotH=H-top-bottom;
  const add=(tag,attrs,text='')=>{
    const el=document.createElementNS(ns,tag);
    Object.entries(attrs).forEach(([key,value])=>el.setAttribute(key,value));
    if(text)el.textContent=text;
    svg.appendChild(el);return el;
  };
  const x=i=>left+(i/48)*plotW, y=v=>top+(1-v)*plotH;
  [0,.25,.5,.75,1].forEach(value=>{
    const yy=y(value);
    add('line',{x1:left,y1:yy,x2:W-right,y2:yy,stroke:'#e3e9ee','stroke-width':1});
    add('text',{x:left-12,y:yy+4,'text-anchor':'end',fill:'#607484','font-size':11},value.toFixed(2));
  });
  [0,12,24,36,48].forEach(frame=>{
    const xx=x(frame);
    add('line',{x1:xx,y1:top+plotH,x2:xx,y2:top+plotH+5,stroke:'#566b7b','stroke-width':1});
    add('text',{x:xx,y:top+plotH+22,'text-anchor':'middle',fill:'#607484','font-size':11},String(frame));
  });
  add('line',{x1:left,y1:top,x2:left,y2:top+plotH,stroke:'#566b7b','stroke-width':1.2});
  add('line',{x1:left,y1:top+plotH,x2:W-right,y2:top+plotH,stroke:'#566b7b','stroke-width':1.2});
  add('text',{x:left+plotW/2,y:H-12,'text-anchor':'middle',fill:'#31495b','font-size':13},'Frame index');
  add('text',{x:18,y:top+plotH/2,'text-anchor':'middle',fill:'#31495b','font-size':13,transform:`rotate(-90 18 ${top+plotH/2})`},'Average ACC');
  DATA.all_video_averages.forEach((row,index)=>{
    for(const [key,style,label] of [['frame_means','', 'raw'],['aligned_frame_means','7 5','RAFT-aligned']]){
      const means=row[key]||[];
      const points=means.map((value,i)=>value==null?'':`${i===0?'M':'L'}${x(i)},${y(value)}`).filter(Boolean).join(' ');
      if(points)add('path',{d:points,fill:'none',stroke:averageColors[row.motion],'stroke-width':2.8,'stroke-linejoin':'round','stroke-linecap':'round','stroke-dasharray':style});
      const lx=left+index*215, ly=label==='raw'?H-66:H-42;
      add('line',{x1:lx,y1:ly,x2:lx+24,y2:ly,stroke:averageColors[row.motion],'stroke-width':3,'stroke-dasharray':style});
      add('text',{x:lx+31,y:ly+4,fill:'#354e60','font-size':12},`${row.motion} · ${label}`);
    }
  });
}
drawAggregateChart();

const list = document.getElementById('sample-list');
const results = document.getElementById('results');
const search = document.getElementById('search');

function safeSourceUrl(value) {
  if (!value) return '';
  return value.startsWith('http://') || value.startsWith('https://') ? value : `https://${value}`;
}
function renderList() {
  const query = search.value.trim().toLowerCase();
  list.replaceChildren();
  DATA.samples.filter(s => `${s.id} ${s.name} ${s.category}`.toLowerCase().includes(query)).forEach(sample => {
    const button=document.createElement('button'); button.type='button'; button.className='sample-option';
    button.setAttribute('aria-pressed',String(sample.id===selectedSampleId));
    const strong=document.createElement('strong'); strong.textContent=`${sample.id} · ${sample.category}`;
    const small=document.createElement('small'); small.textContent=sample.name;
    button.append(strong,small);
    button.addEventListener('click',()=>{selectedSampleId=sample.id;render();});
    list.appendChild(button);
  });
}
function makeCurveControls(sample) {
  const wrap=document.createElement('section'); wrap.className='curve-controls'; wrap.setAttribute('aria-label',`Motion curve selection for ${sample.id}`);
  const heading=document.createElement('strong'); heading.textContent='Visible curves · solid raw / dashed RAFT-aligned'; wrap.appendChild(heading);
  const actions=document.createElement('div'); actions.className='curve-actions';
  const all=document.createElement('button'); all.type='button'; all.textContent='Select all curves';
  all.addEventListener('click',()=>{visibleCurves.set(sample.id,new Set(sample.motions.flatMap(m=>[`${m.motion}:raw`,...(m.aligned_values?.length?[`${m.motion}:aligned`]:[])])));renderCards();});
  const none=document.createElement('button'); none.type='button'; none.textContent='Deselect all curves';
  none.addEventListener('click',()=>{visibleCurves.set(sample.id,new Set());renderCards();});
  actions.append(all,none);wrap.appendChild(actions);
  const selected=visibleCurves.get(sample.id)||new Set();
  sample.motions.forEach(motion=>{
    for(const [variant,values,labelText] of [['raw',motion.values,'raw'],['aligned',motion.aligned_values,'RAFT-aligned']]){
      if(!values?.length)continue;
      const label=document.createElement('label');label.className='curve-option';
      const key=`${motion.motion}:${variant}`;
      const input=document.createElement('input');input.type='checkbox';input.checked=selected.has(key);
      input.addEventListener('change',()=>{const next=new Set(visibleCurves.get(sample.id)||[]);input.checked?next.add(key):next.delete(key);visibleCurves.set(sample.id,next);renderCards();});
      const text=document.createElement('span');text.textContent=`${motion.motion} · ${labelText}`;
      label.append(input,text);wrap.appendChild(label);
    }
  });
  return wrap;
}
function drawChart(sample, shown) {
  const ns='http://www.w3.org/2000/svg', W=920, H=430;
  const left=72, right=22, top=20, bottom=65, plotW=W-left-right, plotH=H-top-bottom;
  const svg=document.createElementNS(ns,'svg');
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.classList.add('chart');
  svg.setAttribute('role','img'); svg.setAttribute('aria-label',`Per-frame ACC curves for ${sample.id}`);
  const add=(tag,attrs,parent=svg,text='')=>{
    const el=document.createElementNS(ns,tag);
    Object.entries(attrs).forEach(([key,value])=>el.setAttribute(key,value));
    if(text) el.textContent=text;
    parent.appendChild(el); return el;
  };
  const colors={static:'#17212b',pan:'#0782a0',tilt:'#d97706',zoom:'#609542'};
  const x=i=>left+(i/48)*plotW, y=v=>top+(1-v)*plotH;
  [0,.25,.5,.75,1].forEach(value=>{
    const yy=y(value);
    add('line',{x1:left,y1:yy,x2:W-right,y2:yy,class:'gridline'});
    add('text',{x:left-12,y:yy+4,'text-anchor':'end',class:'tick-label'},svg,value.toFixed(2));
  });
  [0,8,16,24,32,40,48].forEach(value=>{
    const xx=x(value);
    add('line',{x1:xx,y1:top,x2:xx,y2:H-bottom,class:'gridline'});
    add('text',{x:xx,y:H-bottom+21,'text-anchor':'middle',class:'tick-label'},svg,String(value));
  });
  add('line',{x1:left,y1:top,x2:left,y2:H-bottom,class:'axis'});
  add('line',{x1:left,y1:H-bottom,x2:W-right,y2:H-bottom,class:'axis'});
  add('text',{x:left+plotW/2,y:H-12,'text-anchor':'middle',class:'axis-label'},svg,'Frame index');
  add('text',{x:19,y:top+plotH/2,'text-anchor':'middle',class:'axis-label',transform:`rotate(-90 19 ${top+plotH/2})`},svg,'Bit accuracy (ACC)');
  const series=sample.motions.flatMap(motion=>[
    ...(shown.has(`${motion.motion}:raw`)?[{motion,variant:'raw',values:motion.values}]:[]),
    ...(shown.has(`${motion.motion}:aligned`)?[{motion,variant:'aligned',values:motion.aligned_values}]:[]),
  ]).filter(item=>item.values?.length);
  if(!series.length){add('text',{x:left+plotW/2,y:top+plotH/2,'text-anchor':'middle',class:'legend'},svg,'No curves selected');return svg;}
  series.forEach((item,index)=>{
    const {motion,variant,values}=item, color=colors[motion.motion]||'#58697a';
    const dash=variant==='aligned'?'7 5':'';
    let d='', drawing=false;
    values.forEach((value,i)=>{
      if(value===null || !Number.isFinite(Number(value))) { drawing=false; return; }
      d+=(drawing?' L':' M')+x(i).toFixed(2)+' '+y(Number(value)).toFixed(2);
      drawing=true;
    });
    add('path',{d,class:'curve',stroke:color,...(dash?{'stroke-dasharray':dash}:{})});
    values.forEach((value,index)=>{
      if(value===null || !Number.isFinite(Number(value))) return;
      const point=add('circle',{cx:x(index),cy:y(Number(value)),r:3.0,fill:color,class:'curve-point'});
      add('title',{},point,`Frame ${index} · ACC ${Number(value).toFixed(4)} · ${motion.motion} · ${variant==='raw'?'raw':'RAFT-aligned'}`);
    });
    const lx=left+(index%4)*205, ly=H-49+Math.floor(index/4)*18;
    add('line',{x1:lx,y1:ly,x2:lx+24,y2:ly,stroke:color,'stroke-width':3,...(dash?{'stroke-dasharray':dash}:{})});
    add('text',{x:lx+30,y:ly+4,class:'legend'},svg,`${motion.motion} ${variant==='raw'?'raw':'RAFT'}`);
  });
  return svg;
}
function makeVideoPanel(sample) {
  const panel=document.createElement('section');panel.className='video-panel';panel.setAttribute('aria-label',`Generated video preview for ${sample.id}`);
  const head=document.createElement('div');head.className='video-panel-head';
  const title=document.createElement('h3');title.className='video-panel-title';title.textContent='Video view · embedded 480×320 preview';
  const select=document.createElement('select');select.className='video-select';select.setAttribute('aria-label','Choose motion and alignment view');
  sample.motions.forEach(motion=>{
    for(const [variant,label,urlKey] of [['raw','Raw','raw_video_url'],['aligned','RAFT-aligned','aligned_video_url'],['heatmap','Aligned + decoder heatmap','heatmap_video_url']]){
      if(!motion[urlKey])continue;
      const option=document.createElement('option');option.value=`${motion.motion}|${variant}`;option.textContent=`${motion.motion} (${motion.camera_setting}) · ${label}`;select.appendChild(option);
    }
  });
  head.append(title,select);
  const video=document.createElement('video');video.className='sample-video';video.controls=true;video.playsInline=true;video.preload='metadata';video.setAttribute('aria-label',`Generated video for ${sample.id}`);
  const source=document.createElement('source');source.type='video/mp4';video.appendChild(source);
  const actions=document.createElement('div');actions.className='video-actions';
  const enlarge=document.createElement('button');enlarge.type='button';enlarge.textContent='Enlarge video';
  enlarge.addEventListener('click',async()=>{
    if(panel.style.position==='fixed'){panel.style.cssText='';video.style.cssText='';enlarge.textContent='Enlarge video';return;}
    try {
      if(video.requestFullscreen) await video.requestFullscreen();
      else if(video.webkitEnterFullscreen) video.webkitEnterFullscreen();
      else { panel.style.position='fixed'; panel.style.inset='0'; panel.style.zIndex='1000'; panel.style.display='flex'; panel.style.flexDirection='column'; panel.style.justifyContent='center'; panel.style.borderRadius='0'; panel.style.background='#101820'; video.style.maxHeight='none'; video.style.flex='1'; video.style.background='#000'; enlarge.textContent='Close enlarged video'; }
    } catch(error) { console.warn('Could not enter fullscreen video mode.',error); }
  });
  actions.appendChild(enlarge);panel.append(head,video,actions);
  const updateVideo=()=>{
    const [motionName,variant]=select.value.split('|');
    const motion=sample.motions.find(item=>item.motion===motionName);
    const sourceKey=variant==='raw'?'raw_video_url':variant==='aligned'?'aligned_video_url':'heatmap_video_url';
    source.src=motion?.[sourceKey]||'';
    video.load();
  };
  select.addEventListener('change',updateVideo);
  select.value=sample.motions.some(motion=>motion.motion==='static'&&motion.raw_video_url)?'static|raw':`${sample.motions[0]?.motion||'static'}|raw`;
  updateVideo();
  return panel;
}
function renderCards() {
  results.replaceChildren();
  const sample=DATA.samples.find(item=>item.id===selectedSampleId);
  if(!sample){const empty=document.createElement('div');empty.className='empty';empty.textContent='No source samples are available.';results.appendChild(empty);return;}
  const card=document.createElement('article');card.className='sample-card';
  const head=document.createElement('div');head.className='card-head';
  const image=document.createElement('img');image.className='sample-image';image.src=sample.image_data;image.alt=`Canonical source image for ${sample.id}`;
  const meta=document.createElement('div');meta.className='sample-meta';
  const h=document.createElement('h2');h.className='sample-title';h.textContent=`${sample.id} · Source ${sample.selected_index}`;
  const name=document.createElement('p');name.className='sample-name';name.textContent=sample.name;
  const badges=document.createElement('div');badges.className='badges';
  [`Category: ${sample.category}`,`${sample.motions.length} motion conditions`].forEach(v=>{const b=document.createElement('span');b.className='badge';b.textContent=v;badges.appendChild(b);});
  meta.append(h,name,badges);
  const url=safeSourceUrl(sample.source_url);
  if(url){const a=document.createElement('a');a.className='source-link';a.href=url;a.target='_blank';a.rel='noopener noreferrer';a.textContent='Open source image reference';meta.appendChild(a);}
  head.append(image,meta,makeVideoPanel(sample));card.appendChild(head);
  card.appendChild(makeCurveControls(sample));
  const graphWrap=document.createElement('div');graphWrap.className='graph-wrap';
  const graphTitle=document.createElement('p');graphTitle.className='graph-title';graphTitle.textContent='Per-frame ACC curve · hover points for frame-level values';
  graphWrap.append(graphTitle,drawChart(sample,visibleCurves.get(sample.id)||new Set()));
  card.appendChild(graphWrap);results.appendChild(card);
}
function render(){renderList();renderCards();}
search.addEventListener('input',renderList);
render();
</script>
</body>
</html>
'''
    inner_data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    inner_html = html_template.replace("__DATA__", inner_data_json)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(inner_html, encoding="utf-8")
    OUT_OUTER.write_text(inner_html, encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} and {OUT_OUTER.relative_to(ROOT)} with {len(samples)} grouped sources.")


if __name__ == "__main__":
    main()
