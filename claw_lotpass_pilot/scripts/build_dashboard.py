#!/usr/bin/env python3
"""Build a portable CLAW results dashboard with embedded H.264 previews."""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "claw_lotpass_pilot"
ANALYSIS = EXP / "analysis"
PREVIEWS = ANALYSIS / "dashboard_previews"
OUTPUTS = (ROOT / "claw_lotpass_dashboard.html", ANALYSIS / "dashboard.html")
PREVIEW_SIZE = (480, 320)
CONDITIONS = ("static_control", "broad_fluid_motion")
CONDITION_LABELS = {"static_control": "Frozen-scene control", "broad_fluid_motion": "Broad fluid motion"}
VIDEO_LABELS = {"raw_video_url": "Raw video", "aligned_video_url": "RAFT-aligned", "heatmap_video_url": "Aligned + decoder heatmap"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def keyed(path: Path, key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in read_rows(path)}


def numeric(value, default=None):
    try:
        result = float(value)
        return result if result == result and abs(result) != float("inf") else default
    except (TypeError, ValueError):
        return default


def sha_mtime_needs_refresh(source: Path, destination: Path) -> bool:
    return not destination.is_file() or destination.stat().st_mtime < source.stat().st_mtime


def video_data_url(case_id: str, field: str, source: Path, ffmpeg: str) -> str:
    if not source.is_file():
        return ""
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    preview = PREVIEWS / f"{case_id}_{field}.mp4"
    if sha_mtime_needs_refresh(source, preview):
        partial = preview.with_name(f".{preview.stem}.partial.mp4")
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-threads", "1", "-filter_threads", "1", "-i", str(source),
            "-vf", f"fps=8,scale={PREVIEW_SIZE[0]}:{PREVIEW_SIZE[1]}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos,pad={PREVIEW_SIZE[0]}:{PREVIEW_SIZE[1]}:(ow-iw)/2:(oh-ih)/2,setsar=1",
            "-an", "-c:v", "libx264", "-threads", "1", "-preset", "veryfast", "-crf", "28",
            "-profile:v", "baseline", "-level", "3.0", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(partial),
        ]
        subprocess.run(command, check=True, cwd=ROOT)
        os.replace(partial, preview)
    payload = base64.b64encode(preview.read_bytes()).decode("ascii")
    return "data:video/mp4;base64," + payload


def image_data_url(path: Path) -> str:
    if not path.is_file():
        return ""
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=84, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def image_png_data_url(path: Path) -> str:
    if not path.is_file():
        return ""
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_data() -> dict:
    manifest = json.loads((EXP / "selection/generation_manifest.json").read_text(encoding="utf-8"))
    raw_cases = keyed(ANALYSIS / "raw_ba/case_summary.csv", "case_id")
    stable_cases = keyed(ANALYSIS / "stability/per_video_stability.csv", "case_id")
    aligned_cases = keyed(ANALYSIS / "optical_flow/per_video_aligned_ba.csv", "case_id")
    raw_frames: dict[str, list[float | None]] = {}
    for row in read_rows(ANALYSIS / "raw_ba/per_frame_accuracy.csv"):
        raw_frames.setdefault(row["case_id"], []).append(numeric(row["ba"]))
    aligned_frames: dict[str, list[float | None]] = {}
    for row in read_rows(ANALYSIS / "optical_flow/per_frame_aligned_ba.csv"):
        aligned_frames.setdefault(row["case_id"], []).append(numeric(row["aligned_BA"]))
    analysis = {row["image_id"]: row for row in read_rows(EXP / "selection/claw_sample_analysis.csv")}
    prompts: dict[str, dict[str, dict[str, str]]] = {}
    for row in read_rows(EXP / "selection/claw_prompt_pairs.csv"):
        prompts.setdefault(row["image_id"], {})[row["condition"]] = row
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    samples = []
    preview_jobs = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for image in sorted(manifest["selected_images"], key=lambda row: int(row["sample_index"])):
            image_id = image["image_id"]
            watermarked = ROOT / image["canonical_image_path"].replace("images/canonical_clean", "images/watermarked")
            source = EXP / "images/source" / image["source_filename"]
            encoder_heatmap = ANALYSIS / "heatmaps/encoder/canonical_clean" / f"{image_id}_heatmap.png"
            sample = {
                "id": image_id, "sample_index": int(image["sample_index"]), "name": image_id,
                "category": analysis[image_id]["scene_category"],
                "dynamic_coverage": int(analysis[image_id]["estimated_dynamic_coverage"]),
                "dense_candidate": analysis[image_id]["dense_dynamic_candidate"],
                "notes": analysis[image_id]["notes"],
                "source_image_url": image_data_url(source),
                "watermarked_input_url": image_data_url(watermarked),
                "encoder_heatmap_url": image_png_data_url(encoder_heatmap),
                "conditions": [],
            }
            for condition in CONDITIONS:
                case_id = f"CLAW{int(image['sample_index']):03d}_{condition}"
                run = EXP / "runs" / case_id / "seed_42"
                output = ANALYSIS / "optical_flow" / case_id
                raw, stable, aligned = raw_cases[case_id], stable_cases[case_id], aligned_cases[case_id]
                prompt = prompts[image_id][condition]["generated_prompt"]
                clip = {
                    "case_id": case_id, "condition": condition, "label": CONDITION_LABELS[condition],
                    "prompt": prompt, "status": raw["status"],
                    "mean_raw_ba": numeric(raw.get("mean_raw_ba_excluding_copy_frame")),
                    "minimum_raw_ba": numeric(raw.get("minimum_raw_ba")),
                    "last_third_raw_ba": numeric(raw.get("last_third_raw_ba_excluding_copy_frame")),
                    "mean_aligned_ba": numeric(aligned.get("aligned_mean_BA_excluding_copy_frame")),
                    "aligned_minus_raw_ba": numeric(aligned.get("aligned_minus_raw_BA_excluding_copy_frame")),
                    "stable_region_ratio": numeric(stable.get("mean_stable_region_ratio")),
                    "largest_stable_region_ratio": numeric(stable.get("mean_largest_stable_region_ratio")),
                    "mean_similarity": numeric(stable.get("mean_similarity")),
                    "aligned_stable_region_ratio": numeric(aligned.get("mean_stable_region_ratio_aligned")),
                    "decoder_stable_overlap": numeric(aligned.get("mean_decoder_stable_overlap")),
                    "encoder_stable_overlap": numeric(aligned.get("mean_encoder_stable_overlap")),
                    "raw_values": raw_frames.get(case_id, []), "aligned_values": aligned_frames.get(case_id, []),
                    "video_urls": {},
                    "raw_path": str((run / "video.mp4").relative_to(ROOT)),
                    "aligned_path": str((output / "aligned_video.mp4").relative_to(ROOT)),
                    "heatmap_path": str((output / "decoder_heatmap_overlay.mp4").relative_to(ROOT)),
                }
                for field, path in (
                    ("raw_video_url", run / "video.mp4"),
                    ("aligned_video_url", output / "aligned_video.mp4"),
                    ("heatmap_video_url", output / "decoder_heatmap_overlay.mp4"),
                ):
                    preview_jobs.append((clip, field, path, pool.submit(video_data_url, case_id, field, path, ffmpeg)))
                sample["conditions"].append(clip)
            samples.append(sample)
        for clip, field, _, future in preview_jobs:
            clip["video_urls"][field] = future.result()

    condition_summary = read_rows(ANALYSIS / "summary/condition_summary.csv")
    condition_summary_by_id = {row["condition"]: row for row in condition_summary}
    averages = []
    for condition in CONDITIONS:
        clips = [s["conditions"][0 if condition == CONDITIONS[0] else 1] for s in samples]
        frame_count = min((len(clip["raw_values"]) for clip in clips), default=0)
        aligned_count = min((len(clip["aligned_values"]) for clip in clips), default=0)
        averages.append({
            "condition": condition, "label": CONDITION_LABELS[condition], "video_count": len(clips),
            "mean_raw_ba": numeric(condition_summary_by_id[condition].get("mean_mean_raw_ba")),
            "mean_aligned_ba": numeric(condition_summary_by_id[condition].get("mean_mean_aligned_ba")),
            "mean_stable_region_ratio": numeric(condition_summary_by_id[condition].get("mean_stable_region_ratio")),
            "mean_largest_stable_region_ratio": numeric(condition_summary_by_id[condition].get("mean_largest_stable_region_ratio")),
            "mean_decoder_stable_overlap": numeric(condition_summary_by_id[condition].get("mean_decoder_stable_overlap")),
            "raw_frame_means": [sum(clip["raw_values"][i] or 0 for clip in clips) / sum(clip["raw_values"][i] is not None for clip in clips) for i in range(frame_count)],
            "aligned_frame_means": [sum(clip["aligned_values"][i] or 0 for clip in clips) / sum(clip["aligned_values"][i] is not None for clip in clips) for i in range(aligned_count)],
        })

    charts = {}
    for filename in ("coverage_vs_raw_acc.png", "coverage_vs_stability.png", "coverage_vs_decoder_overlap.png", "case_raw_aligned_acc.png"):
        charts[filename] = image_png_data_url(ANALYSIS / "summary" / filename)
    correlations = read_rows(ANALYSIS / "summary/dynamic_coverage_correlations.csv")
    return {
        "title": "CLAW · LoT-Pass × CogVideoX", "seed": 42, "frames": 49, "fps": 8,
        "samples": samples, "condition_averages": averages, "charts": charts,
        "correlations": correlations,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CLAW · LoT-Pass experiment</title>
<style>
:root{color-scheme:light;--ink:#162a3a;--muted:#637585;--line:#dce5ec;--paper:#f4f7fa;--card:#fff;--blue:#197b9b;--navy:#173c59;--orange:#cf6c35;font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--paper)}header{padding:28px clamp(18px,4vw,52px) 22px;color:#f7fbff;background:linear-gradient(120deg,#13334d,#1c6880)}h1{margin:0;font-size:clamp(1.45rem,2.7vw,2.1rem);letter-spacing:-.025em}.subtitle{max-width:1000px;margin:8px 0 17px;color:#d7e8f1}.stats{display:flex;flex-wrap:wrap;gap:8px}.stat{border:1px solid #ffffff3a;border-radius:999px;padding:5px 11px;background:#ffffff12;font-size:.84rem}
.page-nav{display:flex;gap:8px;max-width:1600px;margin:0 auto;padding:18px clamp(14px,3vw,40px) 0}button{border:1px solid #c7d4df;border-radius:7px;padding:8px 10px;background:#f9fbfd;color:var(--ink);font:inherit;cursor:pointer}button:hover{border-color:#77aabd;background:#eff8fb}.page-nav button[aria-pressed=true]{background:#173c59;border-color:#173c59;color:#fff}.hidden{display:none!important}.overview,.explorer{max-width:1600px;margin:0 auto;padding:18px clamp(14px,3vw,40px) 52px}.section-title{margin:0 0 4px;font-size:1.2rem}.muted{color:var(--muted)}.table-wrap{overflow:auto;margin:12px 0 20px;border:1px solid var(--line);border-radius:10px;background:#fff}.result-table{width:100%;border-collapse:collapse;min-width:700px}.result-table th,.result-table td{padding:10px 13px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}.result-table th{background:#f6f9fb;color:#395468;font-size:.82rem}.result-table tr:last-child td{border-bottom:0}.chart-card,.figure-card{border:1px solid var(--line);border-radius:11px;background:#fff;padding:14px 16px}.chart-card h3,.figure-card h3{margin:0 0 7px;font-size:.96rem;color:#395468}.chart{display:block;width:100%;height:360px;background:#fff}.figure-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-top:15px}.figure-card img{display:block;width:100%;height:auto}.figure-card p{margin:6px 0 0;color:var(--muted);font-size:.8rem}.metric-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:9px;margin:13px 0}.metric{border:1px solid var(--line);border-radius:9px;background:#fff;padding:10px 12px}.metric small{display:block;color:var(--muted)}.metric strong{font-size:1.04rem}
.explorer-layout{display:grid;grid-template-columns:280px minmax(0,1fr);gap:20px;align-items:start}.sidebar{position:sticky;top:12px;border:1px solid var(--line);border-radius:12px;background:#fff;padding:14px}.sidebar h2{margin:0 0 4px;font-size:1rem}.search{width:100%;margin:8px 0;border:1px solid #c7d4df;border-radius:7px;padding:9px;font:inherit}.sample-list{display:grid;gap:4px;max-height:65vh;overflow:auto}.sample-option{width:100%;text-align:left;border-color:transparent;background:transparent;padding:8px}.sample-option[aria-pressed=true]{border-color:#9fc8d5;background:#eaf5f8}.sample-option strong,.sample-option small{display:block}.sample-option small{color:var(--muted)}.sample-card{border:1px solid var(--line);border-radius:12px;background:#fff;overflow:hidden}.card-head{display:grid;grid-template-columns:180px minmax(0,1fr) 180px;gap:14px;align-items:start;padding:15px;border-bottom:1px solid var(--line);background:linear-gradient(120deg,#fff,#f8fbfd)}.card-image{width:100%;aspect-ratio:3/2;object-fit:cover;border:1px solid var(--line);border-radius:8px;background:#edf2f5}.sample-name{overflow-wrap:anywhere;margin:3px 0 6px;color:#42596b;font-size:.86rem}.heatmap-image{width:100%;aspect-ratio:3/2;object-fit:cover;border:1px solid var(--line);border-radius:8px}.tag{display:inline-block;margin:3px 5px 0 0;padding:3px 8px;border-radius:999px;background:#eaf3f7;color:#22556b;font-size:.74rem}.condition-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;padding:14px}.condition-card{min-width:0;border:1px solid var(--line);border-radius:10px;padding:11px;background:#fff}.condition-card h3{margin:0 0 8px;font-size:.95rem}.condition-card select{width:100%;border:1px solid #c7d4df;border-radius:6px;padding:7px;margin-bottom:8px;background:#fff;font:inherit}.sample-video{display:block;width:100%;aspect-ratio:3/2;max-height:280px;object-fit:contain;border-radius:6px;background:#101820}.video-error{min-height:1.3em;color:#a13e2a;font-size:.78rem}.case-metrics{width:100%;border-collapse:collapse;font-size:.78rem}.case-metrics th,.case-metrics td{padding:5px 4px;border-bottom:1px solid #edf1f4;text-align:left}.case-metrics th{color:#637585;font-weight:550}.curve-controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;margin:0 14px 5px;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:#f8fbfd}.curve-controls label{display:inline-flex;align-items:center;gap:5px;font-size:.8rem}.curve-controls input{accent-color:var(--blue)}.prompt-section{margin:0 14px 14px;border:1px solid var(--line);border-radius:8px;padding:10px 12px}.prompt-section summary{cursor:pointer;color:#395468;font-weight:650}.prompt-section p{font-size:.84rem}.footer{max-width:1500px;margin:0 auto;padding:0 28px 26px;color:var(--muted);font-size:.78rem}
@media(max-width:1000px){.explorer-layout{grid-template-columns:1fr}.sidebar{position:static}.sample-list{max-height:240px;grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}.card-head{grid-template-columns:130px minmax(0,1fr)}.heatmap-image{grid-column:1/-1;width:min(260px,100%)}}@media(max-width:680px){.condition-grid{grid-template-columns:1fr}.card-head{grid-template-columns:90px minmax(0,1fr);gap:10px;padding:11px}.card-image{width:90px}.heatmap-image{width:180px}.chart{height:280px}}
</style>
</head>
<body>
<header><h1 id="title"></h1><p class="subtitle">Explore the paired frozen-scene and broad-fluid-motion conditions. Switch each player between raw video, full-frame RAFT alignment, and the aligned decoder-heatmap overlay. All video previews are embedded for offline playback.</p><div class="stats" id="stats"></div></header>
<nav class="page-nav"><button id="overview-tab" aria-pressed="true">Overview</button><button id="explorer-tab" aria-pressed="false">Per-image explorer</button></nav>
<section class="overview" id="overview">
  <h2 class="section-title">Condition summary</h2><p class="muted">Clip means exclude a detected frame-zero conditioning copy. Curves show per-frame means across all 15 clips in each condition.</p>
  <div class="table-wrap"><table class="result-table"><thead><tr><th>Condition</th><th>Clips</th><th>Raw mean ACC</th><th>RAFT-aligned mean ACC</th><th>Alignment change</th><th>Stable-region ratio</th><th>Largest stable region</th><th>Decoder / stable overlap</th></tr></thead><tbody id="summary-body"></tbody></table></div>
  <div class="chart-card"><h3>Raw and RAFT-aligned ACC by frame</h3><canvas class="chart" id="overview-chart" height="360"></canvas></div>
  <h2 class="section-title" style="margin-top:22px">Image coverage and stability analyses</h2><div class="figure-grid" id="figures"></div>
  <h2 class="section-title" style="margin-top:22px">Coverage correlations</h2><p class="muted">Coverage correlations average each image’s two prompt conditions first (n = 15 images); they are exploratory, not causal.</p>
  <div class="table-wrap"><table class="result-table"><thead><tr><th>Outcome</th><th>n images</th><th>Pearson r (p)</th><th>Spearman ρ (p)</th></tr></thead><tbody id="correlation-body"></tbody></table></div>
</section>
<section class="explorer hidden" id="explorer"><div class="explorer-layout"><aside class="sidebar"><h2>Sample images</h2><p class="muted">Uniform sample of 15 · seed 42</p><input class="search" id="search" type="search" placeholder="Filter image IDs or scene types"><div class="sample-list" id="sample-list"></div></aside><main id="sample-detail"></main></div></section>
<footer class="footer">CLAW v1 · LoT-Pass watermark · CogVideoX-5b-I2V · seed 42 · 49 frames · 8 fps. Alignment is a diagnostic view; raw videos remain unchanged. Review the linked experiment report for methodology and interpretation.</footer>
<script>
const DATA = __CLAW_DATA__;
const COLORS={static_control:'#315b7d',broad_fluid_motion:'#d77732'};
document.title=DATA.title; document.getElementById('title').textContent=DATA.title;
const stats=document.getElementById('stats');
for(const t of [`${DATA.samples.length} sampled images`,`30 generated clips`,`${DATA.frames} frames · ${DATA.fps} fps`,`720 × 480`,`RAFT alignment + decoder heatmaps`]){const s=document.createElement('span');s.className='stat';s.textContent=t;stats.appendChild(s)}
function fmt(v,d=4){return Number.isFinite(Number(v))?Number(v).toFixed(d):'—'}
function lineChart(canvas, series, title=''){
 const rect=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
 const w=Math.max(320,rect.width),h=Math.max(240,rect.height||360);canvas.width=w*dpr;canvas.height=h*dpr;
 const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
 const L=50,R=15,T=15,B=38,CW=w-L-R,CH=h-T-B;
 ctx.font='11px system-ui';ctx.textAlign='right';ctx.textBaseline='middle';
 for(let j=0;j<=4;j++){let val=j/4,y=T+CH*(1-val);ctx.strokeStyle='#e3e9ee';ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(w-R,y);ctx.stroke();ctx.fillStyle='#607484';ctx.fillText(val.toFixed(2),L-7,y)}
 ctx.strokeStyle='#566b7b';ctx.beginPath();ctx.moveTo(L,T);ctx.lineTo(L,T+CH);ctx.lineTo(w-R,T+CH);ctx.stroke();
 const n=Math.max(...series.map(s=>(s.values||[]).length),1);ctx.textAlign='center';ctx.fillStyle='#607484';
 for(const idx of [0,Math.floor((n-1)/2),n-1]){const x=L+CW*idx/Math.max(1,n-1);ctx.fillText(String(idx),x,T+CH+17)}
 ctx.fillStyle='#31495b';ctx.fillText('Frame index',L+CW/2,h-7);ctx.save();ctx.translate(13,T+CH/2);ctx.rotate(-Math.PI/2);ctx.fillText('Bit accuracy',0,0);ctx.restore();
 for(const s of series){const vals=s.values||[];if(!vals.length)continue;ctx.beginPath();ctx.strokeStyle=s.color;ctx.lineWidth=2.2;ctx.setLineDash(s.dash?[6,4]:[]);let started=false;for(let i=0;i<vals.length;i++){if(vals[i]===null||!Number.isFinite(Number(vals[i]))){started=false;continue}const x=L+CW*i/Math.max(1,vals.length-1),y=T+CH*(1-Number(vals[i]));if(!started){ctx.moveTo(x,y);started=true}else ctx.lineTo(x,y)}ctx.stroke();ctx.setLineDash([])}
}
function overview(){const series=[];for(const c of DATA.condition_averages){series.push({values:c.raw_frame_means,color:COLORS[c.condition],label:c.label+' raw'});series.push({values:c.aligned_frame_means,color:COLORS[c.condition],dash:true,label:c.label+' aligned'})}lineChart(document.getElementById('overview-chart'),series)}
const summaryBody=document.getElementById('summary-body');
for(const c of DATA.condition_averages){const tr=summaryBody.insertRow();for(const v of [c.label,c.video_count,fmt(c.mean_raw_ba),fmt(c.mean_aligned_ba),fmt(Number(c.mean_aligned_ba)-Number(c.mean_raw_ba)),fmt(c.mean_stable_region_ratio),fmt(c.mean_largest_stable_region_ratio),fmt(c.mean_decoder_stable_overlap)])tr.insertCell().textContent=v}
const figureInfo=[['coverage_vs_raw_acc.png','Estimated fluid coverage vs raw ACC'],['coverage_vs_stability.png','Estimated fluid coverage vs stable regions'],['coverage_vs_decoder_overlap.png','Estimated fluid coverage vs decoder / stable overlap'],['case_raw_aligned_acc.png','Raw and aligned mean ACC by case']];
for(const [file,label] of figureInfo){const src=DATA.charts[file];if(!src)continue;const card=document.createElement('article');card.className='figure-card';const h=document.createElement('h3');h.textContent=label;const img=document.createElement('img');img.src=src;img.alt=label;card.append(h,img);document.getElementById('figures').appendChild(card)}
const corrBody=document.getElementById('correlation-body');for(const c of DATA.correlations){const tr=corrBody.insertRow();for(const v of [c.y,c.n_images,fmt(c.pearson_r,3)+' ('+fmt(c.pearson_p,3)+')',fmt(c.spearman_rho,3)+' ('+fmt(c.spearman_p,3)+')'])tr.insertCell().textContent=v}
const list=document.getElementById('sample-list'),detail=document.getElementById('sample-detail'),search=document.getElementById('search');let selected=DATA.samples[0]?.id;
function renderList(){list.replaceChildren();const q=search.value.trim().toLowerCase();for(const s of DATA.samples){if(q&&!(`${s.name} ${s.category}`.toLowerCase().includes(q)))continue;const b=document.createElement('button');b.className='sample-option';b.setAttribute('aria-pressed',String(s.id===selected));const strong=document.createElement('strong');strong.textContent=`${String(s.sample_index).padStart(2,'0')} · ${s.category}`;const small=document.createElement('small');small.textContent=s.name;b.append(strong,small);b.onclick=()=>{selected=s.id;renderList();renderDetail()};list.appendChild(b)}}
function createConditionCard(s,c){const card=document.createElement('article');card.className='condition-card';const h=document.createElement('h3');h.textContent=c.label;const select=document.createElement('select');for(const [key,label] of [['raw_video_url','Raw video'],['aligned_video_url','RAFT-aligned'],['heatmap_video_url','Aligned + decoder heatmap']]){const o=document.createElement('option');o.value=key;o.textContent=label;select.appendChild(o)}select.value='heatmap_video_url';const video=document.createElement('video');video.className='sample-video';video.controls=true;video.playsInline=true;video.preload='metadata';const error=document.createElement('p');error.className='video-error';function update(){error.textContent='';video.src=c.video_urls[select.value]||'';video.load()}video.onerror=()=>{error.textContent='Embedded MP4 could not be played in this browser.'};select.onchange=update;update();card.append(h,select,video,error);const table=document.createElement('table');table.className='case-metrics';const fields=[['Raw mean ACC',c.mean_raw_ba],['RAFT-aligned mean ACC',c.mean_aligned_ba],['Aligned − raw ACC',c.aligned_minus_raw_ba],['Last-third raw ACC',c.last_third_raw_ba],['Minimum raw ACC',c.minimum_raw_ba],['Stable-region ratio',c.stable_region_ratio],['Largest stable region',c.largest_stable_region_ratio],['Aligned decoder / stable overlap',c.decoder_stable_overlap]];for(const [name,val] of fields){const tr=table.insertRow();tr.insertCell().textContent=name;tr.insertCell().textContent=fmt(val)}card.appendChild(table);return card}
function renderDetail(){const s=DATA.samples.find(x=>x.id===selected);if(!s)return;detail.replaceChildren();const card=document.createElement('article');card.className='sample-card';const head=document.createElement('div');head.className='card-head';const source=document.createElement('img');source.className='card-image';source.src=s.source_image_url;source.alt='Original CLAW source image';const meta=document.createElement('div');const title=document.createElement('h2');title.textContent=`Sample ${String(s.sample_index).padStart(2,'0')}`;const name=document.createElement('p');name.className='sample-name';name.textContent=s.name;const tags=document.createElement('div');for(const t of [`Scene: ${s.category}`,`Estimated fluid coverage: ${s.dynamic_coverage}%`,`Dense-motion candidate: ${s.dense_candidate}`]){const span=document.createElement('span');span.className='tag';span.textContent=t;tags.appendChild(span)}const notes=document.createElement('p');notes.className='muted';notes.textContent=s.notes;meta.append(title,name,tags,notes);const heat=document.createElement('img');heat.className='heatmap-image';heat.src=s.encoder_heatmap_url;heat.alt='LoT-Pass encoder residual heatmap';head.append(source,meta,heat);card.appendChild(head);
 const panels=document.createElement('div');panels.className='condition-grid';for(const c of s.conditions)panels.appendChild(createConditionCard(s,c));card.appendChild(panels);
 const controls=document.createElement('div');controls.className='curve-controls';const label=document.createElement('strong');label.textContent='ACC curves:';controls.appendChild(label);const series=[];for(const c of s.conditions)for(const type of ['raw','aligned']){const id=`${c.condition}-${type}`,cb=document.createElement('input');cb.type='checkbox';cb.checked=true;cb.id=id;const text=document.createElement('label');text.append(cb,document.createTextNode(`${c.label} · ${type==='raw'?'raw':'RAFT-aligned'}`));controls.appendChild(text);series.push({id,values:type==='raw'?c.raw_values:c.aligned_values,color:COLORS[c.condition],dash:type==='aligned'})}const plot=document.createElement('canvas');plot.className='chart';plot.height=350;card.append(controls,plot);detail.appendChild(card);
 function redraw(){lineChart(plot,series.filter(s=>document.getElementById(s.id)?.checked))}controls.addEventListener('change',redraw);redraw();
 for(const c of s.conditions){const d=document.createElement('details');d.className='prompt-section';const summary=document.createElement('summary');summary.textContent=`${c.label} prompt · ${c.prompt.split(/\s+/).length} words`;const p=document.createElement('p');p.textContent=c.prompt;d.append(summary,p);detail.appendChild(d)}
}
search.addEventListener('input',renderList);document.getElementById('overview-tab').onclick=()=>{document.getElementById('overview').classList.remove('hidden');document.getElementById('explorer').classList.add('hidden');document.getElementById('overview-tab').setAttribute('aria-pressed','true');document.getElementById('explorer-tab').setAttribute('aria-pressed','false');overview()};document.getElementById('explorer-tab').onclick=()=>{document.getElementById('overview').classList.add('hidden');document.getElementById('explorer').classList.remove('hidden');document.getElementById('overview-tab').setAttribute('aria-pressed','false');document.getElementById('explorer-tab').setAttribute('aria-pressed','true');renderList();renderDetail()};
overview();
</script></body></html>'''


def main() -> None:
    data = build_data()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__CLAW_DATA__", payload)
    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(f".{output.name}.partial")
        partial.write_text(html, encoding="utf-8")
        os.replace(partial, output)
        print(f"Wrote {output.relative_to(ROOT)} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
