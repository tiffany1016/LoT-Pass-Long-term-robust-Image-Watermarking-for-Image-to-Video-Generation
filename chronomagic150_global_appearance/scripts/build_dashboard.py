#!/usr/bin/env python3
"""Build a self-contained, offline-playable ChronoMagic results dashboard."""
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
EXP = ROOT / "chronomagic150_global_appearance"
ANALYSIS = EXP / "analysis"
PREVIEWS = ANALYSIS / "dashboard_previews"
OUTS = (ROOT / "chronomagic150_dashboard.html", ANALYSIS / "dashboard.html")
SIZE = (480, 320)
VIDEO_FIELDS = ("raw_video_url", "similarity_video_url", "stable_mask_video_url", "decoder_heatmap_video_url")
VIDEO_LABELS = {"raw_video_url": "Raw video", "similarity_video_url": "Similarity overlay",
                "stable_mask_video_url": "Stable-mask overlay", "decoder_heatmap_video_url": "Decoder heatmap overlay"}
COLORS = {"transition": "#cf6c35", "control": "#287c9a"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def numeric(value):
    try:
        x = float(value)
        return x if x == x and abs(x) != float("inf") else None
    except (TypeError, ValueError):
        return None


def image_url(path: Path, max_size=SIZE, fmt="JPEG") -> str:
    if not path.is_file():
        return ""
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format=fmt, quality=86 if fmt == "JPEG" else None, optimize=True)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def video_url(case_id: str, field: str, source: Path, ffmpeg: str) -> str:
    if not source.is_file():
        return ""
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    target = PREVIEWS / f"{case_id}_{field}.mp4"
    if not target.is_file() or target.stat().st_mtime < source.stat().st_mtime:
        partial = target.with_name(f".{target.stem}.partial.mp4")
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-threads", "1",
                   "-filter_threads", "1", "-i", str(source), "-vf",
                   f"fps=8,scale={SIZE[0]}:{SIZE[1]}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos,pad={SIZE[0]}:{SIZE[1]}:(ow-iw)/2:(oh-ih)/2,setsar=1",
                   "-an", "-c:v", "libx264", "-threads", "1", "-preset", "veryfast", "-crf", "28",
                   "-profile:v", "baseline", "-level", "3.0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial)]
        subprocess.run(command, check=True, cwd=ROOT)
        os.replace(partial, target)
    return "data:video/mp4;base64," + base64.b64encode(target.read_bytes()).decode("ascii")


def build_data() -> dict:
    samples = read_rows(EXP / "dataset/chronomagic150_selected16.csv")
    cases = read_rows(EXP / "dataset/chronomagic150_generation_manifest.csv")
    per_frame = read_rows(EXP / "decoding/per_frame_raw_ba.csv")
    videos = {row["case_id"]: row for row in read_rows(ANALYSIS / "per_video_metrics.csv")}
    appearance = {row["case_id"]: row for row in read_rows(ANALYSIS / "appearance_summary.csv")}
    validation = {row["case_id"]: row for row in read_rows(ANALYSIS / "transition_validation.csv")}
    averages = read_rows(ANALYSIS / "condition_summary.csv")
    corr = read_rows(ANALYSIS / "paired_change_correlations.csv")
    flow = ANALYSIS / "optical_flow"
    aligned = {}
    aligned_curves = {}
    if (flow / "summary.json").is_file():
        aligned = {r["case_id"]: r for r in read_rows(flow / "per_video_aligned_ba.csv")}
        for r in read_rows(flow / "per_frame_aligned_ba.csv"):
            aligned_curves.setdefault(r["case_id"], {})[int(r["frame_idx"])] = numeric(r["aligned_BA"])
    by_frame: dict[str, dict[int, float | None]] = {}
    for row in per_frame:
        by_frame.setdefault(row["case_id"], {})[int(row["frame_idx"])] = numeric(row["raw_bit_accuracy"])
    cases_by_sample: dict[str, list[dict]] = {}
    for row in cases:
        sample_id = row["sample_id"]
        if sample_id not in cases_by_sample:
            cases_by_sample[sample_id] = []
        case = {**row,
                "mean_ba": numeric(videos.get(row["case_id"], {}).get("mean_ba")),
                "last_third_ba": numeric(videos.get(row["case_id"], {}).get("last_third_mean_ba")),
                "minimum_ba": numeric(videos.get(row["case_id"], {}).get("minimum_ba")),
                "mean_srr": numeric(videos.get(row["case_id"], {}).get("mean_stable_region_ratio")),
                "mean_largest_srr": numeric(videos.get(row["case_id"], {}).get("mean_largest_stable_region_ratio")),
                "mean_similarity": numeric(videos.get(row["case_id"], {}).get("mean_similarity")),
                "encoder_stable_overlap": numeric(videos.get(row["case_id"], {}).get("mean_encoder_stable_overlap")),
                "decoder_stable_overlap": numeric(videos.get(row["case_id"], {}).get("mean_decoder_stable_overlap")),
                "mean_luminance_delta": numeric(appearance.get(row["case_id"], {}).get("final_minus_initial_mean_luminance")),
                "changed_area": numeric(appearance.get(row["case_id"], {}).get("final_fraction_pixels_luma_change_gt_0_08")),
                "raw_curve": [by_frame.get(row["case_id"], {}).get(i) for i in range(49)],
                "validation_status": validation.get(row["case_id"], {}).get("status", "pending")}
        alignment = aligned.get(row["case_id"])
        if alignment and alignment["status"] == "completed":
            case["aligned_mean"] = numeric(alignment["aligned_mean_BA_excluding_copy_frame"])
            case["alignment_delta"] = numeric(alignment["aligned_minus_raw_BA_excluding_copy_frame"])
            case["aligned_curve"] = [aligned_curves[row["case_id"]].get(i) for i in range(49)]
        cases_by_sample[sample_id].append(case)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    all_jobs = []
    sample_data = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for sample in sorted(samples, key=lambda x: int(x["selection_index"])):
            sample_id = sample["sample_id"]
            watermarked = EXP / "watermarked_sources" / sample_id / "watermarked.png"
            alias = f"CM{int(sample['selection_index']):03d}"
            encoder_heat = EXP / "heatmaps/encoder/source_images" / f"{alias}_heatmap.png"
            item = {
                "sample_id": sample_id, "selection_index": int(sample["selection_index"]),
                "analysis_group": sample["analysis_group"], "project_label": sample["project_label"],
                "sub_category": sample["sub_category"], "official_caption": sample["official_caption"],
                "selection_reason": sample["selection_reason"], "transition_prompt": sample["transition_prompt"],
                "control_prompt": sample["control_prompt"],
                "source_image_url": image_url(ROOT / sample["source_frame_original"]),
                "watermarked_image_url": image_url(watermarked),
                "encoder_heatmap_url": image_url(encoder_heat, fmt="PNG"),
                "conditions": [],
            }
            for condition in ("transition", "control"):
                case = next((c for c in cases_by_sample.get(sample_id, []) if c["condition"] == condition), None)
                if case is None:
                    continue
                case_id = case["case_id"]
                urls = {}
                sources = {
                    "raw_video_url": EXP / "generated_videos" / sample_id / f"{condition}.mp4",
                    "similarity_video_url": EXP / "stable_region" / sample_id / condition / "similarity_overlay.mp4",
                    "stable_mask_video_url": EXP / "stable_region" / sample_id / condition / "stable_mask_overlay.mp4",
                    "decoder_heatmap_video_url": EXP / "heatmaps/decoder" / sample_id / condition / "decoder_heatmap_overlay.mp4",
                }
                if "aligned_curve" in case:
                    sources["aligned_video_url"] = ROOT / aligned[case_id]["aligned_video"]
                    sources["aligned_heatmap_video_url"] = ROOT / aligned[case_id]["decoder_heatmap_overlay_video"]
                for field, path in sources.items():
                    all_jobs.append((case, field, pool.submit(video_url, case_id, field, path, ffmpeg)))
                    urls[field] = ""
                case["video_urls"] = urls
                item["conditions"].append(case)
            sample_data.append(item)
        for case, field, future in all_jobs:
            case["video_urls"][field] = future.result()

    chart_paths = ("raw_ba_trajectories.png", "paired_delta_mean_ba.png", "ba_vs_stability.png", "ba_vs_heatmap_overlap.png")
    charts = {name: image_url(ANALYSIS / name, max_size=(1200, 700), fmt="PNG") for name in chart_paths}
    report = (ANALYSIS / "final_report.md").read_text(encoding="utf-8")
    if aligned:
        report += "\n\n" + (flow / "report.md").read_text(encoding="utf-8")
        for name in ("raw_vs_aligned_acc.png", "acc_curve_raw_vs_aligned_by_condition.png"):
            charts[name] = image_url(flow / name, max_size=(1200, 700), fmt="PNG")
    return {"title": "ChronoMagic-150 · LoT-Pass × CogVideoX", "seed": 42, "frames": 49, "fps": 8,
            "samples": sample_data, "condition_summary": averages, "correlations": corr,
            "charts": charts, "report": report}


TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ChronoMagic-150 results</title>
<style>__SHARED_STYLE__
.report{white-space:pre-wrap;overflow-wrap:anywhere}
#plots{grid-template-columns:repeat(2,minmax(0,1fr))}
#plots .figure-card{min-width:0}
@media(max-width:760px){#plots{grid-template-columns:minmax(0,1fr)}}
</style></head><body><header><h1 id="title"></h1><p class="subtitle">Paired first-frame I2V evaluation of global appearance transitions. Select raw video, source-anchored overlays, or the per-frame decoder heatmap overlay. Video previews and figures are embedded for offline viewing. RAFT aligns each clip to its generated first frame. Dashed curves in the explorer show aligned accuracy.</p><div class="stats" id="chips"></div></header>
<nav class="page-nav"><button id="overview-tab" aria-pressed="true">Overview</button><button id="explorer-tab" aria-pressed="false">Per-image explorer</button><button id="report-tab" aria-pressed="false">Report</button></nav>
<section id="overview" class="overview"><h2>Condition summary</h2><p class="muted">Independent unit: source image/pair; videos and frames are not independent samples.</p><div class="table-wrap"><table class="result-table"><thead><tr><th>Group</th><th>Condition</th><th>n sources</th><th>Mean video raw BA</th><th>Last-third BA</th><th>Stable-region ratio</th><th>Largest stable region</th><th>Decoder overlap</th></tr></thead><tbody id="summary"></tbody></table></div><div class="chart-card"><h3>Mean raw BA by frame</h3><canvas id="overview-chart" class="chart" height="340"></canvas></div><h2 style="margin-top:22px">Paired and stability analyses</h2><div id="plots" class="figure-grid"></div><h2 style="margin-top:22px">Paired stability correlations</h2><div class="table-wrap"><table class="result-table"><thead><tr><th>Metric</th><th>n pairs</th><th>Pearson r (p)</th><th>Spearman ρ (p)</th></tr></thead><tbody id="corr"></tbody></table></div></section>
<section id="explorer" class="explorer hidden"><div class="explorer-layout"><aside class="sidebar"><strong>Selected sources</strong><p class="muted">16 samples · see IDs and prompts</p><input class="search" id="search" placeholder="Filter ID or subcategory"><div id="samples" class="sample-list"></div></aside><main id="detail"></main></div></section>
<section id="report-section" class="overview hidden"><h2>Final report</h2><div id="report" class="report"></div></section><footer class="footer">ChronoMagic-Bench-150-derived first-frame I2V adaptation · CogVideoX-5b-I2V · seed 42 · 720×480 · 49 frames · 8 FPS. Decoder heatmaps show attribution sensitivity, not the physical location of watermark bits.</footer>
<script>
const DATA=__DATA__;const COLORS={transition:'#cf6c35',control:'#287c9a'};document.title=DATA.title;document.getElementById('title').textContent=DATA.title;
const chips=document.getElementById('chips');for(const x of [`${DATA.samples.length} selected sources`,`32 paired videos`,`${DATA.frames} frames · ${DATA.fps} FPS`,`raw and aligned decoding`,`offline embedded previews`]){let s=document.createElement('span');s.className='stat';s.textContent=x;chips.appendChild(s)}
function fmt(v,d=4){if(v===null||v===undefined||v==='')return '—';return Number.isFinite(Number(v))?Number(v).toFixed(d):String(v)}
function chart(canvas,series){const r=canvas.getBoundingClientRect(),d=window.devicePixelRatio||1,w=Math.max(320,r.width),h=Math.max(240,r.height||340);canvas.width=w*d;canvas.height=h*d;const c=canvas.getContext('2d');c.scale(d,d);const L=48,R=14,T=12,B=34,CW=w-L-R,CH=h-T-B;c.font='11px system-ui';c.textAlign='right';c.textBaseline='middle';for(let j=0;j<=4;j++){let v=j/4,y=T+CH*(1-v);c.strokeStyle='#e3e9ee';c.beginPath();c.moveTo(L,y);c.lineTo(w-R,y);c.stroke();c.fillStyle='#627584';c.fillText(v.toFixed(2),L-6,y)}c.strokeStyle='#536b7a';c.beginPath();c.moveTo(L,T);c.lineTo(L,T+CH);c.lineTo(w-R,T+CH);c.stroke();const n=Math.max(1,...series.map(s=>s.values.length));for(let i of [0,Math.floor((n-1)/2),n-1]){let x=L+CW*i/Math.max(1,n-1);c.fillStyle='#627584';c.textAlign='center';c.fillText(String(i),x,T+CH+14)}c.fillStyle='#31495b';c.fillText('Frame index',L+CW/2,h-5);for(const s of series){c.beginPath();c.strokeStyle=s.color;c.lineWidth=2;c.setLineDash(s.dash?[6,4]:[]);let open=false;s.values.forEach((v,i)=>{if(v===null||!Number.isFinite(Number(v))){open=false;return}let x=L+CW*i/Math.max(1,s.values.length-1),y=T+CH*(1-Number(v));if(!open){c.moveTo(x,y);open=true}else c.lineTo(x,y)});c.stroke();c.setLineDash([])}}
function curveControls(canvas, series){
  const controls=document.createElement('div');
  controls.className='curve-controls';
  const title=document.createElement('strong');
  title.textContent='Accuracy curves';
  controls.appendChild(title);
  const inputs=series.map(line=>{
    const label=document.createElement('label');
    const checkbox=document.createElement('input');
    checkbox.type='checkbox';
    checkbox.checked=true;
    label.append(checkbox,document.createTextNode(line.label));
    controls.appendChild(label);
    return checkbox;
  });
  const redraw=()=>chart(canvas,series.filter((_,i)=>inputs[i].checked));
  for(const input of inputs)input.addEventListener('change',redraw);
  for(const [text,checked] of [['Select all',true],['Deselect all',false]]){
    const button=document.createElement('button');
    button.type='button';
    button.textContent=text;
    button.onclick=()=>{for(const input of inputs)input.checked=checked;redraw()};
    controls.appendChild(button);
  }
  canvas.before(controls);
  redraw();
}
function overview(){const values=[];for(const group of ['all'])for(const cond of ['transition','control']){const r=DATA.trajectories[group]?.[cond]||[];values.push({values:r,color:COLORS[cond],label:cond})}chart(document.getElementById('overview-chart'),values)}
for(const row of DATA.condition_summary){let tr=document.getElementById('summary').insertRow();for(const x of [row.analysis_group,row.condition,row.n_independent_sources,fmt(row.mean_video_ba),fmt(row.mean_last_third_ba),fmt(row.mean_stable_region_ratio),fmt(row.mean_largest_stable_region_ratio),fmt(row.mean_decoder_stable_overlap)])tr.insertCell().textContent=x}
for(const [file,label] of [['raw_ba_trajectories.png','Raw bit accuracy by frame'],['paired_delta_mean_ba.png','Transition minus control per source'],['ba_vs_stability.png','Raw BA and stable regions'],['ba_vs_heatmap_overlap.png','Raw BA and decoder/stable overlap'],['raw_vs_aligned_acc.png','Raw vs aligned mean accuracy'],['acc_curve_raw_vs_aligned_by_condition.png','Raw and aligned accuracy by frame']]){if(!DATA.charts[file])continue;let card=document.createElement('article');card.className='figure-card';let h=document.createElement('h3');h.textContent=label;let im=document.createElement('img');im.src=DATA.charts[file]||'';im.alt=label;card.append(h,im);document.getElementById('plots').appendChild(card)}
for(const r of DATA.correlations){let tr=document.getElementById('corr').insertRow();for(const x of [r.metric,r.n,fmt(r.pearson_r,3)+' ('+fmt(r.pearson_p,3)+')',fmt(r.spearman_rho,3)+' ('+fmt(r.spearman_p,3)+')'])tr.insertCell().textContent=x}
document.getElementById('report').textContent=DATA.report;
const list=document.getElementById('samples'),detail=document.getElementById('detail'),search=document.getElementById('search');let selected=DATA.samples[0]?.sample_id;
function renderList(){list.replaceChildren();let q=search.value.trim().toLowerCase();for(const s of DATA.samples){if(q&&!`${s.sample_id} ${s.sub_category}`.toLowerCase().includes(q))continue;let b=document.createElement('button');b.className='sample-option';b.setAttribute('aria-pressed',String(selected===s.sample_id));let a=document.createElement('strong');a.textContent=`${String(s.selection_index).padStart(2,'0')} · ${s.analysis_group}`;let z=document.createElement('small');z.textContent=`${s.sample_id} · ${s.sub_category}`;b.append(a,z);b.onclick=()=>{selected=s.sample_id;renderList();renderDetail()};list.appendChild(b)}}
function createCase(c){let box=document.createElement('article');box.className='condition-card';let h=document.createElement('h3');h.textContent=c.condition==='transition'?'Appearance transition':'No-transition control';let sel=document.createElement('select');for(const [k,v] of [['raw_video_url','Raw video'],['similarity_video_url','Similarity overlay'],['stable_mask_video_url','Stable-mask overlay'],['decoder_heatmap_video_url','Decoder heatmap overlay'],['aligned_video_url','RAFT-aligned video'],['aligned_heatmap_video_url','RAFT-aligned decoder heatmap']]){if(!c.video_urls[k])continue;let o=document.createElement('option');o.value=k;o.textContent=v;sel.appendChild(o)}sel.value='decoder_heatmap_video_url';let vid=document.createElement('video');vid.className='sample-video';vid.controls=true;vid.playsInline=true;vid.preload='metadata';let err=document.createElement('p');err.className='muted';function upd(){err.textContent='';vid.src=c.video_urls[sel.value]||'';vid.load()}sel.onchange=upd;vid.onerror=()=>err.textContent='This embedded preview could not be played by this browser.';upd();box.append(h,sel,vid,err);let table=document.createElement('table');table.className='result-table';for(const [k,v] of [['Mean raw BA',c.mean_ba],['Aligned mean BA (excl. detected copy)',c.aligned_mean],['Aligned minus raw BA (matched frames)',c.alignment_delta],['Last-third BA',c.last_third_ba],['Minimum BA',c.minimum_ba],['Stable-region ratio',c.mean_srr],['Largest stable region',c.mean_largest_srr],['Encoder/stable overlap',c.encoder_stable_overlap],['Decoder/stable overlap',c.decoder_stable_overlap],['Transition review status',c.validation_status],['Final luminance change',c.mean_luminance_delta],['Changed area fraction',c.changed_area]]){let tr=table.insertRow();tr.insertCell().textContent=k;tr.insertCell().textContent=fmt(v)}box.appendChild(table);return box}
function renderDetail(){const s=DATA.samples.find(x=>x.sample_id===selected);if(!s)return;detail.replaceChildren();let card=document.createElement('article');card.className='sample-card';let intro=document.createElement('div');intro.className='card-head';let original=document.createElement('img');original.className='card-image';original.src=s.source_image_url;original.alt='Official source first frame';let meta=document.createElement('div');let title=document.createElement('h2');title.textContent=`${String(s.selection_index).padStart(2,'0')} · ${s.sample_id}`;let cat=document.createElement('p');cat.className='muted';cat.textContent=`${s.analysis_group} · ${s.sub_category} · ${s.project_label}`;let reason=document.createElement('p');reason.textContent=s.selection_reason;meta.append(title,cat,reason);let heat=document.createElement('img');heat.className='heatmap-image';heat.src=s.encoder_heatmap_url;heat.alt='Encoder residual heatmap';intro.append(original,meta,heat);card.appendChild(intro);let cases=document.createElement('div');cases.className='condition-grid';for(const c of s.conditions)cases.appendChild(createCase(c));card.appendChild(cases);let canvas=document.createElement('canvas');canvas.className='chart';canvas.height=320;canvas.style.padding='12px';let series=[];for(const c of s.conditions){series.push({values:c.raw_curve,color:COLORS[c.condition],label:c.condition+' · raw'});if(c.aligned_curve)series.push({values:c.aligned_curve,color:COLORS[c.condition],label:c.condition+' · RAFT-aligned',dash:true});}card.appendChild(canvas);detail.appendChild(card);curveControls(canvas,series);for(const c of s.conditions){let d=document.createElement('details');d.className='prompt-section';let summary=document.createElement('summary');summary.textContent=`${c.condition==='transition'?'Transition':'No-transition'} prompt · ${c.validation_status}`;let p=document.createElement('p');p.textContent=c.prompt;d.append(summary,p);detail.appendChild(d)}let citation=document.createElement('details');citation.className='prompt-section';let summary=document.createElement('summary');summary.textContent='Official ChronoMagic caption';let p=document.createElement('p');p.textContent=s.official_caption;citation.append(summary,p);detail.appendChild(citation)}
search.addEventListener('input',renderList);
function tab(which){for(const x of ['overview','explorer','report-section'])document.getElementById(x).classList.toggle('hidden',x!==which);for(const [id,name] of [['overview-tab','overview'],['explorer-tab','explorer'],['report-tab','report-section']])document.getElementById(id).setAttribute('aria-pressed',String(name===which));if(which==='overview')overview();if(which==='explorer'){renderList();renderDetail()}}
document.getElementById('overview-tab').onclick=()=>tab('overview');document.getElementById('explorer-tab').onclick=()=>tab('explorer');document.getElementById('report-tab').onclick=()=>tab('report-section');overview();
</script></body></html>'''


def main() -> None:
    data = build_data()
    # Turn per-frame curves and summary strings into finite JS-friendly values.
    trajectory_rows = read_rows(ANALYSIS / "paired_frame_trajectories.csv")
    trajectories = {}
    for group in ("all", "illumination", "weather_visibility"):
        trajectories[group] = {}
        for condition in ("transition", "control"):
            trajectories[group][condition] = [numeric(r["raw_bit_accuracy_mean"]) for r in trajectory_rows
                                                if r["analysis_group"] == group and r["condition"] == condition]
    data["trajectories"] = trajectories
    for row in data["condition_summary"]:
        for field in ("n_independent_sources", "mean_video_ba", "mean_last_third_ba", "mean_stable_region_ratio", "mean_largest_stable_region_ratio", "mean_decoder_stable_overlap"):
            row[field] = int(float(row[field])) if field == "n_independent_sources" else numeric(row[field])
    for row in data["correlations"]:
        row["n"] = int(float(row["n"]))
        for field in ("pearson_r", "pearson_p", "spearman_rho", "spearman_p"):
            row[field] = numeric(row[field])
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    claw_builder = ROOT / "claw_lotpass_pilot/scripts/build_dashboard.py"
    shared_style = claw_builder.read_text(encoding="utf-8").split("<style>", 1)[1].split("</style>", 1)[0]
    html = TEMPLATE.replace("__SHARED_STYLE__", shared_style).replace("__DATA__", payload)
    for output in OUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(f".{output.name}.partial")
        partial.write_text(html, encoding="utf-8")
        os.replace(partial, output)
        print(f"Wrote {output.relative_to(ROOT)} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
