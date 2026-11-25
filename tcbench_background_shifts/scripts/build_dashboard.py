#!/usr/bin/env python3
"""Build the standalone TC-Bench dashboard using the VBench dashboard as its template."""

import argparse
import base64
import csv
import json
import shutil
import statistics
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "tcbench_background_shifts"
ANALYSIS = EXPERIMENT / "analysis"
TEMPLATE = ROOT / "vbench_lotpass_pilot_dashboard.html"
OUTPUT = ROOT / "tcbench_lotpass_dashboard.html"
CONDITIONS = {"transition": "Transition", "control": "Control"}


def read_rows(path, key):
    with path.open(encoding="utf-8", newline="") as source:
        return {key(row): row for row in csv.DictReader(source)}


def clip_key(row):
    return row["case_id"], row["condition"]


def frame_key(row):
    return (*clip_key(row), int(row["frame_idx"]))


def data_url(path, mime):
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def encode_preview(ffmpeg, source, destination):
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-threads", "1", "-filter_threads", "1", "-i", str(source),
        "-vf", "scale=480:320:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos,pad=480:320:(ow-iw)/2:(oh-ih)/2,setsar=1",
        "-an", "-c:v", "libx264", "-threads", "1", "-preset", "veryfast", "-crf", "28",
        "-profile:v", "baseline", "-level", "3.0", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(destination),
    ], check=True)
    return data_url(destination, "video/mp4")


def build_data(ffmpeg):
    raw = read_rows(ANALYSIS / "per_video_raw_ba.csv", clip_key)
    aligned = read_rows(ANALYSIS / "optical_flow/per_video_aligned_ba.csv", clip_key)
    raw_frames = read_rows(ANALYSIS / "per_frame_raw_ba.csv", frame_key)
    aligned_frames = read_rows(ANALYSIS / "optical_flow/per_frame_aligned_ba.csv", frame_key)
    reviews = read_rows(ANALYSIS / "transition_validation.csv", clip_key)
    prompts = read_rows(EXPERIMENT / "approval/tcbench_all_36pairs.csv", clip_key)
    case_ids = sorted({case_id for case_id, _ in raw})
    samples, paired = [], []
    frames = int(next(iter(raw.values()))["frame_count"])
    fps = float(next(iter(raw.values()))["fps"])
    seed = int(next(iter(raw.values()))["seed"])

    with tempfile.TemporaryDirectory(dir=ANALYSIS, prefix=".dashboard-") as scratch:
        with ThreadPoolExecutor(max_workers=4) as pool:
            previews = []
            for index, case_id in enumerate(case_ids, 1):
                transition = prompts[case_id, "transition"]
                sample = {
                    "id": case_id, "selected_index": index,
                    "name": transition["prompt_used"], "category": "Background shifts",
                    "source_url": transition["reference_video_url"],
                    "image_data": data_url(ROOT / transition["source_image_model_ready_path"], "image/png"),
                    "motions": [], "review": [],
                }
                samples.append(sample)
                paired_row = [case_id]
                for condition, label in CONDITIONS.items():
                    key = case_id, condition
                    raw_row, aligned_row, review = raw[key], aligned[key], reviews[key]
                    motion = {
                        "motion": label,
                        "values": [float(raw_frames[*key, i]["ba"]) for i in range(frames)],
                        "aligned_values": [float(aligned_frames[*key, i]["aligned_BA"]) for i in range(frames)],
                        "raw_mean": float(raw_row["mean_ba_excluding_copy_frame"]),
                        "aligned_mean": float(aligned_row["aligned_mean_BA_excluding_copy_frame"]),
                    }
                    sample["motions"].append(motion)
                    for field, path in {
                        "raw_video_url": raw_row["video_path"],
                        "aligned_video_url": aligned_row["aligned_video"],
                        "heatmap_video_url": aligned_row["decoder_heatmap_overlay_video"],
                    }.items():
                        destination = Path(scratch) / f"{case_id}_{condition}_{field}.mp4"
                        previews.append((motion, field, pool.submit(
                            encode_preview, ffmpeg, ROOT / path, destination,
                        )))
                    sample["review"].append({
                        "condition": label, "prompt": prompts[key]["prompt_used"],
                        "notes": review["review_notes"],
                    })
                    paired_row.extend([motion["raw_mean"], motion["aligned_mean"]])
                review = reviews[case_id, "transition"]
                sample["badges"] = [
                    f"Transition validity: {review['transition_validity']}",
                    f"Global change: {review['observed_global_change']}",
                    f"Control leakage: {reviews[case_id, 'control']['control_transition_leakage']}",
                ]
                paired.append([*paired_row, review["transition_validity"]])

            for motion, field, future in previews:
                motion[field] = future.result()

    averages = []
    for index, label in enumerate(CONDITIONS.values()):
        clips = [sample["motions"][index] for sample in samples]
        raw_mean = statistics.mean(clip["raw_mean"] for clip in clips)
        aligned_mean = statistics.mean(clip["aligned_mean"] for clip in clips)
        averages.append({
            "motion": label, "video_count": len(clips), "total_video_frames": len(clips) * frames,
            "mean_acc_all_frames": raw_mean, "mean_aligned_acc_all_frames": aligned_mean,
            "aligned_minus_raw_acc": aligned_mean - raw_mean,
            "video_mean_sd": statistics.stdev(clip["raw_mean"] for clip in clips),
            "frame_means": [statistics.mean(column) for column in zip(*(clip["values"] for clip in clips))],
            "aligned_frame_means": [statistics.mean(column) for column in zip(*(clip["aligned_values"] for clip in clips))],
        })
    return {
        "title": "TC-Bench · LoT-Pass bit accuracy",
        "frames": frames, "fps": fps, "seed": seed, "samples": samples,
        "all_video_averages": averages, "paired": paired,
    }


def render(data):
    template = TEMPLATE.read_text(encoding="utf-8")
    start = template.index("const DATA = ")
    end = template.index("\nlet selectedSampleId", start)
    template = template[:start] + "const DATA = __TC_DATA__;" + template[end:]
    replacements = {
        "VBench++": "TC-Bench",
        "across all TC-Bench videos": "across official transition and paired control videos",
        "Per-source explorer": "Per-case explorer",
        "Source samples": "Source cases",
        "Camera setting": "Condition",
        "by camera setting": "by condition",
        "The table averages all 49 frames from each completed video.": (
            "Table means exclude a detected frame-0 conditioning copy. The chart shows all frames."
        ),
        "each video’s 49-frame mean; it describes variation among the 10 videos per setting.": (
            "each video’s mean after excluding a detected conditioning copy. "
            "It describes variation across the 18 videos per condition."
        ),
        "{static:'#17212b',pan:'#0782a0',tilt:'#d97706',zoom:'#609542'}": (
            "{Transition:'#0782a0',Control:'#d97706'}"
        ),
        "index*215": "index*245",
        "Choose motion and alignment view": "Choose condition and video view",
        "${motion.motion} (${motion.camera_setting}) · ${label}": "${motion.motion} · ${label}",
        "select.value=sample.motions.some(motion=>motion.motion==='static'&&motion.raw_video_url)?'static|raw':`${sample.motions[0]?.motion||'static'}|raw`;": (
            "select.selectedIndex=0;"
        ),
        "Motion curve selection": "Condition curve selection",
        "Open source image reference": "Open reference video",
        "Category: ${sample.category}`,`${sample.motions.length} motion conditions`": (
            "Category: ${sample.category}`,...sample.badges"
        ),
        "const showAverage=page==='average';": """const showAverage=page==='average';
  if (showAverage) explorerPage.querySelector('video')?.pause();""",
        "actions.appendChild(enlarge);panel.append(head,video,actions);": """const status=document.createElement('p');
  status.className='hint';
  status.setAttribute('role','status');
  source.addEventListener('error',()=>{
    status.textContent='This viewer could not play the embedded MP4. Open the HTML in a browser with H.264 support.';
  });
  actions.appendChild(enlarge);panel.append(head,video,actions,status);""",
        "source.src=motion?.[sourceKey]||'';": """status.textContent='';
    source.src=motion[sourceKey];""",
        "card.appendChild(graphWrap);results.appendChild(card);": """card.appendChild(graphWrap);
  const review=document.createElement('details');
  review.className='graph-wrap';
  const summary=document.createElement('summary');
  summary.textContent='Prompts and visual review';
  review.append(summary);
  for (const row of sample.review) {
    const prompt=document.createElement('p');
    const notes=document.createElement('p');
    prompt.textContent=`${row.condition} prompt: ${row.prompt}`;
    notes.textContent=row.notes;
    review.append(prompt,notes);
  }
  card.appendChild(review);
  results.appendChild(card);""",
    }
    for old, new in replacements.items():
        if old not in template:
            raise ValueError(f"VBench template text changed: {old}")
        template = template.replace(old, new)

    template = template.replace(
        '</section>\n<div class="layout',
        """  <h2>Paired case ACC</h2>
  <div class="average-table-wrap">
    <table class="average-table">
      <thead><tr><th>Case</th><th>Transition raw</th><th>Transition aligned</th>
        <th>Control raw</th><th>Control aligned</th><th>Transition validity</th></tr></thead>
      <tbody id="paired-table-body"></tbody>
    </table>
  </div>
</section>
<div class="layout""",
        1,
    )
    template = template.replace("drawAggregateChart();", """drawAggregateChart();
const pairedBody=document.getElementById('paired-table-body');
for (const row of DATA.paired) {
  const tr=pairedBody.insertRow();
  for (const value of row) {
    tr.insertCell().textContent=typeof value==='number'?value.toFixed(4):value;
  }
}""", 1)
    # Keep playback when the user toggles curves.
    template = template.replace("renderCards();});", "refreshCurves(sample);});")
    template = template.replace("function drawChart(sample, shown) {", """function refreshCurves(sample) {
  const card=results.querySelector('.sample-card');
  card.querySelector('.curve-controls').replaceWith(makeCurveControls(sample));
  card.querySelector('.chart').replaceWith(drawChart(sample,visibleCurves.get(sample.id)));
}
function drawChart(sample, shown) {""")
    return template.replace("__TC_DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"), help="Path to a CPU FFmpeg build")
    args = parser.parse_args()
    if args.ffmpeg is None:
        parser.error("FFmpeg is required. Pass --ffmpeg or add it to PATH.")
    html = render(build_data(str(Path(args.ffmpeg).resolve())))
    with tempfile.NamedTemporaryFile(dir=ROOT, prefix=".tcbench-dashboard-", delete=False) as staged:
        temporary_output = Path(staged.name)
        try:
            staged.write(html.encode("utf-8"))
            staged.flush()
            temporary_output.replace(OUTPUT)
        finally:
            temporary_output.unlink(missing_ok=True)
    print(f"Wrote {OUTPUT.name} ({len(html.encode('utf-8')) / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
