#!/usr/bin/env python3
"""Plot raw and RAFT-aligned bit accuracy for the large-motion experiment."""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


IMAGE_HEADING = re.compile(r"^### `([^`]+)`$")
PROMPT_HEADING = re.compile(r"^\*\*([A-Z0-9]+-[A-Z]) — (.+)\*\*$")


def load_prompt_catalog(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Parse prompt metadata, keyed by (source image, canonical motion ID)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    source_image: str | None = None
    prompts: dict[tuple[str, str], dict[str, str]] = {}

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        image_match = IMAGE_HEADING.match(line)
        if image_match:
            source_image = image_match.group(1)
            index += 1
            continue

        prompt_match = PROMPT_HEADING.match(line)
        if not prompt_match or source_image is None:
            index += 1
            continue

        prompt_id, title = prompt_match.groups()
        body_start = None
        next_index = index + 1
        while next_index < len(lines):
            candidate = lines[next_index].strip()
            if IMAGE_HEADING.match(candidate) or PROMPT_HEADING.match(candidate):
                break
            if candidate == "```text":
                body_start = next_index + 1
                break
            next_index += 1

        if body_start is None:
            # Incomplete optional entries (currently C2-C) have no description.
            index = next_index
            continue

        body_end = body_start
        while body_end < len(lines) and not lines[body_end].strip().startswith("``"):
            body_end += 1
        if body_end == len(lines):
            raise ValueError(f"Unclosed prompt block for {source_image} / {prompt_id}")
        description = "\n".join(lines[body_start:body_end]).strip()

        # The catalog accidentally labels thing2's T2-C prompt as T1-C.
        canonical_id = "T2-C" if source_image == "thing2.png" and prompt_id == "T1-C" else prompt_id
        key = (source_image, canonical_id)
        if key in prompts:
            raise ValueError(f"Duplicate prompt metadata for {source_image} / {canonical_id}")
        prompts[key] = {"motion_id": canonical_id, "title": title, "description": description}
        index = body_end + 1

    return prompts


def load_video_metadata(run_dir: Path, prompt_catalog: Path, names: list[str]) -> dict[str, dict[str, str]]:
    with (run_dir / "prompt_manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    catalog = load_prompt_catalog(prompt_catalog)
    manifest_by_pair = {row["pair_id"]: row for row in manifest}
    if len(manifest_by_pair) != len(manifest):
        raise ValueError("Prompt manifest contains duplicate pair IDs")

    metadata: dict[str, dict[str, str]] = {}
    for filename in names:
        pair_id = Path(filename).stem
        row = manifest_by_pair.get(pair_id)
        if row is None:
            raise ValueError(f"No prompt manifest entry for video {filename}")
        key = (row["source_image"], row["prompt_id"])
        prompt = catalog.get(key)
        if prompt is None:
            raise ValueError(f"No prompt catalog entry for {row['source_image']} / {row['prompt_id']}")
        if prompt["description"] != row["prompt"].strip():
            raise ValueError(f"Manifest/catalog prompt text differs for {pair_id}")
        metadata[filename] = prompt
    return metadata


def html_json(value: object) -> str:
    """Serialize JSON safely for an inline script element."""
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def interactive_html(
    names: list[str],
    curves: dict[str, np.ndarray],
    metadata: dict[str, dict[str, str]],
    title: str,
    output_path: Path,
    combined: bool = False,
    aligned_curves: dict[str, np.ndarray] | None = None,
) -> None:
    if combined and aligned_curves is None:
        raise ValueError("Combined graph requires aligned curves")
    series = {
        name: {
            "values": [float(value) for value in curves[name]],
            "after_values": [float(value) for value in aligned_curves[name]] if aligned_curves is not None else None,
            "motion_id": metadata[name]["motion_id"],
            "description": metadata[name]["description"],
            "color": plt.get_cmap("turbo")(index / max(len(names) - 1, 1))[:3],
        }
        for index, name in enumerate(names)
    }
    for item in series.values():
        item["color"] = "#" + "".join(f"{round(channel * 255):02x}" for channel in item["color"])

    data = {
        "title": title,
        "combined": combined,
        "series": series,
        "frame_count": len(next(iter(curves.values()))),
    }
    script_data = html_json(data)
    page = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root { color-scheme: light; font: 14px/1.4 system-ui, sans-serif; color: #1d2733; }
body { margin: 0; padding: 20px; background: #fff; }
h1 { margin: 0 0 12px; font-size: 1.35rem; }
.layout { display: grid; grid-template-columns: minmax(0, 1fr) 270px; gap: 18px; align-items: start; }
.chart-wrap { min-width: 0; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
.axis-label { fill: #25313d; font-size: 13px; }
.tick-label { fill: #4c5864; font-size: 11px; }
.grid { stroke: #dce2e8; stroke-width: 1; }
.chance { stroke: #777; stroke-width: 1.4; stroke-dasharray: 2 5; }
.trace { fill: none; stroke-width: 1.65; opacity: .86; }
.trace.after { stroke-dasharray: 6 4; }
.controls { border: 1px solid #ccd3da; border-radius: 6px; padding: 12px; }
.controls h2 { margin: 0 0 8px; font-size: 1rem; }
.line-key { margin: 0 0 10px; color: #4c5864; font-size: .88rem; }
.line-key .solid { color: #24364b; }
.line-key .dashed { color: #24364b; border-bottom: 2px dashed #24364b; }
.selection-actions { display: flex; gap: 8px; margin: 0 0 10px; }
.selection-actions button { padding: 5px 9px; border: 1px solid #aeb8c2; border-radius: 4px; background: #f7f9fb; color: inherit; cursor: pointer; }
.selection-actions button:hover { background: #edf2f6; }
.series-list { max-height: 68vh; overflow: auto; display: grid; gap: 4px; }
.series-toggle { display: grid; grid-template-columns: 16px 12px minmax(0, 1fr); align-items: center; gap: 7px; cursor: pointer; font: 12px/1.3 ui-monospace, monospace; overflow-wrap: anywhere; }
.series-toggle input { margin: 0; }
.swatch { width: 12px; height: 3px; display: inline-block; }
#tooltip { position: fixed; display: none; z-index: 10; max-width: min(440px, calc(100vw - 24px)); max-height: 65vh; overflow: auto; padding: 10px 12px; border: 1px solid #43505d; border-radius: 5px; background: #fff; box-shadow: 0 3px 14px #0003; white-space: pre-wrap; overflow-wrap: anywhere; pointer-events: none; font-size: 12px; }
.point { fill: transparent; stroke: transparent; pointer-events: all; }
.point:focus { stroke: #111; stroke-width: 1.5; outline: none; }
@media (max-width: 850px) { .layout { grid-template-columns: 1fr; } .series-list { max-height: 240px; } }
</style>
</head>
<body>
<h1 id="page-title"></h1>
<div class="layout">
  <main class="chart-wrap"><svg id="chart" role="img" aria-label="Per-video watermark bit accuracy by frame"></svg></main>
  <aside class="controls">
    <h2>Videos</h2>
    <p class="line-key" id="line-key"></p>
    <div class="selection-actions">
      <button type="button" id="select-all">Select all</button>
      <button type="button" id="deselect-all">Deselect all</button>
    </div>
    <div class="series-list" id="series-list"></div>
  </aside>
</div>
<div id="tooltip" role="status" aria-live="polite"></div>
<script>
const DATA = __DATA__;
document.title = DATA.title;
document.getElementById("page-title").textContent = DATA.title;
document.getElementById("line-key").textContent = DATA.combined
  ? "Solid: before alignment · dashed: after RAFT alignment"
  : (DATA.title.includes("after") ? "After RAFT alignment" : "Before alignment");

const svg = document.getElementById("chart");
const NS = "http://www.w3.org/2000/svg";
const W = 1120, H = 650, M = {left: 72, right: 22, top: 22, bottom: 62};
const plotW = W - M.left - M.right, plotH = H - M.top - M.bottom;
const x = frame => M.left + frame * plotW / Math.max(DATA.frame_count - 1, 1);
const y = accuracy => M.top + (1 - accuracy) * plotH;
svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
svg.setAttribute("preserveAspectRatio", "xMinYMin meet");

function node(tag, attrs = {}, text = null) {
  const element = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  if (text !== null) element.textContent = text;
  return element;
}

for (let tick = 0; tick <= 5; tick++) {
  const value = tick / 5;
  const yy = y(value);
  svg.append(node("line", {x1: M.left, x2: W - M.right, y1: yy, y2: yy, class: value === .5 ? "chance" : "grid"}));
  svg.append(node("text", {x: M.left - 10, y: yy + 4, "text-anchor": "end", class: "tick-label"}, value.toFixed(1)));
}
const frameTicks = [...new Set([0, Math.round((DATA.frame_count - 1) / 4), Math.round((DATA.frame_count - 1) / 2), Math.round(3 * (DATA.frame_count - 1) / 4), DATA.frame_count - 1])];
for (const frame of frameTicks) {
  const xx = x(frame);
  svg.append(node("line", {x1: xx, x2: xx, y1: M.top, y2: H - M.bottom, class: "grid"}));
  svg.append(node("text", {x: xx, y: H - M.bottom + 20, "text-anchor": "middle", class: "tick-label"}, String(frame)));
}
svg.append(node("text", {x: M.left + plotW / 2, y: H - 12, "text-anchor": "middle", class: "axis-label"}, "Frame index"));
const yLabel = node("text", {x: 18, y: M.top + plotH / 2, transform: `rotate(-90 18 ${M.top + plotH / 2})`, "text-anchor": "middle", class: "axis-label"}, "Watermark bit accuracy");
svg.append(yLabel);

const tooltip = document.getElementById("tooltip");
const list = document.getElementById("series-list");
const checkboxes = [];
function setAllVisible(visible) {
  for (const checkbox of checkboxes) {
    checkbox.checked = visible;
    checkbox.dispatchEvent(new Event("change"));
  }
}
document.getElementById("select-all").addEventListener("click", () => setAllVisible(true));
document.getElementById("deselect-all").addEventListener("click", () => setAllVisible(false));
function showTooltip(point, event) {
  const name = point.dataset.filename;
  const item = DATA.series[name];
  const state = point.dataset.state;
  const frame = Number(point.dataset.frame);
  const accuracy = Number(point.dataset.accuracy);
  tooltip.textContent = `Filename: ${name}\nMotion ID: ${item.motion_id}\nDescription: ${item.description}\nAlignment: ${state}\nFrame index: ${frame}\nBit accuracy: ${accuracy.toFixed(4)}`;
  tooltip.style.display = "block";
  const left = event && Number.isFinite(event.clientX) ? event.clientX + 14 : 20;
  const top = event && Number.isFinite(event.clientY) ? event.clientY + 14 : 20;
  tooltip.style.left = `${Math.min(left, window.innerWidth - 460)}px`;
  tooltip.style.top = `${Math.min(top, window.innerHeight - 100)}px`;
}
function hideTooltip() { tooltip.style.display = "none"; }

for (const [filename, item] of Object.entries(DATA.series)) {
  const group = node("g", {"data-video": filename});
  const states = DATA.combined ? ["Before alignment", "After RAFT alignment"] : [DATA.title.includes("after") ? "After RAFT alignment" : "Before alignment"];
  for (const state of states) {
    const after = state.startsWith("After");
    const values = after && item.after_values ? item.after_values : item.values;
    const pathData = values.map((value, frame) => `${frame ? "L" : "M"}${x(frame).toFixed(2)},${y(value).toFixed(2)}`).join(" ");
    group.append(node("path", {d: pathData, stroke: item.color, class: `trace${after ? " after" : ""}`, "data-state": state}));
    values.forEach((value, frame) => {
      const point = node("circle", {cx: x(frame), cy: y(value), r: 6, class: "point", tabindex: "0", "data-filename": filename, "data-frame": frame, "data-state": state, "data-accuracy": value, "aria-label": `${filename}, ${item.motion_id}, ${state}, frame ${frame}, bit accuracy ${value.toFixed(4)}`});
      point.addEventListener("pointerenter", event => showTooltip(point, event));
      point.addEventListener("pointermove", event => showTooltip(point, event));
      point.addEventListener("pointerleave", hideTooltip);
      point.addEventListener("focus", event => showTooltip(point, event));
      point.addEventListener("blur", hideTooltip);
      group.append(point);
    });
  }
  svg.append(group);

  const label = document.createElement("label");
  label.className = "series-toggle";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = true;
  checkbox.setAttribute("aria-label", `Show or hide ${filename}`);
  checkbox.addEventListener("change", () => { group.style.display = checkbox.checked ? "" : "none"; });
  checkboxes.push(checkbox);
  const swatch = document.createElement("span");
  swatch.className = "swatch";
  swatch.style.backgroundColor = item.color;
  label.append(checkbox, swatch, document.createTextNode(filename));
  list.append(label);
}
</script>
</body>
</html>'''
    page = page.replace("__TITLE__", html.escape(title)).replace("__DATA__", script_data)
    output_path.write_text(page, encoding="utf-8")


def load_curves(run_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with (run_dir / "prompt_manifest.json").open(encoding="utf-8") as stream:
        manifest = __import__("json").load(stream)
    expected = {f"{row['pair_id']}.mp4" for row in manifest}

    with np.load(run_dir / "video_acc_map.npz", allow_pickle=True) as raw_npz:
        raw = np.asarray(raw_npz["CogVideoX"], dtype=float)
        raw_names = [Path(name).name for name in raw_npz["CogVideoX_filenames"].tolist()]
    with np.load(run_dir / "video_acc_map_optical.npz", allow_pickle=True) as aligned_npz:
        aligned = np.asarray(aligned_npz["CogVideoX"], dtype=float)
        aligned_names = [Path(name).name for name in aligned_npz["CogVideoX_filenames"].tolist()]

    raw_rows = {name: row for name, row in zip(raw_names, raw)}
    aligned_rows = {
        (Path(name).stem.removesuffix("_aligned") + Path(name).suffix): row
        for name, row in zip(aligned_names, aligned)
    }
    if set(raw_rows) != expected or set(aligned_rows) != expected:
        raise ValueError(
            f"Expected {len(expected)} videos; raw missing/extra="
            f"{sorted(expected-set(raw_rows))}/{sorted(set(raw_rows)-expected)}, aligned="
            f"{sorted(expected-set(aligned_rows))}/{sorted(set(aligned_rows)-expected)}"
        )
    names = sorted(expected)
    raw_curves = np.stack([raw_rows[name] for name in names])
    aligned_curves = np.stack([aligned_rows[name] for name in names])
    if raw_curves.shape != aligned_curves.shape:
        raise ValueError(f"Raw/aligned shapes differ: {raw_curves.shape} vs {aligned_curves.shape}")
    return names, raw_curves, aligned_curves


def make_axes(title: str, names: list[str], curves: np.ndarray):
    frames = np.arange(curves.shape[1])
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_axes([0.07, 0.31, 0.91, 0.63])
    colors = plt.get_cmap("turbo")(np.linspace(0, 1, len(names)))
    ax.set(
        title=title,
        xlabel="Frame index",
        ylabel="Watermark bit accuracy",
        xlim=(frames[0], frames[-1]),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.25)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.1)
    return fig, ax, frames, colors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("Validation_Results/Motion_Exp2_CogVideoX_seed42"))
    parser.add_argument(
        "--prompt-catalog",
        type=Path,
        default=Path(__file__).resolve().with_name("large_motion_exp2_prompt_list.md"),
    )
    parser.add_argument(
        "--interactive-only",
        action="store_true",
        help="Write the HTML graphs without regenerating existing PNGs or the CSV.",
    )
    args = parser.parse_args()
    names, raw, aligned = load_curves(args.run_dir)
    metadata = load_video_metadata(args.run_dir, args.prompt_catalog, names)
    output = args.run_dir / "analysis_plots"
    output.mkdir(parents=True, exist_ok=True)

    raw_by_name = dict(zip(names, raw))
    aligned_by_name = dict(zip(names, aligned))
    interactive_html(
        names,
        raw_by_name,
        metadata,
        "Watermark bit accuracy before RAFT alignment",
        output / "before_alignment_interactive.html",
    )
    interactive_html(
        names,
        aligned_by_name,
        metadata,
        "Watermark bit accuracy after RAFT alignment",
        output / "after_alignment_interactive.html",
    )
    interactive_html(
        names,
        raw_by_name,
        metadata,
        "Watermark bit accuracy before and after RAFT alignment",
        output / "before_after_alignment_interactive.html",
        combined=True,
        aligned_curves=aligned_by_name,
    )

    if not args.interactive_only:
        fig, ax, frames, colors = make_axes("Watermark bit accuracy before RAFT alignment", names, raw)
        for name, curve, color in zip(names, raw, colors):
            ax.plot(frames, curve, color=color, linewidth=1.25, alpha=0.88, label=name)
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.015),
                   ncol=4, fontsize=7, frameon=True)
        fig.savefig(output / "before_alignment.png", dpi=180)
        plt.close(fig)

        fig, ax, frames, colors = make_axes("Watermark bit accuracy after RAFT alignment", names, aligned)
        for name, curve, color in zip(names, aligned, colors):
            ax.plot(frames, curve, color=color, linewidth=1.25, alpha=0.88, label=name)
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.015),
                   ncol=4, fontsize=7, frameon=True)
        fig.savefig(output / "after_alignment.png", dpi=180)
        plt.close(fig)

        fig, ax, frames, colors = make_axes("Watermark bit accuracy before and after RAFT alignment", names, raw)
        for curve_raw, curve_aligned, color in zip(raw, aligned, colors):
            ax.plot(frames, curve_raw, color=color, linewidth=1.2, alpha=0.88)
            ax.plot(frames, curve_aligned, color=color, linewidth=1.2, linestyle="--", alpha=0.88)
        ax.legend(
            handles=[
                Line2D([0], [0], color="black", linestyle="-", label="Before alignment"),
                Line2D([0], [0], color="black", linestyle="--", label="After RAFT alignment"),
                Line2D([0], [0], color="gray", linestyle=":", label="Random guess (0.5)"),
            ],
            loc="upper right", fontsize=8,
        )
        video_handles = [Line2D([0], [0], color=color, linewidth=2, label=name) for name, color in zip(names, colors)]
        fig.legend(handles=video_handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
                   ncol=4, fontsize=7, frameon=True)
        fig.savefig(output / "before_after_alignment.png", dpi=180)
        plt.close(fig)

        with (output / "per_video_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("video_filename", "raw_mean_bit_accuracy", "raft_aligned_mean_bit_accuracy", "change"))
            for name, raw_curve, aligned_curve in zip(names, raw, aligned):
                raw_mean = float(raw_curve.mean())
                aligned_mean = float(aligned_curve.mean())
                writer.writerow((name, f"{raw_mean:.6f}", f"{aligned_mean:.6f}", f"{aligned_mean-raw_mean:+.6f}"))
    print(f"Saved three interactive per-video graphs for {len(names)} videos to {output}")


if __name__ == "__main__":
    main()
