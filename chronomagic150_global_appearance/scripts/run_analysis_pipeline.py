#!/usr/bin/env python3
"""Run ChronoMagic post-generation analysis, checking GPU availability before each GPU stage."""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "chronomagic150_global_appearance"
VENV_PYTHON = ROOT / "vbench_lotpass_pilot/.venv/bin/python"


def free_gpu_zero() -> None:
    while True:
        result = subprocess.run(["nvidia-smi"], cwd=ROOT, text=True, capture_output=True, check=True)
        output = result.stdout
        if "No running processes found" in output:
            return
        print("GPU 0 is occupied; waiting 60 seconds per the device policy.", flush=True)
        time.sleep(60)


def run(script: str, gpu: bool = False) -> None:
    if gpu:
        free_gpu_zero()
    command = ["uv", "run", "--python", str(VENV_PYTHON), "python", str(EXP / "scripts" / script)]
    env = os.environ.copy()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    log_path = EXP / "logs" / f"{Path(script).stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"{script} failed (exit {result.returncode}); last log lines:\n{tail}")


def main() -> None:
    manifest = list(csv.DictReader((EXP / "dataset/chronomagic150_generation_manifest.csv").open(encoding="utf-8")))
    if len(manifest) != 32:
        raise RuntimeError(f"Expected 32 approved cases, found {len(manifest)}")
    incomplete = []
    for row in manifest:
        status_path = EXP / "runs" / row["sample_id"] / row["condition"] / "run_status.json"
        if not status_path.is_file() or '"status": "completed"' not in status_path.read_text(encoding="utf-8"):
            incomplete.append(row["case_id"])
    if incomplete:
        raise RuntimeError("Paired generation is incomplete; do not analyze a partial dataset: " + ", ".join(incomplete))
    run("decode_raw.py", gpu=True)
    run("stable_regions.py")
    run("encoder_heatmaps.py")
    run("decoder_heatmaps.py", gpu=True)
    run("summarize_analysis.py")
    print("Raw decoding, source-anchored stability, attribution overlap, and quantitative summaries are complete.", flush=True)
    print("Inspect analysis/transition_review/*.jpg and finalize analysis/transition_validation.csv before the final report/dashboard refresh.", flush=True)


if __name__ == "__main__":
    main()
