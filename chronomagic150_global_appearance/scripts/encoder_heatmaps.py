#!/usr/bin/env python3
"""Run the repository's encoder-residual heatmap implementation on sources."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "chronomagic150_global_appearance"
DATA = EXP / "dataset"
WORK = EXP / "watermarked_sources/encoding_work"
OUT = EXP / "heatmaps/encoder"


def main() -> None:
    rows = list(csv.DictReader((DATA / "chronomagic150_selected16.csv").open(encoding="utf-8")))
    expected = {f"CM{int(row['selection_index']):03d}": row["sample_id"] for row in rows}
    OUT.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / "encoder_heatmap.py"),
               "--pair", f"{WORK / 'source_images'}={WORK / 'Encoded'}",
               "--out-dir", str(OUT)]
    subprocess.run(command, cwd=ROOT, check=True)
    raw = OUT / "raw_maps/source_images"
    missing = sorted(alias for alias in expected if not (raw / f"{alias}.npy").is_file())
    if missing:
        raise RuntimeError(f"Existing encoder heatmap script missed aliases: {missing}")
    (OUT / "source_id_map.json").write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Existing encoder heatmap method processed {len(expected)} approved sources.", flush=True)


if __name__ == "__main__":
    main()
