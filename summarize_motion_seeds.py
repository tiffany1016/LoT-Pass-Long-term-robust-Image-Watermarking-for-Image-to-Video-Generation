#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PROMPTS = ("s1_crouch", "s1_run_exit", "s1_walk")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_accuracy(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        values = data["CogVideoX"]
        filenames = data["CogVideoX_filenames"].tolist()
    if values.ndim != 2 or values.shape[0] != len(filenames):
        raise ValueError(f"Unexpected metric shape in {path}: {values.shape}")
    result = {Path(name).stem.removesuffix("_aligned"): values[i] for i, name in enumerate(filenames)}
    missing = set(PROMPTS) - result.keys()
    if missing:
        raise ValueError(f"{path} is missing prompts: {sorted(missing)}")
    return result


def summarize(root: Path) -> tuple[Path, Path]:
    runs = {42: root / "Motion_CogVideoX"}
    runs.update({seed: root / f"Motion_CogVideoX_seed{seed}" for seed in (43, 44)})
    per_frame_path = root / "Motion_CogVideoX_seed_per_frame.csv"
    summary_path = root / "Motion_CogVideoX_seed_summary.csv"
    frame_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    source_path = Path("data/motion_sources/s1.png")
    prompt_path = Path("data/video_prompts.json")
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))

    for seed, run_dir in runs.items():
        raw = load_accuracy(run_dir / "video_acc_map.npz")
        aligned = load_accuracy(run_dir / "video_acc_map_optical.npz")
        if seed != 42:
            message_map_path = run_dir / "message_map.npz"
            with np.load(message_map_path, allow_pickle=True) as message_map:
                checkpoint = str(message_map["ckpt"].item())
            metadata = {
                "run_id": f"motion_s1_seed{seed}",
                "generation_seed": seed,
                "source_image": str(source_path),
                "source_image_sha256": sha256(source_path),
                "input_state": "watermarked",
                "watermark_checkpoint": checkpoint,
                "message_map_sha256": sha256(message_map_path),
                "encoded_images": {
                    prompt: {
                        "path": str(run_dir / "Encoded" / f"{prompt}.png"),
                        "sha256": sha256(run_dir / "Encoded" / f"{prompt}.png"),
                    }
                    for prompt in PROMPTS
                },
                "prompts": {prompt: prompts[prompt] for prompt in PROMPTS},
                "model_id": "THUDM/CogVideoX-5b-I2V",
                "resolution": "480p (720x480)",
                "num_frames": 49,
                "num_inference_steps": 50,
                "guidance_scale": 5.0,
                "fps": 8,
                "selected_physical_gpu": 6,
                "decode": {
                    "checkpoint": checkpoint,
                    "raw_metrics": str(run_dir / "video_acc_map.npz"),
                    "raft_aligned_metrics": str(run_dir / "video_acc_map_optical.npz"),
                },
                "location_maps": {
                    "method": "decoder gradient saliency",
                    "raw": str(run_dir / "Gradient_Saliency"),
                    "raft_aligned": str(run_dir / "Gradient_Saliency_Aligned"),
                },
            }
            (run_dir / "experiment_metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        for prompt in PROMPTS:
            if raw[prompt].shape != aligned[prompt].shape:
                raise ValueError(f"Raw/aligned frame counts differ for seed={seed}, prompt={prompt}")
            raw_mean = float(raw[prompt].mean())
            aligned_mean = float(aligned[prompt].mean())
            summary_rows.append(
                {
                    "seed": seed,
                    "prompt_id": prompt,
                    "frames": int(raw[prompt].size),
                    "raw_mean_bit_accuracy": raw_mean,
                    "raft_mean_bit_accuracy": aligned_mean,
                    "raft_minus_raw": aligned_mean - raw_mean,
                }
            )
            for frame_index, (raw_value, aligned_value) in enumerate(zip(raw[prompt], aligned[prompt])):
                frame_rows.append(
                    {
                        "seed": seed,
                        "prompt_id": prompt,
                        "frame_index": frame_index,
                        "raw_bit_accuracy": float(raw_value),
                        "raft_bit_accuracy": float(aligned_value),
                    }
                )

        all_raw = np.concatenate([raw[prompt] for prompt in PROMPTS])
        all_aligned = np.concatenate([aligned[prompt] for prompt in PROMPTS])
        raw_mean = float(all_raw.mean())
        aligned_mean = float(all_aligned.mean())
        summary_rows.append(
            {
                "seed": seed,
                "prompt_id": "ALL",
                "frames": int(all_raw.size),
                "raw_mean_bit_accuracy": raw_mean,
                "raft_mean_bit_accuracy": aligned_mean,
                "raft_minus_raw": aligned_mean - raw_mean,
            }
        )

    with per_frame_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=frame_rows[0].keys())
        writer.writeheader()
        writer.writerows(frame_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    return per_frame_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("Validation_Results"))
    args = parser.parse_args()
    per_frame_path, summary_path = summarize(args.root)
    print(f"per_frame={per_frame_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
