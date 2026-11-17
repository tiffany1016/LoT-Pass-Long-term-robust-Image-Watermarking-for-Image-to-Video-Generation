#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEEDS = (42, 43, 44)
PROMPTS = ("s1_crouch", "s1_run_exit", "s1_walk")


def load_video_curves(
    root: Path,
) -> tuple[dict[tuple[int, str], np.ndarray], dict[tuple[int, str], np.ndarray]]:
    raw_by_video: dict[tuple[int, str], np.ndarray] = {}
    aligned_by_video: dict[tuple[int, str], np.ndarray] = {}
    for seed in SEEDS:
        run_dir = root / ("Motion_CogVideoX" if seed == 42 else f"Motion_CogVideoX_seed{seed}")
        with np.load(run_dir / "video_acc_map.npz", allow_pickle=False) as raw_data:
            raw = np.asarray(raw_data["CogVideoX"], dtype=float)
            raw_names = [Path(name).name for name in raw_data["CogVideoX_filenames"].tolist()]
        with np.load(run_dir / "video_acc_map_optical.npz", allow_pickle=False) as aligned_data:
            aligned = np.asarray(aligned_data["CogVideoX"], dtype=float)
            aligned_names = [
                Path(name).stem.removesuffix("_aligned")
                for name in aligned_data["CogVideoX_filenames"].tolist()
            ]
        raw_rows = {Path(name).stem: (name, row) for name, row in zip(raw_names, raw)}
        aligned_rows = {name: row for name, row in zip(aligned_names, aligned)}
        if not set(PROMPTS).issubset(raw_rows) or not set(PROMPTS).issubset(aligned_rows):
            raise ValueError(f"Missing one or more s1 prompts for seed {seed}")
        for prompt in PROMPTS:
            filename, raw_curve = raw_rows[prompt]
            aligned_curve = aligned_rows[prompt]
            if raw_curve.shape != aligned_curve.shape:
                raise ValueError(f"Raw/aligned frame counts differ for seed {seed}, {filename}")
            key = (seed, filename)
            raw_by_video[key] = raw_curve
            aligned_by_video[key] = aligned_curve
    return raw_by_video, aligned_by_video


def save_plot(
    values: dict[tuple[int, str], np.ndarray],
    output: Path,
    *,
    title: str,
    ylabel: str = "Watermark bit accuracy",
) -> None:
    frame_indices = np.arange(next(iter(values.values())).size)
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {key: plt.get_cmap("tab10")(index) for index, key in enumerate(values)}
    for (seed, filename), accuracy in values.items():
        ax.plot(frame_indices, accuracy, color=colors[(seed, filename)], linewidth=1.8,
                label=f"Seed {seed} · {filename}")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, label="Random guess (0.5)")
    ax.set(title=title, xlabel="Frame index", ylabel=ylabel, xlim=(0, frame_indices[-1]), ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("Validation_Results"))
    args = parser.parse_args()
    raw, aligned = load_video_curves(args.root)
    out_dir = args.root / "Motion_CogVideoX_seed_accuracy_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_plot(raw, out_dir / "before_alignment.png", title="Watermark bit accuracy before RAFT alignment")
    save_plot(aligned, out_dir / "after_alignment.png", title="Watermark bit accuracy after RAFT alignment")

    frame_indices = np.arange(next(iter(raw.values())).size)
    fig, ax = plt.subplots(figsize=(13, 8))
    colors = {key: plt.get_cmap("tab10")(index) for index, key in enumerate(raw)}
    for (seed, filename), accuracy in raw.items():
        ax.plot(frame_indices, accuracy, color=colors[(seed, filename)], linewidth=1.8,
                label=f"Seed {seed} · {filename} · before")
        ax.plot(frame_indices, aligned[(seed, filename)], color=colors[(seed, filename)], linestyle="--",
                linewidth=1.8, label=f"Seed {seed} · {filename} · after RAFT")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, label="Random guess (0.5)")
    ax.set(
        title="Watermark bit accuracy before and after RAFT alignment",
        xlabel="Frame index",
        ylabel="Watermark bit accuracy",
        xlim=(0, frame_indices[-1]),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out_dir / "before_after_alignment.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(out_dir)


if __name__ == "__main__":
    main()
