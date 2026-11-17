"""Analyze existing LoT-Pass encoded source images without running the model."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import sobel, uniform_filter


ROOT = Path(__file__).resolve().parent
DEFAULT_PAIRS = {
    ROOT / "data/motion_sources": ROOT / "Validation_Results/Motion_CogVideoX/Encoded",
    ROOT / "data/motion_sources_2": ROOT / "Validation_Results/Motion_Exp2_CogVideoX_seed42/Encoded_Source",
}
OUT = ROOT / "Validation_Results/encoder_heatmap"


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", default=[], metavar="SOURCE_DIR=ENCODED_DIR",
                        help="Override/add an input pair; encoded images must have matching filenames")
    args = parser.parse_args()
    pairs = dict(DEFAULT_PAIRS)
    for pair in args.pair:
        source, encoded = pair.split("=", 1)
        pairs[Path(source).resolve()] = Path(encoded).resolve()

    OUT.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    energy_by_band = {name: [] for name in ("low", "medium", "high")}
    for source_dir, encoded_dir in pairs.items():
        if not source_dir.is_dir():
            missing.append(f"Missing source directory: {source_dir}")
            continue
        for source_path in sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}):
            exact = encoded_dir / source_path.relative_to(source_dir)
            # Older motion runs name encoded files by prompt (for example,
            # s1_walk.png) while the source is s1.png. Analyze every variant.
            watermarked_paths = [exact] if exact.is_file() else sorted(encoded_dir.glob(f"{source_path.stem}_*"))
            watermarked_paths = [p for p in watermarked_paths if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}]
            if not watermarked_paths:
                missing.append(f"{source_path}: no matching image at {exact} or {source_path.stem}_* in {encoded_dir}")
                continue
            for watermarked_path in watermarked_paths:
              try:
                original, watermarked = load_rgb(source_path), load_rgb(watermarked_path)
                if original.shape != watermarked.shape:
                    missing.append(f"{source_path}: dimensions differ ({original.shape} vs {watermarked.shape})")
                    continue
                residual = watermarked - original
                residual_energy = np.mean(residual ** 2, axis=2)
                gray = np.mean(original, axis=2)
                gx, gy = sobel(gray, axis=1), sobel(gray, axis=0)
                texture = np.hypot(gx, gy)
                # Rank-based thirds give every image comparable texture bands.
                low_cut, high_cut = np.quantile(texture, [1 / 3, 2 / 3])
                masks = {"low": texture <= low_cut,
                         "medium": (texture > low_cut) & (texture <= high_cut),
                         "high": texture > high_cut}
                values = {}
                for band, mask in masks.items():
                    value = float(np.mean(residual_energy[mask])) if np.any(mask) else float("nan")
                    values[band] = value
                    if np.isfinite(value):
                        energy_by_band[band].append(value)
                out_dir = OUT / source_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)
                heat = np.sqrt(residual_energy)
                scale = float(np.quantile(heat, 0.99)) or 1.0
                heat_norm = np.clip(heat / scale, 0, 1)
                cmap = plt.get_cmap("magma")
                overlay = (0.55 * original + 0.45 * cmap(heat_norm)[..., :3]).clip(0, 1)
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                for ax, image, title in zip(axes, (original, watermarked, heat_norm, overlay),
                                             ("Original", "Watermarked", "Residual strength", "Heatmap overlay")):
                    ax.imshow(image, cmap="magma" if title == "Residual strength" else None)
                    ax.set_title(title); ax.axis("off")
                fig.tight_layout()
                fig.savefig(out_dir / f"{watermarked_path.stem}_heatmap.png", dpi=150)
                plt.close(fig)
                rows.append({"source_dir": source_dir.name, "filename": source_path.name, "watermarked_filename": watermarked_path.name,
                             "mean_residual_energy": float(np.mean(residual_energy)),
                             "low_texture_energy": values["low"],
                             "medium_texture_energy": values["medium"],
                             "high_texture_energy": values["high"]})
              except (OSError, ValueError) as exc:
                missing.append(f"{source_path}: {exc}")

    with (OUT / "statistics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["source_dir", "filename", "watermarked_filename", "mean_residual_energy", "low_texture_energy", "medium_texture_energy", "high_texture_energy"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    averages = {band: (float(np.mean(vals)) if vals else None) for band, vals in energy_by_band.items()}
    description = "; ".join(f"{b}: {v:.8g}" if v is not None else f"{b}: no data" for b, v in averages.items())
    summary = ["# Encoder heatmap analysis", "", f"Processed {len(rows)} image(s); skipped or missing {len(missing)}.",
               "", f"Mean per-image residual energy by Sobel texture tercile: {description}.",
               "", "Residual heatmaps show embedding perturbation strength. They do not measure decoder sensitivity or actual watermark information density."]
    if missing:
        summary += ["", "## Missing or skipped inputs", ""] + [f"- {item}" for item in missing]
    (OUT / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Processed {len(rows)} images; see {OUT}")


if __name__ == "__main__":
    main()
