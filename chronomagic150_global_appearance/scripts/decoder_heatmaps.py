#!/usr/bin/env python3
"""Apply existing LoT-Pass gradient saliency to every frame and export overlays."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "chronomagic150_global_appearance"
DATA = EXP / "dataset"
VIDEOS = EXP / "generated_videos"
WATERMARKED = EXP / "watermarked_sources"
STABLE = EXP / "stable_region"
OUT = EXP / "heatmaps/decoder"
CHECKPOINT = ROOT / "checkpoints/model-99.pth"
INPUT_RES, ALPHA = 256, 0.35


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def overlap(heat: np.ndarray, mask: np.ndarray) -> float:
    total = float(np.sum(heat))
    return float(np.sum(heat * mask) / total) if total > 1e-12 else float("nan")


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU policy requires CUDA_VISIBLE_DEVICES=0")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in configured uv environment")
    torch.cuda.init()
    import cv2
    import decord
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision import transforms
    from gradient_saliency import compute_saliency, load_decoder, make_visualization

    with np.load(WATERMARKED / "source_message_map.npz", allow_pickle=True) as archive:
        messages = {key: np.asarray(archive[key]).reshape(-1).astype(np.float32)
                    for key in archive.files if key != "ckpt"}
    alias_map = json.loads((EXP / "heatmaps/encoder/source_id_map.json").read_text())
    alias_by_id = {sample_id: alias for alias, sample_id in alias_map.items()}
    raw_encoder = EXP / "heatmaps/encoder/raw_maps/source_images"
    raw_decode = {}
    per_frame_path = EXP / "decoding/per_frame_raw_ba.csv"
    if per_frame_path.is_file():
        for row in csv.DictReader(per_frame_path.open(encoding="utf-8")):
            try:
                raw_decode[(row["case_id"], int(row["frame_idx"]))] = float(row["raw_bit_accuracy"])
            except (KeyError, ValueError):
                continue
    decoder = load_decoder(CHECKPOINT, CHECKPOINT.parent, torch.device("cuda:0"))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((INPUT_RES, INPUT_RES)),
                                    transforms.Normalize([0.5] * 3, [0.5] * 3)])
    cases = list(csv.DictReader((DATA / "chronomagic150_generation_manifest.csv").open(encoding="utf-8")))
    OUT.mkdir(parents=True, exist_ok=True)
    overlap_rows, video_rows = [], []
    for row in cases:
        sample_id, condition, case_id = row["sample_id"], row["condition"], row["case_id"]
        video = VIDEOS / sample_id / f"{condition}.mp4"
        status_path = EXP / "runs" / sample_id / condition / "run_status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        common = {"case_id": case_id, "sample_id": sample_id, "analysis_group": row["analysis_group"],
                  "condition": condition, "seed": int(row["seed"])}
        if status.get("status") != "completed" or not video.is_file():
            video_rows.append({**common, "status": "generation_" + str(status.get("status", "missing"))})
            continue
        if sample_id not in messages or sample_id not in alias_by_id:
            raise KeyError(f"Missing message or encoder heatmap mapping for {sample_id}")
        source_rgb = np.asarray(Image.open(WATERMARKED / sample_id / "watermarked.png").convert("RGB"), dtype=np.uint8)
        encoder_map = np.load(raw_encoder / f"{alias_by_id[sample_id]}.npy").astype(np.float32)
        if encoder_map.shape != source_rgb.shape[:2]:
            raise ValueError(f"Encoder map geometry mismatch for {sample_id}: {encoder_map.shape}")
        reader = decord.VideoReader(str(video), ctx=decord.cpu(0))
        fps = float(reader.get_avg_fps())
        case_out = OUT / sample_id / condition
        (case_out / "raw_gradients").mkdir(parents=True, exist_ok=True)
        (case_out / "overlays").mkdir(parents=True, exist_ok=True)
        temp = case_out / ".decoder_heatmap_overlay.partial.mp4"
        writer = imageio.get_writer(str(temp), fps=fps, codec="libx264", quality=8, macro_block_size=1,
                                    ffmpeg_params=["-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        message = torch.as_tensor(messages[sample_id], dtype=torch.float32, device="cuda:0").unsqueeze(0)
        this_rows = []
        try:
            for frame_idx in range(len(reader)):
                frame = reader[frame_idx].asnumpy()
                image_tensor = transform(Image.fromarray(frame).convert("RGB")).unsqueeze(0).to("cuda:0")
                _, saliency = compute_saliency(decoder, image_tensor, message, target_bit=-1)
                saliency = np.asarray(saliency, dtype=np.float32)
                np.save(case_out / "raw_gradients" / f"frame_{frame_idx:04d}.npy", saliency)
                saliency_up = cv2.resize(saliency, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                mask_path = STABLE / sample_id / condition / "binary_masks" / f"{frame_idx:04d}.png"
                mask = (np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0).astype(np.float32)
                if mask.shape != frame.shape[:2]:
                    raise ValueError(f"Stable-mask geometry mismatch: {case_id} frame {frame_idx}")
                enc_overlap = overlap(encoder_map, mask)
                dec_overlap = overlap(saliency_up, mask)
                srr = float(mask.mean())
                enrichment = dec_overlap / srr if srr > 1e-12 and np.isfinite(dec_overlap) else float("nan")
                _, overlay = make_visualization(Image.fromarray(frame), saliency, alpha=ALPHA)
                overlay_rgb = np.asarray(overlay, dtype=np.uint8)
                writer.append_data(overlay_rgb)
                overlay.save(case_out / "overlays" / f"frame_{frame_idx:04d}.png")
                if frame_idx in (0, 12, 24, 36, 48):
                    stable_overlay = frame.copy(); green = np.zeros_like(frame); green[..., 1] = 255
                    stable_overlay[mask.astype(bool)] = np.clip(
                        (1-ALPHA)*frame[mask.astype(bool)].astype(np.float32) + ALPHA*green[mask.astype(bool)].astype(np.float32),
                        0, 255).astype(np.uint8)
                    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
                    panels = [source_rgb, frame, stable_overlay, overlay_rgb]
                    titles = ["Watermarked source", f"Generated {case_id} · frame {frame_idx}", "Stable patches", "Decoder gradient overlay"]
                    encoder_display = encoder_map / (float(np.quantile(encoder_map, 0.99)) or 1.0)
                    cmap = matplotlib.colormaps["magma"](np.clip(encoder_display, 0, 1))[..., :3]
                    enc_overlay = np.clip((1-ALPHA)*source_rgb.astype(np.float32) + ALPHA*cmap*255, 0, 255).astype(np.uint8)
                    panels.append(enc_overlay); titles.append("Encoder residual overlay")
                    for ax, panel, title in zip(axes, panels, titles):
                        ax.imshow(panel); ax.set_title(title, fontsize=9); ax.axis("off")
                    fig.tight_layout(); fig.savefig(case_out / "overlays" / f"comparison_{frame_idx:04d}.png", dpi=120); plt.close(fig)
                item = {**common, "frame_idx": frame_idx, "raw_bit_accuracy": raw_decode.get((case_id, frame_idx), float("nan")),
                        "encoder_stable_overlap": enc_overlap, "decoder_stable_overlap": dec_overlap,
                        "stable_region_ratio": srr, "decoder_stable_enrichment": enrichment,
                        "decoder_objective": "mean ground-truth signed bit score",
                        "attribution_method": "existing gradient_saliency.compute_saliency; mean absolute input gradient across RGB",
                        "display_alpha": ALPHA}
                this_rows.append(item); overlap_rows.append(item)
        finally:
            writer.close()
        final_video = case_out / "decoder_heatmap_overlay.mp4"
        os.replace(temp, final_video)
        video_rows.append({**common, "status": "completed", "frame_count": len(reader), "fps": fps,
                           "decoder_heatmap_overlay": str(final_video.relative_to(ROOT)),
                           "mean_encoder_stable_overlap": float(np.nanmean([r["encoder_stable_overlap"] for r in this_rows])),
                           "mean_decoder_stable_overlap": float(np.nanmean([r["decoder_stable_overlap"] for r in this_rows]))})
        print(f"[DECODER_HEATMAP_DONE] {case_id}: {len(this_rows)} frame maps and overlay video", flush=True)
    write_csv(OUT / "per_frame_overlap.csv", overlap_rows, list(dict.fromkeys(k for r in overlap_rows for k in r)))
    write_csv(OUT / "per_video_overlap.csv", video_rows, list(dict.fromkeys(k for r in video_rows for k in r)))
    (OUT / "method.json").write_text(json.dumps({
        "encoder": "Existing encoder_heatmap.py residual sqrt(mean((watermarked-clean)^2 over RGB)); raw map retained.",
        "decoder": "Existing gradient_saliency.compute_saliency; objective is mean ground-truth signed bit score; absolute input gradient averaged over RGB.",
        "frame_rule": "Attribution computed independently on every generated frame; no optical-flow alignment or temporal interpolation.",
        "display_alpha": ALPHA,
        "overlap": "sum(nonnegative attribution * same-coordinate stable mask) / sum(attribution)",
        "stable_enrichment": "decoder overlap / stable pixel ratio",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
