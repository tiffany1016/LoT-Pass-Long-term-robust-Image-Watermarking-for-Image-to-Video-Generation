# TC-Bench-derived first-frame I2V evaluation: all background-shifts entries

## Coverage

- Official category: 18 entries; all official IDs retained; source inputs available: 18/18.
- Approved entries: 18/18; retrievable source entries: 18/18; planned clips: 36; successful clips: 36/36; generation failures: 0/36.
- Condition counts: official transition 18/18 generated; custom control 18/18 generated.
- Raw decoder completed: transition 18/18, control 18/18; total raw-frame rows: 1764.
- Visual validation: 36/36 prompt/video rows visually reviewed; transition valid=4/18, observed global change=yes=2/18, control transition leakage=yes=3/18.
- LoT-Pass watermark was encoded once per unique source and reused unchanged for both conditions. Generation settings and source/message hashes are stored per run.

## Raw bit accuracy

- Pre-I2V source decode check across 18 images: mean clean-image BA 0.4931; mean watermarked-image BA 1.0000.
- Mean per-video BA, excluding a frame-0 exact conditioning copy when detected: transition 0.9298; control 0.9341; mean paired control-minus-transition difference 0.0043 across 18/18 pairs with both decoded conditions.
- Preliminary low-BA candidates (late-third mean BA < 0.80): transition 4/18 (TC101, TC102, TC110, TC117); control 1/18 (TC117). Paired candidates (|control − transition| > 0.15): 0/18 (none).
- Paired clips use the same watermarked source and seed. Frame counts are repeated measurements within clips, not independent samples.
- Preliminary screening only: late-third BA < 0.80; paired BA difference flagged when |control − transition| > 0.15. These thresholds are not population failure-rate estimates.

## Source-anchored appearance stability and heatmaps

- Stable regions compare each frame at the same pixel coordinates with its watermarked source; no temporal stability or optical-flow alignment is used.
- Patch size 16×16; normalized RGB MAE threshold <0.05; 7×7-window RGB SSIM threshold >0.90; continuous score = 0.5×clip(1−MAE/0.15,0,1)+0.5×clip(SSIM,0,1); connected components use 4-connectivity.
- Source-anchored luminance measures (Rec.709; all 49 frames averaged per video): mean absolute whole-frame luminance delta transition 0.0543, control 0.0482; mean fraction of 16×16 patches with absolute mean-luminance delta >0.05 transition 0.1815, control 0.1369.
- Similarity and stable-mask overlay opacity is 0.35 for display only. Raw MAE/SSIM/similarity maps and masks are saved separately.
- Encoder maps reuse the existing encoder-residual method; decoder maps reuse existing input-gradient saliency. Overlap is attribution mass inside the stable mask divided by total mass; enrichment divides overlap by stable area and is NA for zero denominators.

- Decoder attribution was recomputed independently for every generated frame: 1764/1764 frame maps recorded across 36 videos; each clip has a `decoder_heatmap_overlay.mp4` with the frame visible under the heatmap at alpha 0.35.

## Supplemental RAFT alignment and decoding

- LoT-Pass RAFT alignment and aligned-frame decoding completed for 36/36 clips. Mean per-video BA excluding detected copy frames: raw 0.9320, aligned 0.9522, mean within-video change +0.0202.
- Every clip has a generated-frame-0-anchored `aligned_video.mp4` and a `decoder_heatmap_overlay.mp4` showing decoder gradients computed on the aligned frames. These outputs are under `analysis/optical_flow/`; raw BA and source-anchored stability results remain unchanged.

## Transition interpretation

TC-Bench’s official `background shifts` category includes local object events and does not guarantee whole-frame appearance change. Global-change labels are reported separately from prompt success. Inspect `transition_validation.csv` and video storyboards; any unreviewed rows remain uncertain.

- Visual transition validation counts: {'valid': 4, 'partial': 5, 'invalid': 8, 'uncertain': 1}; whole-frame/global appearance change marked yes in 2/18 transition videos; target-event leakage in 3/18 controls.
- Unsuccessful transition examples from the full reviewed set (not selected based on BA): TC102, TC103, TC104, TC108, TC109, TC110, TC113, TC116.
- High raw-BA examples from a fixed top-three ranking: TC105 (1.0000), TC109 (1.0000), TC111 (1.0000); low examples from a fixed bottom-three ranking: TC117 (0.7404), TC110 (0.7691), TC101 (0.8202). Examples are descriptive and not cherry-picked evidence.

- Surprising reviewed cases: TC103 starts from an already sunset-like frame; TC104 has a scene replacement unrelated to its prompt; TC114, TC117, and TC119 show target-like activity in their custom controls. These remain included and flagged.

## Limits

This covers one seed per entry in this benchmark category under one model/checkpoint/configuration. It is descriptive benchmark coverage, not a broad causal estimate for global appearance changes. Cases with partial, invalid, or visually surprising generations remain in the denominators.

## Outputs

- `analysis/per_frame_raw_ba.csv`, `per_video_raw_ba.csv`, `paired_ba_summary.csv`
- `analysis/stable_region_metrics.csv` and per-video `analysis/stable_region/` overlays/maps
- `analysis/heatmap_overlap.csv`, raw encoder/decoder maps, comparison panels, and per-video `decoder_heatmap_overlay.mp4`
- `analysis/transition_validation.csv` and `analysis/video_review/` storyboards
