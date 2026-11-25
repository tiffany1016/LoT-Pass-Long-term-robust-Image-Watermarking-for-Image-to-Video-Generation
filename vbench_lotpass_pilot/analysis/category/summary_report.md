# VBench++ LoT-Pass camera-motion pilot: results summary

Scheduled 40 image-prompt pairs; raw-decode/stability-ready videos: 40; remaining/failed: 0.

## Category summary

| Condition | Completed / scheduled | Mean BA (frames 1–48) | Last-third BA | Mean SRR | Mean LSRR |
|---|---:|---:|---:|---:|---:|
| static | 10/10 | 0.9156 | 0.8723 | 0.1709 | 0.1138 |
| pan | 10/10 | 0.8141 | 0.6930 | 0.1304 | 0.0805 |
| tilt | 10/10 | 0.7911 | 0.6900 | 0.1222 | 0.0684 |
| zoom | 10/10 | 0.7867 | 0.6670 | 0.1216 | 0.0808 |

## Paired motion versus static

Each within-image comparison uses shared generated frames 1–48 (frame 0 excluded consistently). Positive delta means static BA exceeded motion BA.

- pan: n=10 paired sources; mean static-minus-motion delta=0.1014
- tilt: n=10 paired sources; mean static-minus-motion delta=0.1245
- zoom: n=10 paired sources; mean static-minus-motion delta=0.1288

## Stability associations and counterexamples

Descriptive Spearman associations (per-video averages; frames are not independent): `{"srr": {"spearman_rho": 0.25803425528135077, "p_value_descriptive_only": 0.10793183658416267, "n_cases": 40}, "lsrr": {"spearman_rho": 0.24292752251760621, "p_value_descriptive_only": 0.13093691123339557, "n_cases": 40}}`.
Median-split counterexamples (case IDs): high BA / low SRR: VB001_static, VB003_static, VB004_pan, VB004_tilt, VB005_tilt, VB006_pan, VB006_zoom, VB007_tilt; low BA / high SRR: VB002_static, VB002_pan, VB009_static, VB009_pan, VB009_tilt, VB009_zoom, VB010_pan, VB010_tilt.

## Decoder attribution heatmaps

Existing decoder gradient attribution was recomputed for every frame: 1960 maps across 40 videos, with all-frame `decoder_heatmap_overlay.mp4` outputs present for 40 videos. Overlay alpha is 0.35; per-frame overlap metrics and raw gradients are saved under `analysis/heatmaps/`. Frames within a video are repeated observations, not independent samples.

## Supplemental RAFT alignment and decoding

All-frame LoT-Pass RAFT alignment completed for 40/40 videos. Mean per-video BA excluding detected copy frames: raw 0.8304, aligned 0.9579, mean within-video change +0.1275. This supplemental path uses generated frame 0 as the reference; raw decoding and same-coordinate stability remain unchanged. Aligned video and after-alignment decoder heatmap overlays exist for 40 and 40 videos, respectively, under `analysis/optical_flow/`.

## Motion validity and limits

Motion-validation rows are marked `not_reviewed` until a human inspects `analysis/motion/review_sheet.png`; requested motion is not treated as realized motion. No realized-motion subset rate is claimed.
This is a stratified 10-image, one-seed exploratory pilot. Its frequencies do not generalize to all VBench++ images. Same-coordinate stability can drop under geometric motion and is not proof of watermark survival. Decoder gradients are sensitivity attributions, not causal or physical watermark locations.

## Incomplete cases

- None.
