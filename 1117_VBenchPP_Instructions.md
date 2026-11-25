# VBench++ I2V × LoT-Pass: Complete Agent Implementation Instructions (v2)

**Project:** Systematic evaluation of LoT-Pass raw watermark decoding under VBench++ camera-motion conditions  
**Scope:** 10 source images × 4 official motion conditions (static / pan / tilt / zoom) = **40 image–prompt pairs**, **one generation seed per pair**  
**Important gate:** **STOP after image/prompt selection. Obtain explicit user approval before any watermark encoding, I2V generation, or decoding.**  
**Current study type:** Pre-registered, stratified-sample **pilot study**, **not** an evaluation of all 763 VBench++ pairs.

**Post-study addendum:** The user later authorized a supplemental LoT-Pass RAFT alignment/recovery analysis. Keep the original raw results unchanged; save aligned BA, aligned videos, and post-alignment decoder heatmaps separately under `analysis/optical_flow/`.

## 1. Research objectives and non-negotiable constraints

This experiment investigates:

- **RQ1:** In an unmodified VBench++ I2V camera-motion subset, how do **pan**, **tilt**, and **zoom** affect LoT-Pass raw bit accuracy relative to a paired **static-camera** condition?
- **RQ2:** Does raw bit accuracy relate to the amount and spatial concentration of source-image content that remains visually similar **at its original pixel coordinates**?
- **RQ3:** Do the existing **encoder and decoder attribution heatmaps** concentrate on the stable regions, particularly in high-accuracy frames?

**Do not:**

- Select images based on expected LoT-Pass failures or observed watermark results.
- Rewrite or strengthen the official VBench++ prompts.
- Switch generator/backbone or tune prompts in response to poor motion adherence without documenting a separate, explicitly approved experiment.
- Treat the requested motion as proof that it actually occurred.
- Present high similarity / attribution overlap as proof of causal reliance.
- Reimplement existing LoT-Pass encoder/decoder heatmap algorithms.
- Compute **temporal stability** (frame-to-frame comparison), unless separately requested.
- Run optical-flow **alignment/recovery** or multi-seed experiments in this version.

All test outcomes must be preserved, including high-BA cases, low-BA cases, static cases, and videos where motion control failed.

## 2. Authoritative source and initial inspection

Use the official **VBench++ I2V** code in the repository folder named `vbench2_beta_i2v` (this legacy folder name does **not** mean the independent VBench-2.0 benchmark).

- Repository: <https://github.com/Vchitect/VBench>
- I2V documentation: <https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/README.md>
- Official image–prompt list: <https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/vbench2_i2v_full_info.json>
- Download script: <https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/download_data.sh>
- Optional camera-motion evaluation code: <https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/camera_motion.py>

The official README lists **763 `camera_motion` prompt entries** across **109 background images**. These are expected reference counts, not numbers to hard-code. Validate and log what is actually found. The official full evaluation recommends five samples per pair; we intentionally use **one seed** to reduce GPU cost, so conclusions must be labeled exploratory.

Before modifying any project:

1. Discover the **existing working LoT-Pass project and CogVideoX scripts/checkpoints**, including local modifications, message mapping, config files, and result parsers. Do not overwrite them.
2. Locate **previously implemented encoder heatmap and decoder heatmap scripts**, their methods (e.g. residual magnitude, gradient attribution, occlusion), model checkpoints, data conventions, and output formats. Reuse as-is where possible.
3. Record repository commit hashes, Python/environment versions, source data checksum, configured image preprocessing, model settings, and decoder settings.
4. Verify storage and memory availability. Use the existing Python environment/venv where practical; do not require sudo or conda.
5. Implement new orchestration/wrappers in a separate experiment directory; keep dependencies minimal.

## 3. Stage A — Construct a reproducible subset (NO GPU generation)

### 3.1 Extract the official camera-motion entries

Read `vbench2_i2v_full_info.json`. Filter entries where `"camera_motion" in item["dimension"]`.

Preserve **the exact** `prompt_en`, `image_name`, `image_type`, and full original JSON record or its stable index. Classify the motion using the following **exact prompt suffixes** (log and fail loudly on unknown suffixes):

| Label | Exact suffix |
|---|---|
| `static` | `camera static` |
| `pan_left` | `camera pans left` |
| `pan_right` | `camera pans right` |
| `tilt_up` | `camera tilts up` |
| `tilt_down` | `camera tilts down` |
| `zoom_in` | `camera zooms in` |
| `zoom_out` | `camera zooms out` |

Group by **original** `image_name`. Only consider images with all seven official variants. Reject/report duplicate `(image_name, motion_full)` records unless duplicates are truly identical. Do not depend on JSON ordering.

### 3.2 Choose exactly 10 source images

Use deterministic stratified random sampling with **selection seed = 42**:

- `architecture`: **3** images
- `indoor`: **3** images
- `scenery`: **3** images
- `abstract`: **1** image

Sort filenames before sampling so the result is stable. If any category has insufficient qualifying images, report actual counts and use a **documented deterministic fallback** to fill to 10 from remaining categories (in priority order `architecture → indoor → scenery → abstract`). Never silently replace low-quality-looking images with preferred ones.

Assign indices `VB001`–`VB010` **after** the final deterministic selection order is fixed.

### 3.3 Choose exactly four OFFICIAL conditions per image

One of each: `static`, `pan`, `tilt`, `zoom`. Directions are determined by selected image index (1-based):

| Selected index | Static | Pan | Tilt | Zoom |
|---|---|---|---|---|
| Odd (1, 3, …, 9) | static | left | up | in |
| Even (2, 4, …, 10) | static | right | down | out |

Expected final totals: **10 static; 5 pan-left + 5 pan-right; 5 tilt-up + 5 tilt-down; 5 zoom-in + 5 zoom-out**.

**Do not compose new prompts.** The final `prompt` field must be copied verbatim from the metadata entry matched by image and full motion type.

### 3.4 Download and prepare selected images

Use VBench++'s official cropped image suite, selecting a consistent crop variant (`crop/16-9` or other available ratio) and documenting why. Download only necessary archives if practical; do not download all generated benchmark videos. `i2v-bench-info.json` provides crop/source provenance, whereas `vbench2_i2v_full_info.json` provides the official prompt pairings.

The current CogVideoX setup may operate at **720×480 (3:2)**, which does **not** match the available **16:9** crop exactly. Implement and record **one fixed, content-preserving crop/resize policy** for all 10 images. **No uncontrolled stretching or different geometry per condition.**

- Produce one canonical **pre-watermark, generator-input image** per selected source.
- Keep the untouched official image, and save the exact preprocessing transform/size.
- Make the preview show the canonical image that will be given to the watermark encoder, not only a different-aspect-ratio original.
- Verify every selected image exists and can be decoded.

### 3.5 Mandatory approval package and STOP

Create:

- `selection/official_camera_motion_all.csv` (all official camera-motion entries)
- `selection/eligible_images.csv` (eligible image list, scene types and seven-condition checks)
- `selection/selected_10_images.csv` (10 images, original names, official scene types and local image paths)
- `selection/vbench_40pairs.csv` (40 rows; exact prompt, identifiers and paths)
- `selection/contact_sheet.png` (10 canonical source images, each with ID, scene type, shortened original name)
- `selection/selection_manifest.json` (selection seed, category distribution, metadata checksum, git commit, preprocessing settings, full list of selected IDs)
- `selection/approval_request.md` (human-readable 10-image preview reference and a table of 40 prompts)

`vbench_40pairs.csv` required columns:

`case_id, image_id, selected_index, image_name, image_type, motion_group, motion_full, prompt, official_image_path, canonical_image_path, source, selection_seed`

Example `case_id`: `VB001_static`, `VB001_pan`, `VB001_tilt`, `VB001_zoom`.

**STOP HERE. Show the contact sheet and selection table to the user and request explicit approval.** The agent must not proceed to encoding, generation, decoding, or GPU-heavy analysis on its own. If the user asks to replace an image, record the modification in `selection_manifest.json` (the subset is no longer a purely random sample; disclose it).

## 4. Stage B — Proceed ONLY after explicit approval

### 4.1 Pre-flight and baseline checks

1. Save the user's approval (e.g. `approval.json` containing the selection manifest hash and approval marker). Refuse to run if selection and approval hashes differ.
2. Reuse the currently working **LoT-Pass** encoder, decoder, checkpoint and message convention. Do not retrain models.
3. Reuse the currently working **CogVideoX-I2V** generation pipeline and config. **No model swap or prompt optimization.** Set the generation seed to **42**, unless the local working pipeline uses a documented seed procedure. Record the actual seed per video.
4. Use a consistent full test duration; for CogVideoX-5B-I2V, target **49 frames / 8 FPS / 720×480**, **subject to confirmation of the existing working configuration**. Test one case first and record any difference before starting the remaining 39.
5. Ensure each image's canonical pre-watermark preprocessing is complete **before encoding**; otherwise I2V resizing could introduce an untracked watermark degradation.
6. Confirm all 40 case IDs resolve to their intended image and prompt. Never use `video1`, `video2` as the only identifiers.

### 4.2 LoT-Pass encoding — once per source image

For each of the 10 canonical images:

- Embed **one LoT-Pass message** into the canonical image; use the same resulting **watermarked image and original message** for that image's four motion conditions.
- Save clean source, canonical source, exact watermarked generator input, message-map/ground-truth bits, encoding seed, and pre-I2V raw BA.
- Check that the actual condition image fed to CogVideoX is the same encoded image used as the **source-anchored stability reference** (after the explicitly recorded preprocessing pipeline).
- Validate correct decoder/message mapping before launching expensive generations. Also log decoding of the clean source as a sanity check, without using it to change the selected sample.

Do not generate a new watermark or new message for each motion condition.

### 4.3 Generate the 40 I2V clips

For each approved `case_id`:

1. Load the corresponding watermarked source image.
2. Use its exact official VBench++ prompt.
3. Run CogVideoX with the frozen configuration and seed.
4. Save the original generated `video.mp4`, frame sequence (or reproducible extraction metadata), effective input image, generation config, stdout/stderr logs and run status.
5. Validate basic output integrity (frame count, duration, image size, readable video).
6. Never retry with edited prompts or a different generator to make a motion appear stronger. Transient execution errors can be retried with the **same settings**.

**Resume capability required:** maintain per-case status `pending/running/completed/failed`, perform atomic output writes, verify outputs before marking completed, and skip completed valid cases on rerun. Use `case_id` and `seed` in all paths/records.

### 4.4 Raw decoding — all frames of all videos

- Decode **raw generated frames**, without optical-flow alignment or geometric registration.
- Use the correct ground-truth message for each image.
- Save **per-frame bit accuracy (BA)**, individual decoded bits if available, and full metadata.
- Mark whether frame 0 is a direct/copy-like input frame. Report BA **with and without copied input frames** where applicable; do not inflate estimates by counting the untouched source as generated content.
- Compute summary metrics: mean BA, last-third mean BA, minimum BA, and first-versus-last-third BA change.
- Persist failures/errors; do not silently discard problematic clips.

## 5. Analyses (automatic Stage B outputs)

### Analysis 1 — Per-frame raw BA

For each `image_id`, plot four labeled curves: static, pan, tilt and zoom. All x-axes are actual frame indices; log FPS, model, full motion direction and seed.

Outputs: `analysis/raw_ba/per_frame_accuracy.csv`, `analysis/raw_ba/per_image/<image_id>_ba.png`, `analysis/raw_ba/case_summary.csv`.

### Analysis 2 — Paired Motion vs Static comparison

For each source image and non-static motion `m`, compute using an identical, declared frame set:

`delta_ba_m = mean_BA(static) - mean_BA(m)`.

Compare static/pan/tilt/zoom **within image** before aggregating across images. Export paired tables and a paired-difference plot. If static accuracy is already low, explicitly flag that camera motion cannot be isolated as the only explanation.

Outputs: `analysis/paired/motion_vs_static.csv`, `analysis/paired/paired_ba_comparison.png`.

### Analysis 3 — Camera-motion adherence/validation

Motion prompt != realized motion. Produce:

- An automated, reproducible motion summary if the existing optical-flow/point-tracking code can estimate global displacement and direction without modifying generated video.
- A compact **motion review contact sheet or clips** for all 40 videos.
- Fields `motion_validation_status` (`valid`, `partial`, `invalid`, `not_reviewed`) and `validation_method`; do not invent human-reviewed labels.
- Optionally integrate the official VBench++ CoTracker-based motion metric if available at reasonable cost. Do **not** require the full official metric to complete raw BA analysis.

Report *all-prompt* results and *motion-realized* results separately (the latter only where validated), alongside the motion success fraction. Do not discard motion-invalid clips from the all-prompt denominator.

Outputs: `analysis/motion/motion_validation.csv`, `analysis/motion/review_sheet.png`.

### Analysis 4 — Source-anchored, PATCH-BASED Stable Region Analysis (required)

**Important:** This is a comparison to the original **watermarked condition image** at the **same spatial pixel coordinates**. **Do not compute temporal stability between consecutive generated frames.** **Do not perform optical-flow alignment before this analysis.**

For each generated frame `I_t`, compare it to the exact watermarked input `I_W` used for generation:

- Use identical coordinate resolution, color space (`RGB`), data range (`[0,1]`) and consistent image loading. If a geometry conversion is required, use only the pre-recorded source-to-generation preprocessing transform; **do not optimize an alignment transform** to maximize similarity.
- Partition into **16×16 pixel patches by default**, configurable. If dimensions are not divisible, handle edge patches consistently and weight area by actual patch size.
- For each patch compute **RGB MAE** (mean absolute error) and **SSIM** (local structural similarity); retain both raw arrays. SSIM's window and data range must be recorded and valid for patch size.
- Create a **binary stable patch mask** using pre-declared provisional criteria `MAE < 0.05 AND SSIM > 0.90`. These are starting calibration values, **not scientific constants**. Calibrate once on a small, explicitly recorded set **without inspecting BA outcomes**, then freeze for the whole 40-video evaluation.
- Create a **continuous similarity map in [0,1]** using a documented fixed mapping of MAE and SSIM (e.g. `0.5*clip(1-MAE/0.15,0,1) + 0.5*clip(SSIM,0,1)`). **Do not normalize independently for each frame**, as this destroys temporal comparability. Keep raw MAE/SSIM grids for alternative thresholds later.
- Compute **SRR** = stable area / total area (area-weighted); **LSRR** = largest 4-connected stable-component area / total area; **mean similarity** = area-weighted mean continuous similarity.
- Save patch grids, binary masks, and continuous maps for every frame. Export `per_frame_stability.csv` joined by case ID and frame index with raw BA.

**Interpretation boundary:** low same-coordinate stability may be caused by *geometric movement*, even when the scene content remains recoverable elsewhere. Do not equate SRR with physical watermark survival or source-content visibility.

#### Required overlays: video must stay CLEARLY visible

Overlay maps **on the actual generated video** (same resolution, same frame order, same FPS):

1. **Continuous similarity overlay**: hue encodes similarity. Suggested consistent legend: **blue/green = high similarity**, **yellow/red = low similarity**. Render as a block/patch heatmap that follows the patch grid, or smoothly interpolate for display (preserve raw patch arrays).
2. **Binary stable-region overlay**: tint **stable patches only** green; **unstable patches remain unaltered**.
3. Optional combined diagnostic frames showing existing encoder/decoder heatmap overlays, while keeping the original visible.

**Transparency requirement (user-specified):** overlay transparency is **only a viewing control**, not a similarity metric.

- Use a **fixed configurable blending alpha, default `alpha=0.35`**.
- Formula on overlay pixels: `overlay = (1-alpha)*original_frame + alpha*colorized_map`.
- For the binary overlay, apply blending **only where stable mask is 1**; other pixels stay exactly the original frame.
- **Do not** vary alpha according to similarity, BA, or motion; the **color** alone represents similarity.
- Keep original scene, objects and movement easy to recognize; use a clear colorbar/legend in a companion figure or unobtrusive margin.
- **Never** use the overlay videos or overlaid images as input to watermark decoding or numerical stability calculations.
- Save both `similarity_overlay.mp4` and `stable_mask_overlay.mp4` per case, plus representative PNG overlays.

Outputs per case: `similarity_maps/*.npy`, `mae_maps/*.npy`, `ssim_maps/*.npy`, `binary_masks/*.png`, `overlays/*.png`, `similarity_overlay.mp4`, `stable_mask_overlay.mp4`. Output globally: `analysis/stability/per_frame_stability.csv`, `analysis/stability/srr_vs_ba.png`, `analysis/stability/lsrr_vs_ba.png`.

### Analysis 5 — Reuse existing Encoder / Decoder Heatmaps (required)

**Locate and reuse** the prior working heatmap implementations; no independent replacement algorithm unless separately approved.

- **Encoder heatmap:** 1 per image (10 total). Save source, computational definition, raw numeric map, and visualization. If encoder map means embedding residual magnitude, label it as such; do not call it decoder attention.
- **Decoder heatmap:** compute the existing decoder attribution map for **every generated frame** in every video. Save every raw map and per-frame overlap metric; do not interpolate or reuse maps between frames. Render `decoder_heatmap_overlay.mp4` for each video, preserving source frame order, dimensions, and FPS, with the generated frame visible under the heatmap at alpha `0.35` by default.
- Do not pretend gradient attribution is an actual physical watermark location. Record the decoder objective (ground-truth bit score, predicted bit, etc.), attribution normalization, and method.
- Align numeric heatmap resolution to stability-mask coordinates correctly; avoid silent axis/crop mistakes. Maintain raw numeric attribution magnitude separate from display-color normalization.

Compute for every video frame:

`encoder_stable_overlap = sum(H_enc * M_stable) / sum(H_enc)`

`decoder_stable_overlap = sum(H_dec * M_stable) / sum(H_dec)`

`decoder_stable_enrichment = decoder_stable_overlap / SRR`, if `SRR > epsilon`.

Use nonnegative magnitudes or the defined attribution importance measure, record any clipping / absolute-value operation, and handle zero-sum attribution and zero-SRR as missing (`NaN`) rather than fabricated zeros.

Also compute/plot BA vs `decoder_stable_overlap`, BA vs `decoder_stable_enrichment`, and paired visual comparisons using all per-frame maps. No frame may be omitted or filled by interpolation.

**Visualization:** make clear overlays of encoder/decoder heatmaps using fixed alpha `0.35` (configurable) so the underlying **source/generated frame** remains visible. The heatmap's **color** encodes attribution; alpha is purely for readability. In comparison figures, show the source image, generated frame, stable-region mask/overlay, encoder heatmap and decoder heatmap with shared case/frame identifiers.

If the old heatmap code cannot be located or is incompatible, **report the blocker with exact missing paths/dependencies and still complete all other analyses**. Do not quietly substitute a new attribution method.

Outputs: `analysis/heatmaps/encoder/`, `analysis/heatmaps/decoder/`, `analysis/heatmaps/heatmap_metrics.csv`, `analysis/heatmaps/heatmap_vs_ba.png`, `analysis/heatmaps/comparison_figures/`.

### Analysis 6 — Category-level summary and evidence limits

Summarize across **10 paired source images** for `static / pan / tilt / zoom`:

- Number of scheduled/completed/valid-motion videos, generation success fraction.
- Mean BA and last-third BA by condition, within-image delta BA to static.
- SRR / LSRR distributions and associations with raw BA.
- Failure candidates: save **two distinct flags** (`last_third_BA < 0.80` and `static_last_third_BA - motion_last_third_BA > 0.15`), plus optional combined `either` flag. These are **pilot screening thresholds**, not established failure definitions.
- All-prompt failure rates, and a separately labeled realized-motion subset rate if validity labels are available.
- Explicitly identify counterexamples (high BA with low SRR, low BA with high SRR) without removing any records.

Use per-image paired aggregates for category summaries; do not treat dozens of correlated video frames as independent samples for confidence claims. This 10-image/1-seed study **cannot establish universal camera-motion failures**.

Outputs: `analysis/category/category_summary.csv`, `analysis/category/category_ba_distribution.png`, `analysis/category/category_failure_rate.png`, `analysis/category/summary_report.md`.

## 6. Stage C — Deferred / NOT authorized in this run

Do not automatically execute:

- Optical-flow alignment / post-alignment bit accuracy recovery.
- Additional seeds (official VBench++ recommends 5; only 1 is approved here).
- A replacement video generation model or FlexTraj camera/point-trajectory control.
- Causal stable-region ablation, patch insertion/removal, or global-edit experiments.
- Prompt editing, reselecting samples after seeing BA, or handpicked failure-only reporting.

The agent may **recommend** these next actions in the summary, but must not run them without new instructions.

## 7. Required directory layout

```text
vbench_lotpass_pilot/
├── README.md
├── config.yaml
├── environment_info.json
├── selection/
│   ├── official_camera_motion_all.csv
│   ├── eligible_images.csv
│   ├── selected_10_images.csv
│   ├── vbench_40pairs.csv
│   ├── selection_manifest.json
│   ├── contact_sheet.png
│   ├── approval_request.md
│   └── approval.json                  # created ONLY after user approval
├── images/
│   ├── official/
│   ├── canonical_clean/
│   └── watermarked/
├── runs/
│   └── VB001/
│       ├── encoding/
│       ├── static/seed_42/
│       ├── pan/seed_42/
│       ├── tilt/seed_42/
│       └── zoom/seed_42/
│           ├── video.mp4
│           ├── generation_config.json
│           ├── raw_ba.csv
│           ├── run_status.json
│           └── stability/
│               ├── similarity_maps/
│               ├── mae_maps/
│               ├── ssim_maps/
│               ├── binary_masks/
│               ├── overlays/
│               ├── similarity_overlay.mp4
│               └── stable_mask_overlay.mp4
├── analysis/
│   ├── raw_ba/
│   ├── paired/
│   ├── motion/
│   ├── stability/
│   ├── heatmaps/
│   └── category/
└── scripts/
    ├── build_selection.py
    ├── verify_selection.py
    ├── run_generation.py
    ├── run_raw_decode.py
    ├── compute_stability.py
    ├── make_overlays.py
    ├── run_existing_heatmaps.py
    └── make_report.py
```

Scripts are suggested names; the agent may reuse existing functional scripts and implement small adapters rather than duplicate working code. Ensure directory paths in CSV/JSON actually exist.

## 8. Config and data integrity

`config.yaml` should expose at least:

```yaml
selection_seed: 42
generation_seed: 42
scene_quota: {architecture: 3, indoor: 3, scenery: 3, abstract: 1}
motions_per_image: [static, pan, tilt, zoom]
generator: CogVideoX-I2V
frames: 49                 # verify working configuration
fps: 8                     # verify working configuration
width: 720                 # verify working configuration
height: 480                # verify working configuration
patch_size: 16
mae_stable_threshold: 0.05
ssim_stable_threshold: 0.90
similarity_mae_scale: 0.15
overlay_alpha: 0.35
heatmap_all_frames: true
optical_flow_alignment: false
temporal_stability: false
```

Record actual model checkpoint, guidance scale, inference steps, watermark message length and encoding settings in the run-specific manifest; do not assume unspecified values. Record all preprocess/crop/resize parameters. Maintain hashes for official metadata, selected source images, canonical clean images, encoded images and the approved selection manifest. Every figure/CSV row should expose `image_id`, `case_id`, `motion_full`, `seed`, and `frame_idx` when applicable.

## 9. Verification and acceptance criteria

### Stage A (before approval)

- [ ] Exactly **10** unique selected images, selected using seed 42 and documented stratification/fallback.
- [ ] Each source has all seven official motion variants available.
- [ ] Exactly **40** final pairs: exactly one static/pan/tilt/zoom per image.
- [ ] Balanced directions: pan left/right 5/5, tilt up/down 5/5, zoom in/out 5/5.
- [ ] All **40 prompts copied verbatim** from the official JSON; no edits.
- [ ] All selected image files exist; verified preprocessed images and manifest available.
- [ ] Review contact sheet and 40-prompt table are supplied to the user.
- [ ] **No watermark encoding, I2V generation, decoding, or heavy analysis executed.**
- [ ] Explicit user approval is awaited; no auto-continue.

### Stage B (after approval)

- [ ] Approval manifest matches current 40-pair selection.
- [ ] Exactly **10** source-image encodings, each reused across its 4 conditions.
- [ ] **40** videos produced or individually accounted for with an explicit failure status; reruns skip completed valid outputs.
- [ ] All actual settings/prompts/messages/seeds logged and mapped to original filenames.
- [ ] Raw per-frame BA exists for all successfully generated clips; no optical-flow alignment.
- [ ] Paired static-vs-motion comparisons and motion-adherence review artifacts exist.
- [ ] Source-anchored patch MAE/SSIM, stable masks, SRR and LSRR exist; **NO temporal stability**.
- [ ] Every generated video has two visualization outputs: **semi-transparent color similarity overlay** and **semi-transparent green stable-mask overlay**, retaining clear original-video visibility (`alpha=0.35` default).
- [ ] Existing encoder and decoder heatmap implementation was reused (or a documented blocker exists), with per-frame overlap metrics and one all-frame decoder heatmap overlay video per generated video.
- [ ] If the user authorizes the supplemental RAFT analysis, report aligned BA separately and preserve the raw BA and source-anchored stability results.
- [ ] Results include **all** cases, including non-failures and motion-invalid clips; category conclusions are explicitly labeled **pilot / exploratory**.

## 10. Agent handoff and reporting

**Handoff A (mandatory STOP):** Return the contact sheet, 10 image IDs/names/scenes, all 40 official prompts, source/metadata counts, output paths and any selection/download problems. **Ask for approval.**

**Handoff B (after approval and completion):** Return generation/decoding completion table, 10 per-image BA figures, category comparisons, stable-region overlays and MP4 paths, reused heatmap locations and overlap plots, summary report and caveats. Identify any cases where CogVideoX failed to realize the requested camera motion. Do not start Stage C automatically.

**Estimated generation cost:** 40 clips × ~15 min/clip ≈ **10 GPU hours**, excluding encoding, decoding, attribution calculations and rendering; this is a rough budget based on the user's observed speed, not a promise of runtime.

## 11. Interpretation guardrails

1. VBench++ labels *requested camera motion*; measure *realized motion* separately.
2. Same-coordinate source stability is not equivalent to source-content retention after 3D camera motion, and not a direct watermark-location detector.
3. Decoder gradients indicate attribution/sensitivity under the chosen objective, not proof of physical watermark placement or causal necessity.
4. The 10-image fixed subset with one seed is for evaluating feasibility and identifying candidate category-level patterns. Do not generalize its frequency estimates to all VBench++ images without expanded sampling and uncertainty analysis.
5. Do not select, replace, or omit cases after seeing LoT-Pass accuracy without reporting how that affects sample representativeness.

---

**Primary official reference:** VBench++ I2V README and `vbench2_i2v_full_info.json` at <https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v>.
