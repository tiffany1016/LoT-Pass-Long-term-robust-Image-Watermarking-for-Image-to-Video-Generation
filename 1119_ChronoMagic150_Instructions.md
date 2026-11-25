# ChronoMagic-Bench-150 Meteorological × LoT-Pass × CogVideoX

## Coding Agent Instructions — 16 Paired Global Appearance Transitions

## 0. Goal and scope

Build a paired single-image-to-video (I2V) evaluation to investigate whether **global appearance transitions**—especially day-to-night, night-to-day, sunrise, sunset, and weather/visibility transitions—cause additional degradation in **LoT-Pass raw watermark decoding** relative to normal I2V generation from the **same watermarked source image**.

**Experiment budget:** approximately 10 hours total, with approximately 15 minutes per CogVideoX video. Prioritize completing the paired generation; account for preprocessing, evaluation, and failures.

**Fixed design:**
- Data source: **official ChronoMagic-Bench-150**, not the full 1,649-sample version.
- Candidate scope: `main_category == meteorological` in the official 150-sample metadata.
- Select **16 independent source samples**, preferentially **12 illumination** and **4 weather/visibility** transitions, subject to verified eligibility.
- For **each** selected sample, produce exactly **one Transition** and **one paired No-transition Control** video: **16 pairs / 32 videos**.
- **No extra seeds** or repeated generation for statistical replication.
- Approval gate: **complete Stage A, then STOP for the user's approval** before encoding or generating videos.
- Preserve and reuse existing LoT-Pass, CogVideoX, stable-region, encoder/decoder heatmap, and overlay code wherever possible. **Do not rebuild working analysis modules.**

This is a **ChronoMagic-150-derived I2V evaluation**, not the original ChronoMagic T2V evaluation.

## 1. Non-negotiable experimental principles

1. **Paired controls are mandatory.** If a source image is used, generate both Transition and No-transition; do not produce an unpaired partial final dataset.
2. **No-transition is NOT static.** It preserves the initial lighting/weather/visibility condition while allowing reasonable natural scene dynamics. Do not demand an entirely frozen video.
3. **Same source image and watermark.** Encode each source image **once** and reuse that exact watermarked image, payload, and encoder configuration for both conditions. Use the **same seed within the pair**, and otherwise match model, scheduler, resolution, frames, steps, and guidance.
4. **Control only the target appearance transition.** Keep common scene/motion/camera wording as closely matched as practical; the primary change between paired prompts should be whether the intended appearance transition takes place.
5. **Source/image fidelity.** Keep original IDs, captions, reference-video URLs, extraction timestamps, and derived prompts. Never silently replace missing reference assets or invent official prompts.
6. **Selection before BA.** Selection must not depend on known/observed LoT-Pass bit accuracy. Record pre-selection criteria and exclusions.
7. **Prompt compliance is not guaranteed.** Measure actual transition success and actual scene-wide change. Report invalid generations rather than quietly removing them.
8. **No temporal-stability analysis.** Stable-region analysis compares each generated frame to the original watermarked source image at the same coordinates; **do not** calculate adjacent-frame temporal stability. Optical-flow alignment is not required in this experiment.
9. **User approval at end of Stage A.** Do not run LoT-Pass encoding, CogVideoX generation, or decoding before explicit approval.

## 2. Official data sources

- Dataset: https://huggingface.co/datasets/BestWishYsh/ChronoMagic-Bench
- GitHub: https://github.com/PKU-YuanGroup/ChronoMagic-Bench
- Use the official **ChronoMagic-Bench-150** captions/metadata file (check repository contents for actual filename and column spelling; do not assume unverified paths).
- Verify source files, schema, total sample count, meteorological subset count, and actual available reference-video assets. Preserve the raw metadata untouched.

Do not substitute the 1,649-sample version, the model-generated result videos, or unrelated web images as official reference source material. If the required **initial reference frames are unavailable**, report precisely what is missing and **STOP** instead of silently improvising replacements.

---

# Stage A — Data preparation and user approval (RUN THIS FIRST)

## A1. Inspect and inventory

1. Locate the existing VBench++/TC-Bench pipeline, LoT-Pass checkpoint and code, CogVideoX I2V generation script, stable-region analyzer, and existing encoder/decoder heatmap scripts.
2. Download/read **ChronoMagic-Bench-150** metadata.
3. Filter the official `meteorological` main category (normalize case/spacing only for filtering; preserve original strings).
4. Produce a full inventory including each official ID, main category, subcategory, caption, reference URL/asset location, and asset availability.
5. Report **actual counted totals** for the 150-sample version and each meteorological subcategory. Do not hard-code a previously quoted count without verifying.

**Output:** `chronomagic150_meteorological_all.csv` and a metadata-count summary.

## A2. Predefine eligibility and classify candidates

Primary focus: **global appearance transitions** expected to change a substantial fraction of the frame while scene composition is reasonably preserved.

Prioritize:
- `day2night`, `night2day`, `sunrise`, `sunset`
- other samples **only if the actual caption describes broad illumination / color-temperature changes** (for example, qualifying `day`, `night`, `nature` samples)
- major weather / atmosphere / visibility changes (storm formation, fog clearing, etc.) affecting large parts of the scene

Deprioritize:
- moving clouds alone with otherwise unchanged scene
- stars/aurora movement without broad illumination change
- primarily local object movement, object growth, unrelated scene replacement, or little visible appearance change
- clips whose initial state is inconsistent with the reference caption

Assign reproducible, pre-BA labels such as:
- `illumination_decrease`, `illumination_increase`, `color_temperature`, `weather_visibility`
- `global_appearance_candidate` (`yes`/`no`/`uncertain`)
- brief `selection_reason` and `exclusion_reason`

These are **project-defined labels, not official ChronoMagic category names**.

## A3. Select 16 source samples

Select **exactly 16 distinct source samples**, provisionally:
- **12 illumination-related**
- **4 weather / visibility-related**

Selection criteria: faithful initial condition, broadly affecting scene appearance, diversity across scenes/directions, downloadable reference material, and avoidance of near duplicates. Use a deterministic, documented selection rule. Do not select for likely watermark failures. If this allocation is not feasible, report the eligible counts and recommend the closest valid 16-sample allocation for **user approval**; do not fabricate additional official samples.

## A4. Extract initial I2V frames

For each selected sample:
1. Retrieve its corresponding **real/reference video**, if accessible under the official dataset resources.
2. Extract the **first frame of the relevant reference clip** by default, using consistent extraction settings. If a different frame is necessary to match the stated initial condition, explicitly document why and record its timestamp.
3. Verify visually that the frame actually matches the *initial* appearance specified in the caption (e.g., daytime for day-to-night, nighttime for night-to-day).
4. Preserve original frame and a separate properly resized/cropped model input. Record all resizing/cropping.
5. Do not replace unavailable clips with fabricated frames or unrelated images without user approval.

Required columns: `sample_id`, `sub_category`, `official_caption`, `reference_video_url`, `source_frame_original`, `source_frame_model_input`, `frame_timestamp`, `asset_status`, `initial_state_match`.

## A5. Create paired I2V prompts

Generate **two English prompts per sample**.

**Transition:** Adapt the official ChronoMagic caption to a single-first-frame CogVideoX I2V instruction while preserving its original transition semantics. Keep the scene composition, avoid unrelated new objects, and avoid introducing camera motion unless essential to the original source.

**No-transition Control:** Keep the *same original appearance state* (lighting / weather / visibility) throughout the generated sequence; allow the same natural motion as the transition prompt and keep all other instructions as close as practical. This is **not** a frozen-frame/static video.

Examples (illustrative custom prompts, not official benchmark captions):

| Type | Transition | No-transition |
|---|---|---|
| Day → night | The same city gradually transitions from daytime to night as the overall scene darkens; camera remains stationary. | The same city remains under consistent daytime lighting throughout; natural motion continues; camera remains stationary. |
| Sunrise | The scene gradually brightens from predawn darkness into morning light; camera remains stationary. | The scene remains under consistent predawn low light throughout; natural motion continues; camera remains stationary. |
| Storm | The originally clear scene gradually develops stormy weather and darkening atmosphere; camera remains stationary. | The scene maintains its original clear weather and consistent lighting; natural motion continues; camera remains stationary. |

Avoid wording that forces different amounts of object motion across conditions. Record both prompts verbatim and mark them as **adapted** / **custom**, not official original text.

## A6. Review artifacts

Generate:
- `chronomagic150_meteorological_all.csv`
- `chronomagic150_selected16.csv` (16 rows with official caption, chosen frame, both prompts, category, selection reason, and source references)
- `chronomagic150_generation_manifest.csv` (32 rows: sample ID + `transition`/`control`, image path, prompt, seed policy, etc.)
- `selected16_contact_sheet.png` with **original IDs**, readable thumbnail labels, and an index linking back to the full prompts
- `selection_report.md`: total available records, per-category counts, selected-case rationale, unavailable assets, reference-frame validity, and any deviations from target allocation

**STOP. Present these assets and request the user's approval.** Do not execute Stage B until approval.

---

# Stage B — LoT-Pass encoding and CogVideoX I2V generation (AFTER APPROVAL)

## B1. Encode images

For each approved source:
- Encode **once** with the existing LoT-Pass encoder/checkpoint.
- Save the exact watermarked source image, message/bit payload, embedding config, and mapping to original source ID.
- Reuse **the same watermarked image and message** for its Transition and Control outputs.

## B2. Generate 32 paired videos

For each of the 16 source images:
1. Generate one Transition video.
2. Generate one No-transition Control video.
3. Use the same random seed within each pair, with otherwise identical generation settings, except the two prompts.
4. Keep the same CogVideoX model/version, input size, FPS, frame count, inference steps, guidance, sampling method, and preprocessing.
5. Save complete prompt text, seed and generation metadata, frame resolution/FPS, paths, logs, and success/failure status.
6. Implement **resume/checkpoint behavior** to avoid regenerating successfully completed videos. Do not increase seed count or alter generation settings to cherry-pick successful watermark behavior.

Target: **32 videos**, approximately 8 hours at 15 minutes/video, leaving nominal time for preparation/evaluation. Actual runtime may vary.

---

# Stage C — Evaluation and analysis

## C1. Raw LoT-Pass bit accuracy

Use the existing **RAW** decoder, **without optical-flow alignment**, on every frame. Report:
- per-frame `raw_bit_accuracy`
- video mean BA
- minimum BA
- last-third mean BA
- per-frame BA trajectory and labeled sample/condition

Keep original filenames and IDs fully traceable.

## C2. Matched paired comparison

For each source sample and each comparable frame index, compare Transition vs Control.

Compute:
- `delta_mean_ba = mean_ba_transition - mean_ba_control`
- `delta_last_third_ba`
- paired BA trajectories
- distribution/summary of the 16 sample-level differences
- category-stratified results (illumination vs weather/visibility, and direction where sample size permits)

**Do not treat video frames as independent benchmark samples.** Report the actual number of independent source images. Descriptive results are acceptable for this limited pilot.

## C3. Source-anchored stable regions

Reuse the existing stable-region implementation:
- compare generated frame `I_t` to its own **watermarked source image** at the **same coordinates**
- 16×16 patch size by default
- record MAE, SSIM, binary stable mask, continuous similarity map, stable-region ratio (SRR), largest connected stable-region ratio (LSRR), mean similarity
- preserve configured thresholds and report them; do not silently retune thresholds based on BA
- **do not compute temporal stability**

Control and Transition should use identical similarity definitions and color scales.

## C4. Overlay videos

For **every generated video**, export:
- `similarity_overlay.mp4`: continuous patch similarity represented by color, semi-transparently over the original generated frames
- `stable_mask_overlay.mp4`: stable patches highlighted in semi-transparent green, with other areas left unchanged

**Alpha default `0.35`, configurable.** Transparency controls overlay visibility only, **not** similarity. Original video content must remain clearly visible. Preserve FPS, resolution, and frame order. Save underlying numeric maps for reproducibility.

## C5. Encoder / decoder heatmaps

**Reuse existing watermark heatmap implementations**; do not invent or rewrite a new localization method.
- Encoder heatmap: once per source image.
- Decoder heatmap: existing representative-frame strategy on both Transition and Control.
- Where maps are aligned and nonnegative, compute encoder/decoder attribution overlap with binary stable regions.
- Save comparisons/overlays without obscuring the video content.

Interpret gradient/occlusion sensitivity as **decoder attribution**, not automatically as the physical location of watermark bits.

## C6. Transition-success and control validation

Check **both** videos in each pair. Mark:
- `transition_status`: `valid` / `partial` / `invalid`
- `control_status`: `valid` / `partial` / `invalid`
- camera-motion contamination, if apparent

For illumination, inspect whole-frame luminance change, patch/region-level luminance coverage, and changes in non-sky areas where relevant. For fog/storm, inspect visibility/contrast and target weather transition, with manual review where necessary.

A prompt asking for day-to-night is **not** evidence the model performed day-to-night. Report all 16 attempted pairs and separately analyze valid/partial/invalid generations. Do not quietly discard failure-to-follow-prompt examples.

## C7. Final analysis

Answer, with quantitative figures and paired examples:
1. Does Transition reduce raw BA relative to its same-source No-transition Control?
2. Which successfully generated global appearance transitions are associated with the largest BA drops?
3. Does paired change in SRR/LSRR track paired change in BA?
4. Does decoder heatmap attribution overlap retained source-anchored stable regions?
5. Are low-BA outcomes attributable to actual appearance transitions, or also present in No-transition Controls?

Avoid claiming causal isolation beyond what matched prompts and generated outputs support. Both branches remain stochastic generative transformations.

---

# Stage D — Final deliverables

Deliver a clear directory hierarchy with:

```text
chronomagic150_global_appearance/
├── dataset/
│   ├── chronomagic150_meteorological_all.csv
│   ├── chronomagic150_selected16.csv
│   ├── chronomagic150_generation_manifest.csv
│   └── source_frames/
├── stage_a_review/
│   ├── selected16_contact_sheet.png
│   └── selection_report.md
├── watermarked_sources/
├── generated_videos/
│   └── <sample_id>/{transition,control}.mp4
├── decoding/
├── stable_region/
│   └── <sample_id>/{transition,control}/
│       ├── similarity_overlay.mp4
│       └── stable_mask_overlay.mp4
├── heatmaps/
├── analysis/
│   ├── per_frame_raw_ba.csv
│   ├── per_video_summary.csv
│   ├── paired_comparison.csv
│   ├── transition_validation.csv
│   └── plots/
└── final_report.md
```

All tables must use stable source IDs and original benchmark captions. Record dataset provenance, extracted timestamps, selection criteria, prompt transformations, software/model versions, random seed, thresholds, and generation settings. If a file or stage fails, log the failure without fabricating a result.

## Immediate agent task

**Execute Stage A only.** Inspect the official ChronoMagic-150 meteorological sample list, prepare a rigorously defined 16-source/32-prompt paired review package, and **STOP for user approval before generating any videos**.
