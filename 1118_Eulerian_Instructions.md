# Eulerian Test Images × LoT-Pass × CogVideoX
## Coding Agent Instructions

## Goal
Build a pilot benchmark for **dense global natural background motion** using the **Eulerian Motion Dataset testing images**.

The main research question is:

> Does LoT-Pass fail when a large portion of the image undergoes continuous non-rigid natural motion (e.g., ocean waves, river flow, smoke, clouds), even when the camera remains stationary?

This experiment is **not** an official Eulerian benchmark evaluation. It is an **Eulerian-derived I2V evaluation** using:
- Eulerian test images as source images
- VLM/LLM-generated I2V prompts
- CogVideoX for video generation
- LoT-Pass for watermark encoding/decoding
- Existing stable-region and heatmap analysis pipelines

---

## Important execution policy

### Download policy
The agent should **first try to download the Eulerian testing images automatically**.

If automatic download succeeds:
- continue with Stage A normally.

If automatic download fails because of network restrictions, permissions, broken links, or missing tools:
- **stop and ask the user to download the file manually**.
- do **not** silently substitute another dataset.

### Approval policy
The agent must stop after Stage A and wait for user approval.

Do **not** start LoT-Pass encoding or CogVideoX generation before the user explicitly approves the selected images and generated prompts.

---

## Dataset source
Use the official Eulerian Motion Dataset testing images.

Official dataset page:
- https://eulerian.cs.washington.edu/dataset/

Target asset:
- `Testing images` zip file

Assume this contains a set of `.png` test images suitable for image animation.

---

## Experiment scope
This pilot should use the Eulerian **testing images only**.

The agent should:
1. enumerate all test images
2. analyze every image
3. recommend a dense-motion subset
4. present the results for approval
5. after approval, run the full LoT-Pass + CogVideoX evaluation on the approved subset

### Initial target subset size
Recommend **10–20 images** for the pilot.

Selection should prioritize images likely to exhibit:
- large-area natural motion
- stationary camera assumption
- minimal semantic ambiguity
- minimal dependence on moving rigid foreground objects

Do not pre-select images based on expected LoT-Pass success or failure.

---

# Stage A — Dataset acquisition, inspection, prompt generation, and approval

## A1. Acquire testing images
1. Try downloading the official Eulerian `Testing images` zip.
2. Save the downloaded archive under a reproducible path.
3. Extract the images.
4. Build a manifest containing:
   - `image_id`
   - `filename`
   - `absolute_path`
   - file extension
   - width
   - height

If download fails, report the failure and request manual download from the user.

## A2. Count images
Count the total number of test images.

Output:
- `eulerian_test_manifest.csv`
- total image count in the summary report

## A3. Analyze all images using a VLM / LLM-assisted procedure
For each image, estimate the following:

- `scene_category` (e.g., ocean, river, waterfall, clouds, smoke, fog, fire, mixed)
- `dynamic_elements` (what naturally dynamic content is present)
- `estimated_dynamic_coverage` (0–100%)
- `estimated_static_coverage` (0–100%)
- `dense_dynamic_candidate` (`yes` / `no` / `uncertain`)
- `camera_stationary_suitability` (`high` / `medium` / `low`)
- `notes`

### Definition guidance
`estimated_dynamic_coverage` should represent the approximate fraction of image area occupied by naturally dynamic content.

Examples:
- full-frame ocean waves → high
- river with large static banks → medium
- waterfall surrounded by large static rocks/forest → lower

This is a **pre-generation estimate**, not a ground-truth measurement.

## A4. Generate one I2V prompt per image
Use a fixed prompt-generation template.

### Prompt-generation rules
For each image, generate **one concise English prompt** that:
1. preserves the original scene composition
2. describes realistic motion only for naturally dynamic elements
3. encourages visible motion across all naturally dynamic regions
4. keeps the camera completely stationary
5. does not add new objects
6. does not introduce scene transitions, lighting transitions, weather transitions, or camera motion
7. does not invent motion for clearly static objects
8. is approximately 25–60 words

### Required output per image
Store:
- `generated_prompt`
- `scene_category`
- `estimated_dynamic_coverage`
- `dense_dynamic_candidate`

### Suggested internal prompt template for the VLM / LLM
Use a prompt equivalent to:

> You are generating an Image-to-Video prompt for a scientific benchmark evaluating natural background dynamics. Given an input image, identify the naturally dynamic elements (such as water, waves, smoke, clouds, fog, or fire). Describe realistic continuous motion of these elements while preserving the scene and keeping the camera completely stationary. Do not add new objects, scene transitions, lighting changes, weather changes, or camera movement. Output one concise English prompt, plus scene category and estimated dynamic coverage.

## A5. Create a recommended pilot subset
From all test images, recommend **10–20 images** for the pilot.

### Selection criteria
Prioritize:
- higher estimated dynamic coverage
- clearer natural-motion scenes
- fewer dominant static regions
- diversity across scene types when possible
- good fit for stationary-camera generation

Do not use LoT-Pass or CogVideoX results during selection.

## A6. Produce approval artifacts
Create the following for user review:

1. `eulerian_test_manifest.csv`
2. `eulerian_test_analysis.csv`
3. `eulerian_prompt_candidates.csv`
4. `eulerian_recommended_subset.csv`
5. `contact_sheet_all_images.png` (or multiple pages if needed)
6. `contact_sheet_recommended_subset.png`
7. a short markdown report summarizing:
   - total image count
   - scene-category distribution
   - estimated dynamic-coverage distribution
   - recommended subset and reasons

## A7. STOP for approval
After producing the above artifacts:
- stop execution
- request user approval
- do not proceed to Stage B until approved

---

# Stage B — LoT-Pass encoding and CogVideoX generation

Run this stage only after user approval.

## B1. Prepare approved source images
Use only the user-approved subset.

## B2. LoT-Pass encoding
For each approved source image:
1. apply LoT-Pass encoding once
2. save the watermarked image
3. record the source-to-watermarked mapping

Keep the watermark message setup consistent across all approved images unless the existing pipeline requires otherwise.

## B3. CogVideoX generation
For each approved watermarked image:
1. use the generated I2V prompt
2. keep the camera stationary
3. use consistent generation settings across all cases
4. save the output video

Record all inference settings, including:
- model name
- resolution
- number of frames
- inference steps
- guidance scale if applicable
- random seed
- any scheduler or sampling settings

Support resume if generation is interrupted.

---

# Stage C — LoT-Pass decoding and quantitative analysis

## C1. Raw decoding
For every generated video:
1. decode LoT-Pass on all frames
2. compute per-frame raw bit accuracy
3. save:
   - `mean_raw_ba`
   - `min_raw_ba`
   - `last_third_raw_ba`
   - per-frame BA curve

## C2. Stable-region analysis
Reuse the existing **source-anchored stable-region pipeline**.

### Required settings
- compare every generated frame to the original watermarked source image
- do **not** compute temporal stability
- use patch-based analysis
- default patch size: `16x16`
- compute at least:
  - MAE
  - SSIM
  - stable_region_ratio
  - largest_stable_region_ratio
  - mean_similarity

## C3. Overlay visualization
Reuse the existing visualization convention.

For every video, produce:
- `similarity_overlay.mp4`
- `stable_mask_overlay.mp4`

### Visualization rules
- use the original generated frame as the background
- overlay alpha defaults to `0.35`
- alpha controls visibility only; it does not encode similarity
- similarity values should be shown using a color map
- stable-mask overlays should show stable regions with semi-transparent green
- the original generated video must remain clearly visible underneath the overlay

## C4. Encoder / decoder heatmaps
Reuse the existing heatmap implementations.

Do not reimplement watermark localization.

### Required outputs
- encoder heatmaps for each approved source image
- decoder heatmaps for representative frames in each generated video
- overlap analysis between decoder heatmaps and stable regions

Compute at least:
- `encoder_stable_overlap`
- `decoder_stable_overlap`

## C5. Dynamic-scene summary analysis
At the end, analyze relationships among:
- estimated dynamic coverage
- stable_region_ratio
- largest_stable_region_ratio
- decoder_stable_overlap
- raw bit accuracy

The main question is whether denser natural motion is associated with lower BA and reduced stable regions.

---

# Expected outputs

## Data manifests
- `eulerian_test_manifest.csv`
- `eulerian_test_analysis.csv`
- `eulerian_prompt_candidates.csv`
- `eulerian_recommended_subset.csv`

## Preview assets
- `contact_sheet_all_images.png`
- `contact_sheet_recommended_subset.png`
- `selection_report.md`

## Generation outputs
- approved watermarked images
- generated videos
- generation logs

## Analysis outputs
- per-video raw BA CSVs
- summary CSVs
- stable-region metrics
- heatmap overlap metrics
- `similarity_overlay.mp4`
- `stable_mask_overlay.mp4`
- summary plots
- final analysis markdown report

---

# Naming and reproducibility
Use stable IDs and preserve original filenames wherever possible.

Record:
- dataset source URL
- download date
- total image count
- approved subset IDs
- model versions
- all generation settings
- all analysis thresholds

---

# Final reminder
This is an **Eulerian-derived Dense Natural Motion I2V evaluation**, not an official Eulerian prompt benchmark.

The agent must not claim that the dataset itself defines a benchmark category called “dense global background motion.” That label is part of the research framing for this project.
