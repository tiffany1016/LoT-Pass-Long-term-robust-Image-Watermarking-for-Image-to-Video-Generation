# TC-Bench Background Shifts × LoT-Pass — Coding Agent Instructions (All 18 Cases)

## 0. Scope and research objective

Build a reproducible **TC-Bench-derived, first-frame I2V evaluation** covering **ALL official `background shifts` entries**, not a random subset. The currently verified metadata has **18 official entries**; the script must compute the exact count from the pinned metadata and fail loudly if it unexpectedly changes. No random selection, no curated replacement, no cherry-picking.

For **each official entry**, prepare its reference-video first frame, (a) the **unchanged official transition prompt**, and (b) one clearly labeled **custom same-scene no-transition control prompt**. With all assets available and user approval, this means **18 distinct source images, 18 transition clips + 18 controls = 36 generated clips**. If some official reference footage is inaccessible, retain those entries as missing in the denominator, and generate only for entries with verified, approved inputs. Do not silently substitute images.

**Research distinction:** TC-Bench originally evaluates temporal transitions using reference-video data and may use endpoint-conditioned I2V/frame interpolation; this study instead adapts official cases to **single-first-frame CogVideoX-I2V**. Name it **"TC-Bench-derived first-frame I2V evaluation of all official background-shifts entries"**. Do not claim to reproduce the original official TC-Bench endpoint-conditioned protocol.

**Post-study addendum:** The user later authorized supplemental LoT-Pass RAFT alignment/recovery across all generated clips. Keep the original raw BA and source-anchored stability results unchanged; place aligned videos, aligned BA, and decoder heatmap overlays in a separate `analysis/optical_flow/` tree.

**Hard stop / approval gate:** Stage A includes ALL metadata preparation, reference retrieval, first-frame extraction, control-prompt drafting, and review. At the end, the agent MUST STOP and ask the user to approve all inputs and prompt pairs. **No LoT-Pass encoding, CogVideoX generation, or expensive decoding/heatmap work before explicit approval.**

## 1. Official sources

- Repo: https://github.com/weixi-feng/TC-Bench
- Official metadata: https://github.com/weixi-feng/TC-Bench/blob/master/i2v_prompts.json
- Original reference-video download script: https://github.com/weixi-feng/TC-Bench/blob/master/download_youtube.py
- The official `background shifts` category has **18 entries** in the currently verified metadata (IDs 101–106 and 108–119; ID 107 belongs to another category). The script must compute and report the actual count rather than relying solely on this figure.

Each entry resembles:

```json
{
  "id": 101,
  "prompt": "A cityscape transitioning from sunset to night.",
  "type": "background shifts",
  "video": ["https://www.youtube.com/watch?v=-SxbMeocXSM", "3,9"],
  "keywords": ["outdoor", "reality", "timelapse"]
}
```

Here `video[1]` is a comma-separated pair of **seconds in the original reference video**, not a generated-video frame index. Parse floats, including whitespace. If the original video is first trimmed to the designated interval, extract the initial image at local clip time 0; **do not apply the start timestamp a second time**.

## 2. Stage A — Dataset preparation, selection, and USER APPROVAL

### A1. Inspect and reuse the previous environment

1. Locate the existing VBench++ experiment workspace, model checkpoints, scripts, raw decoder, encoder/decoder attribution scripts, and stable-region visualization code. Reuse them wherever possible; do not rewrite their algorithms.
2. Create a separate experiment root without modifying previous VBench++ outputs.
3. Record source repository URL and commit hash, generation model version, LoT-Pass checkpoint, Python environment, and dependencies.
4. Do not install heavyweight camera-control, video-editing, or evaluation models in Stage A.

### A2. Construct the official candidate pool

1. Clone/fetch the official TC-Bench repository.
2. Read `i2v_prompts.json`; select entries with exact `type == "background shifts"`.
3. Preserve the full original JSON fields and original prompt text, including punctuation and capitalization. Do not paraphrase or enhance the official transition prompt.
4. Verify that each entry has an ID, nonempty prompt, a reference URL, and a valid `start,end` range with `end > start`.
5. Export `official_background_shifts_all.csv` and a JSON snapshot. Include every official entry, even if it looks unsuitable for whole-frame attribute change.

### A3. Census of ALL official entries — NO SAMPLING

1. Sort the complete category by integer official `id`; retain every entry in `official_background_shifts_all.csv` and in the generation manifest.
2. Verify the set and count from the actual pinned `i2v_prompts.json` (expected **18**). **Do not randomly select 10** and do not apply eligibility filters based on scene type, anticipated full-frame change, ease of generation, or expected LoT-Pass accuracy.
3. Save the metadata commit/hash, full set of original IDs, deduplicated image/video reference provenance, and the exact parsing code. Report if some IDs share a reference asset or produce matching frames; still preserve entries and IDs.
4. Save `all_18_initial.csv` (or use a count-based filename if official count changes) and `dataset_census_manifest.json` **before** looking at model outcomes.
5. Keep unreachable or anomalous official entries in the manifest as explicit missing/excluded-from-generation cases with reason; **never replace or silently drop them**.

### A4. Retrieve references and extract input images

1. Attempt retrieval of the official reference clip for **every official `background shifts` entry** (expected 18) using its URL and recorded `[start,end]` seconds. Do not run an unmodified download script that downloads unrelated categories.
2. Only download when legally and technically permitted. If a URL is unavailable, record `source_unavailable`, the official ID, exact URL, error, and reason. **Do not substitute a visually similar or unrelated image.**
3. For a full reference video, seek accurately to the specified official start time and extract that frame. For an already-trimmed clip, extract its frame at local time 0; never double-apply the start offset.
4. Save the source clip (if available), **original extracted first frame**, and a separate reproducibly resized/cropped model-ready image. Avoid arbitrary exposure/white-balance corrections or extra preprocessing that would alter the intended initial state.
5. Validate each image's dimensions, hashes, reference provenance, and compatibility with the *initial state* of the official transition prompt. If a first frame contradicts its prompt, label `input_matches_initial_state=no/uncertain` and raise it at the approval gate rather than hiding or fixing the discrepancy.
6. **All 18 entries must appear in the preview report**, even if some only have an error/placeholder instead of a photograph. The maximum runnable size is 18 images, or fewer if assets are genuinely inaccessible.

### A5. Prepare matched transition and custom no-transition control prompts

For **each official case**, create exactly two planned conditions:

- `transition`: **original TC-Bench prompt unchanged**, with no extra camera, time, weather, or quality-control instructions appended.
- `control`: **custom no-background-shift prompt** anchored to the extracted source frame's initial state. Keep the same scene, subjects, existing natural movement, and camera behavior as far as the original prompt permits, **but do not perform its targeted background/illumination/weather/season/appearance transition**.

**Crucial distinction: no-transition does NOT mean a frozen video.** The model should still produce a natural I2V clip (leaves/water/people may move); the background state alone remains consistent. Do NOT use the same generic `static video` instruction for every item. Do NOT append "camera completely stationary" just to the control if the original prompt has no equivalent restriction.

Control writing rules:

1. Determine initial state from **both** the official prompt **and the extracted first frame**. If ambiguous, mark `control_needs_review` and draft the minimal possible interpretation; do not silently invent scene details.
2. Remove only the specific state-change clause from the official prompt; preserve ongoing non-target dynamics, subject identity, and general camera-motion instructions where possible. Do not add new actors, props, weather, action, or stylistic cues.
3. Express state persistence naturally, not simply "do not change" without scene context. Keep prompt length and descriptive complexity reasonably comparable, although they cannot be perfectly matched. No manual prompt tuning based on generated BA or failure results.
4. Store the **initial state**, **target state**, **removed transition**, **preserved dynamics**, full control text, drafting rationale, and approval status.
5. Examples (custom control text is illustrative, not official):
   - Official: `A cityscape transitioning from sunset to night.` → Control: `A cityscape at sunset. The scene continues naturally, maintaining the sunset lighting throughout the video.`
   - Summer forest → snowy winter → Control describes the **same summer forest remaining in summer**, allowing natural tree/leaf movement but no snow or seasonal shift.
   - Snow melting → Control describes the **same snow-covered scene with snow remaining**, allowing unrelated natural movement.
6. If a reference source's initial state disagrees with the official prompt, flag it and request user review; do not conceal mismatch by silently rewriting the official transition prompt.
7. Controls are **study-specific additions**, not official TC-Bench entries. Distinguish `is_official_prompt=true` for transitions and `false` for controls.

Store `prompt_official`, `prompt_control`, `prompt_used`, `control_initial_state`, `control_target_state`, `control_removed_transition`, `control_preserved_dynamics`, `control_authoring_note`, `control_needs_review`, `is_official_prompt`, `condition`.

Prepare **36 planned prompt-image pairs (18 × 2)** when official data count is 18. Generate both conditions from the **identical approved, once-watermarked source image** and identical encoded message/seed/settings. If source retrieval fails, keep both planned pairs in the manifest but mark them `not_runnable_missing_source` rather than creating synthetic replacements.

**No-transition validity check after generation:** verify the control approximately preserves the target attribute's initial state; a control that accidentally performs the shift must be flagged `control_transition_leakage`, not treated as clean control.

### A6. Add pre-generation labels without cherry-picking

Annotate every official entry with non-exclusionary descriptive labels:

- `expected_change_mechanism`: illumination / weather / environment-material / local-object-event / mixed / uncertain.
- `expected_whole_frame_change`: likely / unlikely / uncertain.
- `possible_camera_motion`: yes / no / unknown.
- `input_matches_initial_state`: yes / no / uncertain.

These labels help distinguish **TC-Bench background shifts** from **whole-frame appearance transitions**. Do not exclude official entries on these labels without a separate, predeclared secondary experiment. Never use LoT-Pass accuracy to assign or change these labels.

### A7. Build preview and STOP FOR APPROVAL

Before expensive work, output:

- `contact_sheet_all_18.png`: all 18 official IDs; show extracted frames when available and explicit labeled placeholders for inaccessible clips, with shortened official prompt and initial-state note.
- `selection_review_all_18.md`: all 18 original prompts **verbatim**, all 18 drafted custom control prompts, source URL/interval, reference status, source-initial-state compatibility, expected-change labels, and problems requiring user decisions.
- `tcbench_all_18_images.csv` (18 entry-level rows) and `tcbench_all_36pairs.csv` (36 planned prompt pairs, 2 per official entry) with source paths where available, `runnable`/`missing_source` flags, stable official IDs.
- `dataset_census_manifest.json`, source hashes, and available/unavailable summary. No sampling metadata.

**STOP HERE.** Request user approval of retrieved images, source-to-prompt matching, and custom controls before Stage B. If assets are missing, preserve entries and explain that the runnable set has fewer than 18; do not automatically fill with unofficial images.

## 3. Stage B — After explicit approval ONLY

### B1. Reuse LoT-Pass encoding

1. Use **all approved, retrievable** model-ready first frames (up to 18 distinct official entries). If two official entries share the same image, maintain separate official records and note duplication; encoding may be cached by exact content hash only with explicit traceability and consistent messages.
2. Run the existing LoT-Pass encoder **once per unique source image**. Both transition/control conditions for that image must use the **identical watermarked image and encoded message**.
3. Before generation, verify encoded image raw-decoding performance; log baseline BA and failures without excluding them silently.
4. Preserve original and watermarked images, watermark message/map, checkpoint, preprocessing, and logs.
5. Run the existing encoder heatmap implementation once per unique watermarked input, reusing existing code.

### B2. Generate up to 36 I2V videos

1. Reuse the established CogVideoX-I2V model and runtime settings from VBench++. Keep frame dimensions, frame count, FPS, inference steps, CFG, and checkpoint constant where feasible.
2. Generate **one official transition clip + one custom no-transition control clip for each approved, available official entry** (up to **36 total**). Pair conditions using the same predeclared random seed (e.g., seed 42 for all), same source, watermark, model, and settings.
3. Preserve the official prompt verbatim. Do not append stronger transition instructions, camera constraints, or scene descriptions to it; control remains marked custom.
4. Maintain exact mappings `TC{official_id} -> source -> watermarked source -> condition/prompt -> generated clip -> BA/heatmaps`. Log checkpoint, prompt, seed, model version, output path, and failure status.
5. Implement robust resume and atomic completion markers; skip existing verified completed cases. Never overwrite VBench++ or earlier TC-Bench outputs.
6. Budget reference: **36 × ~15 min ≈ 9 hours** for I2V generation if all 18 cases are runnable, excluding preprocessing and analysis; actual runtime can vary. Log actual runtime.
7. Report total official entries, retrievable entries, approved entries, planned clips, successful clips, and failures as **separate denominators**.

### B3. LoT-Pass raw decoding

1. Run the existing raw decoder on every generated frame. Optical-flow alignment is **not part of this stage**.
2. Record per-frame BA, mean BA (excluding an exact conditioning frame if the model reproduces it), last-third BA, minimum BA, and early-to-late BA drop.
3. Pair each transition result with its corresponding same-source control result. Compute paired BA difference (`control - transition`) using the same frame aggregation.
4. Do not discard high-BA cases, failed transitions, unusual subjects, or generated clips that do not visually follow the prompt. Report generation success separately from watermark outcome.

### B4. Attribute/background transition validation

The official label `background shifts` does **not** guarantee whole-frame change; in fact some official cases depict a local event in front of a largely unchanged scene.

1. Review all available generated videos (up to 36); annotate whether the requested transition occurred: `valid`, `partial`, `invalid`, `uncertain`.
2. For lighting cases, measure whole-frame luminance and patch-level luminance change, without treating brightness alone as a complete success metric.
3. For weather/environment changes, use human/visual inspection first; optionally add a lightweight, already-available vision classifier or VLM for support, but do not make new heavyweight dependencies a prerequisite.
4. Record unintended camera drift or scene cuts; these can confound appearance analysis.
5. Report both (a) prompt-conditioned results over **all runnable official cases**, and (b) results conditioned on successfully realized transitions, with the denominator for each.
6. Keep the original descriptive `expected_whole_frame_change` flag separate from **observed** global-change coverage.

### B5. Stable-region analysis — SOURCE-ANCHORED ONLY

**Do not compute temporal stability**. Compare each generated frame at matching pixel coordinates against the **watermarked source image**, not the previous video frame.

1. Work with aligned dimensions/color conventions; keep any analysis resizing explicit and consistent.
2. Split into configurable **16 × 16 patches (default)**. Compute per-patch **normalized RGB MAE** and **SSIM**.
3. Save raw MAE maps and SSIM maps; derive a continuous, documented similarity score in `[0,1]` from a reproducible monotone mapping. Do not invent an undocumented combination formula; record all parameters.
4. Build binary stable masks using fixed, calibrated thresholds (initial candidates: `MAE < 0.05` and `SSIM > 0.90`, not universal truths). Keep the continuous maps so thresholds can be changed without rerunning I2V.
5. Compute per-frame `stable_region_ratio`, `largest_stable_region_ratio` (area of the largest connected component divided by full image area), and `mean_similarity`. Make connectivity convention explicit (4- or 8-connected).
6. Treat these as **same-coordinate appearance preservation** measurements; they do not prove that source content was lost, nor that watermark bits are spatially located there.

#### Required overlays (every generated video)

- `similarity_overlay.mp4`: original generated frame is clearly visible; color-map the **continuous similarity score**, with a visible legend (high similarity = cool colors/green; low = yellow/red). Use fixed **overlay opacity `alpha = 0.35` by default**, configurable. Alpha controls **visual blending only**, not similarity or confidence. Preserve frame order, dimensions, and FPS.
- `stable_mask_overlay.mp4`: original generated frame remains fully visible outside stable patches; mark only stable patches with semitransparent green (`alpha = 0.35` default), leaving other patches unchanged.
- Optionally save `.npy` raw patch maps, PNG masks, and selected preview frames. Raw metrics must be preserved independently from visualization.

### B6. Reuse existing Encoder / Decoder heatmaps

1. Locate and reuse the **existing** encoder and decoder localization implementations and their checkpoints. Do not implement alternative attribution or occlusion algorithms without approval.
2. Identify what each existing map actually represents (encoder residual, decoder gradient, occlusion sensitivity, etc.) and preserve that metadata.
3. Compute encoder heatmaps once per watermarked input. Compute the existing decoder attribution map for **every frame of every generated video**. Save each raw map and frame-level overlap metric; do not interpolate or reuse maps between frames.
4. Register all maps and stable masks in consistent pixel coordinates; preserve unnormalized numerical maps.
5. Compute encoder/decoder attribution overlap with stable regions, plus stable-region area-normalized enrichment if meaningful for the existing attribution definition. Use a small epsilon/NA rule for zero denominators.
6. Create readable comparison panels containing source, generated frame, stable mask, similarity overlay, encoder heatmap, decoder heatmap, and frame BA. Also render `decoder_heatmap_overlay.mp4` for each video, retaining frame order, dimensions, and FPS with the generated image visible under the heatmap at alpha `0.35` by default.
7. For all attribution and stable overlays, keep the actual image visible; `alpha = 0.35` is only a display setting.
8. If old heatmap scripts are unavailable or incompatible, report that fact and continue the other analyses; do not silently substitute an unrelated method.

### B7. Summary analysis and figures

Required analyses:

- Per-frame raw BA for transition and no-transition control, up to 18 paired cases (all official entries; report missing pairs).
- Paired transition vs control differences (same source, same encoded watermark).
- Descriptive category-level BA and failure-candidate counts across **all 18 official entries** (with missing-asset denominators reported); do not imply a reliable population-wide failure rate.
- Transition-success validation and realized whole-frame change statistics.
- Stable-region ratio / largest connected stable region / similarity vs BA.
- Existing encoder/decoder heatmap overlap with stable regions vs BA.
- Representative high-BA, low-BA, unsuccessful-transition, and surprising cases, **all flagged as examples rather than cherry-picked evidence**.

Candidate BA-drop criteria may use late BA < 0.80 or paired BA difference > 0.15, but label these **preliminary screening thresholds**, report their exact definitions, and preserve raw numbers. Do not treat all video frames as statistically independent observations. Even complete coverage of 18 official entries with one seed per entry supports claims only about **this benchmark category under these tested conditions**, not broad causal generalization to all global attribute changes. More seeds and independent datasets remain valuable. Do not conflate multiple frames with independent samples.

## 4. Files and folder structure

```text
tcbench_background_shifts/
  source_repo/                      # pinned TC-Bench source
  metadata/
    official_background_shifts_all.csv
    all_18_initial.csv
    dataset_census_manifest.json
    selection_review.md
  references/
    original_videos/               # optional, downloaded only as permitted
    reference_clips/
    source_frames_original/
    source_frames_model_ready/
  approval/
    contact_sheet_all_18.png
    tcbench_all_18_images.csv
    tcbench_all_36pairs.csv
    approval_record.md
  encoded/
    {case_id}/
      watermarked.png
      message_reference.npz
      encoder_heatmap.npy
  generations/
    {case_id}/
      transition/
        video.mp4
        run_config.json
      control/
        video.mp4
        run_config.json
  analysis/
    per_frame_raw_ba.csv
    paired_ba_summary.csv
    transition_validation.csv
    stable_region_metrics.csv
    heatmap_overlap.csv
    plots/
    stable_region/{case_id}/{condition}/
      similarity_overlay.mp4
      stable_mask_overlay.mp4
      raw_maps/
    heatmap_comparisons/
  reports/
    all_18_background_shifts_summary.md
    exceptions_and_missing_assets.md
  scripts/
```

IDs should be stable and traceable: e.g., `TC101` references official `id=101`, and filenames can include `TC101_transition` / `TC101_control`. Do **not** relabel official IDs as video1/video2. Paths in CSV must point to actual files.

## 5. Suggested CSV schema

Minimum unified columns in `tcbench_all_36pairs.csv`:

`pair_id, case_id, official_id, condition, is_official_prompt, prompt_official, prompt_control, prompt_used, reference_video_url, reference_start_sec, reference_end_sec, source_frame_time_original_seconds, source_image_original_path, source_image_model_ready_path, keywords, expected_change_mechanism, expected_whole_frame_change, possible_camera_motion, input_matches_initial_state, download_status, official_category_count, source_runnable, approval_status, control_initial_state, control_target_state, control_removed_transition, control_preserved_dynamics, control_needs_review, control_authoring_note, sample_status`

Later, append or join by pair ID: `generation_seed, generator_model, generation_success, observed_change_type, transition_validity, raw_BA_mean, raw_BA_last_third, control_minus_transition_BA, stable_region_ratio_mean, largest_stable_region_ratio_mean, encoder_stable_overlap, decoder_stable_overlap`.

## 6. Completion and acceptance criteria

**Stage A acceptance (then STOP):**

- The complete official `background shifts` category is loaded and the actual count verified (expected **18**). **No random sampling.**
- All official IDs have original prompts, source URL/time range, reference retrieval status, and descriptive change-mechanism labels.
- All available initial frames are correctly extracted and tied to official IDs; unavailable videos remain visible with exact errors, not replacements.
- The preview displays **all 18** official entries (image or missing-source placeholder), including source-image compatibility judgments.
- The agent has prepared **18 custom no-transition controls**, each preserving the initial background state but allowing natural dynamics, with any ambiguities flagged for user approval.
- A 36-row planned prompt-pair CSV is complete, explicitly marking unrunnable entries. The contact sheet and side-by-side prompt review are ready.
- **STOP; no watermark encoding or I2V generation before user approval.**

**Stage B acceptance (only after explicit Stage A approval):**

- All **approved retrievable** official entries are evaluated, up to 18 input images and 36 clips (18 official transitions + 18 custom controls). Report missing assets and failures instead of claiming complete coverage falsely.
- Paired images, LoT-Pass messages/seeds, and generation settings are controlled consistently. A control is **not** treated as a frozen video and is validated for absence of the target transition.
- Per-frame raw BA and paired BA differences; transition success, control validity, and observed global appearance-change coverage.
- Patch-based, source-anchored 16×16 stability metrics; `similarity_overlay.mp4` and `stable_mask_overlay.mp4` with alpha default 0.35; no temporal stability.
- Encoder/decoder heatmap reuse, per-frame overlap, representative comparison panels, and an all-frame decoder heatmap overlay video per generated video (or explicit error if old scripts are unavailable).
- Everything traceable to official source IDs, reference frame provenance, original and custom prompts, seeds, checkpoints, preprocessing, and errors.
- The report separates **TC-Bench official background-shifts category** from the narrower, *observed* **global appearance change** mechanism; it does not assume all 18 shifts affect the entire frame.

## 7. Non-goals and safeguards

- Do not run optical-flow recovery, camera-control methods, additional model variants, or multi-seed expansions in this first round.
- Do not rewrite official transition prompts to force a stronger transition.
- Do not sample or select images: include **every** official background-shifts entry, irrespective of anticipated/observed LoT-Pass failure.
- Do not replace failed downloads with unrelated images while still calling them official benchmark inputs.
- Do not report this as the original official TC-Bench first/last-frame I2V evaluation.
- Do not claim broad global-appearance causality simply because the entire 18-entry official category was covered once; include success/failure denominators and caveats.

## 8. Agent's required first response to the user

When Stage A finishes, report: official category count (expected 18), **all official IDs**, all-18 preview contact sheet, per-entry source retrieval/initial-state matching status, **18 unchanged official transition prompts and 18 clearly custom no-transition controls**, expected whole-frame-change labels, actual available/missing source count, 18-entry CSV and planned 36-pair CSV paths, and unresolved source/control ambiguities. **Then STOP for explicit user approval before Stage B.**
