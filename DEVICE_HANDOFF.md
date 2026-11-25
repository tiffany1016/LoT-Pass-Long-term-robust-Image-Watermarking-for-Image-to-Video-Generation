# LoT-Pass repository handoff

This note is for continuing work on another device with the same repository. It records the project structure and experiment workflow, but intentionally contains no validation results, generated data, or result tables.

## Repository snapshot

- Project: official implementation of **LoT-Pass: Long-term-robust Image Watermarking for Image to Video Generation** (paper linked in `README.md`).
- Handoff snapshot: branch `main`, commit `37d5dcd` (`Add custom LoT-Pass experiments`), based on upstream commit `b210bd1`.
- The working tree was clean before adding this handoff file.
- `data/`, `checkpoints/`, and `Validation_Results/` are git-ignored. The new device will need the appropriate source data and checkpoints supplied separately; this note does not contain them.

## Code map

- `test_I2V.py`: image watermark encoding, I2V generation entry point, and video decoding (raw and optical-flow/RAFT aligned).
- `models/`: encoder/decoder architectures and JND module. `utils/`: data loading, video generation, noise handling, config helpers, and training parameters.
- `Noise_option/`, `noise_opt.yml`: image perturbation/noise operators and settings for traditional robustness experiments.
- `video_config.yml`: I2V model selection and generation defaults. Current selection is CogVideoX at 480p, guidance 5, 50 steps, 8 fps, quality 10.
- `run_large_motion_exp2.py`, `large_motion_exp2_prompt_list.md`: staged large-motion experiment workflow and its prompt catalog.
- `run_mask_ablation_generation.py`, `run_mask_ablation_c1a.py`: CogVideoX generation for the regional-residual mask ablation, including a corrected/regenerated `car1__C1-A` input/prompt.
- `gradient_saliency.py`, `encoder_heatmap.py`: decoder gradient-saliency/localization and encoder residual-energy analysis.
- `plot_exp2_bit_accuracy.py`, `plot_seed_bit_accuracy.py`, `summarize_motion_seeds.py`: analyses for prompt-wise video accuracy and generation-seed comparisons.

## Experiments represented in this checkout

### Existing motion seed comparison

The analysis scripts expect three source prompts (`s1_crouch`, `s1_run_exit`, `s1_walk`) under `data/video_prompts.json`, with runs for seeds 42, 43, and 44. Seed 42 uses `Validation_Results/Motion_CogVideoX`; seeds 43 and 44 use `Motion_CogVideoX_seed43` and `Motion_CogVideoX_seed44`. Each run has raw decode maps and RAFT-aligned decode maps. `summarize_motion_seeds.py` writes CSV summaries; `plot_seed_bit_accuracy.py` writes plots. These scripts analyze existing run outputs; they do not generate the seed-43/44 runs.

### Large-motion Experiment 2

The catalog pairs 18 source PNGs (`data/motion_sources_2`) with 42 complete prompts, covering people, animals, cars, birds, and other large subject/background motions. It omits the incomplete `C2-C` catalog entry and corrects the tennis-ball scale-down prompt ID from `T1-C` to `T2-C`. The intended protocol is one 32-bit watermarked encoding per source image, reused for every prompt for that image; encoder input resolution is 256. CogVideoX is `THUDM/CogVideoX-5b-I2V`, generated at 720x480 / 49 frames / 8 fps, with seed 42, guidance 5, and 50 inference steps.

The staged runner defaults to checkpoint `checkpoints/model-99.pth` and output directory `Validation_Results/Motion_Exp2_CogVideoX_seed42`:

```bash
python run_large_motion_exp2.py --stage prepare
python run_large_motion_exp2.py --stage encode
python run_large_motion_exp2.py --stage generate
python run_large_motion_exp2.py --stage decode
python plot_exp2_bit_accuracy.py
```

The stages create a prompt manifest and run metadata, encode sources/messages, generate prompt-specific videos, then decode both raw and RAFT-aligned frames. Generation can skip already existing videos; preparation guards against unapproved edits to an existing prompt manifest. Pass `--replace-pair PAIR_ID` only when intentionally updating that prompt's metadata. All paths and default assumptions can be overridden with the runner's CLI options.

### Regional residual mask ablation

The generation script expects prepared files under `Validation_Results/Motion_Mask_Ablation_CogVideoX_seed42/car1`, including `generation_prompts.json`, `Generation_Input/Encoded/{no_sky,no_mountain,no_car,no_road_all}.png`, and watermark messages in `message_map.npz`. It generates four CogVideoX videos using seed 42. `run_mask_ablation_c1a.py` separately regenerates `car1__C1-A` with the annotation-matched drive-in-frame prompt and maps it to the `no_sky` message. These scripts rely on ablation inputs from the earlier mask-preparation workflow; they do not construct all mask inputs from scratch.

## Setup and portability notes

- Follow `README.md` for the upstream dataset/checkpoint locations. The repository README has spelling/path inconsistencies, so use the actual paths expected by the scripts and check each script's CLI/defaults.
- `requirements.txt` pins the environment (notably PyTorch 2.6.0, torchvision 0.21.0, diffusers 0.35.0, transformers 4.51.3). Install a PyTorch/CUDA build compatible with the new machine before running GPU workflows; CogVideoX model weights may need to be downloaded/cached separately.
- Main experiment runners assume a CUDA device. The old machine's physical GPU number recorded in some metadata is machine-specific; choose the appropriate device on the new machine.
- Keep source images, checkpoints, generated videos, decode maps, and analysis outputs in their expected ignored directories. This handoff intentionally does not preserve any of those files or report their outcomes.
- To understand the current experiment implementation, start with `run_large_motion_exp2.py` and its prompt catalog, then inspect `test_I2V.py` and `utils/video_generation.py` for encoding/decoding and generator seeding behavior.

## Relevant implementation details

- The latest commit makes CogVideoX input ordering deterministic, supports per-call generation seeds and skipping existing videos, and can report each completed video.
- Video decode now sorts video paths, saves corresponding filenames in the `.npz` accuracy maps, reads `train_param.json` beside the checkpoint, and can skip the auxiliary SD-VAE similarity metric when the model cache is unavailable. The experiment runner enables that skip.
- `utils/dataloader.py` converts source images to RGB before transforms, which avoids mode/channel mismatches for non-RGB inputs.

