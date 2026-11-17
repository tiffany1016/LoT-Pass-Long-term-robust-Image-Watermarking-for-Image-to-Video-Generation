#!/usr/bin/env python3
"""Generate the four regional-residual ablation videos with CogVideoX."""
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel

from utils.video_generation import choose_resolution
from utils.yml import NoiseOptions


ROOT = Path("Validation_Results/Motion_Mask_Ablation_CogVideoX_seed42/car1")
MODEL = "THUDM/CogVideoX-5b-I2V"
NEGATIVE = (
    "Vibrant colors, overexposed, static, blurry details, subtitles, style, artwork, painting, scene, "
    "static, overall grayish, poor quality, low quality, JPEG compression artifacts, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused "
    "fingers, static scene, cluttered background, three legs, many people in the background, walking backward"
)


def main() -> None:
    device = torch.device("cuda")
    names = ["no_sky", "no_mountain", "no_car", "no_road_all"]
    prompts = json.loads((ROOT / "generation_prompts.json").read_text())
    setting = NoiseOptions("video_config.yml").get_noise_opt()["model"]["CogVideoX"]

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        MODEL, subfolder="transformer", torch_dtype=torch.float16
    )
    text_encoder = T5EncoderModel.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKLCogVideoX.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float16)
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        MODEL, text_encoder=text_encoder, transformer=transformer, vae=vae, torch_dtype=torch.float16
    )
    pipe.enable_sequential_cpu_offload()
    resolution = choose_resolution(setting["resolution"])
    video_dir = ROOT / "Videos/CogVideoX"
    video_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        target = video_dir / f"{name}.mp4"
        if target.exists():
            print(f"[SKIP_EXISTS] {target}", flush=True)
            continue
        image_path = ROOT / "Generation_Input/Encoded" / f"{name}.png"
        image = load_image(str(image_path)).resize(resolution)
        generator = torch.Generator(device=device).manual_seed(42)
        frames = pipe(
            image=image,
            prompt=prompts[name],
            guidance_scale=setting["guidance_scale"],
            use_dynamic_cfg=True,
            num_inference_steps=setting["inference_steps"],
            negative_prompt=NEGATIVE,
            generator=generator,
        ).frames[0]
        export_to_video(frames, str(target), fps=setting["fps"], quality=setting["quality"])
        print(f"[PAIR_COMPLETE] {name} {target}", flush=True)


if __name__ == "__main__":
    main()
