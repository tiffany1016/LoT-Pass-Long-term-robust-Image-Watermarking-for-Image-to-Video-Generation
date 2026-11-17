#!/usr/bin/env python3
"""Regenerate C1-A for the annotation-matched car1 mask-ablation source."""
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel

from utils.video_generation import choose_resolution
from utils.yml import NoiseOptions


ROOT = Path("Validation_Results/Motion_Mask_Ablation_CogVideoX_seed42/car1")
MODEL = "THUDM/CogVideoX-5b-I2V"
PROMPT = (
    "The yellow sports car drives smoothly along the curved mountain road toward the right, "
    "maintaining its original shape and appearance. The car remains visible throughout the video. "
    "The camera stays completely stationary."
)
NEGATIVE = (
    "Vibrant colors, overexposed, static, blurry details, subtitles, style, artwork, painting, scene, "
    "static, overall grayish, poor quality, low quality, JPEG compression artifacts, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused "
    "fingers, static scene, cluttered background, three legs, many people in the background, walking backward"
)


def main() -> None:
    name = "car1__C1-A"
    image_path = ROOT / "Generation_Input/Encoded" / f"{name}.png"
    image_path.write_bytes((ROOT / "Encoded_Source/car1.png").read_bytes())
    with np.load(ROOT / "message_map.npz", allow_pickle=True) as archive:
        messages = {key: np.array(archive[key]) for key in archive.files}
    if "no_sky" not in messages:
        raise KeyError("Expected the ablation message map to contain no_sky")
    messages[name] = messages["no_sky"]
    np.savez(ROOT / "message_map.npz", **messages)
    prompt_path = ROOT / "c1a_prompt.json"
    prompt_path.write_text(json.dumps({name: PROMPT}, indent=2))

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
    image = load_image(str(image_path)).resize(choose_resolution(setting["resolution"]))
    frames = pipe(
        image=image,
        prompt=PROMPT,
        guidance_scale=setting["guidance_scale"],
        use_dynamic_cfg=True,
        num_inference_steps=setting["inference_steps"],
        negative_prompt=NEGATIVE,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]
    target = ROOT / "Videos/CogVideoX" / f"{name}.mp4"
    export_to_video(frames, str(target), fps=setting["fps"], quality=setting["quality"])
    print(f"[PAIR_COMPLETE] {name} {target}", flush=True)


if __name__ == "__main__":
    main()
