#!/usr/bin/env python3
"""Prepare and run the seed-42 large-motion I2V watermark experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_HEADING = re.compile(r"^### `([^`]+)`$")
PROMPT_HEADING = re.compile(r"^\*\*([A-Z0-9]+-[A-Z]) — (.+)\*\*$")
EXPECTED_IMAGES = 18
EXPECTED_PAIRS = 42
SEED = 42
INPUT_RES = 256
MESSAGE_BITS = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_prompt_catalog(catalog: Path, source_dir: Path) -> list[dict[str, str]]:
    lines = catalog.read_text(encoding="utf-8").splitlines()
    source_name: str | None = None
    pairs: list[dict[str, str]] = []
    for index, line in enumerate(lines):
        image_match = IMAGE_HEADING.match(line.strip())
        if image_match:
            source_name = image_match.group(1)
            continue
        prompt_match = PROMPT_HEADING.match(line.strip())
        if not prompt_match or source_name is None:
            continue

        prompt_id, title = prompt_match.groups()
        opening = None
        for pos in range(index + 1, len(lines)):
            candidate = lines[pos].strip()
            if IMAGE_HEADING.match(candidate) or PROMPT_HEADING.match(candidate):
                break
            if candidate == "```text":
                opening = pos
                break
        if opening is None:
            continue  # Catalog entries without prompt text, e.g. optional C2-C.
        closing = next(
            (pos for pos in range(opening + 1, len(lines)) if lines[pos].strip().startswith("``")),
            None,
        )
        if closing is None:
            raise ValueError(f"Unclosed prompt block for {source_name} / {prompt_id}")
        prompt = "\n".join(lines[opening + 1 : closing]).strip()
        if not prompt:
            continue

        # The catalog's tennis-ball scale-down entry is mislabeled T1-C.
        if source_name == "thing2.png" and prompt_id == "T1-C":
            prompt_id = "T2-C"
        source_path = source_dir / source_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Prompt source image not found: {source_path}")
        pair_id = f"{Path(source_name).stem}__{prompt_id}"
        pairs.append(
            {
                "pair_id": pair_id,
                "source_image": source_name,
                "prompt_id": prompt_id,
                "title": title,
                "prompt": prompt,
            }
        )

    image_files = sorted(path.name for path in source_dir.glob("*.png"))
    used_images = {pair["source_image"] for pair in pairs}
    if len(image_files) != EXPECTED_IMAGES:
        raise ValueError(f"Expected {EXPECTED_IMAGES} source PNGs; found {len(image_files)}")
    if used_images != set(image_files):
        raise ValueError(f"Catalog/source mismatch: unused images={sorted(set(image_files)-used_images)}")
    if len(pairs) != EXPECTED_PAIRS:
        raise ValueError(f"Expected {EXPECTED_PAIRS} complete prompt pairs; parsed {len(pairs)}")
    ids = [pair["pair_id"] for pair in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("Prompt catalog produced duplicate pair IDs")
    return pairs


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> None:
    pairs = parse_prompt_catalog(args.prompt_list, args.source_dir)
    previous_metadata = {}
    if args.run_dir.exists():
        existing = args.run_dir / "prompt_manifest.json"
        if existing.exists():
            previous = json.loads(existing.read_text(encoding="utf-8"))
            current_by_id = {pair["pair_id"]: pair for pair in pairs}
            previous_by_id = {pair["pair_id"]: pair for pair in previous}
            if not set(previous_by_id).issubset(current_by_id):
                raise FileExistsError(f"The new catalog removes existing pairs from {args.run_dir}")
            changed = {pair_id for pair_id, pair in previous_by_id.items() if current_by_id[pair_id] != pair}
            allowed_changes = set(args.replace_pair)
            if changed != allowed_changes:
                raise FileExistsError(
                    f"Prompt changes {sorted(changed)} do not match --replace-pair values {sorted(allowed_changes)}"
                )
            metadata_path = args.run_dir / "experiment_metadata.json"
            if metadata_path.exists():
                previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        elif any(args.run_dir.iterdir()):
            raise FileExistsError(f"Refusing to prepare over existing run data: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    json_write(args.run_dir / "prompt_manifest.json", pairs)
    json_write(args.run_dir / "generation_prompts.json", {pair["pair_id"]: pair["prompt"] for pair in pairs})
    shutil.copy2("video_config.yml", args.run_dir / "video_config.yml")

    metadata = {
        "run_id": "large_motion_exp2_seed42",
        "seed": SEED,
        "source_directory": str(args.source_dir),
        "prompt_catalog": str(args.prompt_list),
        "skipped_incomplete_prompt_ids": ["C2-C"],
        "corrected_catalog_prompt_id": {"source_image": "thing2.png", "from": "T1-C", "to": "T2-C"},
        "num_source_images": EXPECTED_IMAGES,
        "num_prompt_pairs": len(pairs),
        "source_image_sha256": {
            name: sha256(args.source_dir / name) for name in sorted({p["source_image"] for p in pairs})
        },
        "generation": {
            "model_id": "THUDM/CogVideoX-5b-I2V",
            "resolution": "480p (720x480)",
            "guidance_scale": 5.0,
            "num_inference_steps": 50,
            "num_frames": 49,
            "fps": 8,
            "quality": 10,
        },
        "watermark": {
            "checkpoint": str(args.checkpoint),
            "message_bits": MESSAGE_BITS,
            "input_resolution": INPUT_RES,
            "reuse_policy": "one encoded source image/message reused for all prompts for that source",
        },
        "pairs": pairs,
    }
    for key in ("encoded_source_sha256", "message_map_sha256"):
        if key in previous_metadata:
            metadata[key] = previous_metadata[key]
    json_write(args.run_dir / "experiment_metadata.json", metadata)
    print(f"Prepared {len(pairs)} prompt pairs from {EXPECTED_IMAGES} images in {args.run_dir}")


def encode(args: argparse.Namespace) -> None:
    run_dir = args.run_dir
    manifest = json.loads((run_dir / "prompt_manifest.json").read_text(encoding="utf-8"))
    source_encoded = run_dir / "Encoded_Source"
    pair_encoded = run_dir / "Generation_Input" / "Encoded"
    message_map_path = run_dir / "message_map.npz"
    source_map_path = run_dir / "source_message_map.npz"
    if source_encoded.exists() or pair_encoded.exists() or message_map_path.exists():
        if not source_encoded.is_dir() or not pair_encoded.is_dir() or not message_map_path.is_file() or not source_map_path.is_file():
            raise FileExistsError(f"Encoded data is incomplete under {run_dir}; refusing to re-encode or overwrite")
        with np.load(source_map_path, allow_pickle=True) as archive:
            source_messages = {name: np.array(archive[name]) for name in archive.files}
        with np.load(message_map_path, allow_pickle=True) as archive:
            pair_messages = {name: np.array(archive[name]) for name in archive.files}
        expected = {f"{pair['pair_id']}.png" for pair in manifest}
        existing = {path.name for path in pair_encoded.glob("*.png")}
        if not existing.issubset(expected):
            raise ValueError(f"Unexpected generation inputs: {sorted(existing-expected)}")
        updated = False
        for pair in manifest:
            pair_id = pair["pair_id"]
            source_stem = Path(pair["source_image"]).stem
            if source_stem not in source_messages or not (source_encoded / f"{source_stem}.png").is_file():
                raise KeyError(f"No encoded watermark source for {source_stem}")
            if not (pair_encoded / f"{pair_id}.png").is_file():
                shutil.copy2(source_encoded / f"{source_stem}.png", pair_encoded / f"{pair_id}.png")
                updated = True
            if pair_id not in pair_messages:
                pair_messages[pair_id] = source_messages[source_stem]
                updated = True
        if updated:
            np.savez(message_map_path, **pair_messages)
        metadata_path = run_dir / "experiment_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["message_map_sha256"] = sha256(message_map_path)
        json_write(metadata_path, metadata)
        print(f"Reused encoded source watermarks; prepared {len(manifest)} prompt-specific inputs")
        return

    import torch
    from test_I2V import encode_unfix
    from utils.utils import setup_seed

    setup_seed(SEED)
    encode_unfix(
        str(args.source_dir), str(run_dir), torch.device("cuda"),
        ckpt=str(args.checkpoint), input_res=INPUT_RES, message=MESSAGE_BITS,
    )
    generated_source = run_dir / "Encoded"
    if not generated_source.is_dir():
        raise FileNotFoundError(f"Encoder did not produce {generated_source}")
    generated_source.rename(source_encoded)

    with np.load(run_dir / "message_map.npz", allow_pickle=True) as source_map:
        source_messages = {name: np.array(source_map[name]) for name in source_map.files}
    for pair in manifest:
        source_stem = Path(pair["source_image"]).stem
        if source_stem not in source_messages:
            raise KeyError(f"No encoded watermark message for source {source_stem}")
    shutil.copy2(run_dir / "message_map.npz", run_dir / "source_message_map.npz")
    pair_encoded.mkdir(parents=True)
    pair_messages: dict[str, Any] = {"ckpt": source_messages["ckpt"]}
    for pair in manifest:
        source_stem = Path(pair["source_image"]).stem
        shutil.copy2(source_encoded / f"{source_stem}.png", pair_encoded / f"{pair['pair_id']}.png")
        pair_messages[pair["pair_id"]] = source_messages[source_stem]
    np.savez(run_dir / "message_map.npz", **pair_messages)

    metadata_path = run_dir / "experiment_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["encoded_source_sha256"] = {
        source: sha256(source_encoded / f"{Path(source).stem}.png")
        for source in sorted({pair["source_image"] for pair in manifest})
    }
    metadata["message_map_sha256"] = sha256(run_dir / "message_map.npz")
    json_write(metadata_path, metadata)
    print(f"Encoded {EXPECTED_IMAGES} sources; prepared {len(manifest)} prompt-specific inputs")


def generate(args: argparse.Namespace) -> None:
    run_dir = args.run_dir
    manifest = json.loads((run_dir / "prompt_manifest.json").read_text(encoding="utf-8"))
    input_dir = run_dir / "Generation_Input" / "Encoded"
    output_dir = run_dir / "Videos" / "CogVideoX"
    expected = {f"{pair['pair_id']}.png" for pair in manifest}
    actual = {path.name for path in input_dir.glob("*.png")}
    if actual != expected:
        raise ValueError(f"Generation input mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    existing = {path.name for path in output_dir.glob("*.mp4")} if output_dir.exists() else set()
    expected_outputs = {f"{pair['pair_id']}.mp4" for pair in manifest}
    if not existing.issubset(expected_outputs):
        raise ValueError(f"Unexpected existing videos under {output_dir}: {sorted(existing-expected_outputs)}")
    if existing == expected_outputs:
        print(f"All {len(manifest)} videos are already present; nothing to generate")
        return

    import torch
    from utils.video_generation import choose_resolution, test_CogVideoX
    from utils.yml import NoiseOptions

    setting = NoiseOptions("video_config.yml").get_noise_opt()["model"]["CogVideoX"]
    completed = {pair["pair_id"]: pair for pair in manifest}
    seen = len(existing)

    def on_complete(filename: str, output_path: str) -> None:
        nonlocal seen
        pair_id = Path(filename).stem
        pair = completed[pair_id]
        seen += 1
        print(
            f"[PAIR_COMPLETE] {seen}/{len(manifest)} {pair['source_image']} · {pair['prompt_id']} · {output_path}",
            flush=True,
        )

    test_CogVideoX(
        input_image_path=str(input_dir),
        out_video_path=str(run_dir / "Videos"),
        resolution=choose_resolution(setting["resolution"]),
        guidance_scale=setting["guidance_scale"],
        steps=setting["inference_steps"], fps=setting["fps"], quality=setting["quality"],
        test_num=len(manifest), device=torch.device("cuda"), seed=SEED,
        prompt=str(run_dir / "generation_prompts.json"), on_video_complete=on_complete, skip_existing=True,
    )
    if seen != len(manifest):
        raise RuntimeError(f"Only {seen}/{len(manifest)} prompt videos completed")
    print(f"Generated all {seen} videos")


def decode(args: argparse.Namespace) -> None:
    import torch
    from test_I2V import decode_Videos, decode_Videos_opticflow

    device = torch.device("cuda")
    # The SD-VAE is only used for an auxiliary similarity metric, not decoding.
    # Skip that metric locally when its model cache is unavailable.
    common = {"input_res": INPUT_RES, "message": MESSAGE_BITS, "skip_vae_metric": True}
    decode_Videos(str(args.run_dir), None, device, **common)
    decode_Videos_opticflow(str(args.run_dir), None, device, **common)
    print("Completed raw decode and RAFT alignment/decode")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "encode", "generate", "decode"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/motion_sources_2"))
    parser.add_argument("--prompt-list", type=Path, default=Path("large_motion_exp2_prompt_list.md"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model-99.pth"))
    parser.add_argument("--run-dir", type=Path, default=Path("Validation_Results/Motion_Exp2_CogVideoX_seed42"))
    parser.add_argument(
        "--replace-pair",
        action="append",
        default=[],
        help="Allow this existing pair's prompt metadata to be replaced during prepare.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "encode":
        encode(args)
    elif args.stage == "generate":
        generate(args)
    elif args.stage == "decode":
        decode(args)


if __name__ == "__main__":
    main()
