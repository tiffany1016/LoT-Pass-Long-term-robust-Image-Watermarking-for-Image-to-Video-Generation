import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models import convnext
from utils.train_param import TrainParam
from utils.video_generation import load_video, export_to_video


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def find_train_param(ckpt_path, datapath):
    """
    LoT-Pass repo 不同 function 對 train_param.json 的相對位置
    寫法有些不一致，因此這裡自動搜尋幾個合理位置。
    """
    ckpt_path = Path(ckpt_path)

    candidates = [
        ckpt_path.parent / "train_param.json",
        ckpt_path.parent.parent / "train_param.json",
        Path(datapath) / "train_param.json",
        Path("train_param.json"),
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "找不到 train_param.json。\n"
        "已搜尋：\n" + "\n".join(str(p) for p in candidates)
    )


def load_decoder(ckpt_path, datapath, device):
    param_path = find_train_param(ckpt_path, datapath)

    print(f"[INFO] checkpoint: {ckpt_path}")
    print(f"[INFO] train_param: {param_path}")

    with open(param_path, "r") as f:
        param_dict = json.load(f)

    params = TrainParam(**param_dict)

    decoder = convnext.ConvNextDecoder(params)

    ckpt = torch.load(
        ckpt_path,
        map_location=torch.device("cpu")
    )

    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    decoder.to(device)

    # 我們只需要 input gradient，不需要 model parameter gradient
    for p in decoder.parameters():
        p.requires_grad_(False)

    return decoder


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_bit_accuracy(message, decoded):
    """
    跟 LoT-Pass 官方 calculate_acc 的概念相同：
    decoder output -> round -> clip(0,1)
    """
    pred = (
        decoded.detach()
        .cpu()
        .numpy()
        .round()
        .clip(0, 1)
    )

    gt = message.detach().cpu().numpy()

    return 1.0 - np.mean(np.abs(pred - gt))


# ---------------------------------------------------------
# Gradient saliency
# ---------------------------------------------------------

def compute_saliency(
    decoder,
    image_tensor,
    message,
    target_bit=-1
):
    """
    image_tensor:
        [1, 3, H, W], 已經 normalization

    message:
        [1, message_length]

    target_bit:
        -1: 對整個 watermark message 計算 aggregate saliency
        >=0: 只看某一個 bit

    回傳:
        decoded
        saliency [H, W]
    """

    # 讓 decoder input 可以累積 gradient
    image_tensor = image_tensor.clone().detach()
    image_tensor.requires_grad_(True)

    decoded = decoder(image_tensor)

    # message: 0 -> -1
    #          1 -> +1
    #
    # 目的：
    # bit=1 時，希望 decoder output 越大越好
    # bit=0 時，希望 decoder output 越小越好
    sign = message * 2.0 - 1.0

    if target_bit >= 0:
        if target_bit >= decoded.shape[1]:
            raise ValueError(
                f"target_bit={target_bit}, "
                f"但 decoder 只有 {decoded.shape[1]} bits"
            )

        score = (
            sign[:, target_bit]
            * decoded[:, target_bit]
        ).mean()

    else:
        # 所有 bits 一起考慮
        score = (sign * decoded).mean()

    # 對 input image 求 gradient
    score.backward()

    grad = image_tensor.grad.detach()

    # RGB channel 取平均 magnitude
    saliency = grad.abs().mean(dim=1)[0]

    saliency = saliency.cpu().numpy()

    return decoded.detach(), saliency


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

def normalize_saliency(saliency):
    """
    percentile normalization 比單純 max normalization 穩定，
    避免少數超大 gradient 把其他地方全部壓黑。
    """

    saliency = np.maximum(saliency, 0)

    low = np.percentile(saliency, 1)
    high = np.percentile(saliency, 99)

    saliency = (saliency - low) / (high - low + 1e-8)
    saliency = np.clip(saliency, 0, 1)

    return saliency


def make_visualization(frame, saliency, alpha=0.45):
    """
    frame: PIL image
    saliency: decoder input resolution 的 heatmap
    """

    frame_np = np.array(frame.convert("RGB"))

    h, w = frame_np.shape[:2]

    saliency = normalize_saliency(saliency)

    # resize 回原始影片大小
    saliency_large = cv2.resize(
        saliency,
        (w, h),
        interpolation=cv2.INTER_CUBIC
    )

    # 稍微 smooth，讓空間結構容易觀察
    saliency_large = cv2.GaussianBlur(
        saliency_large,
        (0, 0),
        sigmaX=5
    )

    heat_uint8 = np.uint8(
        np.clip(saliency_large * 255, 0, 255)
    )

    # OpenCV heatmap
    heatmap_bgr = cv2.applyColorMap(
        heat_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB
    )

    overlay = (
        (1 - alpha) * frame_np.astype(np.float32)
        + alpha * heatmap_rgb.astype(np.float32)
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    return (
        Image.fromarray(heatmap_rgb),
        Image.fromarray(overlay)
    )


# ---------------------------------------------------------
# Main video processing
# ---------------------------------------------------------

def process_video(
    video_path,
    message,
    decoder,
    transform,
    device,
    output_dir,
    fps=8,
    target_bit=-1
):
    video_name = Path(video_path).stem

    print()
    print("=" * 70)
    print(f"[VIDEO] {video_path}")
    print("=" * 70)

    video = load_video(video_path)

    save_dir = Path(output_dir) / video_name
    heatmap_dir = save_dir / "heatmaps"
    overlay_dir = save_dir / "overlays"
    npy_dir = save_dir / "saliency_npy"

    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    overlay_frames = []

    rows = []

    for frame_idx, frame in enumerate(video):

        image = transform(frame).unsqueeze(0).to(device)

        # 千萬不要包 torch.no_grad()
        decoded, saliency = compute_saliency(
            decoder=decoder,
            image_tensor=image,
            message=message,
            target_bit=target_bit,
        )

        acc = calculate_bit_accuracy(
            message,
            decoded
        )

        heatmap, overlay = make_visualization(
            frame,
            saliency
        )

        heatmap.save(
            heatmap_dir / f"frame_{frame_idx:03d}.png"
        )

        overlay.save(
            overlay_dir / f"frame_{frame_idx:03d}.png"
        )

        # 留 raw gradient，之後做 quantitative analysis 很重要
        np.save(
            npy_dir / f"frame_{frame_idx:03d}.npy",
            saliency
        )

        overlay_frames.append(overlay)

        rows.append(
            [
                frame_idx,
                acc,
                float(saliency.mean()),
                float(saliency.max()),
                float(np.percentile(saliency, 95)),
            ]
        )

        print(
            f"Frame {frame_idx:03d} | "
            f"Acc={acc:.4f} | "
            f"GradMean={saliency.mean():.6e}"
        )

    # -----------------------------------------------------
    # save metrics
    # -----------------------------------------------------

    csv_path = save_dir / "frame_metrics.csv"

    with open(csv_path, "w") as f:
        f.write(
            "frame,bit_accuracy,"
            "gradient_mean,gradient_max,gradient_p95\n"
        )

        for row in rows:
            f.write(
                ",".join(map(str, row))
                + "\n"
            )

    # -----------------------------------------------------
    # save overlay video
    # -----------------------------------------------------

    output_video = save_dir / f"{video_name}_gradient_overlay.mp4"

    export_to_video(
        overlay_frames,
        str(output_video),
        fps=fps
    )

    print(f"[SAVE] {output_video}")


# ---------------------------------------------------------
# Entry
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datapath",
        required=True,
        type=str,
        help="Validation_Results/VideoXXXX..."
    )

    parser.add_argument(
        "--ckpt",
        default=None,
        type=str,
        help="Optional. 不給時從 message_map.npz 取得。"
    )

    parser.add_argument(
        "--method",
        default=None,
        type=str,
        help="只跑 Videos 下某個 method，例如 CogVideoX"
    )

    parser.add_argument(
        "--video",
        default=None,
        type=str,
        help="只跑指定影片，例如 s1_run_exit.mp4"
    )

    parser.add_argument(
        "--videos",
        nargs="+",
        help="只跑指定影片清單，例如 --videos s1_walk.mp4 s1_crouch.mp4"
    )

    parser.add_argument(
        "--aligned",
        action="store_true",
        help="分析 RAFT-aligned videos under Aligned/ instead of Videos/"
    )

    parser.add_argument(
        "--input_res",
        default=256,
        type=int
    )

    parser.add_argument(
        "--fps",
        default=8,
        type=int
    )

    parser.add_argument(
        "--bit",
        default=-1,
        type=int,
        help="-1 = 所有 bits；例如 7 = 只看 bit 7"
    )

    args = parser.parse_args()
    if args.video and args.videos:
        raise ValueError("Use either --video or --videos, not both")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"[DEVICE] {device}")

    # -----------------------------------------------------
    # message map
    # -----------------------------------------------------

    message_map_path = os.path.join(
        args.datapath,
        "message_map.npz"
    )

    message_map = np.load(
        message_map_path,
        allow_pickle=True
    )

    if args.ckpt is None:
        ckpt_path = message_map["ckpt"].item()
    else:
        ckpt_path = args.ckpt

    # -----------------------------------------------------
    # Decoder
    # -----------------------------------------------------

    decoder = load_decoder(
        ckpt_path,
        args.datapath,
        device
    )

    # 官方 test_I2V.py 使用相同 preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (args.input_res, args.input_res)
        ),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ),
    ])

    video_parent = "Aligned" if args.aligned else "Videos"
    videos_root = Path(args.datapath) / video_parent

    if not videos_root.exists():
        raise FileNotFoundError(
            f"找不到 {videos_root}"
        )

    # -----------------------------------------------------
    # method folders
    # -----------------------------------------------------

    if args.method is not None:
        method_dirs = [
            videos_root / args.method
        ]
    else:
        method_dirs = [
            p for p in videos_root.iterdir()
            if p.is_dir()
        ]

    for method_dir in method_dirs:

        if not method_dir.exists():
            print(
                f"[WARNING] method 不存在：{method_dir}"
            )
            continue

        print(f"\n[METHOD] {method_dir.name}")

        output_root = Path(args.datapath) / (
            "Gradient_Saliency_Aligned" if args.aligned else "Gradient_Saliency"
        ) / method_dir.name

        output_root.mkdir(
            parents=True,
            exist_ok=True
        )

        video_paths = sorted(
            list(method_dir.glob("*.mp4"))
        )

        if args.video is not None:
            video_paths = [
                p for p in video_paths
                if p.name == args.video
            ]
        if args.videos is not None:
            requested_videos = set(args.videos)
            video_paths = [p for p in video_paths if p.name in requested_videos]
            missing_videos = requested_videos - {p.name for p in video_paths}
            if missing_videos:
                raise FileNotFoundError(f"Videos not found under {method_dir}: {sorted(missing_videos)}")

        for video_path in video_paths:

            video_id = video_path.stem.removesuffix("_aligned")

            if video_id not in message_map:
                print(
                    f"[WARNING] message_map 找不到 "
                    f"{video_id}，skip"
                )
                continue

            message_np = np.array(
                message_map[video_id],
                dtype=np.float32
            )

            message = torch.tensor(
                message_np,
                device=device
            ).unsqueeze(0)

            process_video(
                video_path=str(video_path),
                message=message,
                decoder=decoder,
                transform=transform,
                device=device,
                output_dir=output_root,
                fps=args.fps,
                target_bit=args.bit
            )


if __name__ == "__main__":
    main()
