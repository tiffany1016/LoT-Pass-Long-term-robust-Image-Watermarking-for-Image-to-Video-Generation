"""Metric primitives. Images are uint8 RGB arrays unless documented otherwise."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from scipy.ndimage import correlate1d
from scipy.stats import wasserstein_distance
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.feature import graycomatrix, graycoprops

METRIC_FAMILIES = (
    "ihd",
    "color_w1",
    "glcm_texture",
    "patch_de00",
    "appearance_coverage",
    "lpips",
)
GLCM_PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
DEFAULT_COVERAGE_THRESHOLDS = (3.0, 5.0, 10.0, 20.0)


def _rgb8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an HxWx3 uint8 RGB image")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("image dimensions must be nonzero")
    return image


def ihd_working_size(width: int, height: int, working_side: int = 512) -> tuple[int, int]:
    if width <= 0 or height <= 0 or working_side <= 0:
        raise ValueError("width, height, and IHD working side must be positive")
    scale = working_side / max(height, width)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def ihc_illumination(image_rgb: np.ndarray, working_side: int = 512) -> np.ndarray:
    """Port IHC-Metric's exported illumination map (blurred red channel)."""
    image = _rgb8(image_rgb)
    height, width = image.shape[:2]
    target_w, target_h = ihd_working_size(width, height, working_side)
    scale = working_side / max(height, width)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    working = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    red = working[..., 0].astype(np.float64)

    # MATLAB fspecial('gaussian',[height,width],80), normalized to unit sum.
    y = np.arange(target_h, dtype=np.float64) - (target_h - 1) / 2
    x = np.arange(target_w, dtype=np.float64) - (target_w - 1) / 2
    kernel_y = np.exp(-(y**2) / (2 * 80.0**2))
    kernel_x = np.exp(-(x**2) / (2 * 80.0**2))
    kernel_y /= kernel_y.sum()
    kernel_x /= kernel_x.sum()
    # The 2D Gaussian is exactly separable; keep its image-sized support while
    # avoiding a quadratic 2D convolution for every decoded frame.
    blurred = correlate1d(red, kernel_y, axis=0, mode="nearest", origin=0)
    blurred = correlate1d(blurred, kernel_x, axis=1, mode="nearest", origin=0)
    if not np.isfinite(blurred).all():
        raise ValueError("IHC illumination contains non-finite values")
    # MATLAB uint8 conversion rounds positive ties away from zero. The
    # released code then replaces zero-valued blurred samples before export.
    illumination = np.floor(np.clip(blurred, 0, 255) + 0.5).astype(np.uint8)
    positive = illumination[illumination > 0]
    if positive.size == 0:
        raise ValueError("IHC illumination has no positive uint8 samples after blur")
    illumination[illumination == 0] = positive.min()
    return illumination


def illumination_histogram(image_rgb: np.ndarray, working_side: int = 512) -> np.ndarray:
    illumination = ihc_illumination(image_rgb, working_side=working_side)
    # Equivalent to 255 bins over [0,255], including 255 in the last bin.
    counts, _ = np.histogram(illumination, bins=255, range=(0, 255))
    histogram = counts.astype(np.float64) / illumination.size
    if not np.isclose(histogram.sum(), 1.0, rtol=0, atol=1e-12):
        raise ArithmeticError("illumination histogram does not have unit mass")
    return histogram


def ihd_scores(histograms: np.ndarray) -> dict[str, np.ndarray | float]:
    """Return per-frame normalized mean-reference and frame-0 IHD values."""
    histograms = np.asarray(histograms, dtype=np.float64)
    if histograms.ndim != 2 or histograms.shape[0] == 0:
        raise ValueError("histograms must be a nonempty frame-by-bin matrix")
    if not np.isfinite(histograms).all() or np.any(histograms < 0):
        raise ValueError("histograms must contain finite nonnegative probabilities")
    if not np.allclose(histograms.sum(axis=1), 1.0, rtol=0, atol=1e-10):
        raise ValueError("each histogram must have unit mass")
    mean_reference = 0.5 * np.abs(histograms - histograms.mean(axis=0)).sum(axis=1)
    frame0_reference = 0.5 * np.abs(histograms - histograms[0]).sum(axis=1)
    return {
        "ihd_mean_ref_norm": mean_reference,
        "ihd_video_norm": float(mean_reference.mean()),
        "ihd_ref0_norm": frame0_reference,
        "ihd_ref0_norm_mean": float(frame0_reference[1:].mean()) if len(histograms) > 1 else float("nan"),
    }


def color_histogram_wasserstein(reference_rgb: np.ndarray, image_rgb: np.ndarray) -> dict[str, float]:
    reference = _rgb8(reference_rgb)
    image = _rgb8(image_rgb)
    if reference.shape != image.shape:
        raise ValueError("reference and image geometry must match")
    support = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    distances: dict[str, float] = {}
    for channel, name in enumerate(("r", "g", "b")):
        ref_hist = np.bincount(reference[..., channel].ravel(), minlength=256).astype(np.float64)
        cur_hist = np.bincount(image[..., channel].ravel(), minlength=256).astype(np.float64)
        ref_hist /= ref_hist.sum()
        cur_hist /= cur_hist.sum()
        distances[f"color_w1_{name}"] = float(
            wasserstein_distance(support, support, u_weights=ref_hist, v_weights=cur_hist)
        )
    distances["color_w1_mean"] = float(np.mean(tuple(distances.values())))
    return distances


def glcm_features(image_rgb: np.ndarray) -> np.ndarray:
    image = _rgb8(image_rgb)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    matrix = graycomatrix(
        gray,
        distances=[1],
        angles=GLCM_ANGLES,
        levels=256,
        symmetric=False,
        normed=True,
    )
    return np.concatenate(
        [graycoprops(matrix, prop).reshape(-1) for prop in GLCM_PROPERTIES]
    ).astype(np.float64)


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).ravel()
    second = np.asarray(second, dtype=np.float64).ravel()
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("cosine feature vectors must be finite and have equal shape")
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0:
        raise ValueError("GLCM cosine similarity is undefined for a zero feature vector")
    return float(np.dot(first, second) / denominator)


def lab_patch_means(image_rgb: np.ndarray, patch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    image = _rgb8(image_rgb)
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    lab = rgb2lab(image.astype(np.float64) / 255.0, illuminant="D65", observer="2")
    height, width = image.shape[:2]
    means: list[np.ndarray] = []
    weights: list[int] = []
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            patch = lab[top : top + patch_size, left : left + patch_size]
            means.append(patch.mean(axis=(0, 1)))
            weights.append(int(patch.shape[0] * patch.shape[1]))
    return np.stack(means), np.asarray(weights, dtype=np.int64)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if values.shape != weights.shape or values.size == 0 or np.any(weights <= 0):
        raise ValueError("weighted median requires matching nonempty values and positive weights")
    order = np.argsort(values, kind="stable")
    ordered_values, ordered_weights = values[order], weights[order]
    index = int(np.searchsorted(np.cumsum(ordered_weights), weights.sum() / 2, side="left"))
    return float(ordered_values[index])


def patch_delta_e(
    reference_means: np.ndarray,
    image_means: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    reference_means = np.asarray(reference_means, dtype=np.float64)
    image_means = np.asarray(image_means, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if reference_means.shape != image_means.shape or reference_means.ndim != 2 or reference_means.shape[1] != 3:
        raise ValueError("patch means must have matching shape (patches, 3)")
    if weights.shape != (reference_means.shape[0],) or np.any(weights <= 0):
        raise ValueError("one positive pixel-area weight is required per patch")
    scores = np.asarray(
        deltaE_ciede2000(reference_means, image_means, kL=1, kC=1, kH=1, channel_axis=-1),
        dtype=np.float64,
    )
    return {
        "patch_de00_mean": float(np.average(scores, weights=weights)),
        "patch_de00_median": weighted_median(scores, weights),
    }, scores


def appearance_coverage(
    patch_scores: np.ndarray,
    patch_weights: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_COVERAGE_THRESHOLDS,
) -> dict[str, float]:
    scores = np.asarray(patch_scores, dtype=np.float64).ravel()
    weights = np.asarray(patch_weights, dtype=np.float64).ravel()
    thresholds = tuple(float(value) for value in thresholds)
    if scores.shape != weights.shape or scores.size == 0 or np.any(weights <= 0):
        raise ValueError("coverage requires one positive pixel-area weight per patch")
    if not np.isfinite(scores).all() or any(not np.isfinite(t) or t < 0 for t in thresholds):
        raise ValueError("patch scores and thresholds must be finite and thresholds nonnegative")
    total = weights.sum()
    result = {
        f"appearance_coverage_de00_gt_{threshold:g}": float(weights[scores > threshold].sum() / total)
        for threshold in thresholds
    }
    ordered = sorted(result.items(), key=lambda item: float(item[0].rsplit("_", 1)[1]))
    if any(ordered[i][1] < ordered[i + 1][1] for i in range(len(ordered) - 1)):
        raise ArithmeticError("coverage must be non-increasing as the threshold increases")
    return result


def lpips_alex_v01(
    reference_rgb: np.ndarray,
    image_rgb: np.ndarray,
    model: object,
    device: str = "cpu",
) -> float:
    """Score a frame pair with a caller-loaded official LPIPS AlexNet v0.1 model."""
    reference = _rgb8(reference_rgb)
    image = _rgb8(image_rgb)
    if reference.shape != image.shape:
        raise ValueError("reference and image geometry must match")
    import torch

    reference_tensor = torch.from_numpy(reference.copy()).permute(2, 0, 1).unsqueeze(0)
    image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
    reference_tensor = reference_tensor.to(device=device, dtype=torch.float32).div_(127.5).sub_(1.0)
    image_tensor = image_tensor.to(device=device, dtype=torch.float32).div_(127.5).sub_(1.0)
    model.eval()
    with torch.inference_mode():
        return float(model(reference_tensor, image_tensor).reshape(-1)[0].detach().cpu())
