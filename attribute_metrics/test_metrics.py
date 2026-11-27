"""Deterministic correctness checks for the attribute metric primitives."""

import unittest
from pathlib import Path

import numpy as np
import cv2
from scipy.ndimage import correlate
from skimage.feature import graycomatrix, graycoprops

from attribute_metrics.metrics import (
    appearance_coverage,
    color_histogram_wasserstein,
    cosine_similarity,
    glcm_features,
    ihd_scores,
    ihc_illumination,
    lab_patch_means,
    lpips_alex_v01,
    patch_delta_e,
    weighted_median,
)
from attribute_metrics.cli import _fill_lpips_rows


class MetricTests(unittest.TestCase):
    def test_ihd_identity_disjoint_histograms_and_single_frame(self):
        identity = ihd_scores(np.array([[1.0, 0.0], [1.0, 0.0]]))
        np.testing.assert_allclose(identity["ihd_mean_ref_norm"], [0, 0])
        np.testing.assert_allclose(identity["ihd_ref0_norm"], [0, 0])
        disjoint = ihd_scores(np.array([[1.0, 0.0], [0.0, 1.0]]))
        np.testing.assert_allclose(disjoint["ihd_mean_ref_norm"], [0.5, 0.5])
        self.assertAlmostEqual(disjoint["ihd_video_norm"], 0.5)
        np.testing.assert_allclose(disjoint["ihd_ref0_norm"], [0, 1])
        self.assertAlmostEqual(disjoint["ihd_ref0_norm_mean"], 1.0)
        single = ihd_scores(np.array([[0.25, 0.75]]))
        self.assertTrue(np.isnan(single["ihd_ref0_norm_mean"]))

    def test_ihc_illumination_uses_red_channel_full_kernel_and_nearest_padding(self):
        rgb = np.zeros((5, 8, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(40, dtype=np.uint8).reshape(5, 8) * 5
        rgb[..., 1] = 220
        rgb[..., 2] = 120
        image = rgb[..., 0].astype(np.float64)
        y = np.arange(5, dtype=np.float64) - 2
        x = np.arange(8, dtype=np.float64) - 3.5
        kernel = np.exp(-(y[:, None] ** 2 + x[None, :] ** 2) / (2 * 80.0**2))
        kernel /= kernel.sum()
        expected = np.floor(correlate(image, kernel, mode="nearest", origin=0) + 0.5).astype(np.uint8)
        positive = expected[expected > 0]
        expected[expected == 0] = positive.min()
        np.testing.assert_array_equal(ihc_illumination(rgb, working_side=8), expected)

    def test_ihc_replaces_quantized_zero_with_minimum_positive(self):
        rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        rgb[:, :256, 0] = 255
        illumination = ihc_illumination(rgb)
        self.assertGreater(int(illumination.min()), 0)
        self.assertEqual(int(illumination.min()), 1)

    def test_color_histogram_wasserstein_uses_weighted_rgb_support(self):
        reference = np.zeros((2, 3, 3), dtype=np.uint8)
        changed = reference.copy()
        changed[..., 0] = 255
        values = color_histogram_wasserstein(reference, changed)
        self.assertAlmostEqual(values["color_w1_r"], 1.0)
        self.assertAlmostEqual(values["color_w1_g"], 0.0)
        self.assertAlmostEqual(values["color_w1_b"], 0.0)
        self.assertAlmostEqual(values["color_w1_mean"], 1.0 / 3)
        np.testing.assert_allclose(list(color_histogram_wasserstein(reference, reference).values()), [0, 0, 0, 0])

    def test_glcm_20_features_identity_and_cosine(self):
        image = np.zeros((5, 7, 3), dtype=np.uint8)
        image[::2, 1::2] = 255
        features = glcm_features(image)
        self.assertEqual(features.shape, (20,))
        self.assertTrue(np.isfinite(features).all())
        self.assertAlmostEqual(cosine_similarity(features, features), 1.0)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        matrix = graycomatrix(
            gray, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256, symmetric=False, normed=True,
        )
        expected = np.concatenate([
            graycoprops(matrix, name).reshape(-1)
            for name in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
        ])
        np.testing.assert_array_equal(features, expected)

    def test_ciede2000_sharma_reference_pairs(self):
        # Published Sharma et al. test pairs (the first three hue-wrap cases).
        lab1 = np.array([
            [50, 2.6772, -79.7751],
            [50, 3.1571, -77.2803],
            [50, 2.8361, -74.0200],
            [50, -1.3802, -84.2814],
        ])
        lab2 = np.array([[50, 0.0, -82.7485]] * 4)
        stats, scores = patch_delta_e(lab1, lab2, np.ones(4))
        np.testing.assert_allclose(scores, [2.0425, 2.8615, 3.4412, 1.0], atol=5e-5)
        self.assertAlmostEqual(stats["patch_de00_mean"], scores.mean())

    def test_patch_grid_area_weighting_and_coverage_threshold_edges(self):
        image = np.zeros((17, 17, 3), dtype=np.uint8)
        means, weights = lab_patch_means(image)
        np.testing.assert_array_equal(weights, [256, 16, 16, 1])
        self.assertEqual(int(weights.sum()), 17 * 17)
        stats, identical_scores = patch_delta_e(means, means, weights)
        self.assertEqual(stats["patch_de00_mean"], 0.0)
        self.assertEqual(stats["patch_de00_median"], 0.0)
        np.testing.assert_array_equal(identical_scores, np.zeros(4))

        values = appearance_coverage(
            np.array([3.0, 3.01, 5.0, 20.0]),
            np.array([256, 16, 16, 1]),
            [3, 5, 20],
        )
        self.assertAlmostEqual(values["appearance_coverage_de00_gt_3"], 33 / 289)
        self.assertAlmostEqual(values["appearance_coverage_de00_gt_5"], 1 / 289)
        self.assertEqual(values["appearance_coverage_de00_gt_20"], 0.0)
        self.assertEqual(weighted_median(np.array([1.0, 3.0]), np.array([16, 16])), 1.0)

    def test_lpips_wrapper_matches_official_package_when_weights_are_cached(self):
        checkpoint = Path.home() / ".cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
        if not checkpoint.is_file():
            self.skipTest("official AlexNet weights are not cached; avoid a network download in tests")
        import lpips
        import torch

        model = lpips.LPIPS(net="alex", version="0.1", pretrained=True).eval().to("cpu")
        reference = np.full((128, 128, 3), 64, dtype=np.uint8)
        changed = np.full((128, 128, 3), 96, dtype=np.uint8)
        row = {"frame_idx": 1}
        errors = []
        _fill_lpips_rows([(1, changed)], [row], reference, model, torch, "cpu", "fixture", errors)
        reference_tensor = torch.from_numpy(reference.copy()).permute(2, 0, 1).unsqueeze(0).float()
        changed_tensor = torch.from_numpy(changed.copy()).permute(2, 0, 1).unsqueeze(0).float()
        reference_tensor = reference_tensor.div(127.5).sub(1.0)
        changed_tensor = changed_tensor.div(127.5).sub(1.0)
        with torch.inference_mode():
            direct = float(model(reference_tensor, changed_tensor).item())
            identity = float(model(reference_tensor, reference_tensor).item())
        self.assertFalse(errors)
        self.assertAlmostEqual(row["lpips_alex_v01"], direct, places=7)
        self.assertAlmostEqual(lpips_alex_v01(reference, changed, model), direct, places=7)
        self.assertAlmostEqual(identity, 0.0, places=7)
        self.assertGreater(direct, 0.0)


if __name__ == "__main__":
    unittest.main()
