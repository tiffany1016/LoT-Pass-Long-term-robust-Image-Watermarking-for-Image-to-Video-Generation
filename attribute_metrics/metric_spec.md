# Metric Specification and Source Notes

This evaluator processes raw generated video frames only. It decodes every frame in presentation order, preserves zero-based frame indices, converts decoded frames to RGB uint8, and requires constant frame geometry within each clip. No alignment, segmentation, exposure correction, frame dropping, or per-frame normalization is performed. Frame 0 of each generated clip is the reference for frame-referenced measures.

## Metric definitions

### Normalized Illumination Histogram Difference (IHD)

The illumination preprocessing follows the released [IHC-Metric MATLAB code](https://github.com/LongChenCV/IHC-Metric/blob/main/SSR_Code.m): apply an image-sized Gaussian kernel with sigma 80 to the first (red) RGB channel using replicate boundaries, and quantize the blurred image to uint8 using round-to-nearest with positive half-values rounded up. The released code replaces zero-valued blurred samples with the minimum positive blurred value before saving. The script's illumination export is this blurred red channel, not its computed log-difference reflectance image. The evaluator keeps the Gaussian's full image-sized support and uses two separable 1D convolutions, mathematically equivalent to the 2D Gaussian. For a non-square frame, IHD first resizes the RGB image to fit a 512-pixel maximum side while preserving aspect ratio (OpenCV INTER_AREA when downscaling and INTER_CUBIC when upscaling); it adds no padding. The effective working dimensions are recorded per frame as ihd_working_width and ihd_working_height.

The released [histogram code](https://github.com/LongChenCV/IHC-Metric/blob/main/IHC_histogram.py) uses 255 bins over [0,255] and a hard-coded 512*512 denominator. This implementation uses the same bin count and support, includes the rightmost 255 value in the final bin, and normalizes by the actual number of pixels. Histograms are unit mass.

For frame histograms h_t, it exports:

- ihd_mean_ref_norm[t] = 0.5 * sum(abs(h_t - mean(h))); ihd_video_norm averages this over all decoded frames, including frame 0.
- ihd_ref0_norm[t] = 0.5 * sum(abs(h_t - h_0)); ihd_ref0_norm_mean averages frames 1 onward and is unavailable for a one-frame clip.

These are normalized total-variation distances: zero means identical histograms and one means disjoint histograms. The mean-reference variant is an IHD adaptation with explicit additional normalization by 2; the frame-0 version is a custom temporal extension. Neither IHC score is computed.

The paper [Eq. (6)](https://arxiv.org/html/2405.09716#S2) defines IHC = 2 - IHD; the released Python code uses IHC = 1 - IHD. The released MATLAB code's zero handling applies to reflectance and the blurred map after blur, with the latter saved as illumination. If quantization leaves no positive blurred value, the released code's minimum-positive replacement is undefined; this implementation marks that frame's IHD unavailable. MATLAB numerical parity was not run in this environment; the separable Gaussian port follows the visible kernel, padding, channel, and quantization semantics, but exact numerical parity remains unverified.

### Color histogram Wasserstein distance

For each RGB channel, compute a 256-bin histogram and use 256 fixed support positions uniformly spaced in [0,1]. Pass histogram probabilities as SciPy wasserstein_distance weights. Export color_w1_r, color_w1_g, color_w1_b, and their arithmetic mean color_w1_mean, comparing each frame with frame 0. Larger values indicate greater marginal color-distribution change. This is a full-frame adaptation of VAMP [Section 3.2.1, Eqs. (1)-(2)](https://arxiv.org/html/2411.13609v2#S3.SS2.SSS1), not its object-tracked adjacent-frame score. RGB marginals discard spatial layout and cross-channel dependencies.

### GLCM texture similarity

Convert RGB uint8 to grayscale using OpenCV COLOR_RGB2GRAY, then call scikit-image graycomatrix with levels 256, distance 1, angles 0, pi/4, pi/2, and 3pi/4, symmetric=False, and normed=True. Call graycoprops for contrast, dissimilarity, homogeneity, energy, and correlation; concatenate the four orientations for each property in that order into a 20-value vector. Export those feature values, cosine similarity to the frame-0 vector (glcm_texture_similarity), and 1 - similarity (glcm_texture_distance). An undefined zero-norm cosine is marked unavailable and logged.

This adapts VAMP [Section 3.2.1, Eq. (6)](https://arxiv.org/html/2411.13609v2#S3.SS2.SSS1) from resized object pairs to full native-resolution frames. The feature vector is not standardized, so contrast can dominate. Grayscale texture responds to illumination; a one-pixel offset is resolution-dependent.

### Patch CIEDE2000

Convert each frame from sRGB uint8 to float sRGB in [0,1], then to CIELAB with scikit-image rgb2lab (D65 illuminant, 2-degree observer). Split into non-overlapping 16x16 patches anchored at the upper left; retain partial bottom/right edge patches without padding. Compute each patch's mean Lab color, then compare each frame's mean to frame 0 with deltaE_ciede2000(kL=1,kC=1,kH=1). This patchwise temporal aggregation is a project adaptation, not a standard video metric.

Export pixel-area-weighted patch_de00_mean and patch_de00_median. The weighted median is the smallest sorted score whose cumulative patch area reaches at least half the image area. Larger scores mean greater same-location color difference; scores are not normalized.

### Raw appearance change coverage

Reuse the patch CIEDE2000 scores and real pixel areas. For each threshold tau, coverage is the fraction of image pixels in patches with score strictly greater than tau. Default thresholds are 3, 5, 10, and 20; they are configurable. Export appearance_coverage_de00_gt_3, _gt_5, _gt_10, and _gt_20. Values are in [0,1]; coverage is area weighted and non-increasing with threshold. This custom metric measures extent of raw appearance change, not cause or motion-compensated change.

### LPIPS

Use the official [lpips package](https://github.com/richzhang/PerceptualSimilarity), AlexNet backbone, version 0.1, calibrated pretrained weights, eval mode, and inference mode. Convert RGB uint8 to NCHW float in [-1,1]; compare every frame to frame 0 with no alignment or mask. Export lpips_alex_v01; larger values mean greater perceptual difference. LPIPS is not assumed to be bounded by one or specific to attributes. Native resolution is retained; if the model cannot process an input geometry, that frame's LPIPS value is unavailable and the error is recorded.

## Input and output behavior

The CSV manifest requires unique clip_id and video_path fields; relative paths resolve from the manifest directory. Explicit --videos paths use their resolved paths as IDs. No directory recursion is performed. Results are emitted to an explicit output directory. CSV blanks represent unavailable values; JSON metadata uses null. Errors are kept in errors.jsonl; a clip may retain successful metrics when another family fails. A decode failure marks the clip failed and keeps any already decoded frame results.

Per-video frame-0-reference means exclude frame 0. IHD's ihd_video_norm includes frame 0, and ihd_ref0_norm_mean excludes it. Single-frame temporal means are unavailable. No combined score is produced. Resolution differences are recorded per clip because GLCM distance and patch scale are in pixels.

The configuration records selected metrics, thresholds, patch size, IHD working side, decoder, device, dependency versions, input paths, and SHA-256 video hashes. A run resumes from per-clip checkpoints only when the configuration and input hashes match. Use a separate output directory for another configuration.

## Primary sources

- Chen et al. (2024), *Illumination Histogram Consistency Metric for Quantitative Assessment of Video Sequences*, Section II, Eqs. (1)-(6): https://arxiv.org/html/2405.09716
- IHC-Metric MATLAB illumination source: https://github.com/LongChenCV/IHC-Metric/blob/main/SSR_Code.m
- IHC-Metric histogram source: https://github.com/LongChenCV/IHC-Metric/blob/main/IHC_histogram.py
- Wang et al. (2024), *What You See Is What Matters: A Novel Visual and Physics-Based Metric for Evaluating Video Generation Quality* (VAMP), Section 3.2.1, Eqs. (1), (2), (6): https://arxiv.org/html/2411.13609v2
- Rubner et al. (2000), *The Earth Mover's Distance as a Metric for Image Retrieval*: https://doi.org/10.1023/A:1026543900054
- SciPy wasserstein_distance: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
- Haralick et al. (1973), *Textural Features for Image Classification*: https://doi.org/10.1109/TSMC.1973.4309314
- scikit-image GLCM API: https://scikit-image.org/docs/stable/api/skimage.feature.html
- Sharma, Wu, and Dalal (2005), *The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary Test Data, and Mathematical Observations*: https://doi.org/10.1002/col.20070
- Sharma et al. supplementary data: https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/
- scikit-image color API: https://scikit-image.org/docs/stable/api/skimage.color.html
- Zhang et al. (2018), *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*: https://arxiv.org/abs/1801.03924
- Official LPIPS implementation: https://github.com/richzhang/PerceptualSimilarity

Library versions are recorded in run_config.json for each run. Source repositories were reviewed on 2026-11-27; their mutable main branch revision was not pinned in the available source view.
