"""STEP 4+5 — Resize to grid + Smart Color Quantization (CIELAB + KMeans)."""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from core.context import ProcessingContext, ClusterStats
from utils.logger import get_logger

log = get_logger(__name__)


def resize_to_grid(ctx: ProcessingContext, img: np.ndarray) -> np.ndarray:
    """
    STEP 4: Resize image so 1px = 1 stitch.
    Uses INTER_NEAREST for pixel-perfect stitch boundaries.
    Respects aspect ratio within ctx.grid_width / ctx.grid_height.
    """
    ctx.report_progress(4, "Resizing to stitch grid...")

    h, w = img.shape[:2]
    target_w = ctx.grid_width
    target_h = ctx.grid_height

    # Fit inside target grid preserving aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    log.info("Resize: %dx%d -> %dx%d stitches", w, h, new_w, new_h)

    # Resize foreground mask to match the new grid dimensions
    if ctx.fg_mask is not None:
        mask_u8 = ctx.fg_mask.astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        ctx.fg_mask = mask_resized > 127
        log.info("Resize: fg_mask resized to %dx%d, fg=%.1f%%",
                 new_w, new_h, 100.0 * ctx.fg_mask.mean())

    return resized


def quantize_colors(ctx: ProcessingContext, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    STEP 5: Reduce colors using KMeans in CIELAB space.
    Pre-quantization blur reduces noise so color regions are cleaner.

    Returns:
        quantized_bgr  — BGR image with reduced colors
        palette_lab    — (N, 3) array of cluster centers in LAB
    """
    ctx.report_progress(5, "Quantizing colors...")
    log.info("Quantize: input %dx%d, target ~%d colors", img.shape[1], img.shape[0], ctx.target_colors)

    # ── Pre-quantization blur — clean up noise before clustering ──────────────
    smooth = cv2.GaussianBlur(img, (3, 3), sigmaX=0.8)

    # ── Convert BGR -> LAB ────────────────────────────────────────────────────
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab.reshape(-1, 3)

    n_pixels = pixels.shape[0]
    n_colors = int(ctx.target_colors)
    n_colors = max(2, min(n_colors, 80))

    # Use MiniBatchKMeans for large images (faster, similar quality)
    if n_pixels > 50_000:
        kmeans = MiniBatchKMeans(
            n_clusters=n_colors,
            random_state=42,
            batch_size=min(10_000, n_pixels),
            n_init=3,
            max_iter=150,
        )
    else:
        kmeans = KMeans(
            n_clusters=n_colors,
            random_state=42,
            n_init=5,
            max_iter=200,
        )

    labels = kmeans.fit_predict(pixels)
    centers_lab = kmeans.cluster_centers_  # (N, 3) in LAB

    # Count unique colors actually used
    unique_labels = np.unique(labels)
    actual_colors = len(unique_labels)
    log.info("Quantize: %d colors -> %d actual clusters", n_colors, actual_colors)

    ctx.cluster_stats = ClusterStats(
        requested_colors=n_colors,
        actual_colors=actual_colors,
        inertia=float(kmeans.inertia_) if hasattr(kmeans, 'inertia_') else 0.0,
        iterations=int(kmeans.n_iter_) if hasattr(kmeans, 'n_iter_') else 0,
    )

    # ── Reconstruct quantized image ───────────────────────────────────────────
    quantized_lab = centers_lab[labels].reshape(img.shape).astype(np.uint8)
    quantized_bgr = cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2BGR)

    return quantized_bgr, centers_lab


def apply_dithering(img: np.ndarray, palette_lab: np.ndarray) -> np.ndarray:
    """
    Optional Floyd-Steinberg dithering for smoother color transitions.
    Works in LAB space for perceptually uniform diffusion.
    """
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    result_indices = np.zeros((h, w), dtype=np.int32)

    for y in range(h):
        for x in range(w):
            old_pixel = lab[y, x].copy()
            # Find nearest palette color
            diffs = palette_lab - old_pixel
            distances = np.sum(diffs ** 2, axis=1)
            nearest_idx = int(np.argmin(distances))
            new_pixel = palette_lab[nearest_idx]
            result_indices[y, x] = nearest_idx

            # Quantization error
            error = old_pixel - new_pixel

            # Diffuse error to neighbors (Floyd-Steinberg weights)
            if x + 1 < w:
                lab[y, x + 1] += error * (7 / 16)
            if y + 1 < h:
                if x > 0:
                    lab[y + 1, x - 1] += error * (3 / 16)
                lab[y + 1, x] += error * (5 / 16)
                if x + 1 < w:
                    lab[y + 1, x + 1] += error * (1 / 16)

    # Reconstruct image
    dithered_lab = palette_lab[result_indices].reshape(h, w, 3).astype(np.uint8)
    return cv2.cvtColor(dithered_lab, cv2.COLOR_LAB2BGR)
