"""Agent 2: Vision Analyst — PIL-based image metrics extractor."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter
from utils.logger import get_logger

log = get_logger(__name__)


class VisionAgent:
    """
    Extracts visual metrics from the pattern image using PIL + numpy.
    No neural network required — fully portable.
    Returns a dict of metrics for use by the Llama optimizer.
    """

    def analyze_image(self, image_path: str) -> dict:
        """
        Analyzes the pattern image and returns visual metrics.

        Returns:
            dict with keys:
              fg_coverage_pct   — % of non-white pixels (foreground)
              unique_colors     — number of unique RGB colors
              mean_brightness   — average pixel brightness 0-255
              color_std         — std deviation of brightness (contrast)
              edge_density_pct  — % of pixels that are edges
              dominant_rgb      — most frequent color as [R,G,B]
              width, height     — image dimensions
        """
        log.info("VisionAgent: analyzing image %s", image_path)
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            log.error("VisionAgent: cannot open image: %s", e)
            return self._empty_metrics()

        arr = np.array(img, dtype=np.uint8)
        h, w = arr.shape[:2]

        # ── Foreground coverage (non-white pixels) ──────────────────────────
        # White background = pixels where R>240 AND G>240 AND B>240
        is_white = (arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240)
        fg_pixels = int((~is_white).sum())
        total_pixels = h * w
        fg_coverage_pct = round(fg_pixels / total_pixels * 100, 2)

        # ── Unique colors ───────────────────────────────────────────────────
        pixels_flat = arr.reshape(-1, 3)
        # Pack RGB into int32 for fast unique count
        packed = (pixels_flat[:, 0].astype(np.int32) * 65536
                  + pixels_flat[:, 1].astype(np.int32) * 256
                  + pixels_flat[:, 2].astype(np.int32))
        unique_packed, counts = np.unique(packed, return_counts=True)
        unique_colors = int(len(unique_packed))

        # ── Dominant color (most frequent non-white) ────────────────────────
        # Filter out white-ish colors
        white_packed = 240 * 65536 + 240 * 256 + 240
        non_white_mask = unique_packed < white_packed
        if non_white_mask.any():
            best_idx = np.argmax(counts[non_white_mask])
            best_packed = unique_packed[non_white_mask][best_idx]
            dominant_rgb = [
                int((best_packed >> 16) & 0xFF),
                int((best_packed >> 8) & 0xFF),
                int(best_packed & 0xFF),
            ]
        else:
            dominant_rgb = [128, 128, 128]

        # ── Brightness and contrast ─────────────────────────────────────────
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        mean_brightness = float(round(gray.mean(), 1))
        color_std = float(round(gray.std(), 1))

        # ── Edge density (using PIL edge filter) ────────────────────────────
        gray_img = img.convert("L")
        edges = gray_img.filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(edges)
        edge_pixels = int((edge_arr > 30).sum())
        edge_density_pct = round(edge_pixels / total_pixels * 100, 2)

        metrics = {
            "fg_coverage_pct": fg_coverage_pct,
            "unique_colors": unique_colors,
            "mean_brightness": mean_brightness,
            "color_std": color_std,
            "edge_density_pct": edge_density_pct,
            "dominant_rgb": dominant_rgb,
            "width": w,
            "height": h,
        }

        log.info(
            "VisionAgent: fg=%.1f%% colors=%d brightness=%.0f std=%.1f edges=%.1f%%",
            fg_coverage_pct, unique_colors, mean_brightness, color_std, edge_density_pct
        )
        return metrics

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            "fg_coverage_pct": 0.0,
            "unique_colors": 0,
            "mean_brightness": 128.0,
            "color_std": 0.0,
            "edge_density_pct": 0.0,
            "dominant_rgb": [128, 128, 128],
            "width": 0,
            "height": 0,
        }
