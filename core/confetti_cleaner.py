"""STEP 7 — Confetti Cleaner: merge/remove isolated pixel islands < threshold."""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from core.context import ProcessingContext, ConfettiReport
from utils.logger import get_logger

log = get_logger(__name__)

MIN_ISLAND_SIZE = 3          # islands smaller than this are candidates
DELTA_E_SAFE_MERGE = 15.0   # only merge if closest neighbor is within this Delta-E


def clean_confetti(
    ctx: ProcessingContext,
    mapped_bgr: np.ndarray,
    palette_colors: list,
) -> np.ndarray:
    """
    STEP 7: Find and eliminate confetti (isolated pixel islands).

    Strategy per island:
      1. If island Delta-E to nearest neighbor < DELTA_E_SAFE_MERGE → merge.
      2. Otherwise (contrasting color) → keep, just note it.

    Returns cleaned BGR image.
    """
    ctx.report_progress(7, "Cleaning confetti...")
    log.info("Confetti: input %dx%d", mapped_bgr.shape[1], mapped_bgr.shape[0])

    result = mapped_bgr.copy()
    h, w = result.shape[:2]


    unique_bgr, inverse = _get_unique_colors(result)
    color_map = inverse.reshape(h, w)  # (H, W) each pixel = color index

    total_found = 0
    total_merged = 0
    total_removed = 0
    details = []

    for color_idx in range(len(unique_bgr)):
        mask = (color_map == color_idx).astype(np.uint8)


        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        for label in range(1, num_labels):  # skip background (label 0)
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= MIN_ISLAND_SIZE:
                continue

            total_found += 1
            island_mask = (labels == label)


            neighbor_color_idx = _get_dominant_neighbor_color(
                color_map, island_mask, color_idx
            )

            if neighbor_color_idx is None:
                continue


            island_bgr = unique_bgr[color_idx]
            neighbor_bgr = unique_bgr[neighbor_color_idx]
            de = _bgr_delta_e(island_bgr, neighbor_bgr)

            if de < DELTA_E_SAFE_MERGE:

                result[island_mask] = neighbor_bgr
                color_map[island_mask] = neighbor_color_idx
                total_merged += 1
            else:


                result[island_mask] = neighbor_bgr
                color_map[island_mask] = neighbor_color_idx
                total_removed += 1

    ctx.confetti_report = ConfettiReport(
        total_islands_found=total_found,
        islands_merged=total_merged,
        islands_removed=total_removed,
    )

    log.info("Confetti: found=%d merged=%d removed=%d",
             total_found, total_merged, total_removed)
    return result




def _get_unique_colors(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (unique_bgr array, inverse index per pixel)."""
    pixels = img.reshape(-1, 3)

    packed = (pixels[:, 0].astype(np.int32)
              + pixels[:, 1].astype(np.int32) * 256
              + pixels[:, 2].astype(np.int32) * 65536)
    unique_packed, inverse = np.unique(packed, return_inverse=True)
    unique_bgr = np.stack([
        unique_packed & 0xFF,
        (unique_packed >> 8) & 0xFF,
        (unique_packed >> 16) & 0xFF,
    ], axis=1).astype(np.uint8)
    return unique_bgr, inverse


def _get_dominant_neighbor_color(
    color_map: np.ndarray,
    island_mask: np.ndarray,
    island_color_idx: int,
) -> int | None:
    """Find the most common adjacent color around the island."""
    h, w = color_map.shape
    dilated = cv2.dilate(
        island_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    ).astype(bool)

    border = dilated & ~island_mask
    if not border.any():
        return None

    neighbor_indices = color_map[border]
    neighbor_indices = neighbor_indices[neighbor_indices != island_color_idx]
    if len(neighbor_indices) == 0:
        return None

    values, counts = np.unique(neighbor_indices, return_counts=True)
    return int(values[np.argmax(counts)])


def _bgr_delta_e(bgr1: np.ndarray, bgr2: np.ndarray) -> float:
    """Quick Delta-E approximation via LAB Euclidean distance (sufficient for threshold check)."""
    img1 = np.array([[bgr1]], dtype=np.uint8)
    img2 = np.array([[bgr2]], dtype=np.uint8)
    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]

    l1, a1, b1 = lab1[0] * 100 / 255, lab1[1] - 128, lab1[2] - 128
    l2, a2, b2 = lab2[0] * 100 / 255, lab2[1] - 128, lab2[2] - 128
    return float(np.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2))
