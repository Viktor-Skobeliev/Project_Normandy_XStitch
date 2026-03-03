"""STEP 6 — Delta-E 2000 mapping: match quantized colors to thread brand palette."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np

from core.context import ProcessingContext, ThreadColor, ColorStats
from core.exceptions import PaletteNotFoundError
from utils.logger import get_logger

log = get_logger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_PALETTES_FILE = os.path.join(_DATA_DIR, "palettes.json")

_palettes_cache: dict | None = None


def _load_palettes() -> dict:
    global _palettes_cache
    if _palettes_cache is None:
        with open(_PALETTES_FILE, "r", encoding="utf-8") as f:
            _palettes_cache = json.load(f)
    return _palettes_cache


def map_to_palette(
    ctx: ProcessingContext,
    quantized_bgr: np.ndarray,
    palette_lab: np.ndarray,
) -> Tuple[np.ndarray, List[ThreadColor], Dict[int, int]]:
    """
    STEP 6: Map each KMeans cluster center to the nearest thread color
    using Delta-E 2000 distance in CIELAB space.

    Returns:
        mapped_bgr      — BGR image with palette-matched colors
        palette_colors  — list of ThreadColor for used colors
        cluster_to_id   — map from cluster index to color ID (1-based)
    """
    ctx.report_progress(6, "Mapping colors to palette...")

    palettes = _load_palettes()
    brand_key = ctx.palette_selected

    if brand_key not in palettes["brands"]:
        available = list(palettes["brands"].keys())
        raise PaletteNotFoundError(
            f"Brand '{brand_key}' not found. Available: {available}"
        )

    brand_colors = palettes["brands"][brand_key]["colors"]
    log.info("Palette mapper: brand=%s, %d colors available, %d clusters",
             brand_key, len(brand_colors), len(palette_lab))

    # ── Convert brand palette to LAB ─────────────────────────────────────────
    brand_lab = _brand_to_lab(brand_colors)  # (M, 3)

    # ── Match each cluster center to nearest brand color ──────────────────────
    cluster_to_color: Dict[int, ThreadColor] = {}
    cluster_to_id: Dict[int, int] = {}
    used_colors: List[ThreadColor] = []
    used_ids: set = set()

    for cluster_idx, lab_pixel in enumerate(palette_lab):
        nearest_brand_idx = _nearest_delta_e2000(lab_pixel, brand_lab)
        brand = brand_colors[nearest_brand_idx]

        color = ThreadColor(
            brand=brand_key,
            code=brand["code"],
            name=brand["name"],
            rgb=brand["rgb"],
        )
        color_key = brand["code"]

        if color_key not in used_ids:
            color_id = len(used_colors) + 1
            used_colors.append(color)
            used_ids.add(color_key)
        else:
            color_id = next(
                i + 1 for i, c in enumerate(used_colors) if c.code == color_key
            )

        cluster_to_color[cluster_idx] = color
        cluster_to_id[cluster_idx] = color_id

    # ── Rebuild image with exact brand RGB values ─────────────────────────────
    h, w = quantized_bgr.shape[:2]
    orig_lab = cv2.cvtColor(quantized_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = orig_lab.reshape(-1, 3)

    # For each pixel, find nearest cluster center, then map to brand color
    mapped_pixels = np.zeros((pixels.shape[0], 3), dtype=np.uint8)
    for cluster_idx, color in cluster_to_color.items():
        # Find pixels belonging to this cluster
        center = palette_lab[cluster_idx]
        diffs = np.sum((pixels - center) ** 2, axis=1)

        # Assign on first pass using nearest center
        pass  # handled below

    # Efficient batch assignment
    dists = np.linalg.norm(pixels[:, None, :] - palette_lab[None, :, :], axis=2)  # (P, K)
    nearest_clusters = np.argmin(dists, axis=1)

    for i, cluster_idx in enumerate(nearest_clusters):
        color = cluster_to_color[cluster_idx]
        r, g, b = color.rgb
        mapped_pixels[i] = [b, g, r]  # BGR

    mapped_bgr = mapped_pixels.reshape(h, w, 3)

    # ── Color stats ───────────────────────────────────────────────────────────
    total_pixels = h * w
    distribution: Dict[str, float] = {}
    for cluster_idx in range(len(palette_lab)):
        mask = nearest_clusters == cluster_idx
        count = int(mask.sum())
        if count > 0:
            code = cluster_to_color[cluster_idx].code
            distribution[code] = distribution.get(code, 0.0) + count / total_pixels * 100

    dominant = sorted(distribution, key=distribution.get, reverse=True)[:5]

    ctx.color_stats = ColorStats(
        total_colors=len(used_colors),
        color_distribution=distribution,
        dominant_colors=dominant,
    )
    ctx.palette_colors = used_colors
    ctx.color_id_map = {color_id: color
                        for color_id, color in enumerate(used_colors, start=1)}

    log.info("Palette mapper: %d unique thread colors used", len(used_colors))
    return mapped_bgr, used_colors, cluster_to_id


# ── Delta-E 2000 implementation ───────────────────────────────────────────────

import cv2


def _brand_to_lab(brand_colors: list) -> np.ndarray:
    """Convert brand RGB list to OpenCV LAB values (uint8 scale)."""
    rgbs = np.array([[c["rgb"][0], c["rgb"][1], c["rgb"][2]]
                     for c in brand_colors], dtype=np.uint8)
    # BGR for OpenCV
    bgr = rgbs[:, ::-1].reshape(1, -1, 3)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return lab


def _nearest_delta_e2000(lab_pixel: np.ndarray, brand_lab: np.ndarray) -> int:
    """Find index of nearest brand color using Delta-E 2000."""
    # Use vectorized CIE DE2000 approximation
    scores = _delta_e2000_batch(lab_pixel, brand_lab)
    return int(np.argmin(scores))


def _delta_e2000_batch(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    Vectorized Delta-E 2000 between one color lab1 (shape 3,) and
    array lab2 (shape N,3). OpenCV LAB scale: L in [0,255], a,b in [0,255].
    We convert to standard CIE scale first.
    """
    # OpenCV LAB -> CIE LAB
    def to_cie(lab):
        L = lab[..., 0] * (100.0 / 255.0)
        a = lab[..., 1] - 128.0
        b = lab[..., 2] - 128.0
        return L, a, b

    L1, a1, b1 = to_cie(lab1)
    L2, a2, b2 = to_cie(lab2)

    # C (chroma)
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    C_avg = (C1 + C2) / 2.0

    # a' adjustment
    C_avg7 = C_avg ** 7
    G = 0.5 * (1 - np.sqrt(C_avg7 / (C_avg7 + 25 ** 7)))
    a1p = a1 * (1 + G)
    a2p = a2 * (1 + G)

    C1p = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p = np.sqrt(a2p ** 2 + b2 ** 2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    # dH'
    dhp = h2p - h1p
    dhp = np.where(np.abs(dhp) <= 180, dhp,
                   np.where(dhp > 180, dhp - 360, dhp + 360))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)

    # Averages
    Lp_avg = (L1 + L2) / 2.0
    Cp_avg = (C1p + C2p) / 2.0
    Hp_avg = h1p + h2p
    Hp_avg = np.where(
        np.abs(h1p - h2p) <= 180, Hp_avg / 2.0,
        np.where(Hp_avg < 360, (Hp_avg + 360) / 2.0, (Hp_avg - 360) / 2.0)
    )

    T = (1
         - 0.17 * np.cos(np.radians(Hp_avg - 30))
         + 0.24 * np.cos(np.radians(2 * Hp_avg))
         + 0.32 * np.cos(np.radians(3 * Hp_avg + 6))
         - 0.20 * np.cos(np.radians(4 * Hp_avg - 63)))

    SL = 1 + 0.015 * (Lp_avg - 50) ** 2 / np.sqrt(20 + (Lp_avg - 50) ** 2)
    SC = 1 + 0.045 * Cp_avg
    SH = 1 + 0.015 * Cp_avg * T

    Cp_avg7 = Cp_avg ** 7
    RC = 2 * np.sqrt(Cp_avg7 / (Cp_avg7 + 25 ** 7))
    d_theta = 30 * np.exp(-((Hp_avg - 275) / 25) ** 2)
    RT = -np.sin(np.radians(2 * d_theta)) * RC

    kL = kC = kH = 1.0
    de2000 = np.sqrt(
        (dLp / (kL * SL)) ** 2
        + (dCp / (kC * SC)) ** 2
        + (dHp / (kH * SH)) ** 2
        + RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )

    return de2000
