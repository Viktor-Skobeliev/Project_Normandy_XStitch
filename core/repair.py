"""STEP 2 — Auto-Repair: denoise, sharpen, white balance, gamma, CLAHE, bilateral."""

from __future__ import annotations

import cv2
import numpy as np

from core.context import ProcessingContext
from utils.logger import get_logger

log = get_logger(__name__)


def auto_repair(ctx: ProcessingContext, img: np.ndarray) -> np.ndarray:
    """
    Apply full image repair pipeline:
      1. Bilateral filter  (edge-preserving smooth)
      2. NLM denoise       (deep noise removal)
      3. Gamma correction  (exposure fix)
      4. Auto white balance
      5. CLAHE             (adaptive contrast)
      6. Unsharp mask      (sharpening)
    Returns repaired BGR image.
    """
    ctx.report_progress(2, "Repairing image...")
    log.info("Repair: input %dx%d", img.shape[1], img.shape[0])

    result = img.copy().astype(np.uint8)


    result = cv2.bilateralFilter(result, d=5, sigmaColor=35, sigmaSpace=35)
    log.debug("Repair: bilateral filter applied")


    result = cv2.fastNlMeansDenoisingColored(result, None,
                                              h=6, hColor=6,
                                              templateWindowSize=7,
                                              searchWindowSize=21)
    log.debug("Repair: NLM denoise applied")


    result = _auto_gamma(result)
    log.debug("Repair: gamma correction applied")


    result = _gray_world_wb(result)
    log.debug("Repair: white balance applied")


    result = _clahe_lab(result)
    log.debug("Repair: CLAHE applied")


    result = _unsharp_mask(result, strength=0.6)
    log.debug("Repair: unsharp mask applied")

    log.info("Repair: complete")
    return result




def _auto_gamma(img: np.ndarray) -> np.ndarray:
    """Auto gamma correction based on mean brightness."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()


    if mean_brightness < 5 or mean_brightness > 250:
        return img  # skip extreme cases

    gamma = np.log(128.0 / 255.0) / np.log(mean_brightness / 255.0 + 1e-7)
    gamma = float(np.clip(gamma, 0.5, 2.5))

    if abs(gamma - 1.0) < 0.1:
        return img  # no correction needed

    lut = np.array([min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
                    for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def _gray_world_wb(img: np.ndarray) -> np.ndarray:
    """Gray World white balance assumption."""
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)

    mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
    if mean_b < 1 or mean_g < 1 or mean_r < 1:
        return img

    mean_gray = (mean_b + mean_g + mean_r) / 3.0

    b = np.clip(b * (mean_gray / mean_b), 0, 255)
    g = np.clip(g * (mean_gray / mean_g), 0, 255)
    r = np.clip(r * (mean_gray / mean_r), 0, 255)

    return cv2.merge([b, g, r]).astype(np.uint8)


def _clahe_lab(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE to L channel in LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def _unsharp_mask(img: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """Unsharp mask sharpening."""
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
    sharp = cv2.addWeighted(img, 1.0 + strength, blur, -strength, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)
