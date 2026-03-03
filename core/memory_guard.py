"""STEP 1 — Validation + Memory Guard."""

from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np

from core.exceptions import ImageCorruptError, ImageTooLargeError, MemoryGuardError
from core.context import ProcessingContext
from utils.logger import get_logger

log = get_logger(__name__)


MAX_MEGAPIXELS = 50        # auto-downscale above this
RAM_WARN_MB = 500          # warn if free RAM below this
RAM_ABORT_MB = 150         # abort if free RAM below this
MAX_FILE_MB = 200          # reject files larger than this


def _get_free_ram_mb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        return 4096.0  # assume plenty if psutil not available


def _get_total_ram_mb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 * 1024)
    except ImportError:
        return 8192.0


def validate_and_guard(ctx: ProcessingContext, filepath: str) -> np.ndarray:
    """
    STEP 1: Validate file, check RAM, load image.
    Returns the loaded BGR numpy array.
    Raises on critical failures.
    """
    ctx.report_progress(1, "Validating image and checking memory...")


    if not os.path.exists(filepath):
        raise ImageCorruptError(f"File not found: {filepath}")

    file_mb = os.path.getsize(filepath) / (1024 * 1024)
    if file_mb > MAX_FILE_MB:
        raise ImageCorruptError(f"File too large: {file_mb:.1f} MB (max {MAX_FILE_MB} MB)")


    free_mb = _get_free_ram_mb()
    total_mb = _get_total_ram_mb()
    log.info("RAM: %.0f MB free / %.0f MB total", free_mb, total_mb)

    if free_mb < RAM_ABORT_MB:
        raise MemoryGuardError(f"Only {free_mb:.0f} MB RAM free — cannot process safely.")
    if free_mb < RAM_WARN_MB:
        ctx.add_warning(f"Low RAM: {free_mb:.0f} MB free. Processing may be slow.")

    from utils.version import APP_VERSION
    import platform, sys
    ctx.system_info = type(ctx.system_info).__call__() if ctx.system_info else None
    from core.context import SystemInfo
    ctx.system_info = SystemInfo(
        ram_total_mb=total_mb,
        ram_available_mb=free_mb,
        os_version=platform.version(),
        dpi_scale=_get_dpi_scale(),
        python_version=sys.version.split()[0],
    )
    ctx.metadata.source_filename = os.path.basename(filepath)


    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img is None:

        try:
            from PIL import Image
            pil = Image.open(filepath).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ImageCorruptError(f"Cannot decode image: {e}")

    h, w = img.shape[:2]
    megapixels = (h * w) / 1_000_000
    log.info("Image loaded: %dx%d (%.1f MP)", w, h, megapixels)


    if megapixels > MAX_MEGAPIXELS:
        scale = (MAX_MEGAPIXELS / megapixels) ** 0.5
        new_w = int(w * scale)
        new_h = int(h * scale)
        log.info("Downscaling from %dx%d to %dx%d (%.1f MP -> %.1f MP)",
                 w, h, new_w, new_h, megapixels, MAX_MEGAPIXELS)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        ctx.add_warning(f"Image was downscaled from {megapixels:.0f}MP to ~{MAX_MEGAPIXELS}MP for safe processing.")

    return img


def _get_dpi_scale() -> float:
    try:
        from ctypes import windll
        return windll.shcore.GetScaleFactorForDevice(0) / 100.0
    except Exception:
        return 1.0
