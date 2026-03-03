"""STEP 10 — Thread usage calculator: meters and skeins per color."""

from __future__ import annotations

from typing import Dict, List
import numpy as np

from core.context import ProcessingContext, ThreadColor
from utils.logger import get_logger

log = get_logger(__name__)


METERS_PER_SKEIN = 8.0


_THREAD_CM_PER_STITCH: Dict[int, float] = {
    14: 4.5,   # ~4.5 cm per stitch on 14-count canvas
    16: 4.0,
    18: 3.5,
    20: 3.0,
}
_DEFAULT_THREAD_CM = 4.0


def calculate_thread_usage(
    ctx: ProcessingContext,
    stitch_matrix: List[List[int]],
    palette_colors: List[ThreadColor],
    color_id_map: Dict[int, ThreadColor],
) -> Dict[str, ThreadColor]:
    """
    STEP 10: Calculate meters and skeins needed per color.

    Optimized: Now ignores background (ID 0) to avoid 
    massive overestimation of thread consumption.
    """
    ctx.report_progress(10, "Calculating thread usage (filtering background)...")

    matrix = np.array(stitch_matrix, dtype=np.int32)


    stitch_counts: Dict[int, int] = {}
    unique_ids, counts = np.unique(matrix, return_counts=True)
    
    total_background_pixels = 0

    for cid, cnt in zip(unique_ids, counts):
        cid_int = int(cid)
        

        if cid_int == 0 or cid_int not in color_id_map:
            total_background_pixels += int(cnt)
            continue
            
        stitch_counts[cid_int] = int(cnt)

    log.info(f"Filtered out {total_background_pixels} background pixels.")

    cm_per_stitch = _THREAD_CM_PER_STITCH.get(ctx.canvas_count, _DEFAULT_THREAD_CM)
    safety_margin = 1.20  # 20% buffer

    result: Dict[str, ThreadColor] = {}

    for color_id, color in color_id_map.items():
        n_stitches = stitch_counts.get(color_id, 0)
        if n_stitches == 0:
            continue

        total_cm = n_stitches * cm_per_stitch * safety_margin
        total_meters = total_cm / 100.0
        skeins = total_meters / METERS_PER_SKEIN

        updated = ThreadColor(
            brand=color.brand,
            code=color.code,
            name=color.name,
            rgb=color.rgb,
            meters_needed=round(total_meters, 2),
            skeins_needed=round(skeins, 2),
        )
        result[color.code] = updated

    ctx.thread_usage = result
    total_meters_all = sum(c.meters_needed for c in result.values())
    
    log.info(
        "Thread calc: %d active colors, total=%.1fm (~%.1f skeins)",
        len(result), total_meters_all,
        total_meters_all / METERS_PER_SKEIN,
    )
    
    return result