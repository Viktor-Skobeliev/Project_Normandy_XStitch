"""Pipeline orchestrator — runs all 13 steps in sequence and handles exports."""

from __future__ import annotations

import time
import os

from core.context import ProcessingContext
from utils.analytics import PipelineAnalytics
from utils.logger import get_logger

log = get_logger(__name__)


def run_pipeline(ctx: ProcessingContext, image_path: str) -> ProcessingContext:
    """
    Execute the full processing pipeline:
    1. Validation, 2. Repair, 3. Segment, 4. Resize, 5. Quantize, 6. Mapping,
    7. Cleanup, 8. Path Opt, 9. Grid Gen, 10. Threads, 11. Tiling, 12. Meta, 13. AI Audit.
    """
    analytics = PipelineAnalytics()
    log.info("Pipeline START — image: %s", image_path)
    t0 = time.perf_counter()


    log.info("Pipeline [1/13] Validation...")
    ctx.report_progress(1, "Validating image and checking memory...")
    with analytics.step("step1_validate") as timer:
        from core.memory_guard import validate_and_guard
        img = validate_and_guard(ctx, image_path)
        ctx.original_image = img
    analytics.record("step1_validate", timer.elapsed)


    log.info("Pipeline [2/13] Auto-repair...")
    with analytics.step("step2_repair") as timer:
        from core.repair import auto_repair
        repaired = auto_repair(ctx, img)
        ctx.repaired_image = repaired
    analytics.record("step2_repair", timer.elapsed)


    log.info("Pipeline [3/13] Auto-segment...")
    with analytics.step("step3_segment") as timer:
        from core.segment import auto_segment
        segmented = auto_segment(ctx, repaired)
        ctx.segmented_image = segmented
    analytics.record("step3_segment", timer.elapsed)


    log.info("Pipeline [4/13] Resize to grid...")
    with analytics.step("step4_resize") as timer:
        from core.quantizer import resize_to_grid
        resized = resize_to_grid(ctx, segmented)
        ctx.resized_image = resized
    analytics.record("step4_resize", timer.elapsed)


    log.info("Pipeline [5/13] Color quantization...")
    with analytics.step("step5_quantize") as timer:
        from core.quantizer import quantize_colors, apply_dithering
        from utils.config import get as cfg_get

        quantized, palette_lab = quantize_colors(ctx, resized)

        if cfg_get("dithering", True):
            ctx.report_progress(5, "Applying dithering...")
            quantized = apply_dithering(quantized, palette_lab)

        ctx.quantized_image = quantized
    analytics.record("step5_quantize", timer.elapsed)


    log.info("Pipeline [6/13] Palette mapping...")
    ctx.report_progress(6, "Mapping colors to thread palette...")
    with analytics.step("step6_palette") as timer:
        from core.palette_mapper import map_to_palette
        mapped_bgr, palette_colors, cluster_to_id = map_to_palette(
            ctx, quantized, palette_lab
        )
    analytics.record("step6_palette", timer.elapsed)


    log.info("Pipeline [7/13] Confetti cleanup...")
    ctx.report_progress(7, "Cleaning confetti pixels...")
    with analytics.step("step7_confetti") as timer:
        from core.confetti_cleaner import clean_confetti
        from utils.config import get as cfg_get

        cleaner_threshold = cfg_get("confetti_threshold", 3)
        import core.confetti_cleaner as cc_module
        original_threshold = cc_module.MIN_ISLAND_SIZE
        cc_module.MIN_ISLAND_SIZE = cleaner_threshold

        cleaned = clean_confetti(ctx, mapped_bgr, palette_colors)
        cc_module.MIN_ISLAND_SIZE = original_threshold
    analytics.record("step7_confetti", timer.elapsed)


    log.info("Pipeline [8/13] Path optimization...")
    ctx.report_progress(8, "Optimizing stitch paths...")
    with analytics.step("step8_paths") as timer:
        from core.path_optimizer import optimize_paths
        

        color_to_id = {}
        for ci, color in enumerate(palette_colors):
            r, g, b = color.rgb
            color_to_id[(b, g, r)] = ci + 1

        stitch_matrix_temp = _bgr_to_matrix(cleaned, color_to_id)
        paths = optimize_paths(ctx, stitch_matrix_temp)
    analytics.record("step8_paths", timer.elapsed)


    log.info("Pipeline [9/13] Grid + symbol generation...")
    ctx.report_progress(9, "Generating stitch grid and symbols...")
    with analytics.step("step9_grid") as timer:
        from core.grid_generator import generate_grid
        stitch_matrix, symbol_map = generate_grid(ctx, cleaned, color_to_id)
        ctx.stitch_matrix = stitch_matrix
        ctx.symbol_map = symbol_map
    analytics.record("step9_grid", timer.elapsed)


    log.info("Pipeline [10/13] Thread usage calculation...")
    ctx.report_progress(10, "Calculating thread usage...")
    with analytics.step("step10_threads") as timer:
        from core.thread_calculator import calculate_thread_usage

        thread_usage = calculate_thread_usage(
            ctx, stitch_matrix, palette_colors, ctx.color_id_map
        )
    analytics.record("step10_threads", timer.elapsed)


    log.info("Pipeline [11/13] Page tiling...")
    with analytics.step("step11_tiling") as timer:
        from core.page_tiler import tile_pages
        ctx.report_progress(11, "Tiling pages...")
        tiles = tile_pages(ctx, stitch_matrix)
        ctx.metadata.settings["tiles_count"] = len(tiles)
    analytics.record("step11_tiling", timer.elapsed)


    log.info("Pipeline [12/13] Preparing export data...")
    ctx.report_progress(12, "Preparing export data...")
    ctx.metadata.settings.update({
        "palette": ctx.palette_selected,
        "canvas_count": ctx.canvas_count,
        "grid_width": ctx.grid_width,
        "grid_height": ctx.grid_height,
        "total_colors": len(palette_colors),
        "pipeline_time_s": round(time.perf_counter() - t0, 2),
    })


    log.info("Pipeline [13/13] AI Audit...")
    with analytics.step("step13_ai") as timer:
        try:
            from ai.enhancer import run_ai_audit_sync
            ai_result = run_ai_audit_sync(ctx)
            if ai_result:
                ctx.ai_suggestions = ai_result
        except Exception as e:
            log.warning("AI Audit failed (non-blocking): %s", e)
    analytics.record("step13_ai", timer.elapsed)

    analytics.log_summary()
    log.info("Pipeline COMPLETE in %.2f s", time.perf_counter() - t0)
    return ctx


def export_pdf_result(ctx: ProcessingContext, file_path: str) -> bool:
    """
    Экспортирует результат обработки в PDF.
    Решает ошибку 'cannot import name export_pdf_result'.
    """
    try:
        log.info("Exporting PDF to %s", file_path)
        from core.pdf_engine import PDFEngine
        engine = PDFEngine(ctx)
        return engine.generate(file_path)
    except ImportError:
        log.error("PDF engine (reportlab) not found.")
        raise
    except Exception as e:
        log.error("Failed to export PDF: %s", e)
        return False


def _bgr_to_matrix(bgr_img, color_to_id: dict) -> list:
    """Convert BGR image to stitch ID matrix using color_to_id lookup."""
    import numpy as np
    h, w = bgr_img.shape[:2]
    pixels = bgr_img.reshape(-1, 3)


    packed_map = {
        b + g * 256 + r * 65536: cid
        for (b, g, r), cid in color_to_id.items()
    }

    packed = (pixels[:, 0].astype(np.int64)
              + pixels[:, 1].astype(np.int64) * 256
              + pixels[:, 2].astype(np.int64) * 65536)


    ids = np.array([packed_map.get(int(p), 1) for p in packed], dtype=np.int32)
    return ids.reshape(h, w).tolist()