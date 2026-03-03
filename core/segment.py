"""STEP 3 — Auto-Segment: direct ONNX u2net inference (CPU only, no rembg/pooch)."""

from __future__ import annotations

import datetime
import os
import sys
import threading
import time
import traceback

import cv2
import numpy as np
from PIL import Image

from core.context import ProcessingContext
from utils.logger import get_logger

log = get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

def _get_model_path() -> str:
    """Absolute path to u2net.onnx — works as script and as frozen exe."""
    if getattr(sys, 'frozen', False):
        # PyInstaller onedir: _MEIPASS == _internal/ directory
        base = sys._MEIPASS
    else:
        # Running as .py script: project root is parent of core/
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, 'models', 'u2net.onnx')


def _get_crash_report_path() -> str:
    """crash_report.txt next to the exe (or project root when running as script)."""
    if getattr(sys, 'frozen', False):
        return os.path.join(os.path.dirname(sys.executable), 'crash_report.txt')
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'crash_report.txt'
    )


def _write_crash(title: str, exc_text: str) -> None:
    """Append a verbose error entry to crash_report.txt."""
    path = _get_crash_report_path()
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {title}\n")
            f.write(f"{'='*60}\n")
            f.write(exc_text)
            f.write('\n')
        log.info("Crash report written to: %s", path)
    except Exception as e:
        log.warning("Could not write crash report: %s", e)


# ── ONNX Session (cached, thread-safe, CPU only) ──────────────────────────────

_session = None
_session_lock = threading.Lock()


def _get_onnx_session():
    """
    Return cached ort.InferenceSession for u2net.onnx.
    Forces CPUExecutionProvider. Raises on missing model or load failure.
    Thread-safe.
    """
    global _session
    with _session_lock:
        if _session is not None:
            return _session

        model_path = _get_model_path()
        log.info("ONNX: model path = %s", model_path)

        # ── Sanity checks ──────────────────────────────────────────────────
        if not os.path.isfile(model_path):
            msg = (f"Model file not found in /models/.\n"
                   f"Expected: {model_path}\n"
                   f"Create the 'models' folder next to the exe and put u2net.onnx there.")
            _write_crash("MODEL NOT FOUND", msg)
            raise FileNotFoundError(msg)

        size_mb = os.path.getsize(model_path) / 1024 / 1024
        log.info("ONNX: model file size = %.1f MB", size_mb)
        if size_mb < 10:
            msg = f"Model file is suspiciously small ({size_mb:.1f} MB) — likely corrupted."
            _write_crash("MODEL CORRUPTED", msg)
            raise RuntimeError(msg)

        # ── Load session ───────────────────────────────────────────────────
        log.info("ONNX: loading InferenceSession (CPU only)...")
        t0 = time.perf_counter()
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, (os.cpu_count() or 4))
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.log_severity_level = 3  # suppress ort verbose output

            _session = ort.InferenceSession(
                model_path,
                sess_options=opts,
                providers=['CPUExecutionProvider'],
            )

            # Log input/output info for debugging
            inp = _session.get_inputs()[0]
            out = _session.get_outputs()[0]
            log.info("ONNX: session ready in %.2fs | input='%s' %s | output='%s' %s",
                     time.perf_counter() - t0, inp.name, inp.shape, out.name, out.shape)

        except Exception as e:
            tb = traceback.format_exc()
            _write_crash("ONNX SESSION LOAD FAILED", tb)
            log.error("ONNX: session load failed:\n%s", tb)
            raise

    return _session


# ── u2net preprocessing / postprocessing ─────────────────────────────────────

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SIZE = (320, 320)


def _preprocess(pil_img: Image.Image) -> np.ndarray:
    """Resize to 320x320, normalize, return NCHW float32 array."""
    resized = pil_img.convert("RGB").resize(_INPUT_SIZE, Image.Resampling.LANCZOS)
    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD            # HWC
    arr = arr.transpose(2, 0, 1)          # CHW
    return arr[np.newaxis]                # NCHW


def _postprocess(ort_out: list, orig_wh: tuple) -> np.ndarray:
    """Convert u2net output to uint8 mask at original resolution."""
    pred = ort_out[0][:, 0, :, :]        # (1, 320, 320)
    pred = np.squeeze(pred)               # (320, 320)
    mn, mx = pred.min(), pred.max()
    pred = (pred - mn) / (mx - mn + 1e-8)
    mask_small = Image.fromarray((pred * 255).astype(np.uint8), mode='L')
    mask = mask_small.resize(orig_wh, Image.Resampling.LANCZOS)
    return np.array(mask)


# ── Core BG removal ───────────────────────────────────────────────────────────

def _remove_background_direct(img_bgr: np.ndarray) -> np.ndarray:
    """
    Run u2net inference directly via onnxruntime (no rembg, no pooch, no network).
    Returns BGRA uint8 array.
    Raises on any failure — caller handles gracefully.
    """
    h, w = img_bgr.shape[:2]
    log.info("BG removal: input %dx%d px", w, h)

    # Convert BGR -> PIL RGB
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    log.info("BG removal: preprocessing...")
    t0 = time.perf_counter()
    inp = _preprocess(pil_img)
    log.info("BG removal: preprocess done in %.3fs, tensor shape %s", time.perf_counter()-t0, inp.shape)

    log.info("BG removal: loading ONNX session...")
    session = _get_onnx_session()
    input_name = session.get_inputs()[0].name

    log.info("BG removal: running inference on CPU...")
    t1 = time.perf_counter()
    ort_out = session.run(None, {input_name: inp})
    elapsed = time.perf_counter() - t1
    log.info("BG removal: inference done in %.2fs, output shape %s", elapsed, ort_out[0].shape)

    log.info("BG removal: postprocessing mask...")
    mask = _postprocess(ort_out, (w, h))
    log.info("BG removal: mask range [%d, %d], unique values: %d",
             mask.min(), mask.max(), len(np.unique(mask)))

    # Composite original BGR + mask -> BGRA
    bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    log.info("BG removal: BGRA composited, non-transparent px = %d", (mask > 10).sum())

    return bgra


# ── Morphological helpers ─────────────────────────────────────────────────────

def _morph_cleanup(bgra: np.ndarray) -> np.ndarray:
    """Erode fringe pixels, dilate to restore shape, blur alpha edge."""
    alpha = bgra[:, :, 3]
    _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, k3, iterations=1)
    mask = cv2.dilate(mask, k5, iterations=1)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    bgra[:, :, 3] = mask
    return bgra


def _crop_to_content(bgra: np.ndarray) -> np.ndarray:
    """Crop transparent margins around the subject."""
    alpha = bgra[:, :, 3]
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)
    if not rows.any() or not cols.any():
        log.warning("BG removal: crop found no content pixels — returning uncropped")
        return bgra
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    log.info("BG removal: crop %dx%d -> %dx%d", bgra.shape[1], bgra.shape[0], c1-c0+1, r1-r0+1)
    return bgra[r0:r1+1, c0:c1+1]


def _center_on_canvas(bgra: np.ndarray, padding_frac: float = 0.05) -> np.ndarray:
    """Add padding around the cropped subject."""
    h, w = bgra.shape[:2]
    pad = int(max(h, w) * padding_frac)
    canvas = np.zeros((h + 2*pad, w + 2*pad, 4), dtype=np.uint8)
    canvas[pad:pad+h, pad:pad+w] = bgra
    return canvas


def _flatten_to_white(bgra: np.ndarray) -> np.ndarray:
    """Composite BGRA onto white background using PIL alpha_composite.
    Strictly uses RGBA background + explicit .convert('RGB') before returning.
    Writes to crash_report.txt on any failure.
    Returns BGR numpy array for the cv2 pipeline.
    """
    try:
        h, w = bgra.shape[:2]
        log.info("_flatten_to_white: PIL composite %dx%d px...", w, h)
        t = time.perf_counter()

        # BGRA numpy -> RGBA PIL
        rgba_arr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
        log.info("_flatten_to_white: BGRA->RGBA conversion OK, dtype=%s", rgba_arr.dtype)

        fg = Image.fromarray(rgba_arr, mode="RGBA")
        log.info("_flatten_to_white: fg PIL size=%s mode=%s", fg.size, fg.mode)

        # Strict white RGBA background — NOT RGB
        bg = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        log.info("_flatten_to_white: bg created size=%s mode=%s", bg.size, bg.mode)

        # PIL alpha composite — handles edge cases correctly
        bg.alpha_composite(fg)
        log.info("_flatten_to_white: alpha_composite done")

        # Explicit RGB conversion before any further processing
        final_img = bg.convert("RGB")
        log.info("_flatten_to_white: converted to RGB size=%s mode=%s", final_img.size, final_img.mode)

        # Back to BGR numpy for the rest of the pipeline
        result = cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2BGR)
        log.info("_flatten_to_white: done in %.3fs — output shape %s dtype %s",
                 time.perf_counter() - t, result.shape, result.dtype)
        return result

    except Exception as e:
        tb = traceback.format_exc()
        _write_crash("_flatten_to_white FAILED", tb)
        log.error("_flatten_to_white: FAILED — see crash_report.txt\n%s", tb)
        raise


# ── Public API ────────────────────────────────────────────────────────────────

def auto_segment(ctx: ProcessingContext, img: np.ndarray) -> np.ndarray:
    """
    STEP 3: Remove background using u2net (direct ONNX, CPU only).

    If remove_background is False — returns img unchanged.
    On any failure — logs to crash_report.txt, adds warning, returns original img.
    """
    log.info("auto_segment: input %dx%d, remove_background=%s",
             img.shape[1], img.shape[0], ctx.remove_background)

    if not ctx.remove_background:
        ctx.report_progress(3, "Background removal skipped (disabled)")
        log.info("auto_segment: skipped by user setting")
        return img

    ctx.report_progress(3, "Loading AI model (CPU)...")
    log.info("auto_segment: starting background removal")

    MAX_INPUT_PX  = 2000   # threshold: if either side > this, resize
    TARGET_PX     = 1500   # target max side after thumbnail

    t_total = time.perf_counter()
    try:
        # ── Step 3a: thumbnail input if too large (before inference) ──────────
        h, w = img.shape[:2]
        log.info("auto_segment [3a input]: %dx%d px", w, h)
        if max(h, w) > MAX_INPUT_PX:
            scale = TARGET_PX / max(h, w)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            log.info("auto_segment [3a thumbnail]: %dx%d > %dpx — resizing to %dx%d",
                     w, h, MAX_INPUT_PX, new_w, new_h)
            ctx.report_progress(3, f"Thumbnail {w}x{h} -> {new_w}x{new_h} (>2000px limit)...")
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            log.info("auto_segment [3a thumbnail]: done, new shape %s", img.shape)
        else:
            log.info("auto_segment [3a thumbnail]: %dx%d within limit, no resize", w, h)

        # ── Step 3b: u2net inference ──────────────────────────────────────────
        ctx.report_progress(3, "Removing background... (15-30s on CPU, please wait)")
        t = time.perf_counter()
        bgra = _remove_background_direct(img)
        log.info("auto_segment [3b inference]: %.2fs — BGRA shape %s", time.perf_counter()-t, bgra.shape)

        # ── Step 3c: morphological cleanup ───────────────────────────────────
        ctx.report_progress(3, "Cleaning up edges...")
        t = time.perf_counter()
        bgra = _morph_cleanup(bgra)
        log.info("auto_segment [3c morph]: %.3fs", time.perf_counter() - t)

        # ── Step 3d: crop + center ────────────────────────────────────────────
        ctx.report_progress(3, "Cropping to subject...")
        t = time.perf_counter()
        bgra = _crop_to_content(bgra)
        bgra = _center_on_canvas(bgra)
        log.info("auto_segment [3d crop+pad]: %.3fs — result %dx%d",
                 time.perf_counter() - t, bgra.shape[1], bgra.shape[0])

        # ── Step 3e: save foreground mask BEFORE compositing ──────────────────
        alpha = bgra[:, :, 3]
        ctx.fg_mask = alpha > 10  # True=foreground stitch, False=background
        fg_px = int(ctx.fg_mask.sum())
        total_px = bgra.shape[0] * bgra.shape[1]
        log.info("auto_segment [3e fg_mask]: %dx%d, fg=%d/%d (%.1f%%)",
                 bgra.shape[1], bgra.shape[0], fg_px, total_px,
                 100.0 * fg_px / max(1, total_px))

        # ── Step 3f: composite on white (PIL RGBA -> RGB) ─────────────────────
        ctx.report_progress(3, "Compositing on white background...")
        t = time.perf_counter()
        result = _flatten_to_white(bgra)
        log.info("auto_segment [3f composite]: %.3fs", time.perf_counter() - t)

        log.info("auto_segment: COMPLETE in %.2fs — output %dx%d — "
                 "COMPOSITING FINISHED, STARTING QUANTIZATION...",
                 time.perf_counter() - t_total, result.shape[1], result.shape[0])
        ctx.report_progress(3, "Background removed — moving to quantization...")
        return result

    except FileNotFoundError as e:
        ctx.add_warning(str(e))
        ctx.report_progress(3, "Model not found — using original image")
        log.error("auto_segment: model not found: %s", e)
        return img

    except Exception as e:
        tb = traceback.format_exc()
        _write_crash("auto_segment FAILED", tb)
        ctx.add_warning(f"Background removal failed: {e}")
        ctx.report_progress(3, f"BG removal failed — see crash_report.txt")
        log.error("auto_segment: unexpected error:\n%s", tb)
        return img


def prewarm_session() -> None:
    """
    Pre-load ONNX session at app startup so first Generate is fast.
    Safe to call from a background thread. Logs result.
    """
    log.info("prewarm_session: starting...")
    try:
        _get_onnx_session()
        log.info("prewarm_session: OK — model loaded into memory")
    except FileNotFoundError as e:
        log.warning("prewarm_session: model not found (%s) — will show error on first use", e)
    except Exception as e:
        tb = traceback.format_exc()
        _write_crash("prewarm_session FAILED", tb)
        log.warning("prewarm_session: failed:\n%s", tb)
