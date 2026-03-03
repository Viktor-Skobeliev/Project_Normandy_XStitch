"""STEP 12 — PDF Export using ReportLab. Color pattern, symbol pattern, legend, thread usage, page map."""

from __future__ import annotations

import io
import json
import math
import os
from typing import Dict, List, Optional

import numpy as np
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from core.context import ProcessingContext, ThreadColor
from core.page_tiler import PageTile, get_page_map_dimensions
from utils.logger import get_logger

log = get_logger(__name__)

# Page constants
PAGE_W, PAGE_H = A4
MARGIN = 15 * mm
CELL_SIZE = 5 * mm           # grid cell size on PDF page
LEGEND_CELL = 6 * mm         # color swatch in legend

# ── Font setup (Arial for Cyrillic support) ───────────────────────────────────
FONT_R = "Helvetica"       # regular
FONT_B = "Helvetica-Bold"  # bold
_fonts_ok = False


def _register_pdf_fonts() -> None:
    global FONT_R, FONT_B, _fonts_ok
    if _fonts_ok:
        return
    win_fonts = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    try:
        pdfmetrics.registerFont(TTFont("Arial",      os.path.join(win_fonts, "arial.ttf")))
        pdfmetrics.registerFont(TTFont("Arial-Bold", os.path.join(win_fonts, "arialbd.ttf")))
        FONT_R = "Arial"
        FONT_B = "Arial-Bold"
        _fonts_ok = True
        log.info("PDF fonts: Arial registered (Cyrillic supported)")
    except Exception as e:
        log.warning("Arial font registration failed (%s) — Cyrillic may not render", e)

_LOCALES_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "locales.json")
_locales_cache: dict | None = None


def _get_locale(lang: str = "en") -> dict:
    global _locales_cache
    if _locales_cache is None:
        with open(_LOCALES_FILE, "r", encoding="utf-8") as f:
            _locales_cache = json.load(f)
    return _locales_cache.get(lang, _locales_cache["en"])


def export_pdf(
    ctx: ProcessingContext,
    stitch_matrix: List[List[int]],
    symbol_map: Dict[int, str],
    color_id_map: Dict[int, ThreadColor],
    thread_usage: Dict[str, ThreadColor],
    tiles: List[PageTile],
    output_path: str,
    lang: str = "en",
) -> None:
    """
    Export complete PDF with:
      1. Color pattern pages (one tile per page)
      2. Symbol pattern pages (same tiles in B&W with symbols)
      3. Legend page
      4. Thread usage page
      5. Page map
    """
    _register_pdf_fonts()
    log.info("PDF export: %d tiles, output=%s", len(tiles), output_path)
    loc = _get_locale(lang)
    pdf_loc = loc.get("pdf", {})

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    c = rl_canvas.Canvas(output_path, pagesize=A4)

    matrix = np.array(stitch_matrix, dtype=np.int32)

    # ── 0a. Summary / cover page ──────────────────────────────────────────────
    _draw_summary_page(c, ctx, matrix, color_id_map, thread_usage, pdf_loc)
    c.showPage()

    # ── 0b. Full pattern overview ─────────────────────────────────────────────
    _draw_overview_page(c, matrix, color_id_map, pdf_loc)
    c.showPage()

    # ── 1. Color pattern pages ───────────────────────────────────────────────
    for tile in tiles:
        _draw_pattern_page(
            c, matrix, tile, color_id_map, symbol_map,
            mode="color", title=pdf_loc.get("title_pattern", "Pattern"),
            page_total=len(tiles),
        )
        c.showPage()

    # ── 2. Symbol (B&W) pattern pages ────────────────────────────────────────
    for tile in tiles:
        _draw_pattern_page(
            c, matrix, tile, color_id_map, symbol_map,
            mode="symbol", title=pdf_loc.get("title_pattern", "Pattern"),
            page_total=len(tiles),
        )
        c.showPage()

    # ── 3. Legend page ────────────────────────────────────────────────────────
    _draw_legend_page(c, color_id_map, symbol_map, thread_usage, pdf_loc)
    c.showPage()

    # ── 4. Thread usage page ──────────────────────────────────────────────────
    _draw_thread_page(c, thread_usage, pdf_loc)
    c.showPage()

    # ── 5. Page map ───────────────────────────────────────────────────────────
    _draw_page_map(c, tiles, pdf_loc)
    c.showPage()

    c.save()
    log.info("PDF saved: %s", output_path)


# ── Summary / cover page ──────────────────────────────────────────────────────

def _draw_summary_page(
    c: rl_canvas.Canvas,
    ctx: ProcessingContext,
    matrix: np.ndarray,
    color_id_map: Dict[int, ThreadColor],
    thread_usage: Dict[str, ThreadColor],
    pdf_loc: dict,
) -> None:
    """Cover page: project photo + stats + thread shopping list (20% buffer)."""
    # ── Header ────────────────────────────────────────────────────────────────
    c.setFont(FONT_B, 20)
    c.setFillColorRGB(0.16, 0.29, 0.54)
    c.drawCentredString(PAGE_W / 2, PAGE_H - MARGIN - 8 * mm,
                        pdf_loc.get("app_name", "XStitch Pattern Generator"))

    c.setFont(FONT_R, 12)
    c.setFillColorRGB(0.35, 0.35, 0.35)
    c.drawCentredString(PAGE_W / 2, PAGE_H - MARGIN - 16 * mm,
                        pdf_loc.get("title_summary",
                                    "Project Summary & Thread Shopping List"))

    c.setStrokeColorRGB(0.16, 0.29, 0.54)
    c.setLineWidth(1.5)
    c.line(MARGIN, PAGE_H - MARGIN - 21 * mm,
           PAGE_W - MARGIN, PAGE_H - MARGIN - 21 * mm)
    c.setLineWidth(0.5)

    # ── Photo thumbnail (left column) ─────────────────────────────────────────
    top_y = PAGE_H - MARGIN - 26 * mm
    photo_max_w = 75 * mm
    photo_max_h = 80 * mm
    photo_drawn_w = 0.0

    if ctx.segmented_image is not None:
        try:
            img = ctx.segmented_image
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            img_rgb = img[:, :, ::-1].astype(np.uint8)
            pil_img = PILImage.fromarray(img_rgb, mode="RGB")
            iw, ih = pil_img.size
            scale_f = min(photo_max_w / iw, photo_max_h / ih)
            draw_w = iw * scale_f
            draw_h = ih * scale_f
            thumb_w = max(1, int(iw * scale_f * 2))
            thumb_h = max(1, int(ih * scale_f * 2))
            pil_img = pil_img.resize((thumb_w, thumb_h), PILImage.LANCZOS)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            buf.seek(0)
            c.drawImage(ImageReader(buf), MARGIN, top_y - draw_h, draw_w, draw_h)
            c.setStrokeColorRGB(0.6, 0.6, 0.6)
            c.rect(MARGIN, top_y - draw_h, draw_w, draw_h, fill=0, stroke=1)
            photo_drawn_w = draw_w
        except Exception as e:
            log.warning("Summary page photo failed: %s", e)

    # ── Project stats (right column) ──────────────────────────────────────────
    stats_x = MARGIN + max(photo_drawn_w, photo_max_w) + 8 * mm
    grid_h, grid_w = matrix.shape
    canvas_count = getattr(ctx, "canvas_count", 14)
    phys_w_cm = grid_w / canvas_count * 2.54
    phys_h_cm = grid_h / canvas_count * 2.54
    total_m = sum(u.meters_needed for u in thread_usage.values())
    total_skeins_f = sum(u.skeins_needed for u in thread_usage.values())
    total_skeins_buy = sum(math.ceil(u.skeins_needed) for u in thread_usage.values())

    stats = [
        (pdf_loc.get("stats_grid_size",     "Grid Size"),
         f"{grid_w} \u00d7 {grid_h}"),
        (pdf_loc.get("stats_canvas",        "Canvas"),
         f"{canvas_count} {pdf_loc.get('count_unit', 'count (Aida)')}"),
        (pdf_loc.get("stats_physical_size", "Physical Size"),
         f"{phys_w_cm:.1f} \u00d7 {phys_h_cm:.1f} cm"),
        (pdf_loc.get("stats_colors",        "Colors"),
         f"{len(color_id_map)}"),
        (pdf_loc.get("stats_thread_total",  "Thread Total"),
         f"{total_m:.1f} m  (~{total_skeins_f:.1f})"),
        (pdf_loc.get("stats_skeins_buy",    "Skeins to Buy"),
         f"{total_skeins_buy}"),
        (pdf_loc.get("stats_buffer",        "Safety Buffer"),
         pdf_loc.get("stats_buffer_value",  "20% already included")),
    ]

    line_h = 9 * mm
    for i, (label, value) in enumerate(stats):
        y = top_y - i * line_h
        c.setFont(FONT_B, 9)
        c.setFillColorRGB(0.16, 0.29, 0.54)
        c.drawString(stats_x, y, label + ":")
        c.setFont(FONT_R, 9)
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.drawString(stats_x + 42 * mm, y, value)

    # ── Thread shopping list table ─────────────────────────────────────────────
    table_top = top_y - photo_max_h - 10 * mm
    c.setFont(FONT_B, 11)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(MARGIN, table_top,
                 pdf_loc.get("shopping_list_title", "Thread Shopping List"))
    c.setFont(FONT_R, 8)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawString(MARGIN + 54 * mm, table_top,
                 pdf_loc.get("shopping_list_note",
                             "(20% safety buffer already included in all quantities)"))

    sorted_usage = sorted(thread_usage.values(), key=lambda u: u.code)
    headers = [
        pdf_loc.get("col_code",       "Code"),
        pdf_loc.get("col_name",       "Color Name"),
        pdf_loc.get("col_brand",      "Brand"),
        pdf_loc.get("col_meters",     "Meters"),
        pdf_loc.get("col_skeins_buy", "Skeins to Buy"),
    ]
    data_rows: list = [headers]
    for u in sorted_usage:
        data_rows.append([
            u.code,
            u.name,
            u.brand or "DMC",
            f"{u.meters_needed:.1f} m",
            f"\u00d7 {math.ceil(u.skeins_needed)}",
        ])
    total_buy = sum(math.ceil(u.skeins_needed) for u in sorted_usage)
    data_rows.append(["", pdf_loc.get("col_total", "TOTAL"), "",
                       f"{total_m:.1f} m", f"\u00d7 {total_buy}"])

    col_widths = [20 * mm, 68 * mm, 22 * mm, 28 * mm, 27 * mm]
    tbl = Table(data_rows, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0),  (-1, 0),  colors.HexColor("#2a4a8a")),
        ("TEXTCOLOR",     (0, 0),  (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0),  (-1, 0),  FONT_B),
        ("FONTNAME",      (0, -1), (-1, -1), FONT_B),
        ("BACKGROUND",    (0, -1), (-1, -1), colors.HexColor("#ddeeff")),
        ("FONTSIZE",      (0, 0),  (-1, -1), 8),
        ("ALIGN",         (0, 0),  (-1, -1), "CENTER"),
        ("ALIGN",         (1, 1),  (1, -1),  "LEFT"),
        ("ROWBACKGROUNDS",(0, 1),  (-1, -2),
         [colors.white, colors.HexColor("#f0f4ff")]),
        ("GRID",          (0, 0),  (-1, -1), 0.5, colors.HexColor("#bbbbbb")),
        ("TOPPADDING",    (0, 0),  (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0),  (-1, -1), 3),
    ]))
    tbl.wrapOn(c, PAGE_W - 2 * MARGIN, PAGE_H)
    tbl.drawOn(c, MARGIN, table_top - 7 * mm - len(data_rows) * 6.5 * mm)


# ── Full overview page ─────────────────────────────────────────────────────────

def _draw_overview_page(
    c: rl_canvas.Canvas,
    matrix: np.ndarray,
    color_id_map: Dict[int, ThreadColor],
    pdf_loc: dict,
) -> None:
    """Full pattern rendered as a single scaled overview image."""
    title = pdf_loc.get("title_overview", "Full Pattern Overview")
    c.setFont(FONT_B, 14)
    c.setFillColorRGB(0.16, 0.29, 0.54)
    c.drawString(MARGIN, PAGE_H - MARGIN, title)
    c.setFillColorRGB(0, 0, 0)

    h, w = matrix.shape
    img_arr = np.full((h, w, 3), 255, dtype=np.uint8)
    for color_id, thread in color_id_map.items():
        mask = matrix == color_id
        img_arr[mask] = thread.rgb

    # Upscale 3× so each stitch is visible at PDF zoom
    UPSCALE = 3
    img_arr = np.kron(img_arr, np.ones((UPSCALE, UPSCALE, 1), dtype=np.uint8))

    pil_img = PILImage.fromarray(img_arr, mode="RGB")

    avail_w = PAGE_W - 2 * MARGIN
    avail_h = PAGE_H - 2 * MARGIN - 22 * mm
    iw, ih = pil_img.size
    fit_scale = min(avail_w / iw, avail_h / ih)
    draw_w = iw * fit_scale
    draw_h = ih * fit_scale

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    x = MARGIN + (avail_w - draw_w) / 2
    y = PAGE_H - MARGIN - 20 * mm - draw_h
    c.drawImage(ImageReader(buf), x, y, draw_w, draw_h)

    c.setStrokeColorRGB(0.4, 0.4, 0.4)
    c.setLineWidth(0.5)
    c.rect(x, y, draw_w, draw_h, fill=0, stroke=1)

    c.setFont(FONT_R, 8)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawCentredString(PAGE_W / 2, y - 5 * mm,
                        f"{w} \u00d7 {h} stitches  |  {len(color_id_map)} colors")


# ── Pattern page ──────────────────────────────────────────────────────────────

def _draw_pattern_page(
    c: rl_canvas.Canvas,
    matrix: np.ndarray,
    tile: PageTile,
    color_id_map: Dict[int, ThreadColor],
    symbol_map: Dict[int, str],
    mode: str,  # "color" or "symbol"
    title: str,
    page_total: int,
) -> None:
    c.setFont(FONT_B, 9)
    c.drawString(MARGIN, PAGE_H - MARGIN + 3 * mm,
                 f"{title} — Page {tile.page_num}/{page_total}  "
                 f"[col {tile.col_start+1}-{tile.col_end}, row {tile.row_start+1}-{tile.row_end}]")

    rows = tile.row_end - tile.row_start
    cols = tile.col_end - tile.col_start

    # Auto-fit cell size
    available_w = PAGE_W - 2 * MARGIN
    available_h = PAGE_H - 2 * MARGIN - 8 * mm
    cell_w = min(CELL_SIZE, available_w / cols)
    cell_h = min(CELL_SIZE, available_h / rows)
    cell = min(cell_w, cell_h)

    origin_x = MARGIN
    origin_y = PAGE_H - MARGIN - 8 * mm - rows * cell

    for r_idx in range(rows):
        for c_idx in range(cols):
            row = tile.row_start + r_idx
            col = tile.col_start + c_idx
            color_id = int(matrix[row, col])

            x = origin_x + c_idx * cell
            y = origin_y + (rows - 1 - r_idx) * cell  # PDF y goes up

            # color_id=0 → background, leave as empty canvas
            if color_id == 0:
                c.setFillColorRGB(1, 1, 1)
                c.rect(x, y, cell, cell, fill=1, stroke=0)
                c.setStrokeColorRGB(0.88, 0.88, 0.88)
                c.setLineWidth(0.1)
                c.rect(x, y, cell, cell, fill=0, stroke=1)
                continue

            if mode == "color":
                thread = color_id_map.get(color_id)
                if thread:
                    r, g, b = thread.rgb
                    c.setFillColorRGB(r / 255, g / 255, b / 255)
                    c.rect(x, y, cell, cell, fill=1, stroke=0)

                # Grid line
                c.setStrokeColorRGB(0.7, 0.7, 0.7)
                c.setLineWidth(0.2)
                c.rect(x, y, cell, cell, fill=0, stroke=1)

                # Symbol overlay at small size
                sym = symbol_map.get(color_id, "")
                if sym:
                    font_size = max(3, cell * 0.55)
                    c.setFillColorRGB(0, 0, 0)
                    c.setFont(FONT_R, font_size)
                    c.drawCentredString(x + cell / 2, y + cell * 0.2, sym)

            else:  # symbol mode — B&W
                c.setFillColorRGB(1, 1, 1)
                c.rect(x, y, cell, cell, fill=1, stroke=0)

                c.setStrokeColorRGB(0.5, 0.5, 0.5)
                c.setLineWidth(0.2)
                c.rect(x, y, cell, cell, fill=0, stroke=1)

                sym = symbol_map.get(color_id, "")
                if sym:
                    font_size = max(3, cell * 0.6)
                    c.setFillColorRGB(0, 0, 0)
                    c.setFont(FONT_B, font_size)
                    c.drawCentredString(x + cell / 2, y + cell * 0.2, sym)

    # 10-stitch ruler marks
    c.setFont(FONT_R, 5)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    for i in range(0, cols, 10):
        xr = origin_x + i * cell
        c.drawCentredString(xr, origin_y - 4, str(tile.col_start + i + 1))
    for i in range(0, rows, 10):
        yr = origin_y + (rows - 1 - i) * cell + cell / 2
        c.drawRightString(origin_x - 1, yr, str(tile.row_start + i + 1))


# ── Legend page ───────────────────────────────────────────────────────────────

def _draw_legend_page(
    c: rl_canvas.Canvas,
    color_id_map: Dict[int, ThreadColor],
    symbol_map: Dict[int, str],
    thread_usage: Dict[str, ThreadColor],
    pdf_loc: dict,
) -> None:
    title = pdf_loc.get("title_legend", "Color Legend")
    c.setFont(FONT_B, 14)
    c.drawString(MARGIN, PAGE_H - MARGIN, title)

    headers = [
        pdf_loc.get("col_symbol", "Symbol"),
        pdf_loc.get("col_code", "Code"),
        pdf_loc.get("col_name", "Color Name"),
        pdf_loc.get("col_stitches", "Stitches"),
        pdf_loc.get("col_meters", "Meters"),
        pdf_loc.get("col_skeins", "Skeins"),
    ]

    data = [headers]
    for color_id, thread in sorted(color_id_map.items()):
        sym = symbol_map.get(color_id, "?")
        usage = thread_usage.get(thread.code)
        data.append([
            sym,
            thread.code,
            thread.name,
            "-",
            f"{usage.meters_needed:.1f}" if usage else "-",
            f"{usage.skeins_needed:.1f}" if usage else "-",
        ])

    col_widths = [15*mm, 20*mm, 65*mm, 25*mm, 22*mm, 20*mm]
    t = Table(data, colWidths=col_widths)
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a4a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), FONT_B),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (2, 1), (2, -1), "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ])
    t.setStyle(style)

    t.wrapOn(c, PAGE_W - 2 * MARGIN, PAGE_H)
    t.drawOn(c, MARGIN, PAGE_H - MARGIN - 10*mm - len(data) * 7*mm)


# ── Thread usage page ─────────────────────────────────────────────────────────

def _draw_thread_page(
    c: rl_canvas.Canvas,
    thread_usage: Dict[str, ThreadColor],
    pdf_loc: dict,
) -> None:
    title = pdf_loc.get("title_threads", "Thread Usage")
    c.setFont(FONT_B, 14)
    c.drawString(MARGIN, PAGE_H - MARGIN, title)

    sorted_usage = sorted(thread_usage.values(), key=lambda x: -x.meters_needed)
    total_m = sum(u.meters_needed for u in sorted_usage)
    total_skeins = sum(u.skeins_needed for u in sorted_usage)

    headers = [
        pdf_loc.get("col_code", "Code"),
        pdf_loc.get("col_name", "Name"),
        "Color",
        pdf_loc.get("col_meters", "Meters"),
        pdf_loc.get("col_skeins", "Skeins"),
    ]
    data = [headers]
    for u in sorted_usage:
        data.append([u.code, u.name, "", f"{u.meters_needed:.1f}", f"{u.skeins_needed:.1f}"])
    data.append(["", "TOTAL", "", f"{total_m:.1f}", f"{total_skeins:.1f}"])

    col_widths = [22*mm, 70*mm, 20*mm, 28*mm, 28*mm]
    t = Table(data, colWidths=col_widths)
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a4a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), FONT_B),
        ("FONTNAME", (0, -1), (-1, -1), FONT_B),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e8e8e8")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (1, 1), (1, -1), "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#f5f5f5")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ])
    t.setStyle(style)

    # Draw color swatches in "Color" column
    t.wrapOn(c, PAGE_W - 2 * MARGIN, PAGE_H)
    table_y = PAGE_H - MARGIN - 10*mm - len(data) * 6.5*mm
    t.drawOn(c, MARGIN, table_y)

    # Draw color swatches
    row_h = 6.5 * mm
    swatch_x = MARGIN + 22*mm + 70*mm + 2*mm
    swatch_y = table_y + (len(data) - 1) * row_h - row_h * 0.7

    for i, u in enumerate(sorted_usage):
        r, g, b = u.rgb
        c.setFillColorRGB(r / 255, g / 255, b / 255)
        c.rect(swatch_x, swatch_y - i * row_h, 16*mm, 4*mm, fill=1, stroke=1)


# ── Page map ──────────────────────────────────────────────────────────────────

def _draw_page_map(
    c: rl_canvas.Canvas,
    tiles: List[PageTile],
    pdf_loc: dict,
) -> None:
    title = pdf_loc.get("title_map", "Page Map")
    c.setFont(FONT_B, 14)
    c.drawString(MARGIN, PAGE_H - MARGIN, title)

    if not tiles:
        return

    n_cols, n_rows = get_page_map_dimensions(tiles)
    cell_w = min(30*mm, (PAGE_W - 2 * MARGIN) / n_cols)
    cell_h = min(20*mm, (PAGE_H - 2 * MARGIN - 15*mm) / n_rows)

    # Index tiles by (grid_row, grid_col)
    tile_map = {(t.grid_row, t.grid_col): t for t in tiles}

    origin_x = MARGIN
    origin_y = PAGE_H - MARGIN - 15*mm

    for gr in range(n_rows):
        for gc in range(n_cols):
            tile = tile_map.get((gr, gc))
            x = origin_x + gc * cell_w
            y = origin_y - gr * cell_h - cell_h

            c.setStrokeColorRGB(0.3, 0.3, 0.3)
            c.setFillColorRGB(0.9, 0.95, 1.0)
            c.rect(x, y, cell_w, cell_h, fill=1, stroke=1)

            if tile:
                c.setFillColorRGB(0, 0, 0)
                c.setFont(FONT_B, 9)
                c.drawCentredString(x + cell_w / 2, y + cell_h * 0.6,
                                    f"Page {tile.page_num}")
                c.setFont(FONT_R, 6)
                c.drawCentredString(x + cell_w / 2, y + cell_h * 0.25,
                                    f"col {tile.col_start+1}-{tile.col_end}")
                c.drawCentredString(x + cell_w / 2, y + cell_h * 0.1,
                                    f"row {tile.row_start+1}-{tile.row_end}")
