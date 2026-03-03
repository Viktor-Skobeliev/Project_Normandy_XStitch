"""PDF Generation Engine — Full pattern export with summary, legend, grid, tiles."""

from __future__ import annotations

import io
import math
import os
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from utils.logger import get_logger

log = get_logger(__name__)

W, H = A4
MARGIN = 20 * mm
INNER_W = W - 2 * MARGIN
CELL_PT = 6
TILE_CELL_PT = 8
COL_HEADER_BG = colors.HexColor("#1F6AA5")
COL_HEADER_TXT = colors.white
COL_ROW_ALT = colors.HexColor("#f0f4f8")
COL_GRID_LINE = colors.HexColor("#cccccc")
COL_VERDICT_PASS = colors.HexColor("#4CAF50")
COL_VERDICT_WARN = colors.HexColor("#FFC107")
COL_VERDICT_FAIL = colors.HexColor("#f44336")


COL_GRID_MAJOR = colors.HexColor("#333333")  # жирная линия каждые 10
COL_GRID_MINOR = colors.HexColor("#bbbbbb")  # тонкая линия каждую клетку


class PDFEngine:
    def __init__(self, ctx):
        self.ctx = ctx
        self.width = W
        self.height = H

    def generate(self, file_path: str) -> bool:
        try:
            c = rl_canvas.Canvas(file_path, pagesize=A4)
            c.setTitle("Cross-Stitch Pattern Report")
            c.setAuthor("XStitch Pattern Generator v2.0")
            self._page_summary(c)
            c.showPage()
            self._page_full_pattern(c)
            c.showPage()
            self._pages_tiles(c)
            c.save()
            log.info("PDF saved: %s", file_path)
            return True
        except Exception as e:
            log.error("PDF Engine error: %s", e, exc_info=True)
            return False

    def _page_summary(self, c):
        ctx = self.ctx
        y = H - MARGIN

        c.setFont("Helvetica-Bold", 18)
        c.setFillColor(COL_HEADER_BG)
        c.drawString(MARGIN, y, "Cross-Stitch Pattern — Project Summary")
        y -= 8
        c.setStrokeColor(COL_HEADER_BG)
        c.setLineWidth(2)
        c.line(MARGIN, y, W - MARGIN, y)
        y -= 16

        img_w = 70 * mm
        img_h = 65 * mm
        stats_x = MARGIN + img_w + 8 * mm

        preview = self._get_preview_image()
        if preview:
            c.drawImage(preview, MARGIN, y - img_h, width=img_w, height=img_h,
                        preserveAspectRatio=True)

        settings = ctx.metadata.settings
        canvas_count = settings.get("canvas_count", ctx.canvas_count)
        grid_w = settings.get("grid_width", ctx.grid_width)
        grid_h = settings.get("grid_height", ctx.grid_height)
        palette = settings.get("palette", ctx.palette_selected)
        total_colors = len(ctx.thread_usage)
        cm_per_stitch = 2.54 / canvas_count
        phys_w = grid_w * cm_per_stitch
        phys_h = grid_h * cm_per_stitch
        total_meters = sum(t.meters_needed for t in ctx.thread_usage.values())
        total_skeins = math.ceil(sum(t.skeins_needed for t in ctx.thread_usage.values()))

        stats = [
            ("Source file",   ctx.metadata.source_filename or "Unknown"),
            ("Thread brand",  palette),
            ("Canvas",        f"{canvas_count} ct (Aida)"),
            ("Grid size",     f"{grid_w} × {grid_h} stitches"),
            ("Physical size", f"{phys_w:.1f} × {phys_h:.1f} cm"),
            ("Colors used",   str(total_colors)),
            ("Thread total",  f"{total_meters:.1f} m"),
            ("Skeins to buy", f"≈ {total_skeins} skeins"),
            ("Safety buffer", "20% included"),
        ]

        sy = y - 4
        for label, value in stats:
            c.setFont("Helvetica-Bold", 9)
            c.setFillColor(colors.HexColor("#444444"))
            c.drawString(stats_x, sy, f"{label}:")
            c.setFont("Helvetica", 9)
            c.setFillColor(colors.black)
            c.drawString(stats_x + 36 * mm, sy, value)
            sy -= 14

        ai = ctx.ai_suggestions or {}
        verdict = ai.get("verdict", "N/A")
        verdict_color = {
            "PASS": COL_VERDICT_PASS,
            "WARNING": COL_VERDICT_WARN,
            "FAIL": COL_VERDICT_FAIL,
        }.get(verdict, colors.gray)

        sy -= 4
        c.setFillColor(verdict_color)
        c.roundRect(stats_x, sy - 14, 50 * mm, 18, 4, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(stats_x + 25 * mm, sy - 9, f"AI Verdict: {verdict}")

        y -= img_h + 10

        pdf_summary = ai.get("pdf_summary", "")
        if pdf_summary:
            y -= 6
            c.setFillColor(colors.HexColor("#f8f9fa"))
            c.setStrokeColor(colors.HexColor("#dee2e6"))
            c.setLineWidth(0.5)
            summary_h = 62
            c.roundRect(MARGIN, y - summary_h, INNER_W, summary_h, 4, fill=1, stroke=1)
            c.setFillColor(colors.HexColor("#333333"))
            c.setFont("Helvetica-Oblique", 8.5)
            self._draw_wrapped_text(c, pdf_summary, MARGIN + 4 * mm, y - 10,
                                    INNER_W - 8 * mm, line_height=11, max_lines=5)
            y -= summary_h + 10

        y -= 4
        c.setStrokeColor(COL_HEADER_BG)
        c.setLineWidth(1)
        c.line(MARGIN, y, W - MARGIN, y)
        y -= 14

        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(COL_HEADER_BG)
        c.drawString(MARGIN, y, "Thread Shopping List  (20% safety buffer included)")
        y -= 14

        code_to_id = {}
        if ctx.color_id_map:
            for cid, color in ctx.color_id_map.items():
                code_to_id[color.code] = cid

        cols_x = [MARGIN, MARGIN + 12*mm, MARGIN + 22*mm, MARGIN + 60*mm,
                  MARGIN + 100*mm, MARGIN + 120*mm, MARGIN + 140*mm]
        col_labels = ["", "Sym", "Code & Name", "", "Meters", "Skeins", "Buy"]
        row_h = 14

        c.setFillColor(COL_HEADER_BG)
        c.rect(MARGIN, y - row_h + 3, INNER_W, row_h, fill=1, stroke=0)
        c.setFillColor(COL_HEADER_TXT)
        c.setFont("Helvetica-Bold", 8)
        for xi, label in zip(cols_x, col_labels):
            c.drawString(xi + 2, y - 8, label)
        y -= row_h

        total_m = 0.0
        total_sk = 0
        for i, (code, thread) in enumerate(ctx.thread_usage.items()):
            if y < MARGIN + 20:
                c.showPage()
                y = H - MARGIN
            if i % 2 == 0:
                c.setFillColor(COL_ROW_ALT)
                c.rect(MARGIN, y - row_h + 3, INNER_W, row_h, fill=1, stroke=0)

            r, g, b = thread.rgb
            c.setFillColorRGB(r / 255, g / 255, b / 255)
            c.rect(cols_x[0] + 1, y - 9, 8, 8, fill=1, stroke=0)
            c.setStrokeColor(colors.gray)
            c.setLineWidth(0.3)
            c.rect(cols_x[0] + 1, y - 9, 8, 8, fill=0, stroke=1)

            c_id = code_to_id.get(code)
            sym = (ctx.symbol_map.get(c_id, "?") if (ctx.symbol_map and c_id) else "?")
            c.setFillColor(colors.black)
            c.setFont("Helvetica-Bold", 9)
            c.drawString(cols_x[1] + 2, y - 8, sym)

            c.setFont("Helvetica", 8)
            name_short = thread.name[:28] if thread.name else ""
            c.drawString(cols_x[2] + 2, y - 8, f"{thread.brand} {code}")
            c.drawString(cols_x[2] + 2 + 22*mm, y - 8, name_short)

            skeins_buy = math.ceil(thread.skeins_needed)
            c.drawRightString(cols_x[4] + 16*mm, y - 8, f"{thread.meters_needed:.1f} m")
            c.drawRightString(cols_x[5] + 16*mm, y - 8, f"{thread.skeins_needed:.2f}")
            c.setFont("Helvetica-Bold", 8)
            c.drawRightString(cols_x[6] + 16*mm, y - 8, str(skeins_buy))

            total_m += thread.meters_needed
            total_sk += skeins_buy
            y -= row_h

        c.setFillColor(colors.HexColor("#e8edf2"))
        c.rect(MARGIN, y - row_h + 3, INNER_W, row_h, fill=1, stroke=0)
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(MARGIN + 4, y - 8, "TOTAL")
        c.drawRightString(cols_x[4] + 16*mm, y - 8, f"{total_m:.1f} m")
        c.drawRightString(cols_x[6] + 16*mm, y - 8, f"{total_sk} skeins")

    def _page_full_pattern(self, c):
        ctx = self.ctx
        if not ctx.stitch_matrix:
            c.setFont("Helvetica", 12)
            c.drawString(MARGIN, H / 2, "No stitch matrix available.")
            return

        matrix = ctx.stitch_matrix
        rows = len(matrix)
        cols = len(matrix[0]) if matrix else 0

        y = H - MARGIN
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(COL_HEADER_BG)
        c.drawString(MARGIN, y, f"Full Pattern Overview — {cols} × {rows} stitches")
        y -= 6
        c.setStrokeColor(COL_HEADER_BG)
        c.setLineWidth(1.5)
        c.line(MARGIN, y, W - MARGIN, y)
        y -= 18

        ruler_gap = 14
        avail_w = INNER_W - ruler_gap * 2
        avail_h = y - MARGIN - ruler_gap * 2
        cell = min(avail_w / cols, avail_h / rows, CELL_PT)
        cell = max(cell, 1.5)

        draw_w = cols * cell
        draw_h = rows * cell
        ox = MARGIN + ruler_gap + (avail_w - draw_w) / 2
        oy = MARGIN + ruler_gap

        img_data = self._render_matrix_to_pil(matrix, ctx, cell_px=max(2, int(cell * 2)))
        if img_data:
            c.drawImage(img_data, ox, oy, width=draw_w, height=draw_h)


        c.setStrokeColor(COL_GRID_MINOR)
        c.setLineWidth(0.3)
        for i in range(cols + 1):
            x = ox + i * cell
            c.line(x, oy, x, oy + draw_h)
        for i in range(rows + 1):
            yr = oy + i * cell
            c.line(ox, yr, ox + draw_w, yr)


        c.setStrokeColor(COL_GRID_MAJOR)
        c.setLineWidth(1.2)
        for i in range(0, cols + 1, 10):
            x = ox + i * cell
            c.line(x, oy, x, oy + draw_h)
        for i in range(0, rows + 1, 10):
            yr = oy + i * cell
            c.line(ox, yr, ox + draw_w, yr)


        c.setStrokeColor(colors.HexColor("#333333"))
        c.setLineWidth(1.5)
        c.rect(ox, oy, draw_w, draw_h, fill=0, stroke=1)


        c.setFont("Helvetica-Bold", 6)
        c.setFillColor(colors.HexColor("#333333"))
        for i in range(0, cols + 1, 10):
            x = ox + i * cell
            c.drawCentredString(x, oy + draw_h + 5, str(i))  # сверху
            c.drawCentredString(x, oy - 9, str(i))            # снизу
        for i in range(0, rows + 1, 10):
            yr = oy + (rows - i) * cell
            c.drawRightString(ox - 3, yr - 2, str(i))         # слева
            c.drawString(ox + draw_w + 3, yr - 2, str(i))     # справа

    def _pages_tiles(self, c):
        ctx = self.ctx
        if not ctx.stitch_matrix:
            return

        from core.page_tiler import tile_pages
        tiles = tile_pages(ctx, ctx.stitch_matrix)
        matrix = ctx.stitch_matrix
        total_tiles = len(tiles)
        log.info("PDF: rendering %d tile pages", total_tiles)

        rows = len(matrix)
        cols = len(matrix[0]) if matrix else 0
        covered = np.zeros((rows, cols), dtype=bool)
        for tile in tiles:
            covered[tile.row_start:tile.row_end, tile.col_start:tile.col_end] = True
        uncovered = int((~covered).sum())
        if uncovered > 0:
            log.warning("PDF tiles INTEGRITY CHECK FAILED: %d stitches not covered!", uncovered)
        else:
            log.info("PDF tiles integrity check: OK — all %d stitches covered", rows * cols)

        for tile in tiles:
            self._draw_tile_page(c, tile, matrix, ctx, total_tiles, uncovered)
            c.showPage()

    def _draw_tile_page(self, c, tile, matrix, ctx, total_tiles, uncovered_count):
        y = H - MARGIN

        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(COL_HEADER_BG)
        page_label = (f"Page {tile.page_num}/{total_tiles}  —  "
                      f"Rows {tile.row_start}–{tile.row_end-1}, "
                      f"Cols {tile.col_start}–{tile.col_end-1}  "
                      f"({tile.width}×{tile.height} stitches)")
        c.drawString(MARGIN, y, page_label)
        y -= 5
        c.setStrokeColor(COL_HEADER_BG)
        c.setLineWidth(1)
        c.line(MARGIN, y, W - MARGIN, y)
        y -= 12

        if uncovered_count > 0:
            c.setFillColor(colors.red)
            c.setFont("Helvetica-Bold", 8)
            c.drawString(MARGIN, y, f"⚠ INTEGRITY WARNING: {uncovered_count} stitches not covered by tiles!")
            y -= 12

        ruler_gap = 16
        avail_w = INNER_W - ruler_gap * 2
        avail_h = y - MARGIN - ruler_gap * 2 - 28
        cell = min(avail_w / tile.width, avail_h / tile.height, TILE_CELL_PT)
        cell = max(cell, 2.0)

        draw_w = tile.width * cell
        draw_h = tile.height * cell
        ox = MARGIN + ruler_gap
        oy = MARGIN + ruler_gap + 28

        tile_matrix = [row[tile.col_start:tile.col_end]
                       for row in matrix[tile.row_start:tile.row_end]]
        img_data = self._render_matrix_to_pil(tile_matrix, ctx,
                                              cell_px=max(4, int(cell * 1.5)),
                                              draw_symbols=True)
        if img_data:
            c.drawImage(img_data, ox, oy, width=draw_w, height=draw_h)


        c.setStrokeColor(COL_GRID_MINOR)
        c.setLineWidth(0.3)
        for i in range(tile.width + 1):
            x = ox + i * cell
            c.line(x, oy, x, oy + draw_h)
        for i in range(tile.height + 1):
            yr = oy + i * cell
            c.line(ox, yr, ox + draw_w, yr)



        c.setStrokeColor(COL_GRID_MAJOR)
        c.setLineWidth(1.5)
        for i in range(tile.width + 1):
            global_col = tile.col_start + i
            if global_col % 10 == 0:
                x = ox + i * cell
                c.line(x, oy, x, oy + draw_h)
        for i in range(tile.height + 1):
            global_row = tile.row_start + i
            if global_row % 10 == 0:
                yr = oy + i * cell
                c.line(ox, yr, ox + draw_w, yr)


        c.setStrokeColor(colors.HexColor("#333333"))
        c.setLineWidth(1.5)
        c.rect(ox, oy, draw_w, draw_h, fill=0, stroke=1)


        c.setFont("Helvetica-Bold", 6)
        c.setFillColor(colors.HexColor("#333333"))
        for i in range(0, tile.width + 1, 10):
            x = ox + i * cell
            label = str(tile.col_start + i)
            c.drawCentredString(x, oy - 10, label)          # снизу
            c.drawCentredString(x, oy + draw_h + 4, label)  # сверху
        for i in range(0, tile.height + 1, 10):
            yr = oy + (tile.height - i) * cell
            label = str(tile.row_start + i)
            c.drawRightString(ox - 3, yr - 3, label)         # слева
            c.drawString(ox + draw_w + 3, yr - 3, label)     # справа


        used_ids = set()
        for row in tile_matrix:
            for cid in row:
                if cid != 0:
                    used_ids.add(cid)

        if used_ids and ctx.color_id_map and ctx.symbol_map:
            lx = MARGIN
            ly = MARGIN + 20
            c.setFont("Helvetica-Bold", 7)
            c.setFillColor(colors.black)
            c.drawString(lx, ly, "Colors on this page:")
            ly -= 9
            c.setFont("Helvetica", 7)
            items_per_row = max(int(INNER_W / (28 * mm)), 4)
            col_i = 0
            for cid in sorted(used_ids):
                color = ctx.color_id_map.get(cid)
                sym = ctx.symbol_map.get(cid, "?")
                if not color:
                    continue
                lxi = lx + col_i * 28 * mm
                r, g, b = color.rgb
                c.setFillColorRGB(r/255, g/255, b/255)
                c.rect(lxi, ly - 5, 7, 7, fill=1, stroke=0)
                c.setFillColor(colors.black)
                c.drawString(lxi + 9, ly, f"{sym} {color.brand} {color.code}")
                col_i += 1
                if col_i >= items_per_row:
                    col_i = 0
                    ly -= 10
                    if ly < MARGIN:
                        break

    def _get_preview_image(self):
        ctx = self.ctx
        arr = ctx.quantized_image
        if arr is None:
            return None
        try:
            from reportlab.lib.utils import ImageReader
            pil = Image.fromarray(arr)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            buf.seek(0)
            return ImageReader(buf)
        except Exception as e:
            log.warning("PDF: could not embed preview: %s", e)
            return None

    def _render_matrix_to_pil(self, matrix, ctx, cell_px: int = 4,
                               draw_symbols: bool = False):
        if not matrix:
            return None
        try:
            from reportlab.lib.utils import ImageReader
            from PIL import ImageEnhance
            rows = len(matrix)
            cols = len(matrix[0]) if matrix else 0
            img = Image.new("RGB", (cols * cell_px, rows * cell_px), (255, 255, 255))
            draw = ImageDraw.Draw(img)

            font = None
            if draw_symbols and cell_px >= 6:
                try:
                    font = ImageFont.truetype("cour.ttf", max(5, cell_px - 3))
                except Exception:
                    font = ImageFont.load_default()

            for ri, row in enumerate(matrix):
                y0 = ri * cell_px
                for ci, cid in enumerate(row):
                    x0 = ci * cell_px
                    if cid == 0:
                        if draw_symbols and cell_px >= 4:
                            draw.rectangle([x0, y0, x0 + cell_px - 1, y0 + cell_px - 1],
                                           outline=(210, 210, 210))
                        continue
                    color = (ctx.color_id_map or {}).get(cid)
                    fill = (int(color.rgb[0]), int(color.rgb[1]), int(color.rgb[2])) if color else (160, 160, 160)
                    if draw_symbols:
                        draw.rectangle([x0, y0, x0 + cell_px - 1, y0 + cell_px - 1],
                                       fill=fill, outline=(100, 100, 100))
                    else:
                        draw.rectangle([x0, y0, x0 + cell_px - 1, y0 + cell_px - 1], fill=fill)
                    if draw_symbols and font and cell_px >= 6:
                        sym = (ctx.symbol_map or {}).get(cid, "")
                        if sym:
                            lum = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
                            txt_color = (0, 0, 0) if lum > 140 else (255, 255, 255)
                            draw.text((x0 + 1, y0 + 1), sym, fill=txt_color, font=font)

            if not draw_symbols:
                img = ImageEnhance.Contrast(img).enhance(1.25)
                img = ImageEnhance.Sharpness(img).enhance(1.2)

            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            buf.seek(0)
            return ImageReader(buf)
        except Exception as e:
            log.error("PDF: matrix render failed: %s", e)
            return None

    def _draw_wrapped_text(self, c, text: str, x: float, y: float,
                           max_w: float, line_height: float = 12, max_lines: int = 10):
        words = text.replace("\n", " ").split()
        line = ""
        lines_drawn = 0
        for word in words:
            test = f"{line} {word}".strip()
            if c.stringWidth(test, "Helvetica-Oblique", 8.5) <= max_w:
                line = test
            else:
                if lines_drawn >= max_lines:
                    break
                c.drawString(x, y - lines_drawn * line_height, line)
                lines_drawn += 1
                line = word
        if line and lines_drawn < max_lines:
            c.drawString(x, y - lines_drawn * line_height, line)