"""Canvas preview widget — renders stitch grid (color/symbol/grid modes)."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import customtkinter as ctk

from core.context import ThreadColor
from utils.logger import get_logger

log = get_logger(__name__)

CELL_PX = 10       # pixels per stitch cell in preview
GRID_COLOR = (180, 180, 180)
TEXT_COLOR = (0, 0, 0)
BG_COLOR = (255, 255, 255)
RULER_INTERVAL = 10  # draw ruler every N stitches


class CanvasPreview(ctk.CTkFrame):
    """
    Zoomable stitch grid preview with three modes:
      - color   : colored cells with symbol overlay
      - symbol  : B&W cells with symbols
      - grid    : bare grid (color, no symbols)
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._canvas = ctk.CTkCanvas(self, bg="#1a1a2e", highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nsew")

        # Scrollbars
        self._hbar = ctk.CTkScrollbar(self, orientation="horizontal",
                                       command=self._canvas.xview)
        self._hbar.grid(row=1, column=0, sticky="ew")
        self._vbar = ctk.CTkScrollbar(self, orientation="vertical",
                                       command=self._canvas.yview)
        self._vbar.grid(row=0, column=1, sticky="ns")

        self._canvas.configure(
            xscrollcommand=self._hbar.set,
            yscrollcommand=self._vbar.set,
        )
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)
        self._canvas.bind("<Control-MouseWheel>", self._on_zoom)

        # State
        self._stitch_matrix: Optional[List[List[int]]] = None
        self._symbol_map: Optional[Dict[int, str]] = None
        self._color_id_map: Optional[Dict[int, ThreadColor]] = None
        self._mode: str = "color"
        self._zoom: float = 1.0
        self._photo = None  # keep reference to prevent GC

    # ── Public API ────────────────────────────────────────────────────────────

    def set_data(
        self,
        stitch_matrix: List[List[int]],
        symbol_map: Dict[int, str],
        color_id_map: Dict[int, ThreadColor],
    ) -> None:
        self._stitch_matrix = stitch_matrix
        self._symbol_map = symbol_map
        self._color_id_map = color_id_map
        self._render()

    def set_mode(self, mode: str) -> None:
        """mode: 'color', 'symbol', 'grid'"""
        self._mode = mode
        if self._stitch_matrix:
            self._render()

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.3, min(zoom, 5.0))
        if self._stitch_matrix:
            self._render()

    def clear(self) -> None:
        self._canvas.delete("all")
        self._stitch_matrix = None

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self) -> None:
        if not self._stitch_matrix:
            return

        matrix = self._stitch_matrix
        h = len(matrix)
        w = len(matrix[0]) if matrix else 0
        cell = max(2, int(CELL_PX * self._zoom))

        img_w = w * cell
        img_h = h * cell

        img = Image.new("RGB", (img_w, img_h), BG_COLOR)
        draw = ImageDraw.Draw(img)

        # Try to load a small monospace font
        try:
            font_size = max(4, cell - 2)
            font = ImageFont.truetype("cour.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        for row_idx, row in enumerate(matrix):
            y0 = row_idx * cell
            y1 = y0 + cell
            for col_idx, color_id in enumerate(row):
                x0 = col_idx * cell
                x1 = x0 + cell

                # color_id=0 → background cell (no stitch, no symbol)
                if color_id == 0:
                    draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=BG_COLOR)
                    if cell >= 4:
                        draw.rectangle([x0, y0, x1 - 1, y1 - 1],
                                       outline=GRID_COLOR, width=1)
                    continue

                # Fill cell
                if self._mode in ("color", "grid"):
                    thread = (self._color_id_map or {}).get(color_id)
                    if thread:
                        r, g, b = thread.rgb
                        fill = (r, g, b)
                    else:
                        fill = (240, 240, 240)
                else:  # symbol mode
                    fill = BG_COLOR

                draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=fill)

                # Grid line
                if cell >= 4:
                    draw.rectangle([x0, y0, x1 - 1, y1 - 1],
                                   outline=GRID_COLOR, width=1)

                # Symbol
                if self._mode in ("color", "symbol") and cell >= 6:
                    sym = (self._symbol_map or {}).get(color_id, "")
                    if sym:
                        text_color = TEXT_COLOR if self._mode == "symbol" else _contrast_color(fill)
                        draw.text((x0 + 1, y0 + 1), sym, fill=text_color, font=font)

        # Ruler marks every 10 stitches
        if cell >= 4:
            for i in range(0, w, RULER_INTERVAL):
                x = i * cell
                draw.text((x + 1, 0), str(i), fill=(100, 100, 100), font=font)
            for i in range(0, h, RULER_INTERVAL):
                y = i * cell
                draw.text((0, y + 1), str(i), fill=(100, 100, 100), font=font)

        # Display on canvas
        from PIL import ImageTk
        self._photo = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self._canvas.configure(scrollregion=(0, 0, img_w, img_h))

    # ── Events ────────────────────────────────────────────────────────────────

    def _on_mousewheel(self, event) -> None:
        self._canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def _on_zoom(self, event) -> None:
        if event.delta > 0:
            self.set_zoom(self._zoom * 1.2)
        else:
            self.set_zoom(self._zoom / 1.2)


def _contrast_color(bg_rgb: tuple) -> tuple:
    """Return black or white text color based on background luminance."""
    r, g, b = bg_rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)
