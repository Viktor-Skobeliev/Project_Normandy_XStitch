"""STEP 11 — Page Tiler: split large stitch matrix into A4-sized pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from core.context import ProcessingContext
from utils.logger import get_logger

log = get_logger(__name__)

# A4 at 300 DPI = 2480 x 3508 px
# At typical PDF rendering: grid cell = ~10pt (3.5mm)
# On A4 (210x297mm), usable area ~180x267mm -> ~51x76 cells per page
CELLS_PER_PAGE_W = 50   # stitches per page width
CELLS_PER_PAGE_H = 70   # stitches per page height


@dataclass
class PageTile:
    page_num: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    grid_row: int   # position in page grid (0-indexed)
    grid_col: int

    @property
    def width(self) -> int:
        return self.col_end - self.col_start

    @property
    def height(self) -> int:
        return self.row_end - self.row_start


def tile_pages(
    ctx: ProcessingContext,
    stitch_matrix: List[List[int]],
) -> List[PageTile]:
    """
    STEP 11: Divide stitch matrix into A4-sized page tiles.

    Returns list of PageTile objects with coordinates into the stitch matrix.
    """
    ctx.report_progress(11, "Tiling pages...")

    total_rows = len(stitch_matrix)
    total_cols = len(stitch_matrix[0]) if stitch_matrix else 0

    tiles: List[PageTile] = []
    page_num = 1
    grid_row = 0

    for row_start in range(0, total_rows, CELLS_PER_PAGE_H):
        row_end = min(row_start + CELLS_PER_PAGE_H, total_rows)
        grid_col = 0

        for col_start in range(0, total_cols, CELLS_PER_PAGE_W):
            col_end = min(col_start + CELLS_PER_PAGE_W, total_cols)

            tiles.append(PageTile(
                page_num=page_num,
                row_start=row_start,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
                grid_row=grid_row,
                grid_col=grid_col,
            ))

            page_num += 1
            grid_col += 1

        grid_row += 1

    n_grid_rows = grid_row
    n_grid_cols = max(t.grid_col for t in tiles) + 1 if tiles else 0

    log.info(
        "Tiler: %dx%d stitches -> %d pages (%d rows x %d cols of pages)",
        total_cols, total_rows, len(tiles), n_grid_rows, n_grid_cols,
    )
    return tiles


def get_page_map_dimensions(tiles: List[PageTile]) -> Tuple[int, int]:
    """Return (n_cols, n_rows) of page grid."""
    if not tiles:
        return 0, 0
    n_rows = max(t.grid_row for t in tiles) + 1
    n_cols = max(t.grid_col for t in tiles) + 1
    return n_cols, n_rows
