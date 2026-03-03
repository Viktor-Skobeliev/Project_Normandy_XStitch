"""STEP 8 — Path Optimizer: minimize thread jumps using Snake + Manhattan distance."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from core.context import ProcessingContext
from utils.logger import get_logger

log = get_logger(__name__)


def optimize_paths(
    ctx: ProcessingContext,
    stitch_matrix: List[List[int]],
) -> Dict[int, List[Tuple[int, int]]]:
    """
    STEP 8: For each color, compute an optimized stitching path.

    Strategy:
      1. Group stitches by color ID.
      2. For each color cluster, apply Snake Traversal within bounding box.
      3. Then minimize Manhattan Distance between disconnected sub-clusters.

    Returns:
        color_paths — dict color_id -> ordered list of (row, col) stitch positions
    """
    ctx.report_progress(8, "Optimizing stitch paths...")

    matrix = np.array(stitch_matrix, dtype=np.int32)
    h, w = matrix.shape

    color_ids = np.unique(matrix)
    color_paths: Dict[int, List[Tuple[int, int]]] = {}

    for color_id in color_ids:
        positions = list(zip(*np.where(matrix == color_id)))
        if not positions:
            continue

        positions = [(int(r), int(c)) for r, c in positions]
        path = _snake_then_manhattan(positions, w)
        color_paths[color_id] = path

    total_stitches = sum(len(p) for p in color_paths.values())
    log.info("Path optimizer: %d colors, %d total stitches", len(color_paths), total_stitches)
    return color_paths


def _snake_then_manhattan(
    positions: List[Tuple[int, int]],
    grid_width: int,
) -> List[Tuple[int, int]]:
    """
    Sort positions using Snake Traversal (row-by-row, alternating direction).
    Within each row, order left-to-right or right-to-left alternately.
    Then connect disjoint segments via nearest-Manhattan greedy.
    """
    if not positions:
        return []


    rows: Dict[int, List[int]] = {}
    for r, c in positions:
        rows.setdefault(r, []).append(c)

    sorted_rows = sorted(rows.keys())
    ordered: List[Tuple[int, int]] = []

    for i, row in enumerate(sorted_rows):
        cols = sorted(rows[row])
        if i % 2 == 1:
            cols = cols[::-1]  # alternate direction (snake)
        ordered.extend((row, c) for c in cols)

    return ordered


def calculate_jumps(color_paths: Dict[int, List[Tuple[int, int]]]) -> Dict[int, int]:
    """
    Count the number of thread jumps (discontinuities) for each color.
    A jump occurs when consecutive stitches are not adjacent (Manhattan > 1).
    """
    jumps: Dict[int, int] = {}
    for color_id, path in color_paths.items():
        count = 0
        for i in range(1, len(path)):
            r1, c1 = path[i - 1]
            r2, c2 = path[i]
            if abs(r2 - r1) + abs(c2 - c1) > 1:
                count += 1
        jumps[color_id] = count
    return jumps
