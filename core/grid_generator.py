"""STEP 9 — Grid & Symbol Generator: build stitch matrix + assign symbols."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np

from core.context import ProcessingContext
from core.exceptions import GridGenerationError
from utils.logger import get_logger

log = get_logger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_SYMBOLS_FILE = os.path.join(_DATA_DIR, "symbols.json")


_CONFLICT_PAIRS = {
    frozenset(["X", "+"]): True,
    frozenset(["X", "x"]): True,
    frozenset(["|", "I"]): True,
    frozenset(["-", "="]): True,
    frozenset([".", ","]): True,
    frozenset(["!", "|"]): True,
}


def generate_grid(
    ctx: ProcessingContext,
    cleaned_bgr: np.ndarray,
    color_to_id: Dict[tuple, int],
) -> Tuple[List[List[int]], Dict[int, str]]:
    """
    STEP 9: Convert BGR image to 2D stitch matrix + assign symbols.

    Args:
        cleaned_bgr  — BGR image after confetti cleaning
        color_to_id  — mapping (B,G,R) tuple -> color ID (1-based)

    Returns:
        stitch_matrix — 2D list[list[int]], each cell = color ID
        symbol_map    — dict color_id -> symbol character
    """
    ctx.report_progress(9, "Generating stitch grid and symbols...")

    h, w = cleaned_bgr.shape[:2]
    log.info("Grid: %dx%d stitches, %d colors", w, h, len(color_to_id))

    if len(color_to_id) == 0:
        raise GridGenerationError("No color mapping provided to grid generator.")


    stitch_matrix = _build_stitch_matrix(cleaned_bgr, color_to_id, h, w)


    if ctx.fg_mask is not None:
        matrix_np = np.array(stitch_matrix, dtype=np.int32)
        bg_cells = ~ctx.fg_mask
        matrix_np[bg_cells] = 0
        stitch_matrix = matrix_np.tolist()
        bg_count = int(bg_cells.sum())
        log.info("Grid: fg_mask applied — %d background cells set to 0 (no stitch)", bg_count)


    symbols_pool = _load_symbols()
    color_ids = [cid for cid in sorted(set(color_to_id.values())) if cid != 0]
    symbol_map = _assign_symbols(color_ids, symbols_pool, stitch_matrix, w, h)

    log.info("Grid: matrix built, %d symbols assigned", len(symbol_map))
    return stitch_matrix, symbol_map


def _build_stitch_matrix(
    bgr: np.ndarray,
    color_to_id: Dict[tuple, int],
    h: int,
    w: int,
) -> List[List[int]]:
    """Convert each pixel to its color ID."""

    packed_map: Dict[int, int] = {
        b + g * 256 + r * 65536: cid
        for (b, g, r), cid in color_to_id.items()
    }

    pixels = bgr.reshape(-1, 3)
    packed = (pixels[:, 0].astype(np.int64)
              + pixels[:, 1].astype(np.int64) * 256
              + pixels[:, 2].astype(np.int64) * 65536)

    ids = np.array([packed_map.get(int(p), 1) for p in packed], dtype=np.int32)
    matrix_flat = ids.reshape(h, w)

    return matrix_flat.tolist()


def _load_symbols() -> List[str]:
    """Load symbol list from symbols.json."""
    try:
        with open(_SYMBOLS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [s["char"] for s in data["symbols"]]
    except Exception as e:
        log.warning("Could not load symbols.json: %s — using fallback", e)
        return list("ABCDEFGHJKLMNPQRTUVWXYabcdefghjklmnpqrtuvwxy346789+#@%&*=~^><!?$/\\|-.,:;")


def _assign_symbols(
    color_ids: List[int],
    symbols_pool: List[str],
    stitch_matrix: List[List[int]],
    w: int,
    h: int,
) -> Dict[int, str]:
    """
    Assign symbols to colors.
    Tries to avoid placing visually conflicting symbols in adjacent positions.
    Uses greedy approach: for each color, pick symbol with fewest conflicts
    given already-assigned neighbors.
    """
    if len(color_ids) > len(symbols_pool):

        symbols_pool = (symbols_pool * ((len(color_ids) // len(symbols_pool)) + 2))

    symbol_map: Dict[int, str] = {}
    used_symbols: set = set()


    adjacency = _compute_adjacency(stitch_matrix, color_ids, h, w)

    for color_id in color_ids:
        neighbor_ids = adjacency.get(color_id, set())
        neighbor_symbols = {symbol_map[nid] for nid in neighbor_ids if nid in symbol_map}


        chosen = None
        for sym in symbols_pool:
            if sym in used_symbols:
                continue
            if _has_visual_conflict(sym, neighbor_symbols):
                continue
            chosen = sym
            break

        if chosen is None:

            for sym in symbols_pool:
                if sym not in used_symbols:
                    chosen = sym
                    break

        if chosen is None:
            chosen = symbols_pool[color_id % len(symbols_pool)]

        symbol_map[color_id] = chosen
        used_symbols.add(chosen)

    return symbol_map


def _compute_adjacency(
    stitch_matrix: List[List[int]],
    color_ids: List[int],
    h: int,
    w: int,
) -> Dict[int, set]:
    """Build a map of which color IDs are adjacent to each other."""
    matrix = np.array(stitch_matrix, dtype=np.int32)
    adjacency: Dict[int, set] = {cid: set() for cid in color_ids}


    left = matrix[:, :-1]
    right = matrix[:, 1:]
    mask_h = left != right
    for r, c in zip(*np.where(mask_h)):
        a, b = int(left[r, c]), int(right[r, c])
        if a == 0 or b == 0:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)


    top = matrix[:-1, :]
    bottom = matrix[1:, :]
    mask_v = top != bottom
    for r, c in zip(*np.where(mask_v)):
        a, b = int(top[r, c]), int(bottom[r, c])
        if a == 0 or b == 0:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)

    return adjacency


def _has_visual_conflict(sym: str, neighbor_symbols: set) -> bool:
    """Return True if sym visually conflicts with any neighbor symbol."""
    for ns in neighbor_symbols:
        if frozenset([sym, ns]) in _CONFLICT_PAIRS:
            return True
    return False
