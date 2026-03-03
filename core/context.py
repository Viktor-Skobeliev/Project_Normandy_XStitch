"""ProcessingContext — единый контракт данных между всеми модулями пайплайна."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class SystemInfo(BaseModel):
    ram_total_mb: float = 0.0
    ram_available_mb: float = 0.0
    os_version: str = ""
    dpi_scale: float = 1.0
    python_version: str = ""


class Metadata(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    app_version: str = "2.0.0"
    source_filename: str = ""
    settings: Dict[str, Any] = Field(default_factory=dict)


class ThreadColor(BaseModel):
    brand: str
    code: str
    name: str
    rgb: List[int]  # [R, G, B]
    meters_needed: float = 0.0
    skeins_needed: float = 0.0


class ConfettiReport(BaseModel):
    total_islands_found: int = 0
    islands_merged: int = 0
    islands_removed: int = 0
    details: List[Dict[str, Any]] = Field(default_factory=list)


class ColorStats(BaseModel):
    total_colors: int = 0
    color_distribution: Dict[str, float] = Field(default_factory=dict)  # code -> percentage
    dominant_colors: List[str] = Field(default_factory=list)


class ClusterStats(BaseModel):
    requested_colors: int = 0
    actual_colors: int = 0
    inertia: float = 0.0
    iterations: int = 0


class ProcessingContext(BaseModel):
    """
    Центральный объект данных. Передаётся между всеми шагами пайплайна.
    Ни один модуль не обращается к UI напрямую — только через этот контекст.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)


    original_image: Optional[np.ndarray] = None
    repaired_image: Optional[np.ndarray] = None
    segmented_image: Optional[np.ndarray] = None
    resized_image: Optional[np.ndarray] = None
    quantized_image: Optional[np.ndarray] = None



    fg_mask: Optional[np.ndarray] = None


    stitch_matrix: Optional[List[List[int]]] = None   # 2D array of color IDs
    symbol_map: Optional[Dict[int, str]] = None        # color_id -> symbol char


    palette_selected: str = "DMC"
    palette_colors: List[ThreadColor] = Field(default_factory=list)  # итоговые цвета схемы
    color_id_map: Dict[int, ThreadColor] = Field(default_factory=dict)  # id -> color


    color_stats: Optional[ColorStats] = None
    cluster_stats: Optional[ClusterStats] = None
    confetti_report: Optional[ConfettiReport] = None


    thread_usage: Dict[str, ThreadColor] = Field(default_factory=dict)


    ai_suggestions: Optional[Dict[str, Any]] = None


    system_info: Optional[SystemInfo] = None


    metadata: Metadata = Field(default_factory=Metadata)


    current_step: int = 0
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    progress_callbacks: List[Any] = Field(default_factory=list, exclude=True)


    target_colors: int = 30          # желаемое кол-во цветов (UI слайдер)
    canvas_count: int = 14           # стежков на дюйм (14/16/18/20)
    grid_width: int = 150            # ширина схемы в стежках
    grid_height: int = 150           # высота схемы в стежках
    remove_background: bool = True   # использовать rembg
    background_color: str = "white"  # цвет фона после удаления ("white"/"transparent")

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def report_progress(self, step: int, message: str) -> None:
        self.current_step = step
        for cb in self.progress_callbacks:
            try:
                cb(step, message)
            except Exception:
                pass
