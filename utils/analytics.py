"""Pipeline step timing analytics."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)


class StepTimer:
    def __init__(self, name: str) -> None:
        self.name = name
        self._start: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self) -> "StepTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        if self._start is not None:
            self.elapsed = time.perf_counter() - self._start
            log.debug("Step [%s] completed in %.3f s", self.name, self.elapsed)


class PipelineAnalytics:
    def __init__(self) -> None:
        self._timings: Dict[str, float] = {}

    def record(self, step_name: str, elapsed: float) -> None:
        self._timings[step_name] = elapsed

    def step(self, name: str) -> StepTimer:
        t = StepTimer(name)
        return t

    def get_report(self) -> List[Dict]:
        return [
            {"step": k, "seconds": round(v, 3)}
            for k, v in self._timings.items()
        ]

    def total_seconds(self) -> float:
        return sum(self._timings.values())

    def log_summary(self) -> None:
        log.info("─── Pipeline timing summary ───")
        for step, secs in self._timings.items():
            log.info("  %-30s %6.3f s", step, secs)
        log.info("  %-30s %6.3f s", "TOTAL", self.total_seconds())
