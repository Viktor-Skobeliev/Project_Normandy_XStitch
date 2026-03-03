"""Persistent configuration storage (API key, user preferences)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from utils.logger import get_logger

log = get_logger(__name__)

_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".xstitch")
_CONFIG_FILE = os.path.join(_CONFIG_DIR, "config.json")

_DEFAULTS: Dict[str, Any] = {
    "api_key": "",
    "language": "en",
    "theme": "dark_blue",
    "canvas_count": 14,
    "palette": "DMC",
    "target_colors": 30,
    "grid_width": 200,
    "grid_height": 200,
    "dithering": True,
    "remove_background": True,
    "background_color": "white",
    "window_width": 1280,
    "window_height": 800,
}


def _load() -> Dict[str, Any]:
    if not os.path.exists(_CONFIG_FILE):
        return dict(_DEFAULTS)
    try:
        with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Merge with defaults to handle missing keys from older versions
        merged = dict(_DEFAULTS)
        merged.update(data)
        return merged
    except Exception as e:
        log.warning("Failed to read config, using defaults: %s", e)
        return dict(_DEFAULTS)


def _save(cfg: Dict[str, Any]) -> None:
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    try:
        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.error("Failed to save config: %s", e)


# ── Public API ──────────────────────────────────────────────────────────────

_cache: Optional[Dict[str, Any]] = None


def get(key: str, default: Any = None) -> Any:
    global _cache
    if _cache is None:
        _cache = _load()
    return _cache.get(key, default)


def set(key: str, value: Any) -> None:
    global _cache
    if _cache is None:
        _cache = _load()
    _cache[key] = value
    _save(_cache)


def get_api_key() -> str:
    return get("api_key", "")


def set_api_key(key: str) -> None:
    set("api_key", key)


def has_api_key() -> bool:
    return bool(get_api_key())


def get_all() -> Dict[str, Any]:
    global _cache
    if _cache is None:
        _cache = _load()
    return dict(_cache)
