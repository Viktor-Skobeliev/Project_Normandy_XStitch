"""Theme manager — loads themes.json and applies to CustomTkinter."""

from __future__ import annotations

import json
import os

import customtkinter as ctk

from utils.config import get, set as cfg_set
from utils.logger import get_logger

log = get_logger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_THEMES_FILE = os.path.join(_DATA_DIR, "themes.json")

_themes_cache: dict | None = None


def _load_themes() -> dict:
    global _themes_cache
    if _themes_cache is None:
        with open(_THEMES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        _themes_cache = data
    return _themes_cache


def get_available_themes() -> list[str]:
    return list(_load_themes()["themes"].keys())


def get_theme(name: str) -> dict:
    themes = _load_themes()["themes"]
    return themes.get(name, themes["dark_blue"])


def get_current_theme() -> dict:
    return get_theme(get("theme", "dark_blue"))


def apply_theme(name: str) -> None:
    """Apply theme by name — sets CTk appearance and saves preference."""
    theme = get_theme(name)
    appearance = theme.get("appearance", "dark")
    ctk.set_appearance_mode(appearance)
    cfg_set("theme", name)
    log.info("Theme applied: %s (%s)", name, appearance)


def get_theme_labels() -> dict[str, str]:
    """Return {key: label} mapping for UI display."""
    themes = _load_themes()["themes"]
    return {k: v["label"] for k, v in themes.items()}
