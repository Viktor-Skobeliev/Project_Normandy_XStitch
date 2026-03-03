"""Navigation controller — routes between screens."""

from __future__ import annotations

import json
import os

import customtkinter as ctk

from utils.config import get, set as cfg_set
from utils.logger import get_logger

log = get_logger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_LOCALES_FILE = os.path.join(_DATA_DIR, "locales.json")


def _load_locales() -> dict:
    with open(_LOCALES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


class NavigationController:
    """
    Manages which screen is displayed.
    LOCAL MODE: API key screen is disabled.
    Flow:
      1. language_screen (only on very first launch)
      2. main_window     (main app)
    """

    def __init__(self, root: ctk.CTk):
        self._root = root
        self._locales = _load_locales()
        self._lang = get("language", "en")
        self._current_frame: ctk.CTkFrame | None = None
        # Проверяем, установлен ли язык. Если нет — считаем это первым запуском.
        self._first_launch = get("language") is None

    def _loc(self) -> dict:
        return self._locales.get(self._lang, self._locales["en"])

    def start(self) -> None:
        # LOCAL MODE: Больше не вызываем _show_api_key
        if self._first_launch:
            self._show_language(self._on_language_selected)
        else:
            self._show_main()

    # ── Screen transitions ────────────────────────────────────────────────────

    # Метод _show_api_key удален за ненадобностью в локальном режиме

    def _show_language(self, callback) -> None:
        self._clear()
        frame = _LanguageScreen(
            self._root,
            loc=self._loc(),
            current_lang=self._lang,
            on_select=callback,
        )
        frame.pack(fill="both", expand=True)
        self._current_frame = frame
        log.info("Nav: showing language_screen")

    def _show_main(self) -> None:
        self._clear()
        from ui.main_window import MainWindow
        from ui.theme_manager import apply_theme

        apply_theme(get("theme", "dark_blue"))

        frame = MainWindow(
            self._root,
            loc=self._loc(),
            on_settings=self._open_settings,
        )
        frame.pack(fill="both", expand=True)
        self._current_frame = frame
        log.info("Nav: showing main_window (Local AI Mode)")

    def _open_settings(self) -> None:
        from ui.settings_screen import SettingsScreen

        def on_save():
            self._lang = get("language", "en")
            log.info("Settings saved, lang=%s", self._lang)

        SettingsScreen(self._root, loc=self._loc(), on_save=on_save)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_language_selected(self, lang: str) -> None:
        self._lang = lang
        cfg_set("language", lang)
        log.info("Language selected: %s", lang)
        self._show_main()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _clear(self) -> None:
        if self._current_frame:
            self._current_frame.pack_forget()
            self._current_frame.destroy()
            self._current_frame = None


class _LanguageScreen(ctk.CTkFrame):
    """Simple language selection screen."""

    def __init__(self, parent, loc: dict, current_lang: str, on_select, **kwargs):
        super().__init__(parent, **kwargs)
        self._on_select = on_select

        loc_screen = loc.get("language_screen", {})
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        center = ctk.CTkFrame(self, width=360, corner_radius=16)
        center.grid(row=0, column=0)
        center.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            center,
            text=loc_screen.get("title", "Select Language"),
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, pady=(40, 24), padx=40)

        self._var = ctk.StringVar(value=current_lang)

        languages = {
            "en": "English",
            "ru": "Русский",
            "ua": "Українська",
        }
        for i, (key, label) in enumerate(languages.items()):
            ctk.CTkRadioButton(
                center,
                text=label,
                variable=self._var,
                value=key,
                font=ctk.CTkFont(size=14),
            ).grid(row=i + 1, column=0, sticky="w", padx=60, pady=6)

        ctk.CTkButton(
            center,
            text=loc_screen.get("btn_continue", "Continue"),
            command=self._on_continue,
            width=280, height=44,
            font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=10,
        ).grid(row=10, column=0, padx=40, pady=(24, 40))

    def _on_continue(self) -> None:
        self._on_select(self._var.get())