"""Settings screen — algorithm + UI preferences."""

from __future__ import annotations

from typing import Callable

import customtkinter as ctk

from ui.theme_manager import get_available_themes, get_theme_labels, apply_theme
from ui.widgets import SectionLabel, LabeledSlider, LabeledCombo
from utils.config import get, set as cfg_set
from utils.logger import get_logger

log = get_logger(__name__)

_CANVAS_COUNTS = ["14", "16", "18", "20"]
_LANGUAGES = {"en": "English", "ru": "Русский", "ua": "Українська"}
_BG_COLORS = ["white", "transparent"]
_PDF_FORMATS = ["A4", "Letter"]


class SettingsScreen(ctk.CTkToplevel):
    """Modal settings window."""

    def __init__(self, parent, loc: dict, on_save: Callable = None, **kwargs):
        super().__init__(parent, **kwargs)
        self.title(loc.get("settings_screen", {}).get("title", "Settings"))
        self.geometry("560x640")
        self.resizable(False, False)
        self.grab_set()  # modal

        self._loc = loc.get("settings_screen", {})
        self._on_save = on_save
        self.grid_columnconfigure(0, weight=1)

        self._build()

    def _build(self) -> None:
        loc = self._loc

        scroll = ctk.CTkScrollableFrame(self)
        scroll.grid(row=0, column=0, sticky="nsew", padx=16, pady=(16, 8))
        scroll.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        row = 0


        SectionLabel(scroll, loc.get("section_ui", "Interface")).grid(
            row=row, column=0, sticky="w", pady=(0, 8))
        row += 1


        theme_labels = get_theme_labels()
        theme_keys = list(theme_labels.keys())
        theme_display = [theme_labels[k] for k in theme_keys]

        current_theme_key = get("theme", "dark_blue")
        current_theme_label = theme_labels.get(current_theme_key, theme_display[0])

        self._theme_combo = LabeledCombo(
            scroll,
            label=loc.get("label_theme", "Theme"),
            values=theme_display,
            initial=current_theme_label,
        )
        self._theme_combo.grid(row=row, column=0, sticky="ew", pady=4)
        row += 1
        self._theme_keys = theme_keys
        self._theme_display = theme_display


        lang_keys = list(_LANGUAGES.keys())
        lang_display = list(_LANGUAGES.values())
        current_lang = get("language", "en")
        current_lang_label = _LANGUAGES.get(current_lang, lang_display[0])

        self._lang_combo = LabeledCombo(
            scroll,
            label=loc.get("label_language", "Language"),
            values=lang_display,
            initial=current_lang_label,
        )
        self._lang_combo.grid(row=row, column=0, sticky="ew", pady=4)
        row += 1
        self._lang_keys = lang_keys
        self._lang_display = lang_display


        ctk.CTkFrame(scroll, height=1, fg_color=("gray80", "gray30")).grid(
            row=row, column=0, sticky="ew", pady=12)
        row += 1


        SectionLabel(scroll, loc.get("section_algorithm", "Algorithm")).grid(
            row=row, column=0, sticky="w", pady=(0, 8))
        row += 1


        self._confetti_slider = LabeledSlider(
            scroll,
            label=loc.get("label_confetti_threshold", "Confetti Threshold (px)"),
            from_=1, to=10,
            initial=get("confetti_threshold", 3),
        )
        self._confetti_slider.grid(row=row, column=0, sticky="ew", pady=4)
        row += 1


        self._bg_combo = LabeledCombo(
            scroll,
            label=loc.get("label_bg_color", "Background Color"),
            values=_BG_COLORS,
            initial=get("background_color", "white"),
        )
        self._bg_combo.grid(row=row, column=0, sticky="ew", pady=4)
        row += 1


        self._dither_var = ctk.BooleanVar(value=bool(get("dithering", False)))
        dither_row = ctk.CTkFrame(scroll, fg_color="transparent")
        dither_row.grid(row=row, column=0, sticky="ew", pady=4)
        dither_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(dither_row, text=loc.get("label_dithering", "Dithering"),
                     width=120, anchor="w",
                     font=ctk.CTkFont(size=12)).grid(row=0, column=0)
        ctk.CTkSwitch(dither_row, text="", variable=self._dither_var).grid(
            row=0, column=1, sticky="w", padx=8)
        row += 1


        ctk.CTkFrame(scroll, height=1, fg_color=("gray80", "gray30")).grid(
            row=row, column=0, sticky="ew", pady=12)
        row += 1


        SectionLabel(scroll, loc.get("section_export", "Export")).grid(
            row=row, column=0, sticky="w", pady=(0, 8))
        row += 1

        self._pdf_combo = LabeledCombo(
            scroll,
            label=loc.get("label_pdf_paper", "PDF Paper Size"),
            values=_PDF_FORMATS,
            initial=get("pdf_format", "A4"),
        )
        self._pdf_combo.grid(row=row, column=0, sticky="ew", pady=4)
        row += 1


        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 16))
        btn_row.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(
            btn_row, text=loc.get("btn_cancel", "Cancel"),
            command=self.destroy,
            fg_color="transparent", border_width=1,
        ).grid(row=0, column=0, padx=(0, 8), sticky="ew")

        ctk.CTkButton(
            btn_row, text=loc.get("btn_save", "Save"),
            command=self._save,
        ).grid(row=0, column=1, padx=(8, 0), sticky="ew")

    def _save(self) -> None:

        label = self._theme_combo.get()
        if label in self._theme_display:
            key = self._theme_keys[self._theme_display.index(label)]
            apply_theme(key)


        lang_label = self._lang_combo.get()
        if lang_label in self._lang_display:
            lang_key = self._lang_keys[self._lang_display.index(lang_label)]
            cfg_set("language", lang_key)

        cfg_set("confetti_threshold", self._confetti_slider.get())
        cfg_set("background_color", self._bg_combo.get())
        cfg_set("dithering", self._dither_var.get())
        cfg_set("pdf_format", self._pdf_combo.get())

        log.info("Settings saved")

        if self._on_save:
            self._on_save()

        self.destroy()
