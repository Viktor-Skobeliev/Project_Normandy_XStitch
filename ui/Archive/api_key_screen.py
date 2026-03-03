"""API Key entry screen — shown on first launch."""

from __future__ import annotations

from typing import Callable

import customtkinter as ctk

from utils.logger import get_logger

log = get_logger(__name__)


class ApiKeyScreen(ctk.CTkFrame):
    """
    Blocks app until a valid Claude API key is entered.
    On success calls on_success(api_key).
    On skip calls on_skip().
    """

    def __init__(self, parent, loc: dict,
                 on_success: Callable[[str], None],
                 on_skip: Callable[[], None],
                 **kwargs):
        super().__init__(parent, **kwargs)
        self._loc = loc
        self._on_success = on_success
        self._on_skip = on_skip

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build()

    def _build(self) -> None:
        loc = self._loc.get("api_key_screen", {})

        center = ctk.CTkFrame(self, width=480, corner_radius=16)
        center.grid(row=0, column=0)
        center.grid_columnconfigure(0, weight=1)


        ctk.CTkLabel(
            center,
            text=loc.get("title", "Welcome"),
            font=ctk.CTkFont(size=26, weight="bold"),
        ).grid(row=0, column=0, pady=(40, 8), padx=50)


        ctk.CTkLabel(
            center,
            text=loc.get("subtitle", "Enter your Claude API key to get started"),
            font=ctk.CTkFont(size=13),
            text_color=("gray40", "gray70"),
            wraplength=380,
        ).grid(row=1, column=0, pady=(0, 24), padx=50)


        self._entry = ctk.CTkEntry(
            center,
            placeholder_text=loc.get("placeholder", "sk-ant-..."),
            width=380,
            height=44,
            font=ctk.CTkFont(size=13),
            show="•",
        )
        self._entry.grid(row=2, column=0, padx=50, pady=(0, 8))


        self._show_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            center,
            text="Show key",
            variable=self._show_var,
            command=self._toggle_show,
            font=ctk.CTkFont(size=11),
        ).grid(row=3, column=0, padx=50, sticky="w", pady=(0, 16))


        self._error_label = ctk.CTkLabel(
            center,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#e05555",
        )
        self._error_label.grid(row=4, column=0, padx=50)


        self._status_label = ctk.CTkLabel(
            center, text="", font=ctk.CTkFont(size=11),
            text_color=("gray40", "gray60"),
        )
        self._status_label.grid(row=5, column=0, padx=50)


        self._btn_confirm = ctk.CTkButton(
            center,
            text=loc.get("btn_confirm", "Confirm"),
            command=self._on_confirm,
            width=380,
            height=44,
            font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=10,
        )
        self._btn_confirm.grid(row=6, column=0, padx=50, pady=(12, 8))


        ctk.CTkButton(
            center,
            text=loc.get("btn_skip", "Skip AI features"),
            command=self._on_skip,
            width=380,
            height=36,
            font=ctk.CTkFont(size=12),
            fg_color="transparent",
            border_width=1,
            corner_radius=10,
        ).grid(row=7, column=0, padx=50, pady=(0, 40))

        self._loc_screen = loc

    def _toggle_show(self) -> None:
        self._entry.configure(show="" if self._show_var.get() else "•")

    def _on_confirm(self) -> None:
        loc = self._loc_screen
        key = self._entry.get().strip()

        if not key:
            self._show_error(loc.get("error_empty", "API key cannot be empty"))
            return

        self._show_error("")
        self._status_label.configure(text=loc.get("checking", "Checking key..."))
        self._btn_confirm.configure(state="disabled")


        self.after(50, lambda: self._validate_key(key))

    def _validate_key(self, key: str) -> None:
        from ai.enhancer import validate_api_key
        valid, error_msg = validate_api_key(key)

        if valid:
            from utils.config import set_api_key
            set_api_key(key)
            self._status_label.configure(text="")
            self._on_success(key)
        else:
            self._btn_confirm.configure(state="normal")
            self._status_label.configure(text="")
            self._show_error(self._loc_screen.get("error_invalid", "Invalid API key."))

    def _show_error(self, msg: str) -> None:
        self._error_label.configure(text=msg)
