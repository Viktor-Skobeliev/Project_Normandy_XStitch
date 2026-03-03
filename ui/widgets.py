"""Custom reusable widgets."""

from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk


class SectionLabel(ctk.CTkLabel):
    """Bold section header label."""
    def __init__(self, parent, text: str, **kwargs):
        super().__init__(parent, text=text, font=ctk.CTkFont(size=13, weight="bold"), **kwargs)


class StatusBar(ctk.CTkFrame):
    """Bottom status bar with message and progress."""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, height=32, corner_radius=0, **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self._label = ctk.CTkLabel(self, text="Ready", anchor="w",
                                   font=ctk.CTkFont(size=11))
        self._label.grid(row=0, column=0, padx=12, sticky="w")

        self._progress = ctk.CTkProgressBar(self, width=160, height=8)
        self._progress.grid(row=0, column=1, padx=12)
        self._progress.set(0)
        self._progress.grid_remove()

    def set_message(self, msg: str) -> None:
        self._label.configure(text=msg)

    def show_progress(self, value: float) -> None:
        """value: 0.0 to 1.0"""
        self._progress.grid()
        self._progress.set(value)

    def hide_progress(self) -> None:
        self._progress.grid_remove()
        self._progress.set(0)


class IconButton(ctk.CTkButton):
    """Square icon button."""
    def __init__(self, parent, text: str, command: Callable, width: int = 36, **kwargs):
        super().__init__(parent, text=text, command=command,
                         width=width, height=width,
                         font=ctk.CTkFont(size=13),
                         corner_radius=8, **kwargs)


class LabeledSlider(ctk.CTkFrame):
    """Slider with Entry field for manual input and two-way sync."""
    def __init__(self, parent, label: str, from_: int, to: int,
                 initial: int = None, command: Callable = None,
                 slider_width: int = 200, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self._from = from_
        self._to = to
        self._command = command
        self._updating = False

        col = 0
        if label:
            ctk.CTkLabel(self, text=label, width=120, anchor="w",
                         font=ctk.CTkFont(size=12)).grid(row=0, column=col, padx=(0, 8))
            col += 1

        self._slider = ctk.CTkSlider(self, from_=from_, to=to,
                                     width=slider_width, command=self._on_slider)
        self._slider.grid(row=0, column=col, padx=(0, 8))

        self._entry = ctk.CTkEntry(self, width=56, font=ctk.CTkFont(size=12),
                                   justify="center")
        self._entry.grid(row=0, column=col + 1)
        self._entry.bind("<Return>", self._on_entry_commit)
        self._entry.bind("<FocusOut>", self._on_entry_commit)

        init_val = initial if initial is not None else from_
        self._slider.set(init_val)
        self._entry.insert(0, str(int(init_val)))

    def _on_slider(self, value):
        if self._updating:
            return
        self._updating = True
        int_val = int(value)
        self._entry.delete(0, "end")
        self._entry.insert(0, str(int_val))
        self._updating = False
        if self._command:
            self._command(int_val)

    def _on_entry_commit(self, event=None):
        if self._updating:
            return
        try:
            val = int(self._entry.get())
            val = max(self._from, min(self._to, val))
        except ValueError:
            val = int(self._slider.get())
        self._updating = True
        self._slider.set(val)
        self._entry.delete(0, "end")
        self._entry.insert(0, str(val))
        self._updating = False
        if self._command:
            self._command(val)

    def get(self) -> int:
        try:
            return int(self._entry.get())
        except ValueError:
            return int(self._slider.get())

    def set(self, value: int) -> None:
        self._updating = True
        self._slider.set(value)
        self._entry.delete(0, "end")
        self._entry.insert(0, str(int(value)))
        self._updating = False


class LabeledCombo(ctk.CTkFrame):
    """Dropdown with label."""
    def __init__(self, parent, label: str, values: list,
                 initial: str = None, command: Callable = None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self, text=label, width=120, anchor="w",
                     font=ctk.CTkFont(size=12)).grid(row=0, column=0, padx=(0, 8))

        self._combo = ctk.CTkComboBox(self, values=values, command=command,
                                      state="readonly")
        self._combo.grid(row=0, column=1, sticky="ew")

        if initial and initial in values:
            self._combo.set(initial)
        elif values:
            self._combo.set(values[0])

    def get(self) -> str:
        return self._combo.get()

    def set(self, value: str) -> None:
        self._combo.set(value)


class ColorSwatch(ctk.CTkCanvas):
    """Small colored rectangle for legend."""
    def __init__(self, parent, rgb: tuple[int, int, int], size: int = 20, **kwargs):
        super().__init__(parent, width=size, height=size,
                         highlightthickness=0, **kwargs)
        r, g, b = rgb
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        self.create_rectangle(0, 0, size, size, fill=hex_color, outline="#888")
