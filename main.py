"""XStitch Pattern Generator — Entry point."""

from __future__ import annotations

import asyncio
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import get_logger
from utils.version import APP_NAME, APP_VERSION

log = get_logger(__name__)


def main() -> None:
    log.info("Starting %s v%s", APP_NAME, APP_VERSION)

    from utils.gpu import log_compute_device
    log_compute_device()

    import customtkinter as ctk
    from ui.navigation_controller import NavigationController


    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        pass

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title(f"{APP_NAME} v{APP_VERSION}")
    root.minsize(900, 700)


    from utils.config import get
    w = get("window_width", 1280)
    h = get("window_height", 800)
    try:
        import ctypes
        user32 = ctypes.windll.user32
        pw = user32.GetSystemMetrics(0)   # SM_CXSCREEN — primary monitor width
        ph = user32.GetSystemMetrics(1)   # SM_CYSCREEN — primary monitor height
    except Exception:
        root.update_idletasks()
        pw = root.winfo_screenwidth()
        ph = root.winfo_screenheight()
    x = max(0, (pw - w) // 2)
    y = max(0, (ph - h) // 2)
    root.geometry(f"{w}x{h}+{x}+{y}")


    def on_close():
        from utils.config import set as cfg_set
        cfg_set("window_width", root.winfo_width())
        cfg_set("window_height", root.winfo_height())
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    nav = NavigationController(root)
    nav.start()

    root.mainloop()
    log.info("Application closed.")


if __name__ == "__main__":
    main()
