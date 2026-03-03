"""
Main Application Window for Cross-Stitch Pattern Generator.
Handles UI layout, user interactions, and coordinates with the processing pipeline.
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Callable, Optional

import customtkinter as ctk


from ui.canvas_preview import CanvasPreview
from ui.widgets import SectionLabel, StatusBar
from utils.config import get, set as cfg_set
from utils.logger import get_logger


log = get_logger(__name__)


_PALETTE_KEYS = ["DMC", "Anchor", "Gamma", "PNK", "Madeira", "Cosmo", "Sulky", "JP_Coats", "Bucilla", "Dimensions"]
_PALETTE_DISPLAY = [
    "DMC", 
    "Anchor", 
    "Gamma", 
    "ПНК им. Кирова", 
    "Madeira", 
    "Cosmo (Lecien)", 
    "Sulky", 
    "J&P Coats", 
    "Bucilla", 
    "Dimensions"
]

class MainWindow(ctk.CTkFrame):
    """
    Основное окно приложения. 
    Отвечает за визуализацию интерфейса и связь UI с ядром (core).
    """
    def __init__(self, parent, loc: dict, on_settings: Callable = None, **kwargs):
        super().__init__(parent, **kwargs)
        

        self._loc = loc
        self._mw = loc.get("main_window", {})
        self._on_settings = on_settings


        self._ctx = None
        self._image_path: Optional[str] = None
        self._processing = False
        self._zoom = 1.0


        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)


        self._build_statusbar()
        

        self._build_toolbar()
        self._build_left_sidebar()
        self._build_center()
        self._build_right_sidebar()


        self._set_status(self._mw.get("status_ready", "Готов к работе"))


        if bool(get("remove_background", True)):
            threading.Thread(target=self._prewarm_model, daemon=True).start()

    def _set_status(self, msg: str, progress: float = -1):
        """
        Обновляет текстовое сообщение и полосу прогресса в нижней части окна.
        """
        if hasattr(self, '_status') and self._status:
            self._status.set_message(msg)
            if progress >= 0:
                self._status.show_progress(progress)
            else:
                self._status.hide_progress()

    def _build_toolbar(self) -> None:
        """Верхняя панель управления с основными кнопками."""
        mw = self._mw
        tb = ctk.CTkFrame(self, height=48, corner_radius=0)
        tb.grid(row=0, column=0, columnspan=3, sticky="ew")
        tb.grid_columnconfigure(10, weight=1)

        btn_cfg = {"height": 32, "corner_radius": 8, "font": ctk.CTkFont(size=12)}


        ctk.CTkButton(tb, text=mw.get("btn_open", "Открыть изображение"),
                      command=self._open_image, **btn_cfg).grid(row=0, column=0, padx=(8, 4), pady=8)

        self._btn_process = ctk.CTkButton(tb, text=mw.get("btn_process", "Создать схему"),
                                          command=self._start_processing, **btn_cfg, state="disabled")
        self._btn_process.grid(row=0, column=1, padx=4, pady=8)

        self._btn_save = ctk.CTkButton(tb, text=mw.get("btn_save", "Сохранить PDF"),
                                       command=self._save_pdf, **btn_cfg, state="disabled")
        self._btn_save.grid(row=0, column=2, padx=4, pady=8)

        ctk.CTkButton(tb, text=mw.get("btn_load", "Загрузить проект"),
                      command=lambda: None, **btn_cfg).grid(row=0, column=3, padx=4, pady=8)


        ctk.CTkButton(tb, text="⚙", width=36, height=32, command=self._on_settings or (lambda: None),
                      corner_radius=8).grid(row=0, column=11, padx=(4, 8), pady=8)

    def _build_left_sidebar(self) -> None:
        """Боковая панель с параметрами генерации схемы."""
        mw = self._mw
        sb = ctk.CTkScrollableFrame(self, width=360, corner_radius=0)
        sb.grid(row=1, column=0, sticky="ns", padx=0, pady=0)
        sb.grid_columnconfigure(0, weight=1)


        SectionLabel(sb, mw.get("label_palette", "Бренд ниток")).pack(fill="x", padx=12, pady=(12, 4))
        self._palette_var = ctk.StringVar(value=get("palette", "DMC"))
        self._palette_combo = ctk.CTkComboBox(sb, values=_PALETTE_DISPLAY, variable=self._palette_var, state="readonly")
        self._palette_combo.pack(fill="x", padx=12, pady=(0, 8))


        SectionLabel(sb, mw.get("label_count", "Канва (каунт)")).pack(fill="x", padx=12, pady=(4, 4))
        self._count_var = ctk.StringVar(value=str(get("canvas_count", 14)))
        self._count_seg = ctk.CTkSegmentedButton(sb, values=["14", "16", "18", "20"], variable=self._count_var)
        self._count_seg.pack(fill="x", padx=12, pady=(0, 8))


        SectionLabel(sb, mw.get("label_colors", "Макс. цветов")).pack(fill="x", padx=12, pady=(8, 2))
        f_col = ctk.CTkFrame(sb, fg_color="transparent")
        f_col.pack(fill="x", padx=12, pady=2)
        f_col.columnconfigure(0, weight=1)
        
        self._colors_var = tk.StringVar(value=str(get("target_colors", 30)))
        self._colors_slider = ctk.CTkSlider(f_col, from_=4, to=80, number_of_steps=76, 
                                           command=lambda v: self._colors_var.set(str(int(v))))
        self._colors_slider.set(get("target_colors", 30))
        self._colors_slider.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        
        self._colors_entry = ctk.CTkEntry(f_col, width=45, textvariable=self._colors_var, justify="center")
        self._colors_entry.grid(row=0, column=1)


        SectionLabel(sb, mw.get("label_width", "Ширина (стежки)")).pack(fill="x", padx=12, pady=(8, 2))
        f_w = ctk.CTkFrame(sb, fg_color="transparent")
        f_w.pack(fill="x", padx=12, pady=2)
        f_w.columnconfigure(0, weight=1)
        
        self._width_var = tk.StringVar(value=str(get("grid_width", 150)))
        self._width_slider = ctk.CTkSlider(f_w, from_=30, to=500, number_of_steps=470,
                                          command=lambda v: self._width_var.set(str(int(v))))
        self._width_slider.set(get("grid_width", 150))
        self._width_slider.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        
        self._width_entry = ctk.CTkEntry(f_w, width=45, textvariable=self._width_var, justify="center")
        self._width_entry.grid(row=0, column=1)


        SectionLabel(sb, mw.get("label_height", "Высота (стежки)")).pack(fill="x", padx=12, pady=(8, 2))
        f_h = ctk.CTkFrame(sb, fg_color="transparent")
        f_h.pack(fill="x", padx=12, pady=2)
        f_h.columnconfigure(0, weight=1)
        
        self._height_var = tk.StringVar(value=str(get("grid_height", 150)))
        self._height_slider = ctk.CTkSlider(f_h, from_=30, to=500, number_of_steps=470,
                                           command=lambda v: self._height_var.set(str(int(v))))
        self._height_slider.set(get("grid_height", 150))
        self._height_slider.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        
        self._height_entry = ctk.CTkEntry(f_h, width=45, textvariable=self._height_var, justify="center")
        self._height_entry.grid(row=0, column=1)


        self._remove_bg_var = ctk.BooleanVar(value=bool(get("remove_background", True)))
        self._bg_switch = ctk.CTkSwitch(sb, text=mw.get("label_remove_bg", "Удалить фон"), 
                                        variable=self._remove_bg_var)
        self._bg_switch.pack(padx=12, pady=20, anchor="w")

    def _build_center(self) -> None:
        """Центральная область с предпросмотром схемы и вкладками."""
        mw = self._mw
        center = ctk.CTkFrame(self, corner_radius=0)
        center.grid(row=1, column=1, sticky="nsew")
        center.grid_rowconfigure(1, weight=1)
        center.grid_columnconfigure(0, weight=1)


        tab_frame = ctk.CTkFrame(center, height=36, corner_radius=0)
        tab_frame.grid(row=0, column=0, sticky="ew")


        self._preview_mode = ctk.StringVar(value="color")
        modes = [
            ("color", mw.get("tab_color", "Цветное")), 
            ("symbol", mw.get("tab_symbol", "Символьное")), 
            ("grid", mw.get("tab_grid", "Сетка"))
        ]
        for mode, label in modes:
            ctk.CTkRadioButton(tab_frame, text=label, variable=self._preview_mode, value=mode, 
                               command=lambda m=mode: self._on_mode_change(m)).pack(side="left", padx=12, pady=6)


        zoom_f = ctk.CTkFrame(tab_frame, fg_color="transparent")
        zoom_f.pack(side="right", padx=8)
        
        ctk.CTkButton(zoom_f, text="−", width=28, command=self._zoom_out).pack(side="left", padx=2)
        self._zoom_label = ctk.CTkLabel(zoom_f, text="100%", width=40)
        self._zoom_label.pack(side="left")
        ctk.CTkButton(zoom_f, text="+", width=28, command=self._zoom_in).pack(side="left", padx=2)


        self._preview = CanvasPreview(center)
        self._preview.grid(row=1, column=0, sticky="nsew")

    def _build_right_sidebar(self) -> None:
        """Правая панель для отображения легенды ниток."""
        self._legend_frame = ctk.CTkScrollableFrame(self, width=250, corner_radius=0)
        self._legend_frame.grid(row=1, column=2, sticky="ns")

    def _build_statusbar(self) -> None:
        """Создает панель статуса в самом низу окна."""
        self._status = StatusBar(self)
        self._status.grid(row=2, column=0, columnspan=3, sticky="ew")

    def _start_processing(self):
        """Запуск процесса создания схемы в отдельном потоке."""
        if self._processing or not self._image_path:
            return
            
        self._processing = True
        self._btn_process.configure(state="disabled")
        
        try:

            settings = {
                "palette": _PALETTE_KEYS[_PALETTE_DISPLAY.index(self._palette_var.get())],
                "canvas_count": int(self._count_var.get()),
                "target_colors": int(self._colors_var.get()),
                "grid_width": int(self._width_var.get()),
                "grid_height": int(self._height_var.get()),
                "remove_background": self._remove_bg_var.get(),
            }

            threading.Thread(target=self._run_thread, args=(settings,), daemon=True).start()
        except Exception as e:
            self._on_pipeline_error(str(e))

    def _run_thread(self, settings):
        """Рабочий поток пайплайна."""
        try:
            from core.pipeline import run_pipeline
            from core.context import ProcessingContext
            
            ctx = ProcessingContext(**settings)

            ctx.progress_callbacks.append(
                lambda s, m: self.after(0, lambda: self._set_status(m, s / 13.0))
            )

            run_pipeline(ctx, self._image_path)
            self._ctx = ctx
            self.after(0, self._on_pipeline_done)
        except Exception as e:
            self.after(0, lambda m=str(e): self._on_pipeline_error(m))

    def _on_pipeline_done(self):
        """Действия при успешном завершении генерации."""
        self._processing = False
        self._btn_process.configure(state="normal")
        self._btn_save.configure(state="normal")
        
        if self._ctx:
            if self._ctx.stitch_matrix:
                self._preview.set_data(
                    self._ctx.stitch_matrix, 
                    self._ctx.symbol_map, 
                    self._ctx.color_id_map
                )
            self._update_legend_ui()
            
        self._set_status(self._mw.get("status_done", "Схема создана!"))

    def _update_legend_ui(self):
        """Отрисовывает список ниток в правой панели."""
        for child in self._legend_frame.winfo_children():
            child.destroy()

        if not self._ctx or not self._ctx.thread_usage:
            return

        SectionLabel(self._legend_frame, "Легенда ниток").pack(fill="x", padx=10, pady=5)






        code_to_id: dict = {}
        if self._ctx.color_id_map:
            for cid, color in self._ctx.color_id_map.items():
                code_to_id[color.code] = cid

        for code, thread in self._ctx.thread_usage.items():
            f = ctk.CTkFrame(self._legend_frame, fg_color="transparent")
            f.pack(fill="x", padx=5, pady=2)


            r, g, b = thread.rgb
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            ctk.CTkLabel(f, text="", width=16, height=16,
                         fg_color=hex_color, corner_radius=2).pack(side="left", padx=2)


            c_id = code_to_id.get(code)
            sym = self._ctx.symbol_map.get(c_id, "?") if (self._ctx.symbol_map and c_id) else "?"
            ctk.CTkLabel(f, text=sym, width=20,
                         font=("Arial", 12, "bold")).pack(side="left")


            meters = thread.meters_needed
            skeins = thread.skeins_needed
            label_text = f"{thread.brand} {code} · {thread.name[:18]}"
            ctk.CTkLabel(f, text=label_text,
                         font=("Arial", 10)).pack(side="left", padx=4)

            ctk.CTkLabel(f, text=f"{meters:.1f}m ({skeins:.1f} мот.)",
                         font=("Arial", 10),
                         text_color=("gray40", "gray70")).pack(side="right", padx=4)

    def _save_pdf(self):
        """Экспорт результата в PDF."""
        if not self._ctx: return
        p = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if p:
            try:
                from core.pipeline import export_pdf_result
                export_pdf_result(self._ctx, p)
                self._set_status(f"Сохранено: {os.path.basename(p)}")
            except ImportError:
                messagebox.showwarning("Внимание", "Функция экспорта в PDF не найдена в core.pipeline")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить PDF: {e}")

    def _on_pipeline_error(self, msg: str):
        """Обработка ошибок выполнения пайплайна."""
        self._processing = False
        self._btn_process.configure(state="normal")
        messagebox.showerror("Ошибка пайплайна", f"Произошла ошибка: {msg}")
        self._set_status("Ошибка выполнения")

    def _open_image(self):
        """Открывает диалог выбора файла."""
        p = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if p:
            self._image_path = p
            self._btn_process.configure(state="normal")
            self._set_status(f"Файл: {os.path.basename(p)}")

    def _on_mode_change(self, m):
        """Переключает режим отображения на холсте."""
        self._preview.set_mode(m)

    def _zoom_in(self):
        """Увеличить масштаб."""
        self._zoom = min(self._zoom * 1.2, 5.0)
        self._update_zoom()

    def _zoom_out(self):
        """Уменьшить масштаб."""
        self._zoom = max(self._zoom / 1.2, 0.2)
        self._update_zoom()

    def _update_zoom(self):
        """Обновляет текст индикатора и масштаб холста."""
        self._zoom_label.configure(text=f"{int(self._zoom * 100)}%")
        self._preview.set_zoom(self._zoom)

    def _prewarm_model(self):
        """Предварительная загрузка модели AI."""
        try:
            from core.segment import prewarm_session
            prewarm_session()
        except Exception:
            log.warning("Не удалось выполнить prewarm модели.")