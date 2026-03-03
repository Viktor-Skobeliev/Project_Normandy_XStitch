"""Save/load .xstitch project files."""

from __future__ import annotations

import json
import os
import pickle
import zipfile
from typing import TYPE_CHECKING

import numpy as np

from core.exceptions import ProjectFileError
from utils.logger import get_logger

if TYPE_CHECKING:
    from core.context import ProcessingContext

log = get_logger(__name__)

_MANIFEST_NAME = "manifest.json"
_CONTEXT_NAME = "context.pkl"
_IMAGES_DIR = "images"


def save_project(ctx: "ProcessingContext", filepath: str) -> None:
    """Save ProcessingContext to a .xstitch zip archive."""
    if not filepath.endswith(".xstitch"):
        filepath += ".xstitch"

    try:

        images = {
            "original": ctx.original_image,
            "repaired": ctx.repaired_image,
            "segmented": ctx.segmented_image,
            "resized": ctx.resized_image,
            "quantized": ctx.quantized_image,
        }


        ctx_dict = ctx.model_dump(
            exclude={"original_image", "repaired_image", "segmented_image",
                     "resized_image", "quantized_image", "progress_callbacks"}
        )

        with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as zf:

            manifest = {
                "version": ctx.metadata.app_version,
                "timestamp": ctx.metadata.timestamp,
                "source": ctx.metadata.source_filename,
            }
            zf.writestr(_MANIFEST_NAME, json.dumps(manifest, indent=2, ensure_ascii=False))


            zf.writestr("metadata.json", json.dumps(ctx_dict, indent=2,
                                                     ensure_ascii=False, default=str))


            for name, arr in images.items():
                if arr is not None:
                    import io
                    buf = io.BytesIO()
                    np.save(buf, arr)
                    zf.writestr(f"{_IMAGES_DIR}/{name}.npy", buf.getvalue())

        log.info("Project saved: %s", filepath)

    except Exception as e:
        raise ProjectFileError(f"Cannot save project: {e}") from e


def load_project(filepath: str) -> "ProcessingContext":
    """Load .xstitch project file and return ProcessingContext."""
    from core.context import ProcessingContext

    if not os.path.exists(filepath):
        raise ProjectFileError(f"File not found: {filepath}")

    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            names = zf.namelist()

            meta_raw = json.loads(zf.read("metadata.json").decode("utf-8"))
            ctx = ProcessingContext(**{
                k: v for k, v in meta_raw.items()
                if k in ProcessingContext.model_fields
            })

            for img_name in ["original", "repaired", "segmented", "resized", "quantized"]:
                path = f"{_IMAGES_DIR}/{img_name}.npy"
                if path in names:
                    import io
                    arr = np.load(io.BytesIO(zf.read(path)), allow_pickle=False)
                    setattr(ctx, f"{img_name}_image", arr)

        log.info("Project loaded: %s", filepath)
        return ctx

    except ProjectFileError:
        raise
    except Exception as e:
        raise ProjectFileError(f"Cannot load project: {e}") from e
