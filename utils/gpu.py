"""GPU/CPU detection utility."""

from __future__ import annotations

from utils.logger import get_logger

log = get_logger(__name__)

# Priority order: TensorRT > CUDA > CPU
_PROVIDER_PRIORITY = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]


def get_ort_providers() -> list[str]:
    """
    Return ordered list of available ONNX Runtime execution providers.
    Priority: TensorRT > CUDA > CPU.
    Always includes CPUExecutionProvider as final fallback.
    """
    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
        providers = [p for p in _PROVIDER_PRIORITY if p in available]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        return providers
    except ImportError:
        return ["CPUExecutionProvider"]


def get_compute_device() -> str:
    """
    Returns human-readable string describing the active compute device:
    'GPU (CUDA+TensorRT)', 'GPU (CUDA)', or 'CPU'.
    """
    providers = get_ort_providers()
    has_trt = "TensorrtExecutionProvider" in providers
    has_cuda = "CUDAExecutionProvider" in providers

    if has_trt:
        return "GPU (CUDA+TensorRT)"
    if has_cuda:
        return "GPU (CUDA)"
    return "CPU"


def log_compute_device() -> str:
    """Log and return the active compute device string."""
    device = get_compute_device()
    log.info("Compute device: %s", device)
    return device
