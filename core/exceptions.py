"""Custom exceptions for XStitch pipeline."""


class XStitchBaseError(Exception):
    """Base exception for all XStitch errors."""
    pass


class ImageCorruptError(XStitchBaseError):
    """Raised when input image cannot be read or is corrupted."""
    pass


class ImageTooLargeError(XStitchBaseError):
    """Raised when image exceeds safe memory threshold."""
    pass


class SegmentationError(XStitchBaseError):
    """Raised when background removal fails."""
    pass


class QuantizationError(XStitchBaseError):
    """Raised when KMeans color quantization fails."""
    pass


class PaletteNotFoundError(XStitchBaseError):
    """Raised when requested brand palette is missing from data."""
    pass


class GridGenerationError(XStitchBaseError):
    """Raised when stitch matrix cannot be generated."""
    pass


class ExportError(XStitchBaseError):
    """Raised when PDF/file export fails."""
    pass


class APIFailError(XStitchBaseError):
    """Raised when Claude API call fails or times out."""
    pass


class APIKeyInvalidError(XStitchBaseError):
    """Raised when the provided Claude API key is invalid."""
    pass


class ConfigError(XStitchBaseError):
    """Raised when config file is missing or malformed."""
    pass


class ProjectFileError(XStitchBaseError):
    """Raised when .xstitch project file cannot be read/written."""
    pass


class MemoryGuardError(XStitchBaseError):
    """Raised when available RAM is critically low."""
    pass
