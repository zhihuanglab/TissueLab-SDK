"""TissueLab SDK package namespace.

This package provides the public import path `tissuelab_sdk.wrapper`.
"""

# Auto-patch tiffslide for 4D z-stack support
# This happens automatically when tissuelab_sdk is imported
try:
    from . import tiffslide_autopatch
except Exception:
    pass  # Silently ignore if patching fails

__all__ = ["wrapper"]


