import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _import_fastslide():
    try:
        import fastslide

        return fastslide
    except ImportError as exc:
        raise ImportError(
            "fastslide is required to read iSyntax files. "
            "On Windows, import fastslide before cv2, pyvips, or tiffslide."
        ) from exc

def _to_rgba_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert an interleaved image array to RGBA uint8."""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 4:
        return arr
    elif arr.shape[-1] == 3:
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
        return np.concatenate([arr, alpha], axis=-1)
    elif arr.shape[-1] == 1:
        gray = arr[..., 0]
        rgb = np.stack([gray, gray, gray], axis=-1)
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
        return np.concatenate([rgb, alpha], axis=-1)
    else:
        raise ValueError(f"Unsupported channel count: {arr.shape[-1]}")

    alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=-1)


def _fastslide_image_to_pil(image) -> Image.Image:
    arr = _to_rgba_uint8(image.to_interleaved().numpy())
    return Image.fromarray(arr, mode="RGBA")


class ISyntaxImageWrapper:
    """OpenSlide-compatible wrapper for Philips iSyntax files using FastSlide."""

    def __init__(self, isyntax_path):
        self.path = isyntax_path
        self._slide = None
        try:
            fastslide = _import_fastslide()
            self._slide = fastslide.FastSlide.from_file_path(self.path)
            self._init_metadata()
        except Exception as e:
            logger.error("Error opening iSyntax file %s: %s", isyntax_path, e)
            self.dimensions = (100, 100)
            self.level_count = 1
            self.level_dimensions = [(100, 100)]
            self.level_downsamples = [1.0]
            self.properties = {
                "vendor": "ISyntaxImageWrapper",
                "dimensions": "100x100",
                "level_count": "1",
                "error": str(e),
            }
            self.associated_images = {}
            raise

    def close(self):
        """Explicitly close the FastSlide reader."""
        if self._slide is not None:
            try:
                self._slide.close()
            except Exception as e:
                logger.warning("Error closing iSyntax reader: %s", e)
            finally:
                self._slide = None

    def __del__(self):
        try:
            if self._slide is not None:
                self.close()
        except Exception:
            pass

    def __getattr__(self, name):
        if not hasattr(self, "_slide") or self._slide is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'. "
                "The slide may have been closed."
            )
        return getattr(self._slide, name)

    def _init_metadata(self):
        if self._slide is None:
            raise RuntimeError("iSyntax reader not initialized")

        self.dimensions = tuple(self._slide.dimensions)
        self.level_count = self._slide.level_count
        self.level_dimensions = [tuple(d) for d in self._slide.level_dimensions]
        self.level_downsamples = list(self._slide.level_downsamples)

        self.properties = {
            str(key): str(value)
            for key, value in dict(self._slide.properties).items()
        }
        self.properties.update(
            {
                "vendor": "ISyntaxImageWrapper",
                "dimensions": f"{self.dimensions[0]}x{self.dimensions[1]}",
                "level_count": str(self.level_count),
                "format": getattr(self._slide, "format", "ISYNTAX"),
            }
        )

        self.associated_images = {}
        try:
            for name in self._slide.associated_images.keys():
                self.associated_images[name] = _fastslide_image_to_pil(
                    self._slide.associated_images[name]
                )
        except Exception as e:
            logger.warning("Could not load iSyntax associated images: %s", e)
            self.associated_images = {}

    def get_best_level_for_downsample(self, downsample):
        if self._slide is None:
            raise RuntimeError("iSyntax reader not initialized")
        return self._slide.get_best_level_for_downsample(downsample)

    def read_region(self, location, level, size, as_array=False):
        """Read a region using OpenSlide semantics.

        ``location`` is in level-0 coordinates; ``size`` is in level-native
        pixels at the requested pyramid level. Returns RGBA uint8 data.
        """
        if self._slide is None:
            raise RuntimeError("iSyntax reader not initialized")

        if level >= self.level_count:
            level = self.level_count - 1

        x, y = location
        w, h = max(0, int(size[0])), max(0, int(size[1]))
        if w == 0 or h == 0:
            empty = np.zeros((max(h, 0), max(w, 0), 4), dtype=np.uint8)
            return empty if as_array else Image.fromarray(empty, mode="RGBA")

        native_x, native_y = self._slide.convert_level0_to_level_native(
            int(x), int(y), level=level
        )

        level_w, level_h = self.level_dimensions[level]
        native_x = max(0, min(native_x, level_w - 1))
        native_y = max(0, min(native_y, level_h - 1))
        read_w = min(w, level_w - native_x)
        read_h = min(h, level_h - native_y)

        if read_w <= 0 or read_h <= 0:
            empty = np.zeros((h, w, 4), dtype=np.uint8)
            return empty if as_array else Image.fromarray(empty, mode="RGBA")

        image = self._slide.read_region(
            location=(native_x, native_y),
            level=level,
            size=(read_w, read_h),
        )
        img_array = _to_rgba_uint8(image.to_interleaved().numpy())

        if read_w < w or read_h < h:
            padded = np.zeros((h, w, 4), dtype=np.uint8)
            padded[:read_h, :read_w] = img_array
            img_array = padded

        if as_array:
            return img_array
        return Image.fromarray(img_array, mode="RGBA")
