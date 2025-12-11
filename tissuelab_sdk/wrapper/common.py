import tifffile
import tiffslide
from PIL import Image
import pydicom
import numpy as np
import sys
import threading
import nibabel as nib
import io
import logging

# Create logger for wrappers
logger = logging.getLogger(__name__)

TILE_SIZE = 1024

class TiffSlideWrapper:
    """
    Wrapper for tiffslide.TiffSlide
    Handles resource management for tiffslide.TiffSlide.

    This wrapper can be used as a context manager.
    """

    def __init__(self, tiff_path):
        self.path = tiff_path
        self.slide = tiffslide.open_slide(tiff_path)

    def close(self):
        """Explicitly close the slide."""
        # Check for 'slide' attribute existence for safety
        if hasattr(self, "slide") and self.slide:
            self.slide.close()
            self.slide = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Suppress all errors during cleanup to prevent crashes
            pass

    def __getattr__(self, name):
        # delegate all other methods to the slide object
        if not hasattr(self, "slide") or self.slide is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'. "
                "The slide may have been closed."
            )
        return getattr(self.slide, name)

class TiffFileWrapper:

    def __init__(self, tiff_path):
        self.path = tiff_path
        self._tiff = tifffile.TiffFile(tiff_path)
        self._zarr_arrays = {}  # Cache for zarr arrays (non-zstack files)
        self._init_levels()
        self._init_properties()

    def close(self):
        """Explicitly close the TIFF file."""
        if hasattr(self, '_tiff') and self._tiff is not None:
            self._tiff.close()
            self._tiff = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Suppress errors during cleanup
            pass

    def _pick_largest_series(self):
        """Select the series with the largest resolution (most pixels)"""
        if not hasattr(self._tiff, 'series') or len(self._tiff.series) == 0:
            return None
        
        best_series = self._tiff.series[0]
        best_pixels = 0
        
        for series in self._tiff.series:
            if not hasattr(series, 'shape') or not hasattr(series, 'axes'):
                continue
            
            axes = series.axes
            shape = series.shape
            
            # Extract Y and X dimensions
            y_size = shape[axes.index('Y')] if 'Y' in axes else 0
            x_size = shape[axes.index('X')] if 'X' in axes else 0
            pixels = y_size * x_size
            
            if pixels > best_pixels:
                best_pixels = pixels
                best_series = series
        
        return best_series

    def _init_levels(self):
        """init levels, use tiff page as level"""
        # get all page dimensions
        self.level_dimensions = []
        self._is_zstack = False
        self._z_layer_count = 1
        
        # Try using series if available (better for z-stack and complex TIFF files)
        if hasattr(self._tiff, 'series') and len(self._tiff.series) > 0:
            # Select the series with largest resolution (most important for NDPI)
            main_series = self._pick_largest_series()
            if main_series is None:
                main_series = self._tiff.series[0]
            
            # Store main series for later use in read_region
            self._main_series = main_series
            
            # For z-stack files, we want to use the first z-layer as the base
            # series.shape might be (z, height, width, channels) or (height, width, channels)
            if hasattr(main_series, 'shape'):
                shape = main_series.shape
                
                # Handle different series dimensions
                if len(shape) == 4:  # (z, height, width, channels) - z-stack
                    # Use first z-layer dimensions
                    self._is_zstack = True
                    self._z_layer_count = shape[0]
                    
                    # Method 1: Try using series.levels (most reliable for NDPI)
                    if hasattr(main_series, 'levels') and len(main_series.levels) > 0:
                        for level_idx, level_series in enumerate(main_series.levels):
                            if hasattr(level_series, 'shape') and hasattr(level_series, 'axes'):
                                level_shape = level_series.shape
                                level_axes = level_series.axes
                                
                                # Extract X and Y dimensions from this level
                                if 'Y' in level_axes and 'X' in level_axes:
                                    y_idx = level_axes.index('Y')
                                    x_idx = level_axes.index('X')
                                    height = level_shape[y_idx]
                                    width = level_shape[x_idx]
                                    self.level_dimensions.append((width, height))
                        
                    # Method 2: Fallback to page-based detection if series.levels not available
                    if not self.level_dimensions:
                        # Extract pyramid levels from first z-layer by detecting dimension pattern
                        total_pages = len(self._tiff.pages)
                        first_page_dims = None
                        for idx in range(total_pages):
                            try:
                                page = self._tiff.pages[idx]
                                if len(page.shape) == 3:
                                    h, w, _ = page.shape
                                elif len(page.shape) == 2:
                                    h, w = page.shape
                                else:
                                    continue
                                
                                # Check if we've reached the next z-layer (dimensions repeat)
                                if first_page_dims is None:
                                    first_page_dims = (w, h)
                                elif (w, h) == first_page_dims and len(self.level_dimensions) > 0:
                                    # We've looped back to the same dimensions, this is the start of z-layer 1
                                    break
                                
                                self.level_dimensions.append((w, h))
                            except Exception as e:
                                logger.warning(f"[TiffFileWrapper] Could not read page {idx}: {e}")
                                break
                    
                elif len(shape) == 3:  # (height, width, channels) - normal
                    height, width, _ = shape
                    self.level_dimensions.append((width, height))
                elif len(shape) == 2:  # (height, width) - grayscale
                    height, width = shape
                    self.level_dimensions.append((width, height))
            
            # Add additional pyramid levels from other series (for non-zstack)
            if not self._is_zstack:
                for series_idx, series in enumerate(self._tiff.series[1:], 1):
                    if hasattr(series, 'shape'):
                        shape = series.shape
                        if len(shape) >= 2:
                            # Take last two dimensions as height, width
                            height, width = shape[-3:-1] if len(shape) >= 3 else shape[-2:]
                            self.level_dimensions.append((width, height))
        
        # Fallback to page-based if series didn't work or is empty
        if not self.level_dimensions:
            for page in self._tiff.pages:
                # TIFF shape is (height, width, channels), we only need (width, height)
                if len(page.shape) == 3:
                    height, width, _ = page.shape
                elif len(page.shape) == 2:
                    height, width = page.shape
                else:
                    continue
                self.level_dimensions.append((width, height))
        
        # Ensure _main_series is always set
        if not hasattr(self, '_main_series'):
            if hasattr(self._tiff, 'series') and len(self._tiff.series) > 0:
                self._main_series = self._tiff.series[0]
            else:
                self._main_series = None

        # first level dimensions as main dimensions
        self.dimensions = self.level_dimensions[0]
        self.level_count = len(self.level_dimensions)
        
        # Calculate level_downsamples (REQUIRED by frontend for coordinate mapping)
        # downsample = how many base-level pixels per level pixel
        self.level_downsamples = []
        base_width, base_height = self.level_dimensions[0]
        for level_width, level_height in self.level_dimensions:
            # Use width for downsample calculation (should be same as height ratio)
            downsample = base_width / level_width
            self.level_downsamples.append(downsample)
        
        # Expose z-stack info as public properties (REQUIRED by frontend)
        self.is_zstack = self._is_zstack
        self.z_layer_count = self._z_layer_count
        

    def _init_properties(self):
        """init properties, extract useful metadata from TIFF tags"""
        self.properties = {}
        self.fit_page = 3

        # get basic properties from first page
        first_page = self._tiff.pages[0]
        tags = first_page.tags

        # add basic properties
        self.properties.update({
            'vendor':
            'tifffile',
            'level_count':
            str(self.level_count),
            'dimensions':
            f'{self.dimensions[0]}x{self.dimensions[1]}',
            'dtype':
            str(first_page.dtype),
            'channels':
            str(first_page.shape[2] if len(first_page.shape) == 3 else 1),
            'is_zstack':
            str(self.is_zstack),
            'z_layer_count':
            str(self.z_layer_count),
        })

        # add useful TIFF tags
        tag_mapping = {
            'ImageWidth': 'width',
            'ImageLength': 'height',
            'BitsPerSample': 'bits_per_sample',
            'Compression': 'compression',
            'PhotometricInterpretation': 'photometric',
            'SamplesPerPixel': 'samples_per_pixel',
            'Software': 'software',
            'DateTime': 'datetime',
            'Artist': 'artist',
            'HostComputer': 'host_computer',
        }

        for tag_name, prop_name in tag_mapping.items():
            if tag_name in tags:
                tag_value = tags[tag_name].value
                self.properties[prop_name] = str(tag_value)

        # add dimensions for each level
        for i, dims in enumerate(self.level_dimensions):
            if dims[0] > TILE_SIZE and dims[
                    1] > TILE_SIZE and i > self.fit_page:
                self.fit_page = i
            self.properties[f'level_{i}_dimensions'] = f'{dims[0]}x{dims[1]}'

        # Extract MPP (microns per pixel) from TIFF resolution tags
        # This provides compatibility with code that expects 'tiffslide.mpp-x'
        try:
            xres_tag = tags.get('XResolution')
            yres_tag = tags.get('YResolution')
            unit_tag = tags.get('ResolutionUnit')
            
            if xres_tag is not None:
                xres = xres_tag.value
                # XResolution is typically a tuple/fraction (numerator, denominator)
                if isinstance(xres, tuple) and len(xres) == 2:
                    pixels_per_unit = xres[0] / xres[1] if xres[1] != 0 else xres[0]
                else:
                    pixels_per_unit = float(xres)
                
                # Convert to microns based on ResolutionUnit
                # RESUNIT: 1=None, 2=inch, 3=centimeter
                unit_val = unit_tag.value if unit_tag else 1
                # Handle enum types
                if hasattr(unit_val, 'value'):
                    unit_val = unit_val.value
                elif hasattr(unit_val, 'name'):
                    unit_val = 3 if 'CENTIMETER' in str(unit_val).upper() else (2 if 'INCH' in str(unit_val).upper() else 1)
                
                mpp = None
                if unit_val == 3:  # centimeters
                    mpp = 10000.0 / pixels_per_unit  # cm to microns
                elif unit_val == 2:  # inches
                    mpp = 25400.0 / pixels_per_unit  # inches to microns
                
                # Only set if it's a reasonable value for microscopy (0.05 - 10.0 µm/px)
                # Covers 100x oil immersion (~0.1) down to 2x objective (~5.0)
                if mpp is not None and 0.05 < mpp < 10.0:
                    self.properties['tiffslide.mpp-x'] = mpp
                    self.properties['tiffslide.mpp-y'] = mpp  # Assume square pixels
                    
                    # Also compute YResolution if available
                    if yres_tag is not None:
                        yres = yres_tag.value
                        if isinstance(yres, tuple) and len(yres) == 2:
                            pixels_per_unit_y = yres[0] / yres[1] if yres[1] != 0 else yres[0]
                        else:
                            pixels_per_unit_y = float(yres)
                        
                        if unit_val == 3:
                            mpp_y = 10000.0 / pixels_per_unit_y
                        elif unit_val == 2:
                            mpp_y = 25400.0 / pixels_per_unit_y
                        else:
                            mpp_y = mpp
                            
                        if 0.05 < mpp_y < 10.0:
                            self.properties['tiffslide.mpp-y'] = mpp_y
        except Exception as e:
            # MPP extraction failed, properties won't have tiffslide.mpp-x
            logger.debug(f"Could not extract MPP from TIFF tags: {e}")

    def get_thumbnail(self, size):
        """return thumbnail, use the smallest page"""
        img = self._tiff.pages[-1].asarray()  # use the smallest page
        pil_img = Image.fromarray(img)
        return pil_img.thumbnail(size, Image.Resampling.LANCZOS)

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Find the best pyramid level for a given downsample factor.
        
        Returns the level with the largest downsample <= the requested value,
        matching OpenSlide/TiffSlide behavior.
        
        Args:
            downsample: The desired downsample factor (e.g., 4.0 means 4x smaller)
            
        Returns:
            The index of the best level to use
        """
        best_level = 0
        for i, ds in enumerate(self.level_downsamples):
            if ds <= downsample:
                best_level = i
        return best_level

    def read_region(self, location, level, size, as_array=False, z_layer=None):
        """read region from specified level

        Args:
            location: (x, y) start position (based on level 0 coordinates)
            level: level
            size: (width, height) size to read
            as_array: if True, return numpy array instead of PIL Image
            z_layer: z-layer index (for z-stack files), defaults to 0
        """
        if level >= self.level_count:
            raise ValueError(
                f"Invalid level {level}. Max level is {self.level_count-1}")

        # Default z_layer to 0 if not specified
        if z_layer is None:
            z_layer = 0

        # Validate and clamp coordinates to prevent out-of-bounds access
        x, y = location
        W0, H0 = self.dimensions
        
        # Coordinate overshoot protection: reject coordinates far beyond valid range
        OVERSHOOT_LIMIT = 10.0
        if x > W0 * OVERSHOOT_LIMIT or y > H0 * OVERSHOOT_LIMIT:
            logger.error(f"[COORD] Location ({x:.0f}, {y:.0f}) far exceeds Level 0 bounds ({W0}, {H0}) by {x/W0:.1f}x, {y/H0:.1f}x")
            logger.error(f"[COORD] This is likely a frontend coordinate bug. Returning black tile.")
            # Return black tile with appropriate format based on as_array parameter
            black_tile = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            if as_array:
                return black_tile
            else:
                return Image.fromarray(black_tile)
        
        # Clamp to Level 0 bounds
        x = max(0, min(x, W0 - size[0]))
        y = max(0, min(y, H0 - size[1]))

        # calculate actual coordinates in current level
        scale_factor = self.dimensions[0] / self.level_dimensions[level][0]
        scaled_x = int(x / scale_factor)
        scaled_y = int(y / scale_factor)

        # Initialize region variable to avoid UnboundLocalError in exception paths
        region = None

        # read region from corresponding page
        # Check if this is a z-stack file using series
        if hasattr(self, '_main_series') and self._main_series is not None:
            main_series = self._main_series  # Use the largest resolution series we selected
            if hasattr(main_series, 'shape') and len(main_series.shape) == 4:
                # This is a z-stack file with Z dimension
                # Use series.levels[L].aszarr() for proper Z-stack reading (CORRECT approach)
                try:
                    import zarr
                    
                    # Check if series has levels attribute
                    if not hasattr(main_series, 'levels') or level >= len(main_series.levels):
                        raise ValueError(f"Series does not have level {level} (has {len(main_series.levels) if hasattr(main_series, 'levels') else 0} levels)")
                    
                    # Get the specific pyramid level
                    level_series = main_series.levels[level]
                    
                    # IMPORTANT: Use level_series.axes, not main_series.axes!
                    # Level 0 has axes='ZYXS', Level 1+ have axes='IYXS'
                    axes = level_series.axes if hasattr(level_series, 'axes') else 'ZYXS'
                    logger.debug(f"[Z-Stack] Level {level} series - shape: {level_series.shape if hasattr(level_series, 'shape') else 'unknown'}, axes: {axes}")
                    
                    # Get zarr array from this level (this properly handles Z dimension)
                    store = level_series.aszarr()
                    z_obj = zarr.open(store, mode='r')
                    logger.debug(f"[Z-Stack] Zarr object type: {type(z_obj)}, keys: {list(z_obj.keys()) if hasattr(z_obj, 'keys') else 'N/A'}")
                    
                    # If it's a Group, we need to get the actual array
                    if isinstance(z_obj, zarr.hierarchy.Group):
                        # For NDPI z-stack Level 0, the Group contains ALL pyramid levels as keys: '0', '1', '2', etc.
                        # We need to use the key that matches the current level
                        key = str(level)
                        if key in z_obj.keys():
                            z_array = z_obj[key]
                            logger.debug(f"[Z-Stack] Using group key '{key}' for level {level}, array shape: {z_array.shape}, dtype: {z_array.dtype}")
                        else:
                            # Fallback: try first key
                            keys = list(z_obj.keys())
                            if len(keys) > 0:
                                key = keys[0]
                                z_array = z_obj[key]
                                logger.warning(f"[Z-Stack] Level {level} not in group keys {keys}, using first key '{key}'")
                            else:
                                raise ValueError(f"Zarr Group is empty, no arrays found")
                        
                        # Note: axes from group array should match level_series.axes (already set above)
                        # Keep using the axes we got from level_series
                        logger.debug(f"[Z-Stack] Group array selected, using axes: {axes}")
                    else:
                        # Directly got an Array (typical for Level 1+)
                        z_array = z_obj
                        logger.debug(f"[Z-Stack] Direct array - shape: {z_array.shape}, dtype: {z_array.dtype}, axes: {axes}")
                    
                    # Build slicer: select specific Z layer + region
                    # Find Z/I, Y, X axes positions
                    # Note: 'Z' is used for Level 0, 'I' (image index) is used for Level 1+ in NDPI z-stacks
                    z_idx = None
                    if 'Z' in axes:
                        z_idx = axes.index('Z')
                        logger.debug(f"[Z-Stack] Found 'Z' axis at index {z_idx}")
                    elif 'I' in axes:
                        # Only use 'I' as Z if it has multiple planes (length > 1)
                        i_idx = axes.index('I')
                        i_length = z_array.shape[i_idx]
                        if i_length > 1:
                            z_idx = i_idx
                            logger.debug(f"[Z-Stack] Using 'I' axis (length={i_length}) as Z at index {z_idx}")
                        else:
                            logger.warning(f"[Z-Stack] 'I' axis found but length={i_length}, not using as Z")
                    
                    if z_idx is None:
                        raise ValueError(f"No valid Z or I axis found in axes: {axes}, shape: {z_array.shape}")
                    
                    y_idx = axes.index('Y') if 'Y' in axes else None
                    x_idx = axes.index('X') if 'X' in axes else None
                    
                    logger.debug(f"[Z-Stack] Axes indices: z={z_idx}, y={y_idx}, x={x_idx}")
                    
                    # Get the level dimensions for clamping
                    level_dims = self.level_dimensions[level]  # (width, height)
                    level_h, level_w = level_dims[1], level_dims[0]
                    
                    # Check if coordinates are completely out of bounds
                    if scaled_x >= level_w or scaled_y >= level_h:
                        logger.warning(f"[Z-Stack] Coordinates out of bounds: scaled=({scaled_x}, {scaled_y}), level_dims=({level_w}, {level_h})")
                        # Return black tile immediately for out-of-bounds coordinates
                        region = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                    else:
                        # Clamp coordinates to valid range at this level
                        # IMPORTANT: Ensure y_start < level_h and x_start < level_w
                        y_start = max(0, min(scaled_y, level_h - 1))
                        y_end = max(y_start + 1, min(scaled_y + size[1], level_h))  # Ensure at least 1px height
                        x_start = max(0, min(scaled_x, level_w - 1))
                        x_end = max(x_start + 1, min(scaled_x + size[0], level_w))  # Ensure at least 1px width
                        
                        # Build slicer for all dimensions
                        slicer = [slice(None)] * len(axes)
                        slicer[z_idx] = z_layer  # Select specific Z plane
                        if y_idx is not None:
                            slicer[y_idx] = slice(y_start, y_end)
                        if x_idx is not None:
                            slicer[x_idx] = slice(x_start, x_end)
                        
                        region = np.asarray(z_array[tuple(slicer)])
                    
                    # Ensure region is 3D (H, W, C) - only if region was successfully read
                    if region is not None:
                        if region.ndim == 2:
                            region = region[:, :, np.newaxis]
                        elif region.ndim > 3:
                            # Squeeze extra dimensions but keep H, W, C
                            while region.ndim > 3 and region.shape[0] == 1:
                                region = region[0]
                            if region.ndim == 2:
                                region = region[:, :, np.newaxis]
                    
                except Exception as e:
                    logger.error(f"[Z-Stack] Failed to read z_layer={z_layer}, level={level}: {e}")
                    logger.exception(e)
                    region = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            else:
                # Not a z-stack, use zarr if available
                page = self._tiff.pages[level]
                if page.is_tiled:
                    try:
                        if level not in self._zarr_arrays:
                            import zarr
                            store = page.aszarr()
                            self._zarr_arrays[level] = zarr.open(store, mode='r')
                        
                        z_array = self._zarr_arrays[level]
                        region = np.asarray(z_array[scaled_y:scaled_y + size[1], scaled_x:scaled_x + size[0], :])
                    except Exception:
                        img = page.asarray()
                        region = img[scaled_y:scaled_y + size[1], scaled_x:scaled_x + size[0]]
                else:
                    img = page.asarray()
                    region = img[scaled_y:scaled_y + size[1], scaled_x:scaled_x + size[0]]
        else:
            # Fallback to page-based reading
            page = self._tiff.pages[level]
            if page.is_tiled:
                try:
                    if level not in self._zarr_arrays:
                        import zarr
                        store = page.aszarr()
                        self._zarr_arrays[level] = zarr.open(store, mode='r')
                    
                    z_array = self._zarr_arrays[level]
                    region = np.asarray(z_array[scaled_y:scaled_y + size[1], scaled_x:scaled_x + size[0], :])
                except Exception:
                    img = page.asarray()
                    region = img[scaled_y:scaled_y + size[1], scaled_x:scaled_x + size[0]]
            else:
                img = page.asarray()
                region = img[scaled_y:scaled_y + size[1], scaled_x:scaled_x + size[0]]

        # Final safety check: ensure region was successfully read
        if region is None:
            logger.error(f"[read_region] region is None after all attempts, returning black tile")
            region = np.zeros((size[1], size[0], 3), dtype=np.uint8)

        if as_array:
            return region
        return Image.fromarray(region)
    
    def diagnose_zstack_structure(self):
        """Diagnostic function to understand z-stack file structure"""
        if not self._is_zstack:
            logger.info("[Z-Stack Diagnosis] Not a z-stack file")
            return
        
        logger.info(f"[Z-Stack Diagnosis] ==================== NDPI Z-Stack Structure ====================")
        logger.info(f"[Z-Stack Diagnosis] Z layers: {self._z_layer_count}, Pyramid levels: {self.level_count}")
        logger.info(f"[Z-Stack Diagnosis] Total pages in file: {len(self._tiff.pages)}")
        logger.info(f"[Z-Stack Diagnosis] Expected pages (z_layers * levels): {self._z_layer_count * self.level_count}")
        
        # Sample first few pages to understand structure
        logger.info(f"[Z-Stack Diagnosis] Sampling first 10 pages:")
        for idx in range(min(10, len(self._tiff.pages))):
            try:
                page = self._tiff.pages[idx]
                logger.info(f"[Z-Stack Diagnosis]   Page {idx}: shape={page.shape}, is_tiled={page.is_tiled}")
            except Exception as e:
                logger.warning(f"[Z-Stack Diagnosis]   Page {idx}: ERROR - {e}")
        logger.info(f"[Z-Stack Diagnosis] ================================================================")


class SimpleImageWrapper:
    """Wrapper for simple image files (JPEG, PNG) to mimic WSI interface"""

    def __init__(self, image_path):
        self.path = image_path
        self._image = Image.open(image_path)
        self._init_levels()
        self._init_properties()
        self._lock = threading.Lock()
    
    def close(self):
        """Explicitly close the image."""
        if hasattr(self, '_image') and self._image is not None:
            self._image.close()
            self._image = None
    
    def __del__(self):
        try:
            self.close()
        except Exception:
            # Suppress errors during cleanup
            pass

    def _init_levels(self):
        """Initialize pyramid levels for the image"""
        original_width, original_height = self._image.size
        self.dimensions = (original_width, original_height)

        # Create pyramid levels
        self.level_dimensions = []
        width, height = original_width, original_height
        self.level_dimensions.append((width, height))
        self.level_count = len(self.level_dimensions)  # 1

    def _init_properties(self):
        """Initialize image properties"""
        self.properties = {
            'vendor': 'SimpleImageWrapper',
            'level_count': str(self.level_count),
            'dimensions': f'{self.dimensions[0]}x{self.dimensions[1]}',
            'format': self._image.format,
            'mode': self._image.mode
        }

    def read_region(self, location, level, size, as_array=False):
        """Read a region from the image at the specified level"""
        with self._lock:
            # Calculate scale factor for the requested level
            scale_factor = self.dimensions[0] / self.level_dimensions[level][0]

            # Calculate the region in the original image
            x, y = location
            scaled_x = int(x / scale_factor)
            scaled_y = int(y / scale_factor)
            scaled_width = int(size[0] / scale_factor)
            scaled_height = int(size[1] / scale_factor)

            # Extract the region from the original image

            region = self._image.crop((scaled_x, scaled_y, scaled_x + scaled_width,
                                       scaled_y + scaled_height))

            # Convert to RGB if necessary
            if region.mode != 'RGB':
                region = region.convert('RGB')

            if as_array:
                return np.array(region)
            return region


class DicomImageWrapper:
    """Wrapper for DICOM image files to mimic WSI interface"""

    def __init__(self, dicom_path):
        self.path = dicom_path
        self._ds = pydicom.dcmread(dicom_path)
        self._image = Image.fromarray(self._ds.pixel_array)
        self._init_levels()
        self._init_properties()

    def close(self):
        """Explicitly close the DICOM image."""
        if hasattr(self, '_ds') and self._ds is not None:
            self._ds = None
        if hasattr(self, '_image') and self._image is not None:
            self._image = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Suppress errors during cleanup
            pass

    def _init_levels(self):
        original_width, original_height = self._image.size
        self.dimensions = (original_width, original_height)
        self.level_dimensions = [(original_width, original_height)]
        self.level_count = len(self.level_dimensions)

    def _init_properties(self):
        self.properties = {
            'vendor':
            'DicomImageWrapper',
            'level_count':
            str(self.level_count),
            'dimensions':
            f'{self.dimensions[0]}x{self.dimensions[1]}',
            'PhotometricInterpretation':
            self._ds.get('PhotometricInterpretation', 'Unknown'),
            'Modality':
            self._ds.get('Modality', 'Unknown')
        }

    def read_region(self, location, level, size, as_array=False):
        scale_factor = self.dimensions[0] / self.level_dimensions[level][0]
        x, y = location
        scaled_x = int(x / scale_factor)
        scaled_y = int(y / scale_factor)
        scaled_width = int(size[0] / scale_factor)
        scaled_height = int(size[1] / scale_factor)
        region = self._image.crop((scaled_x, scaled_y, scaled_x + scaled_width,
                                   scaled_y + scaled_height))
        if region.mode != 'RGB':
            region = region.convert('RGB')

        if as_array:
            return np.array(region)
        return region





class NiftiImageWrapper:
    """Wrapper for NIfTI image files to mimic WSI interface"""

    def __init__(self, nifti_path):
        self.path = nifti_path
        self._nifti = nib.load(nifti_path)
        self._data = self._nifti.get_fdata()
        self.header = self._nifti.header
        self.zooms = self.header.get_zooms()

        # Calculate global min and max values for the entire dataset
        self._global_min = self._data.min()
        self._global_max = self._data.max()
        
        self._init_levels()
        self._init_properties()
    
    def close(self):
        """Explicitly close the NIfTI image."""
        if hasattr(self, '_nifti') and self._nifti is not None:
            self._nifti = None
    
    def __del__(self):
        try:
            self.close()
        except Exception:
            # Suppress errors during cleanup
            pass

    def _init_levels(self):
        """Initialize image pyramid levels"""
        # Get dimensions from NIfTI data
        # NIfTI data typically has (x, y, z, t) or (x, y, z) format
        shape = self._data.shape
        if len(shape) >= 2:
            # For 3D data, use the middle slice of z dimension
            if len(shape) >= 3:
                z_mid = shape[2] // 2
                original_height, original_width = shape[0], shape[1]
            else:
                original_height, original_width = shape
            
            self.dimensions = (original_width, original_height)

            # Create pyramid levels
            self.level_dimensions = [(original_width, original_height)]

            # Add additional downsampled levels
            width, height = original_width, original_height
            factor = 2
            while width // factor >= 64 and height // factor >= 64:
                level_width = width // factor
                level_height = height // factor
                self.level_dimensions.append((level_width, level_height))
                factor *= 2

            self.level_count = len(self.level_dimensions)
        else:
            # Fallback for unusual shapes
            self.dimensions = (100, 100)
            self.level_dimensions = [(100, 100)]
            self.level_count = 1

    def _init_properties(self):
        """Initialize image properties"""
        self.properties = {
            'vendor': 'NiftiImageWrapper',
            'level_count': str(self.level_count),
            'dimensions': f'{self.dimensions[0]}x{self.dimensions[1]}',
            'format': 'NIfTI',
            'shape': str(self._data.shape),
            'header': str(self.header),
            'global_min': str(self._global_min),
            'global_max': str(self._global_max)
        }

    def read_region(self, location, level, size, as_array=False):
        """Read image region from specified level"""
        # Calculate scale factor for requested level
        if level >= len(self.level_dimensions):
            level = len(self.level_dimensions) - 1

        scale_factor = self.dimensions[0] / self.level_dimensions[level][0]

        # Calculate region in original image
        x, y = location
        scaled_x = int(x / scale_factor)
        scaled_y = int(y / scale_factor)
        scaled_width = int(size[0])
        scaled_height = int(size[1])

        # Extract region from NIfTI data
        # For 3D data, use the middle slice of z dimension
        if len(self._data.shape) >= 3:
            z_mid = self._data.shape[2] // 2
            slice_data = self._data[:, :, z_mid]
        else:
            slice_data = self._data

        # Ensure boundaries are not exceeded
        max_y = min(scaled_y + scaled_height, slice_data.shape[0])
        max_x = min(scaled_x + scaled_width, slice_data.shape[1])

        # Extract region
        region_data = slice_data[scaled_y:max_y, scaled_x:max_x]

        # Use global min and max values for normalization to ensure color consistency
        if self._global_max > self._global_min:
            normalized = ((region_data - self._global_min) / 
                         (self._global_max - self._global_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros(region_data.shape, dtype=np.uint8)

        # Convert to PIL image
        if len(normalized.shape) == 2:  # Grayscale image
            img = Image.fromarray(normalized, 'L')
            img = img.convert('RGB')  # Convert to RGB for consistency
        else:  # Already RGB
            img = Image.fromarray(normalized)

        # Resize to requested size if needed
        current_width, current_height = img.size
        if current_width != scaled_width or current_height != scaled_height:
            img = img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
        
        if as_array:
            return np.array(img)
        return img