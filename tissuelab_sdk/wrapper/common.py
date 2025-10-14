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

    def _init_levels(self):
        """init levels, use tiff page as level"""
        # get all page dimensions
        self.level_dimensions = []
        
        # Try using series if available (better for z-stack and complex TIFF files)
        if hasattr(self._tiff, 'series') and len(self._tiff.series) > 0:
            # Use the first main series
            main_series = self._tiff.series[0]
            
            # For z-stack files, we want to use the first z-layer as the base
            # series.shape might be (z, height, width, channels) or (height, width, channels)
            if hasattr(main_series, 'shape'):
                shape = main_series.shape
                
                # Handle different series dimensions
                if len(shape) == 4:  # (z, height, width, channels) - z-stack
                    # Use first z-layer dimensions
                    _, height, width, _ = shape
                    self.level_dimensions.append((width, height))
                elif len(shape) == 3:  # (height, width, channels) - normal
                    height, width, _ = shape
                    self.level_dimensions.append((width, height))
                elif len(shape) == 2:  # (height, width) - grayscale
                    height, width = shape
                    self.level_dimensions.append((width, height))
            
            # Add additional pyramid levels from other series
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

        # first level dimensions as main dimensions
        self.dimensions = self.level_dimensions[0]
        self.level_count = len(self.level_dimensions)

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

    def get_thumbnail(self, size):
        """return thumbnail, use the smallest page"""
        img = self._tiff.pages[-1].asarray()  # use the smallest page
        pil_img = Image.fromarray(img)
        return pil_img.thumbnail(size, Image.Resampling.LANCZOS)

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

        # calculate actual coordinates in current level
        scale_factor = self.dimensions[0] / self.level_dimensions[level][0]
        x, y = location
        scaled_x = int(x / scale_factor)
        scaled_y = int(y / scale_factor)

        # read region from corresponding page
        # Check if this is a z-stack file using series
        if hasattr(self._tiff, 'series') and len(self._tiff.series) > 0:
            main_series = self._tiff.series[0]
            if hasattr(main_series, 'shape') and len(main_series.shape) == 4:
                # This is a z-stack file (z, height, width, channels)
                try:
                    # Use series to read with z-layer
                    # Get the series data with zarr-like access
                    img_full = main_series.asarray()  # Shape: (z, height, width, channels)
                    
                    # Ensure z_layer is within bounds
                    z_count = img_full.shape[0]
                    if z_layer >= z_count:
                        z_layer = 0
                    
                    # Extract the specific z-layer
                    img = img_full[z_layer]  # Shape: (height, width, channels)
                    
                    # Apply downsampling if level > 0
                    if level > 0:
                        downsample_factor = 2 ** level
                        from skimage.transform import downscale_local_mean
                        # Downsample spatial dimensions only
                        img = downscale_local_mean(img, (downsample_factor, downsample_factor, 1))
                        img = img.astype(np.uint8)
                    
                    # Extract region
                    region = img[scaled_y:scaled_y + size[1], scaled_x:scaled_x + size[0]]
                    
                except MemoryError as e:
                    # Memory error - don't fallback, raise clear error
                    raise MemoryError(
                        f"Insufficient memory to load z-stack file."
                        f"File dimensions: {main_series.shape}, "
                        f"requires approximately {np.prod(main_series.shape) / 1024**3:.1f} GB of memory."
                        f"Please use a smaller file or increase system memory."
                    ) from e
                except Exception as e:
                    # Other errors - also raise clear error for z-stack
                    raise RuntimeError(
                        f"Error reading layer {z_layer} of z-stack: {str(e)}"
                    ) from e
            else:
                # Not a z-stack, use page-based reading
                img = self._tiff.pages[level].asarray()
                region = img[scaled_y:scaled_y + size[1], scaled_x:scaled_x + size[0]]
        else:
            # Fallback to page-based reading
            img = self._tiff.pages[level].asarray()
            region = img[scaled_y:scaled_y + size[1], scaled_x:scaled_x + size[0]]

        if as_array:
            return region
        return Image.fromarray(region)


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
        print(f"NiftiImageWrapper: Global min={self._global_min}, Global max={self._global_max}")
        
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
                print(f"NiftiImageWrapper: 3D data, using middle slice z={z_mid}, shape={original_height}x{original_width}")
            else:
                original_height, original_width = shape
                print(f"NiftiImageWrapper: 2D data, shape={original_height}x{original_width}")
            
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
            print(f"NiftiImageWrapper: Created {self.level_count} pyramid levels")
        else:
            # Fallback for unusual shapes
            self.dimensions = (100, 100)
            self.level_dimensions = [(100, 100)]
            self.level_count = 1
            print(f"NiftiImageWrapper: Abnormal data shape, using default size 100x100")

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