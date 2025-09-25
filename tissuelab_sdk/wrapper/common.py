import tifffile
import tiffslide
from PIL import Image
import pydicom
import numpy as np
import sys
import threading
import nibabel as nib
from isyntax import ISyntax
# Add czifile import for thumbnail reading
import czifile
import io
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

    def __del__(self):
        try:
            if hasattr(self, '_tiff') and self._tiff is not None:
                self._tiff.close()
                self._tiff = None
        except Exception:
            # Suppress errors during cleanup
            pass

    def _init_levels(self):
        """init levels, use tiff page as level"""
        # get all page dimensions
        self.level_dimensions = []
        for page in self._tiff.pages:
            # TIFF shape is (height, width, channels), we only need (width, height)
            if len(page.shape) == 3:
                height, width, _ = page.shape
            else:
                height, width = page.shape
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

    def read_region(self, location, level, size, as_array=False):
        """read region from specified level

        Args:
            location: (x, y) start position (based on level 0 coordinates)
            level: level
            size: (width, height) size to read
        """
        if level >= self.level_count:
            raise ValueError(
                f"Invalid level {level}. Max level is {self.level_count-1}")

        # calculate actual coordinates in current level
        scale_factor = self.dimensions[0] / self.level_dimensions[level][0]
        x, y = location
        scaled_x = int(x / scale_factor)
        scaled_y = int(y / scale_factor)

        # read region from corresponding page
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
    
    def __del__(self):
        try:
            if hasattr(self, '_image') and self._image is not None:
                self._image.close()
                self._image = None
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

    def __del__(self):
        try:
            if hasattr(self, '_ds') and self._ds is not None:
                self._ds = None
            if hasattr(self, '_image') and self._image is not None:
                self._image = None
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
    
    def __del__(self):
        try:
            if hasattr(self, '_nifti') and self._nifti is not None:
                self._nifti = None
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