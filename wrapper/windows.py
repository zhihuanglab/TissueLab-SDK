from .common import *  # re-export shared wrappers
from PIL import Image
import numpy as np
from pylibCZIrw import czi
import threading
import time
import pythoncom
import czifile
import xml.etree.ElementTree as ET
try:
    from isyntax import ISyntax
except Exception:  # pragma: no cover
    ISyntax = None
class CziImageWrapper:

    def __init__(self, czi_path, max_levels=5):
        self.path = czi_path
        # Get pixel scale from CZI file
        self.mpp = self._get_czi_scale()
        if self.mpp is None:
            print('Warning: Unable to get mpp value from CZI file, using default value 0.25')
            self.mpp = 0.25
            self.magnification = 40  # Default magnification
        else:
            # Calculate magnification - at 10x, mpp is about 1.0 microns/pixel
            reference_mpp_10x = 1.0
            self.magnification = reference_mpp_10x / self.mpp * 10
            print(f'Calculated magnification from CZI: {self.magnification:.1f}x')
            
        self._init_metadata()
        self._init_levels(max_levels)
        self.lock = threading.Lock()
        self._com_initialized_thread = None
        self._com_init_logged = False

    def _get_czi_scale(self):
        """
        Extract scaling information (microns/pixel) from CZI file
        
        Returns:
            float: Microns per pixel value, returns None if extraction fails
        """
        try:
            # Open CZI file directly using czifile library
            with czifile.CziFile(self.path) as czi_reader:
                # Get metadata
                metadata = czi_reader.metadata()
                
                # Parse XML metadata
                metadata_root = ET.fromstring(metadata)
                
                # Try different possible metadata paths
                possible_paths = [
                    './/Scaling/Items/Distance[@Id="X"]/Value',
                    './/ImageScaling/ImagePixelSize/X',
                    './/ImageDocument/Metadata/Information/Image/PixelSize/X',
                    './/Image/PixelSize/X'
                ]
                
                for path in possible_paths:
                    element = metadata_root.find(path)
                    if element is not None:
                        # Convert from meters to microns (multiply by 10^6)
                        meters_per_pixel = float(element.text)
                        microns_per_pixel = meters_per_pixel * 1e6
                        print(f"Found pixel size from CZI metadata: {microns_per_pixel:.3f} microns/pixel")
                        return microns_per_pixel
                
                print("Pixel size information not found in CZI metadata")
                return None
        except Exception as e:
            print(f"Error reading CZI file: {str(e)}")
            return None

    def _init_metadata(self):
        with czi.open_czi(self.path) as reader:
            bounds = reader.total_bounding_box
            self.x_range = bounds['X']
            self.y_range = bounds['Y']
            self.dimensions = (self.x_range[1] - self.x_range[0],
                               self.y_range[1] - self.y_range[0])

    def _init_levels(self, max_levels):
        self.level_dimensions = []
        w, h = self.dimensions
        for i in range(max_levels):
            self.level_dimensions.append((int(w >> i), int(h >> i)))
        self.level_count = len(self.level_dimensions)
        self.properties = {
            'vendor': 'CziImageWrapper',
            'dimensions': f'{self.dimensions[0]}x{self.dimensions[1]}',
            'level_count': str(self.level_count),
        }

    def read_region(self, location, level, size, as_array=False):
        current_thread = threading.get_ident()
        
        with self.lock:
            # Check if we need to initialize COM for this thread
            if self._com_initialized_thread != current_thread:
                try:
                    pythoncom.CoInitialize()
                    self._com_initialized_thread = current_thread
                    # Only log this once per thread
                    if not self._com_init_logged:
                        self._com_init_logged = True
                except Exception as e:
                    print(f"COM initialization error: {e}")
            
            try:
                if level >= self.level_count:
                    raise ValueError(
                        f"Requested level {level} exceeds available levels {self.level_count}")

                downsample = 2**level
                x, y = location
                w, h = size

                roi_x = int(x + self.x_range[0])
                roi_y = int(y + self.y_range[0])
                roi_w = int(w * downsample)
                roi_h = int(h * downsample)
                zoom = 1.0 / downsample

                # Add more error handling and retries
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        with czi.open_czi(self.path) as reader:
                            img = reader.read(roi=(roi_x, roi_y, roi_w, roi_h),
                                            zoom=zoom,
                                            scene=0)
                        break
                    except Exception as e:
                        if retry < max_retries - 1:
                            print(f"CZI read error, retrying ({retry+1}/{max_retries}): {e}")
                            time.sleep(0.5)  # Short delay before retry
                        else:
                            print(f"roi_x: {roi_x}, roi_y: {roi_y}, roi_w: {roi_w}, roi_h: {roi_h}, zoom: {zoom}")
                            print(f"Failed to read CZI region after {max_retries} attempts: {e}")
                            # Return black image instead of raising exception
                            img = np.zeros((h, w, 3), dtype=np.uint8)

                # BGR to RGB, fill blank space with white
                if img is not None:
                    img = img[:, :, ::-1]
                    img[img == 0] = 255

                pil_img = Image.fromarray(img)
                return np.array(pil_img) if as_array else pil_img
            
            except Exception as e:
                print(f"Unexpected error in CZI read_region: {e}")
                # Return a blank white image in case of error
                blank_img = np.ones((h, w, 3), dtype=np.uint8) * 255
                pil_img = Image.fromarray(blank_img)
                return np.array(pil_img) if as_array else pil_img
            
            finally:
                # Don't uninitialize COM here - it could be needed for future calls
                pass


class ISyntaxImageWrapper:
    """Wrapper for ISyntax image files using WSI interface"""

    def __init__(self, isyntax_path):
        if ISyntax is None:
            raise ImportError("isyntax is not installed. Please install 'isyntax' to use ISyntaxImageWrapper on Windows.")
        self.path = isyntax_path
        self.isyntax_reader = None
        try:
            self.isyntax_reader = ISyntax.open(self.path)
            self._init_levels()
        except Exception as e:
            print(f"Error opening ISyntax file {isyntax_path}: {e}")
            # Set default values if initialization fails
            self.dimensions = (100, 100)
            self.level_count = 1
            self.level_dimensions = [(100, 100)]
            self.properties = {
                'vendor': 'ISyntaxImageWrapper',
                'dimensions': '100x100',
                'level_count': '1',
                'error': str(e)
            }
            raise
    
    def close(self):
        """Explicitly close the ISyntax reader."""
        if hasattr(self, 'isyntax_reader') and self.isyntax_reader is not None:
            try:
                self.isyntax_reader.close()
            except Exception as e:
                print(f"Error closing ISyntax reader: {e}")
            finally:
                self.isyntax_reader = None
    
    def __del__(self):
        """Safe destructor that handles missing attributes."""
        try:
            if hasattr(self, 'isyntax_reader') and self.isyntax_reader is not None:
                self.close()
        except Exception as e:
            # Suppress errors during cleanup
            pass

    def _init_levels(self):
        if self.isyntax_reader is None:
            raise RuntimeError("ISyntax reader not initialized")
        
        self.dimensions = self.isyntax_reader.dimensions
        self.level_count = self.isyntax_reader.level_count
        self.level_dimensions = self.isyntax_reader.level_dimensions

        self.properties = {
            'vendor': 'ISyntaxImageWrapper',
            'dimensions': f'{self.dimensions[0]}x{self.dimensions[1]}',
            'level_count': str(self.level_count),
        }

    def read_region(self, location, level, size, as_array=False):
        """Read a region from the image at the specified level, array type: RGBA"""
        if self.isyntax_reader is None:
            raise RuntimeError("ISyntax reader not initialized")
        
        # convert location and size to integers
        scale_factor = self.dimensions[0] / self.level_dimensions[level][0]
        x, y = location
        scaled_x = int(x / scale_factor)
        scaled_y = int(y / scale_factor)
        scaled_w = int(size[0] / scale_factor)
        scaled_h = int(size[1] / scale_factor)
        img_array = self.isyntax_reader.read_region(
            scaled_x, scaled_y, scaled_w, scaled_h, level)
        return img_array if as_array else Image.fromarray(img_array, mode='RGBA')