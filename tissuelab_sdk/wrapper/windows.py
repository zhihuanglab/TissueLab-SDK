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
        self.czi_reader = czi.CziReader(czi_path)
        self._init_metadata()
        self._init_levels(max_levels)
        self._init_thumbnail()
    
    def __del__(self):
        try:
            if hasattr(self, 'czi_reader') and self.czi_reader is not None:
                self.czi_reader.close()
                self.czi_reader = None
        except Exception:
            # Suppress errors during cleanup
            pass

    def _init_metadata(self):
        bounds = self.czi_reader.total_bounding_box
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

    def _init_thumbnail(self):
        """Initialize thumbnail from CZI file using CZIFile"""
        try:
            # Initialize associated_images dictionary
            self.associated_images = {}
            
            # Use CZIFile to read thumbnail attachment
            with czifile.CziFile(self.path) as czi_file:
                for attachment in czi_file.attachments():
                    try:
                        entry = attachment.attachment_entry
                        # Check if this is a Thumbnail attachment
                        if hasattr(entry, 'name') and entry.name == 'Thumbnail':
                            
                            # Read the thumbnail data
                            data = attachment.data()
                            if data:
                                # Create PIL image from the data
                                img = Image.open(io.BytesIO(data))
                                
                                # Convert to RGB if needed
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                
                                # Store as macro and overview for compatibility
                                self.associated_images['macro'] = img
                                self.associated_images['overview'] = img
                                break
                                
                    except Exception as e:
                        print(f"Warning: Error processing attachment: {e}")
                        continue
                        
        except Exception as e:
            print(f"Warning: Could not initialize CZI thumbnail: {e}")
            self.associated_images = {}

    def read_region(self, location, level, size, as_array=False):
            if level >= self.level_count:
                raise ValueError(
                    f"Requested level {level} exceeds available levels {self.level_count}"
                )
            downsample = 2**level
            x, y = location
            w, h = size
            roi_x = int(x + self.x_range[0])
            roi_y = int(y + self.y_range[0])
            roi_w = int(w * downsample)
            roi_h = int(h * downsample)
            zoom = 1.0 / downsample
            img = self.czi_reader.read(roi=(roi_x, roi_y, roi_w, roi_h),
                                      zoom=zoom,
                                      scene=0)
            # print(f"finished reading")
            # time.sleep(5)
            # BGR to RGB, fill blank space with white
            img = img[:, :, ::-1]
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            img[img == 0] = 255
            pil_img = Image.fromarray(img)
            return np.array(pil_img) if as_array else pil_img


class ISyntaxImageWrapper:
    """Wrapper for ISyntax image files using WSI interface"""

    def __init__(self, isyntax_path):
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