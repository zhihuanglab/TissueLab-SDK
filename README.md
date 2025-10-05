<div align="center">

# TissueLab SDK

**OS-aware imaging wrappers for medical image processing**

</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-tissuelab-blue.svg)](https://pypi.org/project/tissuelab/)

</div>

<br>

## 🚀 Quick Start

### Installation

```bash
pip install tissuelab_sdk
```

### Basic Usage

```python
from tissuelab_sdk.wrapper import TiffSlideWrapper, DicomImageWrapper

# Load a TIFF slide
with TiffSlideWrapper("path/to/slide.tiff") as slide:
    print(f"Dimensions: {slide.dimensions}")
    region = slide.read_region(location=(0, 0), level=0, size=(512, 512))

# Load a DICOM image
with DicomImageWrapper("path/to/image.dcm") as dicom:
    region = dicom.read_region(location=(0, 0), level=0, size=(256, 256))
```

## 📦 Supported Formats

- **TIFF/TIFFSlide**: Standard TIFF and tiled TIFF files
- **DICOM**: Medical imaging standard
- **NIfTI**: Neuroimaging format
- **CZI**: Zeiss microscopy format (Windows)
- **ISyntax**: Philips pathology format (Windows)
- **Simple Images**: JPEG, PNG, BMP, etc.

## 🔧 API Reference

### Core Wrappers

```python
from tissuelab_sdk.wrapper import (
    TiffSlideWrapper,    # TIFF files
    DicomImageWrapper,   # DICOM files
    NiftiImageWrapper,   # NIfTI files
    SimpleImageWrapper,  # JPEG, PNG, etc.
    CziImageWrapper,     # CZI files (Windows)
    ISyntaxImageWrapper  # ISyntax files (Windows)
)

# All wrappers share the same interface
with TiffSlideWrapper("image.tiff") as wrapper:
    # Properties
    wrapper.dimensions          # (width, height)
    wrapper.level_count         # Number of pyramid levels
    wrapper.properties          # Dictionary of metadata
    
    # Methods
    region = wrapper.read_region(location=(x, y), level=0, size=(w, h))
    thumbnail = wrapper.get_thumbnail((256, 256))
```

## 🏥 Integration with TissueLab

This SDK is part of the [TissueLab](https://github.com/zhihuanglab/TissueLab) ecosystem:

```python
# TissueLab automatically uses this SDK for image loading
from tissuelab_sdk.wrapper import TiffSlideWrapper

def analyze_slide(slide_path):
    with TiffSlideWrapper(slide_path) as slide:
        # Process the slide
        return process_image(slide)
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

- **Main Project**: [TissueLab](https://github.com/zhihuanglab/TissueLab)
- **Issues**: [GitHub Issues](https://github.com/zhihuanglab/TissueLab-SDK/issues)
- **Paper**: [arXiv:2509.20279](https://arxiv.org/abs/2509.20279)

## 📚 Citation

If you use TissueLab SDK in your research, please cite our paper:

```bibtex
@article{li2025co,
  title={A co-evolving agentic AI system for medical imaging analysis},
  author={Li, Songhao and Xu, Jonathan and Bao, Tiancheng and Liu, Yuxuan and Liu, Yuchen and Liu, Yihang and Wang, Lilin and Lei, Wenhui and Wang, Sheng and Xu, Yinuo and Cui, Yan and Yao, Jialu and Koga, Shunsuke and Huang, Zhi},
  journal={arXiv preprint arXiv:2509.20279},
  year={2025}
}
```