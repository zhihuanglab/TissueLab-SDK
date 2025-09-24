import sys
from .common import *  # always expose shared wrappers

def is_windows() -> bool:
    return sys.platform.startswith('win')

# On Windows, also expose the extra CZI wrapper
if is_windows():
    from .windows import CziImageWrapper  # noqa: F401

