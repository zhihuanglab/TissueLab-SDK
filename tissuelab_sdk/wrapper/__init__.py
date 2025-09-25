"""Public wrapper API under tissuelab_sdk.wrapper

This module now directly exposes the implementations located within this
package. Import via: `from tissuelab_sdk.wrapper import ...`.
"""

from .common import *  # noqa: F401,F403

# On Windows, expose the extra CZI and ISyntax wrappers
import sys

def _is_windows() -> bool:
    return sys.platform.startswith('win')

if _is_windows():
    from .windows import CziImageWrapper, ISyntaxImageWrapper  # noqa: F401


