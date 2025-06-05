"""Tensorus core package."""

import os

if not os.environ.get("TENSORUS_MINIMAL_IMPORT"):
    from . import models

    __all__ = ["models"]
else:
    __all__ = []
