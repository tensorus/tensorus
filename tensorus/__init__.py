"""Tensorus core package."""

import os

# The repository previously exposed a large collection of models under
# ``tensorus.models``. These models have been moved to a separate package.
# Tensorus now only attempts to import them if available.

if not os.environ.get("TENSORUS_MINIMAL_IMPORT"):
    try:
        import importlib

        models = importlib.import_module("tensorus.models")
        __all__ = ["models"]
    except ModuleNotFoundError:
        __all__ = []
else:
    __all__ = []
