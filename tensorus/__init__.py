"""Tensorus core package."""

import os
import logging

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Initialize root logger if not already configured."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format=LOG_FORMAT, handlers=[logging.StreamHandler()])


configure_logging()

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
