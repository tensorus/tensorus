"""
Tensorus API Package.

This package contains the FastAPI application and endpoints for interacting
with Tensor Descriptors and Semantic Metadata.

To run the API (assuming Uvicorn is installed and you are in the project
root):
`uvicorn tensorus.api:app --reload --port 7860`
"""

# Re-export the FastAPI application defined in ``tensorus/api.py`` so users
# can simply ``import tensorus.api`` to access ``app``. We load the module by
# path to avoid import order issues with the ``tensorus.api`` package itself.
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "tensorus.api_app", Path(__file__).resolve().parent.parent / "api.py"
)
_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_api)  # type: ignore[call-arg]
app = _api.app

__all__ = ["app"]
