"""
Tensorus API Package.

This package contains the FastAPI application and endpoints for interacting
with Tensor Descriptors and Semantic Metadata.

To run the API (assuming Uvicorn is installed and you are in the project root):
`uvicorn tensorus.api.main:app --reload --port 8000`
"""

from .main import app

__all__ = ["app"]
