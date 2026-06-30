"""Tensorus v1.0 Python SDK.

A thin, ergonomic wrapper over the native Rust engine (``tensorus._native``).
The native module does the heavy lifting (storage, descriptor computation); this
layer adds numpy convenience and a friendlier surface.

Example
-------
>>> import numpy as np
>>> from tensorus import Tensorus
>>> ts = Tensorus.memory()
>>> ts.create_dataset("weights")
>>> tid = ts.insert("weights", np.eye(3, dtype=np.float32), metadata={"name": "I3"})
>>> back = ts.get("weights", tid)
>>> bool(np.allclose(back, np.eye(3)))
True
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

from ._native import Tensorus as _NativeTensorus
from ._native import __version__ as __version__  # re-export

__all__ = ["Tensorus", "__version__"]


class Tensorus:
    """High-level client for a Tensorus instance."""

    def __init__(self, native: _NativeTensorus) -> None:
        self._native = native

    # -- construction ---------------------------------------------------------

    @classmethod
    def memory(cls) -> "Tensorus":
        """Open an ephemeral instance in a temporary directory."""
        return cls(_NativeTensorus.memory())

    @classmethod
    def local(cls, data_dir: Optional[str] = None) -> "Tensorus":
        """Open a local instance rooted at ``data_dir`` (default ``./data``)."""
        return cls(_NativeTensorus.local(data_dir))

    @classmethod
    def connect(cls, uri: str) -> "Tensorus":
        """Connect to a Tensorus instance by URI (local path or ``file://``)."""
        return cls(_NativeTensorus.connect(uri))

    # -- datasets -------------------------------------------------------------

    def create_dataset(self, name: str) -> None:
        self._native.create_dataset(name)

    def list_datasets(self) -> list[str]:
        return self._native.list_datasets()

    # -- tensors --------------------------------------------------------------

    def insert(
        self,
        dataset: str,
        tensor: "np.ndarray",
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Insert a tensor (coerced to ``float32``) and return its id."""
        arr = np.ascontiguousarray(tensor, dtype=np.float32)
        meta = json.dumps(metadata) if metadata is not None else None
        return self._native.insert(dataset, arr, meta)

    def get(self, dataset: str, tensor_id: str) -> "np.ndarray":
        """Fetch a tensor by id as a numpy ``float32`` array."""
        return self._native.get(dataset, tensor_id)

    def delete(self, dataset: str, tensor_id: str) -> None:
        self._native.delete(dataset, tensor_id)
