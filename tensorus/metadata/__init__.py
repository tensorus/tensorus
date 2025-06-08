"""
Tensorus Metadata Package.

This package provides schemas for describing tensors and their semantic meaning,
as well as a basic in-memory storage mechanism for managing these metadata objects.
"""

from .schemas import (
    TensorDescriptor,
    SemanticMetadata,
    DataType,
    StorageFormat,
    AccessControl,
    CompressionInfo
)
from .storage import InMemoryStorage, storage_instance

__all__ = [
    "TensorDescriptor",
    "SemanticMetadata",
    "DataType",
    "StorageFormat",
    "AccessControl",
    "CompressionInfo",
    "InMemoryStorage",
    "storage_instance",
]
