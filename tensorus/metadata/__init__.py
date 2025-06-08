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
    CompressionInfo,
    # Extended Schemas
    LineageSourceType,
    LineageSource,
    ParentTensorLink,
    TransformationStep,
    VersionControlInfo,
    LineageMetadata,
    ComputationalMetadata,
    QualityStatistics,
    MissingValuesInfo,
    OutlierInfo,
    QualityMetadata,
    RelatedTensorLink,
    RelationalMetadata,
    UsageAccessRecord,
    UsageMetadata
)
from .storage import InMemoryStorage, storage_instance

__all__ = [
    # Core Schemas
    "TensorDescriptor",
    "SemanticMetadata",
    "DataType",
    "StorageFormat",
    "AccessControl",
    "CompressionInfo",
    # Extended Schemas - Main Classes
    "LineageMetadata",
    "ComputationalMetadata",
    "QualityMetadata",
    "RelationalMetadata",
    "UsageMetadata",
    # Extended Schemas - Helper Classes & Enums (selectively exported if needed directly)
    "LineageSourceType",
    "LineageSource",
    "ParentTensorLink",
    "TransformationStep",
    "VersionControlInfo",
    "QualityStatistics",
    "MissingValuesInfo",
    "OutlierInfo",
    "RelatedTensorLink",
    "UsageAccessRecord",
    # Storage
    "InMemoryStorage",
    "storage_instance",
]
