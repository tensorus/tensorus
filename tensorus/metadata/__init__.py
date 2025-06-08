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
from tensorus.config import settings
from .storage_abc import MetadataStorage
from .storage import InMemoryStorage
from .postgres_storage import PostgresMetadataStorage
from .schemas_iodata import TensorusExportData, TensorusExportEntry # Import I/O schemas

class ConfigurationError(Exception):
    pass

def get_configured_storage_instance() -> MetadataStorage:
    if settings.STORAGE_BACKEND == "postgres":
        if settings.POSTGRES_DSN:
            return PostgresMetadataStorage(dsn=settings.POSTGRES_DSN)
        elif settings.POSTGRES_HOST and settings.POSTGRES_USER and settings.POSTGRES_DB:
            return PostgresMetadataStorage(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT or 5432,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                database=settings.POSTGRES_DB
            )
        else:
            raise ConfigurationError(
                "PostgreSQL backend selected, but required connection details "
                "(DSN or Host/User/DB) are missing. Please set TENSORUS_POSTGRES_DSN "
                "or TENSORUS_POSTGRES_HOST, TENSORUS_POSTGRES_USER, TENSORUS_POSTGRES_DB."
            )
    elif settings.STORAGE_BACKEND == "in_memory":
        return InMemoryStorage()
    else:
        raise ConfigurationError(f"Unsupported storage backend: {settings.STORAGE_BACKEND}")

storage_instance: MetadataStorage = get_configured_storage_instance()

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
    # Extended Schemas - Helper Classes & Enums
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
    # I/O Schemas
    "TensorusExportData",
    "TensorusExportEntry",
    # Storage Abstraction & Implementations
    "MetadataStorage",
    "InMemoryStorage",
    "PostgresMetadataStorage",
    "storage_instance",
    "ConfigurationError",
    "get_configured_storage_instance"
]
