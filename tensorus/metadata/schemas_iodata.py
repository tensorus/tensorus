from typing import List, Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field

from .schemas import (
    TensorDescriptor, SemanticMetadata,
    LineageMetadata, ComputationalMetadata, QualityMetadata,
    RelationalMetadata, UsageMetadata
)

class TensorusExportEntry(BaseModel):
    tensor_descriptor: TensorDescriptor
    semantic_metadata: Optional[List[SemanticMetadata]] = Field(default_factory=list)
    lineage_metadata: Optional[LineageMetadata] = None
    computational_metadata: Optional[ComputationalMetadata] = None
    quality_metadata: Optional[QualityMetadata] = None
    relational_metadata: Optional[RelationalMetadata] = None
    usage_metadata: Optional[UsageMetadata] = None

class TensorusExportData(BaseModel):
    export_format_version: str = Field(default="1.0") # Renamed for clarity
    exported_at: datetime = Field(default_factory=datetime.utcnow)
    entries: List[TensorusExportEntry]

__all__ = ["TensorusExportEntry", "TensorusExportData"]
