from enum import Enum
from typing import List, Dict, Optional
from uuid import UUID, uuid4
from datetime import datetime

from pydantic import BaseModel, Field, validator

class DataType(str, Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    BOOLEAN = "boolean"
    STRING = "string"

class StorageFormat(str, Enum):
    RAW = "raw"
    COMPRESSED_ZLIB = "compressed_zlib"
    COMPRESSED_GZIP = "compressed_gzip"

class AccessControl(BaseModel):
    read: List[str] = Field(default_factory=list)
    write: List[str] = Field(default_factory=list)
    delete: List[str] = Field(default_factory=list)

class CompressionInfo(BaseModel):
    algorithm: str
    level: Optional[int] = None

class TensorDescriptor(BaseModel):
    tensor_id: UUID = Field(default_factory=uuid4)
    dimensionality: int = Field(..., ge=0) # ge=0 means greater than or equal to 0
    shape: List[int]
    data_type: DataType
    storage_format: StorageFormat = StorageFormat.RAW
    creation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    last_modified_timestamp: datetime = Field(default_factory=datetime.utcnow)
    owner: str
    access_control: AccessControl = Field(default_factory=AccessControl)
    byte_size: int = Field(..., ge=0)
    compression_info: Optional[CompressionInfo] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, str]] = None # Generic metadata

    @validator('shape')
    def validate_shape(cls, v, values):
        if 'dimensionality' in values and len(v) != values['dimensionality']:
            raise ValueError('Shape must have a length equal to dimensionality')
        if not all(isinstance(dim, int) and dim >= 0 for dim in v):
            raise ValueError('All dimensions in shape must be non-negative integers')
        return v

    @validator('last_modified_timestamp')
    def validate_last_modified(cls, v, values):
        if 'creation_timestamp' in values and v < values['creation_timestamp']:
            raise ValueError('Last modified timestamp cannot be before creation timestamp')
        return v

    # Pydantic v2 automatically revalidates on model update,
    # so we can use a simple assignment to update last_modified_timestamp
    def update_last_modified(self):
        self.last_modified_timestamp = datetime.utcnow()


class SemanticMetadata(BaseModel):
    name: str
    description: str
    tensor_id: UUID # To link with a TensorDescriptor

    @validator('name', 'description')
    def check_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Name and description cannot be empty')
        return v
