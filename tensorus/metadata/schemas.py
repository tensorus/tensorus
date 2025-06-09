from enum import Enum
from typing import List, Dict, Optional, Any
from uuid import UUID, uuid4
from datetime import datetime

from pydantic import BaseModel, Field, validator

class DataType(str, Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    FLOAT16 = "float16" # Added
    INT32 = "int32"
    INT64 = "int64"
    INT16 = "int16"   # Added
    INT8 = "int8"     # Added
    UINT8 = "uint8"   # Added
    BOOLEAN = "boolean"
    STRING = "string"
    COMPLEX64 = "complex64" # Added
    COMPLEX128 = "complex128" # Added
    OTHER = "other" # Added

class StorageFormat(str, Enum):
    RAW = "raw"
    NUMPY_NPZ = "numpy_npz" # Added
    HDF5 = "hdf5" # Added
    COMPRESSED_ZLIB = "compressed_zlib"
    COMPRESSED_GZIP = "compressed_gzip"
    CUSTOM = "custom" # Added


class AccessControl(BaseModel):
    read: List[str] = Field(default_factory=list)
    write: List[str] = Field(default_factory=list)
    delete: List[str] = Field(default_factory=list)
    owner_permissions: Optional[str] = None # e.g. "rwd"
    group_permissions: Optional[Dict[str, str]] = None # e.g. {"group_name": "rw"}


class CompressionInfo(BaseModel):
    algorithm: str
    level: Optional[int] = None
    settings: Optional[Dict[str, Any]] = None # For other settings


class TensorDescriptor(BaseModel):
    tensor_id: UUID = Field(default_factory=uuid4)
    dimensionality: int = Field(..., ge=0)
    shape: List[int]
    data_type: DataType
    storage_format: StorageFormat = StorageFormat.RAW
    creation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    last_modified_timestamp: datetime = Field(default_factory=datetime.utcnow)
    owner: str # User or service ID
    access_control: AccessControl = Field(default_factory=AccessControl)
    byte_size: int = Field(..., ge=0)
    checksum: Optional[str] = None # e.g. md5, sha256
    compression_info: Optional[CompressionInfo] = None
    tags: Optional[List[str]] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict) # Generic metadata, allows richer values

    @validator('shape', always=True) # always=True ensures it runs even if shape is not explicitly provided (e.g. during model copy/update)
    def validate_shape(cls, v, values, **kwargs): # Pydantic v2 uses field_validator, this is for v1
        # In Pydantic v1, `values` is the dict of other fields.
        # In Pydantic v2, it's `values: ValidationInfo` object.
        dimensionality = values.get('dimensionality')
        if dimensionality is not None and len(v) != dimensionality:
            raise ValueError('Shape must have a length equal to dimensionality')
        if not all(isinstance(dim, int) and dim >= 0 for dim in v):
            raise ValueError('All dimensions in shape must be non-negative integers')
        return v

    @validator('last_modified_timestamp', always=True)
    def validate_last_modified(cls, v, values, **kwargs):
        creation_timestamp = values.get('creation_timestamp')
        if creation_timestamp and v < creation_timestamp:
            raise ValueError('Last modified timestamp cannot be before creation timestamp')
        return v

    def update_last_modified(self):
        self.last_modified_timestamp = datetime.utcnow()


class SemanticMetadata(BaseModel):
    # Link to TensorDescriptor is implicit via storage key (tensor_id)
    # No, explicit tensor_id is better for standalone validation and clarity.
    tensor_id: UUID
    name: str # Name of this specific semantic annotation (e.g., "primary_class_label", "bounding_boxes")
    description: str

    @validator('name', 'description')
    def check_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Name and description fields cannot be empty or just whitespace.')
        return v

# --- Extended Schemas ---

# Part of LineageMetadata
class LineageSourceType(str, Enum):
    FILE = "file"
    API = "api"
    COMPUTATION = "computation"
    DATABASE = "database"
    STREAM = "stream"
    USER_UPLOAD = "user_upload" # Added
    SYNTHETIC = "synthetic" # Added
    OTHER = "other"

class LineageSource(BaseModel):
    type: LineageSourceType
    identifier: str # e.g., file path, API endpoint URL, query string, stream topic
    details: Optional[Dict[str, Any]] = Field(default_factory=dict) # e.g., API request params, version of source data

# Part of LineageMetadata
class ParentTensorLink(BaseModel):
    tensor_id: UUID
    relationship: Optional[str] = None # e.g., "transformed_from", "derived_from", "aggregated_from"

# Part of LineageMetadata
class TransformationStep(BaseModel):
    operation: str # e.g., "normalize", "resize", "fft", "model_inference"
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    operator: Optional[str] = None # User or service that performed the operation
    software_version: Optional[str] = None # e.g., library version used for the operation

# Part of LineageMetadata
class VersionControlInfo(BaseModel):
    repository: Optional[str] = None # URL of the repository
    commit_hash: Optional[str] = None
    branch: Optional[str] = None
    tag: Optional[str] = None
    path_in_repo: Optional[str] = None # If applicable

class LineageMetadata(BaseModel):
    tensor_id: UUID # Links to the TensorDescriptor
    source: Optional[LineageSource] = None
    parent_tensors: List[ParentTensorLink] = Field(default_factory=list)
    transformation_history: List[TransformationStep] = Field(default_factory=list)
    version: Optional[str] = None # Version string for this tensor instance
    version_control: Optional[VersionControlInfo] = None
    provenance: Optional[Dict[str, Any]] = Field(default_factory=dict) # For other unstructured provenance info

class ComputationalMetadata(BaseModel):
    tensor_id: UUID
    algorithm: Optional[str] = None # e.g., "ResNet50", "PCA", "ARIMA"
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict) # Algorithm parameters
    computational_graph_ref: Optional[str] = None # Reference to a stored graph (e.g., ONNX model path, DVC stage)
    execution_environment: Optional[Dict[str, Any]] = Field(default_factory=dict) # e.g., OS, Python version, library versions
    computation_time_seconds: Optional[float] = None
    hardware_info: Optional[Dict[str, Any]] = Field(default_factory=dict) # e.g., CPU, GPU, RAM

    @validator('computation_time_seconds')
    def check_non_negative_time(cls, v):
        if v is not None and v < 0:
            raise ValueError('Computation time cannot be negative')
        return v

# Part of QualityMetadata
class QualityStatistics(BaseModel):
    min_value: Optional[float] = None # Renamed for clarity
    max_value: Optional[float] = None # Renamed for clarity
    mean: Optional[float] = None
    std_dev: Optional[float] = None
    median: Optional[float] = None
    variance: Optional[float] = None
    percentiles: Optional[Dict[float, float]] = None # e.g. {25: val, 50: val, 75: val}
    histogram: Optional[Dict[str, Any]] = None # e.g. {"bins": [], "counts": []}

# Part of QualityMetadata
class MissingValuesInfo(BaseModel):
    count: int = Field(..., ge=0)
    percentage: float = Field(..., ge=0.0, le=100.0)
    strategy: Optional[str] = None # e.g., "imputed_mean", "removed_rows"

# Part of QualityMetadata
class OutlierInfo(BaseModel):
    count: int = Field(..., ge=0)
    percentage: float = Field(..., ge=0.0, le=100.0)
    method_used: Optional[str] = None # e.g., "IQR", "Z-score"
    severity: Optional[Dict[str, int]] = None # e.g. {"mild": 10, "severe": 2}

class QualityMetadata(BaseModel):
    tensor_id: UUID
    statistics: Optional[QualityStatistics] = None
    missing_values: Optional[MissingValuesInfo] = None
    outliers: Optional[OutlierInfo] = None
    noise_level: Optional[float] = None # Could be SNR or a qualitative score
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    validation_results: Optional[Dict[str, Any]] = Field(default_factory=dict) # e.g., {"schema_conformity": True, "range_checks": "passed"}
    drift_score: Optional[float] = None # Data drift score compared to a reference

# Part of RelationalMetadata
class RelatedTensorLink(BaseModel):
    related_tensor_id: UUID # Renamed for clarity
    relationship_type: str # e.g., "augmentation_of", "projection_of", "component_of", "alternative_view"
    details: Optional[Dict[str, Any]] = Field(default_factory=dict)

class RelationalMetadata(BaseModel):
    tensor_id: UUID
    related_tensors: List[RelatedTensorLink] = Field(default_factory=list)
    collections: List[str] = Field(default_factory=list) # List of collection names or IDs this tensor belongs to
    dependencies: List[UUID] = Field(default_factory=list) # Other tensors this one directly depends on (not necessarily lineage parents)
    dataset_context: Optional[str] = None # Name or ID of the dataset this tensor is part of

# Part of UsageMetadata
class UsageAccessRecord(BaseModel):
    accessed_at: datetime = Field(default_factory=datetime.utcnow)
    user_or_service: str
    operation_type: str # e.g., "read", "write", "query", "transform", "visualize"
    details: Optional[Dict[str, Any]] = Field(default_factory=dict) # e.g., query parameters, sub-selection info
    status: Optional[str] = "success" # "success", "failure"

class UsageMetadata(BaseModel):
    tensor_id: UUID
    access_history: List[UsageAccessRecord] = Field(default_factory=list)
    usage_frequency: Optional[int] = Field(default=0, ge=0) # Could be total accesses or accesses in a time window
    last_accessed_at: Optional[datetime] = None # Explicitly tracked or derived from access_history
    application_references: List[str] = Field(default_factory=list) # Names or IDs of applications/models using this tensor
    purpose: Optional[Dict[str, str]] = Field(default_factory=dict) # e.g. {"training_model_X": "feature_set_A"}

    @validator('last_accessed_at', always=True)
    def sync_last_accessed(cls, v, values):
        if values.get('access_history'):
            latest_access = max(record.accessed_at for record in values['access_history'])
            if v is None or latest_access > v:
                return latest_access
        return v

    @validator('usage_frequency', always=True)
    def sync_usage_frequency(cls, v, values):
        if values.get('access_history'):
            return len(values['access_history']) # Simple count, could be more complex
        return v if v is not None else 0

