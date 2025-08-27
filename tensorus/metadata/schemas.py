from enum import Enum
from typing import List, Dict, Optional, Any
from uuid import UUID, uuid4
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo

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

    @field_validator('shape', mode='before')
    def validate_shape(cls, v, info: ValidationInfo):
        dimensionality = info.data.get('dimensionality')
        if dimensionality is not None and len(v) != dimensionality:
            raise ValueError('Shape must have a length equal to dimensionality')
        if not all(isinstance(dim, int) and dim >= 0 for dim in v):
            raise ValueError('All dimensions in shape must be non-negative integers')
        return v

    @field_validator('last_modified_timestamp') # Defaults to mode='after'
    def validate_last_modified(cls, v: datetime, info: ValidationInfo): # v is now datetime
        # Ensure creation_timestamp is also datetime if accessed from info.data
        # However, it's better to rely on already validated fields if possible,
        # or ensure this validator runs after creation_timestamp is validated and converted.
        # For direct comparison, both should be datetime.
        # Pydantic v2 typically ensures other fields referenced in model_validator or
        # late-stage field_validators are already validated/coerced.
        # Assuming creation_timestamp in info.data is already a datetime due to its type hint and default_factory
        creation_timestamp_from_data = info.data.get('creation_timestamp')
        if isinstance(creation_timestamp_from_data, datetime) and v < creation_timestamp_from_data:
            raise ValueError('Last modified timestamp cannot be before creation timestamp')
        # If creation_timestamp is not yet a datetime (e.g. if it was also mode='before' and a string),
        # this comparison would also fail. Best practice is mode='after' for inter-field validation
        # when types are critical.
        return v

    def update_last_modified(self):
        self.last_modified_timestamp = datetime.utcnow()


class SemanticMetadata(BaseModel):
    # Link to TensorDescriptor is implicit via storage key (tensor_id)
    # No, explicit tensor_id is better for standalone validation and clarity.
    tensor_id: UUID
    name: str # Name of this specific semantic annotation (e.g., "primary_class_label", "bounding_boxes")
    description: str

    @field_validator('name', 'description')
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

    @field_validator('computation_time_seconds')
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

    @model_validator(mode='after')
    def sync_derived_usage_fields(self) -> 'UsageMetadata':
        if self.access_history: # This will be the fully validated list of UsageAccessRecord objects
            # Update usage_frequency
            self.usage_frequency = len(self.access_history)

            # Update last_accessed_at
            latest_access_in_history = max(record.accessed_at for record in self.access_history)
            if self.last_accessed_at is None or latest_access_in_history > self.last_accessed_at:
                self.last_accessed_at = latest_access_in_history
        else:
            # If there's no access_history, ensure frequency is 0 if not explicitly set otherwise
            # and last_accessed_at remains as is (or None).
            # The default Field(default=0) for usage_frequency should handle the initial case.
            # If an empty list is provided for access_history, this will correctly set frequency to 0.
            self.usage_frequency = 0
            # self.last_accessed_at will retain its input or default None if access_history is empty

        return self

# --- Vector Database Specific Metadata Schemas ---

class EmbeddingModelInfo(BaseModel):
    """Information about the embedding model used to generate vectors."""
    model_name: str = Field(..., description="Name/identifier of the embedding model")
    model_version: Optional[str] = None
    provider: Optional[str] = None  # e.g., "sentence-transformers", "openai", "huggingface"
    embedding_dimension: int = Field(..., ge=1, description="Dimensionality of the embedding vectors")
    max_sequence_length: Optional[int] = None
    model_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    normalization: Optional[str] = None  # e.g., "l2", "none"

class VectorIndexInfo(BaseModel):
    """Information about vector indexing for similarity search."""
    index_type: str = Field(..., description="Type of vector index (e.g., 'flat', 'ivf', 'hnsw')")
    index_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    distance_metric: str = Field(default="cosine", description="Distance metric used for similarity")
    is_trained: bool = Field(default=False, description="Whether the index has been trained")
    num_vectors_indexed: Optional[int] = Field(default=0, ge=0)
    index_build_timestamp: Optional[datetime] = None
    index_file_path: Optional[str] = None  # Path to serialized index file

class VectorMetadata(BaseModel):
    """Metadata specific to vector embeddings and similarity search."""
    tensor_id: UUID
    is_embedding: bool = Field(default=True, description="Whether this tensor is an embedding vector")
    embedding_model: Optional[EmbeddingModelInfo] = None
    source_text: Optional[str] = None  # Original text that was embedded
    source_data_type: Optional[str] = None  # e.g., "text", "image", "audio"
    vector_index: Optional[VectorIndexInfo] = None
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    nearest_neighbors: Optional[List[UUID]] = Field(default_factory=list)  # Cached nearest neighbor IDs
    cluster_id: Optional[str] = None  # If vector belongs to a cluster
    vector_tags: Optional[List[str]] = Field(default_factory=list)  # Semantic tags for the vector

    @field_validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError('Similarity threshold must be between 0.0 and 1.0')
        return v

class SimilaritySearchResult(BaseModel):
    """Result from a vector similarity search operation."""
    query_vector_id: Optional[UUID] = None
    result_vector_id: UUID
    similarity_score: float = Field(..., description="Similarity score between query and result")
    distance: Optional[float] = None  # Distance metric value
    rank: int = Field(..., ge=1, description="Rank in search results (1-based)")
    metadata_match: Optional[Dict[str, Any]] = Field(default_factory=dict)  # Additional metadata matches

    @field_validator('similarity_score')
    def validate_similarity_score(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError('Similarity score must be numeric')
        return float(v)

class HybridSearchMetadata(BaseModel):
    """Metadata for hybrid search combining vector similarity and metadata filtering."""
    tensor_id: UUID
    search_query: str = Field(..., description="Original search query")
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for vector similarity")
    metadata_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for metadata matching")
    filters_applied: Optional[Dict[str, Any]] = Field(default_factory=dict)
    search_timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_candidates: Optional[int] = Field(default=None, ge=0)
    results_returned: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode='after')
    def validate_weights_sum(self) -> 'HybridSearchMetadata':
        if abs(self.vector_weight + self.metadata_weight - 1.0) > 1e-6:
            raise ValueError('Vector weight and metadata weight must sum to 1.0')
        return self

