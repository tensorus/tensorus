# Metadata Schemas Overview

The Tensorus Metadata System employs a rich set of Pydantic schemas to define the structure and validation rules for metadata. These schemas ensure data consistency and provide a clear contract for API interactions.

## Core Schema: TensorDescriptor

The `TensorDescriptor` is the fundamental metadata unit. It captures essential information about a tensor's structure, storage, and identity. Key fields include:

*   `tensor_id`: Unique UUID for the tensor.
*   `dimensionality`: Number of dimensions.
*   `shape`: Size of each dimension.
*   `data_type`: Data type of tensor elements (e.g., `float32`, `int64`).
*   `storage_format`: How the tensor is physically stored (e.g., `raw`, `numpy_npz`).
*   `creation_timestamp`, `last_modified_timestamp`.
*   `owner`, `access_control` (which itself is a nested model detailing read/write permissions).
*   `byte_size`, `checksum` (e.g., MD5, SHA256 of the tensor data).
*   `compression_info` (details if tensor data is compressed).
*   `tags`: A list of arbitrary string tags for categorization.
*   `metadata`: A flexible dictionary for any other custom key-value metadata.

## Semantic Metadata

Associated with a `TensorDescriptor` (one-to-many, identified by `name` per tensor), this schema describes the meaning and context of the tensor data. Key fields:

*   `tensor_id`: Links to the parent `TensorDescriptor`.
*   `name`: The specific name of this semantic annotation (e.g., "primary_class_label", "object_bounding_boxes", "feature_description_set").
*   `description`: Detailed explanation of this semantic annotation.
*   *(Note: The original broader concept of a single SemanticMetadata object per tensor with fields like `domain`, `purpose`, etc., has evolved. These broader concepts might be captured in `TensorDescriptor.tags`, `TensorDescriptor.metadata`, or specific named `SemanticMetadata` entries.)*

## Extended Metadata Schemas

These provide more detailed and specialized information, typically one-to-one with a `TensorDescriptor`:

*   **`LineageMetadata`**: Tracks origin (`source`), parent tensors (`parent_tensors`), a history of transformations (`transformation_history`), version string (`version`), version control details (`version_control`), and other provenance information (`provenance`).
    *   `LineageSource`: Details the origin (e.g., file, API, computation).
    *   `ParentTensorLink`: Links to parent tensors and describes the relationship.
    *   `TransformationStep`: Describes an operation in the transformation history.
    *   `VersionControlInfo`: Git-like versioning details for the source or tensor itself.

*   **`ComputationalMetadata`**: Describes how the tensor was computed, including the `algorithm` used, `parameters`, reference to a `computational_graph_ref`, `execution_environment` details, `computation_time_seconds`, and `hardware_info`.

*   **`QualityMetadata`**: Captures information about data quality. Includes:
    *   `statistics` (`QualityStatistics` model: min, max, mean, std_dev, median, variance, percentiles, histogram).
    *   `missing_values` (`MissingValuesInfo` model: count, percentage, imputation strategy).
    *   `outliers` (`OutlierInfo` model: count, percentage, detection method).
    -   `noise_level`, `confidence_score`, `validation_results` (custom checks), `drift_score`.

*   **`RelationalMetadata`**: Describes relationships to other tensors (`related_tensors` via `RelatedTensorLink`), membership in `collections`, explicit `dependencies` on other tensors, and `dataset_context`.

*   **`UsageMetadata`**: Tracks how the tensor is used:
    *   `access_history` (list of `UsageAccessRecord` detailing who/what accessed it, when, and how).
    *   `usage_frequency` (auto-calculated from history or can be set).
    *   `last_accessed_at` (auto-calculated from history or can be set).
    *   `application_references` (list of applications or models that use this tensor).
    *   `purpose` (dictionary describing purposes, e.g. for specific model training).

For the exact field definitions, types, optionality, default values, and validation rules for each schema, please refer to the source code in the `tensorus/metadata/schemas.py` module or consult the schemas provided in the interactive API documentation at `/docs`. The I/O schemas for export/import are defined in `tensorus/metadata/schemas_iodata.py`.
