import pytest
from uuid import UUID, uuid4
from datetime import datetime, timedelta

from pydantic import ValidationError

from tensorus.metadata.schemas import (
    TensorDescriptor,
    SemanticMetadata,
    DataType,
    StorageFormat,
    AccessControl,
    CompressionInfo,
    # Extended Schemas and helpers
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

# --- TensorDescriptor Enhanced Tests (from previous phase, plus new fields) ---

def test_tensor_descriptor_enhanced_valid():
    now = datetime.utcnow()
    td = TensorDescriptor(
        tensor_id=uuid4(),
        dimensionality=2,
        shape=[100, 100],
        data_type=DataType.FLOAT32,
        storage_format=StorageFormat.NUMPY_NPZ, # New enum member
        creation_timestamp=now,
        last_modified_timestamp=now + timedelta(seconds=1),
        owner="test_user_enhanced",
        access_control=AccessControl(read=["user1"], write=["owner"], owner_permissions="rwd"), # New field
        byte_size=40000,
        checksum="md5:abcdef123456", # New field
        compression_info=CompressionInfo(algorithm="gzip", level=6, settings={"comment": "test"}), # New field
        tags=["enhanced", "testing"],
        metadata={"source": "api_test", "quality_approved": True, "nested": {"value": 1}} # Richer metadata
    )
    assert td.storage_format == StorageFormat.NUMPY_NPZ
    assert td.checksum == "md5:abcdef123456"
    assert td.access_control.owner_permissions == "rwd"
    assert td.compression_info.settings["comment"] == "test"
    assert td.metadata["quality_approved"] is True

# --- LineageMetadata and Helpers Tests ---

def test_lineage_source_valid():
    ls = LineageSource(type=LineageSourceType.API, identifier="/api/v1/data/source", details={"param": "value"})
    assert ls.type == LineageSourceType.API
    assert ls.identifier == "/api/v1/data/source"
    assert ls.details["param"] == "value"

def test_parent_tensor_link_valid():
    ptl = ParentTensorLink(tensor_id=uuid4(), relationship="derived_from")
    assert isinstance(ptl.tensor_id, UUID)
    assert ptl.relationship == "derived_from"

def test_transformation_step_valid():
    ts = TransformationStep(operation="normalized", parameters={"mean": 0, "std": 1}, operator="proc_x", software_version="lib-v1.2")
    assert ts.operation == "normalized"
    assert ts.parameters["mean"] == 0
    assert ts.timestamp <= datetime.utcnow()

def test_version_control_info_valid():
    vci = VersionControlInfo(repository="http://git.example.com/repo.git", commit_hash="abcdef123", branch="main")
    assert vci.repository == "http://git.example.com/repo.git"

def test_lineage_metadata_valid():
    tensor_id = uuid4()
    lm = LineageMetadata(
        tensor_id=tensor_id,
        source=LineageSource(type=LineageSourceType.FILE, identifier="/path/to/source.csv"),
        parent_tensors=[ParentTensorLink(tensor_id=uuid4(), relationship="copied_from")],
        transformation_history=[TransformationStep(operation="cleaned")],
        version="v1.0.1",
        version_control=VersionControlInfo(commit_hash="xyz789"),
        provenance={"user_notes": "Initial dataset processing"}
    )
    assert lm.tensor_id == tensor_id
    assert lm.source.type == LineageSourceType.FILE
    assert len(lm.parent_tensors) == 1
    assert lm.version == "v1.0.1"
    assert lm.provenance["user_notes"] == "Initial dataset processing"

# --- ComputationalMetadata Tests ---

def test_computational_metadata_valid():
    cm = ComputationalMetadata(
        tensor_id=uuid4(),
        algorithm="PCA",
        parameters={"n_components": 10},
        computation_time_seconds=120.5,
        hardware_info={"cpu": "Intel Xeon", "ram_gb": 64}
    )
    assert cm.algorithm == "PCA"
    assert cm.computation_time_seconds == 120.5

def test_computational_metadata_invalid_time():
    with pytest.raises(ValidationError, match="Computation time cannot be negative"):
        ComputationalMetadata(tensor_id=uuid4(), computation_time_seconds=-10.0)

# --- QualityMetadata and Helpers Tests ---

def test_quality_statistics_valid():
    qs = QualityStatistics(min_value=0.0, max_value=1.0, mean=0.5, std_dev=0.2, percentiles={50: 0.49})
    assert qs.mean == 0.5
    assert qs.percentiles[50] == 0.49

def test_missing_values_info_valid():
    mvi = MissingValuesInfo(count=10, percentage=1.5, strategy="imputed_mean")
    assert mvi.count == 10
    assert mvi.percentage == 1.5

def test_missing_values_info_invalid_percentage():
    with pytest.raises(ValidationError): # Pydantic v1 shows field name, v2 more verbose
        MissingValuesInfo(count=0, percentage=110.0) # percentage > 100
    with pytest.raises(ValidationError):
        MissingValuesInfo(count=0, percentage=-5.0) # percentage < 0
    with pytest.raises(ValidationError):
        MissingValuesInfo(count=-1, percentage=10.0) # count < 0

def test_outlier_info_valid():
    oi = OutlierInfo(count=5, percentage=0.5, method_used="IQR")
    assert oi.count == 5

def test_quality_metadata_valid():
    qm = QualityMetadata(
        tensor_id=uuid4(),
        statistics=QualityStatistics(mean=10.0),
        missing_values=MissingValuesInfo(count=0, percentage=0.0),
        outliers=OutlierInfo(count=1, percentage=0.01, method_used="Z-score"),
        confidence_score=0.95
    )
    assert qm.statistics.mean == 10.0
    assert qm.confidence_score == 0.95

def test_quality_metadata_invalid_confidence():
    with pytest.raises(ValidationError):
        QualityMetadata(tensor_id=uuid4(), confidence_score=1.5) # > 1.0
    with pytest.raises(ValidationError):
        QualityMetadata(tensor_id=uuid4(), confidence_score=-0.1) # < 0.0

# --- RelationalMetadata and Helpers Tests ---

def test_related_tensor_link_valid():
    rtl = RelatedTensorLink(related_tensor_id=uuid4(), relationship_type="augmented_version")
    assert isinstance(rtl.related_tensor_id, UUID)
    assert rtl.relationship_type == "augmented_version"

def test_relational_metadata_valid():
    rm = RelationalMetadata(
        tensor_id=uuid4(),
        related_tensors=[RelatedTensorLink(related_tensor_id=uuid4(), relationship_type="sample_of")],
        collections=["dataset_A_main_features"],
        dependencies=[uuid4()]
    )
    assert len(rm.related_tensors) == 1
    assert "dataset_A_main_features" in rm.collections

# --- UsageMetadata and Helpers Tests ---

def test_usage_access_record_valid():
    uar = UsageAccessRecord(user_or_service="user_x", operation_type="read", details={"query": "full_tensor"})
    assert uar.user_or_service == "user_x"
    assert uar.status == "success" # Default value

def test_usage_metadata_valid():
    tensor_id = uuid4()
    now = datetime.utcnow()
    um = UsageMetadata(
        tensor_id=tensor_id,
        access_history=[
            UsageAccessRecord(user_or_service="user_a", operation_type="read", accessed_at=now - timedelta(days=1)),
            UsageAccessRecord(user_or_service="service_b", operation_type="transform", accessed_at=now)
        ],
        application_references=["model_training_pipeline_X"]
    )
    assert um.tensor_id == tensor_id
    assert len(um.access_history) == 2
    assert um.usage_frequency == 2 # Auto-calculated by validator
    assert um.last_accessed_at == now # Auto-calculated by validator

def test_usage_metadata_sync_validators():
    um = UsageMetadata(tensor_id=uuid4())
    assert um.usage_frequency == 0
    assert um.last_accessed_at is None

    t1 = datetime.utcnow() - timedelta(minutes=5)
    t2 = datetime.utcnow()

    um.access_history.append(UsageAccessRecord(user_or_service="u1", operation_type="read", accessed_at=t1))
    # Re-validate or create new model to trigger validators in Pydantic v1/v2
    # For Pydantic V1, validators on mutable fields like lists are tricky.
    # The model's validators run on __init__ or model_validate.
    # Simple list append won't trigger it.
    # The current UsageMetadata validators are `always=True`, so they run on init/validate.
    # To test this properly, one might do:
    um_revalidated = UsageMetadata.model_validate(um.model_dump())

    assert um_revalidated.usage_frequency == 1
    assert um_revalidated.last_accessed_at == t1

    um_revalidated.access_history.append(UsageAccessRecord(user_or_service="u2", operation_type="write", accessed_at=t2))
    um_final = UsageMetadata.model_validate(um_revalidated.model_dump())

    assert um_final.usage_frequency == 2
    assert um_final.last_accessed_at == t2

def test_usage_metadata_invalid_frequency():
    with pytest.raises(ValidationError):
        UsageMetadata(tensor_id=uuid4(), usage_frequency=-1)


# --- Original SemanticMetadata Tests (ensure they still pass) ---
def test_semantic_metadata_valid():
    tensor_uuid = uuid4()
    sm = SemanticMetadata(
        name="image_class_label",
        description="Describes the primary class identified in the image tensor.",
        tensor_id=tensor_uuid
    )
    assert sm.name == "image_class_label"
    assert sm.tensor_id == tensor_uuid

def test_semantic_metadata_empty_name():
    with pytest.raises(ValidationError, match="Name and description fields cannot be empty or just whitespace."):
        SemanticMetadata(name="", description="A description", tensor_id=uuid4())

# (Add other existing tests for SemanticMetadata and TensorDescriptor if they were modified)
# This file focuses primarily on the *new* extended schemas and TD enhancements.
# The `test_tensor_descriptor_valid` and `test_tensor_descriptor_defaults` from previous phase
# might need updates if defaults or required fields in TensorDescriptor changed significantly.
# The provided solution for schemas.py already enhanced TensorDescriptor, so a new valid case was added.
# The `test_tensor_descriptor_defaults` from Phase 1 would need to be checked against new defaults.
# For example, `tags` and `metadata` in TensorDescriptor now default to empty list/dict.

def test_tensor_descriptor_new_defaults():
    td = TensorDescriptor(
        dimensionality=1,
        shape=[10],
        data_type=DataType.INT32,
        owner="test_user_defaults",
        byte_size=40,
    )
    assert td.tags == [] # New default
    assert td.metadata == {} # New default
    assert td.checksum is None
    assert td.access_control.owner_permissions is None # New field in AccessControl
    assert td.storage_format == StorageFormat.RAW # Existing default
    assert isinstance(td.tensor_id, UUID)

# Ensure enum extensions are usable
def test_extended_enums_in_tensor_descriptor():
    td = TensorDescriptor(
        dimensionality=1, shape=[1], data_type=DataType.FLOAT16, # New DataType member
        storage_format=StorageFormat.HDF5, # New StorageFormat member
        owner="test", byte_size=2
    )
    assert td.data_type == DataType.FLOAT16
    assert td.storage_format == StorageFormat.HDF5
# Minimal set of tests for brevity in this combined file. More exhaustive tests per class are typical.
