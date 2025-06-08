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
    CompressionInfo
)

# --- TensorDescriptor Tests ---

def test_tensor_descriptor_valid():
    now = datetime.utcnow()
    td = TensorDescriptor(
        tensor_id=uuid4(),
        dimensionality=2,
        shape=[100, 100],
        data_type=DataType.FLOAT32,
        storage_format=StorageFormat.RAW,
        creation_timestamp=now,
        last_modified_timestamp=now + timedelta(seconds=1),
        owner="test_user",
        access_control=AccessControl(read=["user1"], write=["owner"]),
        byte_size=40000,
        compression_info=CompressionInfo(algorithm="zlib", level=5),
        tags=["test", "image"],
        metadata={"source": "synthetic"}
    )
    assert td.dimensionality == 2
    assert td.shape == [100, 100]
    assert td.data_type == DataType.FLOAT32
    assert td.owner == "test_user"
    assert td.compression_info.level == 5

def test_tensor_descriptor_defaults():
    now = datetime.utcnow()
    td = TensorDescriptor(
        dimensionality=1,
        shape=[10],
        data_type=DataType.INT32,
        owner="test_user",
        byte_size=40,
        # creation_timestamp defaults to datetime.utcnow if not provided in Pydantic v1
        # For Pydantic v2, default_factory takes precedence
    )
    assert isinstance(td.tensor_id, UUID)
    assert td.storage_format == StorageFormat.RAW
    assert td.access_control.read == []
    assert td.access_control.write == []
    assert td.tags is None
    assert td.metadata is None
    # With default_factory, creation and last_modified might be very close,
    # ensure last_modified is not before creation.
    assert td.last_modified_timestamp >= td.creation_timestamp

def test_tensor_descriptor_invalid_tensor_id_type():
    with pytest.raises(ValidationError):
        TensorDescriptor(tensor_id="not-a-uuid", dimensionality=1, shape=[1], data_type=DataType.FLOAT32, owner="user", byte_size=4)

def test_tensor_descriptor_negative_dimensionality():
    with pytest.raises(ValidationError, match=r"dimensionality\s*\n\s*ensure this value is greater than or equal to 0"):
        TensorDescriptor(dimensionality=-1, shape=[], data_type=DataType.FLOAT32, owner="user", byte_size=0)

def test_tensor_descriptor_shape_mismatch_dimensionality():
    with pytest.raises(ValidationError, match="Shape must have a length equal to dimensionality"):
        TensorDescriptor(dimensionality=2, shape=[10], data_type=DataType.FLOAT32, owner="user", byte_size=40)

def test_tensor_descriptor_shape_negative_value():
    with pytest.raises(ValidationError, match="All dimensions in shape must be non-negative integers"):
        TensorDescriptor(dimensionality=1, shape=[-10], data_type=DataType.FLOAT32, owner="user", byte_size=40)

def test_tensor_descriptor_invalid_data_type_value():
    with pytest.raises(ValidationError, match=r"data_type\s*\n\s*value is not a valid enumeration member"):
        TensorDescriptor(dimensionality=1, shape=[10], data_type="float128", owner="user", byte_size=40)

def test_tensor_descriptor_invalid_storage_format_value():
    with pytest.raises(ValidationError, match=r"storage_format\s*\n\s*value is not a valid enumeration member"):
        TensorDescriptor(dimensionality=1, shape=[10], data_type=DataType.FLOAT32, storage_format="weird_format", owner="user", byte_size=10)

def test_tensor_descriptor_last_modified_before_created():
    now = datetime.utcnow()
    with pytest.raises(ValidationError, match="Last modified timestamp cannot be before creation timestamp"):
        TensorDescriptor(
            dimensionality=1, shape=[10], data_type=DataType.FLOAT32, owner="user", byte_size=40,
            creation_timestamp=now,
            last_modified_timestamp=now - timedelta(seconds=10)
        )

def test_tensor_descriptor_update_last_modified():
    td = TensorDescriptor(dimensionality=1, shape=[10], data_type=DataType.INT64, owner="user", byte_size=80)
    old_ts = td.last_modified_timestamp
    # Allow some time to pass for the test to be meaningful
    import time; time.sleep(0.001)
    td.update_last_modified()
    assert td.last_modified_timestamp > old_ts

def test_tensor_descriptor_invalid_nested_access_control():
    with pytest.raises(ValidationError): # Match specific field if possible, e.g. access_control.read
        TensorDescriptor(
            dimensionality=1, shape=[10], data_type=DataType.FLOAT32, owner="user", byte_size=40,
            access_control={"read": "not-a-list"} # This should cause error in AccessControl model
        )

def test_tensor_descriptor_invalid_nested_compression_info():
     with pytest.raises(ValidationError): # Match specific field, e.g. compression_info.level
        TensorDescriptor(
            dimensionality=1, shape=[10], data_type=DataType.FLOAT32, owner="user", byte_size=40,
            compression_info={"algorithm": "zip", "level": "high"} # level should be int
        )

def test_tensor_descriptor_byte_size_negative():
    with pytest.raises(ValidationError, match=r"byte_size\s*\n\s*ensure this value is greater than or equal to 0"):
        TensorDescriptor(dimensionality=1, shape=[1], data_type=DataType.FLOAT32, owner="u", byte_size=-100)


# --- SemanticMetadata Tests ---

def test_semantic_metadata_valid():
    tensor_uuid = uuid4()
    sm = SemanticMetadata(
        name="image_class_label",
        description="Describes the primary class identified in the image tensor.",
        tensor_id=tensor_uuid
    )
    assert sm.name == "image_class_label"
    assert sm.description == "Describes the primary class identified in the image tensor."
    assert sm.tensor_id == tensor_uuid

def test_semantic_metadata_empty_name():
    with pytest.raises(ValidationError, match="Name and description cannot be empty"):
        SemanticMetadata(name="", description="A description", tensor_id=uuid4())

def test_semantic_metadata_whitespace_name():
    with pytest.raises(ValidationError, match="Name and description cannot be empty"):
        SemanticMetadata(name="   ", description="A description", tensor_id=uuid4())

def test_semantic_metadata_empty_description():
    with pytest.raises(ValidationError, match="Name and description cannot be empty"):
        SemanticMetadata(name="A name", description="", tensor_id=uuid4())

def test_semantic_metadata_whitespace_description():
    with pytest.raises(ValidationError, match="Name and description cannot be empty"):
        SemanticMetadata(name="A name", description="   ", tensor_id=uuid4())

def test_semantic_metadata_invalid_tensor_id_type():
    with pytest.raises(ValidationError): # Default Pydantic error for UUID type mismatch
        SemanticMetadata(name="test", description="test desc", tensor_id="not-a-uuid")


def test_tensor_descriptor_dict_export():
    td = TensorDescriptor(dimensionality=1, shape=[10], data_type=DataType.INT64, owner="user", byte_size=80)
    # Pydantic v1 .dict() / Pydantic v2 .model_dump()
    td_dict = td.dict() if hasattr(td, 'dict') else td.model_dump()
    assert td_dict['dimensionality'] == 1
    assert td_dict['data_type'] == 'int64'
    assert str(td_dict['tensor_id']) == str(td.tensor_id)

def test_semantic_metadata_dict_export():
    uid = uuid4()
    sm = SemanticMetadata(name="label", description="desc", tensor_id=uid)
    sm_dict = sm.dict() if hasattr(sm, 'dict') else sm.model_dump()
    assert sm_dict['name'] == 'label'
    assert str(sm_dict['tensor_id']) == str(uid)

def test_tensor_descriptor_zero_dimensionality():
    td = TensorDescriptor(
        dimensionality=0,
        shape=[],
        data_type=DataType.FLOAT32,
        owner="test_user",
        byte_size=0
    )
    assert td.dimensionality == 0
    assert td.shape == []

def test_tensor_descriptor_zero_dimensionality_shape_mismatch():
    with pytest.raises(ValidationError, match="Shape must have a length equal to dimensionality"):
        TensorDescriptor(dimensionality=0, shape=[10], data_type=DataType.FLOAT32, owner="user", byte_size=40)

def test_access_control_defaults():
    ac = AccessControl()
    assert ac.read == []
    assert ac.write == []
    assert ac.delete == []

    ac_with_data = AccessControl(read=["user1"])
    assert ac_with_data.read == ["user1"]
    assert ac_with_data.write == []

def test_compression_info_optional_level():
    ci = CompressionInfo(algorithm="gzip")
    assert ci.algorithm == "gzip"
    assert ci.level is None

    ci_with_level = CompressionInfo(algorithm="zstd", level=3)
    assert ci_with_level.level == 3
# Removed the pytest.main call as it's not standard practice for test files.
# Pytest will discover and run tests via the command line.
