import pytest
from uuid import uuid4, UUID
from datetime import datetime

from tensorus.metadata.schemas import TensorDescriptor, SemanticMetadata, DataType, AccessControl
from tensorus.metadata.storage import InMemoryStorage, storage_instance as global_storage_instance

# Fixture to provide a clean InMemoryStorage instance for each test
@pytest.fixture
def mem_storage():
    # Use a fresh instance for each test to ensure isolation
    storage = InMemoryStorage()
    storage.clear_all_data() # Ensure it's empty
    return storage

# Fixture for a sample TensorDescriptor
@pytest.fixture
def sample_td(mem_storage: InMemoryStorage): # Depends on mem_storage to ensure it's added to the test-local storage
    td = TensorDescriptor(
        tensor_id=uuid4(),
        dimensionality=2,
        shape=[10, 20],
        data_type=DataType.FLOAT32,
        owner="test_owner",
        byte_size=800
    )
    # mem_storage.add_tensor_descriptor(td) # Let tests decide when to add
    return td

# Fixture for a sample SemanticMetadata
@pytest.fixture
def sample_sm(sample_td: TensorDescriptor): # Depends on sample_td for tensor_id
    sm = SemanticMetadata(
        name="test_semantic_data",
        description="A piece of semantic info",
        tensor_id=sample_td.tensor_id
    )
    return sm

# --- TensorDescriptor Storage Tests ---

def test_add_and_get_tensor_descriptor(mem_storage: InMemoryStorage, sample_td: TensorDescriptor):
    mem_storage.add_tensor_descriptor(sample_td)
    retrieved_td = mem_storage.get_tensor_descriptor(sample_td.tensor_id)
    assert retrieved_td is not None
    assert retrieved_td.tensor_id == sample_td.tensor_id
    assert retrieved_td.owner == "test_owner"

def test_add_tensor_descriptor_duplicate_id(mem_storage: InMemoryStorage, sample_td: TensorDescriptor):
    mem_storage.add_tensor_descriptor(sample_td)
    with pytest.raises(ValueError, match="already exists"):
        mem_storage.add_tensor_descriptor(sample_td)

def test_get_tensor_descriptor_not_found(mem_storage: InMemoryStorage):
    assert mem_storage.get_tensor_descriptor(uuid4()) is None

def test_update_tensor_descriptor(mem_storage: InMemoryStorage, sample_td: TensorDescriptor):
    mem_storage.add_tensor_descriptor(sample_td)
    update_data = {"owner": "new_owner", "tags": ["updated"]}

    updated_td = mem_storage.update_tensor_descriptor(sample_td.tensor_id, **update_data)
    assert updated_td is not None
    assert updated_td.owner == "new_owner"
    assert updated_td.tags == ["updated"]
    assert updated_td.last_modified_timestamp > sample_td.last_modified_timestamp

    # Check if it's actually updated in storage
    retrieved_td = mem_storage.get_tensor_descriptor(sample_td.tensor_id)
    assert retrieved_td.owner == "new_owner"

def test_update_tensor_descriptor_partial(mem_storage: InMemoryStorage, sample_td: TensorDescriptor):
    mem_storage.add_tensor_descriptor(sample_td)
    original_shape = sample_td.shape
    update_data = {"owner": "partial_owner"}
    updated_td = mem_storage.update_tensor_descriptor(sample_td.tensor_id, **update_data)
    assert updated_td is not None
    assert updated_td.owner == "partial_owner"
    assert updated_td.shape == original_shape # Unchanged field

def test_update_tensor_descriptor_not_found(mem_storage: InMemoryStorage):
    assert mem_storage.update_tensor_descriptor(uuid4(), owner="ghost_owner") is None

def test_update_tensor_descriptor_invalid_field(mem_storage: InMemoryStorage, sample_td: TensorDescriptor):
    mem_storage.add_tensor_descriptor(sample_td)
    # Pydantic model in schema should raise error for invalid data types on update
    # e.g. if byte_size was set to a string.
    # The current update_tensor_descriptor directly uses setattr, so Pydantic validation
    # on the model fields will apply if the schema's field types are violated.
    # Example: Trying to set a field that doesn't exist (should go to metadata dict)
    updated_td = mem_storage.update_tensor_descriptor(sample_td.tensor_id, non_existent_field="value")
    assert updated_td.metadata["non_existent_field"] == "value"

    # Example: Trying to set a defined field with wrong type (Pydantic should catch this if types are enforced on setattr)
    # This depends on how Pydantic model handles direct setattr with wrong types after initialization.
    # Usually, Pydantic validates on __init__ and on model_validate().
    # For direct setattr, if `validate_assignment=True` is set in model config, it would re-validate.
    # The TensorDescriptor model does not have validate_assignment=True by default.
    # The current implementation will allow it, but descriptor.dict() might fail later if types are wrong.
    # For stricter type checking on update, the update method would need to call model_validate.
    # Let's test a case that our schema validator would catch, e.g. shape and dimensionality mismatch
    with pytest.raises(ValueError, match="Shape must have a length equal to dimensionality"):
         mem_storage.update_tensor_descriptor(sample_td.tensor_id, shape=[1,2,3], dimensionality=2)


def test_list_tensor_descriptors(mem_storage: InMemoryStorage, sample_td: TensorDescriptor):
    assert len(mem_storage.list_tensor_descriptors()) == 0
    mem_storage.add_tensor_descriptor(sample_td)

    td2 = TensorDescriptor(dimensionality=1, shape=[5], data_type=DataType.INT16, owner="user2", byte_size=10)
    mem_storage.add_tensor_descriptor(td2)

    descriptors = mem_storage.list_tensor_descriptors()
    assert len(descriptors) == 2
    tensor_ids = [td.tensor_id for td in descriptors]
    assert sample_td.tensor_id in tensor_ids
    assert td2.tensor_id in tensor_ids

def test_delete_tensor_descriptor(mem_storage: InMemoryStorage, sample_td: TensorDescriptor):
    mem_storage.add_tensor_descriptor(sample_td)
    assert mem_storage.delete_tensor_descriptor(sample_td.tensor_id) is True
    assert mem_storage.get_tensor_descriptor(sample_td.tensor_id) is None
    assert len(mem_storage.list_tensor_descriptors()) == 0

def test_delete_tensor_descriptor_not_found(mem_storage: InMemoryStorage):
    assert mem_storage.delete_tensor_descriptor(uuid4()) is False

# --- SemanticMetadata Storage Tests ---

def test_add_and_get_semantic_metadata(mem_storage: InMemoryStorage, sample_td: TensorDescriptor, sample_sm: SemanticMetadata):
    mem_storage.add_tensor_descriptor(sample_td) # Prerequisite
    mem_storage.add_semantic_metadata(sample_sm)

    retrieved_sms = mem_storage.get_semantic_metadata(sample_td.tensor_id)
    assert len(retrieved_sms) == 1
    assert retrieved_sms[0].name == sample_sm.name
    assert retrieved_sms[0].tensor_id == sample_td.tensor_id

def test_add_semantic_metadata_tensor_not_found(mem_storage: InMemoryStorage, sample_sm: SemanticMetadata):
    with pytest.raises(ValueError, match="does not exist"):
        mem_storage.add_semantic_metadata(sample_sm) # sample_sm.tensor_id points to a TD not in storage

def test_get_semantic_metadata_empty(mem_storage: InMemoryStorage, sample_td: TensorDescriptor):
    mem_storage.add_tensor_descriptor(sample_td)
    assert mem_storage.get_semantic_metadata(sample_td.tensor_id) == [] # Should be empty list, not None

def test_get_semantic_metadata_by_name(mem_storage: InMemoryStorage, sample_td: TensorDescriptor, sample_sm: SemanticMetadata):
    mem_storage.add_tensor_descriptor(sample_td)
    mem_storage.add_semantic_metadata(sample_sm)

    sm2 = SemanticMetadata(name="other_name", description="desc2", tensor_id=sample_td.tensor_id)
    mem_storage.add_semantic_metadata(sm2)

    retrieved_sm = mem_storage.get_semantic_metadata_by_name(sample_td.tensor_id, sample_sm.name)
    assert retrieved_sm is not None
    assert retrieved_sm.name == sample_sm.name

    assert mem_storage.get_semantic_metadata_by_name(sample_td.tensor_id, "non_existent_name") is None
    assert mem_storage.get_semantic_metadata_by_name(uuid4(), "any_name") is None # Tensor ID not found

def test_update_semantic_metadata(mem_storage: InMemoryStorage, sample_td: TensorDescriptor, sample_sm: SemanticMetadata):
    mem_storage.add_tensor_descriptor(sample_td)
    mem_storage.add_semantic_metadata(sample_sm)

    updated_sm = mem_storage.update_semantic_metadata(sample_td.tensor_id, sample_sm.name, new_description="Updated Description")
    assert updated_sm is not None
    assert updated_sm.description == "Updated Description"

    retrieved_sm = mem_storage.get_semantic_metadata_by_name(sample_td.tensor_id, sample_sm.name)
    assert retrieved_sm.description == "Updated Description"

def test_update_semantic_metadata_not_found(mem_storage: InMemoryStorage, sample_td: TensorDescriptor):
    mem_storage.add_tensor_descriptor(sample_td)
    assert mem_storage.update_semantic_metadata(sample_td.tensor_id, "non_existent_name", "new_desc") is None
    assert mem_storage.update_semantic_metadata(uuid4(), "any_name", "new_desc") is None # Tensor ID not found

def test_delete_semantic_metadata(mem_storage: InMemoryStorage, sample_td: TensorDescriptor, sample_sm: SemanticMetadata):
    mem_storage.add_tensor_descriptor(sample_td)
    mem_storage.add_semantic_metadata(sample_sm)

    sm2 = SemanticMetadata(name="other_name", description="desc2", tensor_id=sample_td.tensor_id)
    mem_storage.add_semantic_metadata(sm2)

    assert mem_storage.delete_semantic_metadata(sample_td.tensor_id, sample_sm.name) is True
    assert mem_storage.get_semantic_metadata_by_name(sample_td.tensor_id, sample_sm.name) is None
    assert len(mem_storage.get_semantic_metadata(sample_td.tensor_id)) == 1 # sm2 should still be there

    assert mem_storage.delete_semantic_metadata(sample_td.tensor_id, "non_existent_name") is False

def test_delete_tensor_descriptor_cascades_semantic_metadata(mem_storage: InMemoryStorage, sample_td: TensorDescriptor, sample_sm: SemanticMetadata):
    mem_storage.add_tensor_descriptor(sample_td)
    mem_storage.add_semantic_metadata(sample_sm) # Linked to sample_td

    assert len(mem_storage.get_semantic_metadata(sample_td.tensor_id)) == 1

    mem_storage.delete_tensor_descriptor(sample_td.tensor_id)
    assert mem_storage.get_tensor_descriptor(sample_td.tensor_id) is None
    assert mem_storage.get_semantic_metadata(sample_td.tensor_id) == [] # Should be empty after cascade

def test_clear_all_data(mem_storage: InMemoryStorage, sample_td: TensorDescriptor, sample_sm: SemanticMetadata):
    mem_storage.add_tensor_descriptor(sample_td)
    mem_storage.add_semantic_metadata(sample_sm)
    assert len(mem_storage.list_tensor_descriptors()) == 1

    mem_storage.clear_all_data()
    assert len(mem_storage.list_tensor_descriptors()) == 0
    assert len(mem_storage.get_semantic_metadata(sample_td.tensor_id)) == 0

# Test to ensure global storage_instance is not affected by mem_storage fixture if tests were using it.
# This test relies on the fact that `mem_storage` creates a *new* instance.
def test_global_storage_instance_isolation(sample_td: TensorDescriptor):
    # This test doesn't use the mem_storage fixture that clears data
    # It uses the global_storage_instance
    initial_count = len(global_storage_instance.list_tensor_descriptors())

    # Add to global instance
    td_global = TensorDescriptor(dimensionality=1, shape=[1], data_type=DataType.BOOLEAN, owner="global", byte_size=1)
    global_storage_instance.add_tensor_descriptor(td_global)

    assert len(global_storage_instance.list_tensor_descriptors()) == initial_count + 1

    # Clean up by removing the added descriptor
    global_storage_instance.delete_tensor_descriptor(td_global.tensor_id)
    assert len(global_storage_instance.list_tensor_descriptors()) == initial_count

# This test is to ensure the mem_storage fixture is indeed clearing data for each test
def test_mem_storage_is_clear_for_new_test(mem_storage: InMemoryStorage):
    assert len(mem_storage.list_tensor_descriptors()) == 0
    td = TensorDescriptor(dimensionality=1, shape=[1], data_type=DataType.INT8, owner="local", byte_size=1)
    mem_storage.add_tensor_descriptor(td)
    assert len(mem_storage.list_tensor_descriptors()) == 1
    # This instance of mem_storage will be discarded after this test.
    # The next test using mem_storage will get a fresh, empty one.
