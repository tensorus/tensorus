from typing import Dict, Optional, List
from uuid import UUID

from .schemas import TensorDescriptor, SemanticMetadata

# In-memory storage using dictionaries
_tensor_descriptors: Dict[UUID, TensorDescriptor] = {}
_semantic_metadata: Dict[UUID, List[SemanticMetadata]] = {} # tensor_id -> list of semantic metadata

class InMemoryStorage:
    def add_tensor_descriptor(self, descriptor: TensorDescriptor) -> None:
        """Adds a TensorDescriptor to the in-memory store."""
        if descriptor.tensor_id in _tensor_descriptors:
            raise ValueError(f"TensorDescriptor with ID {descriptor.tensor_id} already exists.")
        _tensor_descriptors[descriptor.tensor_id] = descriptor

    def get_tensor_descriptor(self, tensor_id: UUID) -> Optional[TensorDescriptor]:
        """Retrieves a TensorDescriptor from the in-memory store by its ID."""
        return _tensor_descriptors.get(tensor_id)

    def update_tensor_descriptor(self, tensor_id: UUID, **kwargs) -> Optional[TensorDescriptor]:
        """Updates an existing TensorDescriptor."""
        descriptor = self.get_tensor_descriptor(tensor_id)
        if descriptor:
            for key, value in kwargs.items():
                if hasattr(descriptor, key):
                    setattr(descriptor, key, value)
                else:
                    # Potentially update generic metadata if key is not a direct attribute
                    if descriptor.metadata is None:
                         descriptor.metadata = {}
                    descriptor.metadata[key] = value
            descriptor.update_last_modified() # Ensure timestamp is updated
            _tensor_descriptors[tensor_id] = descriptor # Re-assign to storage
            return descriptor
        return None

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """Lists all TensorDescriptors."""
        return list(_tensor_descriptors.values())

    def delete_tensor_descriptor(self, tensor_id: UUID) -> bool:
        """Deletes a TensorDescriptor and its associated SemanticMetadata."""
        if tensor_id in _tensor_descriptors:
            del _tensor_descriptors[tensor_id]
            if tensor_id in _semantic_metadata:
                del _semantic_metadata[tensor_id]
            return True
        return False

    def add_semantic_metadata(self, metadata: SemanticMetadata) -> None:
        """Adds SemanticMetadata to the in-memory store, linked to a TensorDescriptor."""
        if metadata.tensor_id not in _tensor_descriptors:
            raise ValueError(f"Cannot add semantic metadata. TensorDescriptor with ID {metadata.tensor_id} does not exist.")

        if metadata.tensor_id not in _semantic_metadata:
            _semantic_metadata[metadata.tensor_id] = []

        # Optional: Check for duplicate name/description for the same tensor_id if needed
        # for existing_meta in _semantic_metadata[metadata.tensor_id]:
        #     if existing_meta.name == metadata.name:
        #         raise ValueError(f"SemanticMetadata with name '{metadata.name}' already exists for tensor {metadata.tensor_id}")

        _semantic_metadata[metadata.tensor_id].append(metadata)

    def get_semantic_metadata(self, tensor_id: UUID) -> List[SemanticMetadata]:
        """Retrieves all SemanticMetadata associated with a given TensorDescriptor ID."""
        return _semantic_metadata.get(tensor_id, [])

    def get_semantic_metadata_by_name(self, tensor_id: UUID, name: str) -> Optional[SemanticMetadata]:
        """Retrieves a specific SemanticMetadata by name for a given TensorDescriptor ID."""
        for metadata_item in self.get_semantic_metadata(tensor_id):
            if metadata_item.name == name:
                return metadata_item
        return None

    def update_semantic_metadata(self, tensor_id: UUID, name: str, new_description: Optional[str] = None) -> Optional[SemanticMetadata]:
        """Updates the description of a specific SemanticMetadata."""
        for metadata_item in self.get_semantic_metadata(tensor_id):
            if metadata_item.name == name:
                if new_description is not None:
                    metadata_item.description = new_description
                # If other fields were mutable, they could be updated here
                return metadata_item
        return None

    def delete_semantic_metadata(self, tensor_id: UUID, name: str) -> bool:
        """Deletes a specific SemanticMetadata by name for a given TensorDescriptor ID."""
        metadata_list = self.get_semantic_metadata(tensor_id)
        initial_len = len(metadata_list)

        _semantic_metadata[tensor_id] = [item for item in metadata_list if item.name != name]

        return len(_semantic_metadata[tensor_id]) < initial_len

    def clear_all_data(self) -> None:
        """Clears all data from the in-memory storage. Useful for testing."""
        _tensor_descriptors.clear()
        _semantic_metadata.clear()

# Global instance of the storage
# This makes it easy to use across different parts of an application (if simple)
# For more complex scenarios, dependency injection or a more robust service locator pattern would be better.
storage_instance = InMemoryStorage()
