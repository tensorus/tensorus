from typing import Dict, Optional, List, Any
from uuid import UUID
from datetime import datetime, timedelta
import logging

from .schemas import (
    TensorDescriptor, SemanticMetadata,
    LineageMetadata, ComputationalMetadata, QualityMetadata,
    RelationalMetadata, UsageMetadata,
)
from .storage_abc import MetadataStorage
from .schemas_iodata import TensorusExportData, TensorusExportEntry
import copy

# Configure module level logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# In-memory storage using dictionaries
_tensor_descriptors: Dict[UUID, TensorDescriptor] = {}
_semantic_metadata: Dict[UUID, List[SemanticMetadata]] = {} # tensor_id -> list of semantic metadata

# New storage dictionaries for extended metadata (one-to-one with TensorDescriptor)
_lineage_metadata_store: Dict[UUID, LineageMetadata] = {}
_computational_metadata_store: Dict[UUID, ComputationalMetadata] = {}
_quality_metadata_store: Dict[UUID, QualityMetadata] = {}
_relational_metadata_store: Dict[UUID, RelationalMetadata] = {}
_usage_metadata_store: Dict[UUID, UsageMetadata] = {}


class InMemoryStorage(MetadataStorage): # Inherit from MetadataStorage
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
            # Create a new model instance with updated data to trigger Pydantic validation
            try:
                # Get current data as dict
                current_data = descriptor.model_dump()
                # Apply updates
                updated_data = {**current_data, **kwargs}

                # Ensure tensor_id is not accidentally changed if present in kwargs
                if 'tensor_id' in kwargs and UUID(kwargs['tensor_id']) != tensor_id:
                    raise ValueError("Cannot change tensor_id during update.")
                updated_data['tensor_id'] = tensor_id # Ensure it's the correct UUID object

                # Re-validate by creating a new model instance
                # This assumes Pydantic v1. For v2, model_validate(updated_data) is preferred on existing instance
                # or MyModel(**updated_data) for new.
                # If using Pydantic V1:
                new_descriptor = TensorDescriptor(**updated_data)
                new_descriptor.update_last_modified() # Ensure timestamp is updated
                _tensor_descriptors[tensor_id] = new_descriptor
                return new_descriptor
            except Exception as e: # Catch Pydantic ValidationError or other issues
                 # This should ideally be ValidationError from Pydantic
                raise ValueError(f"Update failed validation: {e}")
        return None

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """Lists all TensorDescriptors."""
        return list(_tensor_descriptors.values())

    def delete_tensor_descriptor(self, tensor_id: UUID) -> bool:
        """Deletes a TensorDescriptor and ALL its associated metadata (semantic and extended)."""
        if tensor_id in _tensor_descriptors:
            del _tensor_descriptors[tensor_id]

            # Delete associated semantic metadata
            if tensor_id in _semantic_metadata:
                del _semantic_metadata[tensor_id]

            # Delete associated extended metadata
            if tensor_id in _lineage_metadata_store:
                del _lineage_metadata_store[tensor_id]
            if tensor_id in _computational_metadata_store:
                del _computational_metadata_store[tensor_id]
            if tensor_id in _quality_metadata_store:
                del _quality_metadata_store[tensor_id]
            if tensor_id in _relational_metadata_store:
                del _relational_metadata_store[tensor_id]
            if tensor_id in _usage_metadata_store:
                del _usage_metadata_store[tensor_id]
            return True
        return False

    # --- SemanticMetadata Methods ---
    def add_semantic_metadata(self, metadata: SemanticMetadata) -> None:
        """Adds SemanticMetadata to the in-memory store, linked to a TensorDescriptor."""
        if metadata.tensor_id not in _tensor_descriptors:
            raise ValueError(f"Cannot add semantic metadata. TensorDescriptor with ID {metadata.tensor_id} does not exist.")

        if metadata.tensor_id not in _semantic_metadata:
            _semantic_metadata[metadata.tensor_id] = []

        # Check for duplicate name for the same tensor_id
        for existing_meta in _semantic_metadata[metadata.tensor_id]:
            if existing_meta.name == metadata.name:
                raise ValueError(f"SemanticMetadata with name '{metadata.name}' already exists for tensor {metadata.tensor_id}")

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

    def update_semantic_metadata(self, tensor_id: UUID, name: str, new_description: Optional[str] = None, new_name: Optional[str] = None) -> Optional[SemanticMetadata]:
        """Updates a specific SemanticMetadata. Allows changing name and/or description."""
        metadata_list = self.get_semantic_metadata(tensor_id)
        item_to_update = None
        for item in metadata_list:
            if item.name == name:
                item_to_update = item
                break

        if not item_to_update:
            return None

        if new_name is not None and new_name != name:
            # Check if new name already exists
            for item in metadata_list:
                if item.name == new_name:
                    raise ValueError(f"SemanticMetadata with name '{new_name}' already exists for tensor {tensor_id}.")
            item_to_update.name = new_name

        if new_description is not None:
            item_to_update.description = new_description

        # Re-validate (Pydantic doesn't auto-validate on field change for v1 without config)
        # This is a simple approach; a more robust one might involve creating a new validated instance.
        try:
            SemanticMetadata(name=item_to_update.name, description=item_to_update.description, tensor_id=item_to_update.tensor_id)
        except Exception as e: # Should be ValidationError
            # Revert changes if validation fails (crude rollback)
            # This part is tricky and highlights the need for transactional updates or better validation handling.
            # For now, we raise the error. A real system might need more robust handling.
            raise ValueError(f"Update failed validation: {e}")

        return item_to_update


    def delete_semantic_metadata(self, tensor_id: UUID, name: str) -> bool:
        """Deletes a specific SemanticMetadata by name for a given TensorDescriptor ID."""
        metadata_list = self.get_semantic_metadata(tensor_id)
        initial_len = len(metadata_list)

        _semantic_metadata[tensor_id] = [item for item in metadata_list if item.name != name]

        return len(_semantic_metadata[tensor_id]) < initial_len

    # --- Generic CRUD for Extended Metadata ---
    def _add_extended_metadata(self, store: Dict[UUID, Any], metadata_item: Any) -> None:
        if not hasattr(metadata_item, 'tensor_id'):
            raise ValueError("Metadata item must have a 'tensor_id' attribute.")
        tensor_id = metadata_item.tensor_id # type: ignore

        if tensor_id not in _tensor_descriptors:
            raise ValueError(f"Cannot add or update {metadata_item.__class__.__name__}. Corresponding TensorDescriptor with ID {tensor_id} does not exist.") # type: ignore

        # Upsert behavior: if it exists, update it (by replacing); otherwise, add it.
        # The actual update logic (merging fields for partial updates) is handled by _update_extended_metadata.
        # This _add_extended_metadata now effectively becomes an "add or replace entirely".
        # For POST to be a true upsert (create or full replace), this is fine.
        # If POST was create-only, we'd keep the "if tensor_id in store: raise ValueError".
        # Let's make it an upsert by replacing.
        store[tensor_id] = metadata_item # type: ignore

    def _get_extended_metadata(self, store: Dict[UUID, Any], tensor_id: UUID) -> Optional[Any]:
        return store.get(tensor_id)

    def _update_extended_metadata(self, store: Dict[UUID, Any], tensor_id: UUID, metadata_class: type, **kwargs) -> Optional[Any]:
        existing_metadata = store.get(tensor_id)
        if not existing_metadata:
            return None

        current_data = existing_metadata.model_dump()
        updated_data = {**current_data, **kwargs}

        if 'tensor_id' in kwargs and UUID(kwargs['tensor_id']) != tensor_id:
            raise ValueError("Cannot change tensor_id during update.")
        updated_data['tensor_id'] = tensor_id

        try:
            # Re-validate by creating new instance
            # This is for Pydantic v1. For v2, model_validate(updated_data) or MyModel(**updated_data)
            new_metadata = metadata_class(**updated_data)
            store[tensor_id] = new_metadata
            return new_metadata
        except Exception as e: # Should be ValidationError
            raise ValueError(f"Update for {metadata_class.__name__} failed validation: {e}")

    def _delete_extended_metadata(self, store: Dict[UUID, Any], tensor_id: UUID) -> bool:
        if tensor_id in store:
            del store[tensor_id]
            return True
        return False

    # LineageMetadata methods
    def add_lineage_metadata(self, metadata: LineageMetadata) -> None:
        self._add_extended_metadata(_lineage_metadata_store, metadata)
    def get_lineage_metadata(self, tensor_id: UUID) -> Optional[LineageMetadata]:
        return self._get_extended_metadata(_lineage_metadata_store, tensor_id)
    def update_lineage_metadata(self, tensor_id: UUID, **kwargs) -> Optional[LineageMetadata]:
        return self._update_extended_metadata(_lineage_metadata_store, tensor_id, LineageMetadata, **kwargs)
    def delete_lineage_metadata(self, tensor_id: UUID) -> bool:
        return self._delete_extended_metadata(_lineage_metadata_store, tensor_id)

    # ComputationalMetadata methods
    def add_computational_metadata(self, metadata: ComputationalMetadata) -> None:
        self._add_extended_metadata(_computational_metadata_store, metadata)
    def get_computational_metadata(self, tensor_id: UUID) -> Optional[ComputationalMetadata]:
        return self._get_extended_metadata(_computational_metadata_store, tensor_id)
    def update_computational_metadata(self, tensor_id: UUID, **kwargs) -> Optional[ComputationalMetadata]:
        return self._update_extended_metadata(_computational_metadata_store, tensor_id, ComputationalMetadata, **kwargs)
    def delete_computational_metadata(self, tensor_id: UUID) -> bool:
        return self._delete_extended_metadata(_computational_metadata_store, tensor_id)

    # QualityMetadata methods
    def add_quality_metadata(self, metadata: QualityMetadata) -> None:
        self._add_extended_metadata(_quality_metadata_store, metadata)
    def get_quality_metadata(self, tensor_id: UUID) -> Optional[QualityMetadata]:
        return self._get_extended_metadata(_quality_metadata_store, tensor_id)
    def update_quality_metadata(self, tensor_id: UUID, **kwargs) -> Optional[QualityMetadata]:
        return self._update_extended_metadata(_quality_metadata_store, tensor_id, QualityMetadata, **kwargs) # Corrected self to self._update_extended_metadata
    def delete_quality_metadata(self, tensor_id: UUID) -> bool:
        return self._delete_extended_metadata(_quality_metadata_store, tensor_id)

    # RelationalMetadata methods
    def add_relational_metadata(self, metadata: RelationalMetadata) -> None:
        self._add_extended_metadata(_relational_metadata_store, metadata)
    def get_relational_metadata(self, tensor_id: UUID) -> Optional[RelationalMetadata]:
        return self._get_extended_metadata(_relational_metadata_store, tensor_id)
    def update_relational_metadata(self, tensor_id: UUID, **kwargs) -> Optional[RelationalMetadata]:
        return self._update_extended_metadata(_relational_metadata_store, tensor_id, RelationalMetadata, **kwargs)
    def delete_relational_metadata(self, tensor_id: UUID) -> bool:
        return self._delete_extended_metadata(_relational_metadata_store, tensor_id)

    # UsageMetadata methods
    def add_usage_metadata(self, metadata: UsageMetadata) -> None:
        self._add_extended_metadata(_usage_metadata_store, metadata)
    def get_usage_metadata(self, tensor_id: UUID) -> Optional[UsageMetadata]:
        return self._get_extended_metadata(_usage_metadata_store, tensor_id)
    def update_usage_metadata(self, tensor_id: UUID, **kwargs) -> Optional[UsageMetadata]:
        # Special handling for UsageMetadata to re-trigger validators for derived fields
        # This method specifically needs to handle the re-validation logic of UsageMetadata's
        # derived fields (last_accessed_at, usage_frequency) if access_history is part of kwargs.
        existing_metadata = _usage_metadata_store.get(tensor_id)
        if not existing_metadata:
            return None

        current_data = existing_metadata.model_dump()

        # If 'access_history' is part of kwargs, it means we are modifying it.
        # The Pydantic model for UsageMetadata has validators that derive other fields from access_history.
        # We need to ensure these validators run on the new data.

        updated_fields = {**kwargs} # Fields explicitly passed for update

        # Merge explicitly passed fields with current data
        # This ensures that if only a sub-field of a dict field (like 'purpose') is updated,
        # other sub-fields are not lost. However, for list fields like 'access_history',
        # a PATCH usually means replacing the entire list if 'access_history' is in kwargs.
        # If 'access_history' itself is not in kwargs, it remains unchanged.

        final_data_for_validation = copy.deepcopy(current_data)

        for key, value in updated_fields.items():
            if key == "access_history" and isinstance(value, list):
                # Ensure items in access_history are dicts for Pydantic validation
                final_data_for_validation[key] = [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in value
                ]
            else:
                final_data_for_validation[key] = value

        if 'tensor_id' in final_data_for_validation:
            existing_id = final_data_for_validation['tensor_id']
            existing_uuid = existing_id if isinstance(existing_id, UUID) else UUID(str(existing_id))
            if existing_uuid != tensor_id:
                raise ValueError("Cannot change tensor_id during update.")
        final_data_for_validation['tensor_id'] = tensor_id # Ensure it's the correct UUID object

        try:
            # Create a new UsageMetadata instance to trigger all validations, including derived fields.
            new_metadata = UsageMetadata(**final_data_for_validation)
            _usage_metadata_store[tensor_id] = new_metadata
            return new_metadata
        except Exception as e: # Should be ValidationError
            raise ValueError(f"Update for UsageMetadata failed validation: {e}")

    def delete_usage_metadata(self, tensor_id: UUID) -> bool:
        return self._delete_extended_metadata(_usage_metadata_store, tensor_id)

    def clear_all_data(self) -> None:
        # The 'copy' import was added in the previous step for the UsageMetadata update.
        # It is not strictly needed here but good to keep if other deepcopy operations are added.
        """Clears all data from all in-memory stores. Useful for testing."""
        _tensor_descriptors.clear()
        _semantic_metadata.clear()
        _lineage_metadata_store.clear()
        _computational_metadata_store.clear()
        _quality_metadata_store.clear()
        _relational_metadata_store.clear()
        _usage_metadata_store.clear()

    # --- Search and Aggregation Methods ---
    def _get_value_from_path(self, item: Any, path: str) -> Any:
        """Helper to get a value from an item (dict or Pydantic model) using a dot-separated path."""
        if not path:
            return item

        current = item
        for part in path.split('.'):
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part): # Pydantic model
                current = getattr(current, part)
            else:
                return None # Path does not exist
            if current is None:
                return None
        return current

    def search_tensor_descriptors(self, text_query: str, fields: List[str]) -> List[TensorDescriptor]:
        """Performs case-insensitive substring matching for text_query in specified fields."""
        if not text_query or not fields:
            return []

        results: List[TensorDescriptor] = []
        query_lower = text_query.lower()

        for td_id, td in _tensor_descriptors.items():
            matched = False
            for field_path in fields:
                # Split path, first part is likely the metadata type or 'tensor_descriptor'
                path_parts = field_path.split('.', 1)
                target_item = None
                actual_path_to_value = ""

                if path_parts[0] == "tensor_descriptor" or not len(path_parts) > 1: # Direct field on TD
                    target_item = td
                    actual_path_to_value = path_parts[0] if len(path_parts) == 1 else path_parts[1]
                elif path_parts[0] == "semantic":
                    # Semantic metadata is a list, search all entries
                    semantic_items = self.get_semantic_metadata(td_id)
                    for sm_item in semantic_items:
                        if len(path_parts) > 1:
                            value = self._get_value_from_path(sm_item, path_parts[1])
                            if value and query_lower in str(value).lower():
                                matched = True; break
                    if matched: break; continue # Found in one of the semantic items
                elif path_parts[0] == "lineage": target_item = self.get_lineage_metadata(td_id)
                elif path_parts[0] == "computational": target_item = self.get_computational_metadata(td_id)
                elif path_parts[0] == "quality": target_item = self.get_quality_metadata(td_id)
                elif path_parts[0] == "relational": target_item = self.get_relational_metadata(td_id)
                elif path_parts[0] == "usage": target_item = self.get_usage_metadata(td_id)
                else: # Assume it's a direct field of TensorDescriptor if not a known prefix
                    target_item = td
                    actual_path_to_value = field_path

                if target_item and not matched: # if not matched from semantic list
                    if not actual_path_to_value and len(path_parts) > 1: # if path_parts[0] was a metadata type
                         actual_path_to_value = path_parts[1]

                    value = self._get_value_from_path(target_item, actual_path_to_value)

                    if value is not None:
                        if isinstance(value, list) and all(isinstance(elem, str) for elem in value): # e.g. tags
                            if any(query_lower in str(elem).lower() for elem in value):
                                matched = True; break
                        elif isinstance(value, dict): # e.g. metadata on TD
                             if any(query_lower in str(v_item).lower() for v_item in value.values()):
                                matched = True; break
                        elif query_lower in str(value).lower():
                            matched = True; break

            if matched:
                results.append(td)
        return results

    def aggregate_tensor_descriptors(self, group_by_field: str, agg_function: str, agg_field: Optional[str] = None) -> Dict[Any, Any]:
        """Performs aggregation on tensor descriptors."""
        grouped_data: Dict[Any, List[Any]] = {}

        for td_id, td in _tensor_descriptors.items():
            # Determine the item and path for the group_by_field
            group_by_path_parts = group_by_field.split('.', 1)
            group_by_target_item = None
            actual_group_by_path = ""

            if group_by_path_parts[0] == "tensor_descriptor" or not len(group_by_path_parts) > 1:
                group_by_target_item = td
                actual_group_by_path = group_by_path_parts[0] if len(group_by_path_parts) == 1 else group_by_path_parts[1]
            elif group_by_path_parts[0] == "semantic":
                # For simplicity, if grouping by a semantic field, we might pick the first one or require specific semantic name.
                # This example will simplify: if semantic.name is used, it will use the name of the first semantic entry.
                # A more robust solution would require identifying *which* semantic metadata item's field to use.
                semantic_items = self.get_semantic_metadata(td_id)
                if semantic_items and len(group_by_path_parts) > 1: group_by_target_item = semantic_items[0]; actual_group_by_path = group_by_path_parts[1]
                elif semantic_items : group_by_target_item = semantic_items[0]; actual_group_by_path = "name" # Default to name if just "semantic"
                else: continue # No semantic data to group by
            elif group_by_path_parts[0] == "lineage": group_by_target_item = self.get_lineage_metadata(td_id)
            elif group_by_path_parts[0] == "computational": group_by_target_item = self.get_computational_metadata(td_id)
            elif group_by_path_parts[0] == "quality": group_by_target_item = self.get_quality_metadata(td_id)
            elif group_by_path_parts[0] == "relational": group_by_target_item = self.get_relational_metadata(td_id)
            elif group_by_path_parts[0] == "usage": group_by_target_item = self.get_usage_metadata(td_id)
            else: # Assume direct field of TensorDescriptor
                group_by_target_item = td
                actual_group_by_path = group_by_field

            if not group_by_target_item: continue
            if not actual_group_by_path and len(group_by_path_parts) > 1: actual_group_by_path = group_by_path_parts[1]

            group_key = self._get_value_from_path(group_by_target_item, actual_group_by_path)
            if group_key is None: group_key = "N/A" # Handle items where group key is missing
            if isinstance(group_key, list): group_key = tuple(group_key) # lists are unhashable

            # Determine item and path for aggregation field (if any)
            value_to_aggregate = None
            if agg_field:
                agg_field_path_parts = agg_field.split('.', 1)
                agg_target_item = None
                actual_agg_path = ""

                if agg_field_path_parts[0] == "tensor_descriptor" or not len(agg_field_path_parts) > 1:
                    agg_target_item = td
                    actual_agg_path = agg_field_path_parts[0] if len(agg_field_path_parts) == 1 else agg_field_path_parts[1]
                elif agg_field_path_parts[0] == "semantic":
                     semantic_items = self.get_semantic_metadata(td_id) # Simplified: take first
                     if semantic_items and len(agg_field_path_parts) > 1: agg_target_item = semantic_items[0]; actual_agg_path = agg_field_path_parts[1]
                elif agg_field_path_parts[0] == "lineage": agg_target_item = self.get_lineage_metadata(td_id)
                elif agg_field_path_parts[0] == "computational": agg_target_item = self.get_computational_metadata(td_id)
                # ... and so on for other metadata types ...
                else: # Assume direct field of TensorDescriptor
                    agg_target_item = td
                    actual_agg_path = agg_field

                if agg_target_item:
                    if not actual_agg_path and len(agg_field_path_parts) > 1: actual_agg_path = agg_field_path_parts[1]
                    value_to_aggregate = self._get_value_from_path(agg_target_item, actual_agg_path)

            if group_key not in grouped_data:
                grouped_data[group_key] = []

            # For count, we can just add 1 or any placeholder. For others, add the value.
            grouped_data[group_key].append(value_to_aggregate if agg_function != "count" else 1)

        # Perform aggregation
        result: Dict[Any, Any] = {}
        for key, values in grouped_data.items():
            if agg_function == "count":
                result[key] = len(values)
            elif agg_function == "sum":
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                result[key] = sum(numeric_values) if numeric_values else 0
            elif agg_function == "avg":
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                result[key] = sum(numeric_values) / len(numeric_values) if numeric_values else 0
            elif agg_function == "min":
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                result[key] = min(numeric_values) if numeric_values else None
            elif agg_function == "max":
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                result[key] = max(numeric_values) if numeric_values else None
            else:
                raise NotImplementedError(f"Aggregation function '{agg_function}' is not implemented.")
        return result

    # --- Versioning and Lineage Specific Methods ---
    def get_parent_tensor_ids(self, tensor_id: UUID) -> List[UUID]:
        """Retrieves parent tensor IDs from LineageMetadata.parent_tensors."""
        lineage_meta = self.get_lineage_metadata(tensor_id)
        if lineage_meta and lineage_meta.parent_tensors:
            return [parent.tensor_id for parent in lineage_meta.parent_tensors]
        return []

    def get_child_tensor_ids(self, tensor_id: UUID) -> List[UUID]:
        """
        Retrieves child tensor IDs by searching all other tensors' LineageMetadata
        to find which ones list the given tensor_id as a parent.
        This is inefficient for in-memory storage but functional.
        """
        child_ids: List[UUID] = []
        for other_td_id, lineage_meta in _lineage_metadata_store.items():
            if other_td_id == tensor_id: # Cannot be its own child in this context
                continue
            if lineage_meta and lineage_meta.parent_tensors:
                for parent_link in lineage_meta.parent_tensors:
                    if parent_link.tensor_id == tensor_id:
                        child_ids.append(other_td_id)
                        break # Found as parent, no need to check other parent_links for this other_td_id
        return child_ids

    # --- Export/Import Methods ---
    def get_export_data(self, tensor_ids: Optional[List[UUID]] = None) -> TensorusExportData:
        export_entries: List[TensorusExportEntry] = []

        ids_to_export = tensor_ids
        if ids_to_export is None: # Export all
            ids_to_export = list(_tensor_descriptors.keys())

        for td_id in ids_to_export:
            td = self.get_tensor_descriptor(td_id)
            if not td:
                continue # Skip if tensor descriptor not found for a given ID

            entry = TensorusExportEntry(
                tensor_descriptor=td,
                semantic_metadata=self.get_semantic_metadata(td_id), # Returns list
                lineage_metadata=self.get_lineage_metadata(td_id),
                computational_metadata=self.get_computational_metadata(td_id),
                quality_metadata=self.get_quality_metadata(td_id),
                relational_metadata=self.get_relational_metadata(td_id),
                usage_metadata=self.get_usage_metadata(td_id)
            )
            export_entries.append(entry)

        return TensorusExportData(entries=export_entries)

    def import_data(self, data: TensorusExportData, conflict_strategy: str = "skip") -> Dict[str, int]:
        if conflict_strategy not in ["skip", "overwrite"]:
            raise ValueError("Invalid conflict_strategy. Must be 'skip' or 'overwrite'.")

        summary = {"imported": 0, "skipped": 0, "overwritten": 0, "failed": 0}

        for entry in data.entries:
            td_id = entry.tensor_descriptor.tensor_id
            existing_td = self.get_tensor_descriptor(td_id)

            try:
                if existing_td:
                    if conflict_strategy == "skip":
                        summary["skipped"] += 1
                        continue
                    elif conflict_strategy == "overwrite":
                        # Delete existing tensor and all its metadata first
                        self.delete_tensor_descriptor(td_id)
                        # Note: delete_tensor_descriptor also clears associated extended metadata
                        summary["overwritten"] += 1

                # Add TensorDescriptor
                self.add_tensor_descriptor(entry.tensor_descriptor)

                # Add SemanticMetadata (list)
                if entry.semantic_metadata:
                    # Clear existing semantic metadata for this tensor_id if overwriting,
                    # or handle conflicts per item if semantic names are unique constraints.
                    # Current add_semantic_metadata appends or raises if name conflicts.
                    # For a clean import/overwrite, might need to delete existing semantic first.
                    if existing_td and conflict_strategy == "overwrite":
                         # Assuming _semantic_metadata is the dict storing List[SemanticMetadata]
                        if td_id in _semantic_metadata:
                            _semantic_metadata[td_id] = []

                    for sm_item in entry.semantic_metadata:
                        try: # Individual semantic items might conflict by name
                            self.add_semantic_metadata(sm_item)
                        except ValueError as e: # e.g. name conflict
                             # If overwrite, maybe update existing semantic entry by name?
                             # For now, just count as part of failure or skip.
                             logger.warning(
                                 f"Skipping semantic metadata '{sm_item.name}' for {td_id} due to: {e}"
                             )


                # Add other extended metadata (one-to-one)
                # add_..._metadata methods in InMemoryStorage now perform upsert.
                if entry.lineage_metadata: self.add_lineage_metadata(entry.lineage_metadata)
                if entry.computational_metadata: self.add_computational_metadata(entry.computational_metadata)
                if entry.quality_metadata: self.add_quality_metadata(entry.quality_metadata)
                if entry.relational_metadata: self.add_relational_metadata(entry.relational_metadata)
                if entry.usage_metadata: self.add_usage_metadata(entry.usage_metadata)

                if not existing_td or conflict_strategy != "overwrite": # Count as imported if new or not explicitly overwritten
                    summary["imported"] +=1

            except Exception as e:
                # Log error for this specific entry e.g. entry.tensor_descriptor.tensor_id
                logger.error(f"Failed to import entry for tensor_id {td_id}: {e}")
                summary["failed"] += 1

        return summary

    # --- Health and Metrics Methods ---
    def check_health(self) -> tuple[bool, str]:
        # InMemoryStorage is always healthy if it's running.
        return True, "in_memory"

    def get_tensor_descriptors_count(self) -> int:
        return len(_tensor_descriptors)

    def get_extended_metadata_count(self, metadata_model_name: str) -> int:
        # Map model name to its corresponding store dictionary
        store_map = {
            "LineageMetadata": _lineage_metadata_store,
            "ComputationalMetadata": _computational_metadata_store,
            "QualityMetadata": _quality_metadata_store,
            "RelationalMetadata": _relational_metadata_store,
            "UsageMetadata": _usage_metadata_store,
            "SemanticMetadata": _semantic_metadata # Note: Semantic is List per tensor_id
        }
        target_store = store_map.get(metadata_model_name)
        if target_store is not None:
            if metadata_model_name == "SemanticMetadata":
                # Sum of lengths of all lists in _semantic_metadata dictionary
                return sum(len(v_list) for v_list in target_store.values())
            return len(target_store)

        # Fallback or error for unknown model name
        # Could also count SemanticMetadata entries differently (e.g., unique tensor_ids having semantic data)
        # For now, if model name is not in map, return 0 or raise error.
        # Let's raise error for clarity if a specific known type is misspelled.
        # However, the ABC doesn't restrict model_name, so 0 is safer.
        # For this specific implementation, if it's not one of the above, it's not tracked this way.
        valid_model_names = ["LineageMetadata", "ComputationalMetadata", "QualityMetadata", "RelationalMetadata", "UsageMetadata", "SemanticMetadata"]
        if metadata_model_name not in valid_model_names:
            # This could be a programming error if an unexpected name is passed.
            # Or, if the method is meant to be robust to any string, return 0.
            # For now, let's be strict for known types and allow others to return 0.
             logger.warning(
                 f"get_extended_metadata_count called with unhandled model name '{metadata_model_name}' by InMemoryStorage."
             )

        return 0

    # --- Analytics Methods ---
    def get_co_occurring_tags(self, min_co_occurrence: int = 2, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        tag_pairs_count: Dict[tuple[str, str], int] = {}
        all_tags_set: set[str] = set()

        for td in _tensor_descriptors.values():
            if td.tags and len(td.tags) >= 2:
                sorted_tags = sorted(list(set(td.tags))) # Unique tags, sorted to make pairs canonical (tagA, tagB)
                all_tags_set.update(sorted_tags)
                for i in range(len(sorted_tags)):
                    for j in range(i + 1, len(sorted_tags)):
                        pair = (sorted_tags[i], sorted_tags[j])
                        tag_pairs_count[pair] = tag_pairs_count.get(pair, 0) + 1

        # Organize results by primary tag
        co_occurrence_map: Dict[str, List[Dict[str, Any]]] = {tag: [] for tag in all_tags_set}

        sorted_pairs = sorted(tag_pairs_count.items(), key=lambda item: item[1], reverse=True)

        for (tag1, tag2), count in sorted_pairs:
            if count >= min_co_occurrence:
                # Add to tag1's list
                if len(co_occurrence_map[tag1]) < limit: # Respect limit per primary tag
                    co_occurrence_map[tag1].append({"tag": tag2, "count": count})
                # Add to tag2's list
                if len(co_occurrence_map[tag2]) < limit: # Respect limit per primary tag
                     co_occurrence_map[tag2].append({"tag": tag1, "count": count})

        # Sort sub-lists by count
        for tag_key in co_occurrence_map:
            co_occurrence_map[tag_key].sort(key=lambda x: x["count"], reverse=True)
            # Ensure the overall limit isn't just per sub-list but rather total number of primary tags shown,
            # or total pairs. The current limit is per sub-list.
            # If a global limit on the number of primary tags returned is needed:
            # final_limited_map = {k: co_occurrence_map[k] for k in list(co_occurrence_map.keys())[:limit] if co_occurrence_map[k]}
            # return final_limited_map

        # Return only tags that have co-occurring partners satisfying the criteria
        return {k: v for k, v in co_occurrence_map.items() if v}


    def get_stale_tensors(self, threshold_days: int, limit: int = 100) -> List[TensorDescriptor]:
        stale_tensors: List[TensorDescriptor] = []
        threshold_datetime = datetime.utcnow() - timedelta(days=threshold_days)

        for td_id, td in _tensor_descriptors.items():
            last_relevant_timestamp = td.last_modified_timestamp # Default to last_modified

            usage_meta = self.get_usage_metadata(td_id)
            if usage_meta and usage_meta.last_accessed_at:
                last_relevant_timestamp = max(last_relevant_timestamp, usage_meta.last_accessed_at)

            if last_relevant_timestamp < threshold_datetime:
                stale_tensors.append(td)

        # Sort by the last_relevant_timestamp (oldest first) and apply limit
        stale_tensors.sort(key=lambda t: (
            max(t.last_modified_timestamp, self.get_usage_metadata(t.tensor_id).last_accessed_at if self.get_usage_metadata(t.tensor_id) and self.get_usage_metadata(t.tensor_id).last_accessed_at else datetime.min)
        ))
        return stale_tensors[:limit]

    def get_complex_tensors(self, min_parent_count: Optional[int] = None, min_transformation_steps: Optional[int] = None, limit: int = 100) -> List[TensorDescriptor]:
        if min_parent_count is None and min_transformation_steps is None:
            raise ValueError("At least one criterion (min_parent_count or min_transformation_steps) must be provided.")

        complex_tensors: List[TensorDescriptor] = []
        for td_id, td in _tensor_descriptors.items():
            is_complex = False
            lineage_meta = self.get_lineage_metadata(td_id)

            if lineage_meta:
                if min_parent_count is not None and len(lineage_meta.parent_tensors) >= min_parent_count:
                    is_complex = True
                if not is_complex and min_transformation_steps is not None and len(lineage_meta.transformation_history) >= min_transformation_steps:
                    is_complex = True

            if is_complex:
                complex_tensors.append(td)

            if len(complex_tensors) >= limit:
                break

        return complex_tensors
