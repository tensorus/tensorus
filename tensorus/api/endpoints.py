from typing import List, Dict, Optional, Any
from uuid import UUID
from datetime import datetime # For time-based queries

from fastapi import APIRouter, HTTPException, Body, Query, Path
from pydantic import BaseModel, ValidationError

from tensorus.metadata.schemas import (
    TensorDescriptor, SemanticMetadata, DataType,
    LineageSourceType, LineageMetadata, ParentTensorLink,
    ComputationalMetadata, QualityMetadata, RelationalMetadata, UsageMetadata,
    TensorDescriptor # For response models
)
from tensorus.metadata.storage import storage_instance
from tensorus.storage.connectors import mock_tensor_connector_instance

# For Pydantic models used in PATCH requests, e.g. LineageMetadataUpdate
from pydantic import BaseModel as PydanticBaseModel # Alias to avoid conflict with schema.BaseModel if any
from typing import TypeVar, Generic # For generic PATCH request body

ModelType = TypeVar('ModelType', bound=PydanticBaseModel)

class PatchRequestBody(PydanticBaseModel, Generic[ModelType]):
    # This can be used if we want a generic way to define PATCH bodies,
    # but FastAPI handles dict directly for PATCH bodies which is often simpler.
    # For this task, we'll use Dict[str, Any] for PATCH bodies directly in endpoints.
    pass
import copy # For duplicating tensor descriptor data

# Router for TensorDescriptor
router_tensor_descriptor = APIRouter(
    prefix="/tensor_descriptors",
    tags=["Tensor Descriptors"],
    responses={404: {"description": "Not found"}},
)

# Router for SemanticMetadata
router_semantic_metadata = APIRouter(
    prefix="/semantic_metadata",
    tags=["Semantic Metadata"],
    responses={404: {"description": "Not found"}},
)

# --- TensorDescriptor Endpoints ---

@router_tensor_descriptor.post("/", response_model=TensorDescriptor, status_code=201,
                               summary="Create Tensor Descriptor",
                               description="Adds a new tensor descriptor to the metadata store. "
                                           "If `shape`, `data_type`, or `byte_size` are not provided "
                                           "and the tensor exists in storage, these details will be "
                                           "fetched from the mock tensor store.")
async def create_tensor_descriptor(descriptor_data: Dict[str, Any]):
    tensor_id = descriptor_data.get("tensor_id")
    if not tensor_id:
        # If tensor_id is not provided in the request, Pydantic will create one by default.
        # We need a UUID to check the mock store. Let Pydantic handle generation if fully new.
        # However, if we intend to link to an *existing* tensor in storage, tensor_id *must* be provided.
        pass # Allow Pydantic to generate if it's truly a new tensor not yet in storage

    # Attempt to fetch details from mock storage if some fields are missing
    # and a tensor_id was provided or generated that might exist in the mock store.
    # This logic assumes that if a tensor_id is given, it *might* already be in the mock tensor store.

    # Create a preliminary TensorDescriptor to get a valid tensor_id if not provided
    # This is a bit of a workaround due to how Pydantic default_factory for tensor_id works
    temp_id_for_lookup = UUID(tensor_id) if tensor_id else uuid.uuid4() # Use provided or generate one for lookup

    # Fields to potentially fetch from storage
    fields_to_fetch = ["shape", "data_type", "byte_size", "dimensionality"] # dimensionality added
    missing_fields = [field for field in fields_to_fetch if field not in descriptor_data or descriptor_data[field] is None]

    if missing_fields:
        print(f"Missing fields {missing_fields} for tensor {temp_id_for_lookup}, attempting to fetch from mock storage.")
        # Try to get details from the mock tensor storage
        # This implies the tensor data should already exist in the mock storage if we are to fetch details.
        # For this example, we assume `mock_tensor_connector_instance.store_tensor` might have been called separately,
        # or `get_tensor_details` can generate details for a known (even if not explicitly stored) tensor_id.
        storage_details = mock_tensor_connector_instance.get_tensor_details(temp_id_for_lookup)
        if storage_details:
            print(f"Fetched details from mock storage: {storage_details}")
            for field in missing_fields:
                if field in storage_details and (field not in descriptor_data or descriptor_data[field] is None) :
                    descriptor_data[field] = storage_details[field]

            # Special handling for dimensionality if shape is present
            if "shape" in descriptor_data and descriptor_data["shape"] is not None \
               and ("dimensionality" not in descriptor_data or descriptor_data["dimensionality"] is None):
                descriptor_data["dimensionality"] = len(descriptor_data["shape"])
        else:
            print(f"Could not fetch details for tensor {temp_id_for_lookup} from mock storage.")
            # If still missing after trying to fetch, Pydantic validation will catch it if mandatory

    try:
        # Now create the final TensorDescriptor with potentially auto-filled data
        final_descriptor = TensorDescriptor(**descriptor_data)
        storage_instance.add_tensor_descriptor(final_descriptor)

        # As a demonstration, if we just created a descriptor for a tensor,
        # let's ensure it's "stored" in the mock tensor storage if it wasn't already.
        # This is more for completing the loop in the mock scenario.
        if mock_tensor_connector_instance.retrieve_tensor(final_descriptor.tensor_id) is None:
            # Store some dummy data or the descriptor itself as placeholder
            mock_tensor_data_payload = {
                "shape": final_descriptor.shape,
                "data_type": final_descriptor.data_type.value,
                "byte_size": final_descriptor.byte_size,
                "info": "Placeholder data stored by create_tensor_descriptor endpoint"
            }
            mock_tensor_connector_instance.store_tensor(final_descriptor.tensor_id, mock_tensor_data_payload)
            print(f"Stored placeholder tensor data for {final_descriptor.tensor_id} in mock storage.")

        return final_descriptor
    except ValidationError as e: # Pydantic validation error
        raise HTTPException(status_code=422, detail=e.errors())
    except ValueError as e: # Custom validation errors from schemas or storage
        raise HTTPException(status_code=400, detail=str(e))


@router_tensor_descriptor.get("/", response_model=List[TensorDescriptor],
                              summary="List Tensor Descriptors with Advanced Filtering",
                              description="Lists all tensor descriptors, with optional filtering by core fields and extended metadata.")
async def list_tensor_descriptors(
    # Core TensorDescriptor fields
    owner: Optional[str] = Query(None, description="Filter by owner of the tensor."),
    data_type: Optional[DataType] = Query(None, description="Filter by data type of the tensor (e.g., float32, int64)."),
    tags_contain: Optional[List[str]] = Query(None, description="Filter by tensors containing ALL specified tags."),
    # LineageMetadata fields
    lineage_version: Optional[str] = Query(None, alias="lineage.version", description="Filter by lineage version."),
    lineage_source_type: Optional[LineageSourceType] = Query(None, alias="lineage.source.type", description="Filter by lineage source type."),
    # ComputationalMetadata fields
    comp_algorithm: Optional[str] = Query(None, alias="computational.algorithm", description="Filter by computational algorithm used."),
    comp_gpu_model: Optional[str] = Query(None, alias="computational.hardware_info.gpu_model", description="Filter by GPU model used in computation (example of nested query)."),
    # QualityMetadata fields
    quality_confidence_gt: Optional[float] = Query(None, alias="quality.confidence_score_gt", description="Filter by quality confidence score greater than specified value."),
    quality_noise_lt: Optional[float] = Query(None, alias="quality.noise_level_lt", description="Filter by quality noise level less than specified value."),
    # RelationalMetadata fields
    rel_collection: Optional[str] = Query(None, alias="relational.collection", description="Filter by tensors belonging to a specific collection."),
    rel_has_related_tensor_id: Optional[UUID] = Query(None, alias="relational.has_related_tensor_id", description="Filter by tensors that have a specific related tensor ID."),
    # UsageMetadata fields
    usage_last_accessed_before: Optional[datetime] = Query(None, alias="usage.last_accessed_before", description="Filter by tensors last accessed before this timestamp."),
    usage_used_by_app: Optional[str] = Query(None, alias="usage.used_by_app", description="Filter by tensors used by a specific application.")
):
    all_descriptors = storage_instance.list_tensor_descriptors()
    filtered_descriptors = []

    for desc in all_descriptors:
        # --- Core Field Filtering ---
        if owner is not None and desc.owner != owner:
            continue
        if data_type is not None and desc.data_type != data_type:
            continue
        if tags_contain is not None:
            if not desc.tags or not all(tag in desc.tags for tag in tags_contain):
                continue

        # --- Extended Metadata Filtering ---
        # For simplicity, fetching extended metadata for each descriptor.
        # In a real DB, this would be part of the query.

        # Lineage
        if lineage_version is not None or lineage_source_type is not None:
            lineage_meta = storage_instance.get_lineage_metadata(desc.tensor_id)
            if lineage_meta:
                if lineage_version is not None and lineage_meta.version != lineage_version:
                    continue
                if lineage_source_type is not None and (not lineage_meta.source or lineage_meta.source.type != lineage_source_type):
                    continue
            else: # If lineage filters are active, and no lineage meta, then it doesn't match
                continue

        # Computational
        if comp_algorithm is not None or comp_gpu_model is not None:
            comp_meta = storage_instance.get_computational_metadata(desc.tensor_id)
            if comp_meta:
                if comp_algorithm is not None and comp_meta.algorithm != comp_algorithm:
                    continue
                if comp_gpu_model is not None:
                    # Example for nested dict query:
                    gpu = comp_meta.hardware_info.get("gpu_model") if comp_meta.hardware_info else None
                    if gpu != comp_gpu_model:
                        continue
            else:
                continue

        # Quality
        if quality_confidence_gt is not None or quality_noise_lt is not None:
            qual_meta = storage_instance.get_quality_metadata(desc.tensor_id)
            if qual_meta:
                if quality_confidence_gt is not None and (qual_meta.confidence_score is None or qual_meta.confidence_score <= quality_confidence_gt):
                    continue
                if quality_noise_lt is not None and (qual_meta.noise_level is None or qual_meta.noise_level >= quality_noise_lt):
                    continue
            else:
                continue

        # Relational
        if rel_collection is not None or rel_has_related_tensor_id is not None:
            rel_meta = storage_instance.get_relational_metadata(desc.tensor_id)
            if rel_meta:
                if rel_collection is not None and rel_collection not in rel_meta.collections:
                    continue
                if rel_has_related_tensor_id is not None:
                    found_related = any(rt_link.related_tensor_id == rel_has_related_tensor_id for rt_link in rel_meta.related_tensors)
                    if not found_related:
                        continue
            else:
                continue

        # Usage
        if usage_last_accessed_before is not None or usage_used_by_app is not None:
            usage_meta = storage_instance.get_usage_metadata(desc.tensor_id)
            if usage_meta:
                if usage_last_accessed_before is not None and (usage_meta.last_accessed_at is None or usage_meta.last_accessed_at >= usage_last_accessed_before):
                    continue
                if usage_used_by_app is not None and usage_used_by_app not in usage_meta.application_references:
                    continue
            else: # If usage filters active, and no usage meta, then no match
                continue

        filtered_descriptors.append(desc)

    return filtered_descriptors

@router_tensor_descriptor.get("/{tensor_id}", response_model=TensorDescriptor,
                                summary="Get Tensor Descriptor by ID",
                                description="Retrieves a specific tensor descriptor by its unique ID.")
async def get_tensor_descriptor(tensor_id: UUID = Path(..., description="The UUID of the tensor descriptor.")): # type: ignore
    descriptor = storage_instance.get_tensor_descriptor(tensor_id)
    if descriptor is None:
        raise HTTPException(status_code=404, detail=f"TensorDescriptor with ID {tensor_id} not found.")
    return descriptor

# For PUT, Pydantic models are not directly used in FastAPI for partial updates in the way one might expect.
# We accept a dictionary and then apply the updates.
class TensorDescriptorUpdate(BaseModel):
    dimensionality: Optional[int] = None
    shape: Optional[List[int]] = None
    data_type: Optional[DataType] = None
    storage_format: Optional[str] = None # Assuming StorageFormat enum values are passed as strings
    owner: Optional[str] = None
    access_control: Optional[Dict[str, List[str]]] = None
    byte_size: Optional[int] = None
    compression_info: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, str]] = None

@router_tensor_descriptor.put("/{tensor_id}", response_model=TensorDescriptor,
                               summary="Update Tensor Descriptor",
                               description="Updates an existing tensor descriptor. Allows partial updates of fields.")
async def update_tensor_descriptor(
    tensor_id: UUID = Path(..., description="The UUID of the tensor descriptor to update."), # type: ignore
    updates: TensorDescriptorUpdate = Body(..., description="A dictionary containing the fields to update.") # type: ignore
):
    update_data = updates.dict(exclude_unset=True)

    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided.")

    current_descriptor = storage_instance.get_tensor_descriptor(tensor_id)
    if not current_descriptor:
        raise HTTPException(status_code=404, detail=f"TensorDescriptor with ID {tensor_id} not found for update.")

    try:
        updated_descriptor = storage_instance.update_tensor_descriptor(tensor_id, **update_data)
        if updated_descriptor is None:
             raise HTTPException(status_code=404, detail=f"TensorDescriptor with ID {tensor_id} failed to update.")
        return updated_descriptor
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router_tensor_descriptor.delete("/{tensor_id}", status_code=200,
                                 summary="Delete Tensor Descriptor",
                                 description="Deletes a tensor descriptor, its associated semantic metadata, and also attempts to delete the tensor data from the mock tensor store.")
async def delete_tensor_descriptor(tensor_id: UUID = Path(..., description="The UUID of the tensor descriptor to delete.")): # type: ignore
    metadata_deleted = storage_instance.delete_tensor_descriptor(tensor_id)
    if not metadata_deleted:
        raise HTTPException(status_code=404, detail=f"TensorDescriptor with ID {tensor_id} not found in metadata store.")

    tensor_data_deleted = mock_tensor_connector_instance.delete_tensor(tensor_id)
    if tensor_data_deleted: # This is true if deleted or not found in mock store (idempotent)
        return {"message": f"TensorDescriptor {tensor_id} and associated data deleted successfully from metadata store. Mock tensor store delete attempt finished."}
    # Fallback message if needed, but current mock_delete always returns True if no error
    return {"message": f"TensorDescriptor {tensor_id} deleted from metadata. Mock tensor store delete attempt finished."}


# --- SemanticMetadata Endpoints ---

@router_semantic_metadata.post("/", response_model=SemanticMetadata, status_code=201,
                                 summary="Create Semantic Metadata",
                                 description="Adds new semantic metadata linked to an existing TensorDescriptor.")
async def create_semantic_metadata(metadata: SemanticMetadata): # type: ignore
    if storage_instance.get_tensor_descriptor(metadata.tensor_id) is None:
        raise HTTPException(
            status_code=404,
            detail=f"TensorDescriptor with ID {metadata.tensor_id} not found. Cannot add semantic metadata."
        )
    try:
        storage_instance.add_semantic_metadata(metadata)
        return metadata
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())


@router_semantic_metadata.get("/{tensor_id}", response_model=List[SemanticMetadata],
                                summary="Get Semantic Metadata for a Tensor",
                                description="Retrieves all semantic metadata entries associated with a given TensorDescriptor ID.")
async def get_semantic_metadata_for_tensor(tensor_id: UUID = Path(..., description="The UUID of the tensor descriptor.")): # type: ignore
    if storage_instance.get_tensor_descriptor(tensor_id) is None:
         raise HTTPException(status_code=404, detail=f"TensorDescriptor with ID {tensor_id} not found, so no semantic metadata can be retrieved.")

    metadata_list = storage_instance.get_semantic_metadata(tensor_id)
    return metadata_list

class SemanticMetadataUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


@router_semantic_metadata.put("/{tensor_id}/{name}", response_model=SemanticMetadata,
                                 summary="Update Semantic Metadata",
                                 description="Updates the description and/or name of a specific semantic metadata entry. The entry is identified by the tensor ID and its current name.")
async def update_semantic_metadata_entry(
    tensor_id: UUID = Path(..., description="The UUID of the tensor descriptor."), # type: ignore
    name: str = Path(..., description="The current name of the semantic metadata entry to be updated."), # type: ignore
    updates: SemanticMetadataUpdate = Body(..., description="The fields to update. Both 'name' and 'description' are optional.") # type: ignore
):
    if updates.description is None and updates.name is None:
        raise HTTPException(status_code=400, detail="No update data provided for 'name' or 'description'.")

    if storage_instance.get_tensor_descriptor(tensor_id) is None:
        raise HTTPException(status_code=404, detail=f"TensorDescriptor with ID {tensor_id} not found.")

    try:
        updated_metadata = storage_instance.update_semantic_metadata(
            tensor_id,
            name,
            new_description=updates.description,
            new_name=updates.name
        )
    except ValueError as e: # Catch errors from storage, e.g. new name already exists
        raise HTTPException(status_code=400, detail=str(e))

    if updated_metadata is None:
        raise HTTPException(status_code=404, detail=f"SemanticMetadata with name '{name}' for tensor ID {tensor_id} not found.")
    return updated_metadata


@router_semantic_metadata.delete("/{tensor_id}/{name}", status_code=200,
                                  summary="Delete Specific Semantic Metadata",
                                  description="Deletes a specific semantic metadata entry identified by its name, for a given tensor ID.")
async def delete_specific_semantic_metadata(
    tensor_id: UUID = Path(..., description="The UUID of the tensor descriptor."), # type: ignore
    name: str = Path(..., description="The name of the semantic metadata entry to delete.") # type: ignore
):
    if storage_instance.get_tensor_descriptor(tensor_id) is None:
        raise HTTPException(status_code=404, detail=f"TensorDescriptor with ID {tensor_id} not found.")

    if not storage_instance.delete_semantic_metadata(tensor_id, name):
        raise HTTPException(status_code=404, detail=f"SemanticMetadata with name '{name}' for tensor ID {tensor_id} not found or already deleted.")
    return {"message": f"SemanticMetadata '{name}' for tensor {tensor_id} deleted successfully."}

# --- Search and Aggregation Routers ---
router_search_aggregate = APIRouter(
    tags=["Search & Aggregate"]
)

@router_search_aggregate.get("/search/tensors/", response_model=List[TensorDescriptor],
                             summary="Search Tensor Descriptors",
                             description="Performs a text-based search across specified fields of tensor descriptors and their associated metadata.")
async def search_tensors(
    text_query: str = Query(..., min_length=1, description="The text to search for."),
    fields_to_search: Optional[List[str]] = Query(None, description="List of fields to search (e.g., 'owner', 'tags', 'semantic.description', 'lineage.source.identifier'). If empty, searches a predefined set.")
):
    if not fields_to_search: # Default fields if none provided
        fields_to_search = [
            "tensor_id", "owner", "tags", "metadata", # From TensorDescriptor
            "semantic.name", "semantic.description", # From SemanticMetadata (assuming one per tensor for simplicity here, or search all)
            "lineage.source.identifier", "lineage.version", # From LineageMetadata
            "computational.algorithm" # From ComputationalMetadata
        ]
    try:
        results = storage_instance.search_tensor_descriptors(text_query, fields_to_search)
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router_search_aggregate.get("/aggregate/tensors/", response_model=Dict[str, Any],
                               summary="Aggregate Tensor Descriptors",
                               description="Performs aggregation on tensor descriptors based on a group-by field and an aggregation function.")
async def aggregate_tensors(
    group_by_field: str = Query(..., description="Field to group by (e.g., 'data_type', 'owner', 'semantic.name', 'lineage.source.type')."),
    agg_function: str = Query(..., description="Aggregation function to apply (e.g., 'count', 'avg', 'sum', 'min', 'max')."),
    agg_field: Optional[str] = Query(None, description="Field to aggregate for functions like 'avg', 'sum', 'min', 'max' (e.g., 'byte_size', 'computational.computation_time_seconds').")
):
    # Basic validation for agg_field requirement
    if agg_function in ["avg", "sum", "min", "max"] and not agg_field:
        raise HTTPException(status_code=400, detail=f"'{agg_function}' aggregation requires 'agg_field' to be specified.")
    if agg_function == "count" and agg_field:
        # For 'count', agg_field is not strictly necessary but could be used to count non-null values of that field.
        # For simplicity, we'll ignore it for 'count' here or could raise a warning/error.
        pass

    try:
        results = storage_instance.aggregate_tensor_descriptors(group_by_field, agg_function, agg_field)
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))

# --- Versioning and Lineage Router ---
router_version_lineage = APIRouter(
    tags=["Versioning & Lineage"]
)

class NewTensorVersionRequest(BaseModel):
    new_version_string: str
    # Allow specifying changes for the new TensorDescriptor
    # All fields are optional; if not provided, they might be copied from the parent.
    dimensionality: Optional[int] = None
    shape: Optional[List[int]] = None
    data_type: Optional[DataType] = None
    storage_format: Optional[str] = None
    owner: Optional[str] = None
    access_control: Optional[Dict[str, List[str]]] = None # Simplified for request
    byte_size: Optional[int] = None
    checksum: Optional[str] = None
    compression_info: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    # Add fields for other metadata types if they should be settable for a new version
    # For example, a new LineageSource for this version if it's materially different
    lineage_source_identifier: Optional[str] = None
    lineage_source_type: Optional[LineageSourceType] = None


@router_version_lineage.post("/tensors/{tensor_id}/versions", response_model=TensorDescriptor, status_code=201,
                               summary="Create a New Version of a Tensor",
                               description="Creates a new TensorDescriptor as a new version of an existing tensor. "
                                           "The new version will have its own unique tensor_id and will be linked "
                                           "to the original tensor via LineageMetadata.")
async def create_tensor_version(
    tensor_id: UUID = Path(..., description="The UUID of the tensor to version."), # type: ignore
    version_request: NewTensorVersionRequest = Body(...) # type: ignore
):
    parent_td = storage_instance.get_tensor_descriptor(tensor_id)
    if not parent_td:
        raise HTTPException(status_code=404, detail=f"Parent TensorDescriptor with ID {tensor_id} not found.")

    new_version_id = uuid.uuid4() # Generate new ID for the version

    # Prepare data for the new TensorDescriptor
    # Start by copying parent data, then override with request values
    new_td_data = parent_td.dict(exclude={'tensor_id', 'creation_timestamp', 'last_modified_timestamp'}) # Pydantic v1

    # Override with values from the request
    for field, value in version_request.dict(exclude_unset=True).items():
        if hasattr(TensorDescriptor.__fields__.get(field), 'annotation'): # Check if it's a field of TD
             if value is not None : new_td_data[field] = value
        # Exclude fields that are not part of TensorDescriptor schema directly like 'new_version_string'
        # or lineage_source_... which are handled separately.
        # This simple check might need refinement based on Pydantic version and exact field names.
        elif field not in ['new_version_string', 'lineage_source_identifier', 'lineage_source_type']:
            # If it's not a direct TD field and not a special request field, it might be intended for TD.metadata
            if new_td_data.get("metadata") is None: new_td_data["metadata"] = {}
            new_td_data["metadata"][field] = value


    new_td_data["tensor_id"] = new_version_id
    new_td_data["creation_timestamp"] = datetime.utcnow()
    new_td_data["last_modified_timestamp"] = new_td_data["creation_timestamp"]

    # Ensure required fields like owner, byte_size, etc. are present if not copied or set
    # This relies on Pydantic validation during TensorDescriptor creation.
    # If owner/byte_size are not in version_request and not copied (e.g. if parent_td.dict() excluded them),
    # this would need explicit handling or ensure they are always copied.
    # For now, assume parent_td.dict() includes them.
    if 'owner' not in new_td_data or new_td_data['owner'] is None:
        new_td_data['owner'] = parent_td.owner # Default to parent's owner
    if 'byte_size' not in new_td_data or new_td_data['byte_size'] is None:
        # Byte size might change, this is a placeholder. Ideally, it should be provided or recalculated.
        new_td_data['byte_size'] = parent_td.byte_size

    try:
        new_tensor_descriptor = TensorDescriptor(**new_td_data)
        storage_instance.add_tensor_descriptor(new_tensor_descriptor)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except ValueError as e: # From storage_instance if ID exists (should not happen with uuid4)
        raise HTTPException(status_code=400, detail=str(e))

    # Create LineageMetadata for the new version
    parent_link = ParentTensorLink(tensor_id=tensor_id, relationship="new_version_of")
    lineage_data = {
        "tensor_id": new_version_id,
        "parent_tensors": [parent_link],
        "version": version_request.new_version_string,
    }
    if version_request.lineage_source_identifier and version_request.lineage_source_type:
        lineage_data["source"] = {
            "type": version_request.lineage_source_type,
            "identifier": version_request.lineage_source_identifier
        }

    new_lineage_metadata = LineageMetadata(**lineage_data)
    storage_instance.add_lineage_metadata(new_lineage_metadata)

    # Optionally, copy other metadata types (Semantic, Computational etc.) or create new ones
    # For now, only TensorDescriptor and basic LineageMetadata are created for the new version.

    return new_tensor_descriptor


@router_version_lineage.get("/tensors/{tensor_id}/versions", response_model=List[TensorDescriptor],
                              summary="List Direct Next Versions of a Tensor",
                              description="Lists TensorDescriptors that are direct 'next versions' of the given tensor ID. "
                                          "This is based on LineageMetadata where the given tensor is a parent "
                                          "with a 'new_version_of' relationship. Includes the tensor itself.")
async def list_tensor_versions(tensor_id: UUID = Path(..., description="The UUID of the tensor.")): #type: ignore
    results = []
    # Include the tensor itself if it exists
    current_td = storage_instance.get_tensor_descriptor(tensor_id)
    if current_td:
        results.append(current_td)
    else: # If the base tensor_id itself doesn't exist, we can't find versions of it.
        raise HTTPException(status_code=404, detail=f"TensorDescriptor with ID {tensor_id} not found.")

    # Find direct children that are versions
    # This simplified approach finds only direct children marked as new versions.
    # A full version history might require traversing up to a root and then down all branches.
    all_lineage_meta = [_lm for _lm in storage_instance._lineage_metadata_store.values() if _lm] # type: ignore # Accessing internal for demo

    for lm_entry in all_lineage_meta:
        if lm_entry.parent_tensors:
            for parent_link in lm_entry.parent_tensors:
                if parent_link.tensor_id == tensor_id and parent_link.relationship == "new_version_of":
                    versioned_td = storage_instance.get_tensor_descriptor(lm_entry.tensor_id)
                    if versioned_td and versioned_td not in results: # Avoid duplicates if somehow current_td was a version of itself
                        results.append(versioned_td)
    return results


class LineageRelationshipRequest(BaseModel):
    source_tensor_id: UUID
    target_tensor_id: UUID
    relationship_type: str
    details: Optional[Dict[str, Any]] = None

@router_version_lineage.post("/lineage/relationships/", status_code=201,
                               summary="Create a Lineage Relationship",
                               description="Adds a parent-child lineage relationship between two tensors.")
async def create_lineage_relationship(req: LineageRelationshipRequest): # type: ignore
    source_td = storage_instance.get_tensor_descriptor(req.source_tensor_id)
    target_td = storage_instance.get_tensor_descriptor(req.target_tensor_id)

    if not source_td:
        raise HTTPException(status_code=404, detail=f"Source TensorDescriptor ID {req.source_tensor_id} not found.")
    if not target_td:
        raise HTTPException(status_code=404, detail=f"Target TensorDescriptor ID {req.target_tensor_id} not found.")

    target_lineage = storage_instance.get_lineage_metadata(req.target_tensor_id)
    if not target_lineage:
        target_lineage = LineageMetadata(tensor_id=req.target_tensor_id, parent_tensors=[])

    # Avoid duplicate relationships
    existing_link = next((p for p in target_lineage.parent_tensors if p.tensor_id == req.source_tensor_id and p.relationship == req.relationship_type), None)
    if existing_link:
        # Potentially update details if provided, or just return 200 OK.
        if req.details and existing_link.details != req.details: # Pydantic models don't have details on ParentTensorLink
             # ParentTensorLink does not have a 'details' field in its current schema.
             # If it did, one might update: existing_link.details = req.details
             pass # No details to update on ParentTensorLink
        return {"message": "Relationship already exists.", "lineage": target_lineage}


    new_parent_link = ParentTensorLink(tensor_id=req.source_tensor_id, relationship=req.relationship_type)
    target_lineage.parent_tensors.append(new_parent_link)

    try:
        if storage_instance.get_lineage_metadata(req.target_tensor_id):
            storage_instance.update_lineage_metadata(req.target_tensor_id, parent_tensors=target_lineage.parent_tensors)
        else:
            storage_instance.add_lineage_metadata(target_lineage)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": "Lineage relationship created/updated.", "lineage": target_lineage}


@router_version_lineage.get("/tensors/{tensor_id}/lineage/parents", response_model=List[TensorDescriptor],
                              summary="Get Parent Tensors",
                              description="Retrieves the full TensorDescriptor objects for all direct parent tensors of the given tensor.")
async def get_parent_tensors(tensor_id: UUID = Path(..., description="The UUID of the tensor.")): # type: ignore
    if not storage_instance.get_tensor_descriptor(tensor_id):
        raise HTTPException(status_code=404, detail=f"TensorDescriptor with ID {tensor_id} not found.")

    parent_ids = storage_instance.get_parent_tensor_ids(tensor_id)
    parent_descriptors = [storage_instance.get_tensor_descriptor(pid) for pid in parent_ids if storage_instance.get_tensor_descriptor(pid)]
    return parent_descriptors

@router_version_lineage.get("/tensors/{tensor_id}/lineage/children", response_model=List[TensorDescriptor],
                              summary="Get Child Tensors",
                              description="Retrieves the full TensorDescriptor objects for all direct child tensors of the given tensor.")
async def get_child_tensors(tensor_id: UUID = Path(..., description="The UUID of the tensor.")): # type: ignore
    if not storage_instance.get_tensor_descriptor(tensor_id):
        raise HTTPException(status_code=404, detail=f"TensorDescriptor with ID {tensor_id} not found.")

    child_ids = storage_instance.get_child_tensor_ids(tensor_id)
    child_descriptors = [storage_instance.get_tensor_descriptor(cid) for cid in child_ids if storage_instance.get_tensor_descriptor(cid)]
    return child_descriptors


# --- Router for Extended Metadata (CRUD per type) ---
router_extended_metadata = APIRouter(
    prefix="/tensor_descriptors/{tensor_id}", # Nested under tensor_descriptors
    tags=["Extended Metadata"]
)

# Helper to check if TensorDescriptor exists
def _get_td_or_404(tensor_id: UUID):
    td = storage_instance.get_tensor_descriptor(tensor_id)
    if not td:
        raise HTTPException(status_code=404, detail=f"TensorDescriptor with ID {tensor_id} not found.")
    return td

# --- LineageMetadata CRUD ---
@router_extended_metadata.post("/lineage", response_model=LineageMetadata, status_code=201,
                                summary="Create or Replace Lineage Metadata",
                                description="Creates new lineage metadata for a tensor or fully replaces existing lineage metadata.")
async def upsert_lineage_metadata(
    tensor_id: UUID = Path(..., description="The UUID of the TensorDescriptor."), # type: ignore
    lineage_in: LineageMetadata # Full model for POST/PUT like behavior
):
    _get_td_or_404(tensor_id)
    if lineage_in.tensor_id != tensor_id:
        raise HTTPException(status_code=400, detail=f"tensor_id in path ({tensor_id}) does not match tensor_id in body ({lineage_in.tensor_id}).")
    try:
        # The add_lineage_metadata in storage now acts as upsert (full replace)
        storage_instance.add_lineage_metadata(lineage_in)
        return lineage_in
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router_extended_metadata.get("/lineage", response_model=LineageMetadata,
                               summary="Get Lineage Metadata",
                               description="Retrieves the lineage metadata for a specific tensor.")
async def get_lineage_metadata(tensor_id: UUID = Path(..., description="The UUID of the TensorDescriptor.")): #type: ignore
    _get_td_or_404(tensor_id)
    lineage = storage_instance.get_lineage_metadata(tensor_id)
    if not lineage:
        raise HTTPException(status_code=404, detail=f"LineageMetadata not found for tensor ID {tensor_id}.")
    return lineage

@router_extended_metadata.patch("/lineage", response_model=LineageMetadata,
                                 summary="Update Lineage Metadata (Partial)",
                                 description="Partially updates the lineage metadata for a specific tensor.")
async def patch_lineage_metadata(
    tensor_id: UUID = Path(..., description="The UUID of the TensorDescriptor."), #type: ignore
    updates: Dict[str, Any] = Body(..., description="Fields to update.") #type: ignore
):
    _get_td_or_404(tensor_id)
    if not storage_instance.get_lineage_metadata(tensor_id): # Check if it exists before trying to update
         raise HTTPException(status_code=404, detail=f"LineageMetadata not found for tensor ID {tensor_id} to update.")
    try:
        updated_lineage = storage_instance.update_lineage_metadata(tensor_id, **updates)
        if not updated_lineage: # Should not happen if previous check passed, but good for safety
            raise HTTPException(status_code=404, detail=f"Failed to update LineageMetadata for tensor ID {tensor_id}.")
        return updated_lineage
    except ValueError as e: # Catches validation errors from Pydantic model in storage.update
        raise HTTPException(status_code=422, detail=str(e))


@router_extended_metadata.delete("/lineage", status_code=204,
                                  summary="Delete Lineage Metadata",
                                  description="Deletes the lineage metadata for a specific tensor.")
async def delete_lineage_metadata(tensor_id: UUID = Path(..., description="The UUID of the TensorDescriptor.")): #type: ignore
    _get_td_or_404(tensor_id)
    if not storage_instance.delete_lineage_metadata(tensor_id):
        raise HTTPException(status_code=404, detail=f"LineageMetadata not found for tensor ID {tensor_id} to delete.")
    return None # FastAPI will return 204 No Content


# --- ComputationalMetadata CRUD (Pattern repeated) ---
@router_extended_metadata.post("/computational", response_model=ComputationalMetadata, status_code=201)
async def upsert_computational_metadata(tensor_id: UUID, computational_in: ComputationalMetadata): #type: ignore
    _get_td_or_404(tensor_id)
    if computational_in.tensor_id != tensor_id: raise HTTPException(status_code=400, detail="Path tensor_id and body tensor_id mismatch.")
    try: storage_instance.add_computational_metadata(computational_in); return computational_in
    except ValueError as e: raise HTTPException(status_code=400, detail=str(e))

@router_extended_metadata.get("/computational", response_model=ComputationalMetadata)
async def get_computational_metadata(tensor_id: UUID): #type: ignore
    _get_td_or_404(tensor_id); meta = storage_instance.get_computational_metadata(tensor_id)
    if not meta: raise HTTPException(status_code=404, detail="ComputationalMetadata not found.")
    return meta

@router_extended_metadata.patch("/computational", response_model=ComputationalMetadata)
async def patch_computational_metadata(tensor_id: UUID, updates: Dict[str, Any]): #type: ignore
    _get_td_or_404(tensor_id)
    if not storage_instance.get_computational_metadata(tensor_id): raise HTTPException(status_code=404, detail="ComputationalMetadata not found for update.")
    try: updated = storage_instance.update_computational_metadata(tensor_id, **updates); return updated
    except ValueError as e: raise HTTPException(status_code=422, detail=str(e))

@router_extended_metadata.delete("/computational", status_code=204)
async def delete_computational_metadata(tensor_id: UUID): #type: ignore
    _get_td_or_404(tensor_id)
    if not storage_instance.delete_computational_metadata(tensor_id): raise HTTPException(status_code=404, detail="ComputationalMetadata not found for delete.")
    return None

# --- QualityMetadata CRUD ---
@router_extended_metadata.post("/quality", response_model=QualityMetadata, status_code=201)
async def upsert_quality_metadata(tensor_id: UUID, quality_in: QualityMetadata): #type: ignore
    _get_td_or_404(tensor_id);
    if quality_in.tensor_id != tensor_id: raise HTTPException(status_code=400, detail="Path tensor_id and body tensor_id mismatch.")
    try: storage_instance.add_quality_metadata(quality_in); return quality_in
    except ValueError as e: raise HTTPException(status_code=400, detail=str(e))

@router_extended_metadata.get("/quality", response_model=QualityMetadata)
async def get_quality_metadata(tensor_id: UUID): #type: ignore
    _get_td_or_404(tensor_id); meta = storage_instance.get_quality_metadata(tensor_id)
    if not meta: raise HTTPException(status_code=404, detail="QualityMetadata not found.")
    return meta

@router_extended_metadata.patch("/quality", response_model=QualityMetadata)
async def patch_quality_metadata(tensor_id: UUID, updates: Dict[str, Any]): #type: ignore
    _get_td_or_404(tensor_id)
    if not storage_instance.get_quality_metadata(tensor_id): raise HTTPException(status_code=404, detail="QualityMetadata not found for update.")
    try: updated = storage_instance.update_quality_metadata(tensor_id, **updates); return updated
    except ValueError as e: raise HTTPException(status_code=422, detail=str(e))

@router_extended_metadata.delete("/quality", status_code=204)
async def delete_quality_metadata(tensor_id: UUID): #type: ignore
    _get_td_or_404(tensor_id)
    if not storage_instance.delete_quality_metadata(tensor_id): raise HTTPException(status_code=404, detail="QualityMetadata not found for delete.")
    return None

# --- RelationalMetadata CRUD ---
@router_extended_metadata.post("/relational", response_model=RelationalMetadata, status_code=201)
async def upsert_relational_metadata(tensor_id: UUID, relational_in: RelationalMetadata): #type: ignore
    _get_td_or_404(tensor_id)
    if relational_in.tensor_id != tensor_id: raise HTTPException(status_code=400, detail="Path tensor_id and body tensor_id mismatch.")
    try: storage_instance.add_relational_metadata(relational_in); return relational_in
    except ValueError as e: raise HTTPException(status_code=400, detail=str(e))

@router_extended_metadata.get("/relational", response_model=RelationalMetadata)
async def get_relational_metadata(tensor_id: UUID): #type: ignore
    _get_td_or_404(tensor_id); meta = storage_instance.get_relational_metadata(tensor_id)
    if not meta: raise HTTPException(status_code=404, detail="RelationalMetadata not found.")
    return meta

@router_extended_metadata.patch("/relational", response_model=RelationalMetadata)
async def patch_relational_metadata(tensor_id: UUID, updates: Dict[str, Any]): #type: ignore
    _get_td_or_404(tensor_id)
    if not storage_instance.get_relational_metadata(tensor_id): raise HTTPException(status_code=404, detail="RelationalMetadata not found for update.")
    try: updated = storage_instance.update_relational_metadata(tensor_id, **updates); return updated
    except ValueError as e: raise HTTPException(status_code=422, detail=str(e))

@router_extended_metadata.delete("/relational", status_code=204)
async def delete_relational_metadata(tensor_id: UUID): #type: ignore
    _get_td_or_404(tensor_id)
    if not storage_instance.delete_relational_metadata(tensor_id): raise HTTPException(status_code=404, detail="RelationalMetadata not found for delete.")
    return None

# --- UsageMetadata CRUD ---
@router_extended_metadata.post("/usage", response_model=UsageMetadata, status_code=201)
async def upsert_usage_metadata(tensor_id: UUID, usage_in: UsageMetadata): #type: ignore
    _get_td_or_404(tensor_id)
    if usage_in.tensor_id != tensor_id: raise HTTPException(status_code=400, detail="Path tensor_id and body tensor_id mismatch.")
    try: storage_instance.add_usage_metadata(usage_in); return usage_in
    except ValueError as e: raise HTTPException(status_code=400, detail=str(e))

@router_extended_metadata.get("/usage", response_model=UsageMetadata)
async def get_usage_metadata(tensor_id: UUID): #type: ignore
    _get_td_or_404(tensor_id); meta = storage_instance.get_usage_metadata(tensor_id)
    if not meta: raise HTTPException(status_code=404, detail="UsageMetadata not found.")
    return meta

@router_extended_metadata.patch("/usage", response_model=UsageMetadata)
async def patch_usage_metadata(tensor_id: UUID, updates: Dict[str, Any]): #type: ignore
    _get_td_or_404(tensor_id)
    if not storage_instance.get_usage_metadata(tensor_id): raise HTTPException(status_code=404, detail="UsageMetadata not found for update.")
    try:
        # The update_usage_metadata method in storage handles specific logic for access_history and derived fields.
        updated = storage_instance.update_usage_metadata(tensor_id, **updates)
        return updated
    except ValueError as e: raise HTTPException(status_code=422, detail=str(e))

@router_extended_metadata.delete("/usage", status_code=204)
async def delete_usage_metadata(tensor_id: UUID): #type: ignore
    _get_td_or_404(tensor_id)
    if not storage_instance.delete_usage_metadata(tensor_id): raise HTTPException(status_code=404, detail="UsageMetadata not found for delete.")
    return None

# Note: A general DELETE for all semantic metadata of a tensor might be too broad.
# Deleting a TensorDescriptor already handles deletion of its associated semantic metadata.
# If a specific endpoint for deleting *all* semantic metadata entries for a tensor is needed,
# it can be added here, e.g., DELETE /semantic_metadata/{tensor_id}
# However, this is usually handled by deleting the parent TensorDescriptor.
# For now, we focus on deleting specific named entries.
import uuid # Added for the create_tensor_descriptor change
