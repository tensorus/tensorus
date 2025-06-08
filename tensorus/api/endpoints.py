from typing import List, Dict, Optional, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Body, Query, Path
from pydantic import BaseModel, ValidationError

from tensorus.metadata.schemas import TensorDescriptor, SemanticMetadata, DataType
from tensorus.metadata.storage import storage_instance
from tensorus.storage.connectors import mock_tensor_connector_instance # Import the mock connector

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
                              summary="List Tensor Descriptors",
                              description="Lists all tensor descriptors, with optional filtering by owner and data type.")
async def list_tensor_descriptors(
    owner: Optional[str] = Query(None, description="Filter by owner of the tensor"),
    data_type: Optional[DataType] = Query(None, description="Filter by data type of the tensor (e.g., float32, int64)")
):
    descriptors = storage_instance.list_tensor_descriptors()
    filtered_descriptors = []
    for desc in descriptors:
        match_owner = owner is None or desc.owner == owner
        match_data_type = data_type is None or desc.data_type == data_type # DataType enum will be used for comparison
        if match_owner and match_data_type:
            filtered_descriptors.append(desc)
    return filtered_descriptors

@router_tensor_descriptor.get("/{tensor_id}", response_model=TensorDescriptor,
                                summary="Get Tensor Descriptor by ID",
                                description="Retrieves a specific tensor descriptor by its unique ID.")
async def get_tensor_descriptor(tensor_id: UUID = Path(..., description="The UUID of the tensor descriptor.")):
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
    tensor_id: UUID = Path(..., description="The UUID of the tensor descriptor to update."),
    updates: TensorDescriptorUpdate = Body(..., description="A dictionary containing the fields to update.")
):
    update_data = updates.dict(exclude_unset=True) # Use exclude_unset for partial updates

    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided.")

    # Fetch current descriptor to ensure it exists
    current_descriptor = storage_instance.get_tensor_descriptor(tensor_id)
    if not current_descriptor:
        raise HTTPException(status_code=404, detail=f"TensorDescriptor with ID {tensor_id} not found for update.")

    try:
        # Apply updates. The storage_instance.update_tensor_descriptor should handle validation.
        updated_descriptor = storage_instance.update_tensor_descriptor(tensor_id, **update_data)
        if updated_descriptor is None: # Should not happen if current_descriptor was found, but as a safeguard
             raise HTTPException(status_code=404, detail=f"TensorDescriptor with ID {tensor_id} failed to update.")
        return updated_descriptor
    except ValidationError as e: # Pydantic validation error during update within storage method
        raise HTTPException(status_code=422, detail=e.errors())
    except ValueError as e: # Custom validation errors
        raise HTTPException(status_code=400, detail=str(e))


@router_tensor_descriptor.delete("/{tensor_id}", status_code=200,
                                 summary="Delete Tensor Descriptor",
                                 description="Deletes a tensor descriptor, its associated semantic metadata, and also attempts to delete the tensor data from the mock tensor store.")
async def delete_tensor_descriptor(tensor_id: UUID = Path(..., description="The UUID of the tensor descriptor to delete.")):
    # First, delete from metadata storage
    metadata_deleted = storage_instance.delete_tensor_descriptor(tensor_id)
    if not metadata_deleted:
        raise HTTPException(status_code=404, detail=f"TensorDescriptor with ID {tensor_id} not found in metadata store.")

    # Then, attempt to delete from the mock tensor storage
    tensor_data_deleted = mock_tensor_connector_instance.delete_tensor(tensor_id)
    if tensor_data_deleted:
        return {"message": f"TensorDescriptor {tensor_id} and associated data deleted successfully."}
    else:
        # This case means metadata was deleted, but tensor data might not have been in the mock store
        # or an issue occurred (though mock delete is quite permissive).
        return {"message": f"TensorDescriptor {tensor_id} deleted from metadata. Tensor data may not have been found in mock store or already deleted."}


# --- SemanticMetadata Endpoints ---

@router_semantic_metadata.post("/", response_model=SemanticMetadata, status_code=201,
                                 summary="Create Semantic Metadata",
                                 description="Adds new semantic metadata linked to an existing TensorDescriptor.")
async def create_semantic_metadata(metadata: SemanticMetadata):
    # Ensure the tensor descriptor exists before adding semantic metadata
    if storage_instance.get_tensor_descriptor(metadata.tensor_id) is None:
        raise HTTPException(
            status_code=404,
            detail=f"TensorDescriptor with ID {metadata.tensor_id} not found. Cannot add semantic metadata."
        )
    try:
        storage_instance.add_semantic_metadata(metadata)
        return metadata
    except ValueError as e: # Catch potential errors from storage_instance.add_semantic_metadata
        raise HTTPException(status_code=400, detail=str(e))
    except ValidationError as e: # Catch Pydantic validation errors for SemanticMetadata model itself
        raise HTTPException(status_code=422, detail=e.errors())


@router_semantic_metadata.get("/{tensor_id}", response_model=List[SemanticMetadata],
                                summary="Get Semantic Metadata for a Tensor",
                                description="Retrieves all semantic metadata entries associated with a given TensorDescriptor ID.")
async def get_semantic_metadata_for_tensor(tensor_id: UUID = Path(..., description="The UUID of the tensor descriptor.")):
    if storage_instance.get_tensor_descriptor(tensor_id) is None: # Check if parent tensor descriptor exists
         raise HTTPException(status_code=404, detail=f"TensorDescriptor with ID {tensor_id} not found, so no semantic metadata can be retrieved.")

    metadata_list = storage_instance.get_semantic_metadata(tensor_id)
    # It's not an error if metadata_list is empty; the tensor might simply have no semantic metadata yet.
    return metadata_list

class SemanticMetadataUpdate(BaseModel):
    name: Optional[str] = None # Name is part of the key, usually not updatable like this.
                               # For this example, we'll assume we identify by old name and update description.
                               # A better approach might be a unique ID for each semantic metadata entry.
    description: Optional[str] = None


@router_semantic_metadata.put("/{tensor_id}/{name}", response_model=SemanticMetadata,
                                 summary="Update Semantic Metadata",
                                 description="Updates the description of a specific semantic metadata entry. The entry is identified by the tensor ID and its current name.")
async def update_semantic_metadata_entry(
    tensor_id: UUID = Path(..., description="The UUID of the tensor descriptor."),
    name: str = Path(..., description="The current name of the semantic metadata entry to be updated."),
    updates: SemanticMetadataUpdate = Body(..., description="The fields to update. Currently, only 'description' can be updated.")
):
    if updates.description is None and updates.name is None: # Check if any actual update is provided
        raise HTTPException(status_code=400, detail="No update data provided for 'name' or 'description'.")

    if updates.name is not None:
        # Updating the name is more complex as it's part of the identifier.
        # This mock implementation will disallow name changes via this method for simplicity.
        # A real implementation might require a more complex operation (e.g. delete old, create new).
        raise HTTPException(status_code=400, detail="Updating the 'name' of semantic metadata is not supported via this endpoint. Please delete and create a new entry if a name change is required.")

    if storage_instance.get_tensor_descriptor(tensor_id) is None:
        raise HTTPException(status_code=404, detail=f"TensorDescriptor with ID {tensor_id} not found.")

    updated_metadata = storage_instance.update_semantic_metadata(
        tensor_id,
        name,
        new_description=updates.description
    )

    if updated_metadata is None:
        raise HTTPException(status_code=404, detail=f"SemanticMetadata with name '{name}' for tensor ID {tensor_id} not found.")
    return updated_metadata


@router_semantic_metadata.delete("/{tensor_id}/{name}", status_code=200,
                                  summary="Delete Specific Semantic Metadata",
                                  description="Deletes a specific semantic metadata entry identified by its name, for a given tensor ID.")
async def delete_specific_semantic_metadata(
    tensor_id: UUID = Path(..., description="The UUID of the tensor descriptor."),
    name: str = Path(..., description="The name of the semantic metadata entry to delete.")
):
    if storage_instance.get_tensor_descriptor(tensor_id) is None:
        raise HTTPException(status_code=404, detail=f"TensorDescriptor with ID {tensor_id} not found.")

    if not storage_instance.delete_semantic_metadata(tensor_id, name):
        raise HTTPException(status_code=404, detail=f"SemanticMetadata with name '{name}' for tensor ID {tensor_id} not found or already deleted.")
    return {"message": f"SemanticMetadata '{name}' for tensor {tensor_id} deleted successfully."}

# Note: A general DELETE for all semantic metadata of a tensor might be too broad.
# Deleting a TensorDescriptor already handles deletion of its associated semantic metadata.
# If a specific endpoint for deleting *all* semantic metadata entries for a tensor is needed,
# it can be added here, e.g., DELETE /semantic_metadata/{tensor_id}
# However, this is usually handled by deleting the parent TensorDescriptor.
# For now, we focus on deleting specific named entries.
import uuid # Added for the create_tensor_descriptor change
