from typing import List, Dict, Optional, Any, Annotated, Literal
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, HTTPException, Body, Query, Path
from pydantic import BaseModel, ValidationError

from tensorus.metadata.schemas import (
    TensorDescriptor, SemanticMetadata, DataType,
    LineageSourceType, LineageMetadata, ParentTensorLink,
    ComputationalMetadata, QualityMetadata, RelationalMetadata, UsageMetadata,
)
from tensorus.metadata.storage_abc import MetadataStorage
from tensorus.api.dependencies import get_storage_instance
from tensorus.storage.connectors import mock_tensor_connector_instance
from tensorus.metadata.schemas_iodata import TensorusExportData

from pydantic import BaseModel as PydanticBaseModel
from typing import TypeVar, Generic

from fastapi import Depends, Security, status # Added status for HTTPException
from fastapi.responses import JSONResponse
from .security import verify_api_key, api_key_header_auth
from tensorus.audit import log_audit_event

import copy
import uuid

# Router for TensorDescriptor
router_tensor_descriptor = APIRouter(
    prefix="/tensor_descriptors",
    tags=["Tensor Descriptors"],
    responses={404: {"description": "Not found"}},
)

# Router for SemanticMetadata
router_semantic_metadata = APIRouter(
    prefix="/tensor_descriptors/{tensor_id}/semantic", # Corrected prefix for consistency
    tags=["Semantic Metadata (Per Tensor)"],
    responses={404: {"description": "Not found"}},
)

# --- TensorDescriptor Endpoints ---
@router_tensor_descriptor.post("/", response_model=TensorDescriptor, status_code=status.HTTP_201_CREATED)
async def create_tensor_descriptor(
    descriptor_data: Dict[str, Any],
    storage: MetadataStorage = Depends(get_storage_instance),
    api_key: str = Depends(verify_api_key)
):
    tensor_id_str = descriptor_data.get("tensor_id")
    temp_id_for_lookup = UUID(tensor_id_str) if tensor_id_str else uuid.uuid4()

    fields_to_fetch = ["shape", "data_type", "byte_size", "dimensionality"]
    missing_fields = [field for field in fields_to_fetch if field not in descriptor_data or descriptor_data[field] is None]

    if missing_fields:
        storage_details = mock_tensor_connector_instance.get_tensor_details(temp_id_for_lookup)
        if storage_details:
            for field in missing_fields:
                if field in storage_details and (field not in descriptor_data or descriptor_data[field] is None) :
                    descriptor_data[field] = storage_details[field]
            if "shape" in descriptor_data and descriptor_data["shape"] is not None \
               and ("dimensionality" not in descriptor_data or descriptor_data["dimensionality"] is None):
                descriptor_data["dimensionality"] = len(descriptor_data["shape"])
    try:
        final_descriptor = TensorDescriptor(**descriptor_data)
        storage.add_tensor_descriptor(final_descriptor)
        if mock_tensor_connector_instance.retrieve_tensor(final_descriptor.tensor_id) is None:
            mock_tensor_data_payload = {
                "shape": final_descriptor.shape, "data_type": final_descriptor.data_type.value,
                "byte_size": final_descriptor.byte_size, "info": "Placeholder data by create_tensor_descriptor"
            }
            mock_tensor_connector_instance.store_tensor(final_descriptor.tensor_id, mock_tensor_data_payload)
        log_audit_event(action="CREATE_TENSOR_DESCRIPTOR", user=api_key, tensor_id=str(final_descriptor.tensor_id),
                        details={"owner": final_descriptor.owner, "data_type": final_descriptor.data_type.value})
        return final_descriptor
    except ValidationError as e:
        log_audit_event(action="CREATE_TENSOR_DESCRIPTOR_FAILED_VALIDATION", user=api_key, details={"error": str(e), "input_tensor_id": tensor_id_str})
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.errors())
    except ValueError as e:
        log_audit_event(action="CREATE_TENSOR_DESCRIPTOR_FAILED_STORAGE", user=api_key, details={"error": str(e), "input_tensor_id": tensor_id_str})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router_tensor_descriptor.get("/", response_model=List[TensorDescriptor], summary="List Tensor Descriptors with Advanced Filtering")
async def list_tensor_descriptors(
    owner: Optional[str] = Query(None), data_type: Optional[DataType] = Query(None),
    tags_contain: Optional[List[str]] = Query(None), lineage_version: Optional[str] = Query(None, alias="lineage.version"),
    lineage_source_type: Optional[LineageSourceType] = Query(None, alias="lineage.source.type"),
    comp_algorithm: Optional[str] = Query(None, alias="computational.algorithm"),
    comp_gpu_model: Optional[str] = Query(None, alias="computational.hardware_info.gpu_model"),
    quality_confidence_gt: Optional[float] = Query(None, alias="quality.confidence_score_gt"),
    quality_noise_lt: Optional[float] = Query(None, alias="quality.noise_level_lt"),
    rel_collection: Optional[str] = Query(None, alias="relational.collection"),
    rel_has_related_tensor_id: Optional[UUID] = Query(None, alias="relational.has_related_tensor_id"),
    usage_last_accessed_before: Optional[datetime] = Query(None, alias="usage.last_accessed_before"),
    usage_used_by_app: Optional[str] = Query(None, alias="usage.used_by_app"),
    storage: MetadataStorage = Depends(get_storage_instance)
):
    all_descriptors = storage.list_tensor_descriptors(
        owner=owner, data_type=data_type, tags_contain=tags_contain, lineage_version=lineage_version
    ) # Pass some common filters
    filtered_descriptors = []
    for desc in all_descriptors: # Apply remaining filters in memory
        if lineage_source_type and (not (lm := storage.get_lineage_metadata(desc.tensor_id)) or not lm.source or lm.source.type != lineage_source_type): continue
        if comp_algorithm or comp_gpu_model:
            cm = storage.get_computational_metadata(desc.tensor_id)
            if not cm: continue
            if comp_algorithm and cm.algorithm != comp_algorithm: continue
            if comp_gpu_model and (not cm.hardware_info or cm.hardware_info.get("gpu_model") != comp_gpu_model): continue
        if quality_confidence_gt or quality_noise_lt:
            qm = storage.get_quality_metadata(desc.tensor_id)
            if not qm: continue
            if quality_confidence_gt and (qm.confidence_score is None or qm.confidence_score <= quality_confidence_gt): continue
            if quality_noise_lt and (qm.noise_level is None or qm.noise_level >= quality_noise_lt): continue
        if rel_collection or rel_has_related_tensor_id:
            rm = storage.get_relational_metadata(desc.tensor_id)
            if not rm: continue
            if rel_collection and rel_collection not in rm.collections: continue
            if rel_has_related_tensor_id and not any(rtl.related_tensor_id == rel_has_related_tensor_id for rtl in rm.related_tensors): continue
        if usage_last_accessed_before or usage_used_by_app:
            um = storage.get_usage_metadata(desc.tensor_id)
            if not um: continue
            if usage_last_accessed_before and (not um.last_accessed_at or um.last_accessed_at >= usage_last_accessed_before): continue
            if usage_used_by_app and usage_used_by_app not in um.application_references: continue
        filtered_descriptors.append(desc)
    return filtered_descriptors

@router_tensor_descriptor.get("/{tensor_id}", response_model=TensorDescriptor)
async def get_tensor_descriptor(tensor_id: UUID = Path(...), storage: MetadataStorage = Depends(get_storage_instance)):
    descriptor = storage.get_tensor_descriptor(tensor_id)
    if not descriptor: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"TensorDescriptor ID {tensor_id} not found.")
    return descriptor

class TensorDescriptorUpdate(BaseModel):
    dimensionality: Optional[int]=None; shape: Optional[List[int]]=None; data_type: Optional[DataType]=None
    storage_format: Optional[str]=None; owner: Optional[str]=None; access_control: Optional[Dict[str, List[str]]]=None
    byte_size: Optional[int]=None; compression_info: Optional[Dict[str, Any]]=None
    tags: Optional[List[str]]=None; metadata: Optional[Dict[str, Any]]=None

@router_tensor_descriptor.put("/{tensor_id}", response_model=TensorDescriptor)
async def update_tensor_descriptor(
    tensor_id: UUID = Path(...), updates: TensorDescriptorUpdate = Body(...),
    storage: MetadataStorage = Depends(get_storage_instance), api_key: str = Depends(verify_api_key)
):
    update_data = updates.model_dump(exclude_unset=True)
    if not update_data: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No update data provided.")
    current = storage.get_tensor_descriptor(tensor_id)
    if not current:
        log_audit_event("UPDATE_TENSOR_DESCRIPTOR_FAILED_NOT_FOUND", api_key, str(tensor_id))
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"TensorDescriptor ID {tensor_id} not found.")
    try:
        updated = storage.update_tensor_descriptor(tensor_id, **update_data)
        log_audit_event("UPDATE_TENSOR_DESCRIPTOR", api_key, str(tensor_id), {"updated_fields": list(update_data.keys())})
        return updated
    except (ValidationError, ValueError) as e:
        log_audit_event("UPDATE_TENSOR_DESCRIPTOR_FAILED_VALIDATION", api_key, str(tensor_id), {"error": str(e)})
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY if isinstance(e, ValidationError) else status.HTTP_400_BAD_REQUEST, detail=str(e))

@router_tensor_descriptor.delete("/{tensor_id}", status_code=status.HTTP_200_OK)
async def delete_tensor_descriptor(
    tensor_id: UUID = Path(...), storage: MetadataStorage = Depends(get_storage_instance), api_key: str = Depends(verify_api_key)
):
    if not storage.get_tensor_descriptor(tensor_id):
        log_audit_event("DELETE_TENSOR_DESCRIPTOR_FAILED_NOT_FOUND", api_key, str(tensor_id))
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"TensorDescriptor ID {tensor_id} not found.")
    storage.delete_tensor_descriptor(tensor_id)
    mock_tensor_connector_instance.delete_tensor(tensor_id)
    log_audit_event("DELETE_TENSOR_DESCRIPTOR", api_key, str(tensor_id))
    return {"message": f"TensorDescriptor {tensor_id} and associated data deleted."}

# --- SemanticMetadata Endpoints ---
def _check_td_exists_for_semantic(tensor_id: UUID, storage: MetadataStorage, api_key: Optional[str] = None, action_prefix: str = ""):
    if not storage.get_tensor_descriptor(tensor_id):
        if api_key and action_prefix:
            log_audit_event(f"{action_prefix}_SEMANTIC_METADATA_FAILED_TD_NOT_FOUND", user=api_key, tensor_id=str(tensor_id))
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Parent TensorDescriptor ID {tensor_id} not found.")

@router_semantic_metadata.post("/", response_model=SemanticMetadata, status_code=status.HTTP_201_CREATED)
async def create_semantic_metadata_for_tensor(
    tensor_id: UUID = Path(...), metadata_in: SemanticMetadata = Body(...),
    storage: MetadataStorage = Depends(get_storage_instance), api_key: str = Depends(verify_api_key)
):
    _check_td_exists_for_semantic(tensor_id, storage, api_key, "CREATE")
    if metadata_in.tensor_id != tensor_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="tensor_id in path and body must match.")
    try:
        storage.add_semantic_metadata(metadata_in)
        log_audit_event("CREATE_SEMANTIC_METADATA", api_key, str(tensor_id), {"name": metadata_in.name})
        return metadata_in
    except (ValidationError, ValueError) as e:
        log_audit_event("CREATE_SEMANTIC_METADATA_FAILED", api_key, str(tensor_id), {"name": metadata_in.name, "error": str(e)})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST if isinstance(e, ValueError) else status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

@router_semantic_metadata.get("/", response_model=List[SemanticMetadata])
async def get_all_semantic_metadata_for_tensor(tensor_id: UUID = Path(...), storage: MetadataStorage = Depends(get_storage_instance)):
    _check_td_exists_for_semantic(tensor_id, storage)
    return storage.get_semantic_metadata(tensor_id)

class SemanticMetadataUpdate(BaseModel): name: Optional[str] = None; description: Optional[str] = None

@router_semantic_metadata.put("/{current_name}", response_model=SemanticMetadata)
async def update_named_semantic_metadata_for_tensor(
    tensor_id: UUID = Path(...), current_name: str = Path(...), updates: SemanticMetadataUpdate = Body(...),
    storage: MetadataStorage = Depends(get_storage_instance), api_key: str = Depends(verify_api_key)
):
    _check_td_exists_for_semantic(tensor_id, storage, api_key, "UPDATE")
    if not updates.model_dump(exclude_unset=True):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No update data provided.")
    if not storage.get_semantic_metadata_by_name(tensor_id, current_name):
        log_audit_event("UPDATE_SEMANTIC_METADATA_FAILED_NOT_FOUND", api_key, str(tensor_id), {"name": current_name})
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"SemanticMetadata '{current_name}' not found for tensor {tensor_id}.")
    try:
        updated = storage.update_semantic_metadata(tensor_id, current_name, new_description=updates.description, new_name=updates.name)
        log_audit_event(
            "UPDATE_SEMANTIC_METADATA",
            api_key,
            str(tensor_id),
            {"original_name": current_name, "updated_fields": updates.model_dump(exclude_unset=True)},
        )
        return updated
    except (ValidationError, ValueError) as e:
        log_audit_event("UPDATE_SEMANTIC_METADATA_FAILED", api_key, str(tensor_id), {"name": current_name, "error": str(e)})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST if isinstance(e, ValueError) else status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

@router_semantic_metadata.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_named_semantic_metadata_for_tensor(
    tensor_id: UUID = Path(...), name: str = Path(...),
    storage: MetadataStorage = Depends(get_storage_instance), api_key: str = Depends(verify_api_key)
):
    _check_td_exists_for_semantic(tensor_id, storage, api_key, "DELETE")
    if not storage.get_semantic_metadata_by_name(tensor_id, name):
        log_audit_event("DELETE_SEMANTIC_METADATA_FAILED_NOT_FOUND", api_key, str(tensor_id), {"name": name})
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"SemanticMetadata '{name}' not found for tensor {tensor_id}.")
    storage.delete_semantic_metadata(tensor_id, name)
    log_audit_event("DELETE_SEMANTIC_METADATA", api_key, str(tensor_id), {"name": name})
    return None

# --- Search and Aggregation Routers (GET - No Auth/Audit) ---
router_search_aggregate = APIRouter(tags=["Search & Aggregate"])
@router_search_aggregate.get("/search/tensors/", response_model=List[TensorDescriptor])
async def search_tensors(
    text_query: str = Query(..., min_length=1), fields_to_search: Optional[List[str]] = Query(None),
    storage: MetadataStorage = Depends(get_storage_instance)
):
    default_fields = ["tensor_id", "owner", "tags", "metadata", "semantic.name", "semantic.description", "lineage.source.identifier", "lineage.version", "computational.algorithm"]
    try: return storage.search_tensor_descriptors(text_query, fields_to_search or default_fields)
    except ValueError as e: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router_search_aggregate.get("/aggregate/tensors/", response_model=Dict[str, Any])
async def aggregate_tensors(
    group_by_field: str = Query(...), agg_function: str = Query(...), agg_field: Optional[str] = Query(None),
    storage: MetadataStorage = Depends(get_storage_instance)
):
    if agg_function in ["avg", "sum", "min", "max"] and not agg_field:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"'{agg_function}' requires 'agg_field'.")
    try: return storage.aggregate_tensor_descriptors(group_by_field, agg_function, agg_field)
    except ValueError as e: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except NotImplementedError as e: raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e))

# --- Versioning and Lineage Router ---
router_version_lineage = APIRouter(tags=["Versioning & Lineage"])
class NewTensorVersionRequest(BaseModel):
    new_version_string: str; dimensionality: Optional[int]=None; shape: Optional[List[int]]=None; data_type: Optional[DataType]=None
    storage_format: Optional[str]=None; owner: Optional[str]=None; access_control: Optional[Dict[str, List[str]]]=None
    byte_size: Optional[int]=None; checksum: Optional[str]=None; compression_info: Optional[Dict[str, Any]]=None
    tags: Optional[List[str]]=None; metadata: Optional[Dict[str, Any]]=None
    lineage_source_identifier: Optional[str]=None; lineage_source_type: Optional[LineageSourceType]=None

@router_version_lineage.post("/tensors/{tensor_id}/versions", response_model=TensorDescriptor, status_code=status.HTTP_201_CREATED)
async def create_tensor_version(
    tensor_id: UUID = Path(...), version_request: NewTensorVersionRequest = Body(...),
    storage: MetadataStorage = Depends(get_storage_instance), api_key: str = Depends(verify_api_key)
):
    parent_td = storage.get_tensor_descriptor(tensor_id)
    if not parent_td:
        log_audit_event("CREATE_TENSOR_VERSION_FAILED_PARENT_NOT_FOUND", api_key, str(tensor_id), {"new_version": version_request.new_version_string})
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Parent TensorDescriptor ID {tensor_id} not found.")
    new_version_id = uuid.uuid4()
    new_td_data = parent_td.model_dump(
        exclude={"tensor_id", "creation_timestamp", "last_modified_timestamp"}
    )
    for field, value in version_request.model_dump(exclude_unset=True).items():
        if field in TensorDescriptor.model_fields and value is not None: new_td_data[field] = value
        elif field not in ['new_version_string', 'lineage_source_identifier', 'lineage_source_type']:
            if new_td_data.get("metadata") is None: new_td_data["metadata"] = {}
            new_td_data["metadata"][field] = value
    new_td_data.update({
        "tensor_id": new_version_id, "creation_timestamp": datetime.utcnow(),
        "last_modified_timestamp": datetime.utcnow(),
        "owner": new_td_data.get('owner', parent_td.owner),
        "byte_size": new_td_data.get('byte_size', parent_td.byte_size)
    })
    try:
        new_td = TensorDescriptor(**new_td_data)
        storage.add_tensor_descriptor(new_td)
    except (ValidationError, ValueError) as e:
        log_audit_event("CREATE_TENSOR_VERSION_FAILED_VALIDATION", api_key, str(new_version_id), {"parent_id": str(tensor_id), "error": str(e)})
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY if isinstance(e, ValidationError) else status.HTTP_400_BAD_REQUEST, detail=str(e))

    lineage_details = {"tensor_id": new_version_id, "parent_tensors": [ParentTensorLink(tensor_id=tensor_id, relationship="new_version_of")], "version": version_request.new_version_string}
    if version_request.lineage_source_identifier and version_request.lineage_source_type:
        lineage_details["source"] = LineageSource(type=version_request.lineage_source_type, identifier=version_request.lineage_source_identifier) # type: ignore
    storage.add_lineage_metadata(LineageMetadata(**lineage_details))
    log_audit_event("CREATE_TENSOR_VERSION", api_key, str(new_version_id), {"parent_id": str(tensor_id), "version": version_request.new_version_string})
    return new_td

@router_version_lineage.get("/tensors/{tensor_id}/versions", response_model=List[TensorDescriptor])
async def list_tensor_versions(tensor_id: UUID = Path(...), storage: MetadataStorage = Depends(get_storage_instance)):
    results = []
    current_td = storage.get_tensor_descriptor(tensor_id)
    if not current_td: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"TensorDescriptor ID {tensor_id} not found.")
    results.append(current_td)
    child_ids = storage.get_child_tensor_ids(tensor_id)
    for child_id in child_ids:
        child_lineage = storage.get_lineage_metadata(child_id)
        if child_lineage and any(p.tensor_id == tensor_id and p.relationship == "new_version_of" for p in child_lineage.parent_tensors):
            child_td_obj = storage.get_tensor_descriptor(child_id)
            if child_td_obj: results.append(child_td_obj)
    return results

class LineageRelationshipRequest(BaseModel): source_tensor_id: UUID; target_tensor_id: UUID; relationship_type: str; details: Optional[Dict[str, Any]] = None

@router_version_lineage.post("/lineage/relationships/", status_code=status.HTTP_201_CREATED)
async def create_lineage_relationship(
    req: LineageRelationshipRequest, storage: MetadataStorage = Depends(get_storage_instance),
    api_key: str = Depends(verify_api_key)
):
    audit_details = req.model_dump()
    if not storage.get_tensor_descriptor(req.source_tensor_id):
        log_audit_event("CREATE_LINEAGE_REL_FAILED_SRC_NOT_FOUND", api_key, details=audit_details)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Source TD {req.source_tensor_id} not found.")
    if not storage.get_tensor_descriptor(req.target_tensor_id):
        log_audit_event("CREATE_LINEAGE_REL_FAILED_TGT_NOT_FOUND", api_key, details=audit_details)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Target TD {req.target_tensor_id} not found.")

    target_lineage = storage.get_lineage_metadata(req.target_tensor_id) or LineageMetadata(tensor_id=req.target_tensor_id)
    if any(p.tensor_id == req.source_tensor_id and p.relationship == req.relationship_type for p in target_lineage.parent_tensors):
        return {"message": "Relationship already exists.", "lineage": target_lineage}

    target_lineage.parent_tensors.append(ParentTensorLink(tensor_id=req.source_tensor_id, relationship=req.relationship_type))
    try:
        storage.add_lineage_metadata(target_lineage)
        log_audit_event("CREATE_LINEAGE_RELATIONSHIP", api_key, str(req.target_tensor_id), details=audit_details)
        return {"message": "Lineage relationship created/updated.", "lineage": storage.get_lineage_metadata(req.target_tensor_id)}
    except (ValidationError, ValueError) as e:
        log_audit_event("CREATE_LINEAGE_REL_FAILED_VALIDATION", api_key, str(req.target_tensor_id), {**audit_details, "error": str(e)})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router_version_lineage.get("/tensors/{tensor_id}/lineage/parents", response_model=List[TensorDescriptor])
async def get_parent_tensors(tensor_id: UUID = Path(...), storage: MetadataStorage = Depends(get_storage_instance)):
    if not storage.get_tensor_descriptor(tensor_id): raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"TD {tensor_id} not found.")
    parent_ids = storage.get_parent_tensor_ids(tensor_id)
    return [td for pid in parent_ids if (td := storage.get_tensor_descriptor(pid)) is not None]

@router_version_lineage.get("/tensors/{tensor_id}/lineage/children", response_model=List[TensorDescriptor])
async def get_child_tensors(tensor_id: UUID = Path(...), storage: MetadataStorage = Depends(get_storage_instance)):
    if not storage.get_tensor_descriptor(tensor_id): raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"TD {tensor_id} not found.")
    child_ids = storage.get_child_tensor_ids(tensor_id)
    return [td for cid in child_ids if (td := storage.get_tensor_descriptor(cid)) is not None]

# --- Router for Extended Metadata (CRUD per type) ---
router_extended_metadata = APIRouter(prefix="/tensor_descriptors/{tensor_id}", tags=["Extended Metadata (Per Tensor)"])

def _get_td_or_404_for_extended_meta(tensor_id: UUID, storage: MetadataStorage, api_key: Optional[str]=None, action_prefix: str = ""): # Renamed from previous version
    td = storage.get_tensor_descriptor(tensor_id)
    if not td:
        if api_key and action_prefix: log_audit_event(f"{action_prefix}_METADATA_FAILED_TD_NOT_FOUND", user=api_key, tensor_id=str(tensor_id)) # Generic prefix
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Parent TensorDescriptor ID {tensor_id} not found.")
    return td

async def _upsert_extended_metadata(tensor_id: UUID, metadata_name_cap: str, metadata_in: Any, storage: MetadataStorage, api_key: str):
    _get_td_or_404_for_extended_meta(tensor_id, storage, api_key, f"UPSERT_{metadata_name_cap}")
    if metadata_in.tensor_id != tensor_id:
        log_audit_event(f"UPSERT_{metadata_name_cap}_FAILED_ID_MISMATCH", api_key, str(tensor_id), {"body_id": str(metadata_in.tensor_id)})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Path tensor_id and body tensor_id must match.")
    try:
        add_method = getattr(storage, f"add_{metadata_name_cap.lower()}_metadata")
        add_method(metadata_in)
        log_audit_event(f"UPSERT_{metadata_name_cap}_METADATA", api_key, str(tensor_id))
        return metadata_in
    except (ValidationError, ValueError) as e:
        log_audit_event(f"UPSERT_{metadata_name_cap}_FAILED", api_key, str(tensor_id), {"error": str(e)})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST if isinstance(e, ValueError) else status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

async def _get_extended_metadata_ep(tensor_id: UUID, metadata_name_cap: str, storage: MetadataStorage): # Renamed to avoid clash
    _get_td_or_404_for_extended_meta(tensor_id, storage)
    get_method = getattr(storage, f"get_{metadata_name_cap.lower()}_metadata")
    meta = get_method(tensor_id)
    if not meta: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{metadata_name_cap}Metadata not found for tensor {tensor_id}.")
    return meta

async def _patch_extended_metadata(tensor_id: UUID, metadata_name_cap: str, updates: Dict[str, Any], storage: MetadataStorage, api_key: str):
    _get_td_or_404_for_extended_meta(tensor_id, storage, api_key, f"PATCH_{metadata_name_cap}")
    get_method = getattr(storage, f"get_{metadata_name_cap.lower()}_metadata")
    if not get_method(tensor_id):
        log_audit_event(f"PATCH_{metadata_name_cap}_FAILED_NOT_FOUND", api_key, str(tensor_id))
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{metadata_name_cap}Metadata not found for update.")
    try:
        update_method = getattr(storage, f"update_{metadata_name_cap.lower()}_metadata")
        updated_meta = update_method(tensor_id, **updates)
        log_audit_event(f"PATCH_{metadata_name_cap}_METADATA", api_key, str(tensor_id), {"updated_fields": list(updates.keys())})
        return updated_meta
    except (ValidationError, ValueError) as e:
        log_audit_event(f"PATCH_{metadata_name_cap}_FAILED_VALIDATION", api_key, str(tensor_id), {"error": str(e)})
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY if isinstance(e, ValidationError) else status.HTTP_400_BAD_REQUEST, detail=str(e))

async def _delete_extended_metadata(tensor_id: UUID, metadata_name_cap: str, storage: MetadataStorage, api_key: str):
    _get_td_or_404_for_extended_meta(tensor_id, storage, api_key, f"DELETE_{metadata_name_cap}")
    delete_method = getattr(storage, f"delete_{metadata_name_cap.lower()}_metadata")
    if not delete_method(tensor_id):
        log_audit_event(f"DELETE_{metadata_name_cap}_FAILED_NOT_FOUND", api_key, str(tensor_id))
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{metadata_name_cap}Metadata not found for delete.")
    log_audit_event(f"DELETE_{metadata_name_cap}_METADATA", api_key, str(tensor_id))
    return None

# Explicit CRUD endpoints for each extended metadata type
@router_extended_metadata.post("/lineage", response_model=LineageMetadata, status_code=status.HTTP_201_CREATED)
async def upsert_lineage_metadata_ep(tensor_id: UUID=Path(...), lineage_in: LineageMetadata=Body(...), storage: MetadataStorage=Depends(get_storage_instance), api_key: str=Depends(verify_api_key)):
    return await _upsert_extended_metadata(tensor_id, "Lineage", lineage_in, storage, api_key)
@router_extended_metadata.get("/lineage", response_model=LineageMetadata)
async def get_lineage_metadata_ep(tensor_id: UUID=Path(...), storage: MetadataStorage=Depends(get_storage_instance)):
    return await _get_extended_metadata_ep(tensor_id, "Lineage", storage)
@router_extended_metadata.patch("/lineage", response_model=LineageMetadata)
async def patch_lineage_metadata_ep(tensor_id: UUID=Path(...), updates: Dict[str,Any]=Body(...), storage: MetadataStorage=Depends(get_storage_instance), api_key: str=Depends(verify_api_key)):
    return await _patch_extended_metadata(tensor_id, "Lineage", updates, storage, api_key)
@router_extended_metadata.delete("/lineage", status_code=status.HTTP_204_NO_CONTENT)
async def delete_lineage_metadata_ep(tensor_id: UUID=Path(...), storage: MetadataStorage=Depends(get_storage_instance), api_key: str=Depends(verify_api_key)):
    return await _delete_extended_metadata(tensor_id, "Lineage", storage, api_key)

@router_extended_metadata.post("/computational", response_model=ComputationalMetadata, status_code=status.HTTP_201_CREATED)
async def upsert_computational_metadata_ep(tensor_id: UUID=Path(...), computational_in: ComputationalMetadata=Body(...), storage: MetadataStorage=Depends(get_storage_instance), api_key: str=Depends(verify_api_key)):
    return await _upsert_extended_metadata(tensor_id, "Computational", computational_in, storage, api_key)
@router_extended_metadata.get("/computational", response_model=ComputationalMetadata)
async def get_computational_metadata_ep(tensor_id: UUID=Path(...), storage: MetadataStorage=Depends(get_storage_instance)):
    return await _get_extended_metadata_ep(tensor_id, "Computational", storage)
@router_extended_metadata.patch("/computational", response_model=ComputationalMetadata)
async def patch_computational_metadata_ep(tensor_id: UUID=Path(...), updates: Dict[str,Any]=Body(...), storage: MetadataStorage=Depends(get_storage_instance), api_key: str=Depends(verify_api_key)):
    return await _patch_extended_metadata(tensor_id, "Computational", updates, storage, api_key)
@router_extended_metadata.delete("/computational", status_code=status.HTTP_204_NO_CONTENT)
async def delete_computational_metadata_ep(tensor_id: UUID=Path(...), storage: MetadataStorage=Depends(get_storage_instance), api_key: str=Depends(verify_api_key)):
    return await _delete_extended_metadata(tensor_id, "Computational", storage, api_key)

@router_extended_metadata.post("/quality", response_model=QualityMetadata, status_code=status.HTTP_201_CREATED)
async def upsert_quality_metadata_ep(tensor_id: UUID=Path(...), quality_in: QualityMetadata=Body(...), storage: MetadataStorage=Depends(get_storage_instance), api_key: str=Depends(verify_api_key)):
    return await _upsert_extended_metadata(tensor_id, "Quality", quality_in, storage, api_key)
@router_extended_metadata.get("/quality", response_model=QualityMetadata)
async def get_quality_metadata_ep(tensor_id: UUID=Path(...), storage: MetadataStorage=Depends(get_storage_instance)):
    return await _get_extended_metadata_ep(tensor_id, "Quality", storage)
@router_extended_metadata.patch("/quality", response_model=QualityMetadata)
async def patch_quality_metadata_ep(tensor_id: UUID=Path(...), updates: Dict[str,Any]=Body(...), storage: MetadataStorage=Depends(get_storage_instance), api_key: str=Depends(verify_api_key)):
    return await _patch_extended_metadata(tensor_id, "Quality", updates, storage, api_key)
@router_extended_metadata.delete("/quality", status_code=status.HTTP_204_NO_CONTENT)
async def delete_quality_metadata_ep(tensor_id: UUID=Path(...), storage: MetadataStorage=Depends(get_storage_instance), api_key: str=Depends(verify_api_key)):
    return await _delete_extended_metadata(tensor_id, "Quality", storage, api_key)

@router_extended_metadata.post("/relational", response_model=RelationalMetadata, status_code=status.HTTP_201_CREATED)
async def upsert_relational_metadata_ep(tensor_id: UUID=Path(...), relational_in: RelationalMetadata=Body(...), storage: MetadataStorage=Depends(get_storage_instance), api_key: str=Depends(verify_api_key)):
    return await _upsert_extended_metadata(tensor_id, "Relational", relational_in, storage, api_key)
@router_extended_metadata.get("/relational", response_model=RelationalMetadata)
async def get_relational_metadata_ep(tensor_id: UUID=Path(...), storage: MetadataStorage=Depends(get_storage_instance)):
    return await _get_extended_metadata_ep(tensor_id, "Relational", storage)
@router_extended_metadata.patch("/relational", response_model=RelationalMetadata)
async def patch_relational_metadata_ep(tensor_id: UUID=Path(...), updates: Dict[str,Any]=Body(...), storage: MetadataStorage=Depends(get_storage_instance), api_key: str=Depends(verify_api_key)):
    return await _patch_extended_metadata(tensor_id, "Relational", updates, storage, api_key)
@router_extended_metadata.delete("/relational", status_code=status.HTTP_204_NO_CONTENT)
async def delete_relational_metadata_ep(tensor_id: UUID=Path(...), storage: MetadataStorage=Depends(get_storage_instance), api_key: str=Depends(verify_api_key)):
    return await _delete_extended_metadata(tensor_id, "Relational", storage, api_key)

@router_extended_metadata.post("/usage", response_model=UsageMetadata, status_code=status.HTTP_201_CREATED)
async def upsert_usage_metadata_ep(tensor_id: UUID=Path(...), usage_in: UsageMetadata=Body(...), storage: MetadataStorage=Depends(get_storage_instance), api_key: str=Depends(verify_api_key)):
    return await _upsert_extended_metadata(tensor_id, "Usage", usage_in, storage, api_key)
@router_extended_metadata.get("/usage", response_model=UsageMetadata)
async def get_usage_metadata_ep(tensor_id: UUID=Path(...), storage: MetadataStorage=Depends(get_storage_instance)):
    return await _get_extended_metadata_ep(tensor_id, "Usage", storage)
@router_extended_metadata.patch("/usage", response_model=UsageMetadata)
async def patch_usage_metadata_ep(tensor_id: UUID=Path(...), updates: Dict[str,Any]=Body(...), storage: MetadataStorage=Depends(get_storage_instance), api_key: str=Depends(verify_api_key)):
    return await _patch_extended_metadata(tensor_id, "Usage", updates, storage, api_key)
@router_extended_metadata.delete("/usage", status_code=status.HTTP_204_NO_CONTENT)
async def delete_usage_metadata_ep(tensor_id: UUID=Path(...), storage: MetadataStorage=Depends(get_storage_instance), api_key: str=Depends(verify_api_key)):
    return await _delete_extended_metadata(tensor_id, "Usage", storage, api_key)

# --- I/O Router for Export/Import ---
router_io = APIRouter(prefix="/tensors", tags=["Import/Export"])

@router_io.get("/export", response_model=TensorusExportData)
async def export_tensor_metadata(
    tensor_ids_str: Optional[str] = Query(None, alias="tensor_ids"), # Changed FastAPIQuery to Query
    storage: MetadataStorage = Depends(get_storage_instance)
):
    parsed_tensor_ids: Optional[List[UUID]] = None
    if tensor_ids_str:
        try: parsed_tensor_ids = [UUID(tid.strip()) for tid in tensor_ids_str.split(',')]
        except ValueError: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid UUID format in tensor_ids.")
    export_data = storage.get_export_data(tensor_ids=parsed_tensor_ids)
    filename = f"tensorus_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return JSONResponse(content=export_data.model_dump(mode="json"), headers=headers)

@router_io.post("/import", summary="Import Tensor Metadata")
async def import_tensor_metadata(
    import_data_payload: TensorusExportData,
    conflict_strategy: Annotated[Literal["skip", "overwrite"], Query()] = "skip",
    storage: MetadataStorage = Depends(get_storage_instance),
    api_key: str = Depends(verify_api_key)
):
    try:
        result_summary = storage.import_data(import_data_payload, conflict_strategy=conflict_strategy)
        log_audit_event("IMPORT_DATA", api_key, details={"strategy": conflict_strategy, "summary": result_summary})
        return result_summary
    except NotImplementedError:
        log_audit_event("IMPORT_DATA_FAILED_NOT_IMPLEMENTED", api_key, details={"strategy": conflict_strategy})
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Import functionality is not implemented")
    except Exception as e:
        log_audit_event("IMPORT_DATA_FAILED_UNEXPECTED", api_key, details={"strategy": conflict_strategy, "error": str(e)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error during import: {e}")

# --- Management Router for Health and Metrics ---
router_management = APIRouter(tags=["Management"])
class HealthResponse(PydanticBaseModel): status: str; backend: str; detail: Optional[str] = None
@router_management.get("/health", response_model=HealthResponse)
async def health_check(storage: MetadataStorage = Depends(get_storage_instance)):
    is_healthy, backend_type = storage.check_health()
    if is_healthy: return HealthResponse(status="ok", backend=backend_type)
    else: return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                              content=HealthResponse(status="error", backend=backend_type, detail="Storage backend connection failed.").model_dump())
class MetricsResponse(PydanticBaseModel):
    total_tensor_descriptors: int; semantic_metadata_count: int; lineage_metadata_count: int
    computational_metadata_count: int; quality_metadata_count: int; relational_metadata_count: int; usage_metadata_count: int
@router_management.get("/metrics", response_model=MetricsResponse)
async def get_metrics(storage: MetadataStorage = Depends(get_storage_instance)):
    return MetricsResponse(
        total_tensor_descriptors=storage.get_tensor_descriptors_count(),
        semantic_metadata_count=storage.get_extended_metadata_count("SemanticMetadata"),
        lineage_metadata_count=storage.get_extended_metadata_count("LineageMetadata"),
        computational_metadata_count=storage.get_extended_metadata_count("ComputationalMetadata"),
        quality_metadata_count=storage.get_extended_metadata_count("QualityMetadata"),
        relational_metadata_count=storage.get_extended_metadata_count("RelationalMetadata"),
        usage_metadata_count=storage.get_extended_metadata_count("UsageMetadata")
    )

# --- Analytics Router ---
router_analytics = APIRouter(
    prefix="/analytics",
    tags=["Analytics"]
)

@router_analytics.get("/co_occurring_tags", response_model=Dict[str, List[Dict[str, Any]]],
                       summary="Get Co-occurring Tags",
                       description="Finds tags that frequently co-occur with other tags on tensor descriptors.")
async def api_get_co_occurring_tags(
    min_co_occurrence: int = Query(2, ge=1, description="Minimum number of times tags must appear together."),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of co-occurring tags to return for each primary tag."),
    storage: MetadataStorage = Depends(get_storage_instance)
):
    try:
        return storage.get_co_occurring_tags(min_co_occurrence=min_co_occurrence, limit=limit)
    except NotImplementedError:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Co-occurring tags analytics not implemented for the current storage backend.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error calculating co-occurring tags: {e}")


@router_analytics.get("/stale_tensors", response_model=List[TensorDescriptor],
                       summary="Get Stale Tensors",
                       description="Finds tensors that have not been accessed or modified for a given number of days.")
async def api_get_stale_tensors(
    threshold_days: int = Query(90, ge=1, description="Number of days to consider a tensor stale."),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of stale tensors to return."),
    storage: MetadataStorage = Depends(get_storage_instance)
):
    try:
        return storage.get_stale_tensors(threshold_days=threshold_days, limit=limit)
    except NotImplementedError:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Stale tensor analytics not implemented for the current storage backend.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error fetching stale tensors: {e}")


@router_analytics.get("/complex_tensors", response_model=List[TensorDescriptor],
                       summary="Get Complex Tensors",
                       description="Finds tensors considered complex based on lineage (number of parents or transformation steps).")
async def api_get_complex_tensors(
    min_parent_count: Optional[int] = Query(None, ge=0, description="Minimum number of parent tensors."),
    min_transformation_steps: Optional[int] = Query(None, ge=0, description="Minimum number of transformation steps."),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of complex tensors to return."),
    storage: MetadataStorage = Depends(get_storage_instance)
):
    if min_parent_count is None and min_transformation_steps is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one criterion (min_parent_count or min_transformation_steps) must be provided.")
    try:
        return storage.get_complex_tensors(
            min_parent_count=min_parent_count,
            min_transformation_steps=min_transformation_steps,
            limit=limit
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except NotImplementedError:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Complex tensor analytics not implemented for the current storage backend.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error fetching complex tensors: {e}")
