from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, Field

from tensorus.tensor_storage import (
    TensorStorage,
    DatasetNotFoundError,
    TensorNotFoundError,
)
from ..context import get_tensor_storage
from ..utils import list_to_tensor, tensor_to_list

logger = logging.getLogger(__name__)

router = APIRouter()


class DatasetCreateRequest(BaseModel):
    name: str = Field(..., description="Unique name for the new dataset.")


class TensorInput(BaseModel):
    shape: List[int]
    dtype: str
    data: Union[List[Any], int, float]
    metadata: Optional[Dict[str, Any]] = None


class TensorOutput(BaseModel):
    record_id: str
    shape: List[int]
    dtype: str
    data: Union[List[Any], int, float]
    metadata: Dict[str, Any]


class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None


class UpdateMetadataRequest(BaseModel):
    new_metadata: Dict[str, Any]


@router.post("/datasets/create", response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(req: DatasetCreateRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    try:
        storage.create_dataset(req.name)
        return ApiResponse(success=True, message=f"Dataset '{req.name}' created successfully.")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error creating dataset '%s': %s", req.name, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while creating dataset.")


@router.post("/datasets/{name}/ingest", response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
async def ingest_tensor(
    name: str = Path(...),
    tensor_input: TensorInput = Body(...),
    storage: TensorStorage = Depends(get_tensor_storage),
):
    try:
        tensor = list_to_tensor(tensor_input.shape, tensor_input.dtype, tensor_input.data)
        record_id = storage.insert(name, tensor, tensor_input.metadata)
        return ApiResponse(success=True, message="Tensor ingested successfully.", data={"record_id": record_id})
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid tensor data or parameters: {e}")
    except TypeError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid data type provided: {e}")
    except Exception as e:  # pragma: no cover - unexpected
        logger.exception("Unexpected error ingesting into dataset '%s': %s", name, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during ingestion.")


@router.get("/datasets/{name}/fetch", response_model=ApiResponse)
async def fetch_dataset(name: str = Path(...), storage: TensorStorage = Depends(get_tensor_storage)):
    try:
        records = storage.get_dataset_with_metadata(name)
        output_records: List[TensorOutput] = []
        for i, record in enumerate(records):
            if not isinstance(record, dict) or "tensor" not in record or "metadata" not in record:
                logger.warning("Skipping record index %s in '%s' due to invalid format.", i, name)
                continue
            try:
                shape, dtype, data_list = tensor_to_list(record["tensor"])
                record_id = record["metadata"].get("record_id", f"missing_id_{random.randint(1000,9999)}_{i}")
                output_records.append(
                    TensorOutput(record_id=record_id, shape=shape, dtype=dtype, data=data_list, metadata=record["metadata"])
                )
            except Exception as conversion_err:
                rid = record.get("metadata", {}).get("record_id", f"index_{i}")
                logger.error("Error converting tensor to list for record '%s' in dataset '%s': %s", rid, name, conversion_err)
                continue
        return ApiResponse(success=True, message="Fetched dataset", data=output_records)
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:  # pragma: no cover - unexpected
        logger.exception("Unexpected error fetching dataset '%s': %s", name, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while fetching dataset.")


@router.get("/datasets/{name}/records", response_model=ApiResponse)
async def fetch_dataset_records(
    name: str = Path(...),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1),
    storage: TensorStorage = Depends(get_tensor_storage),
):
    try:
        records = storage.get_records_paginated(name, offset=offset, limit=limit)
        output_records = []
        for i, record in enumerate(records):
            shape, dtype, data_list = tensor_to_list(record["tensor"])
            record_id = record["metadata"].get("record_id", f"missing_id_{offset+i}")
            output_records.append(TensorOutput(record_id=record_id, shape=shape, dtype=dtype, data=data_list, metadata=record["metadata"]))
        return ApiResponse(success=True, message="Records retrieved successfully.", data=output_records)
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error fetching records for dataset '%s': %s", name, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error retrieving records.")


@router.get("/datasets", response_model=ApiResponse)
async def list_datasets(storage: TensorStorage = Depends(get_tensor_storage)):
    try:
        if hasattr(storage, "list_datasets") and callable(storage.list_datasets):
            dataset_names = storage.list_datasets()
        elif hasattr(storage, "datasets") and isinstance(storage.datasets, dict):
            dataset_names = list(storage.datasets.keys())
        else:
            logger.error("TensorStorage instance does not support listing datasets.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="API configuration error: Cannot list datasets.")
        return ApiResponse(success=True, message="Retrieved dataset list successfully.", data=dataset_names)
    except Exception as e:  # pragma: no cover - unexpected
        logger.exception("Unexpected error listing datasets: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while listing datasets.")


@router.get("/datasets/{name}/count", response_model=ApiResponse)
async def count_dataset(name: str = Path(...), storage: TensorStorage = Depends(get_tensor_storage)):
    try:
        count = storage.count(name)
        return ApiResponse(success=True, message="Dataset count retrieved successfully.", data={"count": count})
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:  # pragma: no cover - unexpected
        logger.exception("Unexpected error counting dataset '%s': %s", name, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error counting dataset.")


@router.get("/datasets/{dataset_name}/tensors/{record_id}", response_model=TensorOutput)
async def get_tensor_by_id_api(
    dataset_name: str = Path(...),
    record_id: str = Path(...),
    storage: TensorStorage = Depends(get_tensor_storage),
):
    try:
        record = storage.get_tensor_by_id(dataset_name, record_id)
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except TensorNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:  # pragma: no cover - unexpected
        logger.exception("Error fetching tensor '%s' from dataset '%s': %s", record_id, dataset_name, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error fetching tensor.")

    shape, dtype, data_list = tensor_to_list(record["tensor"])
    return TensorOutput(record_id=record_id, shape=shape, dtype=dtype, data=data_list, metadata=record["metadata"])


@router.delete("/datasets/{dataset_name}", response_model=ApiResponse)
async def delete_dataset_api(dataset_name: str = Path(...), storage: TensorStorage = Depends(get_tensor_storage)):
    try:
        storage.delete_dataset(dataset_name)
        return ApiResponse(success=True, message=f"Dataset '{dataset_name}' deleted successfully.")
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:  # pragma: no cover - unexpected
        logger.exception("Error deleting dataset '%s': %s", dataset_name, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error deleting dataset.")


@router.delete("/datasets/{dataset_name}/tensors/{record_id}", response_model=ApiResponse)
async def delete_tensor_api(
    dataset_name: str = Path(...),
    record_id: str = Path(...),
    storage: TensorStorage = Depends(get_tensor_storage),
):
    try:
        storage.delete_tensor(dataset_name, record_id)
        return ApiResponse(success=True, message=f"Tensor record '{record_id}' deleted successfully.")
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except TensorNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:  # pragma: no cover - unexpected
        logger.exception("Error deleting tensor '%s' from dataset '%s': %s", record_id, dataset_name, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error deleting tensor.")


@router.put("/datasets/{dataset_name}/tensors/{record_id}/metadata", response_model=ApiResponse)
async def update_tensor_metadata_api(
    dataset_name: str = Path(...),
    record_id: str = Path(...),
    update_request: UpdateMetadataRequest = Body(...),
    storage: TensorStorage = Depends(get_tensor_storage),
):
    try:
        storage.update_tensor_metadata(dataset_name, record_id, update_request.new_metadata)
        return ApiResponse(success=True, message="Tensor metadata updated successfully.")
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except TensorNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request: {e}")
    except Exception as e:  # pragma: no cover - unexpected
        logger.exception("Error updating metadata for tensor '%s' in dataset '%s': %s", record_id, dataset_name, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error updating tensor metadata.")


@router.get("/explorer/dataset/{dataset}/tensor/{tensor_id}/metadata")
async def explorer_get_tensor_metadata(
    dataset: str = Path(...),
    tensor_id: str = Path(...),
    storage: TensorStorage = Depends(get_tensor_storage),
):
    try:
        record = storage.get_tensor_by_id(dataset, tensor_id)
        return {"dataset": dataset, "tensor_id": tensor_id, "metadata": record["metadata"]}
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except TensorNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:  # pragma: no cover - unexpected
        logger.exception("Error fetching metadata for tensor '%s' in dataset '%s': %s", tensor_id, dataset, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error retrieving tensor metadata.")
