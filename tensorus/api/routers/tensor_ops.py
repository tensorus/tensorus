from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import APIRouter, Body, Depends, HTTPException, status
from pydantic import BaseModel, Field, root_validator

from tensorus.tensor_storage import TensorStorage, DatasetNotFoundError, TensorNotFoundError
from tensorus.tensor_ops import TensorOps
from ..context import get_tensor_storage
from ..utils import tensor_to_list
from .datasets import TensorOutput

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ops", tags=["Tensor Operations"])


class TensorRef(BaseModel):
    dataset_name: str
    record_id: str


class TensorInputVal(BaseModel):
    tensor_ref: Optional[TensorRef] = None
    scalar_value: Optional[Union[float, int]] = None

    @root_validator(pre=True)
    def check_one_input_provided(cls, values):
        if sum(v is not None for v in values.values()) != 1:
            raise ValueError("Exactly one of 'tensor_ref' or 'scalar_value' must be provided.")
        return values


class OpsBaseRequest(BaseModel):
    output_dataset_name: Optional[str] = None
    output_metadata: Optional[Dict[str, Any]] = None


class OpsReshapeRequestParams(BaseModel):
    new_shape: List[int]


class OpsPermuteRequestParams(BaseModel):
    dims: List[int]


class OpsTransposeRequestParams(BaseModel):
    dim0: int
    dim1: int


class OpsGetSingleDimensionParam(BaseModel):
    dim: Optional[Union[int, List[int]]] = None
    keepdim: bool = False


class OpsUnaryOpRequest(OpsBaseRequest):
    input_tensor: TensorRef
    params: Optional[Dict[str, Any]] = None


class OpsReshapeRequest(OpsBaseRequest):
    input_tensor: TensorRef
    params: OpsReshapeRequestParams


class OpsPermuteRequest(OpsBaseRequest):
    input_tensor: TensorRef
    params: OpsPermuteRequestParams


class OpsTransposeRequest(OpsBaseRequest):
    input_tensor: TensorRef
    params: OpsTransposeRequestParams


class OpsReductionRequest(OpsBaseRequest):
    input_tensor: TensorRef
    params: OpsGetSingleDimensionParam


class OpsMinMaxRequest(OpsBaseRequest):
    input_tensor: TensorRef
    params: Optional[OpsGetSingleDimensionParam] = None


class OpsLogRequest(OpsBaseRequest):
    input_tensor: TensorRef


class OpsBinaryOpRequest(OpsBaseRequest):
    input1: TensorRef
    input2: TensorInputVal


class OpsPowerRequest(OpsBaseRequest):
    base_tensor: TensorRef
    exponent: TensorInputVal


class OpsTensorListRequestParams(BaseModel):
    dim: int = 0


class OpsTensorListRequest(OpsBaseRequest):
    input_tensors: List[TensorRef]
    params: OpsTensorListRequestParams


class OpsEinsumRequestParams(BaseModel):
    equation: str


class OpsEinsumRequest(OpsBaseRequest):
    input_tensors: List[TensorRef]
    params: OpsEinsumRequestParams


class OpsResultResponse(BaseModel):
    success: bool
    message: str
    output_dataset_name: Optional[str] = None
    output_record_id: Optional[str] = None
    output_tensor_details: Optional[TensorOutput] = None


DEFAULT_OPS_OUTPUT_DATASET = "tensor_ops_results"


async def _get_tensor_from_ref(tensor_ref: TensorRef, storage: TensorStorage) -> torch.Tensor:
    try:
        record = storage.get_tensor_by_id(tensor_ref.dataset_name, tensor_ref.record_id)
        if not isinstance(record.get("tensor"), torch.Tensor):
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Retrieved object is not a tensor.")
        return record["tensor"]
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except TensorNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:  # pragma: no cover - unexpected
        logger.exception("Unexpected error fetching tensor '%s' from dataset '%s': %s", tensor_ref.record_id, tensor_ref.dataset_name, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected error fetching input tensor.")


async def _store_and_respond_ops(
    result_tensor: torch.Tensor,
    op_name: str,
    request: OpsBaseRequest,
    storage: TensorStorage,
    input_refs: List[TensorRef],
) -> OpsResultResponse:
    output_dataset_name = request.output_dataset_name or DEFAULT_OPS_OUTPUT_DATASET
    try:
        if not storage.dataset_exists(output_dataset_name):
            storage.create_dataset(output_dataset_name)
    except Exception as e:
        logger.error("Failed to ensure output dataset '%s' exists: %s", output_dataset_name, e)

    final_metadata = {
        "operation": op_name,
        "timestamp": time.time(),
        "source_tensors": [ref.model_dump() for ref in input_refs],
        "source_api_request": request.model_dump(exclude_none=True),
        **(request.output_metadata or {}),
    }
    try:
        record_id = storage.insert(output_dataset_name, result_tensor, final_metadata)
        s, d, dl = tensor_to_list(result_tensor)
        tensor_out_details = TensorOutput(record_id=record_id, shape=s, dtype=d, data=dl, metadata=final_metadata)
        return OpsResultResponse(
            success=True,
            message=f"'{op_name}' operation successful. Result stored in '{output_dataset_name}/{record_id}'.",
            output_dataset_name=output_dataset_name,
            output_record_id=record_id,
            output_tensor_details=tensor_out_details,
        )
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error storing result: {e}")
    except Exception as e:  # pragma: no cover - unexpected
        logger.exception("Unexpected error storing result of '%s': %s", op_name, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error storing operation result.")


async def _get_input_val(input_val: TensorInputVal, storage: TensorStorage) -> Union[torch.Tensor, float, int]:
    if input_val.tensor_ref:
        return await _get_tensor_from_ref(input_val.tensor_ref, storage)
    if input_val.scalar_value is not None:
        return input_val.scalar_value
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid input value structure.")


@router.post("/log", response_model=OpsResultResponse)
async def tensor_log(request: OpsLogRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    input_tensor = await _get_tensor_from_ref(request.input_tensor, storage)
    try:
        result_tensor = TensorOps.log(input_tensor)
    except (ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "log", request, storage, [request.input_tensor])


@router.post("/reshape", response_model=OpsResultResponse)
async def tensor_reshape(request: OpsReshapeRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    input_tensor = await _get_tensor_from_ref(request.input_tensor, storage)
    try:
        result_tensor = TensorOps.reshape(input_tensor, request.params.new_shape)
    except (ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "reshape", request, storage, [request.input_tensor])


@router.post("/transpose", response_model=OpsResultResponse)
async def tensor_transpose(request: OpsTransposeRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    input_tensor = await _get_tensor_from_ref(request.input_tensor, storage)
    try:
        result_tensor = TensorOps.transpose(input_tensor, request.params.dim0, request.params.dim1)
    except (ValueError, TypeError, IndexError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "transpose", request, storage, [request.input_tensor])


@router.post("/permute", response_model=OpsResultResponse)
async def tensor_permute(request: OpsPermuteRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    input_tensor = await _get_tensor_from_ref(request.input_tensor, storage)
    try:
        result_tensor = TensorOps.permute(input_tensor, tuple(request.params.dims))
    except (ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "permute", request, storage, [request.input_tensor])


@router.post("/sum", response_model=OpsResultResponse)
async def tensor_sum(request: OpsReductionRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    input_tensor = await _get_tensor_from_ref(request.input_tensor, storage)
    try:
        result_tensor = TensorOps.sum(input_tensor, dim=request.params.dim, keepdim=request.params.keepdim)
    except (ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "sum", request, storage, [request.input_tensor])


@router.post("/mean", response_model=OpsResultResponse)
async def tensor_mean(request: OpsReductionRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    input_tensor = await _get_tensor_from_ref(request.input_tensor, storage)
    try:
        result_tensor = TensorOps.mean(input_tensor, dim=request.params.dim, keepdim=request.params.keepdim)
    except (ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "mean", request, storage, [request.input_tensor])


@router.post("/min", response_model=OpsResultResponse)
async def tensor_min(request: OpsMinMaxRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    input_tensor = await _get_tensor_from_ref(request.input_tensor, storage)
    message_suffix = ""
    try:
        if request.params and request.params.dim is not None:
            result_tuple = TensorOps.min(input_tensor, dim=request.params.dim, keepdim=request.params.keepdim)
            result_tensor = result_tuple.values
            message_suffix = " (values tensor stored)"
        else:
            result_tensor = TensorOps.min(input_tensor)
    except (ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    response = await _store_and_respond_ops(result_tensor, "min", request, storage, [request.input_tensor])
    if message_suffix:
        response.message += message_suffix
    return response


@router.post("/max", response_model=OpsResultResponse)
async def tensor_max(request: OpsMinMaxRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    input_tensor = await _get_tensor_from_ref(request.input_tensor, storage)
    message_suffix = ""
    try:
        if request.params and request.params.dim is not None:
            result_tuple = TensorOps.max(input_tensor, dim=request.params.dim, keepdim=request.params.keepdim)
            result_tensor = result_tuple.values
            message_suffix = " (values tensor stored)"
        else:
            result_tensor = TensorOps.max(input_tensor)
    except (ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    response = await _store_and_respond_ops(result_tensor, "max", request, storage, [request.input_tensor])
    if message_suffix:
        response.message += message_suffix
    return response


@router.post("/add", response_model=OpsResultResponse)
async def tensor_add(request: OpsBinaryOpRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    input1_tensor = await _get_tensor_from_ref(request.input1, storage)
    input2_val = await _get_input_val(request.input2, storage)
    refs = [request.input1]
    if request.input2.tensor_ref:
        refs.append(request.input2.tensor_ref)
    try:
        result_tensor = TensorOps.add(input1_tensor, input2_val)
    except (ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "add", request, storage, refs)


@router.post("/subtract", response_model=OpsResultResponse)
async def tensor_subtract(request: OpsBinaryOpRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    input1_tensor = await _get_tensor_from_ref(request.input1, storage)
    input2_val = await _get_input_val(request.input2, storage)
    refs = [request.input1]
    if request.input2.tensor_ref:
        refs.append(request.input2.tensor_ref)
    try:
        result_tensor = TensorOps.subtract(input1_tensor, input2_val)
    except (ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "subtract", request, storage, refs)


@router.post("/multiply", response_model=OpsResultResponse)
async def tensor_multiply(request: OpsBinaryOpRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    input1_tensor = await _get_tensor_from_ref(request.input1, storage)
    input2_val = await _get_input_val(request.input2, storage)
    refs = [request.input1]
    if request.input2.tensor_ref:
        refs.append(request.input2.tensor_ref)
    try:
        result_tensor = TensorOps.multiply(input1_tensor, input2_val)
    except (ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "multiply", request, storage, refs)


@router.post("/divide", response_model=OpsResultResponse)
async def tensor_divide(request: OpsBinaryOpRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    input1_tensor = await _get_tensor_from_ref(request.input1, storage)
    input2_val = await _get_input_val(request.input2, storage)
    refs = [request.input1]
    if request.input2.tensor_ref:
        refs.append(request.input2.tensor_ref)
    try:
        result_tensor = TensorOps.divide(input1_tensor, input2_val)
    except (ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "divide", request, storage, refs)


@router.post("/power", response_model=OpsResultResponse)
async def tensor_power(request: OpsPowerRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    base_tensor_val = await _get_tensor_from_ref(request.base_tensor, storage)
    exponent_val = await _get_input_val(request.exponent, storage)
    refs = [request.base_tensor]
    if request.exponent.tensor_ref:
        refs.append(request.exponent.tensor_ref)
    try:
        result_tensor = TensorOps.power(base_tensor_val, exponent_val)
    except (ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "power", request, storage, refs)


@router.post("/matmul", response_model=OpsResultResponse)
async def tensor_matmul(request: OpsBinaryOpRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    input1_tensor = await _get_tensor_from_ref(request.input1, storage)
    input2_val = await _get_input_val(request.input2, storage)
    if not isinstance(input2_val, torch.Tensor):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input2 for matmul must be a tensor, not a scalar.")
    refs = [request.input1, request.input2.tensor_ref]
    try:
        result_tensor = TensorOps.matmul(input1_tensor, input2_val)
    except (ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "matmul", request, storage, refs)


@router.post("/dot", response_model=OpsResultResponse)
async def tensor_dot(request: OpsBinaryOpRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    input1_tensor = await _get_tensor_from_ref(request.input1, storage)
    input2_val = await _get_input_val(request.input2, storage)
    if not isinstance(input2_val, torch.Tensor):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input2 for dot product must be a tensor, not a scalar.")
    refs = [request.input1, request.input2.tensor_ref]
    try:
        result_tensor = TensorOps.dot(input1_tensor, input2_val)
    except (ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "dot", request, storage, refs)


@router.post("/concatenate", response_model=OpsResultResponse)
async def tensor_concatenate(request: OpsTensorListRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    tensors = [await _get_tensor_from_ref(ref, storage) for ref in request.input_tensors]
    if not tensors:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input tensors list cannot be empty for concatenate.")
    try:
        result_tensor = TensorOps.concatenate(tensors, dim=request.params.dim)
    except (ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "concatenate", request, storage, request.input_tensors)


@router.post("/stack", response_model=OpsResultResponse)
async def tensor_stack(request: OpsTensorListRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    tensors = [await _get_tensor_from_ref(ref, storage) for ref in request.input_tensors]
    if not tensors:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input tensors list cannot be empty for stack.")
    try:
        result_tensor = TensorOps.stack(tensors, dim=request.params.dim)
    except (ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "stack", request, storage, request.input_tensors)


@router.post("/einsum", response_model=OpsResultResponse)
async def tensor_einsum(request: OpsEinsumRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    tensors = [await _get_tensor_from_ref(ref, storage) for ref in request.input_tensors]
    if not tensors:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input tensors list cannot be empty for einsum.")
    try:
        result_tensor = TensorOps.einsum(request.params.equation, *tensors)
    except (ValueError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "einsum", request, storage, request.input_tensors)

