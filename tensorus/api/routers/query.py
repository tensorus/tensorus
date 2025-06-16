from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, status
from pydantic import BaseModel, Field

from tensorus.nql_agent import NQLAgent
from ..context import get_nql_agent
from .datasets import TensorOutput

logger = logging.getLogger(__name__)

router = APIRouter()


class NQLQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query string.")


class NQLResponse(BaseModel):
    success: bool
    message: str
    count: Optional[int] = None
    results: Optional[List[TensorOutput]] = None


@router.post("/query", response_model=NQLResponse)
async def execute_nql_query(
    request: NQLQueryRequest, nql_agent_svc: NQLAgent = Depends(get_nql_agent)
):
    logger.info("Received NQL query: %s", request.query)
    try:
        nql_result = nql_agent_svc.process_query(request.query)
        output_results: Optional[List[TensorOutput]] = None
        processed = 0
        if nql_result.get("success") and isinstance(nql_result.get("results"), list):
            output_results = []
            for i, record in enumerate(nql_result["results"]):
                if not isinstance(record, dict) or "tensor" not in record or "metadata" not in record:
                    logger.warning("Skipping NQL result record index %s due to invalid format", i)
                    continue
                try:
                    from ..utils import tensor_to_list

                    shape, dtype, data_list = tensor_to_list(record["tensor"])
                    record_id = record["metadata"].get(
                        "record_id", f"missing_id_{random.randint(1000,9999)}_{i}"
                    )
                    output_results.append(
                        TensorOutput(record_id=record_id, shape=shape, dtype=dtype, data=data_list, metadata=record["metadata"])
                    )
                    processed += 1
                except Exception as err:
                    logger.error("Error converting tensor result at index %s: %s", i, err)
        response = NQLResponse(
            success=nql_result.get("success", False),
            message=nql_result.get("message", "Error: Query processing failed unexpectedly."),
            count=nql_result.get("count", processed if output_results is not None else None),
            results=output_results,
        )
        if not response.success:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=response.message)
        return response
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - unexpected
        logger.exception("Unexpected error processing NQL query '%s': %s", request.query, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during query processing.")
