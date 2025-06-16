from fastapi import APIRouter, Depends, HTTPException
from typing import Any, Dict, List

from ..utils import tensor_to_list
from ...nql_agent import NQLAgent
from ...tensor_storage import TensorStorage
from ...api import NQLQueryRequest, NQLResponse, get_nql_agent

router = APIRouter()

@router.post("/query", response_model=NQLResponse)
async def query_route(
    request: NQLQueryRequest,
    nql_agent: NQLAgent = Depends(get_nql_agent),
):
    """Execute an NQL query and convert tensors to list for JSON responses."""
    try:
        result = nql_agent.process_query(request.query)
        if result.get("success") and isinstance(result.get("results"), list):
            converted: List[Dict[str, Any]] = []
            for rec in result["results"]:
                shape, dtype, data = tensor_to_list(rec["tensor"])
                converted.append(
                    {
                        "record_id": rec["metadata"].get("record_id"),
                        "shape": shape,
                        "dtype": dtype,
                        "data": data,
                        "metadata": rec["metadata"],
                    }
                )
            result["results"] = converted
        return NQLResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
