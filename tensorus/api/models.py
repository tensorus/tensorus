"""API Models for Tensorus."""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class NQLQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query string.", json_schema_extra={"example": "find image tensors from 'my_image_dataset' where metadata.source = 'web_scrape'"})


class TensorOutput(BaseModel):
    record_id: str = Field(..., description="Unique record ID assigned during ingestion.")
    shape: List[int] = Field(..., description="Shape of the retrieved tensor.")
    dtype: str = Field(..., description="Data type of the retrieved tensor.")
    data: Union[List[Any], int, float] = Field(..., description="Tensor data (nested list or scalar).")
    metadata: Dict[str, Any] = Field(..., description="Associated metadata.")


class NQLResponse(BaseModel):
    success: bool = Field(..., description="Indicates if the query was successfully processed (syntax, execution).")
    message: str = Field(..., description="Status message (e.g., 'Query successful', 'Error parsing query').")
    count: Optional[int] = Field(None, description="Number of matching records found.")
    results: Optional[List[TensorOutput]] = Field(None, description="List of matching tensor records.")