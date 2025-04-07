# api.py
"""
Implements the RESTful API for Tensorus using FastAPI.

Exposes endpoints for dataset management, data ingestion (via JSON),
and NQL querying. Includes basic tensor serialization/deserialization helpers.

Future Enhancements:
- Add authentication and authorization.
- Implement asynchronous operations for storage interactions if backend supports it.
- Add endpoints for controlling and monitoring agents (Ingestion, RL, AutoML).
- Implement file upload endpoint for direct tensor/data file ingestion.
- More robust error handling and logging.
- Pagination for endpoints returning lists of data.
- Websocket support for real-time updates.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Body, Depends, Path
from pydantic import BaseModel, Field # Use Pydantic v1 style for compatibility if needed

# Import Tensorus modules
from tensor_storage import TensorStorage
from nql_agent import NQLAgent
# We could potentially import and manage other agents here too

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Tensorus Instances ---
# For simplicity in this example, we create global instances.
# In production, consider using FastAPI's dependency injection features
# with factories or context managers for better resource management.
try:
    tensor_storage_instance = TensorStorage()
    nql_agent_instance = NQLAgent(tensor_storage_instance)
    # Initialize other agents if they need API interaction/control
    # ingestion_agent_instance = DataIngestionAgent(...)
    # rl_agent_instance = RLAgent(...)
    # automl_agent_instance = AutoMLAgent(...)
except Exception as e:
    logger.exception(f"Failed to initialize Tensorus components: {e}")
    # Handle initialization failure appropriately - maybe exit or run in degraded mode
    raise RuntimeError(f"Tensorus initialization failed: {e}") from e


# --- Helper Functions ---

def _validate_tensor_data(data: List[Any], shape: List[int]):
    """Basic validation for nested list data against shape."""
    if not shape: # Scalar tensor
         if not isinstance(data, (int, float)):
              raise ValueError("Scalar tensor data must be a single number.")
         return True

    if not isinstance(data, list):
         raise ValueError(f"Data for shape {shape} must be a list.")

    expected_len = shape[0]
    if len(data) != expected_len:
         raise ValueError(f"Dimension 0 mismatch: Expected length {expected_len}, got {len(data)} for shape {shape}.")

    # Recursively validate nested lists if shape has more dimensions
    if len(shape) > 1:
         for item in data:
             _validate_tensor_data(item, shape[1:]) # Validate sub-structure

    # Basic type check for the innermost list elements
    elif len(shape) == 1:
        if not all(isinstance(x, (int, float)) for x in data):
            raise ValueError("Innermost list elements must be numbers (int/float).")

    return True


def list_to_tensor(shape: List[int], dtype_str: str, data: Union[List[Any], int, float]) -> torch.Tensor:
    """Converts validated list data (potentially nested) to a PyTorch tensor."""
    try:
        # Map string dtype to torch dtype
        dtype_map = {
            'float32': torch.float32, 'float': torch.float,
            'float64': torch.float64, 'double': torch.double,
            'int32': torch.int32, 'int': torch.int,
            'int64': torch.int64, 'long': torch.long,
            'bool': torch.bool,
            # Add other dtypes as needed
        }
        torch_dtype = dtype_map.get(dtype_str.lower())
        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype string: {dtype_str}")

        # Validate structure before conversion attempt
        # _validate_tensor_data(data, shape) # Validation can be strict, maybe relax

        tensor = torch.tensor(data, dtype=torch_dtype)

        # Verify shape after creation (tensor creation might broadcast/squeeze)
        if list(tensor.shape) != shape:
             # Attempt reshape if possible (e.g. scalar created as [])
             try:
                 tensor = tensor.reshape(shape)
             except RuntimeError:
                  raise ValueError(f"Created tensor shape {list(tensor.shape)} does not match requested shape {shape} and couldn't be reshaped.")

        return tensor
    except (TypeError, ValueError) as e:
        logger.error(f"Error converting list to tensor: {e}. Shape: {shape}, Dtype: {dtype_str}")
        raise ValueError(f"Failed to convert data to tensor with shape {shape} and dtype {dtype_str}: {e}") from e
    except Exception as e:
         logger.error(f"Unexpected error during list_to_tensor: {e}", exc_info=True)
         raise ValueError(f"An unexpected error occurred during tensor conversion: {e}") from e


def tensor_to_list(tensor: torch.Tensor) -> Tuple[List[int], str, List[Any]]:
    """Converts a PyTorch tensor to shape, dtype string, and nested list."""
    shape = list(tensor.shape)
    dtype_str = str(tensor.dtype).split('.')[-1] # Get 'float32', 'int64' etc.
    data = tensor.tolist()
    return shape, dtype_str, data


# --- Pydantic Models ---

class DatasetCreateRequest(BaseModel):
    name: str = Field(..., description="Unique name for the new dataset.", example="my_new_dataset")

class TensorInput(BaseModel):
    shape: List[int] = Field(..., description="Shape of the tensor.", example=[2, 3])
    dtype: str = Field(..., description="Data type of the tensor (e.g., 'float32', 'int64').", example="float32")
    data: Union[List[Any], int, float] = Field(..., description="Tensor data as a nested list or scalar.", example=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata dictionary.", example={"source": "api_ingest", "quality": 0.9})

class NQLQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query string.", example="find records from sensor_data where status = 'active'")

# Response Models
class TensorOutput(BaseModel):
    record_id: str = Field(..., description="Unique record ID assigned by TensorStorage.")
    shape: List[int] = Field(..., description="Shape of the tensor.")
    dtype: str = Field(..., description="Data type of the tensor.")
    data: Union[List[Any], int, float] = Field(..., description="Tensor data as a nested list or scalar.")
    metadata: Dict[str, Any] = Field(..., description="Metadata associated with the tensor.")

class NQLResponse(BaseModel):
    success: bool = Field(..., description="Indicates if the NQL query was successfully processed.")
    message: str = Field(..., description="Status message or error description.")
    count: Optional[int] = Field(None, description="Number of results found (if applicable).")
    # Return results as list of TensorOutput for consistency
    results: Optional[List[TensorOutput]] = Field(None, description="List of matching tensor records.")

class ApiResponse(BaseModel):
    success: bool = Field(..., description="Indicates if the operation was successful.")
    message: str = Field(..., description="Status message or result description.")
    data: Optional[Any] = Field(None, description="Optional data payload (e.g., list of dataset names, record ID).")


# --- FastAPI App Instance ---
app = FastAPI(
    title="Tensorus API",
    description="API for interacting with the Tensorus Agentic Tensor Database/Data Lake.",
    version="0.1.0",
)

# --- Dependency Functions (Optional but good practice) ---
# These allow FastAPI to manage the lifecycle if needed, though globals are used here.
async def get_tensor_storage() -> TensorStorage:
     return tensor_storage_instance

async def get_nql_agent() -> NQLAgent:
     return nql_agent_instance


# --- API Endpoints ---

@app.post("/datasets/create", response_model=ApiResponse, status_code=201, tags=["Datasets"])
async def create_dataset(
    request: DatasetCreateRequest,
    storage: TensorStorage = Depends(get_tensor_storage)
):
    """Creates a new, empty dataset in TensorStorage."""
    try:
        storage.create_dataset(request.name)
        return ApiResponse(success=True, message=f"Dataset '{request.name}' created successfully.")
    except ValueError as e:
        logger.warning(f"Failed to create dataset '{request.name}': {e}")
        raise HTTPException(status_code=409, detail=str(e)) # 409 Conflict if already exists
    except Exception as e:
         logger.exception(f"Unexpected error creating dataset '{request.name}': {e}")
         raise HTTPException(status_code=500, detail="An internal server error occurred.")


@app.post("/datasets/{name}/ingest", response_model=ApiResponse, status_code=201, tags=["Data Ingestion"])
async def ingest_tensor(
    name: str = Path(..., description="Name of the target dataset."),
    tensor_input: TensorInput = Body(...),
    storage: TensorStorage = Depends(get_tensor_storage)
):
    """Ingests a single tensor (provided as JSON) into the specified dataset."""
    try:
        # Convert list data back to tensor
        tensor = list_to_tensor(tensor_input.shape, tensor_input.dtype, tensor_input.data)
        # Insert into storage
        record_id = storage.insert(name, tensor, tensor_input.metadata)
        return ApiResponse(
            success=True,
            message=f"Tensor successfully ingested into dataset '{name}'.",
            data={"record_id": record_id}
        )
    except ValueError as e: # Catches dataset not found, tensor conversion errors
        logger.warning(f"Failed to ingest tensor into dataset '{name}': {e}")
        raise HTTPException(status_code=400, detail=str(e)) # 400 Bad Request for conversion/validation errors
    except TypeError as e: # Catches non-tensor data error from storage.insert
        logger.warning(f"Type error during ingestion into dataset '{name}': {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
         logger.exception(f"Unexpected error ingesting into dataset '{name}': {e}")
         raise HTTPException(status_code=500, detail="An internal server error occurred.")


@app.get("/datasets/{name}/fetch", response_model=ApiResponse, tags=["Data Retrieval"])
async def fetch_dataset(
    name: str = Path(..., description="Name of the dataset to fetch."),
    storage: TensorStorage = Depends(get_tensor_storage)
):
    """Retrieves all tensor records (including data and metadata) from a dataset."""
    try:
        records = storage.get_dataset_with_metadata(name)
        output_records = []
        for record in records:
             shape, dtype, data_list = tensor_to_list(record['tensor'])
             output_records.append(TensorOutput(
                 record_id=record['metadata'].get('record_id', 'N/A'), # Get ID from metadata
                 shape=shape,
                 dtype=dtype,
                 data=data_list,
                 metadata=record['metadata']
             ))
        return ApiResponse(
            success=True,
            message=f"Successfully retrieved {len(output_records)} records from dataset '{name}'.",
            data=output_records
            )
    except ValueError as e: # Dataset not found
        logger.warning(f"Failed to fetch dataset '{name}': {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
         logger.exception(f"Unexpected error fetching dataset '{name}': {e}")
         raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/datasets", response_model=ApiResponse, tags=["Datasets"])
async def list_datasets(
     storage: TensorStorage = Depends(get_tensor_storage)
 ):
     """Lists the names of all available datasets."""
     try:
         # Accessing the internal dictionary keys - needs a proper method in TensorStorage ideally
         dataset_names = list(storage.datasets.keys())
         return ApiResponse(success=True, message="Retrieved available datasets.", data=dataset_names)
     except Exception as e:
         logger.exception(f"Unexpected error listing datasets: {e}")
         raise HTTPException(status_code=500, detail="An internal server error occurred.")


@app.post("/query", response_model=NQLResponse, tags=["Querying"])
async def execute_nql_query(
    request: NQLQueryRequest,
    nql_agent_svc: NQLAgent = Depends(get_nql_agent) # Use the alias consistently
):
    """Executes a Natural Query Language (NQL) query against TensorStorage."""
    try:
        # Process the query using the NQL agent
        nql_result = nql_agent_svc.process_query(request.query)

        # Convert Tensorus internal results to API response format if successful
        output_results = None
        if nql_result.get('success') and nql_result.get('results'):
            output_results = []
            for record in nql_result['results']:
                 shape, dtype, data_list = tensor_to_list(record['tensor'])
                 output_results.append(TensorOutput(
                     record_id=record['metadata'].get('record_id', 'N/A'),
                     shape=shape,
                     dtype=dtype,
                     data=data_list,
                     metadata=record['metadata']
                 ))

        # Map NQL agent response to API response model
        response = NQLResponse(
             success=nql_result['success'],
             message=nql_result['message'],
             count=nql_result.get('count'),
             results=output_results
        )

        # If NQL parsing failed, return 400 Bad Request
        if not response.success:
            # Consider distinguishing parsing errors (400) from execution errors (could be 500 or 404)
            raise HTTPException(status_code=400, detail=response.message)

        return response

    except HTTPException as e:
         # Re-raise HTTPException if already handled (like the 400 above)
         raise e
    except Exception as e:
         logger.exception(f"Unexpected error processing NQL query '{request.query}': {e}")
         raise HTTPException(status_code=500, detail="An internal server error occurred during query processing.")


@app.get("/", include_in_schema=False) # Hide from OpenAPI docs
async def read_root():
    return {"message": "Welcome to the Tensorus API! Visit /docs for documentation."}


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Tensorus API Server ---")
    print("Access the API documentation at http://127.0.0.1:8000/docs")
    # Use uvicorn to run the FastAPI app
    # `reload=True` enables auto-reloading during development
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)