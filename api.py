# api.py

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import random # For simulating logs/status
import time # For simulating timestamps
import math # For simulating metrics
import asyncio # Added for potential background tasks later

import torch
import uvicorn
# Added Query import
from fastapi import FastAPI, HTTPException, Body, Depends, Path, status, Query
from pydantic import BaseModel, Field

# Import Tensorus modules - Ensure these files exist in your project path
try:
    from tensor_storage import TensorStorage
    from nql_agent import NQLAgent
    # Import other agents if needed for direct control (less common for API layer)
    # from ingestion_agent import DataIngestionAgent
    # from rl_agent import RLAgent
    # from automl_agent import AutoMLAgent
except ImportError as e:
    print(f"ERROR: Could not import Tensorus modules (TensorStorage, NQLAgent): {e}")
    print("Please ensure tensor_storage.py and nql_agent.py are in the Python path.")
    # Optionally raise the error or exit if these are critical at startup
    raise

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Tensorus Instances ---
try:
    # Ensure TensorStorage and NQLAgent can be instantiated without arguments
    # or provide necessary configuration here.
    tensor_storage_instance = TensorStorage()
    nql_agent_instance = NQLAgent(tensor_storage_instance)
    logger.info("Tensorus components (TensorStorage, NQLAgent) initialized successfully.")
    # NOTE: Actual agent processes (Ingestion, RL, AutoML) are assumed to be
    # running independently for now. This API layer will *coordinate* with them
    # in a full implementation, but currently only manages placeholder state.
except Exception as e:
    logger.exception(f"Failed to initialize Tensorus components: {e}")
    # This is critical, so raise an error to prevent the API from starting incorrectly.
    raise RuntimeError(f"Tensorus initialization failed: {e}") from e

# --- Placeholder Agent State Management ---
# In a real system, this would interact with a process manager, message queue,
# or shared state store (like Redis) to control and monitor actual agent processes.
agent_registry = {
    "ingestion": {
        "name": "Data Ingestion",
        "description": "Monitors sources and ingests data into datasets.",
        "status": "stopped", # Possible statuses: running, stopped, error, starting, stopping
        "config": {"source_directory": "temp_ingestion_source", "polling_interval_sec": 10},
        "last_log_timestamp": None,
        # Add simulation state if needed by metrics endpoint
    },
    "rl_trainer": {
        "name": "RL Trainer",
        "description": "Trains reinforcement learning models using stored experiences.",
        "status": "stopped",
        "config": {"experience_dataset": "rl_experiences", "batch_size": 128, "target_update_freq": 500},
        "last_log_timestamp": None,
        "sim_steps": 0, # Added for metrics simulation
    },
    "automl_search": {
        "name": "AutoML Search",
        "description": "Performs hyperparameter optimization.",
        "status": "stopped",
        "config": {"trials": 50, "results_dataset": "automl_results", "task_type": "regression"},
        "last_log_timestamp": None,
        "sim_trials": 0, # Added for metrics simulation
        "sim_best_score": None, # Added for metrics simulation
    },
     # NQL Agent is stateless, typically part of API request/response, but could be listed
     "nql_query": {
         "name": "NQL Query Service",
         "description": "Processes natural language queries.",
         "status": "running", # Assumed always running as part of API
         "config": {"parser_type": "regex"}, # Example config
         "last_log_timestamp": None,
     },
}

def _simulate_agent_log(agent_id: str) -> str:
    """Generates a simulated log line."""
    # Check if agent_id exists to prevent KeyError
    if agent_id not in agent_registry:
        logger.warning(f"Attempted to simulate log for unknown agent_id: {agent_id}")
        return f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} [ERROR] (Unknown Agent: {agent_id}) Log simulation failed."

    log_levels = ["INFO", "DEBUG", "WARNING", "ERROR"]
    messages = [
        "Processing item batch", "Training epoch completed", "Search trial finished",
        "Connection error detected", "Optimization step", "Query received",
        "Target network synced", "Disk space low", "Operation successful", "Agent starting up",
        "Configuration loaded", "Model evaluated", "Experience stored", "Found new file"
    ]
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    level = random.choice(log_levels)
    msg = random.choice(messages)
    # Safely get the agent name using .get()
    agent_name = agent_registry.get(agent_id, {}).get("name", agent_id) # Default to id if name missing
    return f"{ts} [{level}] ({agent_name}) {msg}"


# --- Helper Functions (Tensor Conversion) ---
def _validate_tensor_data(data: List[Any], shape: List[int]):
    """Validates nested list structure against a given shape (recursive)."""
    if not shape: # Scalar case
         if not isinstance(data, (int, float)):
             raise ValueError("Scalar tensor data must be a single number (int or float).")
         return True
    if not isinstance(data, list):
        # Improved error message
        raise ValueError(f"Data for shape {shape} must be a list, but got {type(data).__name__}.")

    expected_len = shape[0]
    if len(data) != expected_len:
        raise ValueError(f"Dimension 0 mismatch for shape {shape}: Expected length {expected_len}, got {len(data)}.")

    if len(shape) > 1: # Recurse for inner dimensions
         for item in data:
             _validate_tensor_data(item, shape[1:])
    elif len(shape) == 1: # Innermost dimension, check element types
        if not all(isinstance(x, (int, float)) for x in data):
            # Find the first non-numeric type for a more specific error
            first_bad_type = next((type(x).__name__ for x in data if not isinstance(x, (int, float))), "unknown")
            raise ValueError(f"Innermost list elements must be numbers (int or float), found type '{first_bad_type}'.")
    return True

def list_to_tensor(shape: List[int], dtype_str: str, data: Union[List[Any], int, float]) -> torch.Tensor:
    """Converts a nested list or scalar to a PyTorch tensor with validation."""
    try:
        dtype_map = {
            'float32': torch.float32, 'float': torch.float,
            'float64': torch.float64, 'double': torch.double,
            'int32': torch.int32, 'int': torch.int,
            'int64': torch.int64, 'long': torch.long,
            'bool': torch.bool
        }
        torch_dtype = dtype_map.get(dtype_str.lower())
        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype string: '{dtype_str}'. Supported: {list(dtype_map.keys())}")

        # Optional: Perform strict validation before torch.tensor()
        # This can catch structure errors early but might be redundant with torch.tensor checks.
        # try:
        #     _validate_tensor_data(data, shape)
        # except ValueError as val_err:
        #     logger.error(f"Input data validation failed for shape {shape}: {val_err}")
        #     raise ValueError(f"Input data validation failed: {val_err}") from val_err

        # Let torch handle initial conversion and type checking
        tensor = torch.tensor(data, dtype=torch_dtype)

        # Verify shape after creation and attempt reshape if necessary
        if list(tensor.shape) != shape:
             logger.warning(f"Created tensor shape {list(tensor.shape)} differs from requested {shape}. Attempting reshape.")
             try:
                 tensor = tensor.reshape(shape)
                 logger.info(f"Reshape successful to {shape}.")
             except RuntimeError as reshape_err:
                 # Include original error for better debugging
                 logger.error(f"Reshape failed: {reshape_err}")
                 # Improved error message
                 raise ValueError(f"Created tensor shape {list(tensor.shape)} != requested {shape} and reshape failed: {reshape_err}") from reshape_err
        return tensor
    except (TypeError, ValueError) as e:
        # Catch specific conversion errors
        logger.error(f"Error converting list to tensor: {e}. Shape: {shape}, Dtype: {dtype_str}, Data type: {type(data).__name__}")
        raise ValueError(f"Failed tensor conversion: {e}") from e
    except Exception as e:
         # Catch any other unexpected errors
         logger.exception(f"Unexpected error during list_to_tensor: {e}", exc_info=True)
         raise ValueError(f"Unexpected tensor conversion error: {e}") from e

def tensor_to_list(tensor: torch.Tensor) -> Tuple[List[int], str, Union[List[Any], int, float]]:
    """Converts a PyTorch tensor back to shape, dtype string, and nested list/scalar."""
    if not isinstance(tensor, torch.Tensor):
        # More specific error message
        raise TypeError(f"Input must be a PyTorch Tensor, got {type(tensor).__name__}")
    shape = list(tensor.shape)
    # Robustly get dtype string, removing the 'torch.' prefix
    dtype_str = str(tensor.dtype).replace('torch.', '')
    # Handle 0-dim tensors (scalars) correctly
    if tensor.ndim == 0:
        data = tensor.item() # Extract scalar value
    else:
        data = tensor.tolist() # Convert multi-dim tensor to nested list
    return shape, dtype_str, data

# --- Pydantic Models ---
class DatasetCreateRequest(BaseModel):
    name: str = Field(..., description="Unique name for the new dataset.", example="my_image_dataset")

class TensorInput(BaseModel):
    shape: List[int] = Field(..., description="Shape of the tensor (e.g., [height, width, channels]).", example=[2, 3])
    dtype: str = Field(..., description="Data type (e.g., 'float32', 'int64', 'bool').", example="float32")
    data: Union[List[Any], int, float] = Field(..., description="Tensor data as a nested list for multi-dim tensors, or a single number for scalars.", example=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional key-value metadata.", example={"source": "api_ingest", "timestamp": 1678886400})

class NQLQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query string.", example="find image tensors from 'my_image_dataset' where metadata.source = 'web_scrape'")

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

class ApiResponse(BaseModel):
    success: bool = Field(..., description="Indicates if the API operation was successful.")
    message: str = Field(..., description="A descriptive status message.")
    data: Optional[Any] = Field(None, description="Optional data payload relevant to the operation (e.g., record_id, list of names).")

# --- NEW Pydantic Models for Agents ---
class AgentInfo(BaseModel):
    id: str = Field(..., description="Unique identifier for the agent (e.g., 'ingestion', 'rl_trainer').")
    name: str = Field(..., description="User-friendly display name of the agent.")
    description: str = Field(..., description="Brief description of the agent's purpose.")
    status: str = Field(..., description="Current operational status (e.g., running, stopped, error, starting, stopping).")
    config: Dict[str, Any] = Field(..., description="Current configuration parameters the agent is using.")

class AgentStatus(AgentInfo):
     # Inherits fields from AgentInfo
     last_log_timestamp: Optional[float] = Field(None, description="Unix timestamp of the last known log message received or generated for this agent.")

class AgentLogResponse(BaseModel):
     logs: List[str] = Field(..., description="List of recent log entries for the agent.")

# --- NEW Pydantic Model for Dashboard Metrics ---
class DashboardMetrics(BaseModel):
    timestamp: float = Field(..., description="Unix timestamp when the metrics were generated (UTC).")
    dataset_count: int = Field(..., description="Total number of datasets currently managed.")
    total_records_est: int = Field(..., description="Estimated total number of tensor records across all datasets (Simulated).")
    agent_status_summary: Dict[str, int] = Field(..., description="Summary count of agents grouped by their status.")
    data_ingestion_rate: float = Field(..., description="Simulated data ingestion rate (records/sec).")
    avg_query_latency_ms: float = Field(..., description="Simulated average NQL query processing latency (ms).")
    rl_latest_reward: Optional[float] = Field(None, description="Simulated latest reward value obtained by the RL trainer.")
    rl_total_steps: int = Field(..., description="Simulated total training steps taken by the RL trainer.")
    automl_best_score: Optional[float] = Field(None, description="Simulated best score found by the AutoML search so far.")
    automl_trials_completed: int = Field(..., description="Simulated number of AutoML trials completed.")
    system_cpu_usage_percent: float = Field(..., description="Simulated overall system CPU usage percentage.")
    system_memory_usage_percent: float = Field(..., description="Simulated overall system memory usage percentage.")

# --- FastAPI App Instance ---
app = FastAPI(
    title="Tensorus API",
    description="API for interacting with the Tensorus Agentic Tensor Database/Data Lake. Includes dataset management, NQL querying, and agent control placeholders.",
    version="0.2.1", # Incremented version for fixes
    # Add contact, license info if desired
    # contact={
    #     "name": "API Support",
    #     "url": "http://example.com/support",
    #     "email": "support@example.com",
    # },
    # license_info={
    #     "name": "Apache 2.0",
    #     "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    # },
)

# --- Dependency Functions ---
async def get_tensor_storage() -> TensorStorage:
    """Dependency function to get the global TensorStorage instance."""
    # In a more complex app, this might involve connection pooling or session management
    return tensor_storage_instance

async def get_nql_agent() -> NQLAgent:
    """Dependency function to get the global NQLAgent instance."""
    return nql_agent_instance


# --- API Endpoints ---

# --- Dataset Management Endpoints ---
@app.post("/datasets/create", response_model=ApiResponse, status_code=status.HTTP_201_CREATED, tags=["Datasets"])
async def create_dataset(req: DatasetCreateRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """
    Creates a new, empty dataset with the specified unique name.

    - **req**: Request body containing the dataset name.
    - **storage**: Injected TensorStorage instance.
    \f
    Raises HTTPException:
    - 409 Conflict: If a dataset with the same name already exists.
    - 500 Internal Server Error: For unexpected errors during creation.
    """
    try:
        # Assuming TensorStorage.create_dataset raises ValueError if exists
        storage.create_dataset(req.name)
        logger.info(f"Dataset '{req.name}' created successfully.")
        return ApiResponse(success=True, message=f"Dataset '{req.name}' created successfully.")
    except ValueError as e:
        # Catch specific error for existing dataset
        logger.warning(f"Attempted to create existing dataset '{req.name}': {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
         # Catch any other unexpected storage errors
         logger.exception(f"Unexpected error creating dataset '{req.name}': {e}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while creating dataset.")

@app.post("/datasets/{name}/ingest", response_model=ApiResponse, status_code=status.HTTP_201_CREATED, tags=["Data Ingestion"])
async def ingest_tensor(
    name: str = Path(..., description="The name of the target dataset for ingestion."),
    tensor_input: TensorInput = Body(..., description="The tensor data and metadata to ingest."),
    storage: TensorStorage = Depends(get_tensor_storage)
):
    """
    Ingests a single tensor (provided in JSON format) into the specified dataset.

    - **name**: Path parameter for the target dataset name.
    - **tensor_input**: Request body containing shape, dtype, data, and optional metadata.
    - **storage**: Injected TensorStorage instance.
    \f
    Raises HTTPException:
    - 400 Bad Request: If tensor data is invalid, shape/dtype mismatch, or other validation errors occur.
    - 404 Not Found: If the specified dataset name does not exist.
    - 500 Internal Server Error: For unexpected storage or processing errors.
    """
    try:
        # Convert incoming list/scalar data to a tensor
        tensor = list_to_tensor(tensor_input.shape, tensor_input.dtype, tensor_input.data)

        # Insert into storage, assuming it returns a unique record ID
        # Also assuming storage.insert raises ValueError if dataset 'name' not found
        record_id = storage.insert(name, tensor, tensor_input.metadata)

        logger.info(f"Ingested tensor into dataset '{name}' with record_id: {record_id}")
        return ApiResponse(success=True, message="Tensor ingested successfully.", data={"record_id": record_id})

    except ValueError as e: # Catch errors from list_to_tensor or storage.insert
        logger.error(f"ValueError during ingestion into '{name}': {e}")
        # Differentiate between bad data and dataset not found
        # Suggestion: Modify TensorStorage to raise specific exceptions (e.g., DatasetNotFoundError)
        # for more robust error handling here instead of string matching.
        if "Dataset not found" in str(e) or "does not exist" in str(e): # Adapt based on TensorStorage's error messages
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset '{name}' not found.")
        else: # Assume other ValueErrors are due to bad input data/shape/dtype
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid tensor data or parameters: {e}")
    except TypeError as e: # Catch potential type errors during tensor creation
        logger.error(f"TypeError during ingestion into '{name}': {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid data type provided: {e}")
    except Exception as e:
         logger.exception(f"Unexpected error ingesting into dataset '{name}': {e}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during ingestion.")

@app.get("/datasets/{name}/fetch", response_model=ApiResponse, tags=["Data Retrieval"])
async def fetch_dataset(
    name: str = Path(..., description="The name of the dataset to fetch records from."),
    storage: TensorStorage = Depends(get_tensor_storage)
):
    """
    Retrieves all records (including tensor data and metadata) from a specified dataset.

    - **name**: Path parameter for the dataset name.
    - **storage**: Injected TensorStorage instance.
    \f
    Returns:
    - ApiResponse containing a list of TensorOutput objects in the 'data' field.

    Raises HTTPException:
    - 404 Not Found: If the dataset name does not exist.
    - 500 Internal Server Error: For unexpected errors during retrieval or data conversion.
    """
    try:
        # Assuming get_dataset_with_metadata returns list of dicts {'tensor': ..., 'metadata': ...}
        # or raises ValueError if dataset not found
        records = storage.get_dataset_with_metadata(name)
        output_records = []
        processed_count = 0
        skipped_count = 0

        for i, record in enumerate(records):
             # Ensure 'tensor' and 'metadata' keys exist in each record from storage
             if not isinstance(record, dict) or 'tensor' not in record or 'metadata' not in record: # Added type check
                 logger.warning(f"Skipping record index {i} in '{name}' due to missing keys or invalid format.")
                 skipped_count += 1
                 continue
             try:
                 # Convert tensor back to list format for JSON response
                 shape, dtype, data_list = tensor_to_list(record['tensor'])
                 # Ensure record_id is present in metadata, provide a default if missing
                 record_id = record['metadata'].get('record_id', f"missing_id_{random.randint(1000,9999)}_{i}")
                 if record_id.startswith("missing_id_"):
                     logger.warning(f"Record index {i} in '{name}' missing 'record_id' in metadata.")

                 output_records.append(TensorOutput(
                     record_id=record_id,
                     shape=shape,
                     dtype=dtype,
                     data=data_list,
                     metadata=record['metadata']
                 ))
                 processed_count += 1
             except Exception as conversion_err:
                 rec_id_for_log = record.get('metadata', {}).get('record_id', f'index_{i}')
                 logger.error(f"Error converting tensor to list for record '{rec_id_for_log}' in dataset '{name}': {conversion_err}", exc_info=True) # Added exc_info
                 skipped_count += 1
                 # Optionally skip problematic records or handle differently

        log_message = f"Fetched dataset '{name}'. Processed: {processed_count}, Skipped: {skipped_count}."
        logger.info(log_message)
        return ApiResponse(success=True, message=log_message, data=output_records)

    except ValueError as e: # Typically "Dataset not found" from storage
        logger.warning(f"Attempted to fetch non-existent dataset '{name}': {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
         logger.exception(f"Unexpected error fetching dataset '{name}': {e}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while fetching dataset.")

@app.get("/datasets", response_model=ApiResponse, tags=["Datasets"])
async def list_datasets(storage: TensorStorage = Depends(get_tensor_storage)):
    """
    Lists the names of all available datasets managed by the TensorStorage.

    - **storage**: Injected TensorStorage instance.
    \f
    Returns:
    - ApiResponse containing a list of dataset names in the 'data' field.

    Raises HTTPException:
    - 500 Internal Server Error: If there's an error retrieving the list from storage.
    """
    try:
        # Adapt this call based on your TensorStorage implementation
        # Example: dataset_names = storage.list_datasets()
        # Example: dataset_names = list(storage.datasets.keys()) # If it's a simple dict
        if hasattr(storage, 'list_datasets') and callable(storage.list_datasets): # Check if callable
            dataset_names = storage.list_datasets()
        elif hasattr(storage, 'datasets') and isinstance(storage.datasets, dict):
             dataset_names = list(storage.datasets.keys())
        else:
            # Improved error message
            logger.error("TensorStorage instance does not have a recognized method (list_datasets) or attribute (datasets dict) to list datasets.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="API configuration error: Cannot list datasets.")

        logger.info(f"Retrieved dataset list: Count={len(dataset_names)}")
        return ApiResponse(success=True, message="Retrieved dataset list successfully.", data=dataset_names)
    except Exception as e:
         logger.exception(f"Unexpected error listing datasets: {e}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while listing datasets.")

# --- Querying Endpoint ---
@app.post("/query", response_model=NQLResponse, tags=["Querying"])
async def execute_nql_query(
    request: NQLQueryRequest,
    nql_agent_svc: NQLAgent = Depends(get_nql_agent)
):
    """
    Executes a Natural Query Language (NQL) query against the stored tensor data.

    - **request**: Request body containing the NQL query string.
    - **nql_agent_svc**: Injected NQLAgent instance.
    \f
    Returns:
    - NQLResponse containing the query success status, message, count, and results.

    Raises HTTPException:
    - 400 Bad Request: If the NQL query is invalid or fails processing as reported by the NQLAgent.
    - 500 Internal Server Error: For unexpected errors during query processing or result conversion.
    """
    logger.info(f"Received NQL query: '{request.query}'")
    try:
        # Assuming process_query returns a dict like:
        # {'success': bool, 'message': str, 'count': Optional[int], 'results': Optional[List[dict]]}
        # where each dict in 'results' has 'tensor' and 'metadata' keys.
        nql_result = nql_agent_svc.process_query(request.query)

        output_results = None
        processed_count = 0
        skipped_count = 0

        if nql_result.get('success') and isinstance(nql_result.get('results'), list):
            output_results = []
            for i, record in enumerate(nql_result['results']):
                 # Basic validation of expected keys in results from NQLAgent
                 if not isinstance(record, dict) or 'tensor' not in record or 'metadata' not in record:
                     logger.warning(f"Skipping NQL result record index {i} due to missing keys or invalid format.")
                     skipped_count += 1
                     continue
                 try:
                     # Convert tensor to list for response
                     shape, dtype, data_list = tensor_to_list(record['tensor'])
                     # Ensure record_id exists, provide default
                     record_id = record['metadata'].get('record_id', f"missing_id_{random.randint(1000,9999)}_{i}")
                     if record_id.startswith("missing_id_"):
                         logger.warning(f"NQL result record index {i} missing 'record_id' in metadata.")

                     output_results.append(TensorOutput(
                         record_id=record_id,
                         shape=shape,
                         dtype=dtype,
                         data=data_list,
                         metadata=record['metadata']
                     ))
                     processed_count += 1
                 except Exception as conversion_err:
                     rec_id_for_log = record.get('metadata', {}).get('record_id', f'index_{i}')
                     logger.error(f"Error converting tensor to list for NQL result record '{rec_id_for_log}': {conversion_err}", exc_info=True) # Added exc_info
                     skipped_count += 1
                     continue # Skip problematic records

        # Construct response using Pydantic model for validation
        # Safely get values from nql_result dict
        response = NQLResponse(
            success=nql_result.get('success', False),
            message=nql_result.get('message', 'Error: Query processing failed unexpectedly.'),
            count=nql_result.get('count', processed_count if output_results is not None else None), # Use processed count if available
            results=output_results
        )

        if not response.success:
            logger.warning(f"NQL query failed: '{request.query}'. Reason: {response.message}")
            # Return 400 for query parsing/execution issues reported by the agent
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=response.message)

        log_message = f"NQL query successful: '{request.query}'. Found: {response.count}, Processed: {processed_count}, Skipped: {skipped_count}."
        logger.info(log_message)
        # Optionally update response message if counts differ significantly
        # response.message = log_message
        return response

    except HTTPException as e:
        # Re-raise HTTPExceptions that were already handled (like the 400 above)
        raise e
    except Exception as e:
         # Catch unexpected errors during query processing or result conversion
         logger.exception(f"Unexpected error processing NQL query '{request.query}': {e}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during query processing.")


# --- Agent Control Endpoints ---

@app.get("/agents", response_model=List[AgentInfo], tags=["Agents"])
async def list_agents():
    """
    Lists all registered agents and their basic information (name, description, status, config).
    Reads data from the global `agent_registry`.
    \f
    Returns:
    - A list of AgentInfo objects.

    Raises HTTPException:
    - 500 Internal Server Error: If the agent registry is unexpectedly unavailable or malformed.
    """
    try:
        agents_list = []
        for agent_id, details in agent_registry.items():
            # Validate expected keys before creating AgentInfo to prevent Pydantic errors
            if not isinstance(details, dict) or not all(k in details for k in ["name", "description", "status", "config"]):
                 # More detailed logging
                 logger.warning(f"Agent '{agent_id}' in registry is missing required keys or is not a dict. Details: {details}. Skipping.")
                 continue
            try:
                agents_list.append(AgentInfo(
                    id=agent_id,
                    name=details["name"],
                    description=details["description"],
                    status=details["status"],
                    config=details["config"]
                ))
            except Exception as pydantic_err: # Catch potential Pydantic validation errors
                 logger.error(f"Error creating AgentInfo for agent '{agent_id}': {pydantic_err}. Details: {details}", exc_info=True) # Added exc_info
                 continue # Skip malformed entries

        logger.info(f"Retrieved list of {len(agents_list)} agents.")
        return agents_list
    except Exception as e:
        # Catch errors iterating the registry itself
        logger.exception(f"Unexpected error listing agents: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error listing agents.")


@app.get("/agents/{agent_id}/status", response_model=AgentStatus, tags=["Agents"])
async def get_agent_status_api(agent_id: str = Path(..., description="The unique identifier of the agent.")):
    """
    Gets the current status, configuration, and last log timestamp for a specific agent.
    Reads data from the global `agent_registry`.

    - **agent_id**: Path parameter for the agent's unique ID.
    \f
    Returns:
    - AgentStatus object containing the agent's details.

    Raises HTTPException:
    - 404 Not Found: If the agent_id does not exist in the registry.
    - 500 Internal Server Error: If the agent's entry in the registry is malformed.
    """
    logger.debug(f"Request received for status of agent '{agent_id}'.")
    if agent_id not in agent_registry:
        logger.warning(f"Status requested for unknown agent '{agent_id}'.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")

    details = agent_registry[agent_id]
    # Basic validation of expected keys
    if not isinstance(details, dict) or not all(k in details for k in ["name", "description", "status", "config"]):
        logger.error(f"Agent '{agent_id}' registry entry is malformed: {details}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error: Malformed status data for agent '{agent_id}'.")

    # Simulate potential status updates if needed (optional placeholder)
    # if details['status'] in ['starting', 'stopping']: details['status'] = random.choice(['running', 'stopped', 'error'])

    try:
        status_response = AgentStatus(
            id=agent_id,
            name=details["name"],
            description=details["description"],
            status=details["status"],
            config=details["config"],
            last_log_timestamp=details.get("last_log_timestamp") # Safely get optional field
        )
        logger.info(f"Returning status for agent '{agent_id}': {status_response.status}")
        return status_response
    except Exception as pydantic_err: # Catch potential Pydantic validation errors
        logger.error(f"Error creating AgentStatus response for agent '{agent_id}': {pydantic_err}. Details: {details}", exc_info=True) # Added exc_info
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error creating status response for agent '{agent_id}'.")


@app.post("/agents/{agent_id}/start", response_model=ApiResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Agents"])
async def start_agent_api(agent_id: str = Path(..., description="The unique identifier of the agent to start.")):
    """
    Signals an agent to start its operation (Placeholder/Simulated).
    Updates the agent's status in the global `agent_registry`.

    - **agent_id**: Path parameter for the agent's unique ID.
    \f
    Returns:
    - ApiResponse indicating success or failure (if already running/starting).

    Raises HTTPException:
    - 404 Not Found: If the agent_id does not exist.
    """
    logger.info(f"Received start signal for agent '{agent_id}'.")
    if agent_id not in agent_registry:
        logger.warning(f"Start signal received for unknown agent '{agent_id}'.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")

    # Check current status before attempting start
    current_status = agent_registry[agent_id].get("status", "unknown")
    if current_status in ["running", "starting"]:
         logger.info(f"Agent '{agent_id}' is already {current_status}. No action taken.")
         # Return success=False for idempotent-like behavior
         return ApiResponse(success=False, message=f"Agent '{agent_id}' is already {current_status}.")
    if current_status == "error":
        # Added logging for starting from error state
        logger.warning(f"Attempting to start agent '{agent_id}' which is in 'error' state. Resetting status.")
        # Decide if starting from error state is allowed/needs special handling

    logger.info(f"API: Processing start signal for agent '{agent_id}' (Placeholder Action).")
    # Simulate state change - In reality, trigger async start process
    agent_registry[agent_id]["status"] = "starting"
    # TODO: Implement actual agent process starting logic (e.g., message queue, process manager call, background task).
    # Simulate transition to running after a short delay in a real scenario
    # For now, just accept the request and simulate immediate change for simplicity.
    await asyncio.sleep(0.1) # Tiny delay to simulate transition time if desired
    agent_registry[agent_id]["status"] = "running" # Immediate simulation for now
    agent_registry[agent_id]["last_log_timestamp"] = time.time() # Update timestamp on action
    logger.info(f"Agent '{agent_id}' status set to 'running' (simulated).")
    return ApiResponse(success=True, message=f"Start signal sent to agent '{agent_id}'. Status is now 'running' (simulated).")

@app.post("/agents/{agent_id}/stop", response_model=ApiResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Agents"])
async def stop_agent_api(agent_id: str = Path(..., description="The unique identifier of the agent to stop.")):
    """
    Signals an agent to stop its operation gracefully (Placeholder/Simulated).
    Updates the agent's status in the global `agent_registry`.

    - **agent_id**: Path parameter for the agent's unique ID.
    \f
    Returns:
    - ApiResponse indicating success or failure (if already stopped/stopping).

    Raises HTTPException:
    - 404 Not Found: If the agent_id does not exist.
    """
    logger.info(f"Received stop signal for agent '{agent_id}'.")
    if agent_id not in agent_registry:
        logger.warning(f"Stop signal received for unknown agent '{agent_id}'.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")

    # Check current status before attempting stop
    current_status = agent_registry[agent_id].get("status", "unknown")
    # Consider 'error' as effectively stopped for control purposes, or handle separately if needed
    if current_status in ["stopped", "stopping"]:
         # Improved logging
         logger.info(f"Agent '{agent_id}' is already {current_status}. No action taken.")
         return ApiResponse(success=False, message=f"Agent '{agent_id}' is already {current_status}.")

    logger.info(f"API: Processing stop signal for agent '{agent_id}' (Placeholder Action).")
    # Simulate state change - In reality, trigger async stop process
    agent_registry[agent_id]["status"] = "stopping"
    # TODO: Implement actual agent process stopping logic (e.g., sending signal, waiting for confirmation).
    # Simulate transition to stopped after a short delay in a real scenario
    await asyncio.sleep(0.1) # Tiny delay
    agent_registry[agent_id]["status"] = "stopped" # Immediate simulation for now
    agent_registry[agent_id]["last_log_timestamp"] = time.time() # Update timestamp on action
    logger.info(f"Agent '{agent_id}' status set to 'stopped' (simulated).")
    return ApiResponse(success=True, message=f"Stop signal sent to agent '{agent_id}'. Status is now 'stopped' (simulated).")


@app.get("/agents/{agent_id}/logs", response_model=AgentLogResponse, tags=["Agents"])
async def get_agent_logs_api(
    agent_id: str = Path(..., description="The unique identifier of the agent."),
    lines: int = Query(20, ge=1, le=1000, description="Maximum number of recent log lines to retrieve.") # Added Query validation
):
    """
    Retrieves recent logs for a specific agent (Simulated - generates new logs each time).

    - **agent_id**: Path parameter for the agent's unique ID.
    - **lines**: Query parameter for the number of log lines (default 20, min 1, max 1000).
    \f
    Returns:
    - AgentLogResponse containing a list of simulated log strings.

    Raises HTTPException:
    - 404 Not Found: If the agent_id does not exist.
    - 500 Internal Server Error: If log generation fails.
    """
    logger.debug(f"Request received for logs of agent '{agent_id}' (lines={lines}).")
    if agent_id not in agent_registry:
        logger.warning(f"Log request for unknown agent '{agent_id}'.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")

    # Parameter 'lines' is already validated by FastAPI/Pydantic via Query(ge=1, le=1000)

    # TODO: Implement actual log retrieval from agent process, file, or logging service.
    # This simulation generates new logs each time, it doesn't store/retrieve history.
    try:
        simulated_logs = [_simulate_agent_log(agent_id) for _ in range(lines)]
        agent_registry[agent_id]["last_log_timestamp"] = time.time() # Update timestamp on access
        logger.info(f"Generated {len(simulated_logs)} simulated log lines for agent '{agent_id}'.")
        return AgentLogResponse(logs=simulated_logs)
    except Exception as e:
        logger.exception(f"Error generating simulated logs for agent '{agent_id}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error generating logs for agent '{agent_id}'.")


# --- Metrics & Monitoring Endpoint ---
@app.get("/metrics/dashboard", response_model=DashboardMetrics, tags=["Metrics & Monitoring"])
async def get_dashboard_metrics(storage: TensorStorage = Depends(get_tensor_storage)):
    """
    Provides aggregated dashboard metrics, combining real data (like dataset count)
    with simulated data for agent performance and system health.

    NOTE: This endpoint currently modifies simulation state (e.g., sim_steps)
    within a GET request, which is not ideal REST practice. For production,
    simulation updates should occur in a background task.

    - **storage**: Injected TensorStorage instance.
    \f
    Returns:
    - DashboardMetrics object containing various metrics.

    Raises HTTPException:
    - 500 Internal Server Error: If critical metrics cannot be retrieved or calculated.
    """
    logger.debug("Request received for dashboard metrics.")
    current_time = time.time()
    metrics_data = {} # Use a dict to build metrics before creating the Pydantic model

    # --- Real Metrics (with error handling) ---
    try:
        # Adapt based on your actual TensorStorage implementation
        if hasattr(storage, 'list_datasets') and callable(storage.list_datasets):
            dataset_count = len(storage.list_datasets())
        elif hasattr(storage, 'datasets') and isinstance(storage.datasets, dict):
             dataset_count = len(storage.datasets.keys())
        else:
            logger.error("TensorStorage instance lacks list_datasets() method or datasets dict.")
            dataset_count = -1 # Indicate error
        metrics_data["dataset_count"] = dataset_count
        logger.debug(f"Retrieved dataset count: {dataset_count}")
    except Exception as e:
        logger.exception(f"Failed to get dataset count for metrics: {e}")
        metrics_data["dataset_count"] = -1 # Indicate error fetching

    # --- Simulated/Placeholder Metrics ---
    # TODO: Replace simulations with actual metric collection from agents/storage/system.

    # Agent Status Summary (from placeholder registry)
    status_counts = {"running": 0, "stopped": 0, "error": 0, "starting": 0, "stopping": 0, "unknown": 0}
    for agent_id, details in agent_registry.items():
        status = details.get("status", "unknown")
        if status not in status_counts:
            logger.warning(f"Agent '{agent_id}' has unexpected status '{status}'. Counting as 'unknown'.")
            status = "unknown"
        status_counts[status] += 1
    metrics_data["agent_status_summary"] = status_counts

    # Simulate Total Records (Only estimate if dataset_count is valid)
    ds_count = metrics_data.get("dataset_count", -1)
    metrics_data["total_records_est"] = ds_count * random.randint(500, 5000) if ds_count >= 0 else 0

    # Simulate performance metrics (slightly dynamic based on time/status)
    # Use .get() for safe access in case agents are removed from registry later
    ingestion_running = agent_registry.get("ingestion", {}).get("status") == "running"
    rl_running = agent_registry.get("rl_trainer", {}).get("status") == "running"
    automl_running = agent_registry.get("automl_search", {}).get("status") == "running"

    metrics_data["data_ingestion_rate"] = random.uniform(5.0, 50.0) * (1.0 if ingestion_running else 0.1)
    metrics_data["avg_query_latency_ms"] = random.uniform(50.0, 300.0) * (1 + 0.5 * math.sin(current_time / 60)) # Smoother oscillation

    # --- Simulation state update (WARNING: Modifies state in GET request) ---
    # This part modifies the global state. Better practice: use a background task.
    rl_agent_state = agent_registry.setdefault("rl_trainer", {"sim_steps": 0}) # Ensure key and sim_steps exist
    rl_total_steps = int(max(0, rl_agent_state.get("sim_steps", 0) + (random.randint(10, 150) if rl_running else 0))) # Adjusted range
    rl_agent_state["sim_steps"] = rl_total_steps # Store simulated steps back
    metrics_data["rl_total_steps"] = rl_total_steps
    metrics_data["rl_latest_reward"] = random.gauss(10, 5.0) if rl_running else None # Example reward distribution

    automl_agent_state = agent_registry.setdefault("automl_search", {"sim_trials": 0, "sim_best_score": None}) # Ensure keys exist
    automl_trials_completed = int(max(0, automl_agent_state.get("sim_trials", 0) + (random.randint(0, 3) if automl_running else 0)))
    automl_agent_state["sim_trials"] = automl_trials_completed
    metrics_data["automl_trials_completed"] = automl_trials_completed

    current_best = automl_agent_state.get("sim_best_score", None)
    automl_best_score = None
    if automl_running:
        if current_best is None:
             automl_best_score = random.uniform(0.7, 0.95) # Example initial score (e.g., accuracy)
        else:
             # Simulate improvement (higher is better for this example score)
             improvement_factor = random.uniform(1.0, 1.005)
             automl_best_score = min(1.0, current_best * improvement_factor) # Cap at 1.0
        automl_agent_state["sim_best_score"] = automl_best_score # Store back
    elif current_best is not None:
         automl_best_score = current_best # Keep last known best if stopped
    metrics_data["automl_best_score"] = automl_best_score
    # --- End Simulation state update ---

    # Simulate System Health (with bounds checks)
    cpu_load = random.uniform(5.0, 25.0) \
               + (15 if ingestion_running else 0) \
               + (25 if rl_running else 0) \
               + (10 if automl_running else 0)
    # Ensure value is between 0 and 100
    metrics_data["system_cpu_usage_percent"] = min(100.0, max(0.0, cpu_load + random.uniform(-2.0, 2.0)))

    mem_load = random.uniform(15.0, 40.0) \
               + (metrics_data.get("dataset_count", 0) * 0.75) # Memory scales slightly with datasets
    # Ensure value is between 0 and 100
    metrics_data["system_memory_usage_percent"] = min(100.0, max(0.0, mem_load + random.uniform(-3.0, 3.0)))

    # --- Construct Response using Pydantic Model ---
    try:
        # Use the collected metrics_data dictionary
        metrics = DashboardMetrics(
            timestamp=current_time,
            dataset_count=metrics_data["dataset_count"],
            total_records_est=metrics_data["total_records_est"],
            agent_status_summary=metrics_data["agent_status_summary"],
            data_ingestion_rate=round(metrics_data["data_ingestion_rate"], 2),
            avg_query_latency_ms=round(metrics_data["avg_query_latency_ms"], 1),
            rl_latest_reward=(round(metrics_data["rl_latest_reward"], 3)
                              if metrics_data.get("rl_latest_reward") is not None else None),
            rl_total_steps=metrics_data["rl_total_steps"],
            automl_best_score=(round(metrics_data["automl_best_score"], 5)
                               if metrics_data.get("automl_best_score") is not None else None),
            automl_trials_completed=metrics_data["automl_trials_completed"],
            system_cpu_usage_percent=round(metrics_data["system_cpu_usage_percent"], 1),
            system_memory_usage_percent=round(metrics_data["system_memory_usage_percent"], 1)
        )
        logger.info("Successfully generated dashboard metrics.")
        return metrics
    except Exception as e:
        # Catch errors during final model creation (e.g., validation errors)
        logger.exception(f"Error constructing DashboardMetrics response from data: {metrics_data}. Error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error constructing metrics response.")


# --- Root Endpoint ---
@app.get("/", include_in_schema=False)
async def read_root():
    """Provides a simple welcome message for the API root."""
    # Useful for health checks or simple verification that the API is running
    return {"message": "Welcome to the Tensorus API! Visit /docs or /redoc for interactive documentation."}

# --- Main Execution Block ---
if __name__ == "__main__":
    # This block allows running the API directly using `python api.py`

    # Basic check for required local modules if run directly
    modules_ok = True
    try:
        from tensor_storage import TensorStorage
        from nql_agent import NQLAgent
    except ImportError as import_err:
        print(f"\nERROR: Missing required local modules: {import_err}.")
        print("Please ensure tensor_storage.py and nql_agent.py are in the same directory or Python path.\n")
        modules_ok = False
        # exit(1) # Exit if modules are absolutely critical for startup

    if modules_ok:
        print(f"--- Starting Tensorus API Server (v{app.version} with Agent Placeholders) ---")
        print(f"--- Logging level set to: {logging.getLevelName(logger.getEffectiveLevel())} ---")
        print(f"--- Access API documentation at http://127.0.0.1:8000/docs ---")
        print(f"--- Alternative documentation at http://127.0.0.1:8000/redoc ---")
        print("--- Press CTRL+C to stop ---")

        # Use uvicorn to run the app
        uvicorn.run(
            "api:app", # Points to the 'app' instance in the 'api.py' file
            host="127.0.0.1",
            port=8000,
            reload=True, # Enable auto-reload for development (watches for file changes)
            log_level=logging.getLevelName(logger.getEffectiveLevel()).lower(), # Sync uvicorn log level
            # Use workers > 1 only if your app is stateless or handles state carefully
            # workers=1
        )
    else:
        print("--- API Server NOT started due to missing modules. ---")
