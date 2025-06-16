# api.py

import logging
from typing import List, Dict, Any, Optional, Tuple, Union, TYPE_CHECKING
import random  # For simulating logs/status
import time  # For simulating timestamps
import math  # For simulating metrics
import asyncio  # For async operations
import os

import torch
from fastapi import (
    FastAPI, HTTPException, Body, Depends, Path, status,
    Query, APIRouter, Request, Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, root_validator

# Routers containing the metadata-related endpoints
from tensorus.api.endpoints import (
    router_tensor_descriptor,
    router_semantic_metadata,
    router_search_aggregate,
    router_version_lineage,
    router_extended_metadata,
    router_io,
    router_management,
    router_analytics,
)
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Custom exceptions
class APIError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)

class NotFoundError(APIError):
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(status_code=404, detail=detail)

class BadRequestError(APIError):
    def __init__(self, detail: str = "Bad request"):
        super().__init__(status_code=400, detail=detail)

# Custom middleware for logging
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url}")
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            if body and len(body) < 1000:  # Log body if not too large
                logger.debug(f"Request body: {body.decode()}")
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(f"Request failed: {str(exc)}", exc_info=True)
            if not isinstance(exc, APIError):
                exc = APIError(status_code=500, detail="Internal server error")
            response = JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail}
            )
        
        # Log response
        process_time = (time.time() - start_time) * 1000
        logger.info(
            f"Response: {request.method} {request.url} - "
            f"Status: {response.status_code} - {process_time:.2f}ms"
        )
        
        return response

# Security headers middleware
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"

    frame_opt = os.environ.get("TENSORUS_X_FRAME_OPTIONS", "SAMEORIGIN")
    csp = os.environ.get("TENSORUS_CONTENT_SECURITY_POLICY", "default-src 'self'")

    if frame_opt and frame_opt.upper() != "NONE":
        response.headers["X-Frame-Options"] = frame_opt
    else:
        if "X-Frame-Options" in response.headers:
            del response.headers["X-Frame-Options"]

    response.headers["X-XSS-Protection"] = "1; mode=block"

    if csp and csp.upper() != "NONE":
        response.headers["Content-Security-Policy"] = csp
    else:
        if "Content-Security-Policy" in response.headers:
            del response.headers["Content-Security-Policy"]

    return response

# --- FastAPI App Instance ---
app = FastAPI(
    title="Tensorus API",
    description=(
        "API for interacting with the Tensorus Agentic Tensor Database/Data Lake. "
        "Includes dataset management, NQL querying, and agent control."
    ),
    version="0.2.1",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    # contact={
    #     "name": "API Support",
    #     "url": "http://example.com/support",
    #     "email": "support@example.com",
    # },
    # license_info={
    #     "name": "Apache 2.0",
    #     "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    # }
)

# Add middleware
app.add_middleware(LoggingMiddleware)
app.middleware("http")(add_security_headers)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],  # In production, specify trusted hosts
)



# Import Tensorus modules - Ensure these files exist in your project path
try:
    from .tensor_storage import (
        TensorStorage,
        DatasetNotFoundError,
        TensorNotFoundError,
    )
    from .nql_agent import NQLAgent
    from .tensor_ops import TensorOps
    # from rl_agent import RLAgent
    # from automl_agent import AutoMLAgent
except ImportError as e:
    print(f"ERROR: Could not import Tensorus modules (TensorStorage, NQLAgent, TensorOps): {e}")
    print("Please ensure tensor_storage.py, nql_agent.py, and tensor_ops.py are in the Python path.")
    # Optionally raise the error or exit if these are critical at startup
    raise

if TYPE_CHECKING:
    # Import for type checking only to avoid heavy dependency during runtime
    from .ingestion_agent import DataIngestionAgent


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


# --- Agent State Management ---
# agent_registry holds static configuration and metadata.
# live_agents holds actual running instances of agents that are managed by this API.
agent_registry = {
    "ingestion": {
        "name": "Data Ingestion",
        "description": "Monitors sources and ingests data into TensorStorage.",
        "config": {
            "source_directory": "./temp_ingestion_source_api", # API managed source
            "polling_interval_sec": 15, # Slightly different default for API
            "dataset_name": "ingested_data_api" # API specific dataset
        },
        # No 'status' or 'last_log_timestamp' here; these are dynamic for live agents
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
     "nql_query": {
         "name": "NQL Query Service",
         "description": "Processes natural language queries.",
         "status": "running", # Assumed always running as part of API
         "config": {"parser_type": "regex"},
         "last_log_timestamp": None, # NQL agent is stateless, logs not stored this way
     },
}

live_agents: Dict[str, "DataIngestionAgent"] = {}  # Stores live agent instances, key is agent_id

def _get_or_create_ingestion_agent() -> "DataIngestionAgent":
    """
    Retrieves the existing DataIngestionAgent instance or creates a new one
    if it doesn't exist. Ensures its source directory is created.
    """
    from .ingestion_agent import DataIngestionAgent
    global live_agents
    if "ingestion" not in live_agents:
        if "ingestion" not in agent_registry:
            # This should not happen if agent_registry is correctly defined
            logger.error("Ingestion agent configuration missing in agent_registry.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Ingestion agent configuration missing.")

        config = agent_registry["ingestion"]["config"]
        source_dir = config["source_directory"]
        dataset_name = config["dataset_name"]
        polling_interval = config["polling_interval_sec"]

        # Ensure source directory exists
        try:
            os.makedirs(source_dir, exist_ok=True)
            logger.info(f"Ensured source directory exists: {source_dir}")
        except OSError as e:
            logger.error(f"Could not create source directory {source_dir}: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create source directory: {e}")

        logger.info(f"Creating DataIngestionAgent instance for dataset '{dataset_name}' monitoring '{source_dir}'.")
        live_agents["ingestion"] = DataIngestionAgent(
            tensor_storage=tensor_storage_instance, # Global instance
            dataset_name=dataset_name,
            source_directory=source_dir,
            polling_interval_sec=polling_interval
        )
    return live_agents["ingestion"]

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


# --- Background Simulation Task ---
async def _metrics_simulation_loop() -> None:
    """Periodically updates simulated metrics for placeholder agents."""
    while True:
        try:
            rl_state = agent_registry.get("rl_trainer")
            if rl_state is not None:
                rl_state["sim_steps"] = rl_state.get("sim_steps", 0) + random.randint(10, 150)

            automl_state = agent_registry.get("automl_search")
            if automl_state is not None:
                automl_state["sim_trials"] = automl_state.get("sim_trials", 0) + random.randint(0, 3)
                current_best = automl_state.get("sim_best_score")
                if current_best is None:
                    automl_state["sim_best_score"] = random.uniform(0.7, 0.95)
                else:
                    improvement = random.uniform(1.0, 1.005)
                    automl_state["sim_best_score"] = min(1.0, current_best * improvement)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.error(f"Metrics simulation loop error: {exc}")

        await asyncio.sleep(5)


@app.on_event("startup")
async def _start_simulation_task() -> None:
    """Launch background metric simulation when the app starts."""
    app.state.metrics_task = asyncio.create_task(_metrics_simulation_loop())


@app.on_event("shutdown")
async def _stop_simulation_task() -> None:
    """Cancel background metric simulation on shutdown."""
    task = getattr(app.state, "metrics_task", None)
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:  # pragma: no cover
            pass


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

        # Validate incoming list structure before constructing the tensor. This
        # ensures mismatched shapes are caught early rather than relying on
        # PyTorch's reshape logic.
        _validate_tensor_data(data, shape)

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

class UpdateMetadataRequest(BaseModel):
    new_metadata: Dict[str, Any] = Field(..., description="New metadata to update the tensor record with. This will replace the existing metadata.")

# --- Pydantic Models for Tensor Operations ---

class TensorRef(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset containing the tensor.")
    record_id: str = Field(..., description="Record ID of the tensor within the dataset.")

class TensorInputVal(BaseModel):
    tensor_ref: Optional[TensorRef] = None
    scalar_value: Optional[Union[float, int]] = None

    @root_validator(pre=True)
    def check_one_input_provided(cls, values):
        if sum(v is not None for v in values.values()) != 1:
            raise ValueError("Exactly one of 'tensor_ref' or 'scalar_value' must be provided.")
        return values

class OpsBaseRequest(BaseModel):
    output_dataset_name: Optional[str] = Field(None, description="Optional name for a new dataset to store the output tensor. If None, a default or existing dataset might be used by the operation.")
    output_metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata to associate with the output tensor.")

# Specific Unary Operation Parameter Models
class OpsReshapeRequestParams(BaseModel):
    new_shape: List[int] = Field(..., description="The new shape for the tensor.")

class OpsPermuteRequestParams(BaseModel):
    dims: List[int] = Field(..., description="A list of dimensions to permute to.")

class OpsTransposeRequestParams(BaseModel):
    dim0: int = Field(..., description="The first dimension to transpose.")
    dim1: int = Field(..., description="The second dimension to transpose.")

class OpsGetSingleDimensionParam(BaseModel):
    dim: Optional[Union[int, List[int]]] = Field(None, description="Dimension(s) to operate over. If None, operates over all dimensions.")
    keepdim: bool = Field(False, description="Whether the output tensor has dim retained or not.")

# Unary Operation Request Models
class OpsUnaryOpRequest(OpsBaseRequest):
    input_tensor: TensorRef = Field(..., description="Reference to the input tensor.")
    params: Optional[Dict[str, Any]] = Field(None, description="Parameters for the unary operation.")

class OpsReshapeRequest(OpsBaseRequest):
    input_tensor: TensorRef = Field(..., description="Reference to the input tensor for reshape.")
    params: OpsReshapeRequestParams

class OpsPermuteRequest(OpsBaseRequest):
    input_tensor: TensorRef = Field(..., description="Reference to the input tensor for permute.")
    params: OpsPermuteRequestParams

class OpsTransposeRequest(OpsBaseRequest):
    input_tensor: TensorRef = Field(..., description="Reference to the input tensor for transpose.")
    params: OpsTransposeRequestParams

class OpsReductionRequest(OpsBaseRequest): # For sum, mean
    input_tensor: TensorRef = Field(..., description="Reference to the input tensor for reduction (sum, mean).")
    params: OpsGetSingleDimensionParam

class OpsMinMaxRequest(OpsBaseRequest): # For min, max
    input_tensor: TensorRef = Field(..., description="Reference to the input tensor for min/max.")
    params: Optional[OpsGetSingleDimensionParam] = None

class OpsLogRequest(OpsBaseRequest):
    input_tensor: TensorRef = Field(..., description="Reference to the input tensor for logarithm.")
    # No specific params for basic log, could add base if needed

# Binary Operation Request Models
class OpsBinaryOpRequest(OpsBaseRequest): # For add, subtract, multiply, divide, matmul, dot
    input1: TensorRef = Field(..., description="Reference to the first input tensor.")
    input2: TensorInputVal = Field(..., description="Second input: a reference to another tensor or a scalar value.")

class OpsPowerRequest(OpsBaseRequest):
    base_tensor: TensorRef = Field(..., description="Reference to the base tensor.")
    exponent: TensorInputVal = Field(..., description="Exponent: a reference to another tensor or a scalar value.")

# Tensor List Operation Request Models
class OpsTensorListRequestParams(BaseModel):
    dim: int = Field(0, description="Dimension along which to concatenate or stack.")

class OpsTensorListRequest(OpsBaseRequest): # For concatenate, stack
    input_tensors: List[TensorRef] = Field(..., min_items=1, description="List of input tensors to concatenate or stack.")
    params: OpsTensorListRequestParams

# Einsum Operation Request Model
class OpsEinsumRequestParams(BaseModel):
    equation: str = Field(..., description="Einstein summation equation string (e.g., 'ij,jk->ik').")

class OpsEinsumRequest(OpsBaseRequest):
    input_tensors: List[TensorRef] = Field(..., min_items=1, description="List of input tensors for Einsum.")
    params: OpsEinsumRequestParams

# Generic Operation Result Response Model
class OpsResultResponse(BaseModel):
    success: bool = Field(..., description="Indicates if the tensor operation was successful.")
    message: str = Field(..., description="A descriptive status message regarding the operation.")
    output_dataset_name: Optional[str] = Field(None, description="Name of the dataset where the output tensor was stored.")
    output_record_id: Optional[str] = Field(None, description="Record ID of the output tensor in the output dataset.")
    output_tensor_details: Optional[TensorOutput] = Field(None, description="Details of the output tensor, if generated and requested.")


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


# Exception handlers
@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=404,
        content={"detail": "The requested resource was not found"},
    )

@app.exception_handler(500)
async def server_error_handler(request: Request, exc: Exception):
    logger.error(f"Internal server error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# --- Dependency Functions ---
async def get_tensor_storage() -> TensorStorage:
    """Get the global TensorStorage instance with error handling."""
    try:
        if not hasattr(get_tensor_storage, '_instance'):
            get_tensor_storage._instance = tensor_storage_instance
        return get_tensor_storage._instance
    except Exception as e:
        logger.error(f"Failed to get TensorStorage instance: {e}", exc_info=True)
        raise APIError(status_code=500, detail="Failed to initialize storage")

async def get_nql_agent() -> NQLAgent:
    """Get the global NQLAgent instance with error handling."""
    try:
        if not hasattr(get_nql_agent, '_instance'):
            get_nql_agent._instance = nql_agent_instance
        return get_nql_agent._instance
    except Exception as e:
        logger.error(f"Failed to get NQLAgent instance: {e}", exc_info=True)
        raise APIError(status_code=500, detail="Failed to initialize query agent")


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

    except DatasetNotFoundError as e:
        logger.warning(f"Dataset not found during ingestion into '{name}': {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        logger.error(f"Validation error during ingestion into '{name}': {e}")
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

    except DatasetNotFoundError as e:
        logger.warning(f"Attempted to fetch non-existent dataset '{name}': {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        logger.error(f"Validation error fetching dataset '{name}': {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
         logger.exception(f"Unexpected error fetching dataset '{name}': {e}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while fetching dataset.")


@app.get("/datasets/{name}/records", response_model=ApiResponse, tags=["Data Retrieval"])
async def fetch_dataset_records(
    name: str = Path(..., description="The name of the dataset."),
    offset: int = Query(0, ge=0, description="Starting offset of records."),
    limit: int = Query(100, ge=1, description="Maximum number of records to return."),
    storage: TensorStorage = Depends(get_tensor_storage)
):
    """Retrieve records from a dataset using pagination."""
    try:
        records = storage.get_records_paginated(name, offset=offset, limit=limit)
        output_records = []
        for i, record in enumerate(records):
            shape, dtype, data_list = tensor_to_list(record["tensor"])
            record_id = record["metadata"].get("record_id", f"missing_id_{offset+i}")
            output_records.append(
                TensorOutput(record_id=record_id, shape=shape, dtype=dtype, data=data_list, metadata=record["metadata"])
            )
        return ApiResponse(success=True, message="Records retrieved successfully.", data=output_records)
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error fetching records for dataset '{name}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error retrieving records.")

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


@app.get("/datasets/{name}/count", response_model=ApiResponse, tags=["Datasets"])
async def count_dataset(
    name: str = Path(..., description="The name of the dataset."),
    storage: TensorStorage = Depends(get_tensor_storage),
):
    """Return the number of records in the specified dataset."""
    try:
        count = storage.count(name)
        return ApiResponse(success=True, message="Dataset count retrieved successfully.", data={"count": count})
    except DatasetNotFoundError as e:
        logger.warning(f"Attempted to count non-existent dataset '{name}': {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error counting dataset '{name}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error counting dataset.")

@app.get("/datasets/{dataset_name}/tensors/{record_id}", response_model=TensorOutput, tags=["Tensor Management"])
async def get_tensor_by_id_api(
    dataset_name: str = Path(..., description="The name of the dataset."),
    record_id: str = Path(..., description="The unique ID of the tensor record."),
    storage: TensorStorage = Depends(get_tensor_storage)
):
    """Retrieve a tensor by ``record_id`` from ``dataset_name``.

    Returns the tensor along with its metadata. Raises ``HTTPException`` with
    status 404 if either the dataset or the record is not found.
    """

    try:
        record = storage.get_tensor_by_id(dataset_name, record_id)
    except DatasetNotFoundError as e:
        logger.warning(
            f"Dataset not found while fetching tensor '{record_id}' from '{dataset_name}': {e}"
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except TensorNotFoundError as e:
        logger.warning(
            f"Tensor '{record_id}' not found in dataset '{dataset_name}': {e}"
        )
        # TensorStorage now raises TensorNotFoundError instead of returning None
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception(
            f"Error fetching tensor '{record_id}' from dataset '{dataset_name}': {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error fetching tensor."
        )

    try:
        shape, dtype, data_list = tensor_to_list(record["tensor"])
    except ValueError as e:
        logger.error(
            f"Validation error retrieving tensor '{record_id}' from '{dataset_name}': {e}"
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return TensorOutput(
        record_id=record_id,
        shape=shape,
        dtype=dtype,
        data=data_list,
        metadata=record["metadata"],
    )

@app.delete("/datasets/{dataset_name}", response_model=ApiResponse, tags=["Datasets"])
async def delete_dataset_api(
    dataset_name: str = Path(..., description="The name of the dataset to delete."),
    storage: TensorStorage = Depends(get_tensor_storage)
):
    """
    Deletes an entire dataset and all its associated tensor records.
    """
    try:
        storage.delete_dataset(dataset_name)
        logger.info(f"Dataset '{dataset_name}' deleted successfully.")
        return ApiResponse(success=True, message=f"Dataset '{dataset_name}' deleted successfully.")
    except DatasetNotFoundError as e:
        logger.warning(f"Attempted to delete non-existent dataset '{dataset_name}': {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        logger.error(f"Validation error deleting dataset '{dataset_name}': {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Error deleting dataset '{dataset_name}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error deleting dataset.")

@app.delete("/datasets/{dataset_name}/tensors/{record_id}", response_model=ApiResponse, tags=["Tensor Management"])
async def delete_tensor_api(
    dataset_name: str = Path(..., description="The name of the dataset."),
    record_id: str = Path(..., description="The unique ID of the tensor record to delete."),
    storage: TensorStorage = Depends(get_tensor_storage)
):
    """
    Deletes a specific tensor record by its ID from the specified dataset.
    """
    try:
        storage.delete_tensor(dataset_name, record_id)
        logger.info(f"Tensor record '{record_id}' deleted successfully from dataset '{dataset_name}'.")
        return ApiResponse(success=True, message=f"Tensor record '{record_id}' deleted successfully.")
    except DatasetNotFoundError as e:
        logger.warning(f"Error deleting tensor from unknown dataset '{dataset_name}': {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except TensorNotFoundError as e:
        logger.warning(f"Tensor record '{record_id}' not found in '{dataset_name}': {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        logger.error(f"Validation error deleting tensor '{record_id}' from '{dataset_name}': {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Error deleting tensor '{record_id}' from dataset '{dataset_name}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error deleting tensor.")

@app.put("/datasets/{dataset_name}/tensors/{record_id}/metadata", response_model=ApiResponse, tags=["Tensor Management"])
async def update_tensor_metadata_api(
    dataset_name: str = Path(..., description="The name of the dataset."),
    record_id: str = Path(..., description="The unique ID of the tensor record to update."),
    update_request: UpdateMetadataRequest = Body(...),
    storage: TensorStorage = Depends(get_tensor_storage)
):
    """
    Updates the metadata for a specific tensor record.
    This replaces the entire existing metadata with the new metadata provided.
    """
    try:
        storage.update_tensor_metadata(dataset_name, record_id, update_request.new_metadata)
        logger.info(f"Metadata for tensor '{record_id}' in dataset '{dataset_name}' updated successfully.")
        return ApiResponse(success=True, message="Tensor metadata updated successfully.")
    except DatasetNotFoundError as e:
        logger.warning(f"Dataset not found while updating metadata for '{record_id}' in '{dataset_name}': {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except TensorNotFoundError as e:
        logger.warning(f"Tensor '{record_id}' not found in '{dataset_name}' during metadata update: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        logger.error(f"Validation error updating metadata for tensor '{record_id}' in '{dataset_name}': {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request: {e}")
    except Exception as e:
        logger.exception(f"Error updating metadata for tensor '{record_id}' in dataset '{dataset_name}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error updating tensor metadata.")

@app.get(
    "/explorer/dataset/{dataset}/tensor/{tensor_id}/metadata",
    tags=["Explorer"],
)
async def explorer_get_tensor_metadata(
    dataset: str = Path(..., description="The name of the dataset."),
    tensor_id: str = Path(..., description="The ID of the tensor record."),
    storage: TensorStorage = Depends(get_tensor_storage),
):
    """Return metadata for a specific tensor in a dataset."""
    try:
        record = storage.get_tensor_by_id(dataset, tensor_id)
        return {"dataset": dataset, "tensor_id": tensor_id, "metadata": record["metadata"]}
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except TensorNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception(
            f"Error fetching metadata for tensor '{tensor_id}' in dataset '{dataset}': {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error retrieving tensor metadata.",
        )

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
        for agent_id, static_details in agent_registry.items():
            if not isinstance(static_details, dict) or not all(k in static_details for k in ["name", "description", "config"]):
                logger.warning(f"Agent '{agent_id}' in registry is missing required static keys. Skipping.")
                continue

            current_status = "unknown"
            if agent_id == "ingestion":
                # For ingestion agent, always try to get live status, but don't create if not existing
                ingestion_instance = live_agents.get("ingestion")
                if ingestion_instance:
                    current_status = ingestion_instance.get_status()
                else:
                    # If not in live_agents, it's effectively stopped from API's perspective
                    current_status = "stopped"
            elif "status" in static_details: # For placeholder agents
                current_status = static_details["status"]
            else: # Should not happen if registry is well-defined
                logger.warning(f"Agent '{agent_id}' has no status info in registry. Defaulting to 'unknown'.")


            try:
                agents_list.append(AgentInfo(
                    id=agent_id,
                    name=static_details["name"],
                    description=static_details["description"],
                    status=current_status, # Use dynamically determined status
                    config=static_details["config"]
                ))
            except Exception as pydantic_err:
                logger.error(f"Error creating AgentInfo for agent '{agent_id}': {pydantic_err}. Details: {static_details}", exc_info=True)
                continue

        logger.info(f"Retrieved list of {len(agents_list)} agents.")
        return agents_list
    except Exception as e:
        logger.exception(f"Unexpected error listing agents: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error listing agents.")


@app.get("/agents/{agent_id}/status", response_model=AgentStatus, tags=["Agents"])
async def get_agent_status_api(agent_id: str = Path(..., description="The unique identifier of the agent.")):
    """
    Gets the current status, configuration, and last log timestamp for a specific agent.
    For the 'ingestion' agent, status is live. For others, it's from the placeholder registry.

    - **agent_id**: Path parameter for the agent's unique ID.
    \f
    Returns:
    - AgentStatus object containing the agent's details.

    Raises HTTPException:
    - 404 Not Found: If the agent_id does not exist.
    - 500 Internal Server Error: If the agent's entry is malformed or for other errors.
    """
    logger.debug(f"Request received for status of agent '{agent_id}'.")
    if agent_id not in agent_registry:
        logger.warning(f"Status requested for unknown agent '{agent_id}'.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")

    static_details = agent_registry[agent_id]
    if not isinstance(static_details, dict) or not all(k in static_details for k in ["name", "description", "config"]):
        logger.error(f"Agent '{agent_id}' registry entry is malformed: {static_details}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error: Malformed static data for agent '{agent_id}'.")

    current_status: str
    last_log_ts: Optional[float] = None # Placeholder for last log timestamp

    if agent_id == "ingestion":
        agent_instance = _get_or_create_ingestion_agent() # Ensures instance exists for status check
        current_status = agent_instance.get_status()
        # For last_log_timestamp, one could fetch logs and get the last one, or agent could track it.
        # Here, we'll fake it for now or leave it None.
        # To get it from actual logs (could be slightly expensive if logs are long):
        # recent_logs = agent_instance.get_logs(max_lines=1)
        # if recent_logs: last_log_ts = time.time() # Approximation
        last_log_ts = time.time() # Simulate update on access for live agent
    else: # Placeholder agents
        current_status = static_details.get("status", "unknown")
        last_log_ts = static_details.get("last_log_timestamp")

    try:
        status_response = AgentStatus(
            id=agent_id,
            name=static_details["name"],
            description=static_details["description"],
            status=current_status,
            config=static_details["config"],
            last_log_timestamp=last_log_ts
        )
        logger.info(f"Returning status for agent '{agent_id}': {status_response.status}")
        return status_response
    except Exception as pydantic_err:
        logger.error(f"Error creating AgentStatus response for agent '{agent_id}': {pydantic_err}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error creating status response for agent '{agent_id}'.")


@app.post("/agents/{agent_id}/start", response_model=ApiResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Agents"])
async def start_agent_api(agent_id: str = Path(..., description="The unique identifier of the agent to start.")):
    """
    Signals an agent to start its operation.
    For 'ingestion' agent, it's a live command. For others, it's simulated.

    - **agent_id**: Path parameter for the agent's unique ID.
    \f
    Returns:
    - ApiResponse indicating success or failure.

    Raises HTTPException:
    - 404 Not Found: If the agent_id does not exist.
    """
    logger.info(f"Received start signal for agent '{agent_id}'.")
    if agent_id not in agent_registry:
        logger.warning(f"Start signal received for unknown agent '{agent_id}'.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")

    if agent_id == "ingestion":
        agent_instance = _get_or_create_ingestion_agent()
        current_status = agent_instance.get_status()
        if current_status == "running":
            logger.info("Ingestion agent is already running.")
            return ApiResponse(success=False, message="Ingestion agent is already running.")
        if current_status == "error" and agent_instance._monitor_thread and agent_instance._monitor_thread.is_alive():
             logger.warning("Ingestion agent is in error state but thread is alive. Attempting to stop first.")
             agent_instance.stop() # Try to clean up before restart
             await asyncio.sleep(1) # Give it a moment to stop

        logger.info("Attempting to start live DataIngestionAgent...")
        agent_instance.start()
        # Brief pause to allow status to potentially update if start is synchronous
        await asyncio.sleep(0.1)
        new_status = agent_instance.get_status()
        if new_status == "running":
            logger.info("DataIngestionAgent started successfully.")
            return ApiResponse(success=True, message="Data Ingestion Agent started successfully.")
        else:
            logger.error(f"DataIngestionAgent failed to start. Current status: {new_status}")
            return ApiResponse(success=False, message=f"Data Ingestion Agent failed to start. Status: {new_status}")

    else: # Placeholder agents
        current_status = agent_registry[agent_id].get("status", "unknown")
        if current_status in ["running", "starting"]:
            logger.info(f"Placeholder agent '{agent_id}' is already {current_status}. No action taken.")
            return ApiResponse(success=False, message=f"Agent '{agent_id}' is already {current_status}.")
        if current_status == "error":
            logger.warning(f"Attempting to start placeholder agent '{agent_id}' from 'error' state.")

        agent_registry[agent_id]["status"] = "starting"
        await asyncio.sleep(0.1) # Simulate transition
        agent_registry[agent_id]["status"] = "running"
        agent_registry[agent_id]["last_log_timestamp"] = time.time()
        logger.info(f"Placeholder agent '{agent_id}' status set to 'running' (simulated).")
        return ApiResponse(success=True, message=f"Start signal sent to placeholder agent '{agent_id}'. Status is now 'running' (simulated).")

@app.post("/agents/{agent_id}/stop", response_model=ApiResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Agents"])
async def stop_agent_api(agent_id: str = Path(..., description="The unique identifier of the agent to stop.")):
    """
    Signals an agent to stop its operation gracefully.
    For 'ingestion' agent, it's a live command. For others, it's simulated.

    - **agent_id**: Path parameter for the agent's unique ID.
    \f
    Returns:
    - ApiResponse indicating success or failure.

    Raises HTTPException:
    - 404 Not Found: If the agent_id does not exist.
    """
    logger.info(f"Received stop signal for agent '{agent_id}'.")
    if agent_id not in agent_registry:
        logger.warning(f"Stop signal received for unknown agent '{agent_id}'.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")

    if agent_id == "ingestion":
        # Don't use _get_or_create_ingestion_agent() here if we only want to stop an *existing* live agent.
        agent_instance = live_agents.get("ingestion")
        if not agent_instance:
            logger.info(f"Ingestion agent '{agent_id}' is not currently live (already stopped or never started).")
            return ApiResponse(success=True, message="Ingestion agent is already stopped or was never started.")

        current_status = agent_instance.get_status()
        if current_status == "stopped":
            logger.info("Ingestion agent is already stopped.")
            return ApiResponse(success=True, message="Ingestion agent is already stopped.")

        logger.info("Attempting to stop live DataIngestionAgent...")
        agent_instance.stop()
        await asyncio.sleep(0.1) # Allow for status update
        new_status = agent_instance.get_status()
        if new_status == "stopped":
            logger.info("DataIngestionAgent stopped successfully.")
            # Optionally remove from live_agents: del live_agents["ingestion"] to allow full re-creation
            return ApiResponse(success=True, message="Data Ingestion Agent stopped successfully.")
        else:
            logger.warning(f"DataIngestionAgent may not have stopped cleanly. Current status: {new_status}")
            return ApiResponse(success=False, message=f"Data Ingestion Agent stop signal sent. Status: {new_status}")

    else: # Placeholder agents
        current_status = agent_registry[agent_id].get("status", "unknown")
        if current_status in ["stopped", "stopping"]:
            logger.info(f"Placeholder agent '{agent_id}' is already {current_status}. No action taken.")
            return ApiResponse(success=False, message=f"Agent '{agent_id}' is already {current_status}.")

        agent_registry[agent_id]["status"] = "stopping"
        await asyncio.sleep(0.1) # Simulate transition
        agent_registry[agent_id]["status"] = "stopped"
        agent_registry[agent_id]["last_log_timestamp"] = time.time()
        logger.info(f"Placeholder agent '{agent_id}' status set to 'stopped' (simulated).")
        return ApiResponse(success=True, message=f"Stop signal sent to placeholder agent '{agent_id}'. Status is now 'stopped' (simulated).")


@app.get("/agents/{agent_id}/logs", response_model=AgentLogResponse, tags=["Agents"])
async def get_agent_logs_api(
    agent_id: str = Path(..., description="The unique identifier of the agent."),
    lines: int = Query(20, ge=1, le=1000, description="Maximum number of recent log lines to retrieve.")
):
    """
    Retrieves recent logs for a specific agent.
    For 'ingestion' agent, logs are from the live instance. For others, they are simulated.

    - **agent_id**: Path parameter for the agent's unique ID.
    - **lines**: Query parameter for the number of log lines.
    \f
    Returns:
    - AgentLogResponse containing a list of log strings.

    Raises HTTPException:
    - 404 Not Found: If the agent_id does not exist.
    - 500 Internal Server Error: If log generation/retrieval fails.
    """
    logger.debug(f"Request received for logs of agent '{agent_id}' (lines={lines}).")
    if agent_id not in agent_registry:
        logger.warning(f"Log request for unknown agent '{agent_id}'.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")

    if agent_id == "ingestion":
        # For logs, we want to get them even if agent is stopped, so _get_or_create is okay.
        # If it was never started, it will be created in "stopped" state, and logs will be empty.
        agent_instance = _get_or_create_ingestion_agent()
        try:
            actual_logs = agent_instance.get_logs(max_lines=lines)
            logger.info(f"Retrieved {len(actual_logs)} log lines for DataIngestionAgent.")
            return AgentLogResponse(logs=actual_logs)
        except Exception as e:
            logger.exception(f"Error retrieving logs from DataIngestionAgent '{agent_id}': {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error retrieving logs for agent '{agent_id}'.")
    else: # Placeholder agents
        # Parameter 'lines' is already validated by FastAPI/Pydantic
        try:
            simulated_logs = [_simulate_agent_log(agent_id) for _ in range(lines)]
            # For placeholder agents, update last_log_timestamp in agent_registry
            if agent_id in agent_registry and isinstance(agent_registry[agent_id], dict):
                 agent_registry[agent_id]["last_log_timestamp"] = time.time()
            logger.info(f"Generated {len(simulated_logs)} simulated log lines for placeholder agent '{agent_id}'.")
            return AgentLogResponse(logs=simulated_logs)
        except Exception as e:
            logger.exception(f"Error generating simulated logs for agent '{agent_id}': {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error generating logs for agent '{agent_id}'.")

# --- Agent Configuration Endpoints ---

class AgentConfigPayload(BaseModel):
    config: Dict[str, Any]


@app.get("/agents/{agent_id}/config", response_model=Dict[str, Any], tags=["Agents"])
async def get_agent_config_api(agent_id: str = Path(..., description="The unique identifier of the agent.")):
    """Retrieve the stored configuration for a given agent."""
    if agent_id not in agent_registry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")
    return agent_registry[agent_id].get("config", {})


@app.post("/agents/{agent_id}/configure", response_model=ApiResponse, tags=["Agents"])
async def configure_agent_api(
    agent_id: str = Path(..., description="The unique identifier of the agent."),
    payload: AgentConfigPayload = Body(...),
):
    """Update an agent's configuration."""
    if agent_id not in agent_registry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")

    new_config = payload.config
    agent_registry[agent_id]["config"] = new_config

    if agent_id == "ingestion" and "ingestion" in live_agents:
        agent_instance = live_agents["ingestion"]
        if "source_directory" in new_config:
            agent_instance.source_directory = new_config["source_directory"]
        if "dataset_name" in new_config:
            agent_instance.dataset_name = new_config["dataset_name"]
        if "polling_interval_sec" in new_config:
            agent_instance.polling_interval = new_config["polling_interval_sec"]

    return ApiResponse(success=True, message=f"Configuration updated for agent '{agent_id}'.")

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

    # --- Actual Metrics ---

    # Agent Status Summary (from registry)
    status_counts = {"running": 0, "stopped": 0, "error": 0, "starting": 0, "stopping": 0, "unknown": 0}
    for agent_id, details in agent_registry.items():
        status = details.get("status", "unknown")
        if status not in status_counts:
            logger.warning(f"Agent '{agent_id}' has unexpected status '{status}'. Counting as 'unknown'.")
            status = "unknown"
        status_counts[status] += 1
    metrics_data["agent_status_summary"] = status_counts

    # Total records across all datasets
    total_records = 0
    for ds in storage.list_datasets() if hasattr(storage, "list_datasets") else []:
        try:
            total_records += storage.count(ds)
        except Exception as e:
            logger.warning(f"Failed counting records for dataset '{ds}': {e}")
    metrics_data["total_records_est"] = total_records

    # Performance metrics from agents/storage
    ingestion_dataset = agent_registry.get("ingestion", {}).get("config", {}).get("dataset_name")
    ingestion_rate = 0.0
    if ingestion_dataset and getattr(storage, "dataset_exists", lambda d: False)(ingestion_dataset):
        try:
            records = storage.get_dataset_with_metadata(ingestion_dataset)
            recent = [r for r in records if r.get("metadata", {}).get("timestamp_utc", 0) >= current_time - 60]
            ingestion_rate = len(recent) / 60.0
        except Exception as e:
            logger.warning(f"Failed calculating ingestion rate: {e}")
    metrics_data["data_ingestion_rate"] = ingestion_rate

    metrics_data["avg_query_latency_ms"] = 0.0  # No instrumentation yet

    rl_cfg = agent_registry.get("rl_trainer", {}).get("config", {})
    rl_ds = rl_cfg.get("experience_dataset")
    rl_total_steps = 0
    rl_latest_reward = None
    if rl_ds and getattr(storage, "dataset_exists", lambda d: False)(rl_ds):
        try:
            rl_total_steps = storage.count(rl_ds)
            if rl_total_steps > 0:
                last_meta = storage.get_dataset_with_metadata(rl_ds)[-1]["metadata"]
                rl_latest_reward = last_meta.get("reward")
        except Exception as e:
            logger.warning(f"Failed retrieving RL metrics: {e}")
    metrics_data["rl_total_steps"] = rl_total_steps
    metrics_data["rl_latest_reward"] = rl_latest_reward

    auto_cfg = agent_registry.get("automl_search", {}).get("config", {})
    auto_ds = auto_cfg.get("results_dataset")
    auto_trials = 0
    auto_best = None
    if auto_ds and getattr(storage, "dataset_exists", lambda d: False)(auto_ds):
        try:
            auto_trials = storage.count(auto_ds)
            metas = [m["metadata"].get("score") for m in storage.get_dataset_with_metadata(auto_ds) if m["metadata"].get("score") is not None]
            if metas:
                if auto_cfg.get("task_type") == "classification":
                    auto_best = max(metas)
                else:
                    auto_best = min(metas)
        except Exception as e:
            logger.warning(f"Failed retrieving AutoML metrics: {e}")
    metrics_data["automl_trials_completed"] = auto_trials
    metrics_data["automl_best_score"] = auto_best

    # System health metrics
    try:
        load1, _, _ = os.getloadavg()
        cpu_usage = min(100.0, (load1 / float(os.cpu_count() or 1)) * 100.0)
    except Exception as e:
        logger.warning(f"Failed to read CPU load: {e}")
        cpu_usage = 0.0

    try:
        mem_total = mem_available = None
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = float(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_available = float(line.split()[1])
                if mem_total is not None and mem_available is not None:
                    break
        if mem_total and mem_available is not None:
            mem_used = mem_total - mem_available
            mem_percent = (mem_used / mem_total) * 100.0
        else:
            mem_percent = 0.0
    except Exception as e:
        logger.warning(f"Failed to read memory usage: {e}")
        mem_percent = 0.0

    metrics_data["system_cpu_usage_percent"] = cpu_usage
    metrics_data["system_memory_usage_percent"] = mem_percent

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
# --- Tensor Operations Router ---
ops_router = APIRouter(
    prefix="/ops",
    tags=["Tensor Operations"],
    # dependencies=[Depends(get_tensor_storage)], # Optional: Add dependencies for the whole router
)

DEFAULT_OPS_OUTPUT_DATASET = "tensor_ops_results"

async def _get_tensor_from_ref(tensor_ref: TensorRef, storage: TensorStorage) -> torch.Tensor:
    """Helper to fetch a tensor from storage using a TensorRef."""
    try:
        record = storage.get_tensor_by_id(tensor_ref.dataset_name, tensor_ref.record_id)
        if not isinstance(record.get('tensor'), torch.Tensor):
            logger.error(f"Retrieved object for '{tensor_ref.record_id}' in '{tensor_ref.dataset_name}' is not a tensor. Type: {type(record.get('tensor'))}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Retrieved object for input tensor '{tensor_ref.record_id}' is not a valid tensor.")
        return record['tensor']
    except DatasetNotFoundError as e:
        logger.warning(
            f"Dataset not found while fetching tensor '{tensor_ref.record_id}' from '{tensor_ref.dataset_name}': {e}"
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except TensorNotFoundError as e:
        logger.warning(
            f"Tensor '{tensor_ref.record_id}' not found in '{tensor_ref.dataset_name}': {e}"
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        logger.error(
            f"Validation error retrieving tensor '{tensor_ref.record_id}' from '{tensor_ref.dataset_name}': {e}"
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error fetching tensor '{tensor_ref.record_id}' from dataset '{tensor_ref.dataset_name}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Unexpected error fetching input tensor '{tensor_ref.record_id}'.")

async def _store_and_respond_ops(
    result_tensor: torch.Tensor,
    op_name: str,
    request: OpsBaseRequest, # OpsLogRequest, OpsReshapeRequest etc. all inherit from OpsBaseRequest
    storage: TensorStorage,
    input_refs: List[TensorRef] # For logging/metadata
) -> OpsResultResponse:
    """Helper to store operation result and craft the API response."""
    output_dataset_name = request.output_dataset_name or DEFAULT_OPS_OUTPUT_DATASET
    # Ensure the default output dataset exists (TensorStorage.insert should handle this if configured to do so,
    # or we can explicitly create it here if needed)
    try:
        if not storage.dataset_exists(output_dataset_name):
            storage.create_dataset(output_dataset_name)
            logger.info(f"Created default output dataset for tensor operations: '{output_dataset_name}'")
    except Exception as e:
        logger.error(f"Failed to ensure output dataset '{output_dataset_name}' exists: {e}")
        # Decide if this should be a critical failure or if TensorStorage.insert will handle it
        # For now, let's assume insert will fail if dataset is truly needed and not creatable.

    default_metadata = {
        "operation": op_name,
        "timestamp": time.time(),
        "source_tensors": [ref.model_dump() for ref in input_refs],
        "source_api_request": request.model_dump(exclude_none=True)  # Log the request for traceability
    }
    final_metadata = {**default_metadata, **(request.output_metadata or {})}

    try:
        record_id = storage.insert(output_dataset_name, result_tensor, final_metadata)
        logger.info(f"Stored result of '{op_name}' operation in '{output_dataset_name}' with record_id: {record_id}")

        s, d, dl = tensor_to_list(result_tensor)
        tensor_out_details = TensorOutput(
            record_id=record_id,
            shape=s,
            dtype=d,
            data=dl,
            metadata=final_metadata
        )
        return OpsResultResponse(
            success=True,
            message=f"'{op_name}' operation successful. Result stored in '{output_dataset_name}/{record_id}'.",
            output_dataset_name=output_dataset_name,
            output_record_id=record_id,
            output_tensor_details=tensor_out_details
        )
    except DatasetNotFoundError as e:
        logger.warning(f"Output dataset '{output_dataset_name}' not found when storing result of '{op_name}': {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:  # Other validation errors
        logger.error(f"Failed to store result for '{op_name}' in '{output_dataset_name}': {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error storing result: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error storing result of '{op_name}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error storing operation result.")


@ops_router.post("/log", response_model=OpsResultResponse)
async def tensor_log(request: OpsLogRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """Computes the element-wise natural logarithm of the input tensor."""
    input_tensor = await _get_tensor_from_ref(request.input_tensor, storage)
    try:
        result_tensor = TensorOps.log(input_tensor)
    except (ValueError, TypeError) as e:
        logger.error(f"TensorOps.log failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "log", request, storage, [request.input_tensor])

@ops_router.post("/reshape", response_model=OpsResultResponse)
async def tensor_reshape(request: OpsReshapeRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """Reshapes the input tensor to the specified new shape."""
    input_tensor = await _get_tensor_from_ref(request.input_tensor, storage)
    try:
        result_tensor = TensorOps.reshape(input_tensor, request.params.new_shape)
    except (ValueError, TypeError, RuntimeError) as e: # RuntimeError for invalid reshape dims
        logger.error(f"TensorOps.reshape failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "reshape", request, storage, [request.input_tensor])

@ops_router.post("/transpose", response_model=OpsResultResponse)
async def tensor_transpose(request: OpsTransposeRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """Transposes the input tensor along the specified dimensions."""
    input_tensor = await _get_tensor_from_ref(request.input_tensor, storage)
    try:
        result_tensor = TensorOps.transpose(input_tensor, request.params.dim0, request.params.dim1)
    except (ValueError, TypeError, IndexError) as e: # IndexError for invalid dims
        logger.error(f"TensorOps.transpose failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "transpose", request, storage, [request.input_tensor])

@ops_router.post("/permute", response_model=OpsResultResponse)
async def tensor_permute(request: OpsPermuteRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """Permutes the dimensions of the input tensor according to the specified order."""
    input_tensor = await _get_tensor_from_ref(request.input_tensor, storage)
    try:
        result_tensor = TensorOps.permute(input_tensor, tuple(request.params.dims))
    except (ValueError, TypeError, RuntimeError) as e: # RuntimeError for invalid permute dims
        logger.error(f"TensorOps.permute failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "permute", request, storage, [request.input_tensor])

@ops_router.post("/sum", response_model=OpsResultResponse)
async def tensor_sum(request: OpsReductionRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """Computes the sum of elements in the input tensor, optionally along specified dimensions."""
    input_tensor = await _get_tensor_from_ref(request.input_tensor, storage)
    try:
        # Params.dim can be int, List[int], or None. TensorOps.sum handles this.
        result_tensor = TensorOps.sum(input_tensor, dim=request.params.dim, keepdim=request.params.keepdim)
    except (ValueError, TypeError, RuntimeError) as e: # RuntimeError for dim issues
        logger.error(f"TensorOps.sum failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "sum", request, storage, [request.input_tensor])

@ops_router.post("/mean", response_model=OpsResultResponse)
async def tensor_mean(request: OpsReductionRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """Computes the mean of elements in the input tensor, optionally along specified dimensions."""
    input_tensor = await _get_tensor_from_ref(request.input_tensor, storage)
    try:
        # Params.dim can be int, List[int], or None. TensorOps.mean handles this.
        result_tensor = TensorOps.mean(input_tensor, dim=request.params.dim, keepdim=request.params.keepdim)
    except (ValueError, TypeError, RuntimeError) as e: # RuntimeError for dim issues
        logger.error(f"TensorOps.mean failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "mean", request, storage, [request.input_tensor])

@ops_router.post("/min", response_model=OpsResultResponse)
async def tensor_min(request: OpsMinMaxRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """
    Finds the minimum value(s) in the input tensor.
    If 'dim' is specified, returns the minimum values along that dimension.
    Note: When 'dim' is specified, this endpoint stores only the 'values' tensor, not 'indices'.
    """
    input_tensor = await _get_tensor_from_ref(request.input_tensor, storage)
    message_suffix = ""
    try:
        if request.params and request.params.dim is not None:
            result_tuple = TensorOps.min(input_tensor, dim=request.params.dim, keepdim=request.params.keepdim)
            result_tensor = result_tuple.values # Store only values
            message_suffix = " (values tensor stored)"
        else:
            result_tensor = TensorOps.min(input_tensor) # No dim, result is a single tensor
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"TensorOps.min failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    
    response = await _store_and_respond_ops(result_tensor, "min", request, storage, [request.input_tensor])
    if message_suffix: # Append info about storing only values
        response.message += message_suffix
    return response

@ops_router.post("/max", response_model=OpsResultResponse)
async def tensor_max(request: OpsMinMaxRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """
    Finds the maximum value(s) in the input tensor.
    If 'dim' is specified, returns the maximum values along that dimension.
    Note: When 'dim' is specified, this endpoint stores only the 'values' tensor, not 'indices'.
    """
    input_tensor = await _get_tensor_from_ref(request.input_tensor, storage)
    message_suffix = ""
    try:
        if request.params and request.params.dim is not None:
            result_tuple = TensorOps.max(input_tensor, dim=request.params.dim, keepdim=request.params.keepdim)
            result_tensor = result_tuple.values # Store only values
            message_suffix = " (values tensor stored)"
        else:
            result_tensor = TensorOps.max(input_tensor) # No dim, result is a single tensor
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"TensorOps.max failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    response = await _store_and_respond_ops(result_tensor, "max", request, storage, [request.input_tensor])
    if message_suffix: # Append info about storing only values
        response.message += message_suffix
    return response

async def _get_input_val(input_val: TensorInputVal, storage: TensorStorage) -> Union[torch.Tensor, float, int]:
    """Helper to resolve a TensorInputVal to either a torch.Tensor or a scalar."""
    if input_val.tensor_ref:
        return await _get_tensor_from_ref(input_val.tensor_ref, storage)
    elif input_val.scalar_value is not None: # Pydantic validator ensures one is present
        return input_val.scalar_value
    else:
        # This case should ideally be prevented by Pydantic's root_validator on TensorInputVal
        logger.error("Invalid TensorInputVal: neither tensor_ref nor scalar_value is present.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Internal error: Invalid input value structure.")

@ops_router.post("/add", response_model=OpsResultResponse)
async def tensor_add(request: OpsBinaryOpRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """Adds input2 (tensor or scalar) to input1 (tensor) element-wise."""
    input1_tensor = await _get_tensor_from_ref(request.input1, storage)
    input2_val = await _get_input_val(request.input2, storage)
    source_refs = [request.input1]
    if request.input2.tensor_ref:
        source_refs.append(request.input2.tensor_ref)
    try:
        result_tensor = TensorOps.add(input1_tensor, input2_val)
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"TensorOps.add failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "add", request, storage, source_refs)

@ops_router.post("/subtract", response_model=OpsResultResponse)
async def tensor_subtract(request: OpsBinaryOpRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """Subtracts input2 (tensor or scalar) from input1 (tensor) element-wise."""
    input1_tensor = await _get_tensor_from_ref(request.input1, storage)
    input2_val = await _get_input_val(request.input2, storage)
    source_refs = [request.input1]
    if request.input2.tensor_ref:
        source_refs.append(request.input2.tensor_ref)
    try:
        result_tensor = TensorOps.subtract(input1_tensor, input2_val)
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"TensorOps.subtract failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "subtract", request, storage, source_refs)

@ops_router.post("/multiply", response_model=OpsResultResponse)
async def tensor_multiply(request: OpsBinaryOpRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """Multiplies input1 (tensor) by input2 (tensor or scalar) element-wise."""
    input1_tensor = await _get_tensor_from_ref(request.input1, storage)
    input2_val = await _get_input_val(request.input2, storage)
    source_refs = [request.input1]
    if request.input2.tensor_ref:
        source_refs.append(request.input2.tensor_ref)
    try:
        result_tensor = TensorOps.multiply(input1_tensor, input2_val)
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"TensorOps.multiply failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "multiply", request, storage, source_refs)

@ops_router.post("/divide", response_model=OpsResultResponse)
async def tensor_divide(request: OpsBinaryOpRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """Divides input1 (tensor) by input2 (tensor or scalar) element-wise."""
    input1_tensor = await _get_tensor_from_ref(request.input1, storage)
    input2_val = await _get_input_val(request.input2, storage)
    source_refs = [request.input1]
    if request.input2.tensor_ref:
        source_refs.append(request.input2.tensor_ref)
    try:
        result_tensor = TensorOps.divide(input1_tensor, input2_val)
    except (ValueError, TypeError, RuntimeError) as e: # Catches division by zero if it's a RuntimeError or ValueError
        logger.error(f"TensorOps.divide failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "divide", request, storage, source_refs)

@ops_router.post("/power", response_model=OpsResultResponse)
async def tensor_power(request: OpsPowerRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """Raises the base_tensor to the power of the exponent (tensor or scalar)."""
    base_tensor_val = await _get_tensor_from_ref(request.base_tensor, storage)
    exponent_val = await _get_input_val(request.exponent, storage)
    source_refs = [request.base_tensor]
    if request.exponent.tensor_ref:
        source_refs.append(request.exponent.tensor_ref)
    try:
        result_tensor = TensorOps.power(base_tensor_val, exponent_val)
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"TensorOps.power failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "power", request, storage, source_refs)

@ops_router.post("/matmul", response_model=OpsResultResponse)
async def tensor_matmul(request: OpsBinaryOpRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """Performs matrix multiplication between input1 and input2 (both must be tensors)."""
    input1_tensor = await _get_tensor_from_ref(request.input1, storage)
    input2_val = await _get_input_val(request.input2, storage)

    if not isinstance(input2_val, torch.Tensor):
        logger.error("TensorOps.matmul failed: input2 must be a tensor, not a scalar.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input2 for matmul must be a tensor, not a scalar.")
    
    source_refs = [request.input1, request.input2.tensor_ref] # input2.tensor_ref must exist due to above check
    
    try:
        result_tensor = TensorOps.matmul(input1_tensor, input2_val)
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"TensorOps.matmul failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "matmul", request, storage, source_refs)

@ops_router.post("/dot", response_model=OpsResultResponse)
async def tensor_dot(request: OpsBinaryOpRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """Computes the dot product of input1 and input2 (both must be 1D tensors)."""
    input1_tensor = await _get_tensor_from_ref(request.input1, storage)
    input2_val = await _get_input_val(request.input2, storage)

    if not isinstance(input2_val, torch.Tensor):
        logger.error("TensorOps.dot failed: input2 must be a tensor, not a scalar.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input2 for dot product must be a tensor, not a scalar.")
    
    # TensorOps.dot will validate if they are 1D.
    source_refs = [request.input1, request.input2.tensor_ref] # input2.tensor_ref must exist

    try:
        result_tensor = TensorOps.dot(input1_tensor, input2_val)
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"TensorOps.dot failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "dot", request, storage, source_refs)

@ops_router.post("/concatenate", response_model=OpsResultResponse)
async def tensor_concatenate(request: OpsTensorListRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """Concatenates a list of input tensors along the specified dimension."""
    input_tensors_resolved = []
    for tensor_ref in request.input_tensors:
        tensor = await _get_tensor_from_ref(tensor_ref, storage)
        input_tensors_resolved.append(tensor)
    
    if not input_tensors_resolved:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input tensors list cannot be empty for concatenate.")

    try:
        result_tensor = TensorOps.concatenate(input_tensors_resolved, dim=request.params.dim)
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"TensorOps.concatenate failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "concatenate", request, storage, request.input_tensors)

@ops_router.post("/stack", response_model=OpsResultResponse)
async def tensor_stack(request: OpsTensorListRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """Stacks a list of input tensors along a new dimension."""
    input_tensors_resolved = []
    for tensor_ref in request.input_tensors:
        tensor = await _get_tensor_from_ref(tensor_ref, storage)
        input_tensors_resolved.append(tensor)

    if not input_tensors_resolved:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input tensors list cannot be empty for stack.")

    try:
        result_tensor = TensorOps.stack(input_tensors_resolved, dim=request.params.dim)
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"TensorOps.stack failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "stack", request, storage, request.input_tensors)

@ops_router.post("/einsum", response_model=OpsResultResponse)
async def tensor_einsum(request: OpsEinsumRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """Applies Einstein summation to the input tensors based on the provided equation."""
    input_tensors_resolved = []
    for tensor_ref in request.input_tensors:
        tensor = await _get_tensor_from_ref(tensor_ref, storage)
        input_tensors_resolved.append(tensor)

    if not input_tensors_resolved:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input tensors list cannot be empty for einsum.")

    try:
        result_tensor = TensorOps.einsum(request.params.equation, *input_tensors_resolved)
    except (ValueError, TypeError, RuntimeError) as e: # Catches invalid equations or tensor mismatches
        logger.error(f"TensorOps.einsum failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return await _store_and_respond_ops(result_tensor, "einsum", request, storage, request.input_tensors)

# Include routers from tensorus.api.endpoints
app.include_router(router_tensor_descriptor)
app.include_router(router_semantic_metadata)
app.include_router(router_search_aggregate)
app.include_router(router_version_lineage)
app.include_router(router_extended_metadata)
app.include_router(router_io)
app.include_router(router_management)
app.include_router(router_analytics)

# Include the tensor operations router defined above
app.include_router(ops_router)

# --- Root Endpoint ---
@app.get("/", include_in_schema=False)
async def read_root():
    """Provides a simple welcome message for the API root."""
    # Useful for health checks or simple verification that the API is running
    return {"message": "Welcome to the Tensorus API! Visit /docs or /redoc for interactive documentation."}
