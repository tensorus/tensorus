# api.py (Modifications for Step 1)

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import random # For simulating logs/status
import time # For simulating timestamps
import math # For simulating metrics

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Body, Depends, Path, status # Added status
from pydantic import BaseModel, Field

# Import Tensorus modules
from tensor_storage import TensorStorage
from nql_agent import NQLAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Tensorus Instances ---
try:
    tensor_storage_instance = TensorStorage()
    nql_agent_instance = NQLAgent(tensor_storage_instance)
    # NOTE: Actual agent processes (Ingestion, RL, AutoML) are assumed to be
    # running independently for now. This API layer will *coordinate* with them
    # in a full implementation, but currently only manages placeholder state.
except Exception as e:
    logger.exception(f"Failed to initialize Tensorus components: {e}")
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
    },
    "rl_trainer": {
        "name": "RL Trainer",
        "description": "Trains reinforcement learning models using stored experiences.",
        "status": "stopped",
        "config": {"experience_dataset": "rl_experiences", "batch_size": 128, "target_update_freq": 500},
         "last_log_timestamp": None,
    },
    "automl_search": {
        "name": "AutoML Search",
        "description": "Performs hyperparameter optimization.",
        "status": "stopped",
        "config": {"trials": 50, "results_dataset": "automl_results", "task_type": "regression"},
        "last_log_timestamp": None,
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
    return f"{ts} [{level}] ({agent_registry[agent_id]['name']}) {msg}"


# --- Helper Functions (list_to_tensor, tensor_to_list - keep as before) ---
def _validate_tensor_data(data: List[Any], shape: List[int]):
    if not shape:
         if not isinstance(data, (int, float)): raise ValueError("Scalar tensor data must be a single number.")
         return True
    if not isinstance(data, list): raise ValueError(f"Data for shape {shape} must be a list.")
    expected_len = shape[0]
    if len(data) != expected_len: raise ValueError(f"Dimension 0 mismatch: Expected {expected_len}, got {len(data)} for shape {shape}.")
    if len(shape) > 1:
         for item in data: _validate_tensor_data(item, shape[1:])
    elif len(shape) == 1:
        if not all(isinstance(x, (int, float)) for x in data): raise ValueError("Innermost list elements must be numbers.")
    return True

def list_to_tensor(shape: List[int], dtype_str: str, data: Union[List[Any], int, float]) -> torch.Tensor:
    try:
        dtype_map = {'float32': torch.float32,'float': torch.float,'float64': torch.float64,'double': torch.double,'int32': torch.int32,'int': torch.int,'int64': torch.int64,'long': torch.long,'bool': torch.bool}
        torch_dtype = dtype_map.get(dtype_str.lower())
        if torch_dtype is None: raise ValueError(f"Unsupported dtype string: {dtype_str}")
        # _validate_tensor_data(data, shape) # Optional strict validation
        tensor = torch.tensor(data, dtype=torch_dtype)
        if list(tensor.shape) != shape:
             try: tensor = tensor.reshape(shape)
             except RuntimeError: raise ValueError(f"Created tensor shape {list(tensor.shape)} != requested {shape}")
        return tensor
    except (TypeError, ValueError) as e:
        logger.error(f"Error converting list to tensor: {e}. Shape: {shape}, Dtype: {dtype_str}")
        raise ValueError(f"Failed tensor conversion: {e}") from e
    except Exception as e:
         logger.error(f"Unexpected error during list_to_tensor: {e}", exc_info=True)
         raise ValueError(f"Unexpected tensor conversion error: {e}") from e

def tensor_to_list(tensor: torch.Tensor) -> Tuple[List[int], str, List[Any]]:
    shape = list(tensor.shape)
    dtype_str = str(tensor.dtype).split('.')[-1]
    data = tensor.tolist()
    return shape, dtype_str, data

# --- Pydantic Models (keep existing ones: DatasetCreateRequest, TensorInput, etc.) ---
class DatasetCreateRequest(BaseModel):
    name: str = Field(..., description="Unique name for the new dataset.", example="my_new_dataset")

class TensorInput(BaseModel):
    shape: List[int] = Field(..., description="Shape of the tensor.", example=[2, 3])
    dtype: str = Field(..., description="Data type (e.g., 'float32', 'int64').", example="float32")
    data: Union[List[Any], int, float] = Field(..., description="Tensor data (nested list/scalar).", example=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata.", example={"source": "api_ingest"})

class NQLQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query string.", example="find records from sensor_data where status = 'active'")

class TensorOutput(BaseModel):
    record_id: str = Field(..., description="Unique record ID.")
    shape: List[int] = Field(..., description="Shape of the tensor.")
    dtype: str = Field(..., description="Data type.")
    data: Union[List[Any], int, float] = Field(..., description="Tensor data.")
    metadata: Dict[str, Any] = Field(..., description="Metadata.")

class NQLResponse(BaseModel):
    success: bool = Field(..., description="Was query processed.")
    message: str = Field(..., description="Status message.")
    count: Optional[int] = Field(None, description="Number of results.")
    results: Optional[List[TensorOutput]] = Field(None, description="Matching records.")

class ApiResponse(BaseModel):
    success: bool = Field(..., description="Was operation successful.")
    message: str = Field(..., description="Status message.")
    data: Optional[Any] = Field(None, description="Optional data payload.")

# --- NEW Pydantic Models for Agents ---
class AgentInfo(BaseModel):
    id: str = Field(..., description="Unique identifier for the agent.")
    name: str = Field(..., description="Display name of the agent.")
    description: str = Field(..., description="Brief description of the agent's function.")
    status: str = Field(..., description="Current status (e.g., running, stopped, error).")
    config: Dict[str, Any] = Field(..., description="Current configuration parameters.")

class AgentStatus(AgentInfo):
     # Inherits fields from AgentInfo, can add more detailed status fields later
     last_log_timestamp: Optional[float] = Field(None, description="Timestamp of the last known log message.")

class AgentLogResponse(BaseModel):
     logs: List[str] = Field(..., description="List of recent log entries.")

# --- NEW Pydantic Model for Dashboard Metrics ---
class DashboardMetrics(BaseModel):
    timestamp: float = Field(..., description="Timestamp when the metrics were generated (UTC).")
    dataset_count: int = Field(..., description="Total number of datasets.")
    total_records_est: int = Field(..., description="Estimated total number of records across all datasets (Simulated).")
    agent_status_summary: Dict[str, int] = Field(..., description="Summary count of agents by status.")
    data_ingestion_rate: float = Field(..., description="Simulated data ingestion rate (records/sec).")
    avg_query_latency_ms: float = Field(..., description="Simulated average NQL query latency (ms).")
    rl_latest_reward: Optional[float] = Field(None, description="Simulated latest reward from RL trainer.")
    rl_total_steps: int = Field(..., description="Simulated total steps taken by RL trainer.")
    automl_best_score: Optional[float] = Field(None, description="Simulated best score found by AutoML.")
    automl_trials_completed: int = Field(..., description="Simulated number of AutoML trials completed.")
    system_cpu_usage_percent: float = Field(..., description="Simulated system CPU usage percentage.")
    system_memory_usage_percent: float = Field(..., description="Simulated system memory usage percentage.")

# --- FastAPI App Instance ---
app = FastAPI(
    title="Tensorus API",
    description="API for interacting with the Tensorus Agentic Tensor Database/Data Lake. Includes agent control placeholders.",
    version="0.2.0", # Incremented version
)

# --- Dependency Functions (keep as before) ---
async def get_tensor_storage() -> TensorStorage: return tensor_storage_instance
async def get_nql_agent() -> NQLAgent: return nql_agent_instance


# --- Existing API Endpoints (Datasets, Querying - keep mostly as before) ---

@app.post("/datasets/create", response_model=ApiResponse, status_code=status.HTTP_201_CREATED, tags=["Datasets"])
async def create_dataset(req: DatasetCreateRequest, storage: TensorStorage = Depends(get_tensor_storage)):
    """Creates a new, empty dataset."""
    try:
        storage.create_dataset(req.name)
        return ApiResponse(success=True, message=f"Dataset '{req.name}' created successfully.")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
         logger.exception(f"Unexpected error creating dataset '{req.name}': {e}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error.")

@app.post("/datasets/{name}/ingest", response_model=ApiResponse, status_code=status.HTTP_201_CREATED, tags=["Data Ingestion"])
async def ingest_tensor(name: str = Path(..., description="Target dataset name."), tensor_input: TensorInput = Body(...), storage: TensorStorage = Depends(get_tensor_storage)):
    """Ingests a single tensor (JSON) into the specified dataset."""
    try:
        tensor = list_to_tensor(tensor_input.shape, tensor_input.dtype, tensor_input.data)
        record_id = storage.insert(name, tensor, tensor_input.metadata)
        return ApiResponse(success=True, message="Tensor ingested.", data={"record_id": record_id})
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
         logger.exception(f"Unexpected error ingesting into dataset '{name}': {e}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error.")

@app.get("/datasets/{name}/fetch", response_model=ApiResponse, tags=["Data Retrieval"])
async def fetch_dataset(name: str = Path(..., description="Dataset name."), storage: TensorStorage = Depends(get_tensor_storage)):
    """Retrieves all records (tensor data + metadata) from a dataset."""
    try:
        records = storage.get_dataset_with_metadata(name)
        output_records = []
        for record in records:
             shape, dtype, data_list = tensor_to_list(record['tensor'])
             output_records.append(TensorOutput(record_id=record['metadata'].get('record_id', 'N/A'), shape=shape, dtype=dtype, data=data_list, metadata=record['metadata']))
        return ApiResponse(success=True, message=f"Retrieved {len(output_records)} records.", data=output_records)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
         logger.exception(f"Unexpected error fetching dataset '{name}': {e}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error.")

@app.get("/datasets", response_model=ApiResponse, tags=["Datasets"])
async def list_datasets(storage: TensorStorage = Depends(get_tensor_storage)):
    """Lists the names of all available datasets."""
    try:
        dataset_names = list(storage.datasets.keys())
        return ApiResponse(success=True, message="Retrieved dataset list.", data=dataset_names)
    except Exception as e:
         logger.exception(f"Unexpected error listing datasets: {e}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error.")

@app.post("/query", response_model=NQLResponse, tags=["Querying"])
async def execute_nql_query(request: NQLQueryRequest, nql_agent_svc: NQLAgent = Depends(get_nql_agent)):
    """Executes a Natural Query Language (NQL) query."""
    try:
        nql_result = nql_agent_svc.process_query(request.query)
        output_results = None
        if nql_result.get('success') and nql_result.get('results'):
            output_results = []
            for record in nql_result['results']:
                 shape, dtype, data_list = tensor_to_list(record['tensor'])
                 output_results.append(TensorOutput(record_id=record['metadata'].get('record_id', 'N/A'), shape=shape, dtype=dtype, data=data_list, metadata=record['metadata']))
        response = NQLResponse(success=nql_result['success'], message=nql_result['message'], count=nql_result.get('count'), results=output_results)
        if not response.success:
            # Return 400 for query parsing/execution issues reported by the agent
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=response.message)
        return response
    except HTTPException as e: raise e # Re-raise if already handled
    except Exception as e:
         logger.exception(f"Unexpected error processing NQL query '{request.query}': {e}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Query processing error.")


# --- NEW Agent Control Endpoints ---

@app.get("/agents", response_model=List[AgentInfo], tags=["Agents"])
async def list_agents():
    """Lists all registered agents and their basic information."""
    agents_list = []
    for agent_id, details in agent_registry.items():
        agents_list.append(AgentInfo(
            id=agent_id,
            name=details["name"],
            description=details["description"],
            status=details["status"],
            config=details["config"]
        ))
    return agents_list

@app.get("/agents/{agent_id}/status", response_model=AgentStatus, tags=["Agents"])
async def get_agent_status_api(agent_id: str = Path(..., description="Unique ID of the agent.")):
    """Gets the current status and configuration of a specific agent."""
    if agent_id not in agent_registry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")
    details = agent_registry[agent_id]
    # Simulate potential status updates if needed
    # if details['status'] in ['starting', 'stopping']: details['status'] = random.choice(['running', 'stopped', 'error'])
    return AgentStatus(
        id=agent_id,
        name=details["name"],
        description=details["description"],
        status=details["status"],
        config=details["config"],
        last_log_timestamp=details.get("last_log_timestamp")
    )

@app.post("/agents/{agent_id}/start", response_model=ApiResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Agents"])
async def start_agent_api(agent_id: str = Path(..., description="Unique ID of the agent.")):
    """Signals an agent to start (Placeholder)."""
    if agent_id not in agent_registry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")
    if agent_registry[agent_id]["status"] == "running":
         return ApiResponse(success=False, message=f"Agent '{agent_id}' is already running.")

    logger.info(f"API: Received start signal for agent '{agent_id}' (Placeholder Action).")
    # Simulate state change - In reality, trigger async start process
    agent_registry[agent_id]["status"] = "starting"
    # Simulate transition to running after a short delay in a real scenario
    # For now, just accept the request.
    # TODO: Implement actual agent process starting logic.
    agent_registry[agent_id]["status"] = "running" # Immediate simulation for now
    return ApiResponse(success=True, message=f"Start signal sent to agent '{agent_id}'.")

@app.post("/agents/{agent_id}/stop", response_model=ApiResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Agents"])
async def stop_agent_api(agent_id: str = Path(..., description="Unique ID of the agent.")):
    """Signals an agent to stop (Placeholder)."""
    if agent_id not in agent_registry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")
    if agent_registry[agent_id]["status"] == "stopped":
         return ApiResponse(success=False, message=f"Agent '{agent_id}' is already stopped.")

    logger.info(f"API: Received stop signal for agent '{agent_id}' (Placeholder Action).")
    # Simulate state change - In reality, trigger async stop process
    agent_registry[agent_id]["status"] = "stopping"
    # Simulate transition to stopped after a short delay in a real scenario
    # TODO: Implement actual agent process stopping logic.
    agent_registry[agent_id]["status"] = "stopped" # Immediate simulation for now
    return ApiResponse(success=True, message=f"Stop signal sent to agent '{agent_id}'.")


@app.get("/agents/{agent_id}/logs", response_model=AgentLogResponse, tags=["Agents"])
async def get_agent_logs_api(agent_id: str = Path(..., description="Unique ID of the agent."), lines: int = 20):
    """Retrieves recent logs for a specific agent (Simulated)."""
    if agent_id not in agent_registry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found.")

    # TODO: Implement actual log retrieval from agent process or logging service.
    simulated_logs = [_simulate_agent_log(agent_id) for _ in range(lines)]
    agent_registry[agent_id]["last_log_timestamp"] = time.time() # Update timestamp on access
    return AgentLogResponse(logs=simulated_logs)

@app.get("/metrics/dashboard", response_model=DashboardMetrics, tags=["Metrics & Monitoring"])
async def get_dashboard_metrics(storage: TensorStorage = Depends(get_tensor_storage)):
    """Provides aggregated dashboard metrics (partially simulated)."""
    current_time = time.time()

    # --- Real Metrics ---
    try:
        dataset_count = len(storage.datasets.keys())
    except Exception:
        dataset_count = -1 # Indicate error fetching

    # --- Simulated/Placeholder Metrics ---
    # TODO: Replace simulations with actual metric collection from agents/storage/system.

    # Agent Status Summary (from placeholder registry)
    status_counts = {"running": 0, "stopped": 0, "error": 0, "starting": 0, "stopping": 0, "unknown": 0}
    for agent_id in agent_registry:
        status = agent_registry[agent_id].get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    # Simulate Total Records (Inefficient to calculate exactly from current storage)
    total_records_est = dataset_count * random.randint(500, 5000) if dataset_count > 0 else 0

    # Simulate performance metrics (slightly dynamic based on time/status)
    ingestion_running = agent_registry["ingestion"]["status"] == "running"
    rl_running = agent_registry["rl_trainer"]["status"] == "running"
    automl_running = agent_registry["automl_search"]["status"] == "running"

    data_ingestion_rate = random.uniform(5.0, 50.0) * (1.0 if ingestion_running else 0.1) # Higher if running
    avg_query_latency_ms = random.uniform(50.0, 300.0) * (1 + math.sin(current_time / 60)) # Simple oscillation

    rl_latest_reward = random.gauss(0, 5.0) if rl_running else None
    rl_total_steps = int(max(0, agent_registry.get("rl_trainer",{}).get("sim_steps", 0) + (random.randint(0, 100) if rl_running else 0)))
    agent_registry.setdefault("rl_trainer", {})["sim_steps"] = rl_total_steps # Store simulated steps back

    automl_trials_completed = int(max(0, agent_registry.get("automl_search",{}).get("sim_trials", 0) + (random.randint(0, 2) if automl_running else 0)))
    agent_registry.setdefault("automl_search", {})["sim_trials"] = automl_trials_completed
    # Simulate best score improving slowly if running
    current_best = agent_registry.get("automl_search",{}).get("sim_best_score", None)
    automl_best_score = None
    if automl_running:
        if current_best is None:
             automl_best_score = random.uniform(0.5, 10.0) # Initial score (assuming lower is better)
        else:
             automl_best_score = current_best * random.uniform(0.98, 1.0) # Slowly improve
        agent_registry.setdefault("automl_search", {})["sim_best_score"] = automl_best_score
    elif current_best is not None:
         automl_best_score = current_best # Keep last known best if stopped


    # Simulate System Health
    system_cpu_usage_percent = min(100.0, random.uniform(10.0, 70.0) + (20 if ingestion_running else 0) + (30 if rl_running else 0))
    system_memory_usage_percent = min(100.0, random.uniform(20.0, 60.0) + dataset_count * 0.5)

    # --- Construct Response ---
    metrics = DashboardMetrics(
        timestamp=current_time,
        dataset_count=dataset_count,
        total_records_est=total_records_est,
        agent_status_summary=status_counts,
        data_ingestion_rate=round(data_ingestion_rate, 2),
        avg_query_latency_ms=round(avg_query_latency_ms, 1),
        rl_latest_reward=round(rl_latest_reward, 2) if rl_latest_reward is not None else None,
        rl_total_steps=rl_total_steps,
        automl_best_score=round(automl_best_score, 4) if automl_best_score is not None else None,
        automl_trials_completed=automl_trials_completed,
        system_cpu_usage_percent=round(system_cpu_usage_percent, 1),
        system_memory_usage_percent=round(system_memory_usage_percent, 1)
    )

    return metrics

# --- Root Endpoint ---
@app.get("/", include_in_schema=False)
async def read_root():
    return {"message": "Welcome to the Tensorus API! Visit /docs or /redoc for documentation."}

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Tensorus API Server (v0.2.0 with Agent Placeholders) ---")
    print("Access API documentation at http://127.0.0.1:8000/docs or /redoc")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)