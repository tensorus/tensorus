# app.py (Revised)
"""
app.py - Main entry point for the extended Tensorus platform (FastAPI version).

This file exposes FastAPI endpoints for:
  • A Unified Operations Dashboard (via WebSocket for metrics).
  • A Multi-Agent Control Panel (API endpoints for status/control placeholders).
  * A Natural Language Query Chatbot endpoint.
  * An Interactive Data Explorer endpoint (basic tensor operations).
  * Built-in API Playground via FastAPI's /docs and /redoc endpoints.

Prerequisites: The existing modules (tensor_storage.py, rl_agent.py, automl_agent.py,
nql_agent.py, tensor_ops.py, dummy_env.py etc.) must exist in the same project directory or be importable.
"""

import logging
import asyncio
import json
import random
import time
from typing import List, Dict, Any, Optional, Set, Union

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Path, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field # Use Pydantic for request bodies
import uvicorn
import torch

# Import modules from Tensorus project
try:
    from tensor_storage import TensorStorage
    # Assuming RLAgent init requires state/action dims matching DummyEnv + storage
    from rl_agent import RLAgent
    from dummy_env import DummyEnv
    # Assuming AutoMLAgent needs storage, search_space, dims etc.
    from automl_agent import AutoMLAgent
    from nql_agent import NQLAgent
    from tensor_ops import TensorOps # Import TensorOps correctly
except ImportError as e:
    print(f"Error importing Tensorus modules: {e}. Please ensure all required .py files are present.")
    raise

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Initialize FastAPI app ---
app = FastAPI(
    title="Tensorus Agentic Platform API",
    description="Extended interactive service layer for Tensorus: dashboard, agent control, NQL chatbot, data explorer.",
    version="1.0.0" # Updated version
)

# --- CORS Middleware ---
# Allow all origins for development ease, restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # e.g., ["http://localhost:3000", "http://127.0.0.1:8501"] for frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Instances and State ---
# Initialize global TensorStorage (in-memory)
storage = TensorStorage()

# Create default datasets if they don't exist
try:
    storage.create_dataset("rl_experiences")
    storage.create_dataset("rl_states") # Needed by RLAgent implementation
    storage.create_dataset("automl_results")
    # Add a dataset for testing explorer
    storage.create_dataset("sample_data")
    storage.insert("sample_data", torch.randn(5, 3, 10))
    storage.insert("sample_data", torch.rand(5, 3, 10) * 10)

except ValueError:
    logger.info("Datasets may already exist, skipping creation.")

# Initialize agents (ensure parameters match module definitions)
try:
    dummy_env = DummyEnv()
    # Correct RLAgent initialization based on its expected signature
    rl_agent_instance = RLAgent(
        tensor_storage=storage,
        state_dim=dummy_env.state_dim,
        action_dim=dummy_env.action_dim
        # Use default hyperparameters or load from config
    )

    # Correct AutoMLAgent initialization (provide a basic search space)
    # TODO: Define search space properly, potentially load from config
    basic_search_space = {
        'lr': lambda: 10**random.uniform(-5, -2),
        'hidden_size': lambda: random.choice([32, 64, 128]),
    }
    automl_agent_instance = AutoMLAgent(
        tensor_storage=storage,
        search_space=basic_search_space,
        input_dim=10, # Example input dim
        output_dim=1, # Example output dim
        results_dataset="automl_results" # Ensure dataset name matches
    )
    nql_agent_instance = NQLAgent(tensor_storage=storage)
except Exception as e:
     logger.exception(f"Error initializing agents: {e}")
     raise RuntimeError(f"Agent initialization failed: {e}") from e


# Global dictionary for placeholder agent status and in-memory logs
# TODO: Replace simple logs with a proper logging service integration
agent_status: Dict[str, Dict[str, Any]] = {
    "ingestion": {
        "name": "Data Ingestion",
        "running": False,
        "log": ["Ingestion agent initialized."],
        "config": {"source_directory": "simulated_source", "polling_interval": 15}
        },
    "rl_trainer": {
        "name": "RL Trainer",
        "running": False,
        "log": ["RL Trainer initialized."],
        "config": rl_agent_instance.__dict__.get('config', {}) # Try to get config if agent stores it
        },
    "automl_search": {
        "name": "AutoML Search",
        "running": False,
        "log": ["AutoML Search initialized."],
        "config": {"search_space_keys": list(basic_search_space.keys()), "results_dataset": "automl_results"}
        },
    "nql_service": {
        "name": "NQL Query Service",
        "running": True, # Part of API, always running
        "log": ["NQL Service ready."],
        "config": {"parser_type": "regex"}
        },
}

# --- Real-time Dashboard Logic ---
class DashboardManager:
    """Manages WebSocket connections and broadcasts dashboard metrics."""
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.metrics: Dict[str, Any] = {
            "timestamp": time.time(),
            "ingestion_rate": 0.0,
            "query_latency_ms": 0.0,
            "rl_episode": 0,
            "rl_latest_reward": None,
            "automl_trials": 0,
            "automl_best_score": None,
            "active_connections": 0,
            # Add more metrics as needed
        }
        self._lock = asyncio.Lock() # Lock for managing connections

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
            self.metrics["active_connections"] = len(self.active_connections)
        await self.send_update(websocket) # Send current state on connect

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.remove(websocket)
            self.metrics["active_connections"] = len(self.active_connections)
        # Optionally broadcast the updated connection count
        # await self.broadcast()

    async def send_update(self, websocket: WebSocket):
         """Sends the current metrics to a single client."""
         try:
              await websocket.send_text(json.dumps(self.metrics))
         except Exception as e:
              logger.warning(f"Failed to send metrics to client: {e}")


    async def broadcast(self):
        """Broadcasts current metrics to all connected clients."""
        async with self._lock:
             if self.active_connections:
                 message = json.dumps(self.metrics)
                 # Use asyncio.gather for concurrent sending
                 results = await asyncio.gather(
                     *[client.send_text(message) for client in self.active_connections],
                     return_exceptions=True # Don't crash if one client fails
                 )
                 # Log errors or handle disconnected clients based on results
                 for i, result in enumerate(results):
                      if isinstance(result, Exception):
                           # Client likely disconnected, consider removing from active_connections
                           logger.warning(f"Error broadcasting to client {i}: {result}")


dashboard_manager = DashboardManager()

async def update_metrics_periodically():
    """Coroutine to periodically update and broadcast metrics."""
    # TODO: Fetch real metrics from agents/storage when available
    while True:
        try:
            dashboard_manager.metrics["timestamp"] = time.time()
            # Simulate metric updates
            if agent_status["ingestion"].get("running"):
                dashboard_manager.metrics["ingestion_rate"] = round(random.uniform(5.0, 50.0), 1)
            else:
                 dashboard_manager.metrics["ingestion_rate"] = round(random.uniform(0.0, 2.0), 1)

            dashboard_manager.metrics["query_latency_ms"] = round(random.uniform(50.0, 300.0), 1)

            if agent_status["rl_trainer"].get("running"):
                dashboard_manager.metrics["rl_episode"] += random.randint(0, 1)
                dashboard_manager.metrics["rl_latest_reward"] = round(random.gauss(0, 5.0), 2) # Simulate fluctuating reward
            # else: keep last known reward? Or set to None? Set to None if stopped.
            #     dashboard_manager.metrics["rl_latest_reward"] = None


            if agent_status["automl_search"].get("running"):
                 dashboard_manager.metrics["automl_trials"] += random.randint(0, 1)
                 # Simulate best score (assuming lower is better)
                 current_best = dashboard_manager.metrics.get("automl_best_score")
                 if current_best is None or random.random() < 0.1: # Chance of finding better
                      new_best = random.uniform(0.1, 5.0) * (current_best if current_best else 1.0) * 0.95
                      dashboard_manager.metrics["automl_best_score"] = round(new_best, 4)
            # else: keep last known score

            await dashboard_manager.broadcast()
            await asyncio.sleep(2) # Update every 2 seconds
        except Exception as e:
            logger.exception(f"Error in metrics update loop: {e}")
            await asyncio.sleep(10) # Wait longer if error occurs


@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup."""
    logger.info("Starting background metrics update task...")
    # Run the periodic update within the main asyncio loop
    asyncio.create_task(update_metrics_periodically())


# --- API Endpoints ---

# 1. Unified Operations Dashboard
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
async def get_dashboard_html():
    """Serves a simple HTML dashboard using WebSocket for live metrics."""
    # In a real application, this HTML would likely be served by a separate
    # frontend framework (React, Vue, etc.) or a more sophisticated
    # Python dashboarding library integration (like Dash within FastAPI).
    html_content = """
    <!DOCTYPE html>
    <html>
    <head> <title>Tensorus Dashboard</title> </head>
    <body>
      <h1>Tensorus Live Dashboard</h1>
      <pre id="metrics" style="white-space: pre-wrap; word-wrap: break-word;"></pre>
      <script>
        const metricsElement = document.getElementById('metrics');
        let ws;
        function connect() {
          ws = new WebSocket(`ws://${window.location.host}/ws/dashboard`); // Use relative host
          metricsElement.textContent = 'Connecting...';
          ws.onmessage = function(event) {
            try {
              const data = JSON.parse(event.data);
              metricsElement.textContent = JSON.stringify(data, null, 2);
            } catch (e) {
              metricsElement.textContent = 'Error parsing metrics: ' + event.data;
            }
          };
          ws.onclose = function(event) {
            metricsElement.textContent = 'Connection closed. Attempting to reconnect in 5 seconds...';
            setTimeout(connect, 5000); // Attempt to reconnect
          };
          ws.onerror = function(error) {
             metricsElement.textContent = 'WebSocket Error. Check console.';
             console.error('WebSocket Error:', error);
             ws.close(); // Ensure close is called on error before reconnect attempt
          };
        }
        connect(); // Initial connection attempt
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws/dashboard", name="dashboard_websocket")
async def dashboard_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint to stream dashboard metrics."""
    await dashboard_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, optionally handle client pings
            await websocket.receive_text() # Wait for message (or disconnect)
            # await websocket.send_text("Pong") # Example pong response
    except WebSocketDisconnect:
        logger.info("Dashboard client disconnected.")
    except Exception as e:
         logger.error(f"Dashboard WebSocket error: {e}")
    finally:
        await dashboard_manager.disconnect(websocket)


# 2. Multi-Agent Control Panel Endpoints

# Pydantic model for agent configuration payload
class AgentConfigPayload(BaseModel):
     config: Dict[str, Any] = Field(..., description="New configuration dictionary for the agent.")

@app.get("/agents/status", response_model=Dict[str, Dict[str, Any]], tags=["Agents"])
async def get_all_agent_status():
    """Returns the current status, logs (limited), and config for all agents."""
    # Return a copy to avoid direct modification issues if any
    status_copy = {}
    for agent_id, data in agent_status.items():
         status_copy[agent_id] = {
             "name": data["name"],
             "running": data["running"],
             "config": data["config"],
             "logs": data["log"][-10:] # Return last 10 log entries for brevity
         }
    return status_copy


@app.post("/agents/{agent_id}/start", response_model=ApiResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Agents"])
async def start_agent(agent_id: str = Path(..., description="ID of the agent (e.g., 'ingestion', 'rl_trainer').")):
    """Starts a specified agent (Placeholder Action)."""
    if agent_id not in agent_status:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found")
    if agent_status[agent_id]["running"]:
         return ApiResponse(success=False, message=f"Agent '{agent_id}' is already running.")

    logger.info(f"API: Received start signal for agent '{agent_id}' (Placeholder).")
    agent_status[agent_id]["running"] = True
    agent_status[agent_id]["log"].append(f"{time.strftime('%H:%M:%S')} - Agent '{agent_id}' started (simulated).")
    # TODO: Implement actual agent process starting logic (e.g., using subprocess, celery, k8s job)
    return ApiResponse(success=True, message=f"Start signal sent to agent '{agent_id}'.")


@app.post("/agents/{agent_id}/stop", response_model=ApiResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Agents"])
async def stop_agent(agent_id: str = Path(..., description="ID of the agent.")):
    """ Stops a specified agent (Placeholder Action)."""
    if agent_id not in agent_status:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found")
    if not agent_status[agent_id]["running"]:
         return ApiResponse(success=False, message=f"Agent '{agent_id}' is already stopped.")

    logger.info(f"API: Received stop signal for agent '{agent_id}' (Placeholder).")
    agent_status[agent_id]["running"] = False
    agent_status[agent_id]["log"].append(f"{time.strftime('%H:%M:%S')} - Agent '{agent_id}' stopped (simulated).")
    # TODO: Implement actual agent process stopping logic
    return ApiResponse(success=True, message=f"Stop signal sent to agent '{agent_id}'.")


@app.post("/agents/{agent_id}/configure", response_model=ApiResponse, tags=["Agents"])
async def configure_agent(agent_id: str = Path(..., description="ID of the agent."), payload: AgentConfigPayload = Body(...)):
    """Dynamically configures an agent (updates placeholder config)."""
    if agent_id not in agent_status:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found")

    new_config = payload.config
    logger.info(f"API: Received configuration update for agent '{agent_id}': {new_config}")
    # Update the placeholder config
    agent_status[agent_id]["config"].update(new_config)
    agent_status[agent_id]["log"].append(f"{time.strftime('%H:%M:%S')} - Agent '{agent_id}' reconfigured (simulated): {new_config}")
    # TODO: Implement logic to apply configuration to the actual running agent process
    return ApiResponse(success=True, message=f"Agent '{agent_id}' configuration updated.", data={"new_config": agent_status[agent_id]["config"]})


# 3. Natural Language Query Chatbot Endpoint

# Pydantic model for chat query
class ChatQueryRequest(BaseModel):
     query: str = Field(..., description="The natural language query from the user.")

# Pydantic model for chat response
class ChatQueryResponse(BaseModel):
    query: str
    response_text: str
    results: Optional[List[Dict[str, Any]]] = None # Include structured results if applicable
    error: Optional[str] = None

@app.post("/chat/query", response_model=ChatQueryResponse, tags=["Chatbot"])
async def chat_query_endpoint(request: ChatQueryRequest):
    """Processes a natural language query via NQL agent (Regex version)."""
    query_str = request.query
    logger.info(f"Received chat query: '{query_str}'")
    # Use the existing NQL Agent (regex based)
    try:
        nql_result = nql_agent_instance.process_query(query_str)
        # TODO: Integrate real LLM here for better understanding and response generation.

        if nql_result.get("success"):
            response_text = nql_result.get("message", "Query processed.")
            count = nql_result.get("count")
            if count is not None:
                 response_text += f" Found {count} record(s)."

            serialized_results = None
            if nql_result.get("results"):
                 serialized_results = []
                 for record in nql_result["results"]:
                      # Serialize tensor to list for JSON response
                      shape, dtype, data_list = tensor_to_list(record['tensor'])
                      serialized_results.append({
                          "record_id": record['metadata'].get('record_id', 'N/A'),
                          "shape": shape,
                          "dtype": dtype,
                          "data_preview": data_list[:min(len(data_list), 5)], # Preview data
                          "metadata": record['metadata']
                      })
            return ChatQueryResponse(query=query_str, response_text=response_text, results=serialized_results)
        else:
            # NQL agent reported failure (parsing error, etc.)
            return ChatQueryResponse(query=query_str, response_text="Sorry, I couldn't process that query.", error=nql_result.get("message"))

    except Exception as e:
        logger.exception(f"Error processing chat query '{query_str}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing query: {e}")


# 4. Interactive Data Explorer Endpoints

# Pydantic models for Explorer
class ExplorerQueryPayload(BaseModel):
    dataset: str = Field(..., description="Target dataset name.")
    operation: str = Field(..., description="Tensor operation (e.g., 'slice', 'info', 'head').")
    params: Optional[Dict[str, Any]] = Field({}, description="Parameters for the operation.")
    tensor_index: int = Field(0, description="Index of the tensor within the dataset to operate on.")


@app.get("/explorer/datasets", response_model=Dict[str, List[str]], tags=["Data Explorer"])
async def list_explorer_datasets():
    """Lists dataset names available for exploration."""
    # TODO: Add filtering or pagination if many datasets exist
    datasets = list(storage.datasets.keys())
    return {"datasets": datasets}


@app.get("/explorer/dataset/{dataset_name}/preview", response_model=Dict[str, Any], tags=["Data Explorer"])
async def get_dataset_preview(dataset_name: str = Path(..., description="Dataset name."), limit: int = 5):
    """Retrieves metadata and tensor previews for a dataset."""
    try:
        # Fetch records with metadata
        records = storage.get_dataset_with_metadata(dataset_name)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    preview_data = []
    for i, record in enumerate(records):
        if i >= limit:
            break
        shape, dtype, data_list = tensor_to_list(record['tensor'])
        preview_data.append({
             "index": i,
             "record_id": record['metadata'].get('record_id', 'N/A'),
             "shape": shape,
             "dtype": dtype,
             "metadata": record['metadata'],
             "data_preview": data_list[:min(len(data_list), 5)] # Preview only first few elements
        })
    return {"dataset": dataset_name, "record_count": len(records), "preview": preview_data}


@app.post("/explorer/operate", response_model=Dict[str, Any], tags=["Data Explorer"])
async def explorer_operate_on_tensor(payload: ExplorerQueryPayload = Body(...)):
    """Performs a specified operation on a tensor within a dataset."""
    dataset_name = payload.dataset
    operation = payload.operation.lower()
    params = payload.params
    tensor_index = payload.tensor_index

    try:
        # Fetch the specific tensor (more efficient than getting the whole dataset)
        # NOTE: get_dataset() gets all; need a get_tensor_at_index method ideally.
        # Workaround: get all and index. Inefficient for large datasets.
        dataset_tensors = storage.get_dataset(dataset_name)
        if not dataset_tensors or tensor_index >= len(dataset_tensors):
             raise HTTPException(status_code=404, detail=f"Tensor index {tensor_index} out of bounds for dataset '{dataset_name}'")
        tensor = dataset_tensors[tensor_index]
    except ValueError as e: # Dataset not found
        raise HTTPException(status_code=404, detail=str(e))
    except IndexError:
         raise HTTPException(status_code=404, detail=f"Tensor index {tensor_index} invalid.")

    result = None
    result_metadata = {"operation": operation, "params": params, "original_shape": list(tensor.shape)}

    try:
        if operation == "info":
            result_metadata["dtype"] = str(tensor.dtype)
            result_metadata["device"] = str(tensor.device)
            result_metadata["num_elements"] = tensor.numel()
            result = "See metadata for info." # No tensor data returned for 'info'
        elif operation == "head":
            count = params.get("count", 5)
            result = tensor.flatten()[:count] # Get first few flattened elements
        elif operation == "slice":
            # Basic slicing on a specified dimension
            dim = params.get("dim", 0)
            start = params.get("start", 0)
            end = params.get("end", None) # Slice to end if not specified
            if not isinstance(dim, int) or not isinstance(start, int):
                 raise ValueError("Slice 'dim' and 'start' must be integers.")
            if end is not None and not isinstance(end, int):
                 raise ValueError("Slice 'end' must be an integer if provided.")

            # Construct slice object dynamically
            slices = [slice(None)] * tensor.ndim # slice(None) means ':'
            if dim < 0 or dim >= tensor.ndim:
                raise ValueError(f"Dimension {dim} out of range for tensor with {tensor.ndim} dimensions.")
            slices[dim] = slice(start, end) # Create the slice for the target dimension

            result = tensor[tuple(slices)] # Apply the tuple of slices
            result_metadata["result_shape"] = list(result.shape)
        # TODO: Add more operations using TensorOps or torch directly (e.g., 'mean', 'sum', 'reshape')
        # Example using TensorOps if available and has static methods:
        # elif operation == "mean":
        #     dim = params.get("dim", None) # Optional dimension
        #     result = TensorOps.mean(tensor, dim=dim) # Assuming TensorOps.mean exists
        else:
            raise ValueError(f"Unsupported operation: '{operation}'")

        # Serialize result tensor if it's a tensor
        if isinstance(result, torch.Tensor):
            _, _, result_list = tensor_to_list(result)
        else:
             result_list = result # Use result directly if not a tensor (like info message)

        return {"success": True, "metadata": result_metadata, "result_data": result_list}

    except Exception as e:
        logger.error(f"Error during explorer operation '{operation}' on dataset '{dataset_name}': {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Operation failed: {e}")


# 5. API Playground and Documentation Hub (Provided by FastAPI)
# Endpoints /docs and /redoc are automatically available.


# --- Root Endpoint ---
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint providing platform overview and documentation links."""
    return JSONResponse(content={
        "message": "Welcome to Tensorus - Agentic Tensor Database/Data Lake API",
        "version": app.version,
        "documentation": {
            "Swagger_UI": "/docs",
            "ReDoc": "/redoc"
        },
        "features": {
             "Dashboard_Metrics": "/metrics/dashboard (WebSocket at /ws/dashboard)",
             "Agent_Control": "/agents/status",
             "NQL_Chat": "/chat/query (POST)",
             "Data_Explorer": "/explorer/datasets",
        }
    })

# --- Main Execution ---
if __name__ == "__main__":
    logger.info(f"Starting Tensorus FastAPI Server v{app.version}...")
    # Recommended: Use 'uvicorn app:app --reload' from terminal for development
    uvicorn.run(app, host="127.0.0.1", port=8000)