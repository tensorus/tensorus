# Tensorus: Agentic Tensor Database/Data Lake

Tensorus is a specialized data platform focused on the management and agent-driven manipulation of tensor data. It offers a streamlined environment for storing, retrieving, and operating on tensors, laying the groundwork for advanced AI and machine learning workflows.

The core purpose of Tensorus is to simplify and enhance how developers and AI agents interact with tensor datasets. By providing dedicated tools for tensor operations and a framework for agentic integration, Tensorus aims to accelerate tasks like automated data ingestion, reinforcement learning from stored experiences, and AutoML processes, ultimately enabling more efficient and intelligent data utilization in AI projects.

## Key Features

*   **Tensor Storage:** Efficiently store and retrieve PyTorch tensors with associated metadata.
*   **Natural Query Language (NQL):** Query your tensor data using a simple, natural language-like syntax.
*   **Agent Framework:** A foundation for building and integrating intelligent agents that interact with your data.
    *   **Data Ingestion Agent:** Automatically monitors a directory for new files and ingests them as tensors.
    *   **RL Agent:** A Deep Q-Network (DQN) agent that can learn from experiences stored in TensorStorage.
    *   **AutoML Agent:** Performs hyperparameter optimization for a dummy model using random search.
*   **API-Driven:** A FastAPI backend provides a RESTful API for interacting with Tensorus.
*   **Streamlit UI:** A user-friendly Streamlit frontend for exploring data and controlling agents.
*   **Tensor Operations:** A comprehensive library of robust tensor operations for common manipulations. See [Basic Tensor Operations](#basic-tensor-operations) for details.
*   **Extensible:** Designed to be extended with more advanced agents, storage backends, and query capabilities.
*   **Model Context Protocol (MCP) Server:** Provides a standardized interface for AI agents and LLMs to interact with Tensorus capabilities (tensor storage and operations) using the Model Context Protocol. (See [MCP Server Details](#mcp-server-details) below).

## Project Structure

*   `app.py`: The main Streamlit frontend application (located at the project root).
*   `pages/`: Directory containing individual Streamlit page scripts and shared UI utilities for the dashboard.
    *   `pages/ui_utils.py`: Utility functions specifically for the Streamlit UI.
    *   *(Other page scripts like `01_dashboard.py`, `02_control_panel.py`, etc., define the different views of the dashboard)*
*   `tensorus/`: Directory containing the core `tensorus` library modules (this is the main installable package).
    *   `tensorus/__init__.py`: Makes `tensorus` a Python package.
    *   `tensorus/api.py`: The FastAPI application providing the backend API for Tensorus.
    *   `tensorus/tensor_storage.py`: Core TensorStorage implementation for managing tensor data.
    *   `tensorus/tensor_ops.py`: Library of functions for tensor manipulations.
    *   `tensorus/nql_agent.py`: Agent for processing Natural Query Language queries.
    *   `tensorus/ingestion_agent.py`: Agent for ingesting data from various sources.
    *   `tensorus/rl_agent.py`: Agent for Reinforcement Learning tasks.
    *   `tensorus/automl_agent.py`: Agent for AutoML processes.
    *   `tensorus/dummy_env.py`: A simple environment for the RL agent demonstration.
    *   *(Other Python files within `tensorus/` are part of the core library.)*
*   `requirements.txt`: Lists the project's Python dependencies for development and local execution.
*   `pyproject.toml`: Project metadata, dependencies for distribution, and build system configuration (e.g., for PyPI).
*   `mcp_tensorus_server/`: Directory for the Node.js Model Context Protocol (MCP) server.
    *   `mcp_tensorus_server/server.js`: The main MCP server implementation.
    *   `mcp_tensorus_server/package.json`: Node.js project file for the MCP server.
*   `README.md`: This file.
*   `LICENSE`: Project license file.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.

## Huggingface Demo

You can try Tensorus online via Huggingface Spaces:

*   **API Documentation:** [Swagger UI](https://tensorus-api.hf.space/docs) | [ReDoc](https://tensorus-api.hf.space/redoc)
*   **Dashboard UI:** [Streamlit Dashboard](https://tensorus-dashboard.hf.space)

## Tensorus Execution Cycle

```mermaid
graph TD
    %% User Interface Layer
    subgraph UI_Layer ["User Interaction"]
        UI[Streamlit UI]
    end
    
    %% API Gateway Layer
    subgraph API_Layer ["Backend Services"]
        API[FastAPI Backend]
    end
    
    %% Core Storage with Method Interface
    subgraph Storage_Layer ["Core Storage - TensorStorage"]
        TS[TensorStorage Core]
        subgraph Storage_Methods ["Storage Interface"]
            TS_insert[insert data metadata]
            TS_query[query query_fn]
            TS_get[get_by_id id]
            TS_sample[sample n]
            TS_update[update_metadata]
        end
        TS --- Storage_Methods
    end
    
    %% Agent Processing Layer
    subgraph Agent_Layer ["Processing Agents"]
        IA[Ingestion Agent]
        NQLA[NQL Agent]
        RLA[RL Agent]
        AutoMLA[AutoML Agent]
    end
    
    %% Tensor Operations Library
    subgraph Ops_Layer ["Tensor Operations"]
        TOps[TensorOps Library]
    end

    %% Primary UI Flow
    UI -->|HTTP Requests| API

    %% API Orchestration
    API -->|Command Dispatch| IA
    API -->|Command Dispatch| NQLA
    API -->|Command Dispatch| RLA
    API -->|Command Dispatch| AutoMLA
    API -->|Direct Query| TS_query

    %% Agent Storage Interactions
    IA -->|Data Ingestion| TS_insert
    
    NQLA -->|Query Execution| TS_query
    NQLA -->|Record Retrieval| TS_get

    RLA -->|State Persistence| TS_insert
    RLA -->|Experience Sampling| TS_sample
    RLA -->|State Retrieval| TS_get
    
    AutoMLA -->|Trial Storage| TS_insert
    AutoMLA -->|Data Retrieval| TS_query

    %% Computational Operations
    NQLA -->|Vector Operations| TOps
    RLA -->|Policy Evaluation| TOps
    AutoMLA -->|Model Optimization| TOps

    %% Indirect Storage Write-back
    TOps -.->|Intermediate Results| TS_insert
```

## Getting Started

### Prerequisites

*   Python 3.9+
*   PyTorch
*   FastAPI
*   Uvicorn
*   Streamlit
*   Pydantic
*   Requests
*   Pillow (for image preprocessing)
*   Matplotlib (optional, for plotting RL rewards)

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/tensorus/tensorus.git
    cd tensorus
    ```

2.  Create a virtual environment (recommended):

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running the API Server

1.  Navigate to the project root directory (the directory containing the `tensorus` folder and `pyproject.toml`).
2.  Ensure your virtual environment is activated if you are using one.
3.  Start the FastAPI backend server using:

    ```bash
    python -m uvicorn tensorus.api:app --reload --host 127.0.0.1 --port 8000
    ```

    *   The `python -m uvicorn` command ensures that Python runs Uvicorn as a module, and `tensorus.api:app` correctly points to the `app` instance within your `tensorus/api.py` file.
    *   `--reload` enables auto-reload for development.
    *   Access the API documentation at `http://127.0.0.1:8000/docs` or `http://127.0.0.1:8000/redoc`.

### Running the Streamlit UI

1.  In a separate terminal (with the virtual environment activated), navigate to the project root.
2.  Start the Streamlit frontend:

    ```bash
    streamlit run app.py
    ```

    *   Access the UI in your browser at the URL provided by Streamlit (usually `http://localhost:8501`).

### Running the MCP Server

The Tensorus MCP Server is a Node.js application that acts as a bridge to the Python backend, exposing Tensorus capabilities as "tools" via the Model Context Protocol.

**Prerequisites (MCP Server):**

*   Node.js (v16 or later recommended) and npm.
*   The Python FastAPI backend (`api.py`) must be running (see [Running the API Server](#running-the-api-server)).

**Setup & Installation (MCP Server):**

1.  Navigate to the MCP server directory:
    ```bash
    cd mcp_tensorus_server
    ```
2.  Install Node.js dependencies:
    ```bash
    npm install
    ```

**Starting the MCP Server:**

1.  Ensure the Python FastAPI backend is running.
2.  From the `mcp_tensorus_server` directory, run:
    ```bash
    node server.js
    ```
    Or, if a start script is added to `mcp_tensorus_server/package.json` (e.g., `"start": "node server.js"`):
    ```bash
    npm start
    ```
3.  The MCP server will connect via stdio by default. MCP clients will communicate with this server process through its standard input and output.

### Running the Agents (Examples)

You can run the example agents directly from their respective files:

*   **RL Agent:**

    ```bash
    python tensorus/rl_agent.py
    ```

*   **AutoML Agent:**

    ```bash
    python tensorus/automl_agent.py
    ```

*   **Ingestion Agent:**

    ```bash
    python tensorus/ingestion_agent.py
    ```

    *   Note: The Ingestion Agent will monitor the `temp_ingestion_source` directory (created automatically if it doesn't exist in the project root) for new files.

## Using Tensorus

### API Endpoints

The API provides the following main endpoints:

*   **Datasets:**
    *   `POST /datasets/create`: Create a new dataset.
    *   `POST /datasets/{name}/ingest`: Ingest a tensor into a dataset.
    *   `GET /datasets/{name}/fetch`: Retrieve all records from a dataset.
    *   `GET /datasets`: List all available datasets.
*   **Querying:**
    *   `POST /query`: Execute an NQL query.
*   **Agents:**
    *   `GET /agents`: List all registered agents.
    *   `GET /agents/{agent_id}/status`: Get the status of a specific agent.
    *   `POST /agents/{agent_id}/start`: Start an agent.
    *   `POST /agents/{agent_id}/stop`: Stop an agent.
    *   `GET /agents/{agent_id}/logs`: Get recent logs for an agent.
*   **Metrics & Monitoring:**
    *   `GET /metrics/dashboard`: Get aggregated dashboard metrics.

### Streamlit UI

The Streamlit UI provides a user-friendly interface for:

*   **Dashboard:** View basic system metrics and agent status.
*   **Agent Control:** Start, stop, and view logs for agents.
*   **NQL Chat:** Enter natural language queries and view results.
*   **Data Explorer:** Browse datasets, preview data, and perform tensor operations.

## Agent Details

### Data Ingestion Agent

*   **Functionality:** Monitors a source directory for new files, preprocesses them into tensors, and inserts them into TensorStorage.
*   **Supported File Types:** CSV, PNG, JPG, JPEG, TIF, TIFF (can be extended).
*   **Preprocessing:** Uses default functions for CSV and images (resize, normalize).
*   **Configuration:**
    *   `source_directory`: The directory to monitor.
    *   `polling_interval_sec`: How often to check for new files.
    *   `preprocessing_rules`: A dictionary mapping file extensions to custom preprocessing functions.

### RL Agent

*   **Functionality:** A Deep Q-Network (DQN) agent that learns from experiences stored in TensorStorage.
*   **Environment:** Uses a `DummyEnv` for demonstration.
*   **Experience Storage:** Stores experiences (state, action, reward, next_state, done) in TensorStorage.
*   **Training:** Implements epsilon-greedy exploration and target network updates.
*   **Configuration:**
    *   `state_dim`: Dimensionality of the environment state.
    *   `action_dim`: Number of discrete actions.
    *   `hidden_size`: Hidden layer size for the DQN.
    *   `lr`: Learning rate.
    *   `gamma`: Discount factor.
    *   `epsilon_*`: Epsilon-greedy parameters.
    *   `target_update_freq`: Target network update frequency.
    *   `batch_size`: Experience batch size.
    *   `experience_dataset`: Dataset name for experiences.
    *   `state_dataset`: Dataset name for state tensors.

### AutoML Agent

*   **Functionality:** Performs hyperparameter optimization using random search.
*   **Model:** Trains a simple `DummyMLP` model.
*   **Search Space:** Configurable hyperparameter search space (learning rate, hidden size, activation).
*   **Evaluation:** Trains and evaluates models on synthetic data.
*   **Results:** Stores trial results (parameters, score) in TensorStorage.
*   **Configuration:**
    *   `search_space`: Dictionary defining the hyperparameter search space.
    *   `input_dim`: Input dimension for the model.
    *   `output_dim`: Output dimension for the model.
    *   `task_type`: Type of task ('regression' or 'classification').
    *   `results_dataset`: Dataset name for storing results.

## Basic Tensor Operations

This section details the core tensor manipulation functionalities provided by `tensor_ops.py`. These operations are designed to be robust, with built-in type and shape checking where appropriate.

#### Arithmetic Operations

*   `add(t1, t2)`: Element-wise addition of two tensors, or a tensor and a scalar.
*   `subtract(t1, t2)`: Element-wise subtraction of two tensors, or a tensor and a scalar.
*   `multiply(t1, t2)`: Element-wise multiplication of two tensors, or a tensor and a scalar.
*   `divide(t1, t2)`: Element-wise division of two tensors, or a tensor and a scalar. Includes checks for division by zero.

#### Matrix and Dot Operations

*   `matmul(t1, t2)`: Matrix multiplication of two tensors, supporting various dimensionalities (e.g., 2D matrices, batched matrix multiplication).
*   `dot(t1, t2)`: Computes the dot product of two 1D tensors.

#### Reduction Operations

*   `sum(tensor, dim=None, keepdim=False)`: Computes the sum of tensor elements over specified dimensions.
*   `mean(tensor, dim=None, keepdim=False)`: Computes the mean of tensor elements over specified dimensions. Tensor is cast to float for calculation.
*   `min(tensor, dim=None, keepdim=False)`: Finds the minimum value in a tensor, optionally along a dimension. Returns values and indices if `dim` is specified.
*   `max(tensor, dim=None, keepdim=False)`: Finds the maximum value in a tensor, optionally along a dimension. Returns values and indices if `dim` is specified.

#### Reshaping and Slicing

*   `reshape(tensor, shape)`: Changes the shape of a tensor without changing its data.
*   `transpose(tensor, dim0, dim1)`: Swaps two dimensions of a tensor.
*   `permute(tensor, dims)`: Permutes the dimensions of a tensor according to the specified order.

#### Concatenation and Splitting

*   `concatenate(tensors, dim=0)`: Joins a sequence of tensors along an existing dimension.
*   `stack(tensors, dim=0)`: Joins a sequence of tensors along a new dimension.

#### Advanced Operations

*   `einsum(equation, *tensors)`: Applies Einstein summation convention to the input tensors based on the provided equation string.

## MCP Server Details

The Tensorus Model Context Protocol (MCP) Server allows external AI agents, LLM-based applications, and other MCP-compatible clients to interact with Tensorus functionalities in a standardized way. It acts as a bridge, translating MCP requests into calls to the Tensorus Python API.

### Overview

*   **Protocol:** Implements the [Model Context Protocol](https://modelcontextprotocol.io/introduction).
*   **Language:** Node.js, using the `@modelcontextprotocol/sdk`.
*   **Communication:** Typically uses stdio for communication with a single client.
*   **Interface:** Exposes Tensorus capabilities as a set of "tools" that an MCP client can list and call.

### Available Tools

The MCP server provides tools for various Tensorus functionalities. Below is an overview. For detailed input schemas and descriptions, an MCP client can call the standard `tools/list` method on the server, or you can inspect the `toolDefinitions` array within `mcp_tensorus_server/server.js`.

*   **Dataset Management:**
    *   `tensorus_list_datasets`: Lists all available datasets.
    *   `tensorus_create_dataset`: Creates a new dataset.
    *   `tensorus_delete_dataset`: Deletes an existing dataset.
*   **Tensor Management:**
    *   `tensorus_ingest_tensor`: Ingests a new tensor (with data provided as JSON) into a dataset.
    *   `tensorus_get_tensor_details`: Retrieves the data and metadata for a specific tensor.
    *   `tensorus_delete_tensor`: Deletes a specific tensor from a dataset.
    *   `tensorus_update_tensor_metadata`: Updates the metadata of a specific tensor.
*   **Tensor Operations:** These tools allow applying operations from the `TensorOps` library to tensors stored in Tensorus.
    *   `tensorus_apply_unary_operation`: Applies operations like `log`, `reshape`, `transpose`, `permute`, `sum`, `mean`, `min`, `max`.
    *   `tensorus_apply_binary_operation`: Applies operations like `add`, `subtract`, `multiply`, `divide`, `power`, `matmul`, `dot`.
    *   `tensorus_apply_list_operation`: Applies operations like `concatenate` and `stack` that take a list of input tensors.
    *   `tensorus_apply_einsum`: Applies Einstein summation.

*Note on Tensor Operations via MCP:* Input tensors are referenced by their `dataset_name` and `record_id`. The result is typically stored as a new tensor, and the MCP tool returns details of this new result tensor (like its `record_id`).

### Example Client Interaction (Conceptual)

```javascript
// Conceptual MCP client-side JavaScript
// Assuming 'client' is an initialized MCP client connected to the Tensorus MCP Server

async function example() {
  // List available tools
  const { tools } = await client.request({ method: 'tools/list' }, {});
  console.log("Available Tensorus Tools:", tools.map(t => t.name)); // Log only names for brevity

  // Create a new dataset
  const createResponse = await client.request({ method: 'tools/call' }, {
    name: 'tensorus_create_dataset',
    arguments: { dataset_name: 'my_mcp_dataset' }
  });
  console.log(JSON.parse(createResponse.content[0].text).message);

  // Ingest a tensor
  const ingestResponse = await client.request({ method: 'tools/call' }, {
    name: 'tensorus_ingest_tensor',
    arguments: {
      dataset_name: 'my_mcp_dataset',
      tensor_shape: [2, 2],
      tensor_dtype: 'float32',
      tensor_data: [[1.0, 2.0], [3.0, 4.0]],
      metadata: { source: 'mcp_client_example' }
    }
  });
  // Assuming the Python API returns { success, message, data: { record_id, ... } }
  // And MCP server stringifies this whole object in the text content
  const ingestData = JSON.parse(ingestResponse.content[0].text);
  console.log("Ingest success:", ingestData.success, "Record ID:", ingestData.data.record_id);
}
```

## Completed Features

*   **Tensor Storage:** Efficiently stores and retrieves PyTorch tensors with associated metadata, including in-memory and optional file-based persistence. Supports dataset creation, tensor ingestion, querying, sampling, and metadata updates.
*   **Natural Query Language (NQL):** Provides a basic regex-based natural language interface for querying tensor data, supporting retrieval and simple filtering.
*   **Agent Framework:** Includes several operational agents:
    *   **Data Ingestion Agent:** Monitors local directories for CSV and image files, preprocesses them, and ingests them into TensorStorage.
    *   **RL Agent:** Implements a DQN agent that learns from experiences (stored in TensorStorage) in a dummy environment.
    *   **AutoML Agent:** Performs random search hyperparameter optimization for a dummy MLP model, storing trial results in TensorStorage.
*   **API-Driven:** A comprehensive FastAPI backend offers RESTful endpoints for dataset management, NQL querying, tensor operations, and agent control (live for Ingestion Agent, simulated for RL/AutoML).
*   **Streamlit UI:** A multi-page user interface for dashboard overview, agent control, NQL interaction, data exploration, and API interaction.
*   **Tensor Operations:** A library of robust tensor operations (arithmetic, matrix ops, reductions, reshaping, etc.) accessible via the API.
*   **Model Context Protocol (MCP) Server:** A Node.js server exposes Tensorus capabilities (storage and operations) as tools via the Model Context Protocol, bridging to the Python API.
*   **Extensible Design:** The project is structured with modular components, facilitating future extensions.

## Future Implementation

*   **Enhanced NQL:** Integrate a local or remote LLM for more robust natural language understanding.
*   **Advanced Agents:** Develop more sophisticated agents for specific tasks (e.g., anomaly detection, forecasting).
*   **Persistent Storage Backend:** Replace/augment current file-based persistence with more robust database or cloud storage solutions (e.g., PostgreSQL, S3, MinIO).
*   **Scalability & Performance:**
    *   Implement tensor chunking for very large tensors.
    *   Optimize query performance with indexing.
    *   Asynchronous operations for agents and API calls.
*   **Security:** Implement authentication and authorization mechanisms for the API and UI.
*   **Real-World Integration:**
    *   Connect Ingestion Agent to more data sources (e.g., cloud storage, databases, APIs).
    *   Integrate RL Agent with real-world environments or more complex simulations.
*   **Advanced AutoML:**
    *   Implement sophisticated search algorithms (e.g., Bayesian Optimization, Hyperband).
    *   Support for diverse model architectures and custom models.
*   **Model Management:** Add capabilities for saving, loading, versioning, and deploying trained models (from RL/AutoML).
*   **Streaming Data Support:** Enhance Ingestion Agent to handle real-time streaming data.
*   **Dataset Schema Validation:** Introduce mechanisms for defining and enforcing schemas for datasets in TensorStorage.
*   **Resource Management:** Add tools and controls for monitoring and managing the resource consumption (CPU, memory) of agents.
*   **Improved UI/UX:** Continuously refine the Streamlit UI for better usability and richer visualizations.
*   **Comprehensive Testing:** Expand unit, integration, and end-to-end tests.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

MIT License
