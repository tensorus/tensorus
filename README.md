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

## Project Structure

*   `api.py`: The FastAPI application that serves as the backend API.
*   `app.py`: The Streamlit frontend application.
*   `tensor_storage.py`: The core TensorStorage implementation.
*   `nql_agent.py`: The Natural Query Language (NQL) agent.
*   `ingestion_agent.py`: The Data Ingestion agent.
*   `rl_agent.py`: The Reinforcement Learning (DQN) agent.
*   `automl_agent.py`: The AutoML agent.
*   `tensor_ops.py`: A library of tensor operations.
*   `dummy_env.py`: A simple environment for the RL agent.
*   `ui_utils.py`: Utility functions for the Streamlit UI.
*   `requirements.txt`: Lists the project's dependencies.

## Huggingface Demo

You can try Tensorus online via Huggingface Spaces:

*   **API Documentation:** [Swagger UI](https://tensorus-api.hf.space/docs) | [ReDoc](https://tensorus-api.hf.space/redoc)
*   **Dashboard UI:** [Streamlit Dashboard](https://tensorus-dashboard.hf.space)

## Tensorus Execution Cycle

```mermaid
graph TD
    subgraph User_Interaction_Graph ["User Interaction"]
        UI[Streamlit UI (app.py)]
    end

    subgraph Backend_Services_Graph ["Backend Services"]
        API[FastAPI Backend (api.py)]
    end

    subgraph Core_Storage_Graph ["Core Storage"]
        TS[TensorStorage (tensor_storage.py)]
    end

    subgraph Agents_Graph ["Agents"]
        IA[Ingestion Agent (ingestion_agent.py)]
        NQLA[NQL Agent (nql_agent.py)]
        RLA[RL Agent (rl_agent.py)]
        AutoMLA[AutoML Agent (automl_agent.py)]
    end

    subgraph Tensor_Operations_Library_Graph ["Tensor Operations Library"]
        TOps[TensorOps (tensor_ops.py)]
    end

    UI -- HTTP Requests --> API
    API -- Interacts with --> TS
    API -- Controls/Manages --> IA
    API -- Controls/Manages --> NQLA
    API -- Controls/Manages --> RLA
    API -- Controls/Manages --> AutoMLA

    IA -- Stores/Retrieves Data --> TS
    NQLA -- Queries Data from --> TS
    RLA -- Stores/Retrieves Experiences & States --> TS
    AutoMLA -- Stores/Retrieves Trial Data --> TS

    IA -- Uses --> TOps
    RLA -- Uses --> TOps
    AutoMLA -- Uses --> TOps
    TS -- May Use (Internally) --> TOps

    style UI fill:#f9f,stroke:#333,stroke-width:2px
    style API fill:#ccf,stroke:#333,stroke-width:2px
    style TS fill:#cfc,stroke:#333,stroke-width:2px
    style Agents fill:#ff9,stroke:#333,stroke-width:2px
    style TOps fill:#fcf,stroke:#333,stroke-width:2px
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

### Running the API

1.  Navigate to the project root directory.
2.  Start the FastAPI backend:

    ```bash
    uvicorn api:app --reload --host 127.0.0.1 --port 8000
    ```

    *   Access the API documentation at `http://127.0.0.1:8000/docs` or `http://127.0.0.1:8000/redoc`.

### Running the Streamlit UI

1.  In a separate terminal (with the virtual environment activated), navigate to the project root.
2.  Start the Streamlit frontend:

    ```bash
    streamlit run app.py
    ```

    *   Access the UI in your browser at the URL provided by Streamlit (usually `http://localhost:8501`).

### Running the Agents (Examples)

You can run the example agents directly from their respective files:

*   **RL Agent:**

    ```bash
    python rl_agent.py
    ```

*   **AutoML Agent:**

    ```bash
    python automl_agent.py
    ```

*   **Ingestion Agent:**

    ```bash
    python ingestion_agent.py
    ```

    *   Note: The Ingestion Agent will monitor the `temp_ingestion_source` directory (created automatically) for new files.

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

## Future Work

*   **Enhanced NQL:** Integrate a local or remote LLM for more robust natural language understanding.
*   **Advanced Agents:** Develop more sophisticated agents for specific tasks.
*   **Persistent Storage:** Replace in-memory TensorStorage with a persistent backend (e.g., database, cloud storage).
*   **Scalability:** Implement chunking and other optimizations for handling very large tensors.
*   **Security:** Add authentication and authorization.
*   **Real-World Data:** Integrate with real-world data sources and environments.
*   **Advanced Search:** Implement more advanced hyperparameter search algorithms (Bayesian Optimization, Hyperband).
*   **Model Management:** Add capabilities for saving and loading trained models.
*   **Streaming Data:** Support for streaming data sources.
*   **Schema Validation:** Implement schema validation for datasets.
*   **Resource Management:** Add controls for managing agent resource usage.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

### Basic Tensor Operations

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

## License

MIT License
