# Tensorus: Agentic Tensor Database/Data Lake

Tensorus is an experimental platform for managing and querying tensor data, designed with agentic capabilities in mind. It provides a flexible way to store, retrieve, and manipulate tensors, and it's built to integrate with intelligent agents for tasks like data ingestion, reinforcement learning, and AutoML.

## Key Features

*   **Tensor Storage:** Efficiently store and retrieve PyTorch tensors with associated metadata.
*   **Natural Query Language (NQL):** Query your tensor data using a simple, natural language-like syntax.
*   **Agent Framework:** A foundation for building and integrating intelligent agents that interact with your data.
    *   **Data Ingestion Agent:** Automatically monitors a directory for new files and ingests them as tensors.
    *   **RL Agent:** A Deep Q-Network (DQN) agent that can learn from experiences stored in TensorStorage.
    *   **AutoML Agent:** Performs hyperparameter optimization for a dummy model using random search.
*   **API-Driven:** A FastAPI backend provides a RESTful API for interacting with Tensorus.
*   **Streamlit UI:** A user-friendly Streamlit frontend for exploring data and controlling agents.
*   **Tensor Operations:** A library of robust tensor operations for common manipulations.
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
*   `tensor_utils.py`: Utility functions for tensor manipulation and conversion.
*   `ui_utils.py`: Utility functions for the Streamlit UI.
*   `requirements.txt`: Lists the project's dependencies.

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
    git clone <your_repository_url>
    cd Tensorus_MVP
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

## License

[Your License] (e.g., MIT License)