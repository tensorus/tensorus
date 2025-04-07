# Tensorus - Agentic Tensor Database/Data Lake

**Tensorus** is an innovative, proof-of-concept hybrid cloud/open-source agentic tensor database/data lake designed for modern AI and machine learning workflows. It stores all data uniformly as multi-dimensional tensors and integrates autonomous AI agents that can continuously ingest, optimize, and learn from the stored data.

**Current Status:** This implementation is a functional proof-of-concept based on the provided design document. Key features include in-memory tensor storage, basic agent implementations (ingestion, RL, NQL, AutoML), and a FastAPI interface. It serves as a foundation for further development.

## Key Features

* **Unified Tensor Storage:** All data (structured, unstructured, multi-modal) is intended to be stored uniformly as tensors (`tensor_storage.py`, currently in-memory).
* **Robust Tensor Operations:** A library for common tensor manipulations with shape checking (`tensor_ops.py`).
* **Agentic Intelligence:** Embedded autonomous agents:
    * **Data Ingestion Agent:** Monitors local directories and ingests recognized file types (CSV, images) as tensors (`ingestion_agent.py`).
    * **Reinforcement Learning Agent:** Implements DQN, stores experiences (state tensor IDs, actions, rewards) in TensorStorage, samples, and trains (`rl_agent.py`). Includes a `dummy_env.py`.
    * **NQL Query Agent:** Parses basic natural language queries (using regex) to retrieve data from TensorStorage (`nql_agent.py`).
    * **AutoML Agent:** Performs basic hyperparameter random search for a dummy model on synthetic data, storing results (`automl_agent.py`).
* **REST API:** A FastAPI application provides endpoints for dataset management, data ingestion (via JSON), and NQL querying (`api.py`).
* **Modularity:** Designed with distinct modules for easier extension and maintenance.

## Project Structure

tensorus/
├── api.py                     # FastAPI application providing the REST API
├── automl_agent.py            # AutoML agent for hyperparameter search
├── dummy_env.py               # Simple environment for the RL agent example
├── ingestion_agent.py         # Agent for automated data ingestion from files
├── nql_agent.py               # Agent for parsing basic Natural Query Language
├── requirements.txt           # Project dependencies
├── rl_agent.py                # Reinforcement Learning (DQN) agent
├── tensor_ops.py              # Library for robust tensor operations
├── tensor_storage.py          # Core tensor storage module (in-memory)
└── README.md                  # This file

## Module Descriptions

* **`tensor_storage.py`**: Implements the core `TensorStorage` class. Manages named datasets containing PyTorch tensors and associated metadata. Currently uses an **in-memory dictionary** for storage. Provides methods for creating datasets, inserting tensors, retrieving data, querying via functions, and sampling.
* **`tensor_ops.py`**: Defines a static `TensorOps` class providing various tensor operations (arithmetic, linear algebra, reductions, reshaping) with built-in type and shape checking for robustness.
* **`ingestion_agent.py`**: Implements the `DataIngestionAgent`. Monitors a specified directory in a background thread, uses configurable rules (defaulting to CSV/image processing) to convert files into tensors, and inserts them into `TensorStorage`.
* **`rl_agent.py`**: Implements the `RLAgent` using Deep Q-Network (DQN). It interacts with an environment (`DummyEnv`), stores experiences by linking state tensor IDs (stored separately) with action/reward/done metadata in `TensorStorage`, samples these experiences, and performs training steps.
* **`dummy_env.py`**: Defines a very simple `DummyEnv` class (1D state, discrete actions) used for demonstrating the `RLAgent`.
* **`nql_agent.py`**: Implements the `NQLAgent`. Parses simple, predefined natural language query patterns using regular expressions and translates them into queries against `TensorStorage`. **Does not use an LLM.**
* **`automl_agent.py`**: Implements the `AutoMLAgent`. Performs random search over a defined hyperparameter space for a dummy MLP model trained on synthetic data. Stores trial results (parameters and scores) in `TensorStorage`.
* **`api.py`**: The main entry point for interacting with Tensorus programmatically. It uses FastAPI to create REST endpoints for managing datasets, ingesting tensors (via JSON body containing shape, dtype, and data list), fetching data, and executing NQL queries. Includes helpers for tensor serialization/deserialization between lists and `torch.Tensor`.
* **`requirements.txt`**: Lists all the Python packages required to run the project.

## Setup and Installation

1.  **Prerequisites:**
    * Python 3.7+
    * `pip` (Python package installer)

2.  **Clone Repository (if applicable):**
    If this code were in a Git repository:
    ```bash
    git clone <repository-url>
    cd tensorus
    ```
    Otherwise, ensure all the `.py` files listed in the Project Structure are in the same directory.

3.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Execution Guide

### 1. Running the API Server

The primary way to interact with Tensorus is through the FastAPI server.

* **Start the Server:**
    ```bash
    python api.py
    ```
    Or using uvicorn directly (enables auto-reload):
    ```bash
    uvicorn api:app --reload --host 127.0.0.1 --port 8000
    ```
* **Access API Documentation:**
    Once the server is running, open your web browser and navigate to:
    * **Swagger UI:** `http://127.0.0.1:8000/docs`
    * **ReDoc:** `http://127.0.0.1:8000/redoc`
    This provides interactive documentation for all API endpoints.

* **Example API Interactions (using `curl`):**

    * **Create a dataset named "test_data":**
        ```bash
        curl -X POST "[http://127.0.0.1:8000/datasets/create](https://www.google.com/search?q=http://127.0.0.1:8000/datasets/create)" \
             -H "Content-Type: application/json" \
             -d '{"name": "test_data"}'
        ```

    * **Ingest a 2x2 float32 tensor into "test_data":**
        ```bash
        curl -X POST "[http://127.0.0.1:8000/datasets/test_data/ingest](https://www.google.com/search?q=http://127.0.0.1:8000/datasets/test_data/ingest)" \
             -H "Content-Type: application/json" \
             -d '{
                   "shape": [2, 2],
                   "dtype": "float32",
                   "data": [[1.1, 2.2], [3.3, 4.4]],
                   "metadata": {"source": "curl_example"}
                 }'
        ```

    * **Fetch all data from "test_data":**
        ```bash
        curl -X GET "[http://127.0.0.1:8000/datasets/test_data/fetch](https://www.google.com/search?q=http://127.0.0.1:8000/datasets/test_data/fetch)"
        ```

    * **Execute an NQL query:**
        ```bash
        curl -X POST "[http://127.0.0.1:8000/query](https://www.google.com/search?q=http://127.0.0.1:8000/query)" \
             -H "Content-Type: application/json" \
             -d '{"query": "get all data from test_data where source = curl_example"}'
        ```

### 2. Running Individual Module Examples

Most modules (`ingestion_agent.py`, `rl_agent.py`, `nql_agent.py`, `automl_agent.py`, `tensor_storage.py`, `tensor_ops.py`, `dummy_env.py`) contain an `if __name__ == "__main__":` block demonstrating their standalone functionality.

* **Run an Example:**
    Activate your virtual environment first (`source venv/bin/activate` or `venv\Scripts\activate`).
    ```bash
    python <module_name>.py
    # Example:
    python rl_agent.py
    ```

* **Specific Notes:**
    * `ingestion_agent.py`: Creates a temporary directory (`temp_ingestion_source`) and dummy files within it. It monitors this directory. You might need `Pillow` installed (`pip install Pillow`) for the image example.
    * `rl_agent.py`: Runs a training loop using `DummyEnv`. If `matplotlib` is installed (`pip install matplotlib`), it will attempt to plot the episode rewards at the end. Close the plot window to terminate the script.
    * Other scripts (`nql_agent.py`, `automl_agent.py`, etc.) will execute their internal examples and print output to the console.

## Limitations and Future Work

* **In-Memory Storage:** `TensorStorage` currently only stores data in memory. Data is lost when the process stops. Implementing persistent storage (e.g., HDF5, Parquet, database integration) is a crucial next step.
* **Basic Agents:** The agents (Ingestion, RL, NQL, AutoML) are basic implementations. They lack advanced features, robust error handling, and sophisticated algorithms (e.g., NQL uses simple regex, AutoML uses only random search).
* **Serialization:** Tensor serialization in the API uses simple nested lists, which might be inefficient for very large tensors.
* **Scalability & Performance:** The current implementation is single-process and not optimized for performance or large-scale data handling.
* **Security:** The API lacks authentication and authorization mechanisms.

Future work could involve addressing these limitations, implementing cloud integration, adding more sophisticated agent behaviors (e.g., using LLMs for NQL), supporting more data types and operations, and optimizing for performance and scalability.

## License

MIT License

