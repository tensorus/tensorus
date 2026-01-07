---
license: mit
title: Core
sdk: docker
emoji: 🐠
colorFrom: blue
colorTo: yellow
short_description: Tensorus Core
---

# Tensorus: Basic Tensor Database

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Focus: Quality over Quantity** - A simple, reliable tensor database focused on fundamental operations done right.

**Tensorus** is a lightweight, specialized database for storing and managing tensor data with metadata. It provides essential functionality for ML/AI workflows: store tensors, retrieve them efficiently, and perform basic operations.

## 🎯 Core Philosophy

**Quality over Quantity** - Tensorus focuses on implementing fundamental tensor database operations perfectly:

- **📦 Tensor Storage** - Store and retrieve PyTorch tensors with rich metadata
- **🔍 Simple Queries** - Filter tensors by metadata attributes
- **⚡ Basic Operations** - Essential tensor operations (arithmetic, linear algebra, reshaping)
- **🔌 REST API** - Clean, documented API for all operations
- **📝 Clear Documentation** - Every feature is well-documented and tested

The core purpose of Tensorus is to **provide a reliable foundation** for tensor data management without unnecessary complexity.

## 🚀 Quick Start

### Installation

```bash
# Install from source
git clone https://github.com/tensorus/tensorus.git
cd tensorus
./setup.sh  # Installs all dependencies
```

### Basic Usage

```python
from tensorus import Tensorus
import torch

# Initialize Tensorus with minimal configuration
ts = Tensorus()

# 1. Create a dataset
ts.create_dataset("my_dataset")

# 2. Store tensors with metadata
tensor = ts.create_tensor(
    [[1, 2], [3, 4]], 
    name="matrix_a",
    dataset="my_dataset",
    metadata={"source": "experiment_1", "epoch": 0}
)

# 3. Retrieve tensors
tensors = ts.list_tensors("my_dataset")
print(f"Stored {len(tensors)} tensors")

# 4. Perform basic operations
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
result = ts.matmul(a, b)
print(f"Result:\n{result}")
```

### REST API

```bash
# Start the API server
uvicorn tensorus.api:app --reload --port 8000

# Access interactive documentation at http://localhost:8000/docs
```

## 🌟 Core Features

### 📦 Tensor Storage
- **Store & Retrieve** - Save PyTorch tensors with automatic serialization
- **Metadata Support** - Attach rich metadata (source, experiment, epoch, etc.)
- **Dataset Organization** - Group related tensors into named datasets
- **Persistence** - Optional disk or S3 storage for data durability

### 🔍 Simple Queries
- **Metadata Filtering** - Find tensors by metadata attributes
- **List Operations** - Browse all tensors in a dataset
- **Efficient Retrieval** - Fast lookup by tensor ID or name

### ⚡ Basic Tensor Operations
- **Arithmetic** - add, subtract, multiply, divide, power
- **Linear Algebra** - matmul, dot, transpose, svd, qr, eigenvalues
- **Reshaping** - reshape, flatten, squeeze, unsqueeze
- **Reductions** - sum, mean, min, max
- **40+ operations** - All essential tensor manipulations

### 🔌 REST API
- **FastAPI Backend** - Modern, async API with automatic documentation
- **Authentication** - API key-based security
- **OpenAPI Spec** - Interactive documentation at `/docs`
- **Simple Endpoints** - Create datasets, store/retrieve tensors, run operations

## 📋 Current Status

**Version:** 0.1.0  
**Status:** Alpha - Core functionality stable, focused on quality improvements

### What Works Well
- ✅ Tensor storage and retrieval
- ✅ Metadata management
- ✅ Basic tensor operations
- ✅ REST API
- ✅ File and S3 persistence

### What's Simplified
- ⚠️ Advanced features (agents, NQL, vector search) are available but considered experimental
- ⚠️ Focus is on core database operations, not complex workflows
- ⚠️ Performance optimizations ongoing

## 📁 Project Structure

```
tensorus/
├── tensorus/              # Core package
│   ├── __init__.py       # Package initialization
│   ├── sdk.py            # Unified SDK interface
│   ├── tensor_storage.py # Core storage engine
│   ├── tensor_ops.py     # Tensor operations library
│   ├── api.py            # REST API (FastAPI)
│   ├── config.py         # Configuration management
│   └── ...               # Additional modules
├── tests/                 # Test suite (150+ tests)
├── examples/              # Usage examples
├── docs/                  # Documentation
├── requirements.txt       # Dependencies
├── setup.sh              # Installation script
└── README.md             # This file
```

### Core Modules

- **`tensor_storage.py`** - Dataset management, tensor persistence, metadata handling
- **`tensor_ops.py`** - 40+ tensor operations (arithmetic, linear algebra, reshaping, etc.)
- **`sdk.py`** - High-level Python interface (Tensorus class)
- **`api.py`** - REST API endpoints and authentication
- **`config.py`** - Environment-based configuration

### Additional Features (Experimental)

The following modules provide advanced functionality but are considered experimental:
- `nql_agent.py` - Natural query language support
- `embedding_agent.py` - Text embeddings and vector search
- `vector_database.py` - Vector similarity search
- `agent_orchestrator.py` - Multi-agent workflows
- `*_agent.py` - Various specialized agents (RL, AutoML, Ingestion)

## 🌐 Live Demos & Integrations

### 🚀 Try Tensorus Online (No Installation Required)

Experience Tensorus directly in your browser via Huggingface Spaces:

*   **🔗 [Interactive API Documentation](https://tensorus-api.hf.space/docs)** - Full Swagger UI with live examples and real-time testing
*   **📖 [Alternative API Docs](https://tensorus-api.hf.space/redoc)** - Clean ReDoc interface with detailed schemas
*   **📊 [Web Dashboard Demo](https://tensorus-dashboard.hf.space)** - Complete Streamlit UI for data exploration and agent control

### 🤖 AI Agent Integration

**Model Context Protocol (MCP) Support** - Standardized integration for AI agents and LLMs:
*   **Repository:** [tensorus/mcp](https://github.com/tensorus/mcp) - Complete MCP server implementation
*   **Features:** Standardized protocol access to all Tensorus capabilities
*   **Use Cases:** LLM-driven tensor analysis, automated data workflows, intelligent agent interactions

## 🏗️ Simple Architecture

Tensorus follows a straightforward layered architecture:

```
┌─────────────────────────────────────┐
│   Client Applications               │
│   (Python SDK, REST API calls)      │
└────────────┬────────────────────────┘
             │
             v
┌─────────────────────────────────────┐
│   Tensorus SDK                      │
│   - create_tensor()                 │
│   - list_tensors()                  │
│   - Tensor operations (matmul, etc) │
└────────────┬────────────────────────┘
             │
             v
┌─────────────────────────────────────┐
│   TensorStorage                     │
│   - Dataset management              │
│   - Metadata indexing               │
│   - Persistence (disk/S3)           │
└────────────┬────────────────────────┘
             │
             v
┌─────────────────────────────────────┐
│   Storage Backends                  │
│   - File system                     │
│   - S3/Cloud storage                │
│   - PostgreSQL (metadata)           │
└─────────────────────────────────────┘
```

### Core Components

1. **SDK (`sdk.py`)** - High-level Python interface
   - Simplified API for common operations
   - Automatic type conversion
   - Error handling

2. **TensorStorage (`tensor_storage.py`)** - Data management engine
   - In-memory datasets with optional persistence
   - Metadata storage and indexing
   - Transaction support

3. **TensorOps (`tensor_ops.py`)** - Operation library
   - 40+ tensor operations
   - Type checking and validation
   - Memory-efficient implementations

4. **REST API (`api.py`)** - HTTP interface
   - FastAPI with OpenAPI documentation
   - API key authentication
   - JSON request/response

## 💻 Installation

### Requirements
- Python 3.10 or higher
- 4+ GB RAM
- Linux, macOS, or Windows with WSL2

### Development Installation

```bash
# Clone the repository
git clone https://github.com/tensorus/tensorus.git
cd tensorus

# Run setup script (installs dependencies)
./setup.sh

# Verify installation
python -c "from tensorus import Tensorus; print('Tensorus installed successfully!')"
```

### Docker Installation (Optional)

```bash
# Quick start with Docker
docker compose up --build

# Access API at http://localhost:7860/docs
```

## 📖 Basic Usage Examples

### Example 1: Store and Retrieve Tensors

```python
from tensorus import Tensorus
import torch

# Initialize
ts = Tensorus()

# Create a dataset
ts.create_dataset("experiments")

# Store a tensor
tensor = ts.create_tensor(
    [[1, 2, 3], [4, 5, 6]],
    name="sample_matrix",
    dataset="experiments",
    metadata={"experiment": "test_1", "step": 0}
)

# List all tensors
tensors = ts.list_tensors("experiments")
print(f"Stored {len(tensors)} tensors")

# Retrieve by metadata
# (Simple filtering supported through the storage API)
```

### Example 2: Tensor Operations

```python
from tensorus import Tensorus
import torch

ts = Tensorus()

# Create tensors
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# Basic operations
result = ts.matmul(a, b)
print(f"Matrix multiplication result:\n{result}")

transpose = ts.transpose(a)
print(f"Transpose:\n{transpose}")

mean_val = ts.mean(a)
print(f"Mean: {mean_val}")
```

### Example 3: REST API Usage

```bash
# Start the API server
uvicorn tensorus.api:app --port 8000

# Create a dataset
curl -X POST "http://localhost:8000/datasets/create" \
  -H "Content-Type: application/json" \
  -d '{"name": "my_dataset"}'

# List datasets
curl "http://localhost:8000/datasets"

# View interactive docs at http://localhost:8000/docs
```

## 🎯 Use Cases

Tensorus is designed for ML/AI workflows that need reliable tensor storage:

- **Experiment Tracking** - Store training data and results with metadata
- **Model Checkpointing** - Save and retrieve model weights
- **Data Processing** - Organize tensor datasets for preprocessing
- **Research** - Manage scientific computing data

*Note: Tensorus focuses on basic tensor database operations. For advanced features like distributed training, vector search, or AutoML, consider specialized tools.*

## 🧪 Testing

Run the test suite to verify your installation:

```bash
# Install dependencies (if not already done)
./setup.sh

# Run all tests
pytest

# Run specific test file
pytest tests/test_tensor_storage.py

# Run with verbose output
pytest -v
```

## 📚 API Reference

### Core API Endpoints

**Datasets:**
- `POST /datasets/create` - Create a new dataset
- `GET /datasets` - List all datasets
- `DELETE /datasets/{name}` - Delete a dataset

**Tensors:**
- `POST /datasets/{name}/ingest` - Store a tensor
- `GET /datasets/{name}/records` - List tensors (with pagination)
- `GET /datasets/{name}/tensors/{id}` - Get specific tensor
- `DELETE /datasets/{name}/tensors/{id}` - Delete tensor

**Operations:**
- See `/docs` endpoint for interactive API documentation

### Authentication

```bash
# Generate an API key
python generate_api_key.py

# Use in requests
curl -H "Authorization: Bearer tsr_your_key" http://localhost:8000/datasets
```

## 🔧 Configuration

Tensorus can be configured via environment variables:

```bash
# Storage
TENSORUS_STORAGE_BACKEND=in_memory  # or 'postgres'
TENSORUS_TENSOR_STORAGE_PATH=tensor_data  # or S3 URI

# Authentication
TENSORUS_AUTH_ENABLED=true
TENSORUS_API_KEYS=tsr_your_key_here

# Logging
TENSORUS_AUDIT_LOG_PATH=tensorus_audit.log
```

## 📖 Available Tensor Operations

Tensorus provides 40+ tensor operations organized by category:

**Arithmetic:** add, subtract, multiply, divide, power, log

**Linear Algebra:** matmul, dot, transpose, svd, qr, eigenvalues, determinant, inverse

**Reshaping:** reshape, flatten, squeeze, unsqueeze, permute

**Reductions:** sum, mean, min, max, variance

See `tensorus/tensor_ops.py` for the complete list and documentation.

## 🤝 Contributing

We welcome contributions! To get started:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🔗 Resources

- **Documentation:** See `docs/` directory
- **Examples:** See `examples/` directory
- **Issues:** [GitHub Issues](https://github.com/tensorus/tensorus/issues)
- **Discussions:** [GitHub Discussions](https://github.com/tensorus/tensorus/discussions)

---

*Tensorus - A simple, reliable tensor database focused on fundamentals*
