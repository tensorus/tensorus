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


### Embedding Generation

Tensorus supports multiple embedding providers for generating high-quality vector representations of text:

*   **Sentence Transformers**: Local models including all-MiniLM-L6-v2, all-mpnet-base-v2, and specialized models
*   **OpenAI**: Cloud-based models like text-embedding-3-small and text-embedding-3-large
*   **Extensible Architecture**: Easy integration of additional embedding providers

### Vector Indexing

Advanced vector indexing capabilities for efficient similarity search:

*   **Geometric Partitioning**: Automatic distribution of vectors across partitions using k-means clustering
*   **Freshness Layers**: Real-time updates without requiring full index rebuilds
*   **FAISS Integration**: High-performance similarity search with multiple distance metrics
*   **Multi-tenancy**: Namespace and tenant isolation for secure multi-user deployments

### Hybrid Search

Unique hybrid search capabilities that combine semantic similarity with computational tensor properties:

*   **Semantic Scoring**: Traditional vector similarity search based on text embeddings
*   **Computational Scoring**: Mathematical property evaluation including shape compatibility, sparsity, rank analysis
*   **Operation Compatibility**: Scoring tensors based on suitability for specific mathematical operations
*   **Combined Ranking**: Weighted combination of semantic and computational relevance scores

### Tensor Workflows

Execute complex mathematical workflows with full computational lineage tracking:

*   **Workflow Execution**: Chain multiple tensor operations with intermediate result storage
*   **Lineage Tracking**: Complete provenance tracking of tensor transformations
*   **Scientific Reproducibility**: Full audit trail of computational steps for research applications
*   **Intermediate Storage**: Optional preservation of intermediate results for analysis

## Completed Features

The current codebase implements all of the items listed in
[Key Features](#key-features). Tensorus already provides efficient tensor
storage with optional file persistence, a natural query language, a flexible
agent framework, a RESTful API, a Streamlit UI, robust tensor operations, and
advanced vector database capabilities. The modular architecture makes future
extensions straightforward.

## Future Implementation

*   **Enhanced NQL:** Integrate a local or remote LLM for more robust natural language understanding.
*   **Advanced Agents:** Develop more sophisticated agents for specific tasks (e.g., anomaly detection, forecasting).
*   **Persistent Storage Backend:** Replace/augment current file-based persistence with more robust database or cloud storage solutions (e.g., PostgreSQL, S3, MinIO).
*   **Advanced Vector Indexing:** Implement HNSW and IVF-PQ algorithms for even more efficient similarity search.
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
*   **Resource Management:** Add tools and controls for monitoring and managing the resource consumption (CPU, memory) of agents.
*   **Improved UI/UX:** Continuously refine the Streamlit UI for better usability and richer visualizations.
*   **Comprehensive Testing:** Expand unit, integration, and end-to-end tests.
*   **Multi-modal Embeddings:** Support for image, audio, and video embeddings alongside text.
*   **Distributed Architecture:** Multi-node deployments for large-scale vector search workloads.

## 🤝 Community & Contributing

### 💬 Get Help & Support

**Community Resources:**
- **📚 [Documentation Hub](docs/index.md)** - Comprehensive guides and tutorials
- **💬 [GitHub Discussions](https://github.com/tensorus/tensorus/discussions)** - Ask questions and share ideas
- **🐛 [Issue Tracker](https://github.com/tensorus/tensorus/issues)** - Bug reports and feature requests
- **🏷️ [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorus)** - Technical Q&A with the community

**Enterprise Support:**
- **📧 Technical Support**: support@tensorus.com
- **📧 Sales & Partnerships**: sales@tensorus.com  
- **📧 Security Issues**: security@tensorus.com

### 🚀 Contributing to Tensorus

We welcome contributions from the community! Here's how to get involved:

#### 🐛 Report Issues
- Use our [issue templates](https://github.com/tensorus/tensorus/issues/new/choose) for bug reports
- Include system information, reproduction steps, and expected behavior
- Search existing issues before creating new ones

#### 🔧 Code Contributions
1. **Fork** the repository and create a feature branch
2. **Develop** with proper tests and documentation
3. **Test** your changes locally using `pytest`
4. **Submit** a pull request with clear description and examples

#### 📖 Documentation Improvements
- Fix typos, improve clarity, and add examples
- Translate documentation to other languages  
- Create tutorials and use case guides
- Update API documentation and code comments

#### 💡 Feature Requests & Ideas
- Propose new features via [GitHub Discussions](https://github.com/tensorus/tensorus/discussions)
- Provide detailed use cases and implementation suggestions
- Participate in design discussions and RFC processes

**Development Resources:**
- **📋 [Contributing Guide](CONTRIBUTING.md)** - Detailed contribution guidelines
- **📜 [Code of Conduct](CODE_OF_CONDUCT.md)** - Community standards and expectations
- **🏗️ [Development Setup](docs/getting_started.md#development-installation)** - Local development environment

## 📄 License & Legal

**MIT License** - See [LICENSE](LICENSE) file for complete terms.

```
Copyright (c) 2024 Tensorus Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

**Third-Party Licenses:** This project includes dependencies with their own licenses. See `requirements.txt` and individual package documentation for details.

---

<div align="center">

### 🌟 Ready to Transform Your Tensor Workflows?

[![Get Started](https://img.shields.io/badge/📚_Get_Started-blue?style=for-the-badge&logo=rocket)](docs/getting_started.md)
[![Live Demo](https://img.shields.io/badge/🚀_Try_Demo-green?style=for-the-badge&logo=play)](https://tensorus-dashboard.hf.space)
[![API Docs](https://img.shields.io/badge/📖_API_Reference-orange?style=for-the-badge&logo=swagger)](docs/api_reference.md)
[![Enterprise](https://img.shields.io/badge/🏢_Enterprise-purple?style=for-the-badge&logo=building)](mailto:sales@tensorus.com)

### ⭐ **Star us on GitHub** | **🔄 Share with your team** | **📢 Follow for updates**

*Tensorus - Empowering Intelligent Tensor Data Management*

</div>