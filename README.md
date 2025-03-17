# Tensorus: Advanced Agentic Tensor Database

## Understanding Tensorus: A Developer's Guide

### What is Tensorus?
Tensorus is a specialized database designed for AI and machine learning applications. While traditional databases store simple data types (like numbers and text) and regular vector databases store flat arrays, Tensorus can store and process multi-dimensional arrays called tensors.

### Key Concepts for Beginners

#### 1. What are Tensors?
Think of tensors as containers for data that can have multiple dimensions:
- **0D Tensor (Scalar)**: A single number (e.g., `5`)
- **1D Tensor (Vector)**: A list of numbers (e.g., `[1, 2, 3]`)
- **2D Tensor (Matrix)**: A table of numbers:
  ```
  [
    [1, 2, 3],
    [4, 5, 6]
  ]
  ```
- **3D Tensor**: Think of it as a stack of 2D matrices
- **4D+ Tensor**: Even more dimensions!

Real-world examples:
- **Images**: 3D tensors (height × width × color channels)
- **Videos**: 4D tensors (frames × height × width × channels)
- **Language Data**: 2D or 3D tensors (batch × sequence × features)

#### 2. What Makes Tensorus Special?
- **Native Tensor Support**: Works directly with multi-dimensional data
- **AI-Ready**: Designed for machine learning workflows
- **Self-Optimizing**: Uses AI to improve its performance
- **Scalable**: Can handle data from small to massive sizes

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Advanced Features](#advanced-features)
  - [Dynamic Agentic Reconfiguration](#dynamic-agentic-reconfiguration)
  - [Blockchain-based Data Provenance](#blockchain-based-data-provenance)
  - [Tensor Query Language (TQL)](#tensor-query-language-tql)
  - [Distributed Storage & Processing](#distributed-storage--processing)
  - [Security & Governance](#security--governance)
  - [Differential Privacy](#differential-privacy)
- [Examples](#examples)
- [Configuration](#configuration)
- [Performance Benchmarking](#performance-benchmarking)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Features
- **Native Tensor Support**: Store and query multi-dimensional tensor data without flattening or loss of structure
- **Advanced Indexing**: Fast similarity search optimized for high-dimensional tensor data
- **GPU Acceleration**: Hardware-accelerated tensor operations for real-time processing
- **Seamless AI/ML Integration**: Direct compatibility with popular deep learning frameworks
- **Comprehensive API**: RESTful interface for easy integration with various applications
- **Scalable Architecture**: Built for distributed environments and high-throughput workloads

### Advanced Features
- **Agentic Reconfiguration**: Self-optimizing database that learns from access patterns using reinforcement learning
- **Blockchain Provenance**: Immutable audit trail of all tensor operations for data provenance
- **Tensor Query Language**: SQL-like language specifically designed for tensor operations
- **Distributed Processing**: Horizontal scaling with partitioning and federated learning capabilities 
- **Enterprise Security**: Authentication, authorization, and comprehensive audit logging
- **Differential Privacy**: Noise-based privacy protection mechanisms for sensitive tensor data

## System Architecture

Tensorus follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Integration Layer (tensor_integration.py)     │
└─────────────────────────────────────────────────────────────────┘
          ▲               ▲                  ▲              ▲
          │               │                  │              │
┌─────────┴─────┐ ┌──────┴───────┐ ┌────────┴───────┐ ┌────┴─────┐
│   Database    │ │  Advanced    │ │  Security &    │ │ API Layer│
│  Core Layer   │ │  Features    │ │  Governance    │ │ (app.py) │
└───────────────┘ └──────────────┘ └────────────────┘ └──────────┘
      ▲ ▲ ▲              ▲                 ▲
      │ │ │              │                 │
┌─────┘ │ └──────┐ ┌─────┴─────┐ ┌─────────┴──────────┐
│       │        │ │           │ │                     │
▼       ▼        ▼ ▼           ▼ ▼                     ▼
┌───────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  Storage  │ │ Indexing │ │Processing│ │  Agent   │ │Blockchain│
│  Layer    │ │  Layer   │ │  Layer   │ │  Layer   │ │  Layer   │
└───────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
   (HDF5)       (FAISS)     (PyTorch)      (RL)        (Chain)
```

### Core Components:
1. **Storage Layer** (`tensor_data.py`): HDF5-based tensor storage with compression
2. **Indexing Layer** (`tensor_indexer.py`): FAISS-powered similarity search
3. **Processing Layer** (`tensor_processor.py`): Tensor operations with PyTorch/TensorLy
4. **Database Layer** (`tensor_database.py`): Core integration of storage, indexing, and processing
5. **API Layer** (`app.py`): RESTful endpoints for database interaction

### Advanced Components:
1. **Agent Layer** (`tensor_agent.py`): RL-based configuration optimizer
2. **Blockchain Layer** (`tensor_blockchain.py`): Immutable operation logging
3. **Query Language** (`tensor_query_language.py`): TQL parser and executor
4. **Distributed System** (`tensor_distributed.py`): Multi-node capabilities
5. **Security System** (`tensor_security.py`): Auth, permissions, and privacy
6. **Integration Layer** (`tensor_integration.py`): Unified interface for advanced features

## Installation

### Prerequisites

- Python 3.8+
- pip package manager
- For GPU acceleration: CUDA 11.0+ and compatible GPU

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/tensorus.git
cd tensorus

# Create and activate a virtual environment (recommended)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

### Using Docker

```bash
# Build the Docker image
docker build -t tensorus .

# Run the container
docker run -p 5000:5000 -v $(pwd)/data:/app/data tensorus
```

### Using Kubernetes (Production)

Kubernetes configurations are provided in the `kubernetes/` directory for production deployment:

```bash
# Deploy to Kubernetes cluster
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

## Getting Started: A Step-by-Step Guide

### Before You Begin
Make sure you have:
1. **Python 3.8 or newer**
   ```bash
   # Check your Python version
   python --version
   # Should show 3.8.x or higher
   ```

2. **pip (Python package manager)**
   ```bash
   # Check pip version
   pip --version
   ```

3. **Git** for downloading the code
   ```bash
   # Check git version
   git --version
   ```

4. (Optional) **CUDA** for GPU support
   - Only needed if you have an NVIDIA GPU
   - Download from NVIDIA's website

## Your First Tensorus Project: A Tutorial

Let's create a simple project to understand how Tensorus works:

### 1. Setting Up Your First Database
```python
import numpy as np
from tensor_database import TensorDatabase

# Initialize the database
db = TensorDatabase(
    storage_path="my_first_db.h5",  # Where to store the data
    use_gpu=False                    # Start without GPU for simplicity
)

# Create a simple tensor (2D matrix)
data = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

# Add some helpful information about the tensor
metadata = {
    "name": "first_matrix",
    "description": "A 2x3 matrix example",
    "created_by": "tutorial"
}

# Save the tensor to the database
tensor_id = db.save(data, metadata)
print(f"Saved tensor with ID: {tensor_id}")
```

### 2. Basic Operations
```python
# Retrieve the tensor
retrieved_tensor, meta = db.get(tensor_id)
print("Retrieved tensor shape:", retrieved_tensor.shape)
print("Metadata:", meta)

# Search for similar tensors
similar_tensors = db.search_similar(
    query_tensor=retrieved_tensor,
    k=5  # Find 5 most similar tensors
)
```

## API Reference

### Core API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tensor` | POST | Store a tensor |
| `/tensor/<tensor_id>` | GET | Retrieve a tensor |
| `/tensor/<tensor_id>` | PUT | Update a tensor |
| `/tensor/<tensor_id>` | DELETE | Delete a tensor |
| `/tensors` | GET | List all tensors |
| `/search` | POST | Find similar tensors |
| `/process` | POST | Perform tensor operations |
| `/batch` | POST | Batch save multiple tensors |
| `/metrics` | GET | Get database performance metrics |
| `/health` | GET | Health check |

### Advanced API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tql` | POST | Execute TQL queries |
| `/provenance/<tensor_id>` | GET | Get tensor operation history |
| `/federated/model` | POST | Create federated learning model |
| `/federated/train` | POST | Start federated training |
| `/auth/login` | POST | Authenticate and get token |
| `/auth/user` | POST | Create a new user |

## Advanced Features

### Dynamic Agentic Reconfiguration Explained
Think of this as an AI assistant that watches how you use the database and automatically adjusts settings to make it faster:

```python
from tensor_agent import TensorAgent

# Create the AI agent
agent = TensorAgent(
    learning_rate=0.01,    # How quickly it learns from changes
    exploration_rate=0.2,  # How often it tries new settings
    reward_decay=0.9      # How much it values long-term improvements
)

# Start the agent
agent.start_monitoring(database)

# The agent will now:
# 1. Watch query patterns (what kind of operations you do most)
# 2. Monitor performance (how long operations take)
# 3. Adjust settings automatically (like changing index types)
# 4. Learn from the results (remember what works best)
```

### Understanding Blockchain in Tensorus

#### What is Blockchain Data Provenance?
Think of blockchain as a special notebook where every change to your data is recorded permanently:
- Each operation (create, update, delete) is written down
- Once written, entries can't be changed
- You can always see who did what and when
- Perfect for tracking data lineage in AI/ML projects

Here's how to use it:

```python
from tensor_blockchain import TensorBlockchain

# Set up the blockchain system
blockchain = TensorBlockchain(
    chain_file="data/tensor_chain.json",  # Where to store the history
    difficulty=2,                         # How hard to make verification
    auto_mine=True                        # Automatically record changes
)

# Example 1: Recording when you create a new tensor
blockchain.add_operation(
    operation_type="create",              # What was done
    tensor_id="tensor_123",              # Which tensor it was done to
    user_id="alice",                     # Who did it
    metadata={                           # Additional information
        "shape": [3, 4, 5],
        "source": "sensor_data",
        "purpose": "training_data"
    }
)

# Example 2: Recording a modification
blockchain.add_operation(
    operation_type="update",
    tensor_id="tensor_123",
    user_id="bob",
    metadata={
        "modification": "normalization",
        "reason": "prepare_for_training"
    }
)

# Example 3: Viewing the history of a tensor
history = blockchain.get_tensor_history("tensor_123")
for entry in history:
    print(f"Operation: {entry['type']}")
    print(f"Done by: {entry['user_id']}")
    print(f"When: {entry['timestamp']}")
    print(f"Details: {entry['metadata']}")
    print("---")
```

### Understanding TQL (Tensor Query Language)

#### What is TQL?
TQL is like SQL (if you're familiar with databases), but designed specifically for tensor operations. It helps you:
- Find tensors based on their properties
- Transform tensors in various ways
- Manage tensor metadata
- Perform similarity searches

Let's look at common TQL operations with examples:

```python
from tensor_query_language import TQLParser

# Initialize the TQL system
parser = TQLParser(database)

# Example 1: Finding tensors by their properties
result = parser.execute("""
    -- Find all image tensors larger than 224x224
    SELECT * FROM tensors 
    WHERE metadata.type = 'image'
    AND metadata.dimensions[0] >= 224
    AND metadata.dimensions[1] >= 224
""")

# Example 2: Finding recent tensors
result = parser.execute("""
    -- Get tensors created in the last hour
    SELECT * FROM tensors 
    WHERE metadata.created_at > NOW() - INTERVAL '1 hour'
    ORDER BY metadata.created_at DESC
""")

# Example 3: Complex tensor operations
result = parser.execute("""
    -- Decompose a large tensor into smaller factors
    TRANSFORM tensors 
    USING decompose_tucker 
    WHERE id = '12345' 
    WITH rank=[10, 10, 10]  -- Reduce each dimension to size 10
""")

# Example 4: Finding similar tensors
result = parser.execute("""
    -- Find 5 tensors most similar to tensor_id '12345'
    SEARCH tensors 
    USING tensor('12345')
    METRIC 'cosine'  -- Use cosine similarity
    LIMIT 5
""")
```

### Understanding Distributed Storage & Processing

#### What is Distributed Processing?
When your data gets too big for one computer, Tensorus can split the work across multiple machines. Think of it like having multiple computers working together as a team.

Here's a detailed example:

```python
from tensor_distributed import TensorDistributedNode, FederatedLearning

# Step 1: Set up the main computer (coordinator)
coordinator = TensorDistributedNode(
    role="coordinator",           # This is the boss node
    port=5050,                   # Port for communication
    data_dir="data/node1",       # Where to store data
    max_memory="8G"              # Use up to 8GB of RAM
)

# Step 2: Set up helper computers (workers)
worker1 = TensorDistributedNode(
    role="worker",
    port=5051,
    data_dir="data/worker1",
    coordinator_url="http://main-computer:5050",
    max_memory="4G"
)

worker2 = TensorDistributedNode(
    role="worker",
    port=5052,
    data_dir="data/worker2",
    coordinator_url="http://main-computer:5050",
    max_memory="4G"
)

# Step 3: Start all nodes
coordinator.start()
worker1.start()
worker2.start()

# Step 4: Save a large tensor (it will be automatically split)
big_tensor = np.random.rand(100000, 1000)  # A 100,000 x 1,000 matrix
tensor_id = coordinator.save_distributed(
    tensor=big_tensor,
    metadata={"type": "large_matrix"},
    replication_factor=2  # Save two copies for safety
)

# Step 5: Retrieve the tensor (automatically reassembled)
retrieved_tensor = coordinator.get_distributed(tensor_id)

# Example: Distributed Machine Learning
fed_learning = FederatedLearning(coordinator)

# Create a model to train
fed_learning.create_model(
    model_id="image_classifier",
    initial_weights={
        "layer1": np.random.rand(784, 128),  # For MNIST digits
        "layer2": np.random.rand(128, 10)    # 10 classes
    },
    config={
        "learning_rate": 0.01,
        "batch_size": 32,
        "optimizer": "adam"
    }
)

# Start training across all workers
fed_learning.start_training(
    model_id="image_classifier",
    num_rounds=10,          # Train for 10 rounds
    min_workers=2,          # Need at least 2 workers
    aggregation="fedavg"    # Use FedAvg algorithm
)

# Monitor training progress
while True:
    status = fed_learning.get_training_status("image_classifier")
    print(f"Round {status['current_round']}/{status['total_rounds']}")
    print(f"Active Workers: {len(status['active_workers'])}")
    print(f"Current Loss: {status['current_loss']}")
    
    if status['is_complete']:
        break
    time.sleep(10)  # Check every 10 seconds
```

### Security & Governance

Tensorus provides enterprise-grade security features:

```python
from tensor_security import TensorSecurity, secure_operation

# Initialize security system
security = TensorSecurity(
    config_path="config/security_config.json",
    token_expiry=3600,  # 1 hour
    enable_audit_log=True
)

# Authenticate a user
token = security.authenticate("admin", "admin_password")

# Authorize an operation
if security.authorize(token, "tensor", "write"):
    # Perform the operation
    database.save(tensor, metadata)

# Create a new user
security.create_user(
    admin_token=token,
    username="alice",
    password="secure_password",
    roles=["reader"]
)

# Decorator for securing operations
@secure_operation(resource="tensor", operation="read")
def get_tensor(self, tensor_id, token):
    return self.database.get(tensor_id)
```

Key security features:
- JWT-based authentication
- Role-based access control (RBAC)
- Comprehensive audit logging
- User and role management
- Token blacklisting and expiration

### Differential Privacy

Tensorus implements differential privacy to protect sensitive data:

```python
from tensor_security import TensorDifferentialPrivacy

# Initialize privacy system
privacy = TensorDifferentialPrivacy(
    epsilon=1.0,  # Privacy budget
    delta=1e-5,   # Probability of privacy violation
    mechanism="laplace"  # Privacy mechanism
)

# Apply privacy to a tensor
private_tensor = privacy.privatize_tensor(
    tensor=tensor,
    sensitivity=1.0  # Maximum influence of a single record
)

# Execute query with privacy
result = privacy.privatize_query(
    database_ref=database,
    query_func=database.search_similar,
    sensitivity=1.0,
    query_tensor=query,
    k=5
)

# Check privacy budget
budget_status = privacy.get_budget_status()
print(f"Remaining privacy budget: {budget_status['remaining_budget']}")
```

Supported privacy mechanisms:
- Laplace mechanism
- Gaussian mechanism
- Exponential mechanism
- Privacy budget tracking and management

## Examples

The `examples/` directory contains demonstration scripts that showcase Tensorus capabilities:

### Advanced Features Demo

The `examples/advanced_features_demo.py` script demonstrates all advanced features:

```bash
# Run the demo with default configuration
python examples/advanced_features_demo.py

# Run with custom configuration
python examples/advanced_features_demo.py --config config/my_config.json
```

The demo:
1. Creates a test database with sample tensors
2. Demonstrates agentic reconfiguration with workload simulation
3. Shows blockchain provenance tracking for tensor operations
4. Executes TQL queries on the sample data
5. Demonstrates distributed capabilities (if enabled)
6. Shows security features with authentication and authorization
7. Demonstrates differential privacy by comparing normal and private results

### Other Examples

- `examples/core_features_demo.py` - Basic tensor operations demo
- `examples/benchmark_demo.py` - Performance benchmarking tool
- `examples/distributed_demo.py` - Multi-node setup and operations
- `examples/tql_playground.py` - Interactive TQL query testing tool

### Practical Examples

Let's look at some real-world examples:

#### Example 1: Image Database
```python
# Creating an image database for deep learning
from tensor_database import TensorDatabase
import cv2
import numpy as np

# Initialize database
db = TensorDatabase(
    storage_path="image_database.h5",
    use_gpu=True  # Use GPU for faster processing
)

# Load and store an image
def add_image(path, category):
    # Read image
    img = cv2.imread(path)
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Add metadata
    metadata = {
        "type": "image",
        "category": category,
        "original_path": path,
        "shape": img.shape
    }
    # Save to database
    return db.save(img, metadata)

# Add multiple images
image_id1 = add_image("cat.jpg", "cat")
image_id2 = add_image("dog.jpg", "dog")

# Find similar images
similar_images = db.search_similar(
    query_tensor=db.get(image_id1)[0],
    k=5
)
```

#### Example 2: Machine Learning Pipeline
```python
# Creating a machine learning pipeline with Tensorus
import numpy as np
from tensor_database import TensorDatabase
from tensor_processor import TensorProcessor
import torch
from torch import nn

# Initialize database and processor
db = TensorDatabase(storage_path="ml_pipeline.h5")
processor = TensorProcessor(use_gpu=True)

# 1. Store training data
def store_training_batch(x_batch, y_batch, batch_id):
    metadata = {
        "type": "training_data",
        "batch_id": batch_id,
        "x_shape": x_batch.shape,
        "y_shape": y_batch.shape,
        "timestamp": datetime.now().isoformat()
    }
    # Store features and labels together
    combined = np.concatenate([x_batch, y_batch], axis=1)
    return db.save(combined, metadata)

# 2. Create a data loader that works with Tensorus
class TensorusDataLoader:
    def __init__(self, db, batch_size=32):
        self.db = db
        self.batch_size = batch_size
        # Get all training data IDs
        self.tensor_ids = db.search(
            "metadata.type = 'training_data'",
            sort_by="metadata.batch_id"
        )
    
    def __iter__(self):
        self.current = 0
        return self
    
    def __next__(self):
        if self.current >= len(self.tensor_ids):
            raise StopIteration
            
        # Get next batch
        tensor_id = self.tensor_ids[self.current]
        data, metadata = self.db.get(tensor_id)
        
        # Split back into features and labels
        x = data[:, :-1]  # All columns except last
        y = data[:, -1]   # Last column
        
        self.current += 1
        return torch.FloatTensor(x), torch.LongTensor(y)

# 3. Training loop with Tensorus
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

loader = TensorusDataLoader(db, batch_size=32)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for x_batch, y_batch in loader:
        # Forward pass
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store training metrics
        db.save(
            tensor=np.array([loss.item()]),
            metadata={
                "type": "training_metric",
                "metric": "loss",
                "epoch": epoch,
                "timestamp": datetime.now().isoformat()
            }
        )
```

#### Example 3: Time Series Analysis
```python
# Working with time series data in Tensorus
import numpy as np
from tensor_database import TensorDatabase
from datetime import datetime, timedelta

# Initialize database
db = TensorDatabase(storage_path="timeseries.h5")

# 1. Store time series data
def store_timeseries(data, start_time, sampling_rate):
    metadata = {
        "type": "timeseries",
        "start_time": start_time.isoformat(),
        "sampling_rate": sampling_rate,  # in seconds
        "n_samples": len(data),
        "end_time": (start_time + timedelta(seconds=sampling_rate * len(data))).isoformat()
    }
    return db.save(data, metadata)

# 2. Query time series data by time range
def get_timeseries_range(start_time, end_time):
    query = f"""
    SELECT * FROM tensors 
    WHERE metadata.type = 'timeseries'
    AND metadata.start_time >= '{start_time.isoformat()}'
    AND metadata.end_time <= '{end_time.isoformat()}'
    ORDER BY metadata.start_time ASC
    """
    results = db.query(query)
    
    # Combine all tensors in the time range
    combined_data = []
    for tensor_id in results:
        data, metadata = db.get(tensor_id)
        combined_data.append(data)
    
    return np.concatenate(combined_data)

# Example usage
# Generate some sample time series data
start_time = datetime(2024, 3, 1)
sampling_rate = 60  # One sample per minute

# Store 24 hours of data in 1-hour chunks
for hour in range(24):
    # Generate random data for this hour
    hourly_data = np.random.randn(60)  # 60 samples per hour
    chunk_start = start_time + timedelta(hours=hour)
    store_timeseries(hourly_data, chunk_start, sampling_rate)

# Query data for a specific time range
query_start = datetime(2024, 3, 1, 10, 0)  # 10 AM
query_end = datetime(2024, 3, 1, 14, 0)    # 2 PM
data = get_timeseries_range(query_start, query_end)
```

### Security Best Practices

Here's a detailed guide on implementing security in your Tensorus application:

```python
from tensor_security import TensorSecurity, TensorPrivacy, secure_operation
from datetime import datetime, timedelta

# 1. Initialize security with strong settings
security = TensorSecurity(
    config_path="security_config.json",
    token_expiry=3600,          # 1-hour tokens
    max_failed_attempts=5,      # Lock after 5 failed attempts
    password_min_length=12,     # Require strong passwords
    enable_audit_log=True,      # Log all operations
    enable_rate_limiting=True   # Prevent brute force attacks
)

# 2. Define user roles and permissions
security.create_role(
    name="analyst",
    permissions=[
        "read_tensor",
        "search_similar",
        "run_queries"
    ]
)

security.create_role(
    name="data_scientist",
    permissions=[
        "read_tensor",
        "write_tensor",
        "delete_tensor",
        "search_similar",
        "run_queries",
        "train_models"
    ]
)

security.create_role(
    name="admin",
    permissions="all"
)

# 3. Create users with appropriate roles
security.create_user(
    username="alice",
    password="secure_password_123",
    roles=["data_scientist"],
    metadata={
        "department": "Research",
        "email": "alice@company.com"
    }
)

# 4. Implement authentication in your API
@app.route("/login", methods=["POST"])
def login():
    username = request.json["username"]
    password = request.json["password"]
    
    try:
        token = security.authenticate(username, password)
        return jsonify({"token": token})
    except AuthenticationError as e:
        return jsonify({"error": str(e)}), 401

# 5. Secure your operations
@secure_operation(required_permission="write_tensor")
def save_tensor(tensor, metadata, token):
    # The decorator checks the token and permission
    return db.save(tensor, metadata)

# 6. Implement privacy protection
privacy = TensorPrivacy(
    epsilon=1.0,               # Privacy budget
    delta=1e-5,               # Privacy failure probability
    mechanism="gaussian",      # Type of noise to add
    budget_window="24h"       # Reset privacy budget daily
)

# 7. Example of privacy-protected operation
def get_average_with_privacy(tensor_ids, token):
    # First, check authorization
    if not security.authorize(token, "read_tensor"):
        raise PermissionError("Not authorized")
    
    # Then, compute average with privacy
    tensors = [db.get(tid)[0] for tid in tensor_ids]
    return privacy.compute_average(
        tensors,
        sensitivity=1.0,  # Maximum change one record can cause
        clipping_bound=5.0  # Maximum allowed value
    )

# 8. Audit logging
def log_operation(user_id, operation, status, details=None):
    security.log_audit_event({
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "operation": operation,
        "status": status,
        "details": details or {},
        "ip_address": request.remote_addr
    })

# Example usage
try:
    # Login
    token = security.authenticate("alice", "secure_password_123")
    
    # Perform operation with privacy
    result = get_average_with_privacy(
        tensor_ids=["tensor1", "tensor2"],
        token=token
    )
    
    # Log successful operation
    log_operation(
        user_id="alice",
        operation="compute_average",
        status="success",
        details={"n_tensors": 2}
    )
    
except Exception as e:
    # Log failed operation
    log_operation(
        user_id="alice",
        operation="compute_average",
        status="failed",
        details={"error": str(e)}
    )
    raise
```

### Error Handling and Debugging

Tensorus provides comprehensive error handling and debugging capabilities. Here's how to use them effectively:

```python
from tensor_database import TensorDatabase, TensorError
import logging
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tensorus.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('tensorus')

# Custom error handling
def safe_tensor_operation(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TensorError as e:
            logger.error(f"Tensor operation failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    return wrapper

@safe_tensor_operation
def process_tensor(tensor_id):
    db = TensorDatabase()
    tensor, metadata = db.get(tensor_id)
    # Process tensor...
    return result

# Debug utilities
def debug_tensor(tensor_id):
    """Print detailed debug information about a tensor"""
    db = TensorDatabase()
    tensor, metadata = db.get(tensor_id)
    
    print(f"=== Tensor Debug Info ===")
    print(f"ID: {tensor_id}")
    print(f"Shape: {tensor.shape}")
    print(f"dtype: {tensor.dtype}")
    print(f"Memory usage: {tensor.nbytes / 1024:.2f} KB")
    print(f"\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
```

### Performance Optimization

Here are key strategies for optimizing Tensorus performance:

```python
from tensor_database import TensorDatabase
from tensor_processor import TensorProcessor
import numpy as np
from functools import lru_cache

# 1. Cache frequently accessed tensors
@lru_cache(maxsize=1000)
def get_cached_tensor(tensor_id):
    db = TensorDatabase()
    return db.get(tensor_id)

# 2. Batch processing
def process_batch(tensor_ids, batch_size=32):
    db = TensorDatabase()
    processor = TensorProcessor()
    
    results = []
    for i in range(0, len(tensor_ids), batch_size):
        batch = tensor_ids[i:i + batch_size]
        tensors = [db.get(tid)[0] for tid in batch]
        processed = processor.process_batch(np.stack(tensors))
        results.extend(processed)
    
    return results

# 3. Memory-efficient iteration
class TensorIterator:
    def __init__(self, tensor_ids, batch_size=32):
        self.tensor_ids = tensor_ids
        self.batch_size = batch_size
        self.db = TensorDatabase()
    
    def __iter__(self):
        for i in range(0, len(self.tensor_ids), self.batch_size):
            batch = self.tensor_ids[i:i + self.batch_size]
            yield [self.db.get(tid)[0] for tid in batch]
```

### Deployment

Here are examples of deploying Tensorus in different scenarios:

```python
# server.py - Basic REST API
from flask import Flask, request, jsonify
from tensor_database import TensorDatabase
from tensor_security import TensorSecurity
import os

app = Flask(__name__)
db = TensorDatabase(
    storage_path=os.getenv("TENSOR_STORAGE_PATH", "tensors.h5")
)
security = TensorSecurity()

@app.route("/tensor/<tensor_id>", methods=["GET"])
def get_tensor(tensor_id):
    token = request.headers.get("Authorization")
    if not security.authorize(token, "read_tensor"):
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        tensor, metadata = db.get(tensor_id)
        return jsonify({
            "shape": tensor.shape,
            "dtype": str(tensor.dtype),
            "metadata": metadata
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

Dockerfile:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000

CMD ["python", "server.py"]
```

Kubernetes deployment:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tensorus
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tensorus
  template:
    metadata:
      labels:
        app: tensorus
    spec:
      containers:
      - name: tensorus
        image: tensorus:latest
        ports:
        - containerPort: 8000
        env:
        - name: TENSOR_STORAGE_PATH
          value: /data/tensors.h5
        volumeMounts:
        - name: tensor-storage
          mountPath: /data
      volumes:
      - name: tensor-storage
        persistentVolumeClaim:
          claimName: tensor-pvc
```

Production configuration:
```python
config = {
    "database": {
        "storage_path": "/data/tensors.h5",
        "cache_size": 10000,
        "compression": "lz4",
        "backup_interval": 3600
    },
    "security": {
        "token_expiry": 3600,
        "max_failed_attempts": 5,
        "password_min_length": 12,
        "enable_audit_log": True
    },
    "distributed": {
        "coordinator_host": "coordinator.tensorus.internal",
        "coordinator_port": 5000,
        "heartbeat_interval": 30,
        "replication_factor": 3
    }
}

# Save configuration
import json
with open("production_config.json", "w") as f:
    json.dump(config, f, indent=2)
```

These examples demonstrate:
1. Proper error handling and debugging with logging
2. Performance optimization through caching and batching
3. Deployment options (Docker, Kubernetes)
4. Production-ready configuration

Would you like me to provide more specific examples for any of these areas?

## Complete Getting Started Guide

This guide will walk you through setting up and running Tensorus step by step, excluding blockchain functionality for simplicity.

### Step 1: Environment Setup

```bash
# Create a new directory for your project
mkdir tensorus_project
cd tensorus_project

# Create and activate a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install required packages
pip install numpy torch pandas h5py flask faiss-cpu scikit-learn
```

### Step 2: Create Project Structure

```bash
# Create necessary directories
mkdir -p tensorus/core tensorus/advanced tensorus/api data config logs

# Create empty __init__.py files for Python packages
touch tensorus/__init__.py
touch tensorus/core/__init__.py
touch tensorus/advanced/__init__.py
touch tensorus/api/__init__.py
```

### Step 3: Core Components Setup

1. Create `tensorus/core/tensor_database.py`:
```python
import numpy as np
import h5py
from datetime import datetime

class TensorDatabase:
    def __init__(self, storage_path="data/tensors.h5", use_gpu=False):
        self.storage_path = storage_path
        self.use_gpu = use_gpu
        
    def save(self, tensor, metadata=None):
        with h5py.File(self.storage_path, 'a') as f:
            # Generate unique ID
            tensor_id = f"tensor_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Store tensor
            f.create_dataset(f"{tensor_id}/data", data=tensor)
            
            # Store metadata if provided
            if metadata:
                metadata_group = f.create_group(f"{tensor_id}/metadata")
                for key, value in metadata.items():
                    metadata_group.attrs[key] = str(value)
            
            return tensor_id
    
    def get(self, tensor_id):
        with h5py.File(self.storage_path, 'r') as f:
            tensor = f[f"{tensor_id}/data"][:]
            metadata = dict(f[f"{tensor_id}/metadata"].attrs) if "metadata" in f[tensor_id] else {}
            return tensor, metadata
```

2. Create `tensorus/core/tensor_processor.py`:
```python
import numpy as np
import torch

class TensorProcessor:
    def __init__(self, use_gpu=False):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    def process_batch(self, tensors):
        # Convert to PyTorch tensor
        if not isinstance(tensors, torch.Tensor):
            tensors = torch.from_numpy(tensors).to(self.device)
        
        # Example processing (normalize)
        processed = (tensors - tensors.mean()) / tensors.std()
        return processed.cpu().numpy()
```

### Step 4: Advanced Features Setup

1. Create `tensorus/advanced/tensor_security.py`:
```python
import jwt
import datetime
import hashlib

class TensorSecurity:
    def __init__(self, secret_key="your-secret-key"):
        self.secret_key = secret_key
        self.users = {}
        self.roles = {}
    
    def authenticate(self, username, password):
        if username in self.users and self._verify_password(password, self.users[username]["password"]):
            return self._generate_token(username)
        raise Exception("Authentication failed")
    
    def _generate_token(self, username):
        return jwt.encode(
            {
                "user": username,
                "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
            },
            self.secret_key,
            algorithm="HS256"
        )
    
    def _verify_password(self, password, stored_hash):
        return hashlib.sha256(password.encode()).hexdigest() == stored_hash
```

2. Create `tensorus/advanced/tensor_query_language.py`:
```python
class TQLParser:
    def __init__(self, database):
        self.db = database
    
    def execute(self, query):
        # Simple implementation for demonstration
        if query.startswith("SELECT"):
            return self._handle_select(query)
        raise NotImplementedError("Query type not supported")
    
    def _handle_select(self, query):
        # Basic implementation
        return []  # Return matching tensor IDs
```

### Step 5: API Setup

Create `tensorus/api/app.py`:
```python
from flask import Flask, request, jsonify
from tensorus.core.tensor_database import TensorDatabase
from tensorus.advanced.tensor_security import TensorSecurity

app = Flask(__name__)
db = TensorDatabase()
security = TensorSecurity()

@app.route("/tensor", methods=["POST"])
def save_tensor():
    data = request.get_json()
    tensor_id = db.save(data["tensor"], data.get("metadata"))
    return jsonify({"tensor_id": tensor_id})

@app.route("/tensor/<tensor_id>", methods=["GET"])
def get_tensor(tensor_id):
    tensor, metadata = db.get(tensor_id)
    return jsonify({
        "tensor": tensor.tolist(),
        "metadata": metadata
    })

if __name__ == "__main__":
    app.run(debug=True)
```

### Step 6: Configuration

Create `config/config.json`:
```json
{
    "database": {
        "storage_path": "data/tensors.h5",
        "use_gpu": false
    },
    "security": {
        "token_expiry": 3600,
        "enable_audit_log": true
    },
    "api": {
        "host": "localhost",
        "port": 5000
    }
}
```

### Step 7: Running the System

1. Start the API server:
```bash
# From project root
python -m tensorus.api.app
```

2. Test basic operations using Python:
```python
import numpy as np
from tensorus.core.tensor_database import TensorDatabase

# Initialize database
db = TensorDatabase()

# Create and save a tensor
tensor = np.random.randn(3, 3)
metadata = {"description": "Test tensor"}
tensor_id = db.save(tensor, metadata)

# Retrieve the tensor
retrieved_tensor, retrieved_metadata = db.get(tensor_id)
print(f"Retrieved tensor:\n{retrieved_tensor}")
print(f"Metadata: {retrieved_metadata}")
```

3. Test API endpoints:
```bash
# Save a tensor
curl -X POST http://localhost:5000/tensor \
  -H "Content-Type: application/json" \
  -d '{"tensor": [[1,2],[3,4]], "metadata": {"description": "Test"}}'

# Get a tensor
curl http://localhost:5000/tensor/tensor_20240301_123456_789
```

### Step 8: Running Examples

1. Basic tensor operations:
```python
from tensorus.core.tensor_database import TensorDatabase
from tensorus.core.tensor_processor import TensorProcessor

# Initialize components
db = TensorDatabase()
processor = TensorProcessor(use_gpu=False)

# Create sample data
data = np.random.randn(100, 784)  # 100 samples of 784 features
metadata = {"type": "training_data"}

# Save and process
tensor_id = db.save(data, metadata)
processed_data = processor.process_batch(data)
```

2. Using the query language:
```python
from tensorus.advanced.tensor_query_language import TQLParser

parser = TQLParser(db)
results = parser.execute("""
    SELECT * FROM tensors 
    WHERE metadata.type = 'training_data'
""")
```

### Common Issues and Solutions

1. If you get HDF5 errors:
   - Make sure the data directory exists
   - Check file permissions
   - Ensure the HDF5 file isn't opened by another process

2. If GPU acceleration isn't working:
   - Verify CUDA is installed correctly
   - Check PyTorch was installed with CUDA support
   - Confirm your GPU is CUDA-compatible

3. If the API server won't start:
   - Check if port 5000 is already in use
   - Verify all dependencies are installed
   - Check file permissions for logs directory

### Next Steps

After getting the basic system running, you can:
1. Implement more advanced TQL queries
2. Add user authentication to the API
3. Implement distributed processing
4. Add monitoring and logging
5. Optimize performance for your specific use case

Would you like me to provide more detailed examples for any of these components or explain any part in more detail?

## Project Structure and File Descriptions

```
tensorus/
├── core/                     # Core functionality
│   ├── __init__.py
│   ├── tensor_database.py    # Main database operations
│   ├── tensor_processor.py   # Tensor processing utilities
│   └── tensor_indexer.py     # Similarity search indexing
├── advanced/                 # Advanced features
│   ├── __init__.py
│   ├── tensor_security.py    # Security and authentication
│   └── tensor_query_language.py  # TQL implementation
├── api/                      # REST API
│   ├── __init__.py
│   └── app.py               # Flask API endpoints
├── examples/                 # Example scripts
│   ├── basic_demo.py        # Basic usage examples
│   └── advanced_demo.py     # Advanced features demo
├── data/                     # Data storage
│   └── tensors.h5           # HDF5 database file
├── config/                   # Configuration files
│   └── config.json          # Main configuration
├── logs/                     # Log files
│   └── tensorus.log         # Application logs
├── requirements.txt          # Project dependencies
└── README.md                # This file
```

### File Descriptions

#### Core Components

1. `tensor_database.py`:
   - Main database operations
   - Handles tensor storage and retrieval
   - Manages metadata
   ```python
   # Key classes:
   class TensorDatabase:
       def save(self, tensor, metadata)  # Save tensor with metadata
       def get(self, tensor_id)          # Retrieve tensor by ID
       def delete(self, tensor_id)       # Delete tensor
       def search_similar(self, query)   # Find similar tensors
   ```

2. `tensor_processor.py`:
   - Tensor processing utilities
   - GPU acceleration support
   - Batch processing capabilities
   ```python
   # Key classes:
   class TensorProcessor:
       def process_batch(self, tensors)   # Process batch of tensors
       def normalize(self, tensor)        # Normalize tensor
       def transform(self, tensor, op)    # Apply transformation
   ```

3. `tensor_indexer.py`:
   - Similarity search indexing
   - FAISS integration
   - Index management
   ```python
   # Key classes:
   class TensorIndexer:
       def build_index(self, tensors)     # Build search index
       def search(self, query, k)         # Search for similar tensors
       def update_index(self, tensor)     # Update index
   ```

#### Advanced Components

1. `tensor_security.py`:
   - Authentication and authorization
   - Role-based access control
   - Audit logging
   ```python
   # Key classes:
   class TensorSecurity:
       def authenticate(self, username, password)  # User authentication
       def authorize(self, token, operation)      # Operation authorization
       def create_user(self, username, roles)     # User management
   ```

2. `tensor_query_language.py`:
   - TQL parser and executor
   - Query optimization
   - Result formatting
   ```python
   # Key classes:
   class TQLParser:
       def execute(self, query)           # Execute TQL query
       def parse(self, query)             # Parse query string
       def optimize(self, query_plan)     # Optimize query execution
   ```

#### API Components

1. `app.py`:
   - REST API endpoints
   - Request handling
   - Response formatting
   ```python
   # Key endpoints:
   @app.route("/tensor", methods=["POST"])        # Save tensor
   @app.route("/tensor/<tensor_id>", methods=["GET"])  # Get tensor
   @app.route("/search", methods=["POST"])        # Search tensors
   ```

## Running the System

### Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
numpy>=1.21.0
torch>=1.9.0
h5py>=3.3.0
flask>=2.0.0
faiss-cpu>=1.7.0  # Use faiss-gpu for GPU support
scikit-learn>=0.24.0
pandas>=1.3.0
pyjwt>=2.1.0
python-dotenv>=0.19.0
```

### Step 2: Create Required Directories

```bash
# Create project structure
mkdir -p tensorus/core tensorus/advanced tensorus/api tensorus/examples data config logs
touch tensorus/core/__init__.py tensorus/advanced/__init__.py tensorus/api/__init__.py
```

### Step 3: Configure the System

Create `config/config.json`:
```json
{
    "database": {
        "storage_path": "data/tensors.h5",
        "use_gpu": false,
        "index_type": "l2",
        "cache_size": 1000
    },
    "security": {
        "secret_key": "your-secret-key",
        "token_expiry": 3600,
        "enable_audit_log": true
    },
    "api": {
        "host": "localhost",
        "port": 5000,
        "debug": true
    }
}
```

### Step 4: Start the System

1. Start the API server:
```bash
# From project root
python -m tensorus.api.app
```

2. Run the basic demo:
```bash
# From project root
python -m tensorus.examples.basic_demo
```

### Step 5: Run Example Code

1. Basic tensor operations:
```bash
# From project root
python examples/basic_demo.py
```

This will run:
```python
from tensorus.core.tensor_database import TensorDatabase
import numpy as np

# Initialize database
db = TensorDatabase()

# Create and save tensor
tensor = np.random.randn(3, 3)
metadata = {"description": "Test tensor"}
tensor_id = db.save(tensor, metadata)
print(f"Saved tensor with ID: {tensor_id}")

# Retrieve tensor
retrieved_tensor, metadata = db.get(tensor_id)
print(f"Retrieved tensor:\n{retrieved_tensor}")
```

2. Advanced features demo:
```bash
# From project root
python examples/advanced_demo.py
```

This will demonstrate:
- TQL queries
- Security features
- Batch processing
- Performance optimization

### Step 6: Test API Endpoints

1. Save a tensor:
```bash
curl -X POST http://localhost:5000/tensor \
  -H "Content-Type: application/json" \
  -d '{
    "tensor": [[1,2],[3,4]],
    "metadata": {"description": "Test tensor"}
  }'
```

2. Retrieve a tensor:
```bash
curl http://localhost:5000/tensor/tensor_20240301_123456_789
```

3. Search similar tensors:
```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_tensor": [[1,2],[3,4]],
    "k": 5
  }'
```

### Common Operations

1. Start the server in development mode:
```bash
FLASK_ENV=development FLASK_APP=tensorus.api.app flask run
```

2. Run with GPU support:
```bash
TENSOR_USE_GPU=1 python -m tensorus.api.app
```

3. Run with custom configuration:
```bash
TENSOR_CONFIG_PATH=config/custom_config.json python -m tensorus.api.app
```

### Monitoring and Logs

1. View application logs:
```bash
tail -f logs/tensorus.log
```

2. Monitor API endpoints:
```bash
curl http://localhost:5000/metrics
```

### Troubleshooting

1. If you see "Address already in use":
```bash
# Find and kill the process using port 5000
lsof -i :5000
kill -9 <PID>
```

2. If database operations fail:
```bash
# Check data directory permissions
chmod 755 data
chmod 644 data/tensors.h5
```

3. If GPU acceleration isn't working:
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

Would you like me to provide more detailed examples or explain any specific part in more detail? 