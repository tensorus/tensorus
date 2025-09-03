# Tensorus User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Core Concepts](#core-concepts)
3. [Working with Tensors](#working-with-tensors)
4. [Vector Database](#vector-database)
5. [API Usage](#api-usage)
6. [Best Practices](#best-practices)

## Getting Started

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/tensorus.git
cd tensorus

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
```

### Quick Start
```python
from tensorus import Tensorus
import numpy as np

# Initialize Tensorus
ts = Tensorus()

# Create a sample tensor
data = np.random.rand(3, 3)
tensor = ts.create_tensor(data, name="sample_tensor")

# Perform operations
result = tensor.transpose()
print(result)
```

## Core Concepts

### Tensors
- Multi-dimensional arrays with rich metadata
- Support for various data types (float32, int64, etc.)
- Automatic shape inference and validation

### Vector Database
- High-performance similarity search
- Multi-tenant support
- Real-time updates with freshness layers

## Working with Tensors

### Creating Tensors
```python
# From NumPy array
import numpy as np
data = np.array([[1, 2], [3, 4]])
tensor = ts.create_tensor(data, name="matrix_2x2")

# From Python list
list_data = [[1, 2], [3, 4]]
tensor = ts.create_tensor(list_data, dtype="float32")
```

### Tensor Operations
```python
# Basic arithmetic
result = tensor1 + tensor2
result = tensor1 * 2

# Matrix operations
product = ts.matmul(matrix1, matrix2)
inverse = ts.inverse(matrix)

# Reductions
sum_result = ts.sum(tensor, axis=0)
mean_result = ts.mean(tensor)
```

## Vector Database

### Creating and Querying Vectors
```python
# Create vector index
index = ts.create_index("my_index", dimensions=384)

# Add vectors
vectors = np.random.rand(100, 384).astype(np.float32)
ids = [f"vec_{i}" for i in range(100)]
index.add_vectors(ids, vectors)

# Query similar vectors
query_vector = np.random.rand(384).astype(np.float32)
results = index.search(query_vector, k=5)
```

## API Usage

### Interactive Documentation

Tensorus provides interactive API documentation that's always in sync with your code:

1. **Swagger UI** (`/docs`):
   - Interactive API exploration
   - Try-it-now functionality
   - Request/response schemas
   - Authentication testing

2. **ReDoc** (`/redoc`):
   - Clean, responsive documentation
   - Better for reading and reference
   - Shows all endpoints at a glance

To access the documentation:
1. Start the development server:
   ```bash
   python -m uvicorn tensorus.api:app --reload
   ```
2. Visit http://localhost:8000/docs or http://localhost:8000/redoc

### Authentication
```python
import requests

url = "http://localhost:8000/tensors"
headers = {
    "X-API-KEY": "your_api_key",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)
```

### Common API Endpoints
For the complete and up-to-date list of endpoints, please refer to the interactive documentation at `/docs` or `/redoc`.

Example endpoints:
- `GET /tensors` - List all tensors
- `POST /tensors` - Create a new tensor
- `GET /tensors/{id}` - Get tensor by ID
- `POST /search` - Semantic search
- `GET /indices` - List vector indices

## Best Practices

### Performance Tips
- Use batch operations for multiple tensors
- Pre-allocate memory when possible
- Use appropriate data types to save memory

### Error Handling
```python
try:
    result = tensor1 / tensor2
except ZeroDivisionError:
    print("Cannot divide by zero")
except Exception as e:
    print(f"An error occurred: {str(e)}")
```

### Resource Management
```python
# Use context managers for resources
with ts.get_tensor("tensor_id") as tensor:
    # Work with tensor
    pass  # Automatically released when done
```
