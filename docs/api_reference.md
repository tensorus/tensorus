# Tensorus API Reference

## Interactive API Documentation

For the most up-to-date and interactive API documentation, please use:

- **Swagger UI**: [/docs](http://localhost:8000/docs) - Interactive API explorer with "Try it out" functionality
- **ReDoc**: [/redoc](http://localhost:8000/redoc) - Clean, responsive API documentation

## Overview

The Tensorus API provides comprehensive REST endpoints for tensor storage, operations, and analytics. All endpoints follow RESTful conventions and return JSON responses with consistent error handling.

**Base URL**: `http://localhost:8000` (Development) | `https://api.tensorus.com/v1` (Production)

**Authentication**: API Key required for all endpoints (header: `X-API-KEY`)

**Rate Limits**: 1000 requests/minute (Enterprise), 100 requests/minute (Developer)

## Quick Start

1. Start the development server:
   ```bash
   python -m uvicorn tensorus.api:app --reload
   ```

2. Access the interactive documentation at http://localhost:8000/docs

3. Set your API key in the "Authorize" dialog

## Core Concepts

### Tensors
- **Tensor ID**: Unique identifier for stored tensors
- **Dataset**: Named collection of related tensors
- **Metadata**: Flexible key-value attributes associated with tensors

### Operations
- **Synchronous**: Real-time operations returning immediate results
- **Asynchronous**: Long-running operations with job tracking
- **History**: Complete audit trail of all operations

## Tensor Storage Endpoints

### Create Dataset

Create a new dataset for organizing tensors.

```http
POST /datasets/{dataset_name}
```

**Parameters:**
- `dataset_name` (path): Name of the dataset to create

**Request Body:**
```json
{
  "description": "Description of the dataset",
  "metadata": {
    "project": "ml_model_v1",
    "owner": "data_team"
  },
  "compression_preset": "balanced"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Dataset created successfully",
  "dataset_name": "image_vectors",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Store Tensor

Store a tensor in a dataset with optional metadata.

```http
POST /datasets/{dataset_name}/tensors
```

**Request Body:**
```json
{
  "tensor_data": [[1, 2, 3], [4, 5, 6]],
  "metadata": {
    "source": "training_data",
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "compression": {
    "algorithm": "gzip",
    "level": 6
  }
}
```

**Response:**
```json
{
  "success": true,
  "tensor_id": "uuid-4-tensor-id",
  "shape": [2, 3],
  "dtype": "float32",
  "compressed": true,
  "storage_size_bytes": 1024
}
```

### Retrieve Tensor

Get a tensor by its ID with full metadata.

```http
GET /datasets/{dataset_name}/tensors/{tensor_id}
```

**Query Parameters:**
- `decompress` (boolean): Whether to decompress tensor data (default: true)
- `include_metadata` (boolean): Include tensor metadata (default: true)

**Response:**
```json
{
  "success": true,
  "tensor": {
    "tensor_id": "uuid-4-tensor-id",
    "shape": [2, 3],
    "dtype": "float32",
    "data": [[1, 2, 3], [4, 5, 6]],
    "metadata": {
      "source": "training_data",
      "created_at": "2024-01-15T10:30:00Z"
    },
    "compression_info": {
      "algorithm": "gzip",
      "compression_ratio": 1.8,
      "original_size": 1843,
      "compressed_size": 1024
    }
  }
}
```

### Query Tensors (NQL)

Use Natural Language Query to find tensors matching criteria.

```http
POST /query/nql
```

**Request Body:**
```json
{
  "query": "find image tensors from 'training_set' where metadata.accuracy > 0.95",
  "limit": 100,
  "include_data": false
}
```

**Response:**
```json
{
  "success": true,
  "message": "Query executed successfully",
  "count": 42,
  "results": [
    {
      "record_id": "uuid-tensor-1",
      "shape": [3, 224, 224],
      "dtype": "float32",
      "metadata": {
        "accuracy": 0.97,
        "model": "resnet50"
      }
    }
  ],
  "execution_time_ms": 150
}
```

## Tensor Operations Endpoints

### Execute Operation

Perform tensor operations with automatic result storage.

```http
POST /operations/execute
```

**Request Body:**
```json
{
  "operation": "matrix_multiply",
  "inputs": {
    "tensor_a": "uuid-tensor-1",
    "tensor_b": "uuid-tensor-2"
  },
  "parameters": {
    "transpose_a": false,
    "transpose_b": true
  },
  "store_result": true,
  "result_dataset": "computed_results"
}
```

**Response:**
```json
{
  "success": true,
  "operation_id": "uuid-operation-id",
  "result_tensor_id": "uuid-result-tensor",
  "execution_time_ms": 45,
  "result_shape": [128, 256],
  "operation_metadata": {
    "operation_type": "matrix_multiply",
    "inputs_count": 2,
    "device": "cuda:0"
  }
}
```

### Batch Operations

Execute multiple operations efficiently in a single request.

```http
POST /operations/batch
```

**Request Body:**
```json
{
  "operations": [
    {
      "operation": "add",
      "inputs": {"a": "tensor-1", "b": "tensor-2"},
      "name": "step_1"
    },
    {
      "operation": "multiply", 
      "inputs": {"a": "@step_1", "scalar": 2.0},
      "name": "step_2"
    }
  ],
  "store_intermediates": false
}
```

### Asynchronous Operations

Start long-running operations with job tracking.

```http
POST /operations/async
```

**Request Body:**
```json
{
  "operation": "tensor_decomposition",
  "inputs": {
    "tensor": "uuid-large-tensor"
  },
  "parameters": {
    "method": "svd",
    "rank": 50
  },
  "callback_url": "https://your-app.com/webhooks/tensorus"
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "uuid-job-id",
  "status": "queued",
  "estimated_duration_seconds": 300,
  "status_url": "/operations/jobs/uuid-job-id/status"
}
```

### Check Job Status

Monitor asynchronous operation progress.

```http
GET /operations/jobs/{job_id}/status
```

**Response:**
```json
{
  "job_id": "uuid-job-id",
  "status": "running",
  "progress_percent": 65,
  "started_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T10:35:00Z",
  "current_stage": "computing_decomposition",
  "result_preview": null
}
```

## Operation Discovery Endpoints

### List Available Operations

Get all available tensor operations with metadata.

```http
GET /operations/types
```

**Response:**
```json
{
  "success": true,
  "operations": [
    {
      "name": "matrix_multiply",
      "category": "linear_algebra",
      "description": "Matrix multiplication with broadcasting support",
      "parameters": [
        {
          "name": "transpose_a",
          "type": "boolean",
          "default": false,
          "description": "Transpose first input matrix"
        }
      ],
      "input_constraints": {
        "min_inputs": 2,
        "max_inputs": 2,
        "dimension_requirements": "compatible for matrix multiplication"
      },
      "performance_complexity": "O(n³)"
    }
  ]
}
```

### Operation Documentation

Get detailed documentation for a specific operation.

```http
GET /operations/{operation_name}/docs
```

**Response:**
```json
{
  "operation": "matrix_multiply",
  "description": "Performs matrix multiplication between two tensors",
  "examples": [
    {
      "title": "Basic Matrix Multiplication",
      "input": {
        "tensor_a": {"shape": [3, 4], "data": "..."},
        "tensor_b": {"shape": [4, 5], "data": "..."}
      },
      "output": {"shape": [3, 5], "description": "Result matrix"}
    }
  ],
  "performance_notes": "GPU acceleration available for tensors > 1024 elements"
}
```

## Operation History & Lineage Endpoints

### Get Operation History

Retrieve historical operations with filtering.

```http
GET /operations/recent
```

**Query Parameters:**
- `limit` (int): Maximum operations to return (default: 100)
- `operation_type` (string): Filter by operation type
- `status` (string): Filter by status (success, failed, running)
- `start_time` (ISO datetime): Filter operations after this time
- `end_time` (ISO datetime): Filter operations before this time

**Response:**
```json
{
  "success": true,
  "count": 25,
  "operations": [
    {
      "operation_id": "uuid-op-1",
      "operation_type": "matrix_multiply",
      "status": "completed",
      "started_at": "2024-01-15T10:30:00Z",
      "duration_ms": 150,
      "inputs": [
        {
          "tensor_id": "uuid-tensor-1",
          "shape": [100, 200],
          "parameter_name": "tensor_a"
        }
      ],
      "outputs": [
        {
          "tensor_id": "uuid-result-1",
          "shape": [100, 300]
        }
      ]
    }
  ]
}
```

### Get Tensor Lineage

Track computational lineage for a specific tensor.

```http
GET /lineage/tensor/{tensor_id}
```

**Query Parameters:**
- `include_operations` (boolean): Include operation details (default: true)
- `max_depth` (int): Maximum lineage depth to traverse

**Response:**
```json
{
  "success": true,
  "tensor_id": "uuid-target-tensor",
  "lineage_graph": {
    "root_tensors": ["uuid-input-1", "uuid-input-2"],
    "max_depth": 3,
    "total_operations": 5,
    "nodes": [
      {
        "tensor_id": "uuid-target-tensor",
        "depth": 0,
        "operation_id": "uuid-final-op",
        "parent_tensors": ["uuid-intermediate-1"]
      }
    ],
    "operations": [
      {
        "operation_id": "uuid-final-op",
        "operation_type": "softmax",
        "duration_ms": 25
      }
    ]
  }
}
```

### Export Lineage Visualization

Get lineage graph in DOT format for visualization.

```http
GET /lineage/tensor/{tensor_id}/dot
```

**Response:**
```json
{
  "success": true,
  "tensor_id": "uuid-tensor",
  "dot_graph": "digraph lineage {\n  \"tensor-1\" -> \"op-1\" -> \"tensor-2\";\n}"
}
```

## Analytics & Search Endpoints

### Vector Similarity Search

Find tensors similar to a query vector.

```http
POST /search/vector
```

**Request Body:**
```json
{
  "query": [0.1, 0.2, 0.3, 0.4],
  "dataset_name": "embeddings",
  "k": 10,
  "similarity_threshold": 0.8,
  "include_vectors": false,
  "metadata_filters": {
    "category": "image",
    "quality": "high"
  }
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "tensor_id": "uuid-similar-1",
      "similarity_score": 0.95,
      "metadata": {
        "category": "image",
        "filename": "cat_001.jpg"
      },
      "vector": null
    }
  ],
  "search_time_ms": 45
}
```

### Dataset Analytics

Get comprehensive analytics for a dataset.

```http
GET /datasets/{dataset_name}/analytics
```

**Response:**
```json
{
  "dataset_name": "training_images",
  "statistics": {
    "total_tensors": 50000,
    "total_size_bytes": 2147483648,
    "average_tensor_size": 42949,
    "compression_ratio": 2.3,
    "storage_efficiency": "56%"
  },
  "shape_distribution": {
    "[3, 224, 224]": 35000,
    "[3, 256, 256]": 15000
  },
  "metadata_summary": {
    "most_common_tags": ["training", "validated", "augmented"],
    "date_range": {
      "earliest": "2024-01-01T00:00:00Z",
      "latest": "2024-01-15T23:59:59Z"
    }
  }
}
```

## System Management Endpoints

### Health Check

Monitor system health and status.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.2.3",
  "services": {
    "database": "healthy",
    "storage": "healthy", 
    "compute": "healthy",
    "indexing": "healthy"
  },
  "performance": {
    "avg_response_time_ms": 45,
    "requests_per_second": 150,
    "error_rate_percent": 0.1
  }
}
```

### System Metrics

Get detailed system performance metrics.

```http
GET /metrics
```

**Response:**
```json
{
  "system": {
    "cpu_usage_percent": 65.2,
    "memory_usage_percent": 78.5,
    "disk_usage_percent": 45.1,
    "network_io_mbps": 125.3
  },
  "tensorus": {
    "total_tensors_stored": 1250000,
    "operations_executed_today": 50000,
    "cache_hit_ratio": 0.85,
    "average_operation_time_ms": 120
  },
  "storage": {
    "compression_effectiveness": 2.1,
    "index_efficiency": 0.92,
    "storage_growth_rate_gb_day": 15.2
  }
}
```

## Error Handling

### Standard Error Response

All endpoints return consistent error responses:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_TENSOR_ID",
    "message": "Tensor with ID 'invalid-id' not found",
    "details": {
      "dataset": "training_data",
      "requested_id": "invalid-id"
    },
    "request_id": "uuid-request-id",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_API_KEY` | API key invalid or expired | 401 |
| `INSUFFICIENT_PERMISSIONS` | Operation not allowed | 403 |
| `TENSOR_NOT_FOUND` | Tensor ID does not exist | 404 |
| `DATASET_NOT_FOUND` | Dataset does not exist | 404 |
| `INVALID_OPERATION` | Operation parameters invalid | 400 |
| `OPERATION_FAILED` | Operation execution failed | 500 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `STORAGE_FULL` | Storage quota exceeded | 507 |

## SDK Usage Examples

### Python SDK

```python
import tensorus

# Initialize client
client = tensorus.Client(api_key="your-api-key")

# Store tensor
tensor_id = client.store_tensor(
    dataset="training_data",
    tensor=my_numpy_array,
    metadata={"epoch": 5, "accuracy": 0.95}
)

# Execute operation
result = client.execute_operation(
    operation="matrix_multiply",
    inputs={"a": tensor_id, "b": other_tensor_id}
)

# Query with NQL
results = client.query("find tensors where metadata.accuracy > 0.9")
```

### JavaScript SDK

```javascript
const Tensorus = require('@tensorus/sdk');

const client = new Tensorus.Client({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.tensorus.com/v1'
});

// Async operations
const tensorId = await client.storeTensor({
  dataset: 'training_data',
  tensor: [[1, 2], [3, 4]],
  metadata: { source: 'synthetic' }
});

// Batch operations
const results = await client.batchOperations([
  { operation: 'add', inputs: { a: 'tensor-1', b: 'tensor-2' }},
  { operation: 'normalize', inputs: { tensor: '@previous' }}
]);
```

## Rate Limits & Quotas

### Standard Limits

| Tier | Requests/min | Storage GB | Operations/day |
|------|--------------|------------|----------------|
| Developer | 100 | 10 | 10,000 |
| Professional | 1,000 | 100 | 100,000 |
| Enterprise | 10,000 | 1,000 | Unlimited |

### Handling Rate Limits

When rate limited, the API returns HTTP 429 with retry information:

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "retry_after_seconds": 60
  }
}
```

## Support & Resources

- **Documentation**: https://docs.tensorus.com
- **API Status**: https://status.tensorus.com  
- **Support Portal**: https://support.tensorus.com
- **GitHub**: https://github.com/tensorus/tensorus
- **Community**: https://community.tensorus.com

---

**Need help?** Contact our API support team at api-support@tensorus.com