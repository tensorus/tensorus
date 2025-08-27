# Vector Database Integration Guide

This guide covers the comprehensive vector database capabilities integrated into Tensorus, enabling semantic search, embedding generation, and hybrid search functionality.

## Overview

Tensorus now includes full vector database capabilities that complement its existing tensor operations. The integration provides:

- **Vector Operations**: Cosine similarity, Euclidean distance, Manhattan distance, and more
- **Embedding Generation**: Support for sentence-transformers and OpenAI models
- **Approximate Nearest Neighbor (ANN) Indexing**: FAISS-based indexing for fast similarity search
- **Hybrid Search**: Combining semantic similarity with metadata filtering
- **REST API Endpoints**: Complete API for vector operations

## Quick Start

### 1. Install Dependencies

```bash
pip install sentence-transformers faiss-cpu openai tiktoken
```

For GPU acceleration:
```bash
pip install faiss-gpu
```

### 2. Basic Usage

#### Generate and Store Embeddings

```python
from tensorus.embedding_agent import EmbeddingAgent
from tensorus.tensor_storage import TensorStorage

# Initialize storage and agent
storage = TensorStorage()
agent = EmbeddingAgent(storage)

# Generate embeddings from text
texts = ["Hello world", "Machine learning", "Vector databases"]
result = agent.encode_and_store(
    texts=texts,
    dataset_name="my_embeddings",
    model_name="all-MiniLM-L6-v2"
)
```

#### Similarity Search

```python
# Search for similar embeddings
query = "artificial intelligence"
results = agent.similarity_search(
    query_text=query,
    dataset_name="my_embeddings",
    k=5
)

for result in results:
    print(f"Text: {result['metadata']['source_text']}")
    print(f"Similarity: {result['similarity']:.3f}")
```

#### Build Vector Index for Fast Search

```python
from tensorus.vector_index import VectorIndexManager

# Initialize index manager
index_manager = VectorIndexManager(storage)

# Build FAISS index
index = index_manager.build_index(
    dataset_name="my_embeddings",
    index_type="hnsw",  # or "flat", "ivf"
    metric="cosine"
)

# Fast search using index
results = index_manager.search_index(
    index_name="my_embeddings_hnsw",
    query="machine learning",
    k=10
)
```

## API Endpoints

### Embedding Generation

**POST** `/vector/embed`

```json
{
    "texts": ["Hello world", "Machine learning"],
    "dataset_name": "my_embeddings",
    "model_name": "all-MiniLM-L6-v2",
    "store_embeddings": true
}
```

### Similarity Search

**POST** `/vector/search`

```json
{
    "query": "artificial intelligence",
    "dataset_name": "my_embeddings",
    "k": 5,
    "model_name": "all-MiniLM-L6-v2"
}
```

### Hybrid Search

**POST** `/vector/hybrid-search`

```json
{
    "query": "machine learning",
    "dataset_name": "my_embeddings",
    "k": 10,
    "metadata_filters": {"category": "AI"},
    "vector_weight": 0.7,
    "metadata_weight": 0.3
}
```

### Vector Index Management

**POST** `/vector/index/build`

```json
{
    "dataset_name": "my_embeddings",
    "index_type": "hnsw",
    "metric": "cosine"
}
```

**POST** `/vector/index/search`

```json
{
    "index_name": "my_embeddings_hnsw",
    "query": "deep learning",
    "k": 5
}
```

## Supported Models

### Sentence Transformers
- `all-MiniLM-L6-v2` (384 dimensions) - Fast and efficient
- `all-mpnet-base-v2` (768 dimensions) - High quality
- `multi-qa-MiniLM-L6-cos-v1` (384 dimensions) - Q&A optimized

### OpenAI Models
- `text-embedding-ada-002` (1536 dimensions)
- `text-embedding-3-small` (1536 dimensions)
- `text-embedding-3-large` (3072 dimensions)

## Index Types

### Flat Index
- **Use case**: Small datasets (<10K vectors)
- **Pros**: Exact search, simple
- **Cons**: Linear search time

### IVF (Inverted File)
- **Use case**: Medium datasets (10K-1M vectors)
- **Pros**: Good balance of speed and accuracy
- **Cons**: Requires training

### HNSW (Hierarchical Navigable Small World)
- **Use case**: Large datasets (>100K vectors)
- **Pros**: Very fast search, good recall
- **Cons**: Higher memory usage

## Advanced Features

### Custom Metadata Schemas

```python
from tensorus.metadata.schemas import VectorMetadata

# Custom metadata for embeddings
metadata = VectorMetadata(
    source_text="Machine learning is amazing",
    embedding_model="all-MiniLM-L6-v2",
    similarity_threshold=0.8,
    tags=["AI", "ML", "technology"]
)
```

### Batch Operations

```python
# Process large batches efficiently
large_texts = ["text"] * 10000
results = agent.encode_batch(
    texts=large_texts,
    batch_size=100,
    model_name="all-MiniLM-L6-v2"
)
```

### Vector Operations

```python
from tensorus.vector_ops import VectorOps
import torch

# Direct vector operations
vec1 = torch.tensor([1.0, 2.0, 3.0])
vec2 = torch.tensor([4.0, 5.0, 6.0])

similarity = VectorOps.cosine_similarity(vec1, vec2)
distance = VectorOps.euclidean_distance(vec1, vec2)

# Batch operations
vectors = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
query = torch.tensor([1.0, 1.0])

indices, similarities = VectorOps.top_k_similarity(query, vectors, k=2)
```

## Performance Optimization

### Model Caching
Models are automatically cached in memory for faster subsequent use.

### Embedding Caching
Embeddings are cached to avoid recomputation of identical texts.

### GPU Acceleration
Install `faiss-gpu` and ensure CUDA is available for GPU acceleration.

### Batch Processing
Use batch operations for processing multiple texts efficiently.

## Integration with Existing Tensorus Features

### Metadata Integration
Vector embeddings seamlessly integrate with Tensorus metadata schemas.

### Storage Compatibility
All vector data is stored using the existing TensorStorage system.

### Agent Framework
Vector operations work within the existing agent architecture.

### API Consistency
Vector endpoints follow the same patterns as existing Tensorus APIs.

## Error Handling

The vector database integration includes comprehensive error handling:

- Model loading failures
- Dimension mismatches
- Invalid input validation
- Index corruption recovery
- Memory management

## Monitoring and Logging

All vector operations include detailed logging for:

- Performance metrics
- Model usage statistics
- Search accuracy metrics
- Error tracking

## Best Practices

1. **Choose the right model**: Balance between speed and quality
2. **Use appropriate index types**: Match index to dataset size
3. **Normalize vectors**: For cosine similarity
4. **Batch operations**: For better performance
5. **Monitor memory usage**: Especially with large indexes
6. **Regular index updates**: For dynamic datasets

## Troubleshooting

### Common Issues

**FAISS not available**
```bash
pip install faiss-cpu  # or faiss-gpu
```

**Out of memory errors**
- Use smaller batch sizes
- Choose more memory-efficient index types
- Enable GPU if available

**Poor search quality**
- Try different embedding models
- Adjust similarity thresholds
- Use hybrid search with metadata filtering

**Slow search performance**
- Build appropriate indexes
- Use GPU acceleration
- Consider approximate search for large datasets
