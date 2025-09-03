"""
Basic Tensorus Usage Examples

This script demonstrates common operations with the Tensorus library.
"""
import numpy as np
from tensorus import Tensorus

def basic_tensor_operations():
    """Demonstrates basic tensor operations."""
    print("=== Basic Tensor Operations ===")
    ts = Tensorus()
    
    # Create tensors
    a = ts.create_tensor([[1, 2], [3, 4]], name="matrix_a")
    b = ts.create_tensor([[5, 6], [7, 8]], name="matrix_b")
    
    # Basic arithmetic
    add_result = a + b
    print(f"Addition:\n{add_result}")
    
    # Matrix multiplication
    matmul_result = ts.matmul(a, b)
    print(f"\nMatrix Multiplication:\n{matmul_result}")
    
    # Transpose
    transposed = a.transpose()
    print(f"\nTranspose of A:\n{transposed}")

def vector_database_example():
    """Demonstrates vector database functionality."""
    print("\n=== Vector Database Example ===")
    ts = Tensorus()
    
    # Create a vector index
    index_name = "image_embeddings"
    dimensions = 384  # Example dimension for image embeddings
    
    if not ts.index_exists(index_name):
        ts.create_index(index_name, dimensions=dimensions)
    
    # Generate some random embeddings
    num_vectors = 1000
    embeddings = np.random.rand(num_vectors, dimensions).astype(np.float32)
    vector_ids = [f"img_{i}" for i in range(num_vectors)]
    
    # Add to index
    ts.add_vectors(index_name, vector_ids, embeddings)
    print(f"Added {num_vectors} vectors to index '{index_name}'")
    
    # Perform a search
    query = np.random.rand(dimensions).astype(np.float32)
    results = ts.search_vectors(index_name, query, k=3)
    
    print("\nSearch Results:")
    for i, (vector_id, score) in enumerate(zip(results.ids, results.scores), 1):
        print(f"{i}. ID: {vector_id}, Similarity: {score:.4f}")

def tensor_metadata_example():
    """Shows how to work with tensor metadata."""
    print("\n=== Tensor Metadata Example ===")
    ts = Tensorus()
    
    # Create a tensor with metadata
    data = np.random.rand(3, 3)
    metadata = {
        "source": "synthetic_data",
        "creation_date": "2025-09-03",
        "tags": ["example", "test"]
    }
    
    tensor = ts.create_tensor(
        data,
        name="sample_tensor",
        metadata=metadata,
        description="A sample tensor with metadata"
    )
    
    # Retrieve and display metadata
    print(f"Tensor ID: {tensor.id}")
    print(f"Name: {tensor.name}")
    print(f"Shape: {tensor.shape}")
    print(f"Metadata: {tensor.metadata}")
    print(f"Description: {tensor.description}")

if __name__ == "__main__":
    basic_tensor_operations()
    vector_database_example()
    tensor_metadata_example()
