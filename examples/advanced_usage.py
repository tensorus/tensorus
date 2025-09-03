"""
Advanced Tensorus Usage Examples

This script demonstrates advanced usage of the Tensorus library,
including tensor operations, vector similarity search, and metadata management.
"""
import numpy as np
from datetime import datetime
from tensorus import Tensorus

def tensor_operations_demo():
    """Demonstrates advanced tensor operations."""
    print("=== Advanced Tensor Operations ===")
    ts = Tensorus()
    
    # Create a batch of random tensors
    batch_size = 5
    tensors = [
        ts.create_tensor(
            np.random.rand(3, 3),
            name=f"tensor_{i}",
            metadata={"batch_id": 42, "created_at": datetime.utcnow().isoformat()}
        )
        for i in range(batch_size)
    ]
    
    # Batch operations
    print(f"Created {len(tensors)} tensors in batch 42")
    
    # Stack tensors
    stacked = ts.stack(tensors, axis=0)
    print(f"\nStacked tensor shape: {stacked.shape}")
    
    # SVD decomposition
    U, S, V = ts.svd(stacked)
    print(f"\nSVD Results:")
    print(f"U shape: {U.shape}")
    print(f"S shape: {S.shape}")
    print(f"V shape: {V.shape}")
    
    return tensors

def semantic_search_demo():
    """Demonstrates semantic search capabilities."""
    print("\n=== Semantic Search Demo ===")
    ts = Tensorus()
    
    # Sample document collection
    documents = [
        "Machine learning is transforming industries.",
        "Deep learning models require large datasets.",
        "Tensorus provides efficient tensor operations.",
        "Natural language processing is a key AI technology.",
        "Vector databases enable fast similarity search."
    ]
    
    # Create an index for document embeddings
    index_name = "document_embeddings"
    embedding_dim = 768  # Example dimension for sentence embeddings
    
    if not ts.index_exists(index_name):
        ts.create_index(index_name, dimensions=embedding_dim)
    
    # Generate embeddings (in a real scenario, use a proper embedding model)
    # This is a simplified example with random embeddings
    doc_embeddings = np.random.rand(len(documents), embedding_dim).astype(np.float32)
    doc_ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Add to index
    ts.add_vectors(index_name, doc_ids, doc_embeddings)
    
    # Search for similar documents
    query = "AI and machine learning"
    query_embedding = np.random.rand(embedding_dim).astype(np.float32)  # In practice, embed the query
    
    print(f"\nSearching for: {query}")
    results = ts.search_vectors(index_name, query_embedding, k=2)
    
    print("\nTop 2 most similar documents:")
    for i, (doc_id, score) in enumerate(zip(results.ids, results.scores), 1):
        doc_idx = int(doc_id.split('_')[1])
        print(f"{i}. Score: {score:.4f}")
        print(f"   Text: {documents[doc_idx]}")
        print()

def metadata_management_demo():
    """Demonstrates advanced metadata management."""
    print("\n=== Metadata Management ===")
    ts = Tensorus()
    
    # Create a tensor with comprehensive metadata
    data = np.random.rand(5, 5)
    metadata = {
        "source": {
            "type": "synthetic",
            "generator": "numpy.random",
            "version": "1.0.0"
        },
        "processing": {
            "normalized": True,
            "augmented": False,
            "pipeline_steps": ["scaling", "centering"]
        },
        "license": "MIT",
        "tags": ["example", "advanced", "metadata"]
    }
    
    tensor = ts.create_tensor(
        data,
        name="advanced_metadata_demo",
        metadata=metadata,
        description="Example tensor with comprehensive metadata"
    )
    
    # Query tensors by metadata
    print("Searching for tensors with tag 'example':")
    results = ts.search_metadata({"tags": "example"})
    
    for tensor in results:
        print(f"\nFound tensor: {tensor.name}")
        print(f"Metadata tags: {tensor.metadata.get('tags', [])}")
    
    # Update metadata
    print("\nUpdating metadata...")
    tensor.update_metadata({
        "processing.augmented": True,
        "processing.pipeline_steps": ["scaling", "centering", "normalization"],
        "version": "1.0.1"
    })
    
    print("\nUpdated metadata:")
    print(f"Augmented: {tensor.metadata['processing']['augmented']}")
    print(f"Pipeline steps: {tensor.metadata['processing']['pipeline_steps']}")
    print(f"Version: {tensor.metadata.get('version', '1.0.0')}")

if __name__ == "__main__":
    # Run all demos
    tensor_operations_demo()
    semantic_search_demo()
    metadata_management_demo()
