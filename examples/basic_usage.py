import sys
import os
import numpy as np

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensor_database import TensorDatabase

def main():
    """Basic demonstration of Tensorus functionality."""
    print("Tensorus: Basic Usage Example")
    print("=============================")
    
    # Initialize the database with a test storage location
    db = TensorDatabase(
        storage_path="example_db.h5",
        index_path="example_index.pkl"
    )
    
    # Create some example tensor data
    print("\n1. Creating and storing tensors...")
    
    # 2D tensor (matrix)
    matrix = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])
    matrix_id = db.save(matrix, metadata={"name": "example_matrix", "type": "2d"})
    print(f"  Saved matrix with ID: {matrix_id}")
    
    # 3D tensor
    tensor_3d = np.random.rand(2, 3, 4)
    tensor_3d_id = db.save(tensor_3d, metadata={"name": "example_3d", "type": "3d"})
    print(f"  Saved 3D tensor with ID: {tensor_3d_id}")
    
    # Another 3D tensor similar to the first
    similar_tensor = tensor_3d.copy()
    similar_tensor += np.random.randn(2, 3, 4) * 0.1  # Add small noise
    similar_id = db.save(similar_tensor, metadata={"name": "similar_tensor", "type": "3d"})
    print(f"  Saved similar tensor with ID: {similar_id}")
    
    # Retrieve one of the tensors
    print("\n2. Retrieving a tensor...")
    retrieved_tensor, metadata = db.get(matrix_id)
    print(f"  Retrieved tensor with metadata: {metadata}")
    print(f"  Tensor data:\n{retrieved_tensor}")
    
    # List all tensors in the database
    print("\n3. Listing all tensors in the database...")
    tensors = db.list_tensors()
    for t in tensors:
        print(f"  Tensor ID: {t['id']}, Metadata: {t['metadata']}")
    
    # Perform tensor operations
    print("\n4. Performing tensor operations...")
    
    # Create another matrix for operations
    matrix2 = np.array([
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ])
    matrix2_id = db.save(matrix2, metadata={"name": "example_matrix2", "type": "2d"})
    
    # Addition
    print("\n  4.1 Matrix addition:")
    sum_matrix = db.process("add", [matrix_id, matrix2_id])
    print(f"  {retrieved_tensor} + \n  {matrix2} = \n  {sum_matrix}")
    
    # Matrix multiplication
    print("\n  4.2 Matrix multiplication:")
    result = db.process("matmul", [matrix_id, matrix2.T])  # Transpose for compatible dimensions
    print(f"  Result:\n{result}")
    
    # Perform similarity search
    print("\n5. Performing similarity search...")
    query = tensor_3d.copy()
    query += np.random.randn(2, 3, 4) * 0.2  # Add some noise
    
    results = db.search_similar(query, k=2)
    print(f"  Found {len(results)} similar tensors:")
    for i, result in enumerate(results):
        print(f"  {i+1}. ID: {result['tensor_id']}, Distance: {result['distance']}")
        print(f"     Metadata: {result['metadata']}")
    
    # Save database state
    print("\n6. Saving database state...")
    db.save_index()
    print("  Database state saved successfully.")
    
    # Get performance metrics
    print("\n7. Database metrics:")
    metrics = db.get_metrics()
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    # Create example directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    main() 