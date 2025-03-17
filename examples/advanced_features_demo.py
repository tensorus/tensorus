#!/usr/bin/env python
# Advanced Features Demo for Tensorus

import sys
import os
import numpy as np
import argparse
import time
import json
from typing import Dict, List, Any

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from tensor_data import TensorStorage
from tensor_indexer import TensorIndexer
from tensor_processor import TensorProcessor
from tensor_database import TensorDatabase
from tensor_integration import TensorAdvanced

def create_demo_database():
    """Create a demo database instance with some test data."""
    print("Creating demo database...")
    
    # Initialize components
    storage = TensorStorage("demo_data/tensor_storage.h5")
    indexer = TensorIndexer(dimension=128, index_type="flat", metric="l2")
    processor = TensorProcessor(use_gpu=False)
    
    # Create database
    db = TensorDatabase(storage=storage, indexer=indexer, processor=processor)
    
    # Add some test tensors
    print("Adding test tensors...")
    tensor_ids = []
    for i in range(10):
        # Create a random tensor
        tensor = np.random.rand(128).astype('float32')
        # Add metadata
        metadata = {
            "name": f"test_tensor_{i}",
            "description": f"Test tensor {i} for demo",
            "created_at": time.time(),
            "category": "demo"
        }
        # Save tensor
        tensor_id = db.save(tensor, metadata)
        tensor_ids.append(tensor_id)
        print(f"  Added tensor {i} with ID: {tensor_id}")
    
    return db, tensor_ids

def demo_agentic_reconfiguration(advanced: TensorAdvanced):
    """Demo the agentic reconfiguration with reinforcement learning."""
    print("\n=== Agentic Reconfiguration Demo ===")
    print("The agent is automatically monitoring the database and will reconfigure it as needed.")
    print("Let's perform some operations to trigger the agent...")
    
    # Create a sample tensor
    tensor = np.random.rand(256).astype('float32')
    
    # Perform operations to generate metrics
    for i in range(5):
        # Save
        tensor_id = advanced.save_tensor(tensor, {"demo": "agent"})
        print(f"  Saved tensor with ID: {tensor_id}")
        
        # Search
        results = advanced.search_similar(tensor, k=3)
        print(f"  Found {len(results)} similar tensors")
        
        # Let the agent observe the operations
        time.sleep(0.5)
    
    print("The agent is now collecting metrics and may reconfigure the database.")
    print("In a real-world scenario, the agent would observe the workload pattern")
    print("and optimize the database configuration accordingly.")

def demo_blockchain_provenance(advanced: TensorAdvanced, tensor_ids: List[str]):
    """Demo the blockchain for data provenance tracking."""
    print("\n=== Blockchain Provenance Demo ===")
    print("Let's perform some operations and track them on the blockchain.")
    
    # Get an existing tensor
    tensor_id = tensor_ids[0]
    tensor, metadata = advanced.get_tensor(tensor_id)
    print(f"Retrieved tensor with ID: {tensor_id}")
    
    # Perform operations
    print("Modifying the tensor...")
    modified_tensor = tensor * 2
    new_metadata = {**metadata, "modified": True, "operation": "multiply_by_2"}
    
    # Save as a new tensor
    new_tensor_id = advanced.save_tensor(modified_tensor, new_metadata)
    print(f"Saved modified tensor with ID: {new_tensor_id}")
    
    # Get another tensor and delete it
    if len(tensor_ids) > 1:
        delete_id = tensor_ids[1]
        print(f"Deleting tensor with ID: {delete_id}")
        advanced.database.delete(delete_id)
    
    # View tensor history from blockchain
    print("\nViewing tensor history from blockchain:")
    history = advanced.get_tensor_history(tensor_id)
    
    if history:
        for entry in history:
            print(f"  Operation: {entry.get('type', 'unknown')}")
            print(f"  Timestamp: {time.ctime(entry.get('timestamp', 0))}")
            print(f"  User: {entry.get('user_id', 'system')}")
            print(f"  Details: {entry.get('metadata', {})}")
            print()
    else:
        print("  No history found (blockchain may be disabled)")

def demo_tensor_query_language(advanced: TensorAdvanced, tensor_ids: List[str]):
    """Demo the Tensor Query Language (TQL)."""
    print("\n=== Tensor Query Language (TQL) Demo ===")
    print("Let's execute some TQL queries...")
    
    # Get an existing tensor ID
    tensor_id = tensor_ids[0] if tensor_ids else None
    if not tensor_id:
        print("No tensor IDs available. Skipping TQL demo.")
        return
    
    # Demo queries
    queries = [
        f"SELECT * FROM tensors WHERE id = '{tensor_id}'",
        "SELECT * FROM tensors WHERE metadata.category = 'demo' LIMIT 3",
        f"SELECT metadata FROM tensors WHERE id = '{tensor_id}'",
        "SEARCH tensors USING random(128) LIMIT 5"
    ]
    
    for query in queries:
        print(f"\nExecuting query: {query}")
        try:
            result = advanced.execute_tql(query)
            
            # Print results (abbreviated for large tensors)
            if "tensor" in result and isinstance(result["tensor"], np.ndarray):
                tensor_info = f"Tensor shape: {result['tensor'].shape}, dtype: {result['tensor'].dtype}"
                print(f"Result: {tensor_info}")
            elif "tensors" in result and isinstance(result["tensors"], list):
                print(f"Result: {len(result['tensors'])} tensors returned")
            elif "metadata" in result:
                print(f"Result: {result['metadata']}")
            elif "result" in result and isinstance(result["result"], np.ndarray):
                tensor_info = f"Tensor shape: {result['result'].shape}, dtype: {result['result'].dtype}"
                print(f"Result: {tensor_info}")
            else:
                print(f"Result: {result}")
        except Exception as e:
            print(f"Error executing query: {e}")

def demo_distributed_capabilities(advanced: TensorAdvanced):
    """Demo distributed capabilities if enabled."""
    print("\n=== Distributed Capabilities Demo ===")
    
    if not advanced.distributed_node:
        print("Distributed capabilities are not enabled in the configuration.")
        print("To test distributed features, update the configuration with:")
        print("  \"enable_distributed\": true")
        print("  \"distributed\": {")
        print("    \"role\": \"coordinator\",")
        print("    \"port\": 5050")
        print("  }")
        return
    
    print(f"Distributed node initialized with role: {advanced.distributed_node.role}")
    print(f"Node ID: {advanced.distributed_node.node_id}")
    print(f"Address: {advanced.distributed_node.address}:{advanced.distributed_node.port}")
    
    # Create a large tensor for distributed storage
    print("\nCreating a large tensor for distributed storage...")
    large_tensor = np.random.rand(1000, 1000).astype('float32')
    
    # Save with distributed storage
    tensor_id = advanced.save_tensor(large_tensor, {"type": "distributed_demo"}, distributed=True)
    print(f"Saved large tensor with ID: {tensor_id}")
    
    # In a real distributed setup, this would be partitioned across nodes
    print("In a multi-node setup, this tensor would be partitioned across nodes.")
    
    # If federated learning is enabled
    if advanced.federated:
        print("\nInitializing a federated learning model...")
        
        # Create a simple model with random weights
        model_weights = {
            "layer1": np.random.rand(10, 10).astype('float32'),
            "layer2": np.random.rand(10, 1).astype('float32')
        }
        
        # Create model config
        model_config = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 5
        }
        
        # Create federated model
        model_id = "demo_model"
        success = advanced.create_federated_model(model_id, model_weights, model_config)
        
        if success:
            print(f"Created federated model with ID: {model_id}")
            
            # Start federated training
            print("Starting federated training...")
            advanced.start_federated_training(model_id, rounds=2)
            print("Training initiated. In a real deployment, this would train across multiple nodes.")
        else:
            print("Failed to create federated model.")

def demo_security_features(advanced: TensorAdvanced):
    """Demo the security features if enabled."""
    print("\n=== Security Features Demo ===")
    
    if not advanced.security:
        print("Security features are not enabled in the configuration.")
        print("To test security features, update the configuration with:")
        print("  \"enable_security\": true")
        return
    
    print("Authenticating with the system...")
    
    # Try to authenticate with default credentials
    token = advanced.authenticate("admin", "admin")
    
    if token:
        print("Authentication successful!")
        print(f"Token: {token[:20]}... (truncated)")
        
        # Create a new user
        print("\nCreating a new user...")
        new_user = f"demo_user_{int(time.time())}"
        success = advanced.create_user(token, new_user, "password123", ["reader"])
        
        if success:
            print(f"Created user: {new_user} with 'reader' role")
            
            # Authenticate as the new user
            new_token = advanced.authenticate(new_user, "password123")
            
            if new_token:
                print(f"Authenticated as {new_user}")
                
                # Try operations with the new user
                print("\nTrying operations with reader permissions...")
                
                # Create a tensor
                test_tensor = np.random.rand(10).astype('float32')
                
                try:
                    # Should fail because reader role can't write
                    tensor_id = advanced.save_tensor(test_tensor, {"test": True}, token=new_token)
                    print(f"Save operation succeeded (unexpected): {tensor_id}")
                except PermissionError as e:
                    print(f"Save operation failed as expected: {e}")
                
                try:
                    # Should succeed because reader role can search
                    results = advanced.search_similar(test_tensor, token=new_token)
                    print(f"Search operation succeeded: {len(results)} results")
                except PermissionError as e:
                    print(f"Search operation failed (unexpected): {e}")
            else:
                print(f"Failed to authenticate as {new_user}")
        else:
            print("Failed to create new user")
    else:
        print("Authentication failed. Default credentials may have been changed.")

def demo_differential_privacy(advanced: TensorAdvanced, tensor_ids: List[str]):
    """Demo the differential privacy features if enabled."""
    print("\n=== Differential Privacy Demo ===")
    
    if not advanced.privacy:
        print("Differential privacy features are not enabled in the configuration.")
        print("To test privacy features, update the configuration with:")
        print("  \"enable_privacy\": true")
        return
    
    print("Differential privacy is enabled with the following settings:")
    budget_status = advanced.privacy.get_budget_status()
    print(f"Epsilon: {budget_status['epsilon']}")
    print(f"Delta: {budget_status['delta']}")
    print(f"Mechanism: {advanced.privacy.mechanism}")
    
    # Get an existing tensor
    if tensor_ids:
        tensor_id = tensor_ids[0]
        
        print("\nRetrieving tensor with and without privacy...")
        
        # Get without privacy
        tensor_regular, metadata_regular = advanced.get_tensor(tensor_id, apply_privacy=False)
        
        # Get with privacy
        tensor_private, metadata_private = advanced.get_tensor(tensor_id, apply_privacy=True)
        
        # Compare
        print(f"Regular tensor: shape={tensor_regular.shape}, mean={tensor_regular.mean():.4f}, std={tensor_regular.std():.4f}")
        print(f"Private tensor: shape={tensor_private.shape}, mean={tensor_private.mean():.4f}, std={tensor_private.std():.4f}")
        
        # Calculate difference
        diff = np.abs(tensor_regular - tensor_private)
        print(f"Difference: mean={diff.mean():.4f}, max={diff.max():.4f}")
        print("The private tensor has noise added to protect privacy.")
        
        # Show budget usage
        new_budget_status = advanced.privacy.get_budget_status()
        print(f"\nPrivacy budget after operation: {new_budget_status['remaining_budget']:.4f} remaining")
        print(f"Used: {new_budget_status['used_budget']:.4f} ({100-new_budget_status['percent_remaining']:.1f}%)")
    else:
        print("No tensor IDs available. Skipping differential privacy demo.")

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Tensorus Advanced Features Demo')
    parser.add_argument('--config', type=str, default=None, help='Path to advanced configuration file')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Tensorus Advanced Features Demo")
    print("=" * 60)
    
    # Create demo database
    db, tensor_ids = create_demo_database()
    
    # Initialize advanced features
    print("\nInitializing advanced features...")
    advanced = TensorAdvanced(db, config_path=args.config)
    
    # Run demos based on what's enabled
    if advanced.agent:
        demo_agentic_reconfiguration(advanced)
    
    if advanced.blockchain:
        demo_blockchain_provenance(advanced, tensor_ids)
    
    if advanced.tql_parser:
        demo_tensor_query_language(advanced, tensor_ids)
    
    demo_distributed_capabilities(advanced)
    
    demo_security_features(advanced)
    
    demo_differential_privacy(advanced, tensor_ids)
    
    # Cleanup
    print("\n=== Demo Complete ===")
    print("Shutting down components...")
    advanced.shutdown()
    print("Advanced features demo completed successfully!")

if __name__ == "__main__":
    main() 