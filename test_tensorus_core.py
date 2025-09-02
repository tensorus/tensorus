"""
Test script for Tensorus core functionality
"""
import sys
import os
import logging
import torch

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tensorus_test.log')
    ]
)

def test_core():
    """Test basic Tensorus core functionality."""
    try:
        logging.info("Starting Tensorus core test...")
        
        # Import Tensorus core
        try:
            from tensorus.tensorus_operational_core import create_tensorus_core
            logging.info("✅ Successfully imported tensorus_operational_core")
        except ImportError as e:
            logging.error(f"❌ Failed to import tensorus_operational_core: {e}")
            logging.exception("Import error details:")
            return

        # Create test directory
        test_dir = "./test_core"
        try:
            os.makedirs(test_dir, exist_ok=True)
            logging.info(f"✅ Created test directory: {os.path.abspath(test_dir)}")
        except Exception as e:
            logging.error(f"❌ Failed to create test directory: {e}")
            return

        # Initialize core
        try:
            logging.info("Initializing Tensorus core...")
            core = create_tensorus_core(test_dir)
            logging.info("✅ Successfully initialized Tensorus core")
        except Exception as e:
            logging.error(f"❌ Failed to initialize Tensorus core: {e}")
            logging.exception("Initialization error details:")
            return

        # Test tensor storage and retrieval
        try:
            logging.info("\nTesting tensor storage and retrieval...")
            test_tensor = torch.randn(2, 2)
            logging.info(f"Created test tensor with shape: {test_tensor.shape}")
            
            # Store tensor
            tensor_id = core.store_tensor(test_tensor, {"test": "data", "category": "test"})
            logging.info(f"✅ Stored tensor with ID: {tensor_id}")
            
            # List tensors
            tensor_ids = core.list_tensors()
            logging.info(f"Found {len(tensor_ids)} tensors: {tensor_ids}")
            
            # Retrieve tensor
            op_tensor = core.get_tensor(tensor_id)
            logging.info(f"✅ Retrieved tensor with ID: {op_tensor.tensor_id}")
            logging.info(f"Tensor data:\n{op_tensor.data}")
            
            # Verify data
            if not torch.allclose(test_tensor, op_tensor.data):
                raise ValueError("Retrieved tensor data does not match original!")
                
            logging.info("✅ Data verification passed")
            
        except Exception as e:
            logging.error(f"❌ Tensor operation test failed: {e}")
            logging.exception("Error details:")
            return

        logging.info("\n✅ All tests completed successfully!")
        
    except Exception as e:
        logging.error(f"❌ Unexpected error in test: {e}")
        logging.exception("Unexpected error:")
    finally:
        # Clean up
        try:
            if os.path.exists(test_dir):
                import shutil
                shutil.rmtree(test_dir, ignore_errors=True)
                logging.info(f"Cleaned up test directory: {test_dir}")
        except Exception as e:
            logging.warning(f"Warning: Failed to clean up test directory: {e}")

if __name__ == "__main__":
    test_core()
