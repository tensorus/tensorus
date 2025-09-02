import sys
import os
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TensorusTest")

def main():
    try:
        logger.info("Starting Tensorus simple test...")
        
        # Check Python version
        logger.info(f"Python version: {sys.version}")
        
        # Check PyTorch installation
        try:
            import torch
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
        except ImportError as e:
            logger.error(f"Failed to import PyTorch: {e}")
            return
            
        # Try to import Tensorus core
        try:
            from tensorus.tensorus_operational_core import TensorusOperationalCore
            logger.info("✅ Successfully imported TensorusOperationalCore")
        except ImportError as e:
            logger.error(f"Failed to import TensorusOperationalCore: {e}")
            import traceback
            traceback.print_exc()
            return
            
        # Create a test directory
        test_dir = "./test_tensorus_simple"
        os.makedirs(test_dir, exist_ok=True)
        logger.info(f"Using test directory: {os.path.abspath(test_dir)}")
        
        # Initialize core
        try:
            core = TensorusOperationalCore(storage_path=test_dir)
            logger.info("✅ Successfully initialized TensorusOperationalCore")
            
            # Test tensor storage
            test_tensor = torch.randn(2, 2)
            logger.info(f"Created test tensor with shape: {test_tensor.shape}")
            
            # Store tensor
            tensor_id = core.store_tensor(test_tensor, {"test": "data"})
            logger.info(f"✅ Stored tensor with ID: {tensor_id}")
            
            # List tensors
            tensor_ids = core.list_tensors()
            logger.info(f"Found {len(tensor_ids)} tensors: {tensor_ids}")
            
            # Retrieve tensor
            op_tensor = core.get_tensor(tensor_id)
            logger.info(f"✅ Retrieved tensor with ID: {op_tensor.tensor_id}")
            logger.info(f"Tensor data:\n{op_tensor.data}")
            
        except Exception as e:
            logger.error(f"Error during Tensorus operations: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists(test_dir):
            import shutil
            try:
                shutil.rmtree(test_dir)
                logger.info(f"Cleaned up test directory: {test_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up test directory: {e}")

if __name__ == "__main__":
    main()
