"""
Comprehensive test for Tensorus core functionality with detailed error reporting.
"""
import sys
import os
import logging
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tensorus_test.log')
    ]
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if required packages can be imported."""
    try:
        import torch
        logger.info(f"✅ PyTorch version: {torch.__version__}")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import PyTorch: {e}")
        return False

def test_tensorus_import():
    """Test if Tensorus core can be imported."""
    try:
        from tensorus.tensorus_operational_core import create_tensorus_core
        logger.info("✅ Successfully imported tensorus_operational_core")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import tensorus_operational_core: {e}")
        logger.error(traceback.format_exc())
        return False

def test_core_operations():
    """Test basic Tensorus core operations."""
    try:
        from tensorus.tensorus_operational_core import create_tensorus_core
        import torch
        
        # Create a test directory
        test_dir = "./test_tensorus_core"
        os.makedirs(test_dir, exist_ok=True)
        logger.info(f"Created test directory: {os.path.abspath(test_dir)}")
        
        # Initialize core
        logger.info("Initializing Tensorus core...")
        core = create_tensorus_core(test_dir)
        logger.info("✅ Successfully initialized Tensorus core")
        
        # Test tensor storage and retrieval
        logger.info("Testing tensor storage and retrieval...")
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
        
        # Verify data
        if not torch.allclose(test_tensor, op_tensor.data):
            raise ValueError("Retrieved tensor data does not match original")
            
        logger.info("✅ Data verification passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        logger.error(traceback.format_exc())
        return False
    finally:
        # Clean up
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)
            logger.info(f"Cleaned up test directory: {test_dir}")

def main():
    """Main test function."""
    logger.info("=== Starting Tensorus Core Tests ===")
    
    # Test 1: Check Python environment
    logger.info("\n[1/3] Testing Python environment...")
    if not test_imports():
        logger.error("❌ Python environment test failed")
        return 1
    
    # Test 2: Check Tensorus imports
    logger.info("\n[2/3] Testing Tensorus imports...")
    if not test_tensorus_import():
        logger.error("❌ Tensorus import test failed")
        return 1
    
    # Test 3: Test core operations
    logger.info("\n[3/3] Testing core operations...")
    if not test_core_operations():
        logger.error("❌ Core operations test failed")
        return 1
    
    logger.info("\n✅ All tests passed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
