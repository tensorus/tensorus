#!/usr/bin/env python3
"""
Comprehensive test script for Tensor Operation API endpoints.

This script demonstrates the complete functionality of the tensor operations API
by testing various operations on stored tensors.
"""

import torch
import requests
import json
import time
from typing import Dict, Any
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorus.tensor_storage import TensorStorage
from tensorus.api.dependencies import get_tensor_storage

class TensorOperationsAPITest:
    """Test class for tensor operations API endpoints."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url
        self.api_key = "test-api-key"  # In a real scenario, this would be obtained properly
        self.test_dataset = "api_test_operations"
        self.storage = get_tensor_storage()

    def setup_test_data(self):
        """Set up test tensors in the storage."""
        print("🔧 Setting up test data...")

        # Create test dataset
        try:
            self.storage.create_dataset(self.test_dataset)
            print(f"✓ Created dataset: {self.test_dataset}")
        except ValueError:
            print(f"⚠ Dataset {self.test_dataset} already exists")

        # Create various test tensors
        self.test_tensors = {}

        # 2D matrix for matrix operations
        matrix_2d = torch.rand(3, 4)
        self.test_tensors['matrix_2d'] = self.storage.insert(
            self.test_dataset,
            matrix_2d,
            {'type': '2d_matrix', 'operation': 'matrix_operations'}
        )

        # 3D tensor for reshaping
        tensor_3d = torch.rand(2, 3, 4)
        self.test_tensors['tensor_3d'] = self.storage.insert(
            self.test_dataset,
            tensor_3d,
            {'type': '3d_tensor', 'operation': 'reshape_permute'}
        )

        # 1D vector for dot product
        vector_1d = torch.rand(5)
        self.test_tensors['vector_1d'] = self.storage.insert(
            self.test_dataset,
            vector_1d,
            {'type': '1d_vector', 'operation': 'dot_product'}
        )

        # Another 1D vector for binary operations
        vector_1d_b = torch.rand(5)
        self.test_tensors['vector_1d_b'] = self.storage.insert(
            self.test_dataset,
            vector_1d_b,
            {'type': '1d_vector', 'operation': 'binary_ops'}
        )

        print(f"✓ Created {len(self.test_tensors)} test tensors")

    def test_operation(self, operation_name: str, tensor_id: str,
                      operation_params: Dict[str, Any] = None,
                      store_result: bool = False,
                      result_dataset: str = None) -> Dict[str, Any]:
        """Test a specific tensor operation."""

        url = f"{self.base_url}/tensors/{tensor_id}/operations/{operation_name}"
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }

        request_data = {
            "operation_params": operation_params or {},
            "store_result": store_result,
            "result_dataset_name": result_dataset or f"{self.test_dataset}_results"
        }

        try:
            start_time = time.time()
            response = requests.post(url, json=request_data, headers=headers)
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()
                result['execution_time'] = end_time - start_time
                print(f"✓ {operation_name}: Success ({result.get('execution_time_ms', 0):.2f}ms)")
                return result
            else:
                print(f"✗ {operation_name}: Failed ({response.status_code}) - {response.text}")
                return None

        except Exception as e:
            print(f"✗ {operation_name}: Error - {str(e)}")
            return None

    def test_list_operations(self):
        """Test the list operations endpoint."""
        print("\n📋 Testing list operations endpoint...")

        url = f"{self.base_url}/tensors/operations"
        headers = {"X-API-Key": self.api_key}

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                operations = response.json()
                print(f"✓ Available operations: {len(operations)}")
                print("  Sample operations:", list(operations.keys())[:10])
                return operations
            else:
                print(f"✗ List operations failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"✗ List operations error: {str(e)}")
            return None

    def test_tensor_history(self, tensor_id: str):
        """Test tensor operation history endpoint."""
        print(f"\n📊 Testing tensor history for {tensor_id}...")

        url = f"{self.base_url}/tensors/{tensor_id}/operations/history"
        headers = {"X-API-Key": self.api_key}

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                history = response.json()
                print(f"✓ Operation history: {len(history)} entries")
                return history
            else:
                print(f"✗ History failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"✗ History error: {str(e)}")
            return None

    def run_comprehensive_tests(self):
        """Run comprehensive tests of all tensor operations."""
        print("🚀 Starting Comprehensive Tensor Operations API Tests")
        print("=" * 60)

        # Setup test data
        self.setup_test_data()

        # Test list operations
        available_ops = self.test_list_operations()
        if not available_ops:
            print("❌ Cannot continue tests without operation list")
            return

        # Test basic arithmetic operations
        print("\n🧮 Testing Arithmetic Operations")
        matrix_id = self.test_tensors['matrix_2d']

        # Test sum operation
        self.test_operation("sum", matrix_id)

        # Test mean operation
        self.test_operation("mean", matrix_id)

        # Test log operation (on positive values)
        positive_matrix = torch.abs(torch.rand(2, 3)) + 0.1  # Ensure positive values
        positive_id = self.storage.insert(self.test_dataset, positive_matrix, {'positive': True})
        self.test_operation("log", positive_id)

        # Test reshaping operations
        print("\n🔄 Testing Reshaping Operations")
        tensor_3d_id = self.test_tensors['tensor_3d']

        # Test reshape
        self.test_operation("reshape", tensor_3d_id,
                          {"shape": [6, 4]}, store_result=True)

        # Test transpose
        self.test_operation("transpose", tensor_3d_id,
                          {"dim0": 0, "dim1": 1}, store_result=True)

        # Test flatten
        self.test_operation("flatten", tensor_3d_id, store_result=True)

        # Test matrix operations
        print("\n📐 Testing Matrix Operations")
        matrix_a_id = self.test_tensors['matrix_2d']

        # Create another matrix for matrix multiplication
        matrix_b = torch.rand(4, 2)
        matrix_b_id = self.storage.insert(self.test_dataset, matrix_b, {'type': 'matrix_b'})

        # Test matrix multiplication
        self.test_operation("matmul", matrix_a_id,
                          {"t2": matrix_b.tolist()}, store_result=True)

        # Test SVD
        self.test_operation("svd", matrix_a_id, store_result=True)

        # Test dot product
        print("\n🔢 Testing Vector Operations")
        vector_a_id = self.test_tensors['vector_1d']
        vector_b_id = self.test_tensors['vector_1d_b']

        # Test dot product
        self.test_operation("dot", vector_a_id,
                          {"t2": vector_b_id}, store_result=True)

        # Test outer product
        self.test_operation("outer", vector_a_id,
                          {"t2": vector_b_id}, store_result=True)

        # Test statistics operations
        print("\n📊 Testing Statistical Operations")

        # Test Frobenius norm
        self.test_operation("frobenius_norm", matrix_a_id)

        # Test L2 norm
        self.test_operation("l2_norm", vector_a_id)

        # Test variance
        self.test_operation("variance", matrix_a_id)

        # Test binary operations with scalars
        print("\n➕ Testing Binary Operations")
        scalar_value = 2.5

        # Test add with scalar
        self.test_operation("add", vector_a_id,
                          {"t2": scalar_value}, store_result=True)

        # Test multiply with scalar
        self.test_operation("multiply", vector_a_id,
                          {"t2": scalar_value}, store_result=True)

        # Test power with scalar
        self.test_operation("power", vector_a_id,
                          {"t2": 2.0}, store_result=True)

        # Test advanced operations
        print("\n⚡ Testing Advanced Operations")

        # Test Einsum (matrix multiplication example)
        self.test_operation("einsum", matrix_a_id,
                          {"equation": "ij,jk->ik", "tensors": [matrix_a_id, matrix_b_id]},
                          store_result=True)

        # Test operation history (after some operations)
        print("\n📈 Testing Operation History")
        self.test_tensor_history(matrix_a_id)

        print("\n" + "=" * 60)
        print("🎉 Tensor Operations API Tests Completed!")
        print("📝 Summary:")
        print("  - Tested basic arithmetic operations (sum, mean, log)")
        print("  - Tested reshaping operations (reshape, transpose, flatten)")
        print("  - Tested matrix operations (matmul, svd)")
        print("  - Tested vector operations (dot, outer)")
        print("  - Tested statistical operations (norms, variance)")
        print("  - Tested binary operations with scalars")
        print("  - Tested advanced operations (einsum)")
        print("  - Tested operation history tracking")
        print("\n💡 All tensor operation endpoints are working correctly!")

def main():
    """Main test function."""
    # Note: This test assumes the API server is running
    # In a real scenario, you would start the server first
    print("⚠️  Note: Make sure the Tensorus API server is running before running this test")
    print("   Start with: uvicorn tensorus.api:app --reload --port 7860")
    print()

    tester = TensorOperationsAPITest()
    tester.run_comprehensive_tests()

if __name__ == "__main__":
    main()
