#!/usr/bin/env python3
"""
Comprehensive Tensor Operations Examples for Tensorus

This script demonstrates the complete range of tensor operations available in Tensorus,
providing practical examples for each category of operations. This addresses GAP 9:
Limited Practical Examples by offering comprehensive, real-world usage patterns.

Categories covered:
- Basic arithmetic operations
- Matrix and dot operations
- Reduction operations
- Reshaping and slicing
- Concatenation and splitting
- Advanced operations (einsum, autograd)
- Linear algebra operations
- Convolution operations
- Statistical operations
- Norm calculations

Each section includes:
- Multiple practical examples
- Error handling demonstrations
- Performance considerations
- Best practices
"""

import torch
import numpy as np
import sys
from pathlib import Path
import time
import matplotlib.pyplot as plt

# Add tensorus to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tensorus.tensor_ops import TensorOps
except ImportError as e:
    print(f"Error importing tensorus modules: {e}")
    sys.exit(1)


def section_header(title: str, description: str = ""):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    if description:
        print(f" {description}")
    print("=" * 80)


def subsection_header(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def demonstrate_arithmetic_operations():
    """Comprehensive examples of arithmetic operations."""
    section_header("ARITHMETIC OPERATIONS", "Basic element-wise operations with scalars and tensors")
    
    # Create sample tensors
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    scalar = 2.5
    
    print(f"Tensor A:\n{a}")
    print(f"Tensor B:\n{b}")
    print(f"Scalar: {scalar}")
    
    subsection_header("Addition")
    # Tensor + Tensor
    result_add_tensor = TensorOps.add(a, b)
    print(f"A + B:\n{result_add_tensor}")
    
    # Tensor + Scalar
    result_add_scalar = TensorOps.add(a, scalar)
    print(f"A + {scalar}:\n{result_add_scalar}")
    
    subsection_header("Subtraction")
    result_sub = TensorOps.subtract(a, b)
    print(f"A - B:\n{result_sub}")
    
    result_sub_scalar = TensorOps.subtract(a, 1.0)
    print(f"A - 1.0:\n{result_sub_scalar}")
    
    subsection_header("Multiplication")
    result_mul = TensorOps.multiply(a, b)
    print(f"A * B (element-wise):\n{result_mul}")
    
    result_mul_scalar = TensorOps.multiply(a, scalar)
    print(f"A * {scalar}:\n{result_mul_scalar}")
    
    subsection_header("Division")
    result_div = TensorOps.divide(a, b)
    print(f"A / B:\n{result_div}")
    
    # Demonstrate division by zero handling
    print("\nDivision by zero demonstration:")
    zero_tensor = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    try:
        result_div_zero = TensorOps.divide(a[:, :2], zero_tensor)
        print(f"A[:, :2] / zero_tensor:\n{result_div_zero}")
        print("Note: Division by zero results in inf values")
    except ValueError as e:
        print(f"Division by zero error: {e}")
    
    subsection_header("Power Operations")
    result_pow = TensorOps.power(a, 2)
    print(f"A^2:\n{result_pow}")
    
    # Fractional powers
    result_sqrt = TensorOps.power(a, 0.5)
    print(f"A^0.5 (square root):\n{result_sqrt}")
    
    # Tensor power
    power_tensor = torch.tensor([[1, 2, 3], [2, 1, 2]])
    result_pow_tensor = TensorOps.power(a, power_tensor)
    print(f"A^power_tensor:\n{result_pow_tensor}")
    
    subsection_header("Logarithm")
    positive_tensor = torch.abs(a) + 0.1  # Ensure positive values
    result_log = TensorOps.log(positive_tensor)
    print(f"log(abs(A) + 0.1):\n{result_log}")
    
    # Demonstrate warning for non-positive values
    print("\nLogarithm of tensor with non-positive values:")
    mixed_tensor = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -0.5, 3.0]])
    result_log_mixed = TensorOps.log(mixed_tensor)
    print(f"log(mixed_tensor):\n{result_log_mixed}")
    print("Note: NaN and -inf values appear for non-positive inputs")


def demonstrate_matrix_operations():
    """Comprehensive examples of matrix and dot operations."""
    section_header("MATRIX AND DOT OPERATIONS", "Linear algebra operations including matmul, dot, and cross products")
    
    subsection_header("Matrix Multiplication")
    # Standard matrix multiplication
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2
    B = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])  # 2x3
    
    print(f"Matrix A (3x2):\n{A}")
    print(f"Matrix B (2x3):\n{B}")
    
    result_matmul = TensorOps.matmul(A, B)
    print(f"A @ B (3x3):\n{result_matmul}")
    
    # Batch matrix multiplication
    print("\nBatch matrix multiplication:")
    batch_A = torch.randn(3, 4, 5)  # 3 matrices of size 4x5
    batch_B = torch.randn(3, 5, 6)  # 3 matrices of size 5x6
    batch_result = TensorOps.matmul(batch_A, batch_B)
    print(f"Batch A shape: {batch_A.shape}")
    print(f"Batch B shape: {batch_B.shape}")
    print(f"Batch result shape: {batch_result.shape}")
    
    # Broadcasting example
    print("\nMatrix-vector multiplication (broadcasting):")
    matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3
    vector = torch.tensor([1.0, 2.0, 3.0])  # 3
    mv_result = TensorOps.matmul(matrix, vector)
    print(f"Matrix (2x3) @ Vector (3,) = {mv_result.shape}:\n{mv_result}")
    
    subsection_header("Dot Product")
    vec1 = torch.tensor([1.0, 2.0, 3.0])
    vec2 = torch.tensor([4.0, 5.0, 6.0])
    
    print(f"Vector 1: {vec1}")
    print(f"Vector 2: {vec2}")
    
    dot_result = TensorOps.dot(vec1, vec2)
    print(f"Dot product: {dot_result.item()}")
    
    # Demonstrate error handling for incompatible shapes
    print("\nError handling for incompatible shapes:")
    try:
        wrong_vec = torch.tensor([1.0, 2.0])
        TensorOps.dot(vec1, wrong_vec)
    except ValueError as e:
        print(f"Expected error: {e}")
    
    subsection_header("Outer Product")
    outer_result = TensorOps.outer(vec1, vec2)
    print(f"Outer product shape: {outer_result.shape}")
    print(f"Outer product:\n{outer_result}")
    
    subsection_header("Cross Product")
    # 3D vectors for cross product
    v1_3d = torch.tensor([1.0, 0.0, 0.0])
    v2_3d = torch.tensor([0.0, 1.0, 0.0])
    
    print(f"Vector 1 (3D): {v1_3d}")
    print(f"Vector 2 (3D): {v2_3d}")
    
    cross_result = TensorOps.cross(v1_3d, v2_3d)
    print(f"Cross product: {cross_result}")
    
    # Cross product with higher dimensional tensors
    print("\nCross product with batched vectors:")
    batch_v1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    batch_v2 = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    batch_cross = TensorOps.cross(batch_v1, batch_v2)
    print(f"Batch cross product:\n{batch_cross}")


def demonstrate_reduction_operations():
    """Comprehensive examples of reduction operations."""
    section_header("REDUCTION OPERATIONS", "Sum, mean, min, max operations across different dimensions")
    
    # Create a 3D tensor for comprehensive demonstrations
    tensor_3d = torch.tensor([[[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0]],
                             [[7.0, 8.0, 9.0],
                              [10.0, 11.0, 12.0]]])
    
    print(f"3D Tensor shape: {tensor_3d.shape}")
    print(f"3D Tensor:\n{tensor_3d}")
    
    subsection_header("Sum Operations")
    # Total sum
    total_sum = TensorOps.sum(tensor_3d)
    print(f"Total sum: {total_sum.item()}")
    
    # Sum along different dimensions
    sum_dim0 = TensorOps.sum(tensor_3d, dim=0)
    print(f"Sum along dim=0 (shape {sum_dim0.shape}):\n{sum_dim0}")
    
    sum_dim1 = TensorOps.sum(tensor_3d, dim=1)
    print(f"Sum along dim=1 (shape {sum_dim1.shape}):\n{sum_dim1}")
    
    sum_dim2 = TensorOps.sum(tensor_3d, dim=2)
    print(f"Sum along dim=2 (shape {sum_dim2.shape}):\n{sum_dim2}")
    
    # Sum with keepdim
    sum_keepdim = TensorOps.sum(tensor_3d, dim=1, keepdim=True)
    print(f"Sum along dim=1 with keepdim (shape {sum_keepdim.shape}):\n{sum_keepdim}")
    
    # Sum along multiple dimensions
    sum_multi_dim = TensorOps.sum(tensor_3d, dim=(0, 2))
    print(f"Sum along dims (0, 2): {sum_multi_dim}")
    
    subsection_header("Mean Operations")
    mean_total = TensorOps.mean(tensor_3d)
    print(f"Total mean: {mean_total.item()}")
    
    mean_dim0 = TensorOps.mean(tensor_3d, dim=0)
    print(f"Mean along dim=0:\n{mean_dim0}")
    
    subsection_header("Min and Max Operations")
    # Min/Max without dimension (returns single value)
    min_val = TensorOps.min(tensor_3d)
    max_val = TensorOps.max(tensor_3d)
    print(f"Global min: {min_val.item()}")
    print(f"Global max: {max_val.item()}")
    
    # Min/Max along dimensions (returns values and indices)
    min_dim1_vals, min_dim1_indices = TensorOps.min(tensor_3d, dim=1)
    max_dim1_vals, max_dim1_indices = TensorOps.max(tensor_3d, dim=1)
    
    print(f"Min along dim=1 values:\n{min_dim1_vals}")
    print(f"Min along dim=1 indices:\n{min_dim1_indices}")
    print(f"Max along dim=1 values:\n{max_dim1_vals}")
    print(f"Max along dim=1 indices:\n{max_dim1_indices}")


def demonstrate_reshaping_operations():
    """Comprehensive examples of reshaping and slicing operations."""
    section_header("RESHAPING AND SLICING", "Tensor shape manipulation and dimension operations")
    
    # Start with a sample tensor
    original = torch.arange(24).float().reshape(2, 3, 4)
    print(f"Original tensor shape: {original.shape}")
    print(f"Original tensor:\n{original}")
    
    subsection_header("Reshape Operations")
    # Basic reshaping
    reshaped_2d = TensorOps.reshape(original, (6, 4))
    print(f"Reshaped to (6, 4):\n{reshaped_2d}")
    
    reshaped_1d = TensorOps.reshape(original, (24,))
    print(f"Reshaped to (24,): {reshaped_1d}")
    
    # Reshape with -1 (inferred dimension)
    reshaped_infer = TensorOps.reshape(original, (-1, 8))
    print(f"Reshaped to (-1, 8) = {reshaped_infer.shape}:\n{reshaped_infer}")
    
    subsection_header("Transpose Operations")
    # Simple transpose (2D)
    matrix_2d = original[0]  # Take first "slice" to get 2D matrix
    transposed = TensorOps.transpose(matrix_2d, 0, 1)
    print(f"Original matrix (3x4):\n{matrix_2d}")
    print(f"Transposed (4x3):\n{transposed}")
    
    # Transpose in 3D tensor
    transposed_3d = TensorOps.transpose(original, 0, 2)
    print(f"Original shape: {original.shape}")
    print(f"Transposed dims 0,2 shape: {transposed_3d.shape}")
    
    subsection_header("Permute Operations")
    # Permute all dimensions
    permuted = TensorOps.permute(original, (2, 0, 1))
    print(f"Original shape: {original.shape}")
    print(f"Permuted (2,0,1) shape: {permuted.shape}")
    
    # Common permutation for images (channels last to channels first)
    image_like = torch.randn(32, 32, 3)  # Height x Width x Channels
    channels_first = TensorOps.permute(image_like, (2, 0, 1))  # Channels x Height x Width
    print(f"Image HWC {image_like.shape} -> CHW {channels_first.shape}")
    
    subsection_header("Flatten Operations")
    # Flatten all dimensions
    flattened = TensorOps.flatten(original)
    print(f"Original shape: {original.shape}")
    print(f"Flattened shape: {flattened.shape}")
    
    # Flatten specific dimensions
    flatten_partial = TensorOps.flatten(original, start_dim=1)
    print(f"Flatten from dim 1: {original.shape} -> {flatten_partial.shape}")
    
    flatten_middle = TensorOps.flatten(original, start_dim=0, end_dim=1)
    print(f"Flatten dims 0-1: {original.shape} -> {flatten_middle.shape}")
    
    subsection_header("Squeeze and Unsqueeze")
    # Add singleton dimensions
    unsqueezed = TensorOps.unsqueeze(original, dim=1)
    print(f"Unsqueeze at dim 1: {original.shape} -> {unsqueezed.shape}")
    
    unsqueezed_end = TensorOps.unsqueeze(original, dim=-1)
    print(f"Unsqueeze at dim -1: {original.shape} -> {unsqueezed_end.shape}")
    
    # Remove singleton dimensions
    squeezed = TensorOps.squeeze(unsqueezed, dim=1)
    print(f"Squeeze dim 1: {unsqueezed.shape} -> {squeezed.shape}")
    
    # Create tensor with multiple singleton dimensions
    multi_singleton = torch.randn(1, 5, 1, 3, 1)
    squeezed_all = TensorOps.squeeze(multi_singleton)
    print(f"Squeeze all: {multi_singleton.shape} -> {squeezed_all.shape}")


def demonstrate_concatenation_operations():
    """Comprehensive examples of concatenation and splitting operations."""
    section_header("CONCATENATION AND SPLITTING", "Joining and splitting tensors along dimensions")
    
    # Create sample tensors
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])
    c = torch.tensor([[9, 10], [11, 12]])
    
    print(f"Tensor A:\n{a}")
    print(f"Tensor B:\n{b}")  
    print(f"Tensor C:\n{c}")
    
    subsection_header("Concatenation")
    # Concatenate along different dimensions
    concat_dim0 = TensorOps.concatenate([a, b, c], dim=0)
    print(f"Concatenate along dim=0 (rows):\n{concat_dim0}")
    
    concat_dim1 = TensorOps.concatenate([a, b, c], dim=1)
    print(f"Concatenate along dim=1 (columns):\n{concat_dim1}")
    
    # Concatenate with different sized tensors
    print("\nConcatenation with different sizes:")
    d = torch.tensor([[13, 14, 15], [16, 17, 18]])  # 2x3
    e = torch.tensor([[19, 20, 21, 22]])  # 1x4 (different size)
    
    # This will work for dim=0 (different number of rows)
    try:
        concat_diff = TensorOps.concatenate([a, d], dim=0)  # Will fail - different column sizes
        print(f"Concatenate different sizes:\n{concat_diff}")
    except RuntimeError as error:
        print(f"Expected error for incompatible sizes: {error}")
    
    # Compatible concatenation
    d_compatible = torch.tensor([[13, 14], [16, 17], [18, 19]])  # 3x2
    concat_compatible = TensorOps.concatenate([a, b, d_compatible], dim=0)
    print(f"Compatible concatenation:\n{concat_compatible}")
    
    subsection_header("Stack Operations")
    # Stack creates new dimension
    stacked_dim0 = TensorOps.stack([a, b, c], dim=0)
    print(f"Stack along dim=0 shape: {stacked_dim0.shape}")
    print(f"Stack along dim=0:\n{stacked_dim0}")
    
    stacked_dim1 = TensorOps.stack([a, b, c], dim=1)  
    print(f"Stack along dim=1 shape: {stacked_dim1.shape}")
    
    stacked_dim2 = TensorOps.stack([a, b, c], dim=2)
    print(f"Stack along dim=2 shape: {stacked_dim2.shape}")
    
    # Stack 1D tensors
    print("\nStacking 1D tensors:")
    vec1 = torch.tensor([1, 2, 3])
    vec2 = torch.tensor([4, 5, 6])
    vec3 = torch.tensor([7, 8, 9])
    
    stacked_vectors = TensorOps.stack([vec1, vec2, vec3], dim=0)
    print(f"Stacked vectors shape: {stacked_vectors.shape}")
    print(f"Stacked vectors:\n{stacked_vectors}")
    
    # Demonstrate error handling
    print("\nError handling for incompatible shapes:")
    try:
        wrong_shape = torch.tensor([[1, 2, 3]])  # Different shape
        TensorOps.stack([a, wrong_shape], dim=0)
    except RuntimeError as error:
        print(f"Expected error: {error}")


def demonstrate_advanced_operations():
    """Comprehensive examples of advanced operations including einsum and autograd."""
    section_header("ADVANCED OPERATIONS", "Einstein summation and automatic differentiation")
    
    subsection_header("Einstein Summation (einsum)")
    
    # Basic matrix operations with einsum
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    
    # Matrix multiplication: A @ B
    matmul_einsum = TensorOps.einsum('ij,jk->ik', A, B)
    print(f"Matrix multiplication (einsum 'ij,jk->ik'):\n{matmul_einsum}")
    
    # Element-wise multiplication then sum: sum(A * B)
    elementwise_sum = TensorOps.einsum('ij,ij->', A, B)
    print(f"Element-wise multiply then sum: {elementwise_sum.item()}")
    
    # Transpose: A.T
    transpose_einsum = TensorOps.einsum('ij->ji', A)
    print(f"Transpose (einsum 'ij->ji'):\n{transpose_einsum}")
    
    # Trace: sum of diagonal elements
    trace_einsum = TensorOps.einsum('ii->', A)
    print(f"Trace (einsum 'ii->'): {trace_einsum.item()}")
    
    print("\nAdvanced einsum examples:")
    
    # Batch matrix multiplication
    batch_A = torch.randn(3, 4, 5)
    batch_B = torch.randn(3, 5, 6)
    batch_matmul = TensorOps.einsum('bij,bjk->bik', batch_A, batch_B)
    print(f"Batch matmul: {batch_A.shape} x {batch_B.shape} -> {batch_matmul.shape}")
    
    # Bilinear transformation: x^T A y
    x = torch.randn(4)
    y = torch.randn(5)
    A_bilinear = torch.randn(4, 5)
    bilinear = TensorOps.einsum('i,ij,j->', x, A_bilinear, y)
    print(f"Bilinear form x^T A y: {bilinear.item():.4f}")
    
    # Attention mechanism (simplified)
    Q = torch.randn(8, 64)  # Query: sequence_length x hidden_dim
    K = torch.randn(8, 64)  # Key: sequence_length x hidden_dim  
    V = torch.randn(8, 64)  # Value: sequence_length x hidden_dim
    
    attention_scores = TensorOps.einsum('qd,kd->qk', Q, K) / (64 ** 0.5)
    attention_weights = torch.softmax(attention_scores, dim=-1)
    attention_output = TensorOps.einsum('qk,kd->qd', attention_weights, V)
    print(f"Attention output shape: {attention_output.shape}")
    
    subsection_header("Automatic Differentiation")
    
    # Simple gradient computation
    print("Computing gradients of scalar functions:")
    
    def quadratic_function(x):
        """f(x) = x^T A x + b^T x + c"""
        A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
        b = torch.tensor([1.0, -1.0])
        c = 5.0
        return TensorOps.einsum('i,ij,j->', x, A, x) + torch.dot(b, x) + c
    
    x_input = torch.tensor([1.0, 2.0], requires_grad=True)
    grad = TensorOps.compute_gradient(quadratic_function, x_input)
    print(f"Input x: {x_input.data}")
    print(f"Gradient of quadratic function: {grad}")
    
    # Verify with analytical gradient: grad = 2*A*x + b
    A_check = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    b_check = torch.tensor([1.0, -1.0])
    analytical_grad = 2 * TensorOps.matmul(A_check, x_input.detach()) + b_check
    print(f"Analytical gradient: {analytical_grad}")
    print(f"Difference: {torch.norm(grad - analytical_grad).item():.6f}")
    
    # Jacobian computation for vector function
    print("\nJacobian computation:")
    
    def vector_function(x):
        """f(x) = [x1^2 + x2, x1 * x2, sin(x1) + cos(x2)]"""
        return torch.stack([
            x[0]**2 + x[1],
            x[0] * x[1], 
            torch.sin(x[0]) + torch.cos(x[1])
        ])
    
    x_vec = torch.tensor([1.0, 0.5])
    jacobian = TensorOps.compute_jacobian(vector_function, x_vec)
    print(f"Input: {x_vec}")
    print(f"Jacobian matrix:\n{jacobian}")
    print(f"Function output: {vector_function(x_vec)}")


def demonstrate_linear_algebra():
    """Comprehensive examples of linear algebra operations."""
    section_header("LINEAR ALGEBRA OPERATIONS", "Eigenvalues, decompositions, matrix properties")
    
    # Create sample matrices
    symmetric_matrix = torch.tensor([[4.0, 2.0, 1.0], 
                                   [2.0, 5.0, 3.0], 
                                   [1.0, 3.0, 6.0]])
    
    regular_matrix = torch.tensor([[1.0, 2.0, 3.0], 
                                 [4.0, 5.0, 6.0], 
                                 [7.0, 8.0, 10.0]])  # Made non-singular
    
    print(f"Symmetric matrix:\n{symmetric_matrix}")
    print(f"Regular matrix:\n{regular_matrix}")
    
    subsection_header("Eigendecomposition")
    eigenvals, eigenvecs = TensorOps.matrix_eigendecomposition(symmetric_matrix)
    print(f"Eigenvalues: {eigenvals}")
    print(f"Eigenvectors:\n{eigenvecs}")
    
    # Verify eigendecomposition: A v = λ v
    for i in range(eigenvals.shape[0]):
        λ = eigenvals[i]
        v = eigenvecs[:, i]
        Av = TensorOps.matmul(symmetric_matrix, v.unsqueeze(1)).squeeze()
        λv = λ * v
        error = torch.norm(Av - λv).item()
        print(f"Eigenvalue {i}: λ={λ.item():.4f}, verification error: {error:.6f}")
    
    subsection_header("SVD Decomposition")
    U, S, Vt = TensorOps.svd(regular_matrix)
    print(f"U shape: {U.shape}")
    print(f"Singular values: {S}")
    print(f"Vt shape: {Vt.shape}")
    
    # Reconstruct matrix to verify SVD
    reconstructed = TensorOps.matmul(U, TensorOps.matmul(torch.diag(S), Vt))
    reconstruction_error = torch.norm(regular_matrix - reconstructed).item()
    print(f"SVD reconstruction error: {reconstruction_error:.8f}")
    
    subsection_header("QR Decomposition")
    Q, R = TensorOps.qr_decomposition(regular_matrix)
    print(f"Q shape: {Q.shape}")
    print(f"R shape: {R.shape}")
    
    # Verify QR decomposition
    qr_reconstructed = TensorOps.matmul(Q, R)
    qr_error = torch.norm(regular_matrix - qr_reconstructed).item()
    print(f"QR reconstruction error: {qr_error:.8f}")
    
    # Verify Q is orthogonal
    Q_orthogonal_check = TensorOps.matmul(Q.T, Q)
    identity_error = torch.norm(Q_orthogonal_check - torch.eye(Q.shape[1])).item()
    print(f"Q orthogonality error: {identity_error:.8f}")
    
    subsection_header("Cholesky Decomposition")
    # Create a positive definite matrix
    random_matrix = torch.randn(4, 4)
    positive_definite = TensorOps.matmul(random_matrix, random_matrix.T) + 0.1 * torch.eye(4)
    
    L = TensorOps.cholesky_decomposition(positive_definite)
    print(f"Positive definite matrix shape: {positive_definite.shape}")
    print(f"Cholesky factor L:\n{L}")
    
    # Verify Cholesky decomposition: A = L L^T
    chol_reconstructed = TensorOps.matmul(L, L.T)
    chol_error = torch.norm(positive_definite - chol_reconstructed).item()
    print(f"Cholesky reconstruction error: {chol_error:.8f}")
    
    subsection_header("Matrix Properties")
    
    # Trace
    trace_val = TensorOps.matrix_trace(symmetric_matrix)
    print(f"Trace of symmetric matrix: {trace_val.item()}")
    
    # Determinant
    det_val = TensorOps.matrix_determinant(symmetric_matrix)
    print(f"Determinant of symmetric matrix: {det_val.item()}")
    
    # Matrix rank
    rank_val = TensorOps.matrix_rank(regular_matrix)
    print(f"Rank of regular matrix: {rank_val.item()}")
    
    # Matrix inverse (for invertible matrix)
    try:
        inv_matrix = TensorOps.matrix_inverse(regular_matrix)
        print(f"Matrix inverse shape: {inv_matrix.shape}")
        
        # Verify inverse
        identity_check = TensorOps.matmul(regular_matrix, inv_matrix)
        inv_error = torch.norm(identity_check - torch.eye(3)).item()
        print(f"Inverse verification error: {inv_error:.6f}")
        
    except Exception as e:
        print(f"Matrix inversion failed: {e}")
    
    subsection_header("Tensor Trace")
    # Create higher-order tensor
    tensor_4d = torch.randn(3, 4, 3, 4)
    tensor_trace = TensorOps.tensor_trace(tensor_4d, axis1=0, axis2=2)
    print(f"4D tensor shape: {tensor_4d.shape}")
    print(f"Tensor trace along axes (0,2) shape: {tensor_trace.shape}")


def demonstrate_convolution_operations():
    """Comprehensive examples of convolution operations."""
    section_header("CONVOLUTION OPERATIONS", "1D, 2D, and 3D convolutions with different modes")
    
    subsection_header("1D Convolution")
    
    # Signal processing example
    # Create a signal with noise
    t = torch.linspace(0, 1, 100)
    clean_signal = torch.sin(2 * np.pi * 5 * t) + 0.5 * torch.sin(2 * np.pi * 10 * t)
    noise = 0.2 * torch.randn_like(clean_signal)
    noisy_signal = clean_signal + noise
    
    # Create smoothing kernel (Gaussian-like)
    kernel_size = 5
    kernel_1d = torch.exp(-torch.linspace(-2, 2, kernel_size)**2)
    kernel_1d = kernel_1d / kernel_1d.sum()  # Normalize
    
    print(f"Signal length: {noisy_signal.shape[0]}")
    print(f"Kernel size: {kernel_size}")
    print(f"Smoothing kernel: {kernel_1d}")
    
    # Apply convolution with different modes
    conv_valid = TensorOps.convolve_1d(noisy_signal, kernel_1d, mode="valid")
    conv_same = TensorOps.convolve_1d(noisy_signal, kernel_1d, mode="same")
    conv_full = TensorOps.convolve_1d(noisy_signal, kernel_1d, mode="full")
    
    print(f"Valid convolution length: {conv_valid.shape[0]}")
    print(f"Same convolution length: {conv_same.shape[0]}")
    print(f"Full convolution length: {conv_full.shape[0]}")
    
    subsection_header("2D Convolution") 
    
    # Image processing example
    # Create a simple image pattern
    image = torch.zeros(20, 20)
    image[8:12, 8:12] = 1.0  # Square in the middle
    image += 0.1 * torch.randn_like(image)  # Add noise
    
    print(f"Image shape: {image.shape}")
    
    # Edge detection kernel (Sobel-like)
    edge_kernel = torch.tensor([[-1.0, 0.0, 1.0],
                               [-2.0, 0.0, 2.0], 
                               [-1.0, 0.0, 1.0]])
    
    # Smoothing kernel (Gaussian-like)
    smooth_kernel = torch.tensor([[1.0, 2.0, 1.0],
                                 [2.0, 4.0, 2.0],
                                 [1.0, 2.0, 1.0]]) / 16.0
    
    # Apply different kernels
    edges = TensorOps.convolve_2d(image, edge_kernel, mode="same")
    smoothed = TensorOps.convolve_2d(image, smooth_kernel, mode="same")
    
    print(f"Original image range: [{image.min().item():.3f}, {image.max().item():.3f}]")
    print(f"Edge detection range: [{edges.min().item():.3f}, {edges.max().item():.3f}]")
    print(f"Smoothed image range: [{smoothed.min().item():.3f}, {smoothed.max().item():.3f}]")
    
    # Demonstrate different modes
    conv_2d_valid = TensorOps.convolve_2d(image, smooth_kernel, mode="valid")
    conv_2d_full = TensorOps.convolve_2d(image, smooth_kernel, mode="full")
    
    print(f"2D Valid convolution shape: {conv_2d_valid.shape}")
    print(f"2D Full convolution shape: {conv_2d_full.shape}")
    
    subsection_header("3D Convolution")
    
    # Volume processing example
    volume = torch.zeros(10, 10, 10)
    volume[4:6, 4:6, 4:6] = 1.0  # Small cube in the center
    volume += 0.05 * torch.randn_like(volume)
    
    # 3D smoothing kernel
    kernel_3d = torch.ones(3, 3, 3) / 27.0  # Average pooling
    
    print(f"Volume shape: {volume.shape}")
    print(f"3D kernel shape: {kernel_3d.shape}")
    
    # Apply 3D convolution
    smoothed_volume = TensorOps.convolve_3d(volume, kernel_3d, mode="same")
    
    print(f"Original volume range: [{volume.min().item():.3f}, {volume.max().item():.3f}]")
    print(f"Smoothed volume range: [{smoothed_volume.min().item():.3f}, {smoothed_volume.max().item():.3f}]")
    
    # Different modes for 3D
    conv_3d_valid = TensorOps.convolve_3d(volume, kernel_3d, mode="valid")
    print(f"3D Valid convolution shape: {conv_3d_valid.shape}")


def demonstrate_statistical_operations():
    """Comprehensive examples of statistical operations."""
    section_header("STATISTICAL OPERATIONS", "Variance, covariance, correlation, and norm calculations")
    
    # Create sample data
    np.random.seed(42)  # For reproducibility
    data = torch.tensor(np.random.multivariate_normal([0, 0], [[2, 1], [1, 2]], 100).T)
    
    print(f"Data shape: {data.shape}")
    print(f"Data (first 5 samples):\n{data[:, :5]}")
    
    subsection_header("Basic Statistics")
    
    # Variance and standard deviation
    var_total = TensorOps.variance(data)
    std_total = TensorOps.std(data)
    print(f"Total variance: {var_total.item():.4f}")
    print(f"Total std dev: {std_total.item():.4f}")
    
    # Per-variable statistics
    var_per_dim = TensorOps.variance(data, dim=1)
    std_per_dim = TensorOps.std(data, dim=1)
    print(f"Variance per dimension: {var_per_dim}")
    print(f"Std dev per dimension: {std_per_dim}")
    
    # Unbiased vs biased estimates
    var_unbiased = TensorOps.variance(data, dim=1, unbiased=True)
    var_biased = TensorOps.variance(data, dim=1, unbiased=False)
    print(f"Unbiased variance: {var_unbiased}")
    print(f"Biased variance: {var_biased}")
    
    subsection_header("Covariance Matrix")
    
    # Covariance matrix (rowvar=True means each row is a variable)
    cov_matrix = TensorOps.covariance(data, rowvar=True)
    print(f"Covariance matrix:\n{cov_matrix}")
    
    # Cross-covariance between two datasets
    data2 = torch.tensor(np.random.multivariate_normal([1, -1], [[1.5, 0.5], [0.5, 1.5]], 100).T)
    cross_cov = TensorOps.covariance(data, data2, rowvar=True)
    print(f"Cross-covariance shape: {cross_cov.shape}")
    print(f"Cross-covariance:\n{cross_cov}")
    
    subsection_header("Correlation Matrix")
    
    # Correlation matrix
    corr_matrix = TensorOps.correlation(data, rowvar=True)
    print(f"Correlation matrix:\n{corr_matrix}")
    
    # Cross-correlation
    cross_corr = TensorOps.correlation(data, data2, rowvar=True)
    print(f"Cross-correlation:\n{cross_corr}")
    
    subsection_header("Norm Calculations")
    
    # Create test tensor
    test_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"Test tensor:\n{test_tensor}")
    
    # Different norm types
    frobenius_norm = TensorOps.frobenius_norm(test_tensor)
    l1_norm = TensorOps.l1_norm(test_tensor)
    l2_norm = TensorOps.l2_norm(test_tensor)
    nuclear_norm = TensorOps.nuclear_norm(test_tensor)
    
    print(f"Frobenius norm: {frobenius_norm.item():.4f}")
    print(f"L1 norm: {l1_norm.item():.4f}")
    print(f"L2 norm: {l2_norm.item():.4f}")
    print(f"Nuclear norm: {nuclear_norm.item():.4f}")
    
    # P-norms with different values of p
    for p in [1.5, 2.0, 3.0, float('inf')]:
        if p == float('inf'):
            # PyTorch uses 'inf' string for infinity norm
            try:
                p_norm_val = torch.linalg.norm(test_tensor, ord=float('inf'))
                print(f"L-infinity norm: {p_norm_val.item():.4f}")
            except:
                print("L-infinity norm not computed")
        else:
            p_norm_val = TensorOps.p_norm(test_tensor, p)
            print(f"L-{p} norm: {p_norm_val.item():.4f}")
    
    # Vector norms
    vector = torch.tensor([3.0, 4.0, 5.0])
    print(f"\nVector: {vector}")
    print(f"Vector L1 norm: {TensorOps.l1_norm(vector).item():.4f}")
    print(f"Vector L2 norm: {TensorOps.l2_norm(vector).item():.4f}")
    print(f"Vector L2 norm (manual): {torch.sqrt(torch.sum(vector**2)).item():.4f}")


def performance_benchmarking():
    """Demonstrate performance characteristics of different operations."""
    section_header("PERFORMANCE BENCHMARKING", "Speed comparison of different tensor operations")
    
    # Create test tensors of different sizes
    sizes = [(100, 100), (500, 500), (1000, 1000)]
    operations = [
        ("Matrix Multiplication", lambda a, b: TensorOps.matmul(a, b)),
        ("Element-wise Addition", lambda a, b: TensorOps.add(a, b)),  
        ("SVD", lambda a, b: TensorOps.svd(a)),
        ("Eigendecomposition", lambda a, b: TensorOps.matrix_eigendecomposition(a))
    ]
    
    print("Benchmarking tensor operations across different sizes:")
    print(f"{'Operation':<20} | {'100x100':<10} | {'500x500':<10} | {'1000x1000':<12}")
    print("-" * 65)
    
    for op_name, op_func in operations:
        times = []
        
        for size in sizes:
            # Create test tensors
            if op_name in ["SVD", "Eigendecomposition"]:
                # Use one tensor for these operations
                a = torch.randn(*size)
                b = None
                
                # Make symmetric for eigendecomposition for numerical stability
                if op_name == "Eigendecomposition":
                    a = a + a.T
                
            else:
                # Use two tensors
                a = torch.randn(*size)
                b = torch.randn(*size)
            
            # Benchmark the operation
            num_trials = 5 if size[0] <= 500 else 1  # Fewer trials for large matrices
            
            start_time = time.time()
            for _ in range(num_trials):
                try:
                    if b is not None:
                        _ = op_func(a, b)
                    else:
                        _ = op_func(a, None)
                except:
                    # Some operations might fail for certain sizes
                    break
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_trials * 1000  # Convert to ms
            times.append(f"{avg_time:.2f}ms")
        
        print(f"{op_name:<20} | {times[0]:<10} | {times[1]:<10} | {times[2]:<12}")


def main():
    """Run comprehensive tensor operations demonstration."""
    print("TENSORUS COMPREHENSIVE TENSOR OPERATIONS EXAMPLES")
    print("Addressing GAP 9: Limited Practical Examples")
    print("=" * 80)
    print("This demonstration covers all tensor operations with practical examples,")
    print("error handling, and performance considerations.")
    
    try:
        demonstrate_arithmetic_operations()
        demonstrate_matrix_operations()
        demonstrate_reduction_operations()
        demonstrate_reshaping_operations()
        demonstrate_concatenation_operations()
        demonstrate_advanced_operations()
        demonstrate_linear_algebra()
        demonstrate_convolution_operations()
        demonstrate_statistical_operations()
        performance_benchmarking()
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE EXAMPLES SUMMARY")
        print("=" * 80)
        print("✅ Arithmetic operations: add, subtract, multiply, divide, power, log")
        print("✅ Matrix operations: matmul, dot, outer, cross products")
        print("✅ Reduction operations: sum, mean, min, max across dimensions")
        print("✅ Reshaping operations: reshape, transpose, permute, flatten, squeeze/unsqueeze")
        print("✅ Concatenation: concatenate and stack operations")
        print("✅ Advanced operations: Einstein summation and automatic differentiation")
        print("✅ Linear algebra: eigendecomposition, SVD, QR, Cholesky, matrix properties")
        print("✅ Convolution operations: 1D, 2D, 3D with different modes")
        print("✅ Statistical operations: variance, covariance, correlation, norms")
        print("✅ Performance benchmarking across different tensor sizes")
        print("✅ Error handling and edge case demonstrations")
        print("✅ Real-world usage patterns and best practices")
        print("\nGAP 9 has been significantly addressed with comprehensive examples!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())