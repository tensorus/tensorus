import unittest
import numpy as np
import sys

# Add the root directory to sys.path to allow importing from tensorus
sys.path.append('.')

# Import only NumPy-based functions from tensorus.tensor_operations
from tensorus.tensor_operations import (
    add_tensors,
    subtract_tensors,
    multiply_tensors_elementwise,
    divide_tensors_elementwise,
    scalar_multiply_tensor,
    scalar_add_tensor,
    reshape_tensor,
    transpose_tensor,
    tensor_dot_product,
    outer_product,
    einstein_summation,
    frobenius_norm,
    l1_norm,
    matrix_eigendecomposition,
    matrix_trace,
    tensor_trace,
    tensor_mean,
    tensor_variance,
    matrix_covariance,
    matrix_correlation
    # PyTorch, SciPy, and TensorLy functions are intentionally omitted
)

class TestNumPyTensorOperations(unittest.TestCase):

    def setUp(self):
        """Set up some common tensors for testing."""
        self.A_2x2 = np.array([[1., 2.], [3., 4.]])
        self.B_2x2 = np.array([[5., 6.], [7., 8.]])
        self.A_3x2 = np.array([[1., 2.], [3., 4.], [5., 6.]])
        self.B_2x3 = np.array([[1., 2., 3.], [4., 5., 6.]])
        self.vector1 = np.array([1., 2., 3.])
        self.vector2 = np.array([4., 5., 6.])
        self.tensor_3d = np.arange(24).reshape((2, 3, 4))

    # 4. Test Basic Operations (NumPy only)
    def test_tensor_addition(self):
        expected = np.array([[6., 8.], [10., 12.]])
        result = add_tensors(self.A_2x2, self.B_2x2)
        self.assertTrue(np.allclose(result, expected))
        with self.assertRaises(ValueError):
            add_tensors(self.A_2x2, self.A_3x2)

    def test_tensor_subtraction(self):
        expected = np.array([[-4., -4.], [-4., -4.]])
        result = subtract_tensors(self.A_2x2, self.B_2x2)
        self.assertTrue(np.allclose(result, expected))
        with self.assertRaises(ValueError):
            subtract_tensors(self.A_2x2, self.A_3x2)

    def test_elementwise_multiplication(self):
        expected = np.array([[5., 12.], [21., 32.]])
        result = multiply_tensors_elementwise(self.A_2x2, self.B_2x2)
        self.assertTrue(np.allclose(result, expected))
        with self.assertRaises(ValueError):
            multiply_tensors_elementwise(self.A_2x2, self.A_3x2)

    def test_elementwise_division(self):
        C_2x2 = np.array([[2., 2.], [2., 2.]])
        expected = np.array([[0.5, 1.], [1.5, 2.]])
        result = divide_tensors_elementwise(self.A_2x2, C_2x2)
        self.assertTrue(np.allclose(result, expected))
        with self.assertRaises(ValueError):
            divide_tensors_elementwise(self.A_2x2, self.A_3x2)
        with self.assertRaises(ZeroDivisionError):
            divide_tensors_elementwise(self.A_2x2, np.array([[1., 0.], [1., 1.]]))

    def test_scalar_multiplication(self):
        scalar = 3.
        expected = np.array([[3., 6.], [9., 12.]])
        result = scalar_multiply_tensor(scalar, self.A_2x2)
        self.assertTrue(np.allclose(result, expected))

    def test_scalar_addition(self):
        scalar = 3.
        expected = np.array([[4., 5.], [6., 7.]])
        result = scalar_add_tensor(self.A_2x2, scalar)
        self.assertTrue(np.allclose(result, expected))

    # 5. Test Shape Operations (NumPy only)
    def test_tensor_reshape(self):
        tensor = np.arange(6) # 1D array [0, 1, 2, 3, 4, 5]
        new_shape = (2, 3)
        expected = np.array([[0, 1, 2], [3, 4, 5]])
        result = reshape_tensor(tensor, new_shape)
        self.assertTrue(np.allclose(result, expected))
        with self.assertRaises(ValueError): # Product of new shape dimensions must be equal to old
            reshape_tensor(tensor, (2, 4))

    def test_tensor_transpose(self):
        expected = np.array([[1., 3.], [2., 4.]])
        result = transpose_tensor(self.A_2x2) # Default transpose (reverse axes)
        self.assertTrue(np.allclose(result, expected))

        tensor_3d_original = np.arange(24).reshape((2,3,4))
        # Transpose axes (0,1,2) to (1,0,2)
        expected_transposed = np.transpose(tensor_3d_original, (1,0,2))
        result_transposed = transpose_tensor(tensor_3d_original, (1,0,2))
        self.assertTrue(np.allclose(result_transposed, expected_transposed))

        with self.assertRaises(ValueError): # Invalid permutation
            transpose_tensor(self.A_2x2, (0, 0))

    # 6. Test Tensor Products (NumPy only)
    def test_tensor_dot_product(self):
        # Matrix multiplication
        expected = np.dot(self.A_3x2, self.B_2x3) # (3x2) . (2x3) -> (3x3)
        result = tensor_dot_product(self.A_3x2, self.B_2x3, axes=([1],[0]))
        self.assertTrue(np.allclose(result, expected))

        # More complex tensordot
        # A is (2,3,4), B is (3,4,2)
        # Contract axis 1 of A with axis 0 of B, and axis 2 of A with axis 1 of B
        # Result should be (2,2)
        A = np.random.rand(2,3,4)
        B = np.random.rand(3,4,2)
        expected_td = np.tensordot(A,B, axes=([1,2],[0,1]))
        result_td = tensor_dot_product(A,B, axes=([1,2],[0,1]))
        self.assertTrue(np.allclose(result_td, expected_td))

        with self.assertRaises(ValueError): # Incompatible axes for dot product
            tensor_dot_product(self.A_3x2, self.B_2x3, axes=([0],[0]))

    def test_outer_product(self):
        # For vectors
        expected_vec = np.outer(self.vector1, self.vector2)
        result_vec = outer_product(self.vector1, self.vector2)
        self.assertTrue(np.allclose(result_vec, expected_vec))

        # For higher order tensors (equivalent to tensordot with axes=0)
        expected_tensor = np.tensordot(self.A_2x2, self.B_2x2, axes=0)
        result_tensor = outer_product(self.A_2x2, self.B_2x2)
        self.assertTrue(np.allclose(result_tensor, expected_tensor))


    def test_tensor_contraction_einsum(self):
        # Matrix multiplication: ij,jk->ik
        result_matmul = einstein_summation('ij,jk->ik', self.A_3x2, self.B_2x3)
        expected_matmul = np.matmul(self.A_3x2, self.B_2x3)
        self.assertTrue(np.allclose(result_matmul, expected_matmul))

        # Trace: ii->
        result_trace = einstein_summation('ii->', self.A_2x2)
        expected_trace = np.trace(self.A_2x2)
        self.assertTrue(np.allclose(result_trace, expected_trace))

        # Batch matrix multiplication: bij,bjk->bik
        A_batch = np.random.rand(3,2,4) # 3 batches of 2x4 matrices
        B_batch = np.random.rand(3,4,5) # 3 batches of 4x5 matrices
        result_batch_matmul = einstein_summation('bij,bjk->bik', A_batch, B_batch)
        expected_batch_matmul = np.matmul(A_batch, B_batch)
        self.assertTrue(np.allclose(result_batch_matmul, expected_batch_matmul))

        with self.assertRaises(ValueError): # Invalid einsum string due to mismatched inner dimension for 'j'
            # A_3x2 has shape (3,2). For 'ij,jk->ik':
            # First operand A_3x2: j is dimension 1 (size 2)
            # Second operand A_3x2: j is dimension 0 (size 3)
            # These (2 and 3) do not match.
            einstein_summation('ij,jk->ik', self.A_3x2, self.A_3x2)

    # 7. Test Norms and Metrics (NumPy only)
    def test_frobenius_norm(self):
        expected = np.linalg.norm(self.A_2x2, 'fro')
        result = frobenius_norm(self.A_2x2)
        self.assertTrue(np.allclose(result, expected))

    def test_l1_norm(self):
        expected = np.sum(np.abs(self.A_2x2))
        result = l1_norm(self.A_2x2)
        self.assertTrue(np.allclose(result, expected))

    # 8. Test Eigenvalues (NumPy only)
    def test_matrix_eigendecomposition(self):
        # Test with a simple symmetric matrix
        sym_matrix = np.array([[2., 1.], [1., 2.]])
        eigenvalues, eigenvectors = matrix_eigendecomposition(sym_matrix)

        # Verify A @ v = lambda * v for each eigenvector/value pair
        for i in range(len(eigenvalues)):
            lambda_v = eigenvalues[i] * eigenvectors[:, i]
            A_v = np.dot(sym_matrix, eigenvectors[:, i])
            self.assertTrue(np.allclose(A_v, lambda_v))

        with self.assertRaises(TypeError):
            matrix_eigendecomposition([[1,2],[3,4]]) # Not a numpy array
        with self.assertRaises(ValueError): # Not 2D
            matrix_eigendecomposition(np.array([1, 2, 3]))
        with self.assertRaises(ValueError): # Not square
            matrix_eigendecomposition(self.A_3x2)

    # 9. Test Trace Operations (NumPy only)
    def test_matrix_trace(self):
        expected = np.trace(self.A_2x2) # 1 + 4 = 5
        result = matrix_trace(self.A_2x2)
        self.assertEqual(result, expected)

        # Test with non-square 2D matrix (numpy.trace allows this)
        non_square_2d = np.array([[1,2,3],[4,5,6]])
        expected_ns = np.trace(non_square_2d) # 1 + 5 = 6
        result_ns = matrix_trace(non_square_2d)
        self.assertEqual(result_ns, expected_ns)

        with self.assertRaises(TypeError):
            matrix_trace([[1,2],[3,4]])
        with self.assertRaises(ValueError): # Not 2D
            matrix_trace(self.tensor_3d)

    def test_tensor_trace(self):
        # Trace of a 3D tensor along axes 0 and 1 (summing elements where indices for axis 0 and 1 are equal)
        # tensor_3d is (2,3,4)
        # trace(tensor_3d, axis1=0, axis2=1) means sum where i=j for T[i,j,k]
        # This requires dimensions of axis1 and axis2 to be equal.
        # Let's use a tensor where this is possible, e.g., (2,2,4)
        tensor_224 = np.arange(16).reshape((2,2,4))
        # Expected: sum tensor_224[0,0,:] + tensor_224[1,1,:]
        # tensor_224[0,0,:] = [0,1,2,3]
        # tensor_224[1,1,:] = [12,13,14,15]
        # Sum = [12,14,16,18]
        expected = np.trace(tensor_224, axis1=0, axis2=1)
        result = tensor_trace(tensor_224, axis1=0, axis2=1)
        self.assertTrue(np.allclose(result, expected))

        # Test case where axis1 and axis2 dimensions are different (NumPy handles this by using the smaller dimension)
        # self.tensor_3d has shape (2,3,4)
        # trace along axis1=0 (size 2) and axis2=1 (size 3)
        # Expected: sum of tensor_3d[0,0,:] and tensor_3d[1,1,:]
        # tensor_3d[0,0,:] = [0, 1, 2, 3]
        # tensor_3d[1,1,:] = [16, 17, 18, 19]  (Corrected from previous manual calculation)
        # Sum = [0+16, 1+17, 2+18, 3+19] = [16, 18, 20, 22]
        expected_trace_3d_01 = np.array([16., 18., 20., 22.])
        result_trace_3d_01 = tensor_trace(self.tensor_3d, axis1=0, axis2=1)
        self.assertTrue(np.allclose(result_trace_3d_01, expected_trace_3d_01))

        with self.assertRaises(TypeError):
            tensor_trace([[[1]]])
        with self.assertRaises(ValueError): # Axes out of bounds
            tensor_trace(self.tensor_3d, axis1=0, axis2=3)
        # The following case was previously expected to raise ValueError, but np.trace handles it:
        # with self.assertRaises(ValueError): # Dimensions for axes 0 and 1 (2 and 3) do not match
        #     tensor_trace(self.tensor_3d, axis1=0, axis2=1)


    # 10. Test Statistical Operations (NumPy only)
    def test_tensor_mean(self):
        self.assertTrue(np.allclose(tensor_mean(self.A_2x2), np.mean(self.A_2x2)))
        self.assertTrue(np.allclose(tensor_mean(self.A_3x2, axis=0), np.mean(self.A_3x2, axis=0)))
        self.assertTrue(np.allclose(tensor_mean(self.A_3x2, axis=1), np.mean(self.A_3x2, axis=1)))
        with self.assertRaises(TypeError):
            tensor_mean([[1,2],[3,4]])

    def test_tensor_variance(self):
        self.assertTrue(np.allclose(tensor_variance(self.A_2x2), np.var(self.A_2x2)))
        self.assertTrue(np.allclose(tensor_variance(self.A_3x2, axis=0), np.var(self.A_3x2, axis=0)))
        self.assertTrue(np.allclose(tensor_variance(self.A_3x2, axis=1, ddof=1), np.var(self.A_3x2, axis=1, ddof=1)))
        with self.assertRaises(TypeError):
            tensor_variance([[1,2],[3,4]])

    def test_matrix_covariance(self):
        # Assuming rows are variables, observations are columns
        X = np.array([[1, 2, 3], [4, 5, 6]]) # 2 variables, 3 observations
        expected_cov_X_rowvar_true = np.cov(X, rowvar=True)
        result_cov_X_rowvar_true = matrix_covariance(X, rowvar=True)
        self.assertTrue(np.allclose(result_cov_X_rowvar_true, expected_cov_X_rowvar_true))

        # Assuming columns are variables, rows are observations
        Xt = X.T # 3 observations, 2 variables
        expected_cov_Xt_rowvar_false = np.cov(Xt, rowvar=False)
        result_cov_Xt_rowvar_false = matrix_covariance(Xt, rowvar=False)
        self.assertTrue(np.allclose(result_cov_Xt_rowvar_false, expected_cov_Xt_rowvar_false))

        # Test with two matrices
        Y = np.array([[7, 8, 9], [10, 11, 12]])
        expected_cov_XY = np.cov(X, Y, rowvar=True)
        result_cov_XY = matrix_covariance(X, Y, rowvar=True)
        self.assertTrue(np.allclose(result_cov_XY, expected_cov_XY))

        with self.assertRaises(TypeError):
            matrix_covariance([[1,2],[3,4]])
        with self.assertRaises(TypeError):
            matrix_covariance(X, [[1,2],[3,4]])


    def test_matrix_correlation(self):
        X = np.array([[1, 2, 30], [-40, 5, 6]]) # 2 variables, 3 observations
        expected_corr_X_rowvar_true = np.corrcoef(X, rowvar=True)
        result_corr_X_rowvar_true = matrix_correlation(X, rowvar=True)
        self.assertTrue(np.allclose(result_corr_X_rowvar_true, expected_corr_X_rowvar_true))

        Xt = X.T
        expected_corr_Xt_rowvar_false = np.corrcoef(Xt, rowvar=False)
        result_corr_Xt_rowvar_false = matrix_correlation(Xt, rowvar=False)
        self.assertTrue(np.allclose(result_corr_Xt_rowvar_false, expected_corr_Xt_rowvar_false))

        # Test with two matrices
        Y = np.array([[7, 80, 9], [-10, 11, 1.2]])
        expected_corr_XY = np.corrcoef(X, y=Y, rowvar=True) # np.corrcoef uses 'y'
        result_corr_XY = matrix_correlation(X, matrix_Y=Y, rowvar=True)
        self.assertTrue(np.allclose(result_corr_XY, expected_corr_XY))

        with self.assertRaises(TypeError):
            matrix_correlation([[1,2],[3,4]])
        with self.assertRaises(TypeError):
            matrix_correlation(X, [[1,2],[3,4]])


if __name__ == '__main__':
    # Create 'test' directory if it doesn't exist (not strictly necessary for running script directly, but good practice)
    import os
    if not os.path.exists('test'):
        os.makedirs('test')
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
