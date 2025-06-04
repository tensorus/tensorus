import numpy as np

def add_tensors(tensor1, tensor2):
  """
  Adds two tensors element-wise.

  Args:
    tensor1: The first tensor (numpy array).
    tensor2: The second tensor (numpy array).

  Returns:
    A new tensor containing the element-wise sum of tensor1 and tensor2.

  Raises:
    ValueError: If the shapes of the input tensors are not compatible.
  """
  if tensor1.shape != tensor2.shape:
    raise ValueError("Input tensors must have the same shape for addition.")
  return np.add(tensor1, tensor2)

def subtract_tensors(tensor1, tensor2):
  """
  Subtracts the second tensor from the first, element-wise.

  Args:
    tensor1: The first tensor (numpy array).
    tensor2: The second tensor (numpy array).

  Returns:
    A new tensor containing the element-wise difference of tensor1 and tensor2.

  Raises:
    ValueError: If the shapes of the input tensors are not compatible.
  """
  if tensor1.shape != tensor2.shape:
    raise ValueError("Input tensors must have the same shape for subtraction.")
  return np.subtract(tensor1, tensor2)

def multiply_tensors_elementwise(tensor1, tensor2):
  """
  Multiplies two tensors element-wise (Hadamard product).

  Args:
    tensor1: The first tensor (numpy array).
    tensor2: The second tensor (numpy array).

  Returns:
    A new tensor containing the element-wise product of tensor1 and tensor2.

  Raises:
    ValueError: If the shapes of the input tensors are not compatible.
  """
  if tensor1.shape != tensor2.shape:
    raise ValueError("Input tensors must have the same shape for element-wise multiplication.")
  return np.multiply(tensor1, tensor2)

def divide_tensors_elementwise(tensor1, tensor2):
  """
  Divides the first tensor by the second, element-wise.

  Args:
    tensor1: The first tensor (numpy array).
    tensor2: The second tensor (numpy array).

  Returns:
    A new tensor containing the element-wise division of tensor1 by tensor2.

  Raises:
    ValueError: If the shapes of the input tensors are not compatible.
    ZeroDivisionError: If tensor2 contains any zero elements.
  """
  if tensor1.shape != tensor2.shape:
    raise ValueError("Input tensors must have the same shape for element-wise division.")
  if np.any(tensor2 == 0):
    raise ZeroDivisionError("Division by zero is not allowed.")
  return np.divide(tensor1, tensor2)

def scalar_multiply_tensor(scalar, tensor):
  """
  Multiplies a tensor by a scalar.

  Args:
    scalar: The scalar value.
    tensor: The tensor (numpy array).

  Returns:
    A new tensor containing the result of the scalar multiplication.
  """
  return np.multiply(scalar, tensor)

def scalar_add_tensor(tensor, scalar):
  """
  Adds a scalar to a tensor.

  Args:
    tensor: The tensor (numpy array).
    scalar: The scalar value.

  Returns:
    A new tensor containing the result of the scalar addition.
  """
  return np.add(tensor, scalar)

def reshape_tensor(tensor, new_shape):
  """
  Reshapes a tensor to a new shape.

  Args:
    tensor: The tensor (numpy array) to reshape.
    new_shape: A tuple or list representing the new dimensions of the tensor.

  Returns:
    A new tensor with the specified new_shape.

  Raises:
    ValueError: If the new shape is incompatible with the original shape
                (i.e., the product of new dimensions doesn't match the
                 product of old dimensions).
  """
  try:
    return np.reshape(tensor, new_shape)
  except ValueError as e:
    raise ValueError(f"Cannot reshape tensor of size {tensor.size} into shape {new_shape}: {e}")

def transpose_tensor(tensor, axes_permutation=None):
  """
  Transposes a tensor along the specified axes.

  Args:
    tensor: The tensor (numpy array) to transpose.
    axes_permutation: A tuple or list of integers representing the new
                      order of axes. If None (default), it reverses the
                      order of the axes.

  Returns:
    A new tensor with its axes transposed.

  Raises:
    ValueError: If the axes_permutation is invalid (e.g., contains duplicate
                axes, axes out of bounds for the tensor's dimensions).
  """
  try:
    return np.transpose(tensor, axes_permutation)
  except ValueError as e:
    raise ValueError(f"Invalid axes_permutation {axes_permutation} for tensor with {tensor.ndim} dimensions: {e}")

def tensor_dot_product(tensor1, tensor2, axes):
  """
  Computes the tensor dot product (tensor contraction) along specified axes.

  Args:
    tensor1: The first tensor (numpy array).
    tensor2: The second tensor (numpy array).
    axes: An int or a sequence of two ints.
          - If an int N, sum over the last N axes of tensor1 and the first N
            axes of tensor2 in order. The result tensor will have
            shape_tensor1[:-N] + shape_tensor2[N:].
          - If a sequence of two ints (axes_tensor1, axes_tensor2), sum over the
            axes specified by axes_tensor1 for tensor1 and axes_tensor2 for tensor2.

  Returns:
    The tensor dot product of tensor1 and tensor2 along the specified axes.

  Raises:
    ValueError: If the shapes are incompatible for the given axes.
  """
  try:
    return np.tensordot(tensor1, tensor2, axes=axes)
  except ValueError as e:
    raise ValueError(f"Incompatible shapes for tensordot with axes {axes}: {e}")

def outer_product(tensor1, tensor2):
  """
  Computes the outer product of two tensors.
  This is equivalent to np.tensordot(tensor1, tensor2, axes=0).

  Args:
    tensor1: The first tensor (numpy array).
    tensor2: The second tensor (numpy array).

  Returns:
    The outer product of tensor1 and tensor2.
  """
  return np.tensordot(tensor1, tensor2, axes=0) # Or np.outer for 1D vectors specifically

def einstein_summation(subscripts, *operands):
  """
  Evaluates the Einstein summation convention on the operands.

  Args:
    subscripts: Specifies the Einstein summation convention using a string
                like "ij,jk->ik" where letters map to dimensions.
    *operands: A sequence of tensors (numpy arrays) to operate on.

  Returns:
    The result of the Einstein summation.

  Raises:
    ValueError: If the subscripts or operands are invalid (e.g., mismatched
                dimensions, incorrect subscript string).
  """
  try:
    return np.einsum(subscripts, *operands)
  except ValueError as e:
    raise ValueError(f"Error in Einstein summation with subscripts '{subscripts}': {e}")

def frobenius_norm(tensor):
  """
  Computes the Frobenius norm of a tensor.

  Args:
    tensor: The tensor (numpy array).

  Returns:
    The Frobenius norm of the tensor.
  """
  return np.linalg.norm(tensor, 'fro')

def l1_norm(tensor):
  """
  Computes the L1 norm of a tensor.
  L1 norm is the sum of the absolute values of the elements.

  Args:
    tensor: The tensor (numpy array).

  Returns:
    The L1 norm of the tensor.
  """
  return np.sum(np.abs(tensor))

# --- PyTorch based operations ---
# The following functions require PyTorch.
# Inputs to these functions should be PyTorch tensors.

import torch
from torch.autograd.functional import jacobian

def compute_gradient(scalar_function, tensor_input):
  """
  Computes the gradient of a scalar-valued function with respect to a tensor input.
  Requires PyTorch.

  Args:
    scalar_function: A function that takes a PyTorch tensor and returns a scalar PyTorch tensor.
    tensor_input: A PyTorch tensor for which requires_grad will be set to True.

  Returns:
    The gradient of the scalar_function with respect to tensor_input.
    Returns None if tensor_input.grad is None after backward pass.
  """
  if not isinstance(tensor_input, torch.Tensor):
    raise TypeError("tensor_input must be a PyTorch tensor.")

  # Ensure tensor_input requires gradients
  if not tensor_input.requires_grad:
    tensor_input.requires_grad_(True)

  # Zero out any existing gradients
  if tensor_input.grad is not None:
    tensor_input.grad.zero_()

  output_scalar = scalar_function(tensor_input)

  if not isinstance(output_scalar, torch.Tensor) or not output_scalar.ndim == 0:
    raise ValueError("scalar_function must return a scalar PyTorch tensor.")

  output_scalar.backward()
  return tensor_input.grad

def compute_jacobian(vector_function, tensor_input):
  """
  Computes the Jacobian of a vector-valued function with respect to a tensor input.
  Requires PyTorch.

  Args:
    vector_function: A function that takes a PyTorch tensor and returns a PyTorch tensor.
    tensor_input: A PyTorch tensor. It does not need requires_grad=True beforehand.

  Returns:
    The Jacobian matrix (or tensor) of vector_function with respect to tensor_input.
  """
  if not isinstance(tensor_input, torch.Tensor):
    raise TypeError("tensor_input must be a PyTorch tensor.")

  # jacobian function in PyTorch expects the input tensor to not have requires_grad=True set by the user for certain cases,
  # or it might behave unexpectedly or error if it's a non-leaf tensor with grad_fn.
  # It's generally safer to let `jacobian` handle the grad status internally or pass a fresh tensor.
  # If tensor_input already has requires_grad=True and is a leaf, it's usually fine.

  # A common pattern is to clone and detach if you want to be absolutely sure,
  # especially if tensor_input might be part of a larger graph.
  # For simplicity here, we pass it directly as per common `jacobian` usage.

  return jacobian(vector_function, tensor_input)

# --- NumPy based linear algebra operations ---

def matrix_eigendecomposition(matrix_A):
  """
  Computes the eigenvalues and eigenvectors of a square matrix.

  Args:
    matrix_A: A 2D square NumPy array.

  Returns:
    A tuple containing:
      - eigenvalues: A 1D NumPy array of eigenvalues.
      - eigenvectors: A 2D NumPy array where each column is an eigenvector.

  Raises:
    ValueError: If matrix_A is not a 2D square NumPy array.
  """
  if not isinstance(matrix_A, np.ndarray):
    raise TypeError("Input matrix_A must be a NumPy array.")
  if matrix_A.ndim != 2:
    raise ValueError("Input matrix_A must be a 2D array.")
  if matrix_A.shape[0] != matrix_A.shape[1]:
    raise ValueError("Input matrix_A must be a square matrix (number of rows must equal number of columns).")

  eigenvalues, eigenvectors = np.linalg.eig(matrix_A)
  return eigenvalues, eigenvectors

def matrix_trace(matrix_A):
  """
  Computes the trace of a 2D square matrix (sum of diagonal elements).

  Args:
    matrix_A: A 2D NumPy array.

  Returns:
    The trace of matrix_A.

  Raises:
    ValueError: If matrix_A is not a 2D NumPy array.
    TypeError: If matrix_A is not a NumPy array.
  """
  if not isinstance(matrix_A, np.ndarray):
    raise TypeError("Input matrix_A must be a NumPy array.")
  if matrix_A.ndim != 2:
    raise ValueError("Input matrix_A must be a 2D array for matrix_trace. For higher-dimensional arrays, use tensor_trace.")
  # numpy.trace itself doesn't require square for 2D, but traditionally trace is for square matrices.
  # However, we will follow numpy.trace behavior which allows non-square 2D.
  return np.trace(matrix_A)

def tensor_trace(tensor_A, axis1=0, axis2=1):
  """
  Computes the trace (sum of elements along specified diagonal) of a tensor.
  This is a generalization of matrix trace to higher dimensional arrays.
  The trace is computed by summing along `axis1` and `axis2`.

  Args:
    tensor_A: A NumPy array.
    axis1: The first axis defining the diagonal. Default is 0.
    axis2: The second axis defining the diagonal. Default is 1.

  Returns:
    The traced (reduced) NumPy array.

  Raises:
    ValueError: If axis1 or axis2 are out of bounds, or if the dimensions
                along axis1 and axis2 do not match.
    TypeError: If tensor_A is not a NumPy array.
  """
  if not isinstance(tensor_A, np.ndarray):
    raise TypeError("Input tensor_A must be a NumPy array.")

  # np.trace handles axis validation and dimension matching internally,
  # raising ValueError if they are invalid.
  try:
    return np.trace(tensor_A, axis1=axis1, axis2=axis2)
  except ValueError as e:
    # Re-raise with a more informative message or just let numpy's error propagate.
    # For now, let numpy's more specific error propagate.
    raise e
