import numpy as np
# Imports for scipy, torch, tensorly will be moved into their respective functions

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

# import torch # Moved into functions
# from torch.autograd.functional import jacobian # Moved into functions

def compute_gradient(scalar_function, tensor_input):
  import torch # Moved from top
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
  # torch is imported inside this function
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
  from torch.autograd.functional import jacobian # Moved from top
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

# --- SciPy based signal processing operations ---

def convolve_1d(signal_x, kernel_w, mode='valid'):
  from scipy import signal # Moved from top
  """
  Computes the 1D convolution of two NumPy arrays.

  Args:
    signal_x: A 1D NumPy array representing the input signal.
    kernel_w: A 1D NumPy array representing the kernel (weights).
    mode: A string indicating the type of convolution.
          'full': Returns the full discrete linear convolution.
          'valid': Returns only those parts of the convolution that are computed
                   without zero-padding. (Default)
          'same': Returns the central part of the convolution that is the same
                  size as signal_x.

  Returns:
    A 1D NumPy array representing the convolved signal.

  Raises:
    TypeError: If signal_x or kernel_w are not NumPy arrays.
    ValueError: If signal_x or kernel_w are not 1D arrays, or if mode is invalid.
  """
  # scipy.signal is imported inside this function
  if not isinstance(signal_x, np.ndarray) or not isinstance(kernel_w, np.ndarray):
    raise TypeError("Inputs signal_x and kernel_w must be NumPy arrays.")
  if signal_x.ndim != 1:
    raise ValueError("Input signal_x must be a 1D array.")
  if kernel_w.ndim != 1:
    raise ValueError("Input kernel_w must be a 1D array.")

  valid_modes = ['full', 'valid', 'same']
  if mode not in valid_modes:
    raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}.")

  return signal.convolve(signal_x, kernel_w, mode=mode)

def convolve_2d(image_I, kernel_K, mode='valid'):
  """
  Computes the 2D convolution of two NumPy arrays.

  Args:
    image_I: A 2D NumPy array representing the input image.
    kernel_K: A 2D NumPy array representing the kernel.
    mode: A string indicating the type of convolution.
          'full': Returns the full discrete linear convolution.
          'valid': Returns only those parts of the convolution that are computed
                   without zero-padding. (Default)
          'same': Returns the central part of the convolution that is the same
                  size as image_I.

  Returns:
    A 2D NumPy array representing the convolved image.

  Raises:
    TypeError: If image_I or kernel_K are not NumPy arrays.
    ValueError: If image_I or kernel_K are not 2D arrays, or if mode is invalid.
  """
  from scipy import signal # Moved from top
  if not isinstance(image_I, np.ndarray) or not isinstance(kernel_K, np.ndarray):
    raise TypeError("Inputs image_I and kernel_K must be NumPy arrays.")
  if image_I.ndim != 2:
    raise ValueError("Input image_I must be a 2D array.")
  if kernel_K.ndim != 2:
    raise ValueError("Input kernel_K must be a 2D array.")

  valid_modes = ['full', 'valid', 'same']
  if mode not in valid_modes:
    raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}.")

  return signal.convolve2d(image_I, kernel_K, mode=mode)

# --- NumPy based statistical operations ---

def tensor_mean(tensor_A, axis=None, keepdims=False):
  """
  Computes the arithmetic mean along the specified axis.

  Args:
    tensor_A: A NumPy array.
    axis: Axis or axes along which the means are computed.
          The default is to compute the mean of the flattened array.
    keepdims: If this is set to True, the axes which are reduced are left
              in the result as dimensions with size one.

  Returns:
    The mean of the array elements.

  Raises:
    TypeError: If tensor_A is not a NumPy array.
  """
  if not isinstance(tensor_A, np.ndarray):
    raise TypeError("Input tensor_A must be a NumPy array.")
  return np.mean(tensor_A, axis=axis, keepdims=keepdims)

def tensor_variance(tensor_A, axis=None, ddof=0, keepdims=False):
  """
  Computes the variance along the specified axis.

  Args:
    tensor_A: A NumPy array.
    axis: Axis or axes along which the variance is computed.
          The default is to compute the variance of the flattened array.
    ddof: "Delta Degrees of Freedom": the divisor used in the calculation is
          N - ddof, where N represents the number of elements. Default is 0.
    keepdims: If this is set to True, the axes which are reduced are left
              in the result as dimensions with size one.

  Returns:
    The variance of the array elements.

  Raises:
    TypeError: If tensor_A is not a NumPy array.
  """
  if not isinstance(tensor_A, np.ndarray):
    raise TypeError("Input tensor_A must be a NumPy array.")
  return np.var(tensor_A, axis=axis, ddof=ddof, keepdims=keepdims)

def matrix_covariance(matrix_X, matrix_Y=None, rowvar=True, ddof=None, bias=False):
  """
  Estimates a covariance matrix, given data and weights.

  Args:
    matrix_X: A 1D or 2D NumPy array containing variables and observations.
              If rowvar is True (default), then each row represents a
              variable, with observations in the columns. Otherwise, the
              relationship is transposed: each column represents a variable,
              while the rows contain observations.
    matrix_Y: (Optional) A 1D or 2D NumPy array with the same dimensionality
              as matrix_X. This specifies a second set of variables and observations.
    rowvar: If True (default), each row represents a variable, with
            observations in the columns. Otherwise, each column represents a
            variable and rows are observations.
    ddof: Degrees of freedom correction. Default is None, which implies
          N-1 for unweighted, N for weighted, where N is number of observations.
    bias: If False (default), then the sum is divided by (N - 1) where N is
          the number of observations. If True, then the sum is divided by N.
          These values can be overridden by ddof.

  Returns:
    The covariance matrix of the input data.

  Raises:
    TypeError: If matrix_X (or matrix_Y if provided) is not a NumPy array.
    ValueError: If inputs are not 1D or 2D arrays, or other np.cov specific errors.
  """
  if not isinstance(matrix_X, np.ndarray):
    raise TypeError("Input matrix_X must be a NumPy array.")
  if matrix_Y is not None and not isinstance(matrix_Y, np.ndarray):
    raise TypeError("Input matrix_Y must be a NumPy array if provided.")

  # np.cov handles dimension checks (1D or 2D) internally.
  try:
    return np.cov(m=matrix_X, y=matrix_Y, rowvar=rowvar, ddof=ddof, bias=bias)
  except ValueError as e:
    # Let numpy's specific error messages propagate for dimension issues etc.
    raise e

def matrix_correlation(matrix_X, matrix_Y=None, rowvar=True):
  """
  Computes the Pearson product-moment correlation coefficients.

  Args:
    matrix_X: A 1D or 2D NumPy array containing variables and observations.
              If rowvar is True (default), then each row represents a
              variable, with observations in the columns. Otherwise, the
              relationship is transposed.
    matrix_Y: (Optional) A 1D or 2D NumPy array with the same dimensionality
              as matrix_X. This specifies a second set of variables.
    rowvar: If True (default), each row represents a variable, with
            observations in the columns. Otherwise, each column represents a
            variable and rows are observations.

  Returns:
    The correlation matrix of the input data.

  Raises:
    TypeError: If matrix_X (or matrix_Y if provided) is not a NumPy array.
    ValueError: If inputs are not 1D or 2D arrays, or other np.corrcoef specific errors.
  """
  if not isinstance(matrix_X, np.ndarray):
    raise TypeError("Input matrix_X must be a NumPy array.")
  if matrix_Y is not None and not isinstance(matrix_Y, np.ndarray):
    raise TypeError("Input matrix_Y must be a NumPy array if provided.")

  # np.corrcoef handles dimension checks internally.
  # It's built on top of np.cov, so error handling for dimensions will be similar.
  try:
    if matrix_Y is None:
        return np.corrcoef(matrix_X, rowvar=rowvar)
    else:
        return np.corrcoef(matrix_X, y=matrix_Y, rowvar=rowvar)
  except ValueError as e:
    raise e

# --- TensorLy based operations ---
# These functions require the TensorLy library. Install with: pip install tensorly

def estimate_tensor_rank_cp(tensor_A, max_rank=10, n_iter_max=100, tol=1e-7, verbose=False):
  import tensorly as tl # Moved from top
  from tensorly.decomposition import parafac # Moved from top
  """
  Estimates the rank of a tensor using CP decomposition by finding the rank
  that minimizes the reconstruction error up to a specified max_rank.

  Args:
    tensor_A: The input tensor (NumPy array).
    max_rank: The maximum rank to test for CP decomposition. (Default: 10)
    n_iter_max: Maximum number of iterations for the PARAFAC algorithm. (Default: 100)
    tol: Tolerance for convergence of the PARAFAC algorithm. (Default: 1e-7)
    verbose: If True, prints the reconstruction error for each rank. (Default: False)

  Returns:
    A tuple containing:
      - estimated_rank: The rank that yielded the minimum reconstruction error.
      - errors: A list of reconstruction errors for each rank tested.

  Raises:
    TypeError: If tensor_A is not a NumPy array.
    ImportError: If TensorLy is not installed.
  """
  # tensorly and parafac are imported inside this function
  if not isinstance(tensor_A, np.ndarray):
    raise TypeError("Input tensor_A must be a NumPy array.")

  # This check is less critical now as direct import failure would raise ImportError
  # if tl is None or parafac is None:
  #     raise ImportError("TensorLy library is required for estimate_tensor_rank_cp. Please install it via 'pip install tensorly'.")

  errors = []
  norm_tensor_A = tl.norm(tensor_A)

  if norm_tensor_A == 0: # Handle zero tensor case
      if verbose:
          print("Input tensor is a zero tensor. Reconstruction error is 0 for rank 1.")
      return 1, [0.0] * max_rank


  for r in range(1, max_rank + 1):
    try:
      # Perform CP decomposition
      # init='svd' can be helpful for reproducibility and convergence
      weights, factors = parafac(tensor_A, rank=r, n_iter_max=n_iter_max, tol=tol, init='random', random_state=0)

      # Reconstruct the tensor
      reconstructed_tensor = tl.cp_to_tensor((weights, factors))

      # Calculate relative reconstruction error
      error = tl.norm(tensor_A - reconstructed_tensor) / norm_tensor_A
      errors.append(error)

      if verbose:
        print(f"Rank {r}: Relative Reconstruction Error = {error:.4e}")

      # Optional: Check for a sufficiently small error to stop early
      # if error < tol: # A different tolerance might be used for early stopping
      #     break

    except Exception as e:
      # Handle potential errors during decomposition for a specific rank
      if verbose:
        print(f"Rank {r}: Error during CP decomposition: {e}")
      errors.append(float('inf')) # Assign a high error if decomposition fails

  if not errors: # Should not happen if max_rank >= 1
      return 0, []

  min_error = float('inf')
  estimated_rank = 0
  for i, err in enumerate(errors):
      if err < min_error:
          min_error = err
          estimated_rank = i + 1

  return estimated_rank, errors
