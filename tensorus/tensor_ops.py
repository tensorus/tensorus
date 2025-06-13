# tensor_ops.py
"""
Provides a library of robust tensor operations for Tensorus.

This module defines a static class `TensorOps` containing methods for common
tensor manipulations, including arithmetic, linear algebra, reshaping, and
more advanced operations. It emphasizes shape checking and error handling.

Future Enhancements:
- Add more advanced operations (FFT, specific convolutions).
- Implement optional automatic broadcasting checks.
- Add support for sparse tensors.
- Optimize operations further (e.g., using custom kernels if needed).
"""

import torch
import logging
from typing import Tuple, Optional, List, Union

import tensorly as tl
# All decomposition imports (parafac, tucker, tensor_train) are now removed.
# CPTensor is a namedtuple, can be used for type hinting if specific
# from tensorly.cp_tensor import CPTensor

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TensorOps:
    """
    A static library class providing robust tensor operations.
    All methods are static and operate on provided torch.Tensor objects.
    """

    @staticmethod
    def _check_tensor(*tensors: torch.Tensor) -> None:
        """Internal helper to check if inputs are PyTorch tensors."""
        for i, t in enumerate(tensors):
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"Input at index {i} is not a torch.Tensor, but {type(t)}.")

    @staticmethod
    def _check_shape(tensor: torch.Tensor, expected_shape: Tuple[Optional[int], ...], op_name: str) -> None:
        """Internal helper to check tensor shape against an expected shape with wildcards (None)."""
        TensorOps._check_tensor(tensor)
        actual_shape = tensor.shape
        if len(actual_shape) != len(expected_shape):
            raise ValueError(f"{op_name} expects a tensor with {len(expected_shape)} dimensions, but got {len(actual_shape)} dimensions (shape {actual_shape}). Expected pattern: {expected_shape}")

        for i, (actual_dim, expected_dim) in enumerate(zip(actual_shape, expected_shape)):
            if expected_dim is not None and actual_dim != expected_dim:
                raise ValueError(f"{op_name} expects dimension {i} to be {expected_dim}, but got {actual_dim}. Actual shape: {actual_shape}. Expected pattern: {expected_shape}")
        logging.debug(f"{op_name} shape check passed for tensor with shape {actual_shape}")


    # --- Arithmetic Operations ---

    @staticmethod
    def add(t1: torch.Tensor, t2: Union[torch.Tensor, float, int]) -> torch.Tensor:
        """Element-wise addition with type checking."""
        TensorOps._check_tensor(t1)
        if isinstance(t2, torch.Tensor):
            TensorOps._check_tensor(t2)
        try:
            return torch.add(t1, t2)
        except RuntimeError as e:
            logging.error(f"Error during addition: {e}. t1 shape: {t1.shape}, t2 type: {type(t2)}, t2 shape (if tensor): {t2.shape if isinstance(t2, torch.Tensor) else 'N/A'}")
            raise e

    @staticmethod
    def subtract(t1: torch.Tensor, t2: Union[torch.Tensor, float, int]) -> torch.Tensor:
        """Element-wise subtraction with type checking."""
        TensorOps._check_tensor(t1)
        if isinstance(t2, torch.Tensor):
            TensorOps._check_tensor(t2)
        try:
            return torch.subtract(t1, t2)
        except RuntimeError as e:
            logging.error(f"Error during subtraction: {e}. t1 shape: {t1.shape}, t2 type: {type(t2)}, t2 shape (if tensor): {t2.shape if isinstance(t2, torch.Tensor) else 'N/A'}")
            raise e

    @staticmethod
    def multiply(t1: torch.Tensor, t2: Union[torch.Tensor, float, int]) -> torch.Tensor:
        """Element-wise multiplication with type checking."""
        TensorOps._check_tensor(t1)
        if isinstance(t2, torch.Tensor):
            TensorOps._check_tensor(t2)
        try:
            return torch.multiply(t1, t2)
        except RuntimeError as e:
            logging.error(f"Error during multiplication: {e}. t1 shape: {t1.shape}, t2 type: {type(t2)}, t2 shape (if tensor): {t2.shape if isinstance(t2, torch.Tensor) else 'N/A'}")
            raise e

    @staticmethod
    def divide(t1: torch.Tensor, t2: Union[torch.Tensor, float, int]) -> torch.Tensor:
        """Element-wise division with type checking and zero division check."""
        TensorOps._check_tensor(t1)
        if isinstance(t2, torch.Tensor):
            TensorOps._check_tensor(t2)
            if torch.any(t2 == 0):
                 logging.warning("Division by zero encountered in tensor division.")
                 # Depending on policy, could raise error or return inf/nan
                 # raise ValueError("Division by zero in tensor division.")
        elif isinstance(t2, (int, float)):
            if t2 == 0:
                logging.error("Division by zero scalar.")
                raise ValueError("Division by zero.")
        else:
             raise TypeError(f"Divisor must be a tensor or scalar, got {type(t2)}")

        try:
            return torch.divide(t1, t2)
        except RuntimeError as e:
            logging.error(f"Error during division: {e}. t1 shape: {t1.shape}, t2 type: {type(t2)}, t2 shape (if tensor): {t2.shape if isinstance(t2, torch.Tensor) else 'N/A'}")
            raise e

    @staticmethod
    def power(t1: torch.Tensor, t2: Union[torch.Tensor, float, int]) -> torch.Tensor:
        """
        Computes the element-wise power of tensor t1 to the exponent t2.

        Args:
            t1 (torch.Tensor): The base tensor.
            t2 (Union[torch.Tensor, float, int]): The exponent, which can be
                a tensor, a float, or an integer.

        Returns:
            torch.Tensor: The result of t1 raised to the power of t2, element-wise.

        Raises:
            TypeError: If t1 is not a torch.Tensor, or if t2 is a tensor but
                       not a torch.Tensor.
            RuntimeError: If an error occurs during the torch.pow computation.
        """
        TensorOps._check_tensor(t1)
        if isinstance(t2, torch.Tensor):
            TensorOps._check_tensor(t2)
        try:
            return torch.pow(t1, t2)
        except RuntimeError as e:
            logging.error(f"Error during power operation: {e}. t1 shape: {t1.shape}, t2 type: {type(t2)}, t2 shape (if tensor): {t2.shape if isinstance(t2, torch.Tensor) else 'N/A'}")
            raise e

    @staticmethod
    def log(tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes the element-wise natural logarithm of a tensor.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The natural logarithm of the input tensor.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            RuntimeError: If an error occurs during the torch.log computation.
        """
        TensorOps._check_tensor(tensor)
        if torch.any(tensor <= 0):
            logging.warning("Logarithm of non-positive value encountered in tensor. This will result in NaN or -inf values.")
        try:
            return torch.log(tensor)
        except RuntimeError as e:
            logging.error(f"Error during log operation: {e}. Input tensor shape: {tensor.shape}")
            raise e

    # --- Matrix and Dot Operations ---

    @staticmethod
    def matmul(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Matrix multiplication (torch.matmul) with shape checks."""
        TensorOps._check_tensor(t1, t2)
        if t1.ndim < 1 or t2.ndim < 1:
             raise ValueError(f"Matmul requires tensors with at least 1 dimension, got {t1.ndim} and {t2.ndim}")

        # Basic check for standard 2D matrix multiplication
        if t1.ndim == 2 and t2.ndim == 2:
            if t1.shape[1] != t2.shape[0]:
                raise ValueError(f"Matrix multiplication shape mismatch: t1 shape {t1.shape} (inner dim {t1.shape[1]}) and t2 shape {t2.shape} (inner dim {t2.shape[0]}) are incompatible.")
        # Note: torch.matmul handles broadcasting and batch matmul, more complex checks could be added here.
        try:
            return torch.matmul(t1, t2)
        except RuntimeError as e:
            logging.error(f"Error during matmul: {e}. t1 shape: {t1.shape}, t2 shape: {t2.shape}")
            raise e

    @staticmethod
    def dot(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Dot product (torch.dot) for 1D tensors."""
        TensorOps._check_tensor(t1, t2)
        TensorOps._check_shape(t1, (None,), "dot product input 1")
        TensorOps._check_shape(t2, (None,), "dot product input 2")
        if t1.shape[0] != t2.shape[0]:
             raise ValueError(f"Dot product requires 1D tensors of the same size, got shapes {t1.shape} and {t2.shape}")
        try:
            return torch.dot(t1, t2)
        except RuntimeError as e:
            logging.error(f"Error during dot product: {e}. t1 shape: {t1.shape}, t2 shape: {t2.shape}")
            raise e

    @staticmethod
    def outer(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Outer product for two 1-D tensors."""
        TensorOps._check_tensor(t1, t2)
        TensorOps._check_shape(t1, (None,), "outer input 1")
        TensorOps._check_shape(t2, (None,), "outer input 2")
        try:
            return torch.outer(t1, t2)
        except RuntimeError as e:
            logging.error(
                f"Error during outer product: {e}. t1 shape: {t1.shape}, t2 shape: {t2.shape}"
            )
            raise e

    @staticmethod
    def cross(t1: torch.Tensor, t2: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Cross product along a dimension (size 3)."""
        TensorOps._check_tensor(t1, t2)
        if t1.shape != t2.shape:
            raise ValueError(
                f"Cross product requires tensors of the same shape, got {t1.shape} and {t2.shape}"
            )
        rank = t1.ndim
        if rank == 0:
            raise ValueError("Cross product requires tensors with at least 1 dimension")
        if dim < 0:
            dim += rank
        if dim < 0 or dim >= rank:
            raise ValueError(f"dim {dim} out of range for tensors with rank {rank}")
        if t1.shape[dim] != 3:
            raise ValueError(
                f"Cross product is defined for dimension size 3, got {t1.shape[dim]}"
            )
        try:
            return torch.cross(t1, t2, dim=dim)
        except RuntimeError as e:
            logging.error(
                f"Error during cross product: {e}. t1 shape: {t1.shape}, t2 shape: {t2.shape}, dim: {dim}"
            )
            raise e


    # --- Reduction Operations ---

    @staticmethod
    def sum(tensor: torch.Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> torch.Tensor:
        """Sum of tensor elements over given dimensions."""
        TensorOps._check_tensor(tensor)
        try:
            return torch.sum(tensor, dim=dim, keepdim=keepdim)
        except RuntimeError as e:
            logging.error(f"Error during sum: {e}. tensor shape: {tensor.shape}, dim: {dim}")
            raise e

    @staticmethod
    def mean(tensor: torch.Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> torch.Tensor:
        """Mean of tensor elements over given dimensions."""
        TensorOps._check_tensor(tensor)
        try:
            # Ensure float tensor for mean calculation
            return torch.mean(tensor.float(), dim=dim, keepdim=keepdim)
        except RuntimeError as e:
            logging.error(f"Error during mean: {e}. tensor shape: {tensor.shape}, dim: {dim}")
            raise e

    @staticmethod
    def min(tensor: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Min of tensor elements over a given dimension."""
        TensorOps._check_tensor(tensor)
        try:
            return torch.min(tensor, dim=dim, keepdim=keepdim)
        except RuntimeError as e:
            logging.error(f"Error during min: {e}. tensor shape: {tensor.shape}, dim: {dim}")
            raise e

    @staticmethod
    def max(tensor: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Max of tensor elements over a given dimension."""
        TensorOps._check_tensor(tensor)
        try:
            return torch.max(tensor, dim=dim, keepdim=keepdim)
        except RuntimeError as e:
            logging.error(f"Error during max: {e}. tensor shape: {tensor.shape}, dim: {dim}")
            raise e


    # --- Reshaping and Slicing ---

    @staticmethod
    def reshape(tensor: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """Reshape tensor with validation."""
        TensorOps._check_tensor(tensor)
        try:
            return torch.reshape(tensor, shape)
        except RuntimeError as e:
             logging.error(f"Error during reshape: {e}. Original shape: {tensor.shape}, target shape: {shape}")
             raise ValueError(f"Cannot reshape tensor of shape {tensor.shape} to {shape}. {e}")


    @staticmethod
    def transpose(tensor: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
        """Transpose tensor dimensions."""
        TensorOps._check_tensor(tensor)
        try:
            return torch.transpose(tensor, dim0, dim1)
        except Exception as e: # Catches index errors etc.
            logging.error(f"Error during transpose: {e}. tensor shape: {tensor.shape}, dim0: {dim0}, dim1: {dim1}")
            raise e

    @staticmethod
    def permute(tensor: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
        """Permute tensor dimensions."""
        TensorOps._check_tensor(tensor)
        if len(dims) != tensor.ndim:
            raise ValueError(f"Permute dims tuple length {len(dims)} must match tensor rank {tensor.ndim}")
        if len(set(dims)) != len(dims) or not all(0 <= d < tensor.ndim for d in dims):
             raise ValueError(f"Invalid permutation dims {dims} for tensor rank {tensor.ndim}")
        try:
            return tensor.permute(dims) # Use the method directly
        except Exception as e:
            logging.error(f"Error during permute: {e}. tensor shape: {tensor.shape}, dims: {dims}")
            raise e

    @staticmethod
    def flatten(tensor: torch.Tensor, start_dim: int = 0, end_dim: int = -1) -> torch.Tensor:
        """Flatten contiguous dimensions of a tensor."""
        TensorOps._check_tensor(tensor)
        rank = tensor.ndim
        # Normalize negative indices
        if start_dim < 0:
            start_dim += rank
        if end_dim < 0:
            end_dim += rank
        if not (0 <= start_dim < rank) or not (0 <= end_dim < rank):
            raise ValueError(f"start_dim and end_dim must be in [0, {rank-1}], got {start_dim} and {end_dim}")
        if start_dim > end_dim:
            raise ValueError(f"start_dim ({start_dim}) cannot be greater than end_dim ({end_dim})")
        try:
            return torch.flatten(tensor, start_dim=start_dim, end_dim=end_dim)
        except Exception as e:
            logging.error(f"Error during flatten: {e}. tensor shape: {tensor.shape}, start_dim: {start_dim}, end_dim: {end_dim}")
            raise e

    @staticmethod
    def squeeze(tensor: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
        """Remove dimensions of size 1."""
        TensorOps._check_tensor(tensor)
        if dim is not None:
            if dim < -tensor.ndim or dim >= tensor.ndim:
                raise ValueError(f"dim {dim} out of range for tensor with rank {tensor.ndim}")
        try:
            return torch.squeeze(tensor, dim) if dim is not None else torch.squeeze(tensor)
        except Exception as e:
            logging.error(f"Error during squeeze: {e}. tensor shape: {tensor.shape}, dim: {dim}")
            raise e

    @staticmethod
    def unsqueeze(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """Insert a dimension of size 1 at the specified position."""
        TensorOps._check_tensor(tensor)
        rank = tensor.ndim
        if dim < -(rank + 1) or dim > rank:
            raise ValueError(f"dim {dim} out of valid range [-(rank+1), rank] for tensor rank {rank}")
        try:
            return torch.unsqueeze(tensor, dim)
        except Exception as e:
            logging.error(f"Error during unsqueeze: {e}. tensor shape: {tensor.shape}, dim: {dim}")
            raise e


    # --- Concatenation and Splitting ---

    @staticmethod
    def concatenate(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
        """Concatenate tensors along a dimension with checks."""
        if not tensors:
            raise ValueError("Cannot concatenate an empty list of tensors.")
        TensorOps._check_tensor(*tensors)
        # Add checks for shape compatibility along non-concatenated dims if needed
        try:
            return torch.cat(tensors, dim=dim)
        except RuntimeError as e:
            shapes = [t.shape for t in tensors]
            logging.error(f"Error during concatenation: {e}. Input shapes: {shapes}, dim: {dim}")
            raise e

    @staticmethod
    def stack(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
        """Stack tensors along a new dimension with checks."""
        if not tensors:
            raise ValueError("Cannot stack an empty list of tensors.")
        TensorOps._check_tensor(*tensors)
        # Add checks for shape equality if needed
        try:
            return torch.stack(tensors, dim=dim)
        except RuntimeError as e:
            shapes = [t.shape for t in tensors]
            logging.error(f"Error during stack: {e}. Input shapes: {shapes}, dim: {dim}")
            raise e

    # --- Advanced Operations ---

    @staticmethod
    def einsum(equation: str, *tensors: torch.Tensor) -> torch.Tensor:
        """Einstein summation with type checking."""
        TensorOps._check_tensor(*tensors)
        try:
            return torch.einsum(equation, *tensors)
        except RuntimeError as e:
            shapes = [t.shape for t in tensors]
            logging.error(f"Error during einsum: {e}. Equation: '{equation}', Input shapes: {shapes}")
            raise e

    # --- Autograd Operations ---

    @staticmethod
    def compute_gradient(scalar_function, tensor_input: torch.Tensor) -> torch.Tensor:
        """Compute gradient of a scalar function with respect to `tensor_input`."""
        TensorOps._check_tensor(tensor_input)
        if not tensor_input.requires_grad:
            tensor_input.requires_grad_(True)
        if tensor_input.grad is not None:
            tensor_input.grad.zero_()
        output = scalar_function(tensor_input)
        if output.ndim != 0:
            raise ValueError("scalar_function must return a scalar tensor")
        output.backward()
        return tensor_input.grad

    @staticmethod
    def compute_jacobian(vector_function, tensor_input: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian of a vector function with respect to `tensor_input`."""
        TensorOps._check_tensor(tensor_input)
        from torch.autograd.functional import jacobian
        return jacobian(vector_function, tensor_input)

    # --- Linear Algebra Operations ---

    @staticmethod
    def matrix_eigendecomposition(matrix_A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eigenvalues and eigenvectors of a square matrix."""
        TensorOps._check_tensor(matrix_A)
        if matrix_A.ndim != 2:
            raise ValueError("Input matrix_A must be 2-D")
        if matrix_A.shape[0] != matrix_A.shape[1]:
            raise ValueError("Input matrix_A must be square")
        return torch.linalg.eig(matrix_A)

    @staticmethod
    def matrix_trace(matrix_A: torch.Tensor) -> torch.Tensor:
        """Trace of a 2-D matrix."""
        TensorOps._check_tensor(matrix_A)
        if matrix_A.ndim != 2:
            raise ValueError("Input matrix_A must be 2-D")
        return torch.trace(matrix_A)

    @staticmethod
    def tensor_trace(tensor_A: torch.Tensor, axis1: int = 0, axis2: int = 1) -> torch.Tensor:
        """Trace of a tensor along two axes."""
        TensorOps._check_tensor(tensor_A)
        if axis1 >= tensor_A.ndim or axis2 >= tensor_A.ndim:
            raise ValueError("axis1 and axis2 must be valid dimensions")
        if tensor_A.shape[axis1] != tensor_A.shape[axis2]:
            raise ValueError("Dimensions for axis1 and axis2 must match")
        return torch.diagonal(tensor_A, offset=0, dim1=axis1, dim2=axis2).sum(-1)

    @staticmethod
    def svd(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Singular Value Decomposition of a 2-D matrix."""
        TensorOps._check_tensor(matrix)
        if matrix.ndim != 2:
            raise ValueError("Input matrix must be 2-D")
        return torch.linalg.svd(matrix, full_matrices=False)

    @staticmethod
    def qr_decomposition(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """QR decomposition of a 2-D matrix."""
        TensorOps._check_tensor(matrix)
        if matrix.ndim != 2:
            raise ValueError("Input matrix must be 2-D")
        return torch.linalg.qr(matrix)

    @staticmethod
    def lu_decomposition(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """LU decomposition of a 2-D matrix returning P, L, U."""
        TensorOps._check_tensor(matrix)
        if matrix.ndim != 2:
            raise ValueError("Input matrix must be 2-D")
        return torch.linalg.lu(matrix)

    @staticmethod
    def cholesky_decomposition(matrix: torch.Tensor) -> torch.Tensor:
        """Cholesky decomposition of a symmetric positive-definite matrix."""
        TensorOps._check_tensor(matrix)
        if matrix.ndim != 2:
            raise ValueError("Input matrix must be 2-D")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Input matrix must be square")
        if not torch.allclose(matrix, matrix.transpose(-2, -1)):
            raise ValueError("Input matrix must be symmetric")
        return torch.linalg.cholesky(matrix)

    @staticmethod
    def matrix_inverse(matrix: torch.Tensor) -> torch.Tensor:
        """Inverse of a square matrix."""
        TensorOps._check_tensor(matrix)
        if matrix.ndim != 2:
            raise ValueError("Input matrix must be 2-D")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Input matrix must be square")
        orig_dtype = matrix.dtype
        inv = torch.linalg.inv(matrix.double()).to(orig_dtype)
        return inv

    @staticmethod
    def matrix_determinant(matrix: torch.Tensor) -> torch.Tensor:
        """Determinant of a square matrix."""
        TensorOps._check_tensor(matrix)
        if matrix.ndim != 2:
            raise ValueError("Input matrix must be 2-D")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Input matrix must be square")
        return torch.linalg.det(matrix)

    @staticmethod
    def matrix_rank(matrix: torch.Tensor) -> torch.Tensor:
        """Matrix rank of a 2-D tensor."""
        TensorOps._check_tensor(matrix)
        if matrix.ndim != 2:
            raise ValueError("Input matrix must be 2-D")
        return torch.linalg.matrix_rank(matrix)

    # --- Convolution Operations ---

    @staticmethod
    def convolve_1d(signal_x: torch.Tensor, kernel_w: torch.Tensor, mode: str = "valid") -> torch.Tensor:
        """1D convolution implemented with `torch.nn.functional.conv1d`."""
        import torch.nn.functional as F
        TensorOps._check_tensor(signal_x, kernel_w)
        if signal_x.ndim != 1 or kernel_w.ndim != 1:
            raise ValueError("Inputs must be 1D tensors")
        signal = signal_x.unsqueeze(0).unsqueeze(0)
        kernel = kernel_w.flip(0).unsqueeze(0).unsqueeze(0)
        if mode == "full":
            padding = kernel_w.numel() - 1
        elif mode == "same":
            padding = kernel_w.numel() // 2
        elif mode == "valid":
            padding = 0
        else:
            raise ValueError("mode must be one of 'full', 'same', or 'valid'")
        result = F.conv1d(signal, kernel, padding=padding)
        return result.squeeze(0).squeeze(0)

    @staticmethod
    def convolve_2d(image_I: torch.Tensor, kernel_K: torch.Tensor, mode: str = "valid") -> torch.Tensor:
        """2D convolution implemented with `torch.nn.functional.conv2d`."""
        import torch.nn.functional as F
        TensorOps._check_tensor(image_I, kernel_K)
        if image_I.ndim != 2 or kernel_K.ndim != 2:
            raise ValueError("Inputs must be 2D tensors")
        img = image_I.unsqueeze(0).unsqueeze(0)
        ker = kernel_K.flip(0, 1).unsqueeze(0).unsqueeze(0)
        if mode == "full":
            padding = (kernel_K.shape[0]-1, kernel_K.shape[1]-1)
        elif mode == "same":
            padding = (kernel_K.shape[0]//2, kernel_K.shape[1]//2)
        elif mode == "valid":
            padding = (0, 0)
        else:
            raise ValueError("mode must be one of 'full', 'same', or 'valid'")
        result = F.conv2d(img, ker, padding=padding)
        return result.squeeze(0).squeeze(0)

    @staticmethod
    def convolve_3d(volume: torch.Tensor, kernel: torch.Tensor, mode: str = "valid") -> torch.Tensor:
        """3D convolution implemented with `torch.nn.functional.conv3d`."""
        import torch.nn.functional as F
        TensorOps._check_tensor(volume, kernel)
        if volume.ndim != 3 or kernel.ndim != 3:
            raise ValueError("Inputs must be 3D tensors")

        vol = volume.unsqueeze(0).unsqueeze(0)
        ker = kernel.flip(0, 1, 2).unsqueeze(0).unsqueeze(0)

        if mode == "full":
            padding = (kernel.shape[0] - 1,
                       kernel.shape[1] - 1,
                       kernel.shape[2] - 1)
        elif mode == "same":
            padding = (kernel.shape[0] // 2,
                       kernel.shape[1] // 2,
                       kernel.shape[2] // 2)
        elif mode == "valid":
            padding = (0, 0, 0)
        else:
            raise ValueError("mode must be one of 'full', 'same', or 'valid'")

        result = F.conv3d(vol, ker, padding=padding)
        return result.squeeze(0).squeeze(0)

    # --- Statistical Operations ---

    @staticmethod
    def variance(tensor: torch.Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None,
                 unbiased: bool = False, keepdim: bool = False) -> torch.Tensor:
        """Variance of tensor elements."""
        TensorOps._check_tensor(tensor)
        return torch.var(tensor.float(), dim=dim, unbiased=unbiased, keepdim=keepdim)

    @staticmethod
    def std(tensor: torch.Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None,
            unbiased: bool = False, keepdim: bool = False) -> torch.Tensor:
        """Standard deviation of tensor elements."""
        TensorOps._check_tensor(tensor)
        return torch.std(tensor.float(), dim=dim, unbiased=unbiased, keepdim=keepdim)

    @staticmethod
    def covariance(matrix_X: torch.Tensor, matrix_Y: Optional[torch.Tensor] = None,
                   rowvar: bool = True, bias: bool = False, ddof: Optional[int] = None) -> torch.Tensor:
        """Estimate covariance matrix of the given variables."""
        TensorOps._check_tensor(matrix_X)
        if matrix_Y is not None:
            TensorOps._check_tensor(matrix_Y)
            matrices = [matrix_X, matrix_Y]
            mat = torch.cat(matrices, dim=0 if rowvar else 1)
        else:
            mat = matrix_X
        if not rowvar:
            mat = mat.t()
        if ddof is None:
            ddof = 0 if bias else 1
        mean = mat.mean(dim=1, keepdim=True)
        xm = mat - mean
        cov = xm @ xm.t() / (mat.shape[1] - ddof)
        return cov

    @staticmethod
    def correlation(matrix_X: torch.Tensor, matrix_Y: Optional[torch.Tensor] = None,
                    rowvar: bool = True) -> torch.Tensor:
        """Correlation coefficient matrix."""
        cov = TensorOps.covariance(matrix_X, matrix_Y, rowvar=rowvar, bias=False)
        diag = torch.sqrt(torch.diag(cov))
        denom = diag.unsqueeze(0) * diag.unsqueeze(1)
        return cov / denom

    @staticmethod
    def frobenius_norm(tensor: torch.Tensor) -> torch.Tensor:
        """Frobenius norm of a tensor."""
        TensorOps._check_tensor(tensor)
        return torch.linalg.norm(tensor, "fro")

    @staticmethod
    def l1_norm(tensor: torch.Tensor) -> torch.Tensor:
        """L1 norm of a tensor."""
        TensorOps._check_tensor(tensor)
        return torch.sum(torch.abs(tensor))

    @staticmethod
    def l2_norm(tensor: torch.Tensor) -> torch.Tensor:
        """L2 norm of a tensor."""
        TensorOps._check_tensor(tensor)
        return torch.linalg.norm(tensor, 2)

    @staticmethod
    def p_norm(tensor: torch.Tensor, p: float) -> torch.Tensor:
        """General p-norm of a tensor."""
        TensorOps._check_tensor(tensor)
        if not isinstance(p, (int, float)):
            raise TypeError("p must be a numeric value")
        if p <= 0:
            raise ValueError("p must be positive")
        return torch.linalg.norm(tensor, p)

    @staticmethod
    def nuclear_norm(matrix: torch.Tensor) -> torch.Tensor:
        """Nuclear norm (sum of singular values) for a 2-D tensor."""
        TensorOps._check_tensor(matrix)
        if matrix.ndim != 2:
            raise ValueError(
                f"Nuclear norm expects a 2-D tensor, got shape {matrix.shape}"
            )
        return torch.linalg.matrix_norm(matrix, ord="nuc")

