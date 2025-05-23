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


# Example Usage
if __name__ == "__main__":
    t1 = torch.tensor([[1., 2.], [3., 4.]])
    t2 = torch.tensor([[5., 6.], [7., 8.]])
    t3 = torch.tensor([1., 2.])
    t4 = torch.tensor([3., 4.])

    print("--- Arithmetic ---")
    print("Add:", TensorOps.add(t1, t2))
    print("Subtract:", TensorOps.subtract(t1, 5.0))
    print("Multiply:", TensorOps.multiply(t1, t2))
    print("Divide:", TensorOps.divide(t1, 2.0))
    try:
        TensorOps.divide(t1, torch.tensor([[1., 0.], [1., 1.]]))
    except ValueError as e:
        print("Caught expected division by zero warning/error.") # Logging handles the warning


    print("\n--- Matrix/Dot ---")
    print("Matmul:", TensorOps.matmul(t1, t2))
    print("Dot:", TensorOps.dot(t3, t4))
    try:
        TensorOps.matmul(t1, t3) # Incompatible shapes
    except ValueError as e:
        print(f"Caught expected matmul error: {e}")

    print("\n--- Reduction ---")
    print("Sum (all):", TensorOps.sum(t1))
    print("Mean (dim 0):", TensorOps.mean(t1, dim=0))
    print("Max (dim 1):", TensorOps.max(t1, dim=1)) # Returns (values, indices)

    print("\n--- Reshaping ---")
    print("Reshape:", TensorOps.reshape(t1, (4, 1)))
    print("Transpose:", TensorOps.transpose(t1, 0, 1))
    print("Permute:", TensorOps.permute(torch.rand(2,3,4), (1, 2, 0)).shape)

    print("\n--- Concat/Stack ---")
    print("Concatenate (dim 0):", TensorOps.concatenate([t1, t2], dim=0))
    print("Stack (dim 0):", TensorOps.stack([t1, t1], dim=0)) # Stacks along new dim 0

    print("\n--- Advanced ---")
    print("Einsum (trace):", TensorOps.einsum('ii->', t1)) # Trace
    print("Einsum (batch matmul):", TensorOps.einsum('bij,bjk->bik', torch.rand(5, 2, 3), torch.rand(5, 3, 4)).shape)

    print("\n--- Error Handling Example ---")
    try:
        TensorOps.add(t1, "not a tensor")
    except TypeError as e:
        print(f"Caught expected type error: {e}")

    try:
         TensorOps._check_shape(t1, (None, 3), "Example Op") # Wrong dim size
    except ValueError as e:
         print(f"Caught expected shape error: {e}")