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
from tensorly.decomposition import parafac, tucker, tensor_train
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

    # --- Tensor Decomposition Operations ---

    @staticmethod
    def cp_decomposition(tensor: torch.Tensor, rank: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Performs CP (CANDECOMP/PARAFAC) decomposition on a tensor.

        The CP decomposition factorizes a tensor into a sum of rank-one tensors.
        It returns weights and a list of factor matrices.

        Args:
            tensor (torch.Tensor): The input tensor to decompose.
            rank (int): The desired rank of the decomposition.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: A tuple containing:
                - torch_weights (torch.Tensor): A 1D tensor of weights.
                - torch_factors (List[torch.Tensor]): A list of factor matrices.
                  Each factor matrix has `rank` columns.

        Raises:
            TypeError: If the input `tensor` is not a PyTorch tensor.
            ValueError: If `rank` is not a positive integer.
            ValueError: If the input `tensor` has fewer than 2 dimensions.
            RuntimeError: If the CP decomposition fails (e.g., convergence issues).
        """
        TensorOps._check_tensor(tensor)

        if not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"Rank must be a positive integer, but got {rank}.")

        if tensor.ndim < 2:
            raise ValueError(f"CP decomposition requires a tensor with at least 2 dimensions, but got {tensor.ndim} dimensions (shape {tensor.shape}).")

        logging.info(f"Performing CP decomposition with rank {rank} on tensor of shape {tensor.shape}")

        try:
            # Convert to float32 for tensorly operations, common in ML
            tl_tensor = tl.tensor(tensor.float().numpy())

            # Perform CP decomposition using TensorLy
            # parafac returns a CPTensor instance (namedtuple: (weights, factors))
            cp_tensor_tl = parafac(tl_tensor, rank=rank)

            weights_np = cp_tensor_tl.weights
            factors_np = cp_tensor_tl.factors

            # Convert weights and factors back to PyTorch tensors, maintaining float32
            if weights_np is not None:
                torch_weights = torch.from_numpy(weights_np).type(torch.float32)
            else:
                # Handle cases where weights might be absorbed or not returned distinctly
                # For now, assume weights are always returned. If not, this might need adjustment
                # e.g., return torch.ones(rank, dtype=torch.float32) or raise error.
                # Based on TensorLy's standard parafac, weights should be present.
                logging.warning("CP decomposition returned None for weights. This is unexpected for standard parafac.")
                # Fallback or error based on expected behavior. For now, let's assume it's an error if None.
                raise RuntimeError("CP decomposition failed to return weights.")

            torch_factors = [torch.from_numpy(factor).type(torch.float32) for factor in factors_np]

            logging.info(f"CP decomposition successful. Weights shape: {torch_weights.shape}, Number of factor matrices: {len(torch_factors)}")
            if torch_factors:
                 logging.info(f"Shape of first factor matrix: {torch_factors[0].shape}")

            return torch_weights, torch_factors

        except Exception as e:
            logging.error(f"Error during CP decomposition: {e}. Tensor shape: {tensor.shape}, Rank: {rank}")
            # Re-raise as a RuntimeError to signal failure in the operation
            raise RuntimeError(f"CP decomposition failed. Original error: {e}")

    @staticmethod
    def hosvd(tensor: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Performs Higher-Order Singular Value Decomposition (HOSVD) on a tensor.

        HOSVD is a specific case of Tucker decomposition where the factor matrices
        are constrained to be orthogonal. It decomposes the input tensor into a
        core tensor and a list of orthogonal factor matrices, one for each mode.
        The ranks for HOSVD are taken to be the full dimensions of the tensor.

        Args:
            tensor (torch.Tensor): The input tensor to decompose. Must have at
                                   least 1 dimension.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: A tuple containing:
                - torch_core (torch.Tensor): The core tensor of the HOSVD.
                  Its shape will be the same as the input tensor.
                - torch_factors (List[torch.Tensor]): A list of orthogonal
                  factor matrices. `torch_factors[i]` will have shape
                  `(tensor.shape[i], tensor.shape[i])`.

        Raises:
            TypeError: If the input `tensor` is not a PyTorch tensor.
            ValueError: If the input `tensor` has less than 1 dimension.
            RuntimeError: If the HOSVD (via Tucker decomposition) fails.
        """
        TensorOps._check_tensor(tensor)

        if tensor.ndim < 2:
            # HOSVD is typically defined for tensors of order 2 or higher.
            # TensorLy's TuckerTensor representation also expects at least 2 factors.
            raise ValueError(f"HOSVD requires a tensor with at least 2 dimensions, but got {tensor.ndim} dimensions (shape {tensor.shape}).")

        # For HOSVD, ranks are the full dimensions of the tensor.
        ranks = list(tensor.shape)

        logging.info(f"Performing HOSVD (Tucker with full ranks {ranks}) on tensor of shape {tensor.shape}")

        try:
            # Convert to float32 for tensorly operations
            tl_tensor = tl.tensor(tensor.float().numpy())

            # Perform HOSVD using TensorLy's tucker with full ranks and init='svd' (default)
            # This ensures orthogonal factors are computed via SVD of unfoldings.
            core_np, factors_np = tucker(tl_tensor, rank=ranks) # init='svd' is default

            # Convert core tensor and factors back to PyTorch tensors
            torch_core = torch.from_numpy(core_np.copy()).type(torch.float32)
            torch_factors = [torch.from_numpy(factor.copy()).type(torch.float32) for factor in factors_np]

            logging.info(f"HOSVD successful. Core tensor shape: {torch_core.shape}, Number of factor matrices: {len(torch_factors)}")
            if torch_factors:
                 logging.info(f"Shape of first factor matrix: {torch_factors[0].shape}")

            return torch_core, torch_factors

        except Exception as e:
            logging.error(f"Error during HOSVD: {e}. Tensor shape: {tensor.shape}")
            # Re-raise as a RuntimeError to signal failure
            raise RuntimeError(f"HOSVD failed. Original error: {e}")

    @staticmethod
    def tucker_decomposition(tensor: torch.Tensor, ranks: List[int]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Performs Tucker decomposition on a tensor.

        The Tucker decomposition factorizes a tensor into a core tensor and a set
        of factor matrices for each mode.

        Args:
            tensor (torch.Tensor): The input tensor to decompose.
            ranks (List[int]): A list of desired ranks for each mode of the tensor.
                               The length of the list must match the number of
                               dimensions of the input tensor.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: A tuple containing:
                - torch_core (torch.Tensor): The core tensor of the decomposition.
                  Its shape will be according to the specified `ranks`.
                - torch_factors (List[torch.Tensor]): A list of factor matrices.
                  Each factor matrix `torch_factors[i]` will have shape
                  `(tensor.shape[i], ranks[i])`.

        Raises:
            TypeError: If the input `tensor` is not a PyTorch tensor.
            ValueError: If `ranks` is not a list of positive integers,
                        if its length does not match `tensor.ndim`, or
                        if any rank is not `1 <= rank_i <= tensor.shape[i]`.
            RuntimeError: If the Tucker decomposition fails.
        """
        TensorOps._check_tensor(tensor)

        if not isinstance(ranks, list) or not all(isinstance(r, int) and r > 0 for r in ranks):
            raise ValueError(f"Ranks must be a list of positive integers, but got {ranks}.")

        if len(ranks) != tensor.ndim:
            raise ValueError(f"Length of ranks list ({len(ranks)}) must match tensor dimensionality ({tensor.ndim}).")

        for i, r in enumerate(ranks):
            if not (1 <= r <= tensor.shape[i]):
                raise ValueError(f"Rank for mode {i} ({r}) is out of valid range [1, {tensor.shape[i]}].")

        logging.info(f"Performing Tucker decomposition with ranks {ranks} on tensor of shape {tensor.shape}")

        try:
            # Convert to float32 for tensorly operations
            tl_tensor = tl.tensor(tensor.float().numpy())

            # Perform Tucker decomposition using TensorLy
            # tucker returns (core, [factor_matrix_0, ..., factor_matrix_N])
            core_np, factors_np = tucker(tl_tensor, rank=ranks)

            # Convert core tensor and factors back to PyTorch tensors, maintaining float32
            # Using .copy() as TensorLy might return views from its operations
            torch_core = torch.from_numpy(core_np.copy()).type(torch.float32)
            torch_factors = [torch.from_numpy(factor.copy()).type(torch.float32) for factor in factors_np]

            logging.info(f"Tucker decomposition successful. Core tensor shape: {torch_core.shape}, Number of factor matrices: {len(torch_factors)}")
            if torch_factors:
                 logging.info(f"Shape of first factor matrix: {torch_factors[0].shape}")

            return torch_core, torch_factors

        except Exception as e:
            logging.error(f"Error during Tucker decomposition: {e}. Tensor shape: {tensor.shape}, Ranks: {ranks}")
            # Re-raise as a RuntimeError to signal failure in the operation
            raise RuntimeError(f"Tucker decomposition failed. Original error: {e}")

    @staticmethod
    def tt_decomposition(tensor: torch.Tensor, rank: Union[int, List[int]]) -> List[torch.Tensor]:
        """
        Performs Tensor Train (TT) decomposition on a tensor.

        The TT decomposition factorizes a tensor into a sequence of 3D cores
        (factors). Also known as Matrix Product State (MPS) decomposition.

        Args:
            tensor (torch.Tensor): The input tensor to decompose. Must have ndim >= 1.
            rank (Union[int, List[int]]): The TT-ranks.
                - If int: the maximum TT-rank.
                - If List[int]: the list of internal TT-ranks [r_1, ..., r_{N-1}],
                  where N is the order of the tensor. The length of the list must
                  be tensor.ndim - 1. Boundary ranks r_0 and r_N are implicitly 1.

        Returns:
            List[torch.Tensor]: A list of TT factor tensors (cores).
                Each factor G_k is a 3D tensor of shape (rank_{k-1}, tensor.shape[k], rank_k).
                For the first core G_0, shape is (1, tensor.shape[0], rank_1).
                For the last core G_{N-1}, shape is (rank_{N-1}, tensor.shape[N-1], 1).

        Raises:
            TypeError: If the input `tensor` is not a PyTorch tensor.
            ValueError: If `tensor.ndim` is 0.
                        If `rank` is an int and not positive.
                        If `rank` is a list and its length is not `tensor.ndim - 1` (for ndim > 1),
                        or if it's not empty (for ndim == 1).
                        If any rank in the list is not positive.
            RuntimeError: If the TT decomposition fails.
        """
        TensorOps._check_tensor(tensor)

        if tensor.ndim == 0:
            raise ValueError("TT decomposition requires a tensor with at least 1 dimension, but got 0.")

        # Validate user-provided rank and determine the rank parameter for TensorLy's tensor_train
        param_for_tensor_train: Union[int, List[int]]
        if isinstance(rank, int): # User provided a single maximum internal rank
            if rank <= 0:
                raise ValueError(f"If rank is an integer, it must be positive, but got {rank}.")
            if tensor.ndim == 1:
                param_for_tensor_train = 1 # tensor_train for 1D tensor expects rank=1 (int)
            else:
                # Assuming the tensor_train in env expects full N+1 ranks list
                param_for_tensor_train = [1] + [rank] * (tensor.ndim - 1) + [1]
        elif isinstance(rank, list): # User provided a list of N-1 internal ranks
            if tensor.ndim == 1:
                if rank: # Non-empty list for 1D tensor
                    raise ValueError(f"For a 1D tensor, rank list must be empty for user input, but got {rank}.")
                param_for_tensor_train = 1 # tensor_train for 1D tensor expects rank=1 (int)
            else: # tensor.ndim > 1
                if len(rank) != tensor.ndim - 1:
                    raise ValueError(f"Rank list length must be tensor.ndim - 1 ({tensor.ndim - 1}), but got {len(rank)} for tensor of shape {tensor.shape}.")
                if not all(isinstance(r, int) and r > 0 for r in rank):
                    raise ValueError(f"All ranks in the list must be positive integers, but got {rank}.")
                # Assuming the tensor_train in env expects full N+1 ranks list
                param_for_tensor_train = [1] + rank + [1]
        else:
            raise TypeError(f"Rank must be an int or a list of ints, but got {type(rank)}.")

        logging.info(f"Performing TT decomposition with TensorLy rank parameter {param_for_tensor_train} (user input {rank}) on tensor of shape {tensor.shape}")

        try:
            tl_tensor = tl.tensor(tensor.float().numpy())

            # Perform TT decomposition using TensorLy
            result_tl = tensor_train(tl_tensor, rank=param_for_tensor_train)

            factors_np: List[np.ndarray]
            if hasattr(result_tl, 'factors'): # Modern TensorLy (>=0.8.0) returns TensorTrain object
                factors_np = result_tl.factors
            elif isinstance(result_tl, list): # Older TensorLy might return list of factors
                factors_np = result_tl
            elif tensor.ndim == 1 and isinstance(result_tl, tl.ndarray): # Handle 1D case if it returns a single factor array
                factors_np = [result_tl]
            else: # Unexpected return type
                raise RuntimeError(f"TensorLy's tensor_train returned an unexpected type: {type(result_tl)}. Value: {result_tl}")

            # Convert factors back to PyTorch tensors
            torch_factors = [torch.from_numpy(factor.copy()).type(torch.float32) for factor in factors_np]

            logging.info(f"TT decomposition successful. Number of TT cores: {len(torch_factors)}")
            if torch_factors:
                for i, core_factor in enumerate(torch_factors):
                    logging.info(f"  TT Core {i} shape: {core_factor.shape}")

            return torch_factors

        except Exception as e:
            logging.error(f"Error during TT decomposition: {e}. Tensor shape: {tensor.shape}, Rank(s): {rank}")
            raise RuntimeError(f"TT decomposition failed. Original error: {e}")


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

    print("\n--- Tensor Decomposition ---")
    # Example for CP Decomposition
    # Create a synthetic 3D tensor
    data = torch.arange(24, dtype=torch.float32).reshape((2, 3, 4))
    rank_to_decompose = 2
    print(f"Original tensor shape for CP: {data.shape}")
    try:
        weights, factors = TensorOps.cp_decomposition(data, rank=rank_to_decompose)
        print(f"CP Decomposition (Rank {rank_to_decompose}):")
        print(f"  Weights shape: {weights.shape}")
        print(f"  Number of factors: {len(factors)}")
        for i, factor in enumerate(factors):
            print(f"  Factor {i} shape: {factor.shape}")

        # Example with a 2D tensor (matrix)
        matrix_data = torch.rand(5, 4)
        rank_matrix_decompose = 3
        print(f"\nOriginal matrix shape for CP: {matrix_data.shape}")
        weights_matrix, factors_matrix = TensorOps.cp_decomposition(matrix_data, rank=rank_matrix_decompose)
        print(f"CP Decomposition (Rank {rank_matrix_decompose}) for matrix:")
        print(f"  Weights shape: {weights_matrix.shape}")
        print(f"  Number of factors: {len(factors_matrix)}")
        for i, factor in enumerate(factors_matrix):
            print(f"  Factor {i} shape: {factor.shape}")

    except RuntimeError as e:
        print(f"CP Decomposition example failed: {e}")
    except FileNotFoundError:
        print("TensorLy or its dependencies might not be installed. Skipping CP decomposition example.")
    except Exception as e: # Catch any other unexpected errors during example run
        print(f"An unexpected error occurred in CP decomposition example: {e}")


    # Example for Tucker Decomposition
    data_tucker = torch.rand(3, 4, 5, dtype=torch.float32) # Example 3D tensor
    # Ranks for each mode - must be <= corresponding dimension
    ranks_tucker = [2, 3, 2]
    print(f"\nOriginal tensor shape for Tucker: {data_tucker.shape}, Ranks: {ranks_tucker}")
    try:
        core, factors_tucker = TensorOps.tucker_decomposition(data_tucker, ranks_tucker)
        print(f"Tucker Decomposition (Ranks {ranks_tucker}):")
        print(f"  Core tensor shape: {core.shape}") # Expected: (ranks_tucker[0], ranks_tucker[1], ranks_tucker[2])
        print(f"  Number of factors: {len(factors_tucker)}")
        for i, factor in enumerate(factors_tucker):
            print(f"  Factor {i} shape: {factor.shape}") # Expected: (data_tucker.shape[i], ranks_tucker[i])

        # Optional: Reconstruction example
        # Convert core and factors back to NumPy for TensorLy's tucker_to_tensor
        core_np_rec = core.numpy()
        factors_np_rec = [f.numpy() for f in factors_tucker]
        reconstructed_tl = tl.tucker_to_tensor((core_np_rec, factors_np_rec))
        reconstructed_torch = torch.from_numpy(reconstructed_tl).float()

        error = torch.norm(data_tucker - reconstructed_torch) / torch.norm(data_tucker)
        print(f"  Reconstruction error (relative): {error.item():.4f}")

    except RuntimeError as e:
        print(f"Tucker Decomposition example failed: {e}")
    except FileNotFoundError: # In case tensorly was not installed by previous examples
        print("TensorLy or its dependencies might not be installed. Skipping Tucker decomposition example.")
    except Exception as e:
        print(f"An unexpected error occurred in Tucker decomposition example: {e}")


    # Example for HOSVD
    data_hosvd = torch.rand(2, 3, 4, dtype=torch.float32) # Example 3D tensor
    print(f"\nOriginal tensor shape for HOSVD: {data_hosvd.shape}")
    try:
        core_hosvd, factors_hosvd = TensorOps.hosvd(data_hosvd)
        print("HOSVD:")
        print(f"  Core tensor shape: {core_hosvd.shape}") # Expected: same as data_hosvd.shape
        print(f"  Number of factors: {len(factors_hosvd)}")
        for i, factor in enumerate(factors_hosvd):
            print(f"  Factor {i} shape: {factor.shape}") # Expected: (data_hosvd.shape[i], data_hosvd.shape[i])
            # Verify orthogonality: factor.T @ factor should be close to identity
            identity_approx = torch.matmul(factor.T, factor)
            identity_check = torch.allclose(identity_approx, torch.eye(factor.shape[1]), atol=1e-5)
            print(f"  Factor {i} orthogonality check: {identity_check}")

        # Reconstruction (should be near perfect for HOSVD with full ranks)
        # Convert core and factors back to NumPy for TensorLy's tucker_to_tensor
        core_np_rec_h = core_hosvd.numpy()
        factors_np_rec_h = [f.numpy() for f in factors_hosvd]
        reconstructed_tl_h = tl.tucker_to_tensor((core_np_rec_h, factors_np_rec_h))
        reconstructed_torch_h = torch.from_numpy(reconstructed_tl_h).float()

        error_h = torch.norm(data_hosvd - reconstructed_torch_h) / torch.norm(data_hosvd)
        print(f"  Reconstruction error (relative): {error_h.item():.6f}") # Expect very low error

    except RuntimeError as e:
        print(f"HOSVD example failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in HOSVD example: {e}")


    # Example for Tensor Train (TT) Decomposition
    # For a 3D tensor of shape (I0, I1, I2)
    data_tt_3d = torch.rand(3, 4, 5, dtype=torch.float32)
    # Internal ranks [r1, r2] for a 3D tensor (N=3, N-1=2 ranks)
    ranks_tt_3d = [2, 3]
    print(f"\nOriginal tensor shape for TT (3D): {data_tt_3d.shape}, Internal Ranks: {ranks_tt_3d}")
    try:
        factors_tt_3d = TensorOps.tt_decomposition(data_tt_3d, ranks_tt_3d)
        print("TT Decomposition (3D):")
        for i, factor in enumerate(factors_tt_3d):
            print(f"  TT Core {i} shape: {factor.shape}")
            # Expected shapes for 3D tensor (I0,I1,I2) and ranks [r1,r2]:
            # G0: (1, I0, r1)
            # G1: (r1, I1, r2)
            # G2: (r2, I2, 1)

        # Optional: Reconstruction example
        # Convert factors back to NumPy for TensorLy's tt_to_tensor
        factors_np_rec_tt = [f.numpy() for f in factors_tt_3d]
        reconstructed_tl_tt = tl.tt_to_tensor(factors_np_rec_tt) # Pass list of factors directly
        reconstructed_torch_tt = torch.from_numpy(reconstructed_tl_tt).float()

        error_tt_3d = torch.norm(data_tt_3d - reconstructed_torch_tt) / torch.norm(data_tt_3d)
        print(f"  Reconstruction error (3D, relative): {error_tt_3d.item():.4f}")

    except RuntimeError as e:
        print(f"TT Decomposition example (3D) failed: {e}")
    except Exception as e:
        print(f"An unexpected error in TT (3D) example: {e}")

    # For a 1D tensor (vector)
    data_tt_1d = torch.rand(10, dtype=torch.float32)
    ranks_tt_1d = [] # Empty list for 1D tensor, will be handled as rank=1 internally
    print(f"\nOriginal tensor shape for TT (1D): {data_tt_1d.shape}, Internal Ranks: {ranks_tt_1d} (becomes rank=1)")
    try:
        factors_tt_1d = TensorOps.tt_decomposition(data_tt_1d, ranks_tt_1d)
        print("TT Decomposition (1D):")
        for i, factor in enumerate(factors_tt_1d):
            print(f"  TT Core {i} shape: {factor.shape}") # Expected: G0: (1, I0, 1)

        factors_np_rec_tt_1d = [f.numpy() for f in factors_tt_1d]
        reconstructed_tl_tt_1d = tl.tt_to_tensor(factors_np_rec_tt_1d)
        reconstructed_torch_tt_1d = torch.from_numpy(reconstructed_tl_tt_1d).float()
        error_tt_1d = torch.norm(data_tt_1d - reconstructed_torch_tt_1d) / torch.norm(data_tt_1d)
        print(f"  Reconstruction error (1D, relative): {error_tt_1d.item():.6f}")


    except RuntimeError as e:
        print(f"TT Decomposition example (1D) failed: {e}")
    except Exception as e:
        print(f"An unexpected error in TT (1D) example: {e}")