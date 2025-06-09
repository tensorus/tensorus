# tensorus/tensor_decompositions.py
"""
Provides a library of tensor decomposition operations for Tensorus.
This module will contain various tensor decomposition algorithms
like CP, Tucker, TT, TR, etc.
"""

import torch
import tensorly as tl
from typing import List, Tuple, Union, Optional, Dict

from tensorus.tensor_ops import TensorOps
import logging
from tensorly.decomposition import (
    parafac,
    tucker,
    tensor_train,
    tensor_ring,
    non_negative_parafac,
    non_negative_tucker,
    partial_tucker,
    # block_term_decomposition removed as it's not found
)
from tensorly.cp_tensor import cp_to_tensor
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.tt_tensor import tt_to_tensor
from tensorly.tr_tensor import tr_to_tensor
# For BTD reconstruction - no specific import as the decomposition function itself is problematic.

import numpy as np
from scipy.fft import fft, ifft # For t-SVD


class TensorDecompositionOps:
    """
    A static library class providing tensor decomposition operations.
    All methods are static and operate on provided torch.Tensor objects,
    returning decomposition factors also as torch.Tensor objects.
    """

    @staticmethod
    def _t_product(A_torch: torch.Tensor, B_torch: torch.Tensor) -> torch.Tensor:
        """
        Helper function for t-product of two 3-way tensors A and B along the 3rd axis.
        A: (n1, p, n3)
        B: (p, n2, n3)
        Output: (n1, n2, n3)
        """
        A_np = A_torch.numpy()
        B_np = B_torch.numpy()

        if A_np.ndim != 3 or B_np.ndim != 3:
            raise ValueError("t-product is defined for 3-way tensors.")
        if A_np.shape[2] != B_np.shape[2]:
            raise ValueError(f"Third dimensions (tubes) for t-product must match, got {A_np.shape[2]} and {B_np.shape[2]}.")
        # Inner dimension check A.shape[1] vs B.shape[0] is handled by np.matmul per slice.

        A_fft = fft(A_np, axis=2)
        B_fft = fft(B_np, axis=2)

        # Correct initialization for C_fft_np
        C_fft_np = np.zeros((A_np.shape[0], B_np.shape[1], A_np.shape[2]), dtype=complex)

        for i in range(A_fft.shape[2]): # Loop over tubes
            try:
                C_fft_np[:, :, i] = np.matmul(A_fft[:, :, i], B_fft[:, :, i])
            except ValueError as e: # Catch matmul errors e.g. shape mismatch for inner dims
                raise ValueError(f"Slice-wise matrix multiplication failed for slice {i}: {e}") from e

        C_ifft = ifft(C_fft_np, axis=2)

        # If original tensors were real, result should be real
        # (although FFT can introduce small imaginary parts due to precision)
        C_final_np = C_ifft.real if torch.is_floating_point(A_torch) and torch.is_floating_point(B_torch) else C_ifft
        return torch.from_numpy(C_final_np.copy()).type(A_torch.dtype)


    @staticmethod
    def cp_decomposition(tensor: torch.Tensor, rank: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        TensorOps._check_tensor(tensor)
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"Rank must be a positive integer, but got {rank}.")
        if tensor.ndim < 2:
            raise ValueError(f"CP decomposition requires a tensor with at least 2 dimensions, but got {tensor.ndim} dimensions (shape {tensor.shape}).")
        logging.info(f"Performing CP decomposition with rank {rank} on tensor of shape {tensor.shape}")
        try:
            tl_tensor = tl.tensor(tensor.float().numpy())
            cp_tensor_tl = parafac(tl_tensor, rank=rank)
            weights_np = cp_tensor_tl.weights
            factors_np = cp_tensor_tl.factors
            if weights_np is not None:
                torch_weights = torch.from_numpy(weights_np).type(torch.float32)
            else:
                logging.warning("CP decomposition returned None for weights. This is unexpected for standard parafac.")
                raise RuntimeError("CP decomposition failed to return weights.")
            torch_factors = [torch.from_numpy(factor.copy()).type(torch.float32) for factor in factors_np]
            logging.info(f"CP decomposition successful. Weights shape: {torch_weights.shape}, Number of factor matrices: {len(torch_factors)}")
            if torch_factors:
                 logging.info(f"Shape of first factor matrix: {torch_factors[0].shape}")
            return torch_weights, torch_factors
        except Exception as e:
            logging.error(f"Error during CP decomposition: {e}. Tensor shape: {tensor.shape}, Rank: {rank}")
            raise RuntimeError(f"CP decomposition failed. Original error: {e}")

    @staticmethod
    def ntf_cp_decomposition(tensor: torch.Tensor, rank: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        TensorOps._check_tensor(tensor)
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"Rank must be a positive integer, but got {rank}.")
        if torch.any(tensor < 0):
            raise ValueError("Input tensor for NTF-CP must be non-negative.")
        if tensor.ndim < 2:
            raise ValueError(f"NTF-CP decomposition requires a tensor with at least 2 dimensions, but got {tensor.ndim}.")
        logging.info(f"Performing NTF-CP decomposition with rank {rank} on tensor of shape {tensor.shape}")
        try:
            tl_tensor = tl.tensor(tensor.float().numpy())
            cp_tensor_tl = non_negative_parafac(tl_tensor, rank=rank, init='random', tol=1e-6, n_iter_max=100)
            weights_np = cp_tensor_tl.weights
            factors_np = cp_tensor_tl.factors
            if weights_np is not None:
                torch_weights = torch.from_numpy(weights_np.copy()).type(torch.float32)
            else:
                logging.warning("NTF-CP decomposition returned None for weights.")
                raise RuntimeError("NTF-CP decomposition failed to return weights.")
            torch_factors = [torch.from_numpy(factor.copy()).type(torch.float32) for factor in factors_np]
            if torch.any(torch_weights < -1e-6):
                logging.warning("NTF-CP weights have unexpected negative values.")
            for i, f in enumerate(torch_factors):
                if torch.any(f < -1e-6):
                    logging.warning(f"NTF-CP factor matrix {i} has unexpected negative values.")
            logging.info(f"NTF-CP decomposition successful. Weights shape: {torch_weights.shape}, Number of factor matrices: {len(torch_factors)}")
            return torch_weights, torch_factors
        except Exception as e:
            logging.error(f"Error during NTF-CP decomposition: {e}. Tensor shape: {tensor.shape}, Rank: {rank}")
            raise RuntimeError(f"NTF-CP decomposition failed. Original error: {e}")

    @staticmethod
    def non_negative_tucker(tensor: torch.Tensor, ranks: List[int]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Non-negative Tucker decomposition using tensorly's non_negative_tucker."""
        TensorOps._check_tensor(tensor)
        if not isinstance(ranks, list) or not all(isinstance(r, int) and r > 0 for r in ranks):
            raise ValueError(f"Ranks must be a list of positive integers, but got {ranks}.")
        if len(ranks) != tensor.ndim:
            raise ValueError(f"Length of ranks list ({len(ranks)}) must match tensor dimensionality ({tensor.ndim}).")
        for i, r_val in enumerate(ranks):
            if not (1 <= r_val <= tensor.shape[i]):
                raise ValueError(f"Rank for mode {i} ({r_val}) is out of valid range [1, {tensor.shape[i]}].")
        if torch.any(tensor < 0):
            raise ValueError("Input tensor for non-negative Tucker must be non-negative.")

        logging.info(f"Performing non-negative Tucker decomposition with ranks {ranks} on tensor of shape {tensor.shape}")
        try:
            tl_tensor = tl.tensor(tensor.float().numpy())
            core_np, factors_np = non_negative_tucker(tl_tensor, rank=ranks)
            torch_core = torch.from_numpy(core_np.copy()).type(torch.float32)
            torch_factors = [torch.from_numpy(f.copy()).type(torch.float32) for f in factors_np]
            return torch_core, torch_factors
        except Exception as e:
            logging.error(f"Error during non-negative Tucker decomposition: {e}. Tensor shape: {tensor.shape}, Ranks: {ranks}")
            raise RuntimeError(f"Non-negative Tucker decomposition failed. Original error: {e}")

    @staticmethod
    def tucker_decomposition(tensor: torch.Tensor, ranks: List[int]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        TensorOps._check_tensor(tensor)
        if not isinstance(ranks, list) or not all(isinstance(r, int) and r > 0 for r in ranks):
            raise ValueError(f"Ranks must be a list of positive integers, but got {ranks}.")
        if len(ranks) != tensor.ndim:
            raise ValueError(f"Length of ranks list ({len(ranks)}) must match tensor dimensionality ({tensor.ndim}).")
        for i, r_val in enumerate(ranks):
            if not (1 <= r_val <= tensor.shape[i]):
                raise ValueError(f"Rank for mode {i} ({r_val}) is out of valid range [1, {tensor.shape[i]}].")
        logging.info(f"Performing Tucker decomposition with ranks {ranks} on tensor of shape {tensor.shape}")
        try:
            tl_tensor = tl.tensor(tensor.float().numpy())
            core_np, factors_np = tucker(tl_tensor, rank=ranks)
            torch_core = torch.from_numpy(core_np.copy()).type(torch.float32)
            torch_factors = [torch.from_numpy(factor.copy()).type(torch.float32) for factor in factors_np]
            logging.info(f"Tucker decomposition successful. Core tensor shape: {torch_core.shape}, Number of factor matrices: {len(torch_factors)}")
            if torch_factors:
                 logging.info(f"Shape of first factor matrix: {torch_factors[0].shape}")
            return torch_core, torch_factors
        except Exception as e:
            logging.error(f"Error during Tucker decomposition: {e}. Tensor shape: {tensor.shape}, Ranks: {ranks}")
            raise RuntimeError(f"Tucker decomposition failed. Original error: {e}")

    @staticmethod
    def partial_tucker(tensor: torch.Tensor, ranks: List[int]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Higher-Order Orthogonal Iteration (HOOI) via tensorly.partial_tucker."""
        TensorOps._check_tensor(tensor)
        if not isinstance(ranks, list) or not all(isinstance(r, int) and r > 0 for r in ranks):
            raise ValueError(f"Ranks must be a list of positive integers, but got {ranks}.")
        if len(ranks) != tensor.ndim:
            raise ValueError(f"Length of ranks list ({len(ranks)}) must match tensor dimensionality ({tensor.ndim}).")
        for i, r_val in enumerate(ranks):
            if not (1 <= r_val <= tensor.shape[i]):
                raise ValueError(f"Rank for mode {i} ({r_val}) is out of valid range [1, {tensor.shape[i]}].")

        logging.info(f"Performing partial Tucker (HOOI) with ranks {ranks} on tensor of shape {tensor.shape}")
        try:
            tl_tensor = tl.tensor(tensor.float().numpy())
            result = partial_tucker(tl_tensor, rank=ranks)
            if isinstance(result[0], tuple):
                core_np, factors_np = result[0]
            else:
                core_np, factors_np = result
            torch_core = torch.from_numpy(core_np.copy()).type(torch.float32)
            torch_factors = [torch.from_numpy(f.copy()).type(torch.float32) for f in factors_np]
            return torch_core, torch_factors
        except Exception as e:
            logging.error(f"Error during partial Tucker decomposition: {e}. Tensor shape: {tensor.shape}, Ranks: {ranks}")
            raise RuntimeError(f"Partial Tucker decomposition failed. Original error: {e}")

    @staticmethod
    def hosvd(tensor: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        TensorOps._check_tensor(tensor)
        if tensor.ndim < 2:
            raise ValueError(f"HOSVD requires a tensor with at least 2 dimensions, but got {tensor.ndim} dimensions (shape {tensor.shape}).")
        ranks = list(tensor.shape)
        logging.info(f"Performing HOSVD (Tucker with full ranks {ranks}) on tensor of shape {tensor.shape}")
        try:
            tl_tensor = tl.tensor(tensor.float().numpy())
            core_np, factors_np = tucker(tl_tensor, rank=ranks)
            torch_core = torch.from_numpy(core_np.copy()).type(torch.float32)
            torch_factors = [torch.from_numpy(factor.copy()).type(torch.float32) for factor in factors_np]
            logging.info(f"HOSVD successful. Core tensor shape: {torch_core.shape}, Number of factor matrices: {len(torch_factors)}")
            if torch_factors:
                 logging.info(f"Shape of first factor matrix: {torch_factors[0].shape}")
            return torch_core, torch_factors
        except Exception as e:
            logging.error(f"Error during HOSVD: {e}. Tensor shape: {tensor.shape}")
            raise RuntimeError(f"HOSVD failed. Original error: {e}")

    @staticmethod
    def tt_decomposition(tensor: torch.Tensor, rank: Union[int, List[int]]) -> List[torch.Tensor]:
        TensorOps._check_tensor(tensor)
        if tensor.ndim == 0:
            raise ValueError("TT decomposition requires a tensor with at least 1 dimension, but got 0.")
        param_for_tensor_train: Union[int, List[int]]
        if isinstance(rank, int):
            if rank <= 0:
                raise ValueError(f"If rank is an integer, it must be positive, but got {rank}.")
            if tensor.ndim == 1:
                param_for_tensor_train = 1
            else:
                param_for_tensor_train = [1] + [rank] * (tensor.ndim - 1) + [1]
        elif isinstance(rank, list):
            if tensor.ndim == 1:
                if rank:
                    raise ValueError(f"For a 1D tensor, rank list must be empty for user input, but got {rank}.")
                param_for_tensor_train = 1
            else:
                if len(rank) != tensor.ndim - 1:
                    raise ValueError(f"Rank list length must be tensor.ndim - 1 ({tensor.ndim - 1}), but got {len(rank)} for tensor of shape {tensor.shape}.")
                if not all(isinstance(r, int) and r > 0 for r in rank):
                    raise ValueError(f"All ranks in the list must be positive integers, but got {rank}.")
                param_for_tensor_train = [1] + rank + [1]
        else:
            raise TypeError(f"Rank must be an int or a list of ints, but got {type(rank)}.")
        logging.info(f"Performing TT decomposition with TensorLy rank parameter {param_for_tensor_train} (user input {rank}) on tensor of shape {tensor.shape}")
        try:
            tl_tensor = tl.tensor(tensor.float().numpy())
            result_tl = tensor_train(tl_tensor, rank=param_for_tensor_train)
            factors_np: List[tl.ndarray]
            if hasattr(result_tl, 'factors'):
                factors_np = result_tl.factors
            elif isinstance(result_tl, list):
                factors_np = result_tl
            elif tensor.ndim == 1 and isinstance(result_tl, tl.ndarray):
                factors_np = [result_tl]
            else:
                raise RuntimeError(f"TensorLy's tensor_train returned an unexpected type: {type(result_tl)}. Value: {result_tl}")
            torch_factors = [torch.from_numpy(factor.copy()).type(torch.float32) for factor in factors_np]
            logging.info(f"TT decomposition successful. Number of TT cores: {len(torch_factors)}")
            if torch_factors:
                for i, core_factor in enumerate(torch_factors):
                    logging.info(f"  TT Core {i} shape: {core_factor.shape}")
            return torch_factors
        except Exception as e:
            logging.error(f"Error during TT decomposition: {e}. Tensor shape: {tensor.shape}, Rank(s): {rank}")
            raise RuntimeError(f"TT decomposition failed. Original error: {e}")

    @staticmethod
    def tt_svd(tensor: torch.Tensor, rank: Union[int, List[int]]) -> List[torch.Tensor]:
        """Tensor Train decomposition using SVD initialization."""
        TensorOps._check_tensor(tensor)
        if tensor.ndim == 0:
            raise ValueError("TT decomposition requires a tensor with at least 1 dimension, but got 0.")

        param_for_tensor_train: Union[int, List[int]]
        if isinstance(rank, int):
            if rank <= 0:
                raise ValueError(f"If rank is an integer, it must be positive, but got {rank}.")
            param_for_tensor_train = 1 if tensor.ndim == 1 else [1] + [rank] * (tensor.ndim - 1) + [1]
        elif isinstance(rank, list):
            if tensor.ndim == 1:
                if rank:
                    raise ValueError(f"For a 1D tensor, rank list must be empty for user input, but got {rank}.")
                param_for_tensor_train = 1
            else:
                if len(rank) != tensor.ndim - 1:
                    raise ValueError(f"Rank list length must be tensor.ndim - 1 ({tensor.ndim - 1}), but got {len(rank)} for tensor of shape {tensor.shape}.")
                if not all(isinstance(r, int) and r > 0 for r in rank):
                    raise ValueError(f"All ranks in the list must be positive integers, but got {rank}.")
                param_for_tensor_train = [1] + rank + [1]
        else:
            raise TypeError(f"Rank must be an int or a list of ints, but got {type(rank)}.")

        logging.info(f"Performing TT-SVD with TensorLy rank parameter {param_for_tensor_train} (user input {rank}) on tensor of shape {tensor.shape}")
        try:
            tl_tensor = tl.tensor(tensor.float().numpy())
            result_tl = tensor_train(tl_tensor, rank=param_for_tensor_train, svd='truncated_svd')
            factors_np: List[tl.ndarray]
            if hasattr(result_tl, 'factors'):
                factors_np = result_tl.factors
            elif isinstance(result_tl, list):
                factors_np = result_tl
            elif tensor.ndim == 1 and isinstance(result_tl, tl.ndarray):
                factors_np = [result_tl]
            else:
                raise RuntimeError(f"TensorLy's tensor_train returned an unexpected type: {type(result_tl)}. Value: {result_tl}")
            torch_factors = [torch.from_numpy(f.copy()).type(torch.float32) for f in factors_np]
            return torch_factors
        except Exception as e:
            logging.error(f"Error during TT-SVD decomposition: {e}. Tensor shape: {tensor.shape}, Rank(s): {rank}")
            raise RuntimeError(f"TT-SVD decomposition failed. Original error: {e}")

    @staticmethod
    def tr_decomposition(tensor: torch.Tensor, rank: Union[int, List[int]]) -> List[torch.Tensor]:
        TensorOps._check_tensor(tensor)
        if tensor.ndim == 0:
            raise ValueError("TR decomposition requires a tensor with at least 1 dimension, but got 0.")
        param_for_tensor_ring: List[int]
        if isinstance(rank, int):
            if rank <= 0:
                raise ValueError(f"If rank is an integer, it must be positive, but got {rank}.")
            param_for_tensor_ring = [rank] * tensor.ndim + [rank]
        elif isinstance(rank, list):
            if len(rank) != tensor.ndim:
                raise ValueError(f"If rank is a list, its length must be equal to tensor.ndim ({tensor.ndim}), but got {len(rank)} for tensor of shape {tensor.shape}.")
            if not all(isinstance(r_val, int) and r_val > 0 for r_val in rank):
                raise ValueError(f"All ranks in the list must be positive integers, but got {rank}.")
            param_for_tensor_ring = rank + [rank[0]]
        else:
            raise TypeError(f"Rank must be an int or a list of ints, but got {type(rank)}.")
        logging.info(f"Performing TR decomposition with TensorLy rank parameter {param_for_tensor_ring} (user input {rank}) on tensor of shape {tensor.shape}")
        try:
            tl_tensor = tl.tensor(tensor.float().numpy())
            tr_object_tl = tensor_ring(tl_tensor, rank=param_for_tensor_ring)
            factors_np = tr_object_tl.factors
            torch_factors = [torch.from_numpy(factor.copy()).type(torch.float32) for factor in factors_np]
            logging.info(f"TR decomposition successful. Number of TR cores: {len(torch_factors)}")
            if torch_factors:
                for i, core_factor in enumerate(torch_factors):
                    logging.info(f"  TR Core {i} shape: {core_factor.shape}")
            return torch_factors
        except Exception as e:
            logging.error(f"Error during TR decomposition: {e}. Tensor shape: {tensor.shape}, Rank(s): {rank}")
            raise RuntimeError(f"TR decomposition failed. Original error: {e}")

    @staticmethod
    def ht_decomposition(tensor: torch.Tensor, dim_tree, ranks: Dict[int, int]):
        try:
            import htensor
        except ImportError:
            raise RuntimeError("htensor library is not installed, which is required for HT decomposition. Please install h-tucker.") from None
        TensorOps._check_tensor(tensor)
        if not isinstance(dim_tree, htensor.DimensionTree):
            raise TypeError(f"dim_tree must be an htensor.DimensionTree, but got {type(dim_tree)}.")
        if not isinstance(ranks, dict):
            raise TypeError(f"ranks must be a dict, but got {type(ranks)}.")
        if not all(isinstance(k, int) and isinstance(v, int) and v > 0 for k, v in ranks.items()):
            raise ValueError("ranks dictionary must have integer keys and positive integer values.")
        if tensor.ndim == 0:
             raise ValueError("HT decomposition requires a tensor with at least 1 dimension, but got 0.")
        if dim_tree.num_dims() != tensor.ndim:
            raise ValueError(f"Dimension tree number of dimensions ({dim_tree.num_dims()}) must match tensor dimensionality ({tensor.ndim}).")
        logging.info(f"Performing HT decomposition on tensor of shape {tensor.shape} with given dimension tree and ranks.")
        try:
            np_tensor = tensor.float().numpy()
            ht_object = htensor.HTensor.from_tensor(np_tensor, dim_tree, ranks)
            logging.info(f"HT decomposition successful. Resulting HTensor object created.")
            return ht_object
        except Exception as e:
            logging.error(f"Error during HT decomposition: {e}. Tensor shape: {tensor.shape}")
            raise RuntimeError(f"HT decomposition failed. Original error: {e}")

    @staticmethod
    def btd_decomposition(tensor: torch.Tensor, ranks_per_term: List[Tuple[int, int, int]]) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Block Term Decomposition (BTD) using a sequential Tucker-1 approach.

        Each term is computed via :func:`tensorly.decomposition.partial_tucker`
        on the current residual tensor. The list ``ranks_per_term`` defines the
        core sizes for each term as ``[(L_1, M_1, N_1), (L_2, M_2, N_2), ...]``.

        Args:
            tensor: Input 3-way tensor to decompose.
            ranks_per_term: List of tuples specifying core ranks for every term.
                ``ranks_per_term[i]`` corresponds to the shape ``(L_i, M_i, N_i)``
                of the ``i``-th core and, consequently, to the number of columns
                of the factor matrices returned for that term.

        Returns:
            A list ``[(core_1, [A_1, B_1, C_1]), (core_2, [A_2, B_2, C_2]), ...]``
            where each ``core_r`` is a tensor of shape ``(L_r, M_r, N_r)`` and the
            factor matrices have shapes ``(I, L_r)``, ``(J, M_r)``, ``(K, N_r)`` for
            an input tensor of shape ``(I, J, K)``.
        """

        TensorOps._check_tensor(tensor)
        if tensor.ndim != 3:
            raise ValueError(
                f"BTD as sum of Tucker-1 terms is typically for 3-way tensors, but got {tensor.ndim} dimensions."
            )
        if not isinstance(ranks_per_term, list):
            raise TypeError(f"ranks_per_term must be a list of tuples, but got {type(ranks_per_term)}.")
        if not ranks_per_term:
            raise ValueError("ranks_per_term list cannot be empty.")
        for i, term_ranks in enumerate(ranks_per_term):
            if not (
                isinstance(term_ranks, tuple)
                and len(term_ranks) == 3
                and all(isinstance(r, int) and r > 0 for r in term_ranks)
            ):
                raise ValueError(
                    f"Each element in ranks_per_term must be a tuple of 3 positive integers, but term {i} is {term_ranks}."
                )
            if not (
                term_ranks[0] <= tensor.shape[0]
                and term_ranks[1] <= tensor.shape[1]
                and term_ranks[2] <= tensor.shape[2]
            ):
                raise ValueError(
                    f"Ranks for term {i} {term_ranks} exceed tensor dimensions {tensor.shape}."
                )

        logging.info(
            f"Performing BTD with ranks_per_term {ranks_per_term} on tensor of shape {tensor.shape}"
        )

        residual = tensor.float().clone()
        terms: List[Tuple[torch.Tensor, List[torch.Tensor]]] = []

        for term_idx, ranks in enumerate(ranks_per_term):
            try:
                tl_tensor = tl.tensor(residual.numpy())
                (core_np, factors_np), _ = partial_tucker(tl_tensor, rank=list(ranks))
                torch_core = torch.from_numpy(core_np.copy()).type(torch.float32)
                torch_factors = [torch.from_numpy(f.copy()).type(torch.float32) for f in factors_np]
                terms.append((torch_core, torch_factors))

                # Subtract reconstructed term from residual for sequential fit
                reconstructed_np = tucker_to_tensor((core_np, factors_np))
                residual = residual - torch.from_numpy(reconstructed_np).type(residual.dtype)
                logging.info(
                    f"BTD term {term_idx} extracted with core shape {torch_core.shape}"
                )
            except Exception as e:
                logging.error(
                    f"Error extracting BTD term {term_idx}: {e}. Tensor shape: {tensor.shape}, ranks: {ranks}"
                )
                raise RuntimeError(f"BTD decomposition failed at term {term_idx}. Original error: {e}")

        return terms

    @staticmethod
    def t_svd(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs Tensor SVD (t-SVD) on a 3-way tensor.
        Factorizes a 3-way tensor X into U * S * V^T (using t-product).

        Args:
            tensor (torch.Tensor): The input 3-way tensor (n1 x n2 x n3).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - U (torch.Tensor): Orthogonal tensor U (n1 x n1 x n3).
                - S (torch.Tensor): F-diagonal tensor S (n1 x n2 x n3).
                - V (torch.Tensor): Orthogonal tensor V (n2 x n2 x n3)
                                    (returned as V, not V_H from SVD).
        Raises:
            TypeError: If the input `tensor` is not a PyTorch tensor.
            ValueError: If the input `tensor` is not 3-way.
            RuntimeError: If the t-SVD computation fails.
        """
        TensorOps._check_tensor(tensor)
        if tensor.ndim != 3:
            raise ValueError(f"t-SVD is defined for 3-way tensors, but got {tensor.ndim} dimensions.")

        logging.info(f"Performing t-SVD on tensor of shape {tensor.shape}")

        try:
            X_np = tensor.float().numpy()
            n1, n2, n3 = X_np.shape

            X_fft = fft(X_np, axis=2)

            U_fft_np = np.zeros((n1, n1, n3), dtype=complex)
            S_fft_np = np.zeros((n1, n2, n3), dtype=complex)
            # Vh_fft_np = np.zeros((n2, n2, n3), dtype=complex) # Store Vh (Hermitian of V)
            V_fft_np = np.zeros((n2, n2, n3), dtype=complex)


            for i in range(n3):
                U_slice, s_slice, Vh_slice = np.linalg.svd(X_fft[:, :, i], full_matrices=True)

                U_fft_np[:, :U_slice.shape[1], i] = U_slice # U_slice shape (n1, k) k=min(n1,n2) if not full_matrices, (n1,n1) if full_matrices

                S_slice_mat = np.zeros((n1, n2), dtype=complex)
                min_dim = min(n1, n2)
                S_slice_mat[:min_dim, :min_dim] = np.diag(s_slice)
                S_fft_np[:, :, i] = S_slice_mat

                # Vh_fft_np[:Vh_slice.shape[0], :, i] = Vh_slice # Vh_slice shape (k, n2) or (n2,n2)
                V_fft_np[:, :Vh_slice.shape[0], i] = Vh_slice.T.conj()


            U_ifft = ifft(U_fft_np, axis=2)
            S_ifft = ifft(S_fft_np, axis=2)
            V_ifft = ifft(V_fft_np, axis=2) # This is ifft of V

            # Take real part if original tensor was real (which it is via .float())
            U_final_np = U_ifft.real
            S_final_np = S_ifft.real
            V_final_np = V_ifft.real

            # Convert back to PyTorch tensors
            torch_U = torch.from_numpy(U_final_np.copy()).type(torch.float32)
            torch_S = torch.from_numpy(S_final_np.copy()).type(torch.float32)
            torch_V = torch.from_numpy(V_final_np.copy()).type(torch.float32)

            logging.info("t-SVD successful.")
            return torch_U, torch_S, torch_V

        except Exception as e:
            logging.error(f"Error during t-SVD: {e}. Tensor shape: {tensor.shape}")
            raise RuntimeError(f"t-SVD failed. Original error: {e}")


if __name__ == '__main__':
    print("--- TensorDecompositionOps Examples ---")

    # CP Decomposition Example
    print("\n--- CP Decomposition Example ---")
    cp_tensor_data = torch.arange(24, dtype=torch.float32).reshape((2, 3, 4))
    cp_rank = 2
    try:
        weights_cp, factors_cp = TensorDecompositionOps.cp_decomposition(cp_tensor_data, cp_rank)
        print(f"CP weights shape: {weights_cp.shape}")
        print(f"CP factors shapes: {[f.shape for f in factors_cp]}")
        np_weights_cp = weights_cp.numpy()
        np_factors_cp = [f.numpy() for f in factors_cp]
        reconstructed_cp = torch.from_numpy(tl.cp_to_tensor((np_weights_cp, np_factors_cp))).float()
        error_cp = torch.norm(cp_tensor_data - reconstructed_cp) / torch.norm(cp_tensor_data)
        print(f"CP reconstruction error: {error_cp.item():.6f}")
    except Exception as e:
        print(f"CP example failed: {e}")

    # NTF-CP Example
    print("\n--- NTF-CP Decomposition Example ---")
    ntf_cp_tensor_data = torch.rand(2, 3, 4).float()
    ntf_cp_rank = 2
    print(f"Original NTF-CP tensor shape: {ntf_cp_tensor_data.shape}, Rank: {ntf_cp_rank}")
    try:
        weights_ntf_cp, factors_ntf_cp = TensorDecompositionOps.ntf_cp_decomposition(ntf_cp_tensor_data, ntf_cp_rank)
        print(f"NTF-CP weights shape: {weights_ntf_cp.shape}, min value: {weights_ntf_cp.min().item():.4f}")
        print(f"NTF-CP factors shapes: {[f.shape for f in factors_ntf_cp]}")
        for i_f, factor_f in enumerate(factors_ntf_cp):
            print(f"  Factor {i_f} min value: {factor_f.min().item():.4f}")
        np_weights_ntf_cp = weights_ntf_cp.numpy()
        np_factors_ntf_cp = [f.numpy() for f in factors_ntf_cp]
        reconstructed_ntf_cp = torch.from_numpy(tl.cp_to_tensor((np_weights_ntf_cp, np_factors_ntf_cp))).float()
        error_ntf_cp = torch.norm(ntf_cp_tensor_data - reconstructed_ntf_cp) / torch.norm(ntf_cp_tensor_data)
        print(f"NTF-CP reconstruction error: {error_ntf_cp.item():.6f}")
    except Exception as e:
        print(f"NTF-CP example failed: {e}")

    # Tucker Decomposition Example
    print("\n--- Tucker Decomposition Example ---")
    tucker_tensor_data = torch.rand(3, 4, 5, dtype=torch.float32)
    tucker_ranks = [2, 3, 2]
    try:
        core_tucker, factors_tucker = TensorDecompositionOps.tucker_decomposition(tucker_tensor_data, tucker_ranks)
        print(f"Tucker core shape: {core_tucker.shape}")
        print(f"Tucker factors shapes: {[f.shape for f in factors_tucker]}")
        np_core_tucker = core_tucker.numpy()
        np_factors_tucker = [f.numpy() for f in factors_tucker]
        reconstructed_tucker = torch.from_numpy(tl.tucker_to_tensor((np_core_tucker, np_factors_tucker))).float()
        error_tucker = torch.norm(tucker_tensor_data - reconstructed_tucker) / torch.norm(tucker_tensor_data)
        print(f"Tucker reconstruction error: {error_tucker.item():.4f}")
    except Exception as e:
        print(f"Tucker example failed: {e}")

    # HOSVD Example
    print("\n--- HOSVD Example ---")
    hosvd_tensor_data = torch.rand(2, 3, 4, dtype=torch.float32)
    try:
        core_hosvd, factors_hosvd = TensorDecompositionOps.hosvd(hosvd_tensor_data)
        print(f"HOSVD core shape: {core_hosvd.shape}")
        print(f"HOSVD factors shapes: {[f.shape for f in factors_hosvd]}")
        for i, factor_h in enumerate(factors_hosvd):
            identity_approx = torch.matmul(factor_h.T, factor_h)
            print(f"  Factor {i} orthogonality check (vs Identity): {torch.allclose(identity_approx, torch.eye(factor_h.shape[1]), atol=1e-5)}")
        np_core_hosvd = core_hosvd.numpy()
        np_factors_hosvd = [f.numpy() for f in factors_hosvd]
        reconstructed_hosvd = torch.from_numpy(tl.tucker_to_tensor((np_core_hosvd, np_factors_hosvd))).float()
        error_hosvd = torch.norm(hosvd_tensor_data - reconstructed_hosvd) / torch.norm(hosvd_tensor_data)
        print(f"HOSVD reconstruction error: {error_hosvd.item():.6f}")
    except Exception as e:
        print(f"HOSVD example failed: {e}")

    # TT Decomposition Example (3D)
    print("\n--- TT Decomposition Example (3D) ---")
    tt_3d_tensor_data = torch.rand(3, 4, 5, dtype=torch.float32)
    tt_3d_ranks = [2, 3]
    try:
        factors_tt_3d = TensorDecompositionOps.tt_decomposition(tt_3d_tensor_data, tt_3d_ranks)
        print(f"TT 3D factors shapes: {[f.shape for f in factors_tt_3d]}")
        np_factors_tt_3d = [f.numpy() for f in factors_tt_3d]
        reconstructed_tt_3d = torch.from_numpy(tl.tt_to_tensor(np_factors_tt_3d)).float()
        error_tt_3d = torch.norm(tt_3d_tensor_data - reconstructed_tt_3d) / torch.norm(tt_3d_tensor_data)
        print(f"TT 3D reconstruction error: {error_tt_3d.item():.4f}")
    except Exception as e:
        print(f"TT 3D example failed: {e}")

    # TT Decomposition Example (1D - expected to fail based on previous findings)
    print("\n--- TT Decomposition Example (1D) ---")
    tt_1d_tensor_data = torch.rand(10, dtype=torch.float32)
    tt_1d_rank = []
    try:
        factors_tt_1d = TensorDecompositionOps.tt_decomposition(tt_1d_tensor_data, tt_1d_rank)
        print(f"TT 1D factors shapes: {[f.shape for f in factors_tt_1d]}")
    except RuntimeError as e:
        print(f"TT 1D example failed as expected: {e}")
    except Exception as e:
        print(f"An unexpected error in TT 1D example: {e}")

    # TR Decomposition Example
    print("\n--- TR Decomposition Example ---")
    tr_tensor_data = torch.rand(4, 5, 6, dtype=torch.float32)
    tr_ranks = [2, 2, 2]
    try:
        factors_tr = TensorDecompositionOps.tr_decomposition(tr_tensor_data, tr_ranks)
        print(f"TR factors shapes: {[f.shape for f in factors_tr]}")
        np_factors_tr = [f.numpy() for f in factors_tr]
        reconstructed_tr = torch.from_numpy(tl.tr_to_tensor(np_factors_tr)).float()
        error_tr = torch.norm(tr_tensor_data - reconstructed_tr) / torch.norm(tr_tensor_data)
        print(f"TR reconstruction error: {error_tr.item():.4f}")
    except Exception as e:
        print(f"TR example failed: {e}")

    # HT Decomposition Example
    print("\n--- HT Decomposition Example ---")
    try:
        import htensor
        ht_tensor_data = torch.rand(4, 4, 4, 4, dtype=torch.float32)
        ht_dim_tree = htensor.DimensionTree(ht_tensor_data.ndim)
        ht_ranks_full_dict = {i: 2 for i in range(1, ht_dim_tree.max_node_id + 1)}
        print(f"Original HT tensor shape: {ht_tensor_data.shape}, Tree: default balanced, Ranks: all 2")
        ht_object = TensorDecompositionOps.ht_decomposition(ht_tensor_data, ht_dim_tree, ht_ranks_full_dict)
        print(f"HT decomposition successful. HTensor object created: {ht_object}")
        reconstructed_ht_np = ht_object.to_tensor()
        reconstructed_ht_torch = torch.from_numpy(reconstructed_ht_np).type(ht_tensor_data.dtype)
        error_ht = torch.norm(ht_tensor_data - reconstructed_ht_torch) / torch.norm(ht_tensor_data)
        print(f"HT reconstruction error: {error_ht.item():.4f}")
    except RuntimeError as e:
         print(f"HT example failed or skipped: {e}")
    except ImportError:
         print(f"HT example skipped: htensor library not available for example execution.")
    except Exception as e:
        print(f"HT example failed with an unexpected error: {e}")

    # BTD Example
    print("\n--- BTD Example ---")
    btd_tensor_data = torch.rand(6, 7, 8, dtype=torch.float32)
    btd_ranks_per_term = [(2, 2, 2), (3, 3, 3)]
    print(f"Original BTD tensor shape: {btd_tensor_data.shape}, Ranks per term: {btd_ranks_per_term}")
    try:
        btd_terms = TensorDecompositionOps.btd_decomposition(btd_tensor_data, btd_ranks_per_term)
    except NotImplementedError as nie:
        print(f"BTD example skipped: {nie}")
    except ImportError as ie:
        print(f"BTD example skipped due to import error: {ie}")
    except Exception as e:
        print(f"BTD example failed: {e}")

    # t-SVD Example
    print("\n--- t-SVD Example ---")
    tsvd_tensor_data = torch.rand(5, 4, 3).float() # n1, n2, n3
    print(f"Original t-SVD tensor shape: {tsvd_tensor_data.shape}")
    try:
        U_tsvd, S_tsvd, V_tsvd = TensorDecompositionOps.t_svd(tsvd_tensor_data)
        print(f"t-SVD U shape: {U_tsvd.shape}") # (n1, n1, n3)
        print(f"t-SVD S shape: {S_tsvd.shape}") # (n1, n2, n3)
        print(f"t-SVD V shape: {V_tsvd.shape}") # (n2, n2, n3)

        # Reconstruction: X = U * S * V_H (V Hermitian/transpose)
        # V_H_tsvd = torch.permute(V_tsvd, (1, 0, 2)).conj() # Not needed if V is already V
        # For real data, V_H is V.T (slice-wise)
        # V_transpose_np = np.transpose(V_tsvd.numpy(), axes=(1,0,2)) # Transpose frontal slices
        # V_H_tsvd = torch.from_numpy(V_transpose_np).type(tsvd_tensor_data.dtype)

        # For real case, V is orthogonal, so V_H = V_transpose (for each frontal slice)
        # The t-product U * S * V_H. V stores V_H from SVD.
        # So we need V.conj().transpose(1,0,2) for V_H from V.
        # Or, if V_tsvd is already V_H (as from np.linalg.svd's vh), then just use it.
        # The implementation stores Vh.T.conj() as V. So V is U_v in X = U_u S U_v^H.
        # So to reconstruct, we need V_H. V is (n2,n2,n3). V_H is (n2,n2,n3) with frontal slices V_i^H.
        # The t_product(A,B) does A_i @ B_i. So for U*S*V_H, V_H needs to be (n2, n2, n3) with slices V_i^H.
        # V_tsvd is (n2, n2, n3) with slices V_i. So V_H_tsvd is V_tsvd.permute(1,0,2).conj() if complex.
        # If real, V_H_tsvd = V_tsvd.permute(1,0,2).

        # V_tsvd stores V from U,S,V_h = svd(X), where V_h is V_hermitian.
        # So, V_tsvd is the V in X = U*S*V_H. To reconstruct, we need V_H.
        # V_H_for_reconstruction = torch.permute(V_tsvd.conj(), (1,0,2)) # if V_tsvd stores V
        # The svd returns U, s, Vh. We store U, diag(s), Vh.T.conj() (this is V).
        # So to reconstruct X = U*S*V_H, we need V_H.
        # Our V_tsvd is V. So V_H is V_tsvd.permute(1,0,2).conj() if complex, or .T if real.
        # For real data, V_final_np is V. V_H is V.T (slice-wise).
        # So, V_H_torch = torch.from_numpy(np.transpose(V_tsvd.numpy(), axes=(1,0,2))).type(tsvd_tensor_data.dtype)

        # Simpler: if V_tsvd is (n2, n2, n3), its frontal slices are V_i.
        # V_H means each frontal slice V_i becomes V_i^H.
        # The V_tsvd from function is (n2, n2, n3) where each frontal slice is V_i (orthogonal)
        # So V_H_tsvd should be V_tsvd.permute(0,1,2) with slices V_i.conj().T - no, this is not right.
        # V_H means (V_i)^H for each frontal slice, so (n2, n2, n3) with V_i^H.
        # V_tsvd_H = V_tsvd.permute(1,0,2) # Transposing the first two dims of each frontal slice
        # If complex, V_tsvd_H = V_tsvd.permute(1,0,2).conj()
        # Since input is real, V_tsvd is real, so V_tsvd_H = V_tsvd.permute(1,0,2)

        # Reconstruction X_rec = U * S * V^T (t-product)
        # V_tsvd stores V. V^T means transpose of each frontal slice.
        V_transpose_tsvd = torch.transpose(V_tsvd, 0, 1) # Transposes (n2,n2,n3) to (n2,n2,n3) where frontal slices are V_i.T

        temp_prod = TensorDecompositionOps._t_product(U_tsvd, S_tsvd)
        reconstructed_tsvd = TensorDecompositionOps._t_product(temp_prod, V_transpose_tsvd)

        error_tsvd = torch.norm(tsvd_tensor_data - reconstructed_tsvd) / torch.norm(tsvd_tensor_data)
        print(f"t-SVD reconstruction error: {error_tsvd.item():.6f}")
    except Exception as e:
        print(f"t-SVD example failed: {e}")


    print("\n--- All Decomposition Examples Finished ---")
