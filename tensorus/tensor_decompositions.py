# tensorus/tensor_decompositions.py
"""
Provides a library of tensor decomposition operations for Tensorus.
This module will contain various tensor decomposition algorithms
like CP, Tucker, TT, TR, etc.
"""

import torch
import tensorly as tl
from typing import List, Tuple, Union, Optional

# To use utility methods like _check_tensor from TensorOps
# Assuming tensor_ops.py is in the same directory and can be imported.
from tensorus.tensor_ops import TensorOps
import logging
from tensorly.decomposition import parafac, tucker, tensor_train, tensor_ring # Added tensor_ring
from tensorly.cp_tensor import cp_to_tensor
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.tt_tensor import tt_to_tensor
from tensorly.tr_tensor import tr_to_tensor # Added for TR reconstruction


class TensorDecompositionOps:
    """
    A static library class providing tensor decomposition operations.
    All methods are static and operate on provided torch.Tensor objects,
    returning decomposition factors also as torch.Tensor objects.
    """

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
    def tr_decomposition(tensor: torch.Tensor, rank: Union[int, List[int]]) -> List[torch.Tensor]:
        """
        Performs Tensor Ring (TR) decomposition on a tensor.

        The TR decomposition factorizes a tensor into a sequence of 3D cores,
        where the last rank connects back to the first, forming a ring.

        Args:
            tensor (torch.Tensor): The input tensor to decompose. Must have ndim >= 1.
            rank (Union[int, List[int]]): The TR-ranks.
                - If int: The maximum TR-rank (r_k for all k).
                - If List[int]: A list of N ranks [r_0, r_1, ..., r_{N-1}], where N is
                  the order of the tensor. The condition r_N = r_0 is implicit.

        Returns:
            List[torch.Tensor]: A list of TR factor tensors (cores).
                Each factor G_k is a 3D tensor of shape (rank_{k-1}, tensor.shape[k], rank_k).
                Note that rank_N is implicitly equal to rank_0.

        Raises:
            TypeError: If the input `tensor` is not a PyTorch tensor or if `rank` is not an int or list.
            ValueError: If `tensor.ndim` is 0.
                        If `rank` is an int and not positive.
                        If `rank` is a list and its length is not `tensor.ndim`.
                        If any rank in the list is not positive.
            RuntimeError: If the TR decomposition fails.
        """
        TensorOps._check_tensor(tensor)

        if tensor.ndim == 0:
            raise ValueError("TR decomposition requires a tensor with at least 1 dimension, but got 0.")

        # Validate user-provided rank and determine the rank parameter for TensorLy's tensor_ring
        # Based on error messages, this environment's tensor_ring expects N+1 ranks.
        param_for_tensor_ring: List[int]
        if isinstance(rank, int): # User provided a single maximum rank
            if rank <= 0:
                raise ValueError(f"If rank is an integer, it must be positive, but got {rank}.")
            param_for_tensor_ring = [rank] * tensor.ndim + [rank] # N+1 elements, r_N = r_0 = rank
        elif isinstance(rank, list): # User provided a list of N ranks [r_0, ..., r_{N-1}]
            if len(rank) != tensor.ndim:
                raise ValueError(f"If rank is a list, its length must be equal to tensor.ndim ({tensor.ndim}), but got {len(rank)} for tensor of shape {tensor.shape}.")
            if not all(isinstance(r_val, int) and r_val > 0 for r_val in rank):
                raise ValueError(f"All ranks in the list must be positive integers, but got {rank}.")
            param_for_tensor_ring = rank + [rank[0]] # Append r_0 to make r_N = r_0, total N+1 elements
        else:
            raise TypeError(f"Rank must be an int or a list of ints, but got {type(rank)}.")

        logging.info(f"Performing TR decomposition with TensorLy rank parameter {param_for_tensor_ring} (user input {rank}) on tensor of shape {tensor.shape}")

        try:
            tl_tensor = tl.tensor(tensor.float().numpy())

            # tensor_ring returns a TensorRing object which has a .factors attribute
            tr_object_tl = tensor_ring(tl_tensor, rank=param_for_tensor_ring)
            factors_np = tr_object_tl.factors # This is a list of numpy arrays

            torch_factors = [torch.from_numpy(factor.copy()).type(torch.float32) for factor in factors_np]

            logging.info(f"TR decomposition successful. Number of TR cores: {len(torch_factors)}")
            if torch_factors:
                for i, core_factor in enumerate(torch_factors):
                    logging.info(f"  TR Core {i} shape: {core_factor.shape}")

            return torch_factors

        except Exception as e:
            logging.error(f"Error during TR decomposition: {e}. Tensor shape: {tensor.shape}, Rank(s): {rank}")
            raise RuntimeError(f"TR decomposition failed. Original error: {e}")


if __name__ == '__main__':
    print("--- TensorDecompositionOps Examples ---")

    # CP Decomposition Example
    print("\n--- CP Decomposition Example ---")
    cp_tensor_data = torch.arange(24, dtype=torch.float32).reshape((2, 3, 4))
    cp_rank = 2
    print(f"Original CP tensor shape: {cp_tensor_data.shape}, Rank: {cp_rank}")
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

    # Tucker Decomposition Example
    print("\n--- Tucker Decomposition Example ---")
    tucker_tensor_data = torch.rand(3, 4, 5, dtype=torch.float32)
    tucker_ranks = [2, 3, 2]
    print(f"Original Tucker tensor shape: {tucker_tensor_data.shape}, Ranks: {tucker_ranks}")
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
    print(f"Original HOSVD tensor shape: {hosvd_tensor_data.shape}")
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
    print(f"Original TT 3D tensor shape: {tt_3d_tensor_data.shape}, Internal Ranks: {tt_3d_ranks}")
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
    print(f"Original TT 1D tensor shape: {tt_1d_tensor_data.shape}, User Rank: {tt_1d_rank}")
    try:
        factors_tt_1d = TensorDecompositionOps.tt_decomposition(tt_1d_tensor_data, tt_1d_rank)
        print(f"TT 1D factors shapes: {[f.shape for f in factors_tt_1d]}")
    except RuntimeError as e:
        print(f"TT 1D example failed as expected: {e}")
    except Exception as e:
        print(f"An unexpected error in TT 1D example: {e}")

    # TR Decomposition Example
    print("\n--- TR Decomposition Example ---")
    # Using a slightly larger tensor and smaller ranks to avoid SVD issues.
    tr_tensor_data = torch.rand(4, 5, 6, dtype=torch.float32) # 3D tensor
    tr_ranks = [2, 2, 2] # Ranks [r0, r1, r2], r2 implicitly connects to r0 for reconstruction.
                         # TensorLy's tensor_ring takes these N ranks.
    print(f"Original TR tensor shape: {tr_tensor_data.shape}, Ranks: {tr_ranks}")
    try:
        factors_tr = TensorDecompositionOps.tr_decomposition(tr_tensor_data, tr_ranks)
        print(f"TR factors shapes: {[f.shape for f in factors_tr]}")
        # Expected shapes for 3D (I0,I1,I2)=(4,5,6) and ranks [r0,r1,r2]=[2,2,2]:
        # G0: (r2, I0, r0) = (2, 4, 2)
        # G1: (r0, I1, r1) = (2, 5, 2)
        # G2: (r1, I2, r2) = (2, 6, 2)

        np_factors_tr = [f.numpy() for f in factors_tr]
        reconstructed_tr = torch.from_numpy(tl.tr_to_tensor(np_factors_tr)).float()
        error_tr = torch.norm(tr_tensor_data - reconstructed_tr) / torch.norm(tr_tensor_data)
        print(f"TR reconstruction error: {error_tr.item():.4f}")
    except Exception as e:
        print(f"TR example failed: {e}")

    print("\n--- All Decomposition Examples Finished ---")
