import numpy as np
import torch
import logging
from typing import Tuple, List, Dict, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import tensorly as tl
    from tensorly.decomposition import parafac, tucker
    TENSORLY_AVAILABLE = True
except ImportError:
    logger.warning("TensorLy not available. Advanced tensor decompositions will not be supported.")
    TENSORLY_AVAILABLE = False

class TensorProcessor:
    """
    Processing layer for tensor operations.
    Provides functionality for basic and advanced tensor operations.
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize the tensor processor.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"TensorProcessor initialized on device: {self.device}")
        
        if TENSORLY_AVAILABLE:
            tl.set_backend('pytorch')
            logger.info("TensorLy backend set to PyTorch")
    
    def to_torch(self, tensor: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input to PyTorch tensor."""
        if isinstance(tensor, np.ndarray):
            return torch.from_numpy(tensor).to(self.device)
        elif isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        else:
            raise TypeError(f"Unsupported tensor type: {type(tensor)}")
    
    def to_numpy(self, tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert input to NumPy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            raise TypeError(f"Unsupported tensor type: {type(tensor)}")
    
    # Basic operations
    def add(self, t1: Union[np.ndarray, torch.Tensor], 
            t2: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Add two tensors."""
        torch_t1 = self.to_torch(t1)
        torch_t2 = self.to_torch(t2)
        result = torch.add(torch_t1, torch_t2)
        return self.to_numpy(result)
    
    def subtract(self, t1: Union[np.ndarray, torch.Tensor], 
                t2: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Subtract second tensor from first."""
        torch_t1 = self.to_torch(t1)
        torch_t2 = self.to_torch(t2)
        result = torch.sub(torch_t1, torch_t2)
        return self.to_numpy(result)
    
    def multiply(self, t1: Union[np.ndarray, torch.Tensor], 
                t2: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Element-wise multiplication of two tensors."""
        torch_t1 = self.to_torch(t1)
        torch_t2 = self.to_torch(t2)
        result = torch.mul(torch_t1, torch_t2)
        return self.to_numpy(result)
    
    def matmul(self, t1: Union[np.ndarray, torch.Tensor], 
              t2: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Matrix multiplication of two tensors."""
        torch_t1 = self.to_torch(t1)
        torch_t2 = self.to_torch(t2)
        result = torch.matmul(torch_t1, torch_t2)
        return self.to_numpy(result)
    
    # Advanced operations
    def decompose_cp(self, tensor: Union[np.ndarray, torch.Tensor], 
                    rank: int = 10) -> Dict[str, np.ndarray]:
        """
        Perform CP (CANDECOMP/PARAFAC) decomposition.
        
        Args:
            tensor: Input tensor
            rank: Rank of the decomposition
            
        Returns:
            Dictionary containing the factors of decomposition
        """
        if not TENSORLY_AVAILABLE:
            raise ImportError("TensorLy is required for tensor decompositions")
            
        torch_tensor = self.to_torch(tensor)
        factors = parafac(torch_tensor, rank=rank)
        
        # Convert factors to numpy arrays
        result = {
            "weights": self.to_numpy(factors[0]),
            "factors": [self.to_numpy(f) for f in factors[1]]
        }
        
        return result
    
    def decompose_tucker(self, tensor: Union[np.ndarray, torch.Tensor], 
                        ranks: List[int] = None) -> Dict[str, np.ndarray]:
        """
        Perform Tucker decomposition.
        
        Args:
            tensor: Input tensor
            ranks: List of ranks for each mode
            
        Returns:
            Dictionary containing the core tensor and factors
        """
        if not TENSORLY_AVAILABLE:
            raise ImportError("TensorLy is required for tensor decompositions")
            
        torch_tensor = self.to_torch(tensor)
        
        # Default ranks if not provided
        if ranks is None:
            ranks = [tensor.shape[i] // 2 for i in range(len(tensor.shape))]
        
        core, factors = tucker(torch_tensor, rank=ranks)
        
        # Convert to numpy arrays
        result = {
            "core": self.to_numpy(core),
            "factors": [self.to_numpy(f) for f in factors]
        }
        
        return result
    
    def tensor_svd(self, tensor: Union[np.ndarray, torch.Tensor], 
                  mode: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute SVD along a specific mode.
        
        Args:
            tensor: Input tensor
            mode: Mode along which to compute SVD
            
        Returns:
            U, S, V: Factors of the SVD
        """
        torch_tensor = self.to_torch(tensor)
        
        # Reshape tensor for SVD
        n_dim = len(torch_tensor.shape)
        if mode < 0 or mode >= n_dim:
            raise ValueError(f"Mode {mode} out of range for tensor with {n_dim} dimensions")
            
        # Reshape to matrix
        perm = list(range(n_dim))
        perm.pop(mode)
        perm = [mode] + perm
        reshaped = torch_tensor.permute(perm)
        shape = reshaped.shape
        matrix = reshaped.reshape(shape[0], -1)
        
        # Compute SVD
        U, S, V = torch.svd(matrix)
        
        return self.to_numpy(U), self.to_numpy(S), self.to_numpy(V)
        
    def reshape(self, tensor: Union[np.ndarray, torch.Tensor], 
               new_shape: Tuple) -> np.ndarray:
        """
        Reshape a tensor to a new shape.
        
        Args:
            tensor: Input tensor
            new_shape: New shape for the tensor
            
        Returns:
            Reshaped tensor
        """
        torch_tensor = self.to_torch(tensor)
        reshaped = torch_tensor.reshape(new_shape)
        return self.to_numpy(reshaped)
        
    def transpose(self, tensor: Union[np.ndarray, torch.Tensor], 
                 dims: List[int] = None) -> np.ndarray:
        """
        Transpose (permute) the dimensions of a tensor.
        
        Args:
            tensor: Input tensor
            dims: List specifying the permutation of dimensions
            
        Returns:
            Transposed tensor
        """
        torch_tensor = self.to_torch(tensor)
        
        if dims is None:
            # Default to reversing the dimensions
            dims = list(range(len(torch_tensor.shape)))[::-1]
            
        transposed = torch_tensor.permute(dims)
        return self.to_numpy(transposed) 