from typing import List, Any, Tuple, Union
import logging
import torch

logger = logging.getLogger(__name__)


def _validate_tensor_data(data: List[Any], shape: List[int]) -> bool:
    if not shape:
        if not isinstance(data, (int, float)):
            raise ValueError("Scalar tensor data must be a single number.")
        return True
    if not isinstance(data, list):
        raise ValueError(f"Data for shape {shape} must be a list.")
    if len(data) != shape[0]:
        raise ValueError(
            f"Dimension 0 mismatch for shape {shape}: Expected length {shape[0]}, got {len(data)}."
        )
    if len(shape) > 1:
        for item in data:
            _validate_tensor_data(item, shape[1:])
    else:
        if not all(isinstance(x, (int, float)) for x in data):
            bad = next((type(x).__name__ for x in data if not isinstance(x, (int, float))), "unknown")
            raise ValueError(f"Innermost list elements must be numbers, found '{bad}'.")
    return True


def list_to_tensor(shape: List[int], dtype_str: str, data: Union[List[Any], int, float]) -> torch.Tensor:
    try:
        dtype_map = {
            "float32": torch.float32,
            "float": torch.float,
            "float64": torch.float64,
            "double": torch.double,
            "int32": torch.int32,
            "int": torch.int,
            "int64": torch.int64,
            "long": torch.long,
            "bool": torch.bool,
        }
        torch_dtype = dtype_map.get(dtype_str.lower())
        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype string: '{dtype_str}'. Supported: {list(dtype_map.keys())}")
        tensor = torch.tensor(data, dtype=torch_dtype)
        if list(tensor.shape) != shape:
            logger.warning("Created tensor shape %s differs from requested %s. Attempting reshape.", list(tensor.shape), shape)
            tensor = tensor.reshape(shape)
        return tensor
    except (TypeError, ValueError) as e:
        logger.error("Error converting list to tensor: %s", e)
        raise ValueError(f"Failed tensor conversion: {e}") from e
    except Exception as e:  # pragma: no cover - unexpected
        logger.exception("Unexpected error during list_to_tensor: %s", e)
        raise ValueError(f"Unexpected tensor conversion error: {e}") from e


def tensor_to_list(tensor: torch.Tensor) -> Tuple[List[int], str, Union[List[Any], int, float]]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input must be a PyTorch Tensor, got {type(tensor).__name__}")
    shape = list(tensor.shape)
    dtype_str = str(tensor.dtype).replace("torch.", "")
    if tensor.ndim == 0:
        data = tensor.item()
    else:
        data = tensor.tolist()
    return shape, dtype_str, data
