from typing import Any, List, Tuple, Union

import torch


def tensor_to_list(tensor: torch.Tensor) -> Tuple[List[int], str, Union[List[Any], int, float]]:
    """Convert a tensor to (shape, dtype, list data)."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    shape = list(tensor.shape)
    dtype_str = str(tensor.dtype).split(".")[-1]
    data = tensor.tolist()
    return shape, dtype_str, data
