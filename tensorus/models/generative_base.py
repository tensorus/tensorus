from abc import ABC, abstractmethod
from typing import Any
import torch
from torch import nn

from .base import TensorusModel


class GenerativeModel(TensorusModel, ABC):
    """Base class for generative models built with PyTorch."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def fit(self, *args: Any, **kwargs: Any) -> None:
        self._fit(*args, **kwargs)

    @abstractmethod
    def _fit(self, *args: Any, **kwargs: Any) -> None:
        """Internal training implementation for subclasses."""
        raise NotImplementedError

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        return self._sample(*args, **kwargs)

    # ``TensorusModel`` requires a ``predict`` method. For generative models this
    # simply proxies to :meth:`sample`.
    def predict(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self.sample(*args, **kwargs)

    @abstractmethod
    def _sample(self, *args: Any, **kwargs: Any) -> Any:
        """Return generated samples."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict()}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        self.model.load_state_dict(data["state_dict"])
