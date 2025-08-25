from abc import ABC, abstractmethod
from typing import Any, Optional

from .tensor_storage import TensorStorage


class TensorusModel(ABC):
    """Abstract base class for models used within Tensorus."""

    @abstractmethod
    def fit(self, X: Any, y: Optional[Any] = None) -> None:
        """Train the model."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Generate predictions for ``X``."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the model to ``path``."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model parameters from ``path``."""
        raise NotImplementedError

    def from_storage(self, storage: TensorStorage, dataset_name: str) -> Any:
        """Optional hook to load training data from ``TensorStorage``."""
        return storage.get_dataset(dataset_name)