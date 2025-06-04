import torch
import numpy as np
from typing import Any, Optional

from .base import TensorusModel


class LinearRegressionModel(TensorusModel):
    """Simple linear regression model using the normal equation.

    Args:
        None

    Attributes:
        weight (torch.Tensor): Learned weights of shape ``(n_features,)``.
        bias (torch.Tensor): Learned bias scalar.
    """

    def __init__(self) -> None:
        self.weight: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None

    def _to_tensor(self, arr: Any) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.float()
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).float()
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    def fit(self, X: Any, y: Any) -> None:
        """Fit the model.

        Args:
            X: Training data of shape ``(n_samples, n_features)``.
            y: Target values of shape ``(n_samples,)`` or ``(n_samples, 1)``.
        """
        X_t = self._to_tensor(X)
        y_t = self._to_tensor(y).view(-1, 1)

        ones = torch.ones(X_t.shape[0], 1, dtype=X_t.dtype)
        X_aug = torch.cat([X_t, ones], dim=1)
        theta = torch.linalg.pinv(X_aug) @ y_t
        self.weight = theta[:-1, 0]
        self.bias = theta[-1, 0]

    def predict(self, X: Any) -> torch.Tensor:
        """Predict target values for ``X``.

        Args:
            X: Input tensor of shape ``(n_samples, n_features)``.
        Returns:
            torch.Tensor: Predictions of shape ``(n_samples,)``.
        """
        if self.weight is None or self.bias is None:
            raise ValueError("Model parameters not initialized. Call fit() first.")
        X_t = self._to_tensor(X)
        return X_t @ self.weight + self.bias

    def save(self, path: str) -> None:
        """Save model parameters to ``path``."""
        torch.save({"weight": self.weight, "bias": self.bias}, path)

    def load(self, path: str) -> None:
        """Load model parameters from ``path``."""
        data = torch.load(path, map_location="cpu")
        self.weight = data.get("weight")
        self.bias = data.get("bias")
