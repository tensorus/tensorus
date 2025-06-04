import torch
import numpy as np
from typing import Any, Optional

from .base import TensorusModel


def _sigmoid(z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(z)


class LogisticRegressionModel(TensorusModel):
    """Binary logistic regression classifier.

    The model expects ``X`` with shape ``(n_samples, n_features)`` and ``y`` with
    shape ``(n_samples,)`` or ``(n_samples, 1)`` containing 0/1 labels.
    """

    def __init__(self, lr: float = 0.1, epochs: int = 1000) -> None:
        self.lr = lr
        self.epochs = epochs
        self.weight: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None

    def _to_tensor(self, arr: Any) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.float()
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).float()
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    def fit(self, X: Any, y: Any) -> None:
        """Train the classifier using gradient descent."""
        X_t = self._to_tensor(X)
        y_t = self._to_tensor(y).view(-1, 1)
        n_features = X_t.shape[1]
        self.weight = torch.zeros((n_features, 1), dtype=X_t.dtype)
        self.bias = torch.zeros(1, dtype=X_t.dtype)

        for _ in range(self.epochs):
            logits = X_t @ self.weight + self.bias
            probs = _sigmoid(logits)
            error = probs - y_t
            grad_w = X_t.t() @ error / X_t.shape[0]
            grad_b = error.mean()
            self.weight -= self.lr * grad_w
            self.bias -= self.lr * grad_b

    def predict(self, X: Any) -> torch.Tensor:
        """Return binary predictions for ``X``."""
        if self.weight is None or self.bias is None:
            raise ValueError("Model parameters not initialized. Call fit() first.")
        X_t = self._to_tensor(X)
        logits = X_t @ self.weight + self.bias
        return (_sigmoid(logits) >= 0.5).float().view(-1)

    def predict_proba(self, X: Any) -> torch.Tensor:
        """Return probabilities for ``X``."""
        if self.weight is None or self.bias is None:
            raise ValueError("Model parameters not initialized. Call fit() first.")
        X_t = self._to_tensor(X)
        logits = X_t @ self.weight + self.bias
        return _sigmoid(logits).view(-1)

    def save(self, path: str) -> None:
        """Save model parameters to ``path``."""
        torch.save({"weight": self.weight, "bias": self.bias}, path)

    def load(self, path: str) -> None:
        """Load model parameters from ``path``."""
        data = torch.load(path, map_location="cpu")
        self.weight = data.get("weight")
        self.bias = data.get("bias")
