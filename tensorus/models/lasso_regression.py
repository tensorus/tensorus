import numpy as np
from typing import Any, Optional
from sklearn.linear_model import Lasso
import joblib

from .base import TensorusModel


class LassoRegressionModel(TensorusModel):
    """Lasso regression model using ``sklearn.linear_model.Lasso``."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.model: Optional[Lasso] = None

    def _to_array(self, arr: Any) -> np.ndarray:
        if isinstance(arr, np.ndarray):
            return arr
        try:
            import torch

            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().numpy()
        except Exception:
            pass
        raise TypeError("Input must be a numpy array or torch tensor")

    def fit(self, X: Any, y: Any) -> None:
        X_np = self._to_array(X)
        y_np = self._to_array(y)
        self.model = Lasso(alpha=self.alpha)
        self.model.fit(X_np, y_np)

    def predict(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        X_np = self._to_array(X)
        return self.model.predict(X_np)

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() before save().")
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
