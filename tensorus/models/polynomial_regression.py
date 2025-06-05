import numpy as np
from typing import Any, Optional
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
import joblib

from .base import TensorusModel


class PolynomialRegressionModel(TensorusModel):
    """Polynomial regression using sklearn Pipeline."""

    def __init__(self, degree: int = 2, regularization: float = 0.0) -> None:
        self.degree = degree
        self.regularization = regularization
        self.model: Optional[Any] = None

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
        if self.regularization > 0:
            lr = Ridge(alpha=self.regularization)
        else:
            lr = LinearRegression()
        self.model = make_pipeline(PolynomialFeatures(self.degree), lr)
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
