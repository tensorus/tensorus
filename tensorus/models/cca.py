import numpy as np
from typing import Any, Optional, Tuple
from sklearn.cross_decomposition import CCA
import joblib

from .base import TensorusModel


class CCAModel(TensorusModel):
    """Canonical Correlation Analysis using ``sklearn.cross_decomposition.CCA``."""

    def __init__(self, n_components: int = 2, **kwargs) -> None:
        self.n_components = n_components
        self.kwargs = kwargs
        self.model: Optional[CCA] = None

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

    def fit(self, X: Any, Y: Any) -> None:
        X_np = self._to_array(X)
        Y_np = self._to_array(Y)
        self.model = CCA(n_components=self.n_components, **self.kwargs)
        self.model.fit(X_np, Y_np)

    def predict(self, X: Any, Y: Any | None = None) -> Tuple[np.ndarray, np.ndarray]:
        return self.transform(X, Y)

    def transform(self, X: Any, Y: Any | None = None) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        X_np = self._to_array(X)
        if Y is not None:
            Y_np = self._to_array(Y)
            return self.model.transform(X_np, Y_np)
        return self.model.transform(X_np)

    def fit_transform(self, X: Any, Y: Any) -> Tuple[np.ndarray, np.ndarray]:
        X_np = self._to_array(X)
        Y_np = self._to_array(Y)
        self.model = CCA(n_components=self.n_components, **self.kwargs)
        return self.model.fit_transform(X_np, Y_np)

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() before save().")
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
