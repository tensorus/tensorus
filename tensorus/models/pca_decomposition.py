import numpy as np
from typing import Any, Optional
from sklearn.decomposition import PCA
import joblib

from .base import TensorusModel


class PCADecompositionModel(TensorusModel):
    """Principal Component Analysis using ``sklearn.decomposition.PCA``."""

    def __init__(self, n_components: Optional[int] = None, **kwargs) -> None:
        self.n_components = n_components
        self.kwargs = kwargs
        self.model: Optional[PCA] = None

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

    def fit(self, X: Any, y: Any = None) -> None:
        X_np = self._to_array(X)
        self.model = PCA(n_components=self.n_components, **self.kwargs)
        self.model.fit(X_np)

    def predict(self, X: Any) -> np.ndarray:
        """Alias for :meth:`transform`."""
        return self.transform(X)

    def transform(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        X_np = self._to_array(X)
        return self.model.transform(X_np)

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        X_np = self._to_array(X)
        self.model = PCA(n_components=self.n_components, **self.kwargs)
        return self.model.fit_transform(X_np)

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() before save().")
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
