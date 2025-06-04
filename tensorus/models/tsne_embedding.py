import numpy as np
from typing import Any, Optional
from sklearn.manifold import TSNE
import joblib

from .base import TensorusModel


class TSNEEmbeddingModel(TensorusModel):
    """t-SNE dimensionality reduction using ``sklearn.manifold.TSNE``."""

    def __init__(self, n_components: int = 2, **kwargs) -> None:
        self.n_components = n_components
        self.kwargs = kwargs
        self.model: Optional[TSNE] = None

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
        self.model = TSNE(n_components=self.n_components, **self.kwargs)
        self.model.fit(X_np)

    def predict(self, X: Any) -> np.ndarray:
        """Alias for :meth:`transform`."""
        return self.transform(X)

    def transform(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        if hasattr(self.model, "transform"):
            X_np = self._to_array(X)
            return self.model.transform(X_np)
        raise NotImplementedError("TSNE transform is not available in this scikit-learn version")

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        X_np = self._to_array(X)
        self.model = TSNE(n_components=self.n_components, **self.kwargs)
        return self.model.fit_transform(X_np)

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() before save().")
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
