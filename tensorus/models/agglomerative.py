import numpy as np
from typing import Any, Optional
from sklearn.cluster import AgglomerativeClustering
import joblib

from .base import TensorusModel


class AgglomerativeClusteringModel(TensorusModel):
    """Agglomerative clustering using ``sklearn.cluster.AgglomerativeClustering``."""

    def __init__(self, n_clusters: int = 2, **kwargs) -> None:
        self.n_clusters = n_clusters
        self.kwargs = kwargs
        self.model: Optional[AgglomerativeClustering] = None
        self._last_fit_X: Optional[np.ndarray] = None

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
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, **self.kwargs)
        self.model.fit(X_np)
        self._last_fit_X = X_np

    def predict(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        X_np = self._to_array(X)
        if self._last_fit_X is not None and np.array_equal(X_np, self._last_fit_X):
            return self.model.labels_
        model = AgglomerativeClustering(n_clusters=self.n_clusters, **self.kwargs)
        return model.fit_predict(X_np)

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() before save().")
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
        self._last_fit_X = None

