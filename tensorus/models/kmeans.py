import numpy as np
from typing import Any, Optional
from sklearn.cluster import KMeans
import joblib

from .base import TensorusModel


class KMeansClusteringModel(TensorusModel):
    """K-Means clustering using ``sklearn.cluster.KMeans``."""

    def __init__(self, n_clusters: int = 2, **kwargs) -> None:
        self.n_clusters = n_clusters
        self.kwargs = kwargs
        self.model: Optional[KMeans] = None

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
        self.model = KMeans(n_clusters=self.n_clusters, **self.kwargs)
        self.model.fit(X_np)

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
