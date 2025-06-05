import numpy as np
from typing import Any, List, Optional

from .base import TensorusModel


class MixtureOfExpertsModel(TensorusModel):
    """Combine multiple experts using fixed weights."""

    def __init__(self, models: List[TensorusModel], weights: Optional[List[float]] = None) -> None:
        self.models = models
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights

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
        for m in self.models:
            m.fit(X, y)

    def predict(self, X: Any) -> Any:
        preds = []
        for m in self.models:
            p = m.predict(X)
            preds.append(self._to_array(p))
        avg = np.sum([w * p for w, p in zip(self.weights, preds)], axis=0)
        first_pred = self.models[0].predict(X)
        if isinstance(first_pred, np.ndarray):
            return avg
        else:
            import torch
            return torch.tensor(avg, dtype=first_pred.dtype)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({"models": self.models, "weights": self.weights}, path)

    def load(self, path: str) -> None:
        import joblib
        data = joblib.load(path)
        self.models = data["models"]
        self.weights = data["weights"]
