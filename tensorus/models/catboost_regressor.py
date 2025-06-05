import numpy as np
from typing import Any, Optional
from catboost import CatBoostRegressor

from .base import TensorusModel


class CatBoostRegressorModel(TensorusModel):
    """Wrapper for ``catboost.CatBoostRegressor`` with optional GPU support."""

    def __init__(self, use_gpu: bool = False, **kwargs) -> None:
        self.use_gpu = use_gpu
        self.kwargs = kwargs
        self.model: Optional[CatBoostRegressor] = None

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
        params = dict(self.kwargs)
        if self.use_gpu:
            params.setdefault("task_type", "GPU")
        self.model = CatBoostRegressor(**params)
        self.model.fit(X_np, y_np, verbose=False)

    def predict(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        X_np = self._to_array(X)
        return self.model.predict(X_np)

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() before save().")
        self.model.save_model(path)

    def load(self, path: str) -> None:
        self.model = CatBoostRegressor()
        self.model.load_model(path)
