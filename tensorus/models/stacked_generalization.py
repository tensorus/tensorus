import numpy as np
from typing import Any, List

from .base import TensorusModel


class StackedGeneralizationModel(TensorusModel):
    """Simple stacking ensemble using predictions of base models."""

    def __init__(self, base_models: List[TensorusModel], meta_model: TensorusModel) -> None:
        self.base_models = base_models
        self.meta_model = meta_model

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
        for model in self.base_models:
            model.fit(X, y)
        base_preds = [self._to_array(m.predict(X)).reshape(len(self._to_array(X)), -1) for m in self.base_models]
        meta_X = np.hstack(base_preds)
        self.meta_model.fit(meta_X, self._to_array(y))

    def predict(self, X: Any) -> Any:
        base_preds = [self._to_array(m.predict(X)).reshape(len(self._to_array(X)), -1) for m in self.base_models]
        meta_X = np.hstack(base_preds)
        return self.meta_model.predict(meta_X)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({"base_models": self.base_models, "meta_model": self.meta_model}, path)

    def load(self, path: str) -> None:
        import joblib
        data = joblib.load(path)
        self.base_models = data["base_models"]
        self.meta_model = data["meta_model"]
