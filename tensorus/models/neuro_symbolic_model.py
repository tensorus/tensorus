import joblib
from typing import Any, Callable, Optional

from .base import TensorusModel


class NeuroSymbolicModel(TensorusModel):
    """Wrap a neural model with optional symbolic post-processing."""

    def __init__(self, base_model: TensorusModel, symbolic_fn: Optional[Callable[[Any, Any], Any]] = None) -> None:
        self.base_model = base_model
        self.symbolic_fn = symbolic_fn

    def fit(self, X: Any, y: Any) -> None:
        self.base_model.fit(X, y)

    def predict(self, X: Any) -> Any:
        preds = self.base_model.predict(X)
        if self.symbolic_fn is not None:
            preds = self.symbolic_fn(X, preds)
        return preds

    def save(self, path: str) -> None:
        joblib.dump({"base_model": self.base_model, "symbolic_fn": self.symbolic_fn}, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.base_model = data["base_model"]
        self.symbolic_fn = data.get("symbolic_fn")
