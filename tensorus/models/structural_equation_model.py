import pandas as pd
import numpy as np
import joblib
from typing import Any, Optional
from semopy import Model

from .base import TensorusModel


class StructuralEquationModel(TensorusModel):
    """Wrapper around ``semopy.Model`` for structural equation modeling."""

    def __init__(self, description: str) -> None:
        self.description = description
        self.model: Optional[Model] = None

    def _to_dataframe(self, data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, dict):
            return pd.DataFrame(data)
        if isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        try:
            import torch
            if isinstance(data, torch.Tensor):
                return pd.DataFrame(data.detach().cpu().numpy())
        except Exception:
            pass
        raise TypeError("Input must be convertible to pandas DataFrame")

    def fit(self, data: Any) -> None:
        df = self._to_dataframe(data)
        self.model = Model(self.description)
        self.model.fit(df)

    def predict(self, data: Any) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        df = self._to_dataframe(data)
        return self.model.predict(df)

    def save(self, path: str) -> None:
        """Persist the model description to ``path``."""
        joblib.dump({"description": self.description}, path)

    def load(self, path: str) -> None:
        obj = joblib.load(path)
        self.description = obj.get("description")
        self.model = None
