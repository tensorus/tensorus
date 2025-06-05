import numpy as np
import pandas as pd
import joblib
from typing import Any, Optional
from lifelines import CoxPHFitter

from .base import TensorusModel


class CoxPHModel(TensorusModel):
    """Cox Proportional Hazards model using ``lifelines.CoxPHFitter``."""

    def __init__(self) -> None:
        self.model: Optional[CoxPHFitter] = None
        self.feature_names: Optional[list[str]] = None

    def _to_array(self, arr: Any) -> np.ndarray:
        if isinstance(arr, np.ndarray):
            return arr
        if isinstance(arr, pd.DataFrame) or isinstance(arr, pd.Series):
            return arr.to_numpy()
        try:
            import torch
            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().numpy()
        except Exception:
            pass
        raise TypeError("Input must be array-like")

    def _prepare_dataframe(self, X: Any, durations: Any, events: Any) -> pd.DataFrame:
        X_np = self._to_array(X)
        durations_np = self._to_array(durations).reshape(-1)
        events_np = self._to_array(events).reshape(-1)
        df = pd.DataFrame(X_np)
        df["duration"] = durations_np
        df["event"] = events_np
        return df

    def fit(self, X: Any, durations: Any, event_observed: Any) -> None:
        df = self._prepare_dataframe(X, durations, event_observed)
        self.feature_names = list(df.columns[:-2])
        self.model = CoxPHFitter()
        self.model.fit(df, duration_col="duration", event_col="event")

    def predict(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        X_np = self._to_array(X)
        df = pd.DataFrame(X_np, columns=self.feature_names)
        return self.model.predict_expectation(df).to_numpy()

    def predict_partial_hazard(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        X_np = self._to_array(X)
        df = pd.DataFrame(X_np, columns=self.feature_names)
        return self.model.predict_partial_hazard(df).to_numpy()

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() before save().")
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)

    def load(self, path: str) -> None:
        obj = joblib.load(path)
        self.model = obj.get("model")
        self.feature_names = obj.get("feature_names")
