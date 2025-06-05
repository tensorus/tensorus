import numpy as np
import pandas as pd
import joblib
from typing import Any, Optional
from arch import arch_model

from .base import TensorusModel


class GARCHModel(TensorusModel):
    """GARCH model using the ``arch`` package."""

    def __init__(self, p: int = 1, q: int = 1, mean: str = "zero", vol: str = "GARCH",
                 dist: str = "normal", **kwargs) -> None:
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.dist = dist
        self.kwargs = kwargs
        self.model = None
        self.fitted = None

    def _to_series(self, arr: Any) -> pd.Series:
        if isinstance(arr, pd.Series):
            return arr
        if isinstance(arr, np.ndarray):
            return pd.Series(arr)
        try:
            import torch
            if isinstance(arr, torch.Tensor):
                return pd.Series(arr.detach().cpu().numpy())
        except Exception:
            pass
        raise TypeError("Input must be a pandas Series, numpy array or torch tensor")

    def fit(self, y: Any) -> None:
        series = self._to_series(y)
        self.model = arch_model(
            series,
            p=self.p,
            q=self.q,
            mean=self.mean,
            vol=self.vol,
            dist=self.dist,
            **self.kwargs,
        )
        self.fitted = self.model.fit(disp="off")

    def predict(self, steps: Any) -> np.ndarray:
        if self.fitted is None:
            raise ValueError("Model not trained. Call fit() first.")
        n_steps = int(steps)
        forecast = self.fitted.forecast(horizon=n_steps)
        return forecast.mean.iloc[-1].to_numpy()

    def save(self, path: str) -> None:
        if self.fitted is None:
            raise ValueError("Model not trained. Call fit() before save().")
        joblib.dump(self.fitted, path)

    def load(self, path: str) -> None:
        self.fitted = joblib.load(path)
