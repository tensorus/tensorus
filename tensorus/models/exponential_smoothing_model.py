import numpy as np
import pandas as pd
import joblib
from typing import Any, Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .base import TensorusModel


class ExponentialSmoothingModel(TensorusModel):
    """Exponential Smoothing model using ``statsmodels``."""

    def __init__(self, trend: Optional[str] = None, seasonal: Optional[str] = None,
                 seasonal_periods: Optional[int] = None, **kwargs) -> None:
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.kwargs = kwargs
        self.model: Optional[ExponentialSmoothing] = None
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
        self.model = ExponentialSmoothing(
            series,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            **self.kwargs,
        )
        self.fitted = self.model.fit()

    def predict(self, steps: Any) -> np.ndarray:
        if self.fitted is None:
            raise ValueError("Model not trained. Call fit() first.")
        n_steps = int(steps)
        forecast = self.fitted.forecast(n_steps)
        return np.asarray(forecast)

    def save(self, path: str) -> None:
        if self.fitted is None:
            raise ValueError("Model not trained. Call fit() before save().")
        joblib.dump(self.fitted, path)

    def load(self, path: str) -> None:
        self.fitted = joblib.load(path)
