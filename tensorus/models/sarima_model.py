import numpy as np
import pandas as pd
import joblib
from typing import Any, Optional, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import TensorusModel


class SARIMAModel(TensorusModel):
    """Seasonal ARIMA model using ``statsmodels``."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 0, 0),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        **kwargs,
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.kwargs = kwargs
        self.model: Optional[SARIMAX] = None
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
        self.model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            **self.kwargs,
        )
        self.fitted = self.model.fit(disp=False)

    def predict(self, steps: Any) -> np.ndarray:
        if self.fitted is None:
            raise ValueError("Model not trained. Call fit() first.")
        n_steps = int(steps)
        forecast = self.fitted.forecast(steps=n_steps)
        return np.asarray(forecast)

    def save(self, path: str) -> None:
        if self.fitted is None:
            raise ValueError("Model not trained. Call fit() before save().")
        joblib.dump(self.fitted, path)

    def load(self, path: str) -> None:
        self.fitted = joblib.load(path)
