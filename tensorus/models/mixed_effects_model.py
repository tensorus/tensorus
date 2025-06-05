import numpy as np
import pandas as pd
import joblib
from typing import Any, Optional
from statsmodels.regression.mixed_linear_model import MixedLM

from .base import TensorusModel


class MixedEffectsModel(TensorusModel):
    """Multilevel linear model using ``statsmodels.MixedLM``."""

    def __init__(self) -> None:
        self.model: Optional[MixedLM] = None
        self.result = None
        self.exog_names: Optional[list[str]] = None

    def _to_array(self, arr: Any) -> np.ndarray:
        if isinstance(arr, np.ndarray):
            return arr
        if isinstance(arr, pd.Series) or isinstance(arr, pd.DataFrame):
            return np.asarray(arr)
        try:
            import torch
            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().numpy()
        except Exception:
            pass
        raise TypeError("Input must be array-like")

    def fit(self, X: Any, y: Any, groups: Any) -> None:
        X_np = self._to_array(X)
        y_np = self._to_array(y).reshape(-1)
        groups_np = self._to_array(groups).reshape(-1)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        df = pd.DataFrame(X_np, columns=[f"x{i}" for i in range(X_np.shape[1])])
        df["y"] = y_np
        df["groups"] = groups_np
        exog = df[[f"x{i}" for i in range(X_np.shape[1])]]
        endog = df["y"]
        self.exog_names = list(exog.columns)
        self.model = MixedLM(endog, exog, groups=df["groups"])
        self.result = self.model.fit()

    def predict(self, X: Any) -> np.ndarray:
        if self.result is None:
            raise ValueError("Model not trained. Call fit() first.")
        X_np = self._to_array(X)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        df = pd.DataFrame(X_np, columns=self.exog_names)
        return self.result.predict(df)

    def save(self, path: str) -> None:
        if self.result is None:
            raise ValueError("Model not trained. Call fit() before save().")
        joblib.dump({"result": self.result, "exog_names": self.exog_names}, path)

    def load(self, path: str) -> None:
        obj = joblib.load(path)
        self.result = obj.get("result")
        self.exog_names = obj.get("exog_names")
