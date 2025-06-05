import numpy as np
import pandas as pd
import joblib
from typing import Any, Optional
from statsmodels.multivariate.manova import MANOVA

from .base import TensorusModel


class ManovaModel(TensorusModel):
    """Multivariate ANOVA using ``statsmodels``."""

    def __init__(self) -> None:
        self.manova: Optional[MANOVA] = None
        self.feature_cols: Optional[list[str]] = None
        self.target_cols: Optional[list[str]] = None

    def _prepare_df(self, X: Any, Y: Any) -> pd.DataFrame:
        X_np = np.asarray(X)
        Y_np = np.asarray(Y)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        if Y_np.ndim == 1:
            Y_np = Y_np.reshape(-1, 1)
        x_cols = [f"x{i}" for i in range(X_np.shape[1])]
        y_cols = [f"y{i}" for i in range(Y_np.shape[1])]
        df = pd.DataFrame(np.hstack([Y_np, X_np]), columns=y_cols + x_cols)
        self.feature_cols = x_cols
        self.target_cols = y_cols
        return df

    def fit(self, X: Any, Y: Any) -> None:
        df = self._prepare_df(X, Y)
        formula = " + ".join(self.target_cols) + " ~ " + " + ".join(self.feature_cols)
        self.manova = MANOVA.from_formula(formula, data=df)

    def predict(self, X: Any) -> Any:
        raise NotImplementedError("MANOVA model does not implement prediction.")

    def summary(self) -> pd.DataFrame:
        if self.manova is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.manova.mv_test().summary_frame

    def save(self, path: str) -> None:
        if self.manova is None:
            raise ValueError("Model not trained. Call fit() before save().")
        joblib.dump({"manova": self.manova,
                     "feature_cols": self.feature_cols,
                     "target_cols": self.target_cols}, path)

    def load(self, path: str) -> None:
        obj = joblib.load(path)
        self.manova = obj.get("manova")
        self.feature_cols = obj.get("feature_cols")
        self.target_cols = obj.get("target_cols")
