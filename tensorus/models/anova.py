import numpy as np
import pandas as pd
import joblib
from typing import Any, Optional
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from .base import TensorusModel


class AnovaModel(TensorusModel):
    """One-way or multi-way ANOVA using ``statsmodels``."""

    def __init__(self) -> None:
        self.model = None
        self.table: Optional[pd.DataFrame] = None
        self.feature_cols: Optional[list[str]] = None

    def _prepare_df(self, X: Any, y: Any) -> pd.DataFrame:
        X_np = np.asarray(X)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        y_np = np.asarray(y).reshape(-1)
        cols = [f"x{i}" for i in range(X_np.shape[1])]
        df = pd.DataFrame(X_np, columns=cols)
        df["y"] = y_np
        self.feature_cols = cols
        return df

    def fit(self, X: Any, y: Any) -> None:
        df = self._prepare_df(X, y)
        formula = "y ~ " + " + ".join(self.feature_cols)
        self.model = ols(formula, data=df).fit()
        self.table = anova_lm(self.model)

    def predict(self, X: Any) -> Any:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        X_np = np.asarray(X)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        df = pd.DataFrame(X_np, columns=self.feature_cols)
        return self.model.predict(df)

    def summary(self) -> pd.DataFrame:
        if self.table is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.table

    def save(self, path: str) -> None:
        if self.table is None:
            raise ValueError("Model not trained. Call fit() before save().")
        joblib.dump({"model": self.model, "table": self.table, "feature_cols": self.feature_cols}, path)

    def load(self, path: str) -> None:
        obj = joblib.load(path)
        self.model = obj.get("model")
        self.table = obj.get("table")
        self.feature_cols = obj.get("feature_cols")
