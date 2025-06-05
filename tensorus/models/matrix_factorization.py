import torch
import numpy as np
from typing import Any, Optional

from .base import TensorusModel


class MatrixFactorizationModel(TensorusModel):
    """Matrix factorization using stochastic gradient descent."""

    def __init__(
        self, n_factors: int = 8, lr: float = 0.01, reg: float = 0.02, epochs: int = 20
    ) -> None:
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.P: Optional[torch.Tensor] = None
        self.Q: Optional[torch.Tensor] = None
        self.ratings: Optional[torch.Tensor] = None

    def _to_tensor(self, arr: Any) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.float()
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).float()
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    def fit(self, X: Any, y: Any = None) -> None:
        R = self._to_tensor(X)
        self.ratings = R
        n_users, n_items = R.shape
        self.P = torch.randn(n_users, self.n_factors)
        self.Q = torch.randn(n_items, self.n_factors)
        mask = R > 0
        for _ in range(self.epochs):
            for u in range(n_users):
                for i in range(n_items):
                    if not mask[u, i]:
                        continue
                    r_ui = R[u, i]
                    pred = (self.P[u] * self.Q[i]).sum()
                    err = r_ui - pred
                    self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                    self.Q[i] += self.lr * (err * self.P[u] - self.reg * self.Q[i])

    def predict(self, X: Any = None) -> torch.Tensor:
        assert self.P is not None and self.Q is not None
        return self.P @ self.Q.t()

    def save(self, path: str) -> None:
        torch.save(
            {
                "P": self.P,
                "Q": self.Q,
                "config": {
                    "n_factors": self.n_factors,
                    "lr": self.lr,
                    "reg": self.reg,
                    "epochs": self.epochs,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        cfg = data.get("config", {})
        self.n_factors = cfg.get("n_factors", self.n_factors)
        self.lr = cfg.get("lr", self.lr)
        self.reg = cfg.get("reg", self.reg)
        self.epochs = cfg.get("epochs", self.epochs)
        self.P = data.get("P")
        self.Q = data.get("Q")
