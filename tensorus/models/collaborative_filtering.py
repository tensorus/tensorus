import torch
import numpy as np
from typing import Any, Optional

from .base import TensorusModel


class CollaborativeFilteringModel(TensorusModel):
    """Simple memory-based collaborative filtering model."""

    def __init__(self, method: str = "user", k: Optional[int] = None) -> None:
        self.method = method
        self.k = k
        self.ratings: Optional[torch.Tensor] = None
        self.similarity: Optional[torch.Tensor] = None

    def _to_tensor(self, arr: Any) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.float()
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).float()
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    def _compute_similarity(self) -> None:
        assert self.ratings is not None
        if self.method == "user":
            matrix = self.ratings
        else:
            matrix = self.ratings.t()
        norm = matrix.norm(dim=1, keepdim=True) + 1e-8
        normalized = matrix / norm
        self.similarity = normalized @ normalized.t()

    def fit(self, X: Any, y: Any = None) -> None:
        self.ratings = self._to_tensor(X)
        self._compute_similarity()

    def _predict_for_user(self, user: int) -> torch.Tensor:
        assert self.ratings is not None and self.similarity is not None
        if self.method == "user":
            sim = self.similarity[user]
            ratings = self.ratings
        else:
            sim = self.similarity
            ratings = self.ratings
        # compute weighted sum
        if self.method == "user":
            weights = sim.clone()
            if self.k is not None:
                topk = torch.topk(weights, self.k + 1).indices  # include self
                mask = torch.zeros_like(weights, dtype=torch.bool)
                mask[topk] = True
                weights = torch.where(mask, weights, torch.zeros_like(weights))
            weights[user] = 0.0
            pred = weights @ ratings / (weights.abs().sum() + 1e-8)
        else:  # item-based
            user_r = ratings[user]
            weights = user_r.clone()
            if self.k is not None:
                topk = torch.topk(weights, self.k + 1).indices
                mask = torch.zeros_like(weights, dtype=torch.bool)
                mask[topk] = True
                weights = torch.where(mask, weights, torch.zeros_like(weights))
            pred = weights @ sim / (weights.abs().sum() + 1e-8)
        return pred

    def predict(self, X: Any = None) -> torch.Tensor:
        assert self.ratings is not None
        if X is not None:
            input_ratings = self._to_tensor(X)
        else:
            input_ratings = self.ratings
        n_users, n_items = input_ratings.shape
        preds = torch.zeros((n_users, n_items), dtype=input_ratings.dtype)
        for u in range(n_users):
            preds[u] = self._predict_for_user(u)
        return preds

    def save(self, path: str) -> None:
        torch.save({"ratings": self.ratings, "method": self.method, "k": self.k}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        self.method = data.get("method", "user")
        self.k = data.get("k")
        self.ratings = data.get("ratings")
        if self.ratings is not None:
            self._compute_similarity()
