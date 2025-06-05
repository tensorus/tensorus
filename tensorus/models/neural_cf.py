import torch
import numpy as np
from typing import Any, List, Optional
from torch import nn

from .base import TensorusModel


class NeuralCollaborativeFilteringModel(TensorusModel):
    """Neural network based collaborative filtering model."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 8,
        hidden_layers: Optional[List[int]] = None,
        lr: float = 0.01,
        epochs: int = 20,
    ) -> None:
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.hidden_layers = hidden_layers or [16]
        self.lr = lr
        self.epochs = epochs
        self._build_model()

    def _build_model(self) -> None:
        self.user_emb = nn.Embedding(self.n_users, self.embed_dim)
        self.item_emb = nn.Embedding(self.n_items, self.embed_dim)
        layers: List[nn.Module] = []
        input_dim = self.embed_dim * 2
        for h in self.hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def _to_tensor(self, arr: Any, dtype: torch.dtype = torch.long) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.to(dtype)
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).to(dtype)
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    def fit(self, X: Any, y: Any) -> None:
        X_t = self._to_tensor(X, torch.long)
        y_t = self._to_tensor(y, torch.float32).view(-1, 1)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            users = X_t[:, 0]
            items = X_t[:, 1]
            u_emb = self.user_emb(users)
            i_emb = self.item_emb(items)
            out = self.mlp(torch.cat([u_emb, i_emb], dim=1))
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()

    def parameters(self):
        return (
            list(self.user_emb.parameters())
            + list(self.item_emb.parameters())
            + list(self.mlp.parameters())
        )

    def predict(self, X: Any) -> torch.Tensor:
        X_t = self._to_tensor(X, torch.long)
        with torch.no_grad():
            users = X_t[:, 0]
            items = X_t[:, 1]
            u_emb = self.user_emb(users)
            i_emb = self.item_emb(items)
            out = self.mlp(torch.cat([u_emb, i_emb], dim=1))
            return out.view(-1)

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": {
                    "user_emb": self.user_emb.state_dict(),
                    "item_emb": self.item_emb.state_dict(),
                    "mlp": self.mlp.state_dict(),
                },
                "config": {
                    "n_users": self.n_users,
                    "n_items": self.n_items,
                    "embed_dim": self.embed_dim,
                    "hidden_layers": self.hidden_layers,
                    "lr": self.lr,
                    "epochs": self.epochs,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        cfg = data.get("config", {})
        self.n_users = cfg.get("n_users", self.n_users)
        self.n_items = cfg.get("n_items", self.n_items)
        self.embed_dim = cfg.get("embed_dim", self.embed_dim)
        self.hidden_layers = cfg.get("hidden_layers", self.hidden_layers)
        self.lr = cfg.get("lr", self.lr)
        self.epochs = cfg.get("epochs", self.epochs)
        self._build_model()
        state = data.get("state_dict", {})
        self.user_emb.load_state_dict(state.get("user_emb", {}))
        self.item_emb.load_state_dict(state.get("item_emb", {}))
        self.mlp.load_state_dict(state.get("mlp", {}))
