import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Any, Callable, Optional, List

from .base import TensorusModel


class PhysicsInformedNNModel(TensorusModel):
    """Simple feed-forward network with physics-informed loss."""

    def __init__(
        self,
        input_size: int,
        hidden_layers: Optional[List[int]] = None,
        lr: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        physics_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        physics_weight: float = 1.0,
    ) -> None:
        self.input_size = input_size
        self.hidden_layers = hidden_layers or [32]
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.physics_loss_fn = physics_loss_fn
        self.physics_weight = physics_weight
        self._build_model()

    def _build_model(self) -> None:
        modules: List[nn.Module] = []
        in_dim = self.input_size
        for h in self.hidden_layers:
            modules.append(nn.Linear(in_dim, h))
            modules.append(nn.Tanh())
            in_dim = h
        modules.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*modules).to("cpu")

    def _to_tensor(self, arr: Any) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.float()
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).float()
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    def fit(self, X: Any, y: Any) -> None:
        X_t = self._to_tensor(X)
        y_t = self._to_tensor(y).view(-1)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mse = nn.MSELoss()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.requires_grad_(self.physics_loss_fn is not None)
                opt.zero_grad()
                pred = self.model(xb).view(-1)
                loss = mse(pred, yb)
                if self.physics_loss_fn is not None:
                    phys = self.physics_loss_fn(xb, pred)
                    loss = loss + self.physics_weight * phys
                loss.backward()
                opt.step()

    def predict(self, X: Any) -> torch.Tensor:
        X_t = self._to_tensor(X)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_t).view(-1)
        return pred

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(),
                    "config": {
                        "input_size": self.input_size,
                        "hidden_layers": self.hidden_layers,
                        "lr": self.lr,
                        "epochs": self.epochs,
                        "batch_size": self.batch_size,
                        "physics_weight": self.physics_weight,
                    }}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        cfg = data.get("config", {})
        self.input_size = cfg.get("input_size", self.input_size)
        self.hidden_layers = cfg.get("hidden_layers", self.hidden_layers)
        self.lr = cfg.get("lr", self.lr)
        self.epochs = cfg.get("epochs", self.epochs)
        self.batch_size = cfg.get("batch_size", self.batch_size)
        self.physics_weight = cfg.get("physics_weight", self.physics_weight)
        self._build_model()
        self.model.load_state_dict(data["state_dict"])
