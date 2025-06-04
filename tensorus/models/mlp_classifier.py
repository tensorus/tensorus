import torch
import numpy as np
from typing import Any, List, Optional, Dict
from torch import nn

from .base import TensorusModel


class MLPClassifierModel(TensorusModel):
    """Simple configurable multi-layer perceptron classifier."""

    def __init__(
        self,
        input_size: int,
        hidden_layers: Optional[List[int]] = None,
        output_size: int = 2,
        activation: str = "relu",
        lr: float = 1e-3,
        epochs: int = 100,
    ) -> None:
        self.input_size = int(input_size)
        self.hidden_layers = hidden_layers or [64]
        self.output_size = int(output_size)
        self.activation_name = activation
        self.lr = lr
        self.epochs = epochs
        self._build_model()

    def _get_activation(self) -> nn.Module:
        act = self.activation_name.lower()
        if act == "relu":
            return nn.ReLU()
        if act == "tanh":
            return nn.Tanh()
        if act == "sigmoid":
            return nn.Sigmoid()
        raise ValueError(f"Unsupported activation '{self.activation_name}'")

    def _build_model(self) -> None:
        layers: List[nn.Module] = []
        in_dim = self.input_size
        for h in self.hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(self._get_activation())
            in_dim = h
        layers.append(nn.Linear(in_dim, self.output_size))
        self.model = nn.Sequential(*layers).to("cpu")

    def _to_tensor(self, arr: Any) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.float()
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).float()
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    def _to_label_tensor(self, arr: Any) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.long()
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).long()
        raise TypeError("Labels must be a torch.Tensor or numpy.ndarray")

    def fit(self, X: Any, y: Any) -> None:
        X_t = self._to_tensor(X)
        y_t = self._to_label_tensor(y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_t)
            loss = criterion(outputs, y_t)
            loss.backward()
            optimizer.step()

    def predict(self, X: Any) -> torch.Tensor:
        X_t = self._to_tensor(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            return logits.argmax(dim=1)

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": {
                    "input_size": self.input_size,
                    "hidden_layers": self.hidden_layers,
                    "output_size": self.output_size,
                    "activation": self.activation_name,
                    "lr": self.lr,
                    "epochs": self.epochs,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        data: Dict[str, Any] = torch.load(path, map_location="cpu")
        cfg = data.get("config", {})
        self.input_size = cfg.get("input_size", self.input_size)
        self.hidden_layers = cfg.get("hidden_layers", self.hidden_layers)
        self.output_size = cfg.get("output_size", self.output_size)
        self.activation_name = cfg.get("activation", self.activation_name)
        self.lr = cfg.get("lr", self.lr)
        self.epochs = cfg.get("epochs", self.epochs)
        self._build_model()
        self.model.load_state_dict(data["state_dict"])
