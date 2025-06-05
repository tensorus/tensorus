import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATConv
from typing import Any

from .base import TensorusModel


class GATClassifierModel(TensorusModel):
    """Graph Attention Network for node classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 8,
        output_dim: int = 2,
        heads: int = 2,
        lr: float = 1e-2,
        epochs: int = 100,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.heads = int(heads)
        self.lr = lr
        self.epochs = epochs

        self.conv1 = GATConv(self.input_dim, self.hidden_dim, heads=self.heads)
        self.conv2 = GATConv(self.hidden_dim * self.heads, self.output_dim, heads=1)

    def _to_tensor(self, arr: Any) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.float()
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).float()
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    def _to_edge_index(self, arr: Any) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.long()
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).long()
        raise TypeError("Edge index must be a torch.Tensor or numpy.ndarray")

    def _forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def fit(self, X: Any, edge_index: Any, y: Any) -> None:
        x = self._to_tensor(X)
        ei = self._to_edge_index(edge_index)
        if isinstance(y, torch.Tensor):
            y_t = y.long()
        else:
            y_t = torch.tensor(y, dtype=torch.long)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        self.conv1.train()
        self.conv2.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            out = self._forward(x, ei)
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()

    def predict(self, X: Any, edge_index: Any) -> torch.Tensor:
        x = self._to_tensor(X)
        ei = self._to_edge_index(edge_index)
        self.conv1.eval()
        self.conv2.eval()
        with torch.no_grad():
            out = self._forward(x, ei)
            return out.argmax(dim=1)

    def parameters(self):
        return list(self.conv1.parameters()) + list(self.conv2.parameters())

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": {
                    "conv1": self.conv1.state_dict(),
                    "conv2": self.conv2.state_dict(),
                },
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "output_dim": self.output_dim,
                    "heads": self.heads,
                    "lr": self.lr,
                    "epochs": self.epochs,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        cfg = data.get("config", {})
        self.__init__(
            cfg.get("input_dim", self.input_dim),
            cfg.get("hidden_dim", self.hidden_dim),
            cfg.get("output_dim", self.output_dim),
            cfg.get("heads", self.heads),
            cfg.get("lr", self.lr),
            cfg.get("epochs", self.epochs),
        )
        self.conv1.load_state_dict(data["state_dict"]["conv1"])
        self.conv2.load_state_dict(data["state_dict"]["conv2"])
