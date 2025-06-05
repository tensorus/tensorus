import torch
import numpy as np
from typing import Any
from torch import nn

from .base import TensorusModel


class CNNModelBase(TensorusModel):
    """Base class for convolutional neural network models."""

    def __init__(self, model: nn.Module, lr: float = 1e-3, epochs: int = 1) -> None:
        self.model = model
        self.lr = lr
        self.epochs = epochs

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
        torch.save({"state_dict": self.model.state_dict()}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        self.model.load_state_dict(data["state_dict"])
