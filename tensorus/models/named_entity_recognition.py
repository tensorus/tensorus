import numpy as np
import torch
from torch import nn
from TorchCRF import CRF
from typing import Any, Optional

from .base import TensorusModel


class NamedEntityRecognitionModel(TensorusModel, nn.Module):
    """BiLSTM-CRF model for sequence tagging."""

    def __init__(
        self,
        vocab_size: int,
        tagset_size: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        lr: float = 0.1,
        epochs: int = 5,
    ) -> None:
        nn.Module.__init__(self)
        self.vocab_size = int(vocab_size)
        self.tagset_size = int(tagset_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.lr = lr
        self.epochs = epochs

        self.embed = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size)

    def _to_tensor(self, arr: Any, dtype: torch.dtype = torch.long) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.to(dtype)
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).to(dtype)
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    def _forward(self, X: torch.Tensor) -> torch.Tensor:
        x = self.embed(X)
        x, _ = self.lstm(x)
        emissions = self.fc(x)
        return emissions

    def fit(self, X: Any, y: Any) -> None:
        X_t = self._to_tensor(X)
        y_t = self._to_tensor(y)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        mask = torch.ones_like(X_t, dtype=torch.bool)
        self.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            emissions = self._forward(X_t)
            loss = -self.crf(emissions, y_t, mask).mean()
            loss.backward()
            optimizer.step()

    def predict(self, X: Any) -> torch.Tensor:
        X_t = self._to_tensor(X)
        self.eval()
        mask = torch.ones_like(X_t, dtype=torch.bool)
        with torch.no_grad():
            emissions = self._forward(X_t)
            preds = self.crf.viterbi_decode(emissions, mask)
        return torch.tensor(preds)

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "vocab_size": self.vocab_size,
                    "tagset_size": self.tagset_size,
                    "embedding_dim": self.embedding_dim,
                    "hidden_dim": self.hidden_dim,
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
            cfg.get("vocab_size", 1),
            cfg.get("tagset_size", 1),
            cfg.get("embedding_dim", self.embedding_dim),
            cfg.get("hidden_dim", self.hidden_dim),
            cfg.get("lr", self.lr),
            cfg.get("epochs", self.epochs),
        )
        self.load_state_dict(data["state_dict"])
