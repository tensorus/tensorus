import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Any, List, Optional, Dict

from .base import TensorusModel


class _RBM(nn.Module):
    """Simple Bernoulli RBM using CD-1."""

    def __init__(self, n_visible: int, n_hidden: int, lr: float = 0.01) -> None:
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_h(self, v: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(v @ self.W + self.h_bias)
        return torch.bernoulli(prob), prob

    def sample_v(self, h: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(h @ self.W.t() + self.v_bias)
        return torch.bernoulli(prob), prob

    def contrastive_divergence(self, v0: torch.Tensor) -> None:
        h0, ph0 = self.sample_h(v0)
        vk, _ = self.sample_v(h0)
        hk, phk = self.sample_h(vk)
        batch_size = v0.size(0)
        self.W.data += self.lr * ((v0.t() @ ph0 - vk.t() @ phk) / batch_size)
        self.v_bias.data += self.lr * torch.mean(v0 - vk, dim=0)
        self.h_bias.data += self.lr * torch.mean(ph0 - phk, dim=0)

    def fit(self, X: torch.Tensor, epochs: int = 5, batch_size: int = 64) -> None:
        loader = DataLoader(X, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for batch in loader:
                batch = batch.view(-1, self.n_visible)
                self.contrastive_divergence(batch)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        _, prob = self.sample_h(X.view(-1, self.n_visible))
        return prob


class StackedRBMClassifierModel(TensorusModel):
    """Classifier built from stacked RBMs with supervised fine-tuning."""

    def __init__(
        self,
        layer_sizes: List[int],
        n_classes: int,
        rbm_lr: float = 0.01,
        rbm_epochs: int = 5,
        fine_tune_lr: float = 0.001,
        fine_tune_epochs: int = 10,
        batch_size: int = 64,
    ) -> None:
        self.layer_sizes = layer_sizes
        self.n_classes = n_classes
        self.rbm_lr = rbm_lr
        self.rbm_epochs = rbm_epochs
        self.fine_tune_lr = fine_tune_lr
        self.fine_tune_epochs = fine_tune_epochs
        self.batch_size = batch_size
        self._build_model()

    def _build_model(self) -> None:
        modules: List[nn.Module] = []
        in_dim = self.layer_sizes[0]
        self.rbms: List[_RBM] = []
        for h in self.layer_sizes[1:]:
            self.rbms.append(_RBM(in_dim, h, lr=self.rbm_lr))
            modules.append(nn.Linear(in_dim, h))
            modules.append(nn.Sigmoid())
            in_dim = h
        modules.append(nn.Linear(in_dim, self.n_classes))
        self.model = nn.Sequential(*modules).to("cpu")

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

    def _pretrain(self, X: torch.Tensor) -> torch.Tensor:
        data = X
        for rbm in self.rbms:
            rbm.fit(data, epochs=self.rbm_epochs, batch_size=self.batch_size)
            data = rbm.transform(data)
        # copy rbm weights to nn.Linear layers
        idx = 0
        for layer in self.model:
            if isinstance(layer, nn.Linear) and idx < len(self.rbms):
                rbm = self.rbms[idx]
                layer.weight.data = rbm.W.t().clone()
                layer.bias.data = rbm.h_bias.clone()
                idx += 1
        return data

    def fit(self, X: Any, y: Any) -> None:
        X_t = self._to_tensor(X)
        y_t = self._to_label_tensor(y)
        self._pretrain(X_t)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.fine_tune_lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.fine_tune_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
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
                    "layer_sizes": self.layer_sizes,
                    "n_classes": self.n_classes,
                    "rbm_lr": self.rbm_lr,
                    "rbm_epochs": self.rbm_epochs,
                    "fine_tune_lr": self.fine_tune_lr,
                    "fine_tune_epochs": self.fine_tune_epochs,
                    "batch_size": self.batch_size,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        data: Dict[str, Any] = torch.load(path, map_location="cpu")
        cfg = data.get("config", {})
        self.layer_sizes = cfg.get("layer_sizes", self.layer_sizes)
        self.n_classes = cfg.get("n_classes", self.n_classes)
        self.rbm_lr = cfg.get("rbm_lr", self.rbm_lr)
        self.rbm_epochs = cfg.get("rbm_epochs", self.rbm_epochs)
        self.fine_tune_lr = cfg.get("fine_tune_lr", self.fine_tune_lr)
        self.fine_tune_epochs = cfg.get("fine_tune_epochs", self.fine_tune_epochs)
        self.batch_size = cfg.get("batch_size", self.batch_size)
        self._build_model()
        self.model.load_state_dict(data["state_dict"])
