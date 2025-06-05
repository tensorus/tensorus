import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Any

from .generative_base import GenerativeModel


class AffineFlow(nn.Module):
    def __init__(self, data_dim: int) -> None:
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(data_dim))
        self.log_scale = nn.Parameter(torch.zeros(data_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z * torch.exp(self.log_scale) + self.shift

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.shift) * torch.exp(-self.log_scale)


class FlowBasedModel(GenerativeModel):
    """Simple affine normalizing flow."""

    def __init__(
        self,
        data_dim: int = 784,
        lr: float = 1e-3,
        epochs: int = 1,
        batch_size: int = 32,
    ) -> None:
        super().__init__(AffineFlow(data_dim))
        self.data_dim = data_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def _to_tensor(self, arr: Any) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.float()
        import numpy as np

        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).float()
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    def _fit(self, X: Any) -> None:
        X_t = self._to_tensor(X).view(-1, self.data_dim)
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            for (xb,) in loader:
                optim.zero_grad()
                z = self.model.inverse(xb)
                log_prob = -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.data_dim * torch.log(torch.tensor(2 * torch.pi))
                log_det = self.model.log_scale.sum()
                loss = -(log_prob + log_det).mean()
                loss.backward()
                optim.step()

    def _sample(self, num_samples: int) -> torch.Tensor:
        z = torch.randn(num_samples, self.data_dim)
        with torch.no_grad():
            x = self.model(z)
        return x
