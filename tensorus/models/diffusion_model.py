import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Any

from .generative_base import GenerativeModel


class DiffusionDenoiser(nn.Module):
    def __init__(self, data_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffusionModel(GenerativeModel):
    """Very simple denoising diffusion model."""

    def __init__(
        self,
        data_dim: int = 784,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        epochs: int = 1,
        batch_size: int = 32,
        noise_std: float = 0.1,
        steps: int = 10,
    ) -> None:
        super().__init__(DiffusionDenoiser(data_dim, hidden_dim))
        self.data_dim = data_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.steps = steps

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
                noise = torch.randn_like(xb) * self.noise_std
                noisy = xb + noise
                pred = self.model(noisy)
                loss = nn.functional.mse_loss(pred, xb)
                loss.backward()
                optim.step()

    def _sample(self, num_samples: int) -> torch.Tensor:
        x = torch.randn(num_samples, self.data_dim)
        with torch.no_grad():
            for _ in range(self.steps):
                x = self.model(x)
        return torch.sigmoid(x)
