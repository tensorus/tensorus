import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Any

from .generative_base import GenerativeModel


class _GAN(nn.Module):
    def __init__(self, noise_dim: int, data_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)


class GANModel(GenerativeModel):
    """Very small GAN for toy data."""

    def __init__(
        self,
        data_dim: int = 784,
        noise_dim: int = 16,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        epochs: int = 1,
        batch_size: int = 32,
    ) -> None:
        super().__init__(_GAN(noise_dim, data_dim, hidden_dim))
        self.data_dim = data_dim
        self.noise_dim = noise_dim
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
        opt_g = torch.optim.Adam(self.model.generator.parameters(), lr=self.lr)
        opt_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.lr)
        bce = nn.BCELoss()
        for _ in range(self.epochs):
            for (xb,) in loader:
                # train discriminator
                real_labels = torch.ones(xb.size(0), 1)
                fake_labels = torch.zeros(xb.size(0), 1)

                opt_d.zero_grad()
                pred_real = self.model.discriminator(xb)
                loss_real = bce(pred_real, real_labels)
                z = torch.randn(xb.size(0), self.noise_dim)
                fake = self.model.generator(z).detach()
                pred_fake = self.model.discriminator(fake)
                loss_fake = bce(pred_fake, fake_labels)
                loss_d = loss_real + loss_fake
                loss_d.backward()
                opt_d.step()

                # train generator
                opt_g.zero_grad()
                z = torch.randn(xb.size(0), self.noise_dim)
                gen = self.model.generator(z)
                pred = self.model.discriminator(gen)
                loss_g = bce(pred, real_labels)
                loss_g.backward()
                opt_g.step()

    def _sample(self, num_samples: int) -> torch.Tensor:
        z = torch.randn(num_samples, self.noise_dim)
        with torch.no_grad():
            samples = self.model.generator(z)
            return torch.sigmoid(samples).view(num_samples, -1)
