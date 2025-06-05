import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Any

from .generative_base import GenerativeModel


class _VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu_logvar = self.encoder(x)
        mu, logvar = mu_logvar.chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class VAEModel(GenerativeModel):
    """Simple Variational Autoencoder."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        lr: float = 1e-3,
        epochs: int = 1,
        batch_size: int = 32,
    ) -> None:
        super().__init__(_VAE(input_dim, hidden_dim, latent_dim))
        self.input_dim = input_dim
        self.latent_dim = latent_dim
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
        X_t = self._to_tensor(X).view(-1, self.input_dim)
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            for (xb,) in loader:
                optim.zero_grad()
                recon, mu, logvar = self.model(xb)
                recon_loss = nn.functional.binary_cross_entropy_with_logits(
                    recon, xb, reduction="mean"
                )
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kld
                loss.backward()
                optim.step()

    def _sample(self, num_samples: int) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        with torch.no_grad():
            recon = self.model.decode(z)
            return torch.sigmoid(recon).view(num_samples, -1)
