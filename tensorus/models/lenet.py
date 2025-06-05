import torch
from torch import nn

from .cnn_base import CNNModelBase


class LeNet(nn.Module):
    """Simple LeNet-5 architecture."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class LeNetModel(CNNModelBase):
    """LeNet classifier."""

    def __init__(self, num_classes: int = 10, lr: float = 1e-3, epochs: int = 1) -> None:
        super().__init__(LeNet(num_classes=num_classes), lr=lr, epochs=epochs)
