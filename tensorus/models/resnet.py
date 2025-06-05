from torch import nn
from torchvision import models

from .cnn_base import CNNModelBase


class ResNetModel(CNNModelBase):
    """ResNet18 classifier using ``torchvision.models.resnet18``."""

    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = False,
        lr: float = 1e-3,
        epochs: int = 1,
    ) -> None:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        super().__init__(model, lr=lr, epochs=epochs)
