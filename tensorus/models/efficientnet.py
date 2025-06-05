from torch import nn
from torchvision import models

from .cnn_base import CNNModelBase


class EfficientNetModel(CNNModelBase):
    """EfficientNet-B0 classifier using ``torchvision.models.efficientnet_b0``."""

    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = False,
        lr: float = 1e-3,
        epochs: int = 1,
    ) -> None:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        if num_classes != 1000:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        super().__init__(model, lr=lr, epochs=epochs)
