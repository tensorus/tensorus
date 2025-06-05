from torch import nn
from torchvision import models

from .cnn_base import CNNModelBase


class MobileNetModel(CNNModelBase):
    """MobileNetV2 classifier using ``torchvision.models.mobilenet_v2``."""

    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = False,
        lr: float = 1e-3,
        epochs: int = 1,
    ) -> None:
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        if num_classes != 1000:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        super().__init__(model, lr=lr, epochs=epochs)
