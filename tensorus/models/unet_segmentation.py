import numpy as np
import torch
from typing import Any
import segmentation_models_pytorch as smp

from .base import TensorusModel


class UNetSegmentationModel(TensorusModel):
    """UNet model using ``segmentation_models_pytorch``."""

    def __init__(
        self,
        encoder_name: str = "resnet18",
        in_channels: int = 3,
        num_classes: int = 1,
        pretrained: bool = False,
        lr: float = 1e-3,
        epochs: int = 1,
    ) -> None:
        weights = "imagenet" if pretrained else None
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=num_classes,
        )
        self.lr = lr
        self.epochs = epochs

    def _to_tensor(self, arr: Any) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.float()
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).float()
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    def fit(self, X: Any, y: Any) -> None:  # pragma: no cover - training not implemented
        pass

    def predict(self, X: Any) -> torch.Tensor:
        X_t = self._to_tensor(X)
        if X_t.ndim == 3:
            X_t = X_t.unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            out = self.model(X_t)
        return out

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict()}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        self.model.load_state_dict(data["state_dict"])
