import torch
import numpy as np
from typing import Any

from .base import TensorusModel


class YOLOv5Detector(TensorusModel):
    """YOLOv5 object detector loaded via PyTorch Hub."""

    def __init__(self, model_name: str = "yolov5n", pretrained: bool = True, epochs: int = 1) -> None:
        self.model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=pretrained)
        self.epochs = epochs

    def _to_tensor(self, arr: Any) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.float()
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).float()
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    def train(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - delegates to hub
        """Thin wrapper around ``YOLOv5`` training."""
        if hasattr(self.model, "train"):
            self.model.train(*args, epochs=self.epochs, **kwargs)
        else:
            raise NotImplementedError("Training not supported for this model")

    def predict(self, X: Any):
        X_t = self._to_tensor(X)
        if X_t.ndim == 3:
            X_t = X_t.unsqueeze(0)
        results = self.model(X_t)
        return results

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.model.state_dict()}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        self.model.model.load_state_dict(data["state_dict"])
