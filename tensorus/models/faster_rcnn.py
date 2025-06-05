import numpy as np
import torch
from typing import Any, List, Dict
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .base import TensorusModel


class FasterRCNNModel(TensorusModel):
    """Wrapper around ``torchvision`` Faster R-CNN."""

    def __init__(
        self,
        num_classes: int = 91,
        pretrained: bool = False,
        lr: float = 1e-3,
        epochs: int = 1,
    ) -> None:
        weights = (
            models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            if pretrained
            else None
        )
        self.model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        if num_classes != 91:
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.lr = lr
        self.epochs = epochs

    def _to_tensor(self, arr: Any) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.float()
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).float()
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    def fit(self, X: Any, y: Any) -> None:  # pragma: no cover - training not implemented
        """Training is not supported for this model."""
        raise NotImplementedError(
            "FasterRCNNModel currently supports inference only"
        )

    def predict(self, X: Any) -> List[Dict[str, torch.Tensor]]:
        X_t = self._to_tensor(X)
        if X_t.ndim == 3:
            X_t = X_t.unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(list(img for img in X_t))
        return outputs

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict()}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        self.model.load_state_dict(data["state_dict"])
