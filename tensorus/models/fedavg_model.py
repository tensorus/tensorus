import torch
from typing import Any, List
from torch import nn

from .base import TensorusModel


class FedAvgModel(TensorusModel):
    """Federated averaging for a set of client models."""

    def __init__(self, global_model: nn.Module) -> None:
        self.global_model = global_model

    def fit(self, client_state_dicts: List[dict[str, torch.Tensor]]) -> None:
        """Aggregate ``client_state_dicts`` by averaging their parameters."""
        if not client_state_dicts:
            return
        avg_state = {}
        for key in client_state_dicts[0].keys():
            avg_state[key] = torch.stack([sd[key].float() for sd in client_state_dicts]).mean(dim=0)
        self.global_model.load_state_dict(avg_state)

    def predict(self, X: Any) -> Any:
        self.global_model.eval()
        with torch.no_grad():
            return self.global_model(X)

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.global_model.state_dict()}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        self.global_model.load_state_dict(data["state_dict"])
