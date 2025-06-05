import torch
from typing import Any, List, Optional
from transformers import CLIPModel, CLIPProcessor

from .base import TensorusModel


class MultimodalFoundationModel(TensorusModel):
    """CLIP-like model aligning text and image representations."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        lr: float = 1e-4,
        epochs: int = 1,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)

    def fit(self, images: List[Any], texts: List[str]) -> None:
        """Fine-tune on paired ``images`` and ``texts``."""
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        target = torch.arange(len(texts), device=self.device)
        for _ in range(self.epochs):
            optimizer.zero_grad()
            out = self.model(**inputs)
            loss_i = torch.nn.functional.cross_entropy(out.logits_per_image, target)
            loss_t = torch.nn.functional.cross_entropy(out.logits_per_text, target)
            loss = (loss_i + loss_t) / 2
            loss.backward()
            optimizer.step()

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            return self.model.get_text_features(**inputs)

    def encode_image(self, images: List[Any]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.model.get_image_features(**inputs)

    def predict(self, images: List[Any], texts: List[str]) -> torch.Tensor:
        """Return similarity scores between ``images`` and ``texts``."""
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        return out.logits_per_image

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model = CLIPModel.from_pretrained(path)
        self.processor = CLIPProcessor.from_pretrained(path)
        self.model.to(self.device)
