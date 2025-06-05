import torch
from typing import Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import TensorusModel


class LargeLanguageModelWrapper(TensorusModel):
    """Load a pre-trained causal language model and expose generation APIs."""

    def __init__(
        self,
        model_name: str = "gpt2",
        lr: float = 1e-4,
        epochs: int = 1,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device)

    def fit(self, texts: List[str]) -> None:
        """Fine-tune the language model on ``texts``."""
        enc = self.tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            out = self.model(input_ids=input_ids, labels=input_ids)
            out.loss.backward()
            optimizer.step()

    def generate(self, prompts: List[str], max_length: int = 50, **kwargs: Any) -> List[str]:
        """Generate text continuations for each prompt."""
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            output = self.model.generate(**enc, max_length=max_length, **kwargs)
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)

    def predict(self, prompts: List[str]) -> List[str]:  # type: ignore[override]
        return self.generate(prompts)

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
