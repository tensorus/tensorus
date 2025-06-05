import numpy as np
import torch
from torch import nn
from transformers import (
    BertForSequenceClassification,
    BertConfig,
    GPT2LMHeadModel,
    GPT2Config,
    T5ForConditionalGeneration,
    T5Config,
)
from torchvision import models

from .base import TensorusModel
from .cnn_base import CNNModelBase


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerModel(TensorusModel, nn.Module):
    """Generic encoder-decoder transformer."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 32,
        num_heads: int = 2,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 64,
        output_dim: int | None = None,
        lr: float = 1e-3,
        epochs: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.num_encoder_layers = int(num_encoder_layers)
        self.num_decoder_layers = int(num_decoder_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.output_dim = int(output_dim or input_dim)
        self.lr = lr
        self.epochs = epochs

        self.src_embed = nn.Embedding(self.input_dim, self.model_dim)
        self.tgt_embed = nn.Embedding(self.output_dim, self.model_dim)
        self.pos_encoder = PositionalEncoding(self.model_dim)
        self.transformer = nn.Transformer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            batch_first=True,
        )
        self.fc_out = nn.Linear(self.model_dim, self.output_dim)

    def _to_tensor(self, arr: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.long()
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).long()
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_emb = self.pos_encoder(self.src_embed(src))
        tgt_emb = self.pos_encoder(self.tgt_embed(tgt))
        out = self.transformer(src_emb, tgt_emb)
        return self.fc_out(out)

    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> None:
        src = self._to_tensor(X)
        tgt = self._to_tensor(y)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        self.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            output = self.forward(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, self.output_dim), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

    def predict(self, X: np.ndarray | torch.Tensor, max_len: int | None = None) -> torch.Tensor:
        src = self._to_tensor(X)
        if max_len is None:
            max_len = src.size(1)
        tgt = torch.zeros(src.size(0), 1, dtype=torch.long)
        self.eval()
        with torch.no_grad():
            for _ in range(max_len):
                out = self.forward(src, tgt)
                next_token = out[:, -1].argmax(dim=-1, keepdim=True)
                tgt = torch.cat([tgt, next_token], dim=1)
        return tgt[:, 1:]

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "model_dim": self.model_dim,
                    "num_heads": self.num_heads,
                    "num_encoder_layers": self.num_encoder_layers,
                    "num_decoder_layers": self.num_decoder_layers,
                    "dim_feedforward": self.dim_feedforward,
                    "output_dim": self.output_dim,
                    "lr": self.lr,
                    "epochs": self.epochs,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        cfg = data.get("config", {})
        self.__init__(
            cfg.get("input_dim", self.input_dim),
            cfg.get("model_dim", self.model_dim),
            cfg.get("num_heads", self.num_heads),
            cfg.get("num_encoder_layers", self.num_encoder_layers),
            cfg.get("num_decoder_layers", self.num_decoder_layers),
            cfg.get("dim_feedforward", self.dim_feedforward),
            cfg.get("output_dim", self.output_dim),
            cfg.get("lr", self.lr),
            cfg.get("epochs", self.epochs),
        )
        self.load_state_dict(data["state_dict"])


class BERTModel(TensorusModel):
    """Wrapper around ``BertForSequenceClassification``."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        model_name: str = "bert-base-uncased",
        lr: float = 1e-4,
        epochs: int = 1,
    ) -> None:
        self.num_classes = int(num_classes)
        self.pretrained = bool(pretrained)
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs
        if pretrained:
            self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        else:
            cfg = BertConfig(num_labels=num_classes)
            self.model = BertForSequenceClassification(cfg)

    def _to_tensor(self, arr: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.long()
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).long()
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> None:
        X_t = self._to_tensor(X)
        y_t = self._to_tensor(y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            out = self.model(input_ids=X_t, labels=y_t)
            out.loss.backward()
            optimizer.step()

    def predict(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        X_t = self._to_tensor(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids=X_t).logits
            return logits.argmax(dim=1)

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(), "config": self.model.config.to_dict()}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        cfg = BertConfig.from_dict(data["config"])
        self.model = BertForSequenceClassification(cfg)
        self.model.load_state_dict(data["state_dict"])


class GPTModel(TensorusModel):
    """Wrapper around ``GPT2LMHeadModel``."""

    def __init__(
        self,
        pretrained: bool = True,
        model_name: str = "gpt2",
        lr: float = 1e-4,
        epochs: int = 1,
    ) -> None:
        self.pretrained = bool(pretrained)
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs
        if pretrained:
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
        else:
            cfg = GPT2Config()
            self.model = GPT2LMHeadModel(cfg)

    def _to_tensor(self, arr: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.long()
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).long()
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> None:
        X_t = self._to_tensor(X)
        y_t = self._to_tensor(y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            out = self.model(input_ids=X_t, labels=y_t)
            out.loss.backward()
            optimizer.step()

    def predict(self, X: np.ndarray | torch.Tensor, max_length: int | None = None) -> torch.Tensor:
        X_t = self._to_tensor(X)
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=X_t,
                max_length=(max_length or X_t.size(1) + 1),
            )
        return generated

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(), "config": self.model.config.to_dict()}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        cfg = GPT2Config.from_dict(data["config"])
        self.model = GPT2LMHeadModel(cfg)
        self.model.load_state_dict(data["state_dict"])


class T5Model(TensorusModel):
    """Wrapper around ``T5ForConditionalGeneration``."""

    def __init__(
        self,
        pretrained: bool = True,
        model_name: str = "t5-small",
        lr: float = 1e-4,
        epochs: int = 1,
    ) -> None:
        self.pretrained = bool(pretrained)
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs
        if pretrained:
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            cfg = T5Config(bos_token_id=0, eos_token_id=1, decoder_start_token_id=0)
            self.model = T5ForConditionalGeneration(cfg)

    def _to_tensor(self, arr: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.long()
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).long()
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> None:
        X_t = self._to_tensor(X)
        y_t = self._to_tensor(y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            out = self.model(input_ids=X_t, labels=y_t)
            out.loss.backward()
            optimizer.step()

    def predict(self, X: np.ndarray | torch.Tensor, max_length: int | None = None) -> torch.Tensor:
        X_t = self._to_tensor(X)
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(input_ids=X_t, max_length=(max_length or X_t.size(1) + 1))
        return generated

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(), "config": self.model.config.to_dict()}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        cfg = T5Config.from_dict(data["config"])
        self.model = T5ForConditionalGeneration(cfg)
        self.model.load_state_dict(data["state_dict"])


class VisionTransformerModel(CNNModelBase):
    """Vision Transformer classifier using ``torchvision.models.vit_b_16``."""

    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = False,
        lr: float = 1e-3,
        epochs: int = 1,
    ) -> None:
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        if num_classes != 1000:
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        super().__init__(model, lr=lr, epochs=epochs)
