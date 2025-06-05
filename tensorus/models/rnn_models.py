import copy
from typing import Any

import numpy as np
import torch
from torch import nn

from .base import TensorusModel


class LSTMModule(nn.Module):
    """Simple wrapper around ``nn.LSTM`` supporting stacked layers."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


class GRUModule(nn.Module):
    """Simple wrapper around ``nn.GRU`` supporting stacked layers."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out


class BidirectionalWrapper(nn.Module):
    """Bidirectional processing wrapper for RNN modules."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.fwd = module
        self.bwd = copy.deepcopy(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_fwd = self.fwd(x)
        out_bwd = self.bwd(torch.flip(x, dims=[1]))
        out_bwd = torch.flip(out_bwd, dims=[1])
        return torch.cat([out_fwd, out_bwd], dim=-1)


class LSTMClassifierModel(TensorusModel):
    """Sequence classifier using an LSTM backbone."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        n_classes: int = 2,
        bidirectional: bool = False,
        lr: float = 1e-3,
        epochs: int = 10,
    ) -> None:
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.n_classes = int(n_classes)
        self.bidirectional = bool(bidirectional)
        self.lr = lr
        self.epochs = epochs
        base = LSTMModule(self.input_size, self.hidden_size, self.num_layers)
        if bidirectional:
            self.rnn = BidirectionalWrapper(base)
            out_dim = self.hidden_size * 2
        else:
            self.rnn = base
            out_dim = self.hidden_size
        self.fc = nn.Linear(out_dim, self.n_classes)

    def _to_tensor(self, arr: Any) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.float()
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).float()
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    def _to_label_tensor(self, arr: Any) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.long()
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).long()
        raise TypeError("Labels must be a torch.Tensor or numpy.ndarray")

    def fit(self, X: Any, y: Any) -> None:
        X_t = self._to_tensor(X)
        y_t = self._to_label_tensor(y)
        optimizer = torch.optim.Adam(list(self.rnn.parameters()) + list(self.fc.parameters()), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        self.rnn.train()
        self.fc.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            out = self.rnn(X_t)
            logits = self.fc(out[:, -1, :])
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

    def predict(self, X: Any) -> torch.Tensor:
        X_t = self._to_tensor(X)
        self.rnn.eval()
        self.fc.eval()
        with torch.no_grad():
            out = self.rnn(X_t)
            logits = self.fc(out[:, -1, :])
            return logits.argmax(dim=1)

    def save(self, path: str) -> None:
        torch.save(
            {
                'state_dict': {
                    'rnn': self.rnn.state_dict(),
                    'fc': self.fc.state_dict(),
                },
                'config': {
                    'input_size': self.input_size,
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers,
                    'n_classes': self.n_classes,
                    'bidirectional': self.bidirectional,
                    'lr': self.lr,
                    'epochs': self.epochs,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, map_location='cpu')
        cfg = data.get('config', {})
        self.__init__(
            cfg.get('input_size', self.input_size),
            cfg.get('hidden_size', self.hidden_size),
            cfg.get('num_layers', self.num_layers),
            cfg.get('n_classes', self.n_classes),
            cfg.get('bidirectional', self.bidirectional),
            cfg.get('lr', self.lr),
            cfg.get('epochs', self.epochs),
        )
        self.rnn.load_state_dict(data['state_dict']['rnn'])
        self.fc.load_state_dict(data['state_dict']['fc'])


class GRUClassifierModel(LSTMClassifierModel):
    """Sequence classifier using a GRU backbone."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        n_classes: int = 2,
        bidirectional: bool = False,
        lr: float = 1e-3,
        epochs: int = 10,
    ) -> None:
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.n_classes = int(n_classes)
        self.bidirectional = bool(bidirectional)
        self.lr = lr
        self.epochs = epochs
        base = GRUModule(self.input_size, self.hidden_size, self.num_layers)
        if bidirectional:
            self.rnn = BidirectionalWrapper(base)
            out_dim = self.hidden_size * 2
        else:
            self.rnn = base
            out_dim = self.hidden_size
        self.fc = nn.Linear(out_dim, self.n_classes)
