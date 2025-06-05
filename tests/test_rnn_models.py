import numpy as np
import torch

from tensorus.models.rnn_models import (
    LSTMModule,
    GRUModule,
    BidirectionalWrapper,
    LSTMClassifierModel,
    GRUClassifierModel,
)


def _generate_data():
    X = np.array([
        [[0.0], [0.0], [0.0]],
        [[1.0], [1.0], [1.0]],
        [[2.0], [2.0], [2.0]],
        [[3.0], [3.0], [3.0]],
    ])
    y = np.array([0, 0, 1, 1])
    return X, y


def test_lstm_classifier():
    X, y = _generate_data()
    model = LSTMClassifierModel(
        input_size=1,
        hidden_size=4,
        num_layers=2,
        n_classes=2,
        lr=0.1,
        epochs=200,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert torch.equal(preds, torch.tensor(y))


def test_gru_classifier():
    X, y = _generate_data()
    model = GRUClassifierModel(
        input_size=1,
        hidden_size=4,
        num_layers=2,
        n_classes=2,
        lr=0.1,
        epochs=200,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert torch.equal(preds, torch.tensor(y))


def test_bidirectional_wrapper_output_dim():
    module = LSTMModule(input_size=2, hidden_size=3)
    wrapper = BidirectionalWrapper(module)
    x = torch.randn(5, 7, 2)
    out = wrapper(x)
    assert out.shape == (5, 7, 6)
