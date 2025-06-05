import numpy as np
import torch

from tensorus.models.gcn_classifier import GCNClassifierModel
from tensorus.models.gat_classifier import GATClassifierModel


def _simple_graph():
    x = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    edge_index = np.array([[0, 1, 2, 3], [1, 0, 3, 2]])
    y = np.array([0, 0, 1, 1])
    return x, edge_index, y


def test_gcn_classifier():
    x, ei, y = _simple_graph()
    model = GCNClassifierModel(input_dim=2, hidden_dim=4, output_dim=2, lr=0.1, epochs=200)
    model.fit(x, ei, y)
    preds = model.predict(x, ei)
    assert torch.equal(preds, torch.tensor(y))


def test_gat_classifier():
    x, ei, y = _simple_graph()
    model = GATClassifierModel(input_dim=2, hidden_dim=4, output_dim=2, heads=2, lr=0.1, epochs=200)
    model.fit(x, ei, y)
    preds = model.predict(x, ei)
    assert torch.equal(preds, torch.tensor(y))

