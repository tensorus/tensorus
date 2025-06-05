import pytest
import os

os.environ["TENSORUS_MINIMAL_IMPORT"] = "1"

pytest.importorskip("torch")
import torch
import importlib.util
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "tensorus" / "models"
import sys
import types

pkg = types.ModuleType("tensorus.models")
pkg.__path__ = [str(BASE)]
sys.modules.setdefault("tensorus.models", pkg)


def _load(name: str):
    path = BASE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tensorus.models.{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


CollaborativeFilteringModel = _load(
    "collaborative_filtering"
).CollaborativeFilteringModel
MatrixFactorizationModel = _load("matrix_factorization").MatrixFactorizationModel
NeuralCollaborativeFilteringModel = _load("neural_cf").NeuralCollaborativeFilteringModel


def _toy_matrix():
    return torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ]
    )


def test_collaborative_filtering():
    R = _toy_matrix()
    model = CollaborativeFilteringModel()
    model.fit(R)
    preds = model.predict()
    assert preds.shape == R.shape
    assert not torch.isnan(preds).any()


def test_matrix_factorization():
    R = _toy_matrix()
    model = MatrixFactorizationModel(n_factors=2, lr=0.05, epochs=200)
    model.fit(R)
    preds = model.predict()
    mse = ((preds - R) ** 2).mean()
    assert mse < 0.5


def test_neural_cf():
    R = _toy_matrix()
    users, items = torch.nonzero(R >= 0, as_tuple=True)
    ratings = R[users, items]
    X = torch.stack([users, items], dim=1)
    model = NeuralCollaborativeFilteringModel(
        n_users=R.shape[0],
        n_items=R.shape[1],
        embed_dim=4,
        hidden_layers=[8],
        lr=0.05,
        epochs=200,
    )
    model.fit(X, ratings)
    preds = model.predict(X)
    mse = ((preds - ratings) ** 2).mean()
    assert mse < 0.5
