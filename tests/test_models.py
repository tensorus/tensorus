import numpy as np
import torch

from tensorus.models.linear_regression import LinearRegressionModel
from tensorus.models.logistic_regression import LogisticRegressionModel


def test_linear_regression_fit_predict(tmp_path):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])
    model = LinearRegressionModel()
    model.fit(X, y)
    pred = model.predict(X)
    assert torch.allclose(pred, torch.tensor([3.0, 5.0, 7.0, 9.0]), atol=1e-4)

    save_path = tmp_path / "lin.pt"
    model.save(str(save_path))
    model2 = LinearRegressionModel()
    model2.load(str(save_path))
    pred2 = model2.predict(X)
    assert torch.allclose(pred, pred2)


def test_logistic_regression_fit_predict(tmp_path):
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = LogisticRegressionModel(lr=0.5, epochs=2000)
    model.fit(X, y)
    pred = model.predict(X)
    assert torch.equal(pred, torch.tensor([0.0, 0.0, 1.0, 1.0]))

    save_path = tmp_path / "log.pt"
    model.save(str(save_path))
    model2 = LogisticRegressionModel()
    model2.load(str(save_path))
    pred2 = model2.predict(X)
    assert torch.equal(pred, pred2)
