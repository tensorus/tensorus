import numpy as np
import torch  # ensure torch is available for model imports

from tensorus.models.arima_model import ARIMAModel
from tensorus.models.sarima_model import SARIMAModel
from tensorus.models.exponential_smoothing_model import ExponentialSmoothingModel
from tensorus.models.garch_model import GARCHModel


def test_arima_model(tmp_path):
    data = np.arange(20, dtype=float)
    model = ARIMAModel(order=(1, 0, 0))
    model.fit(data)
    preds = model.predict(5)
    assert len(preds) == 5

    save_path = tmp_path / "arima.pkl"
    model.save(str(save_path))
    model2 = ARIMAModel()
    model2.load(str(save_path))
    preds2 = model2.predict(5)
    assert np.allclose(preds2, preds)


def test_sarima_model(tmp_path):
    data = np.sin(np.linspace(0, 6, 30))
    model = SARIMAModel(order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
    model.fit(data)
    preds = model.predict(3)
    assert len(preds) == 3

    save_path = tmp_path / "sarima.pkl"
    model.save(str(save_path))
    model2 = SARIMAModel()
    model2.load(str(save_path))
    preds2 = model2.predict(3)
    assert np.allclose(preds2, preds)


def test_exponential_smoothing_model(tmp_path):
    data = np.linspace(1, 10, 30)
    model = ExponentialSmoothingModel(trend="add")
    model.fit(data)
    preds = model.predict(4)
    assert len(preds) == 4

    save_path = tmp_path / "exp.pkl"
    model.save(str(save_path))
    model2 = ExponentialSmoothingModel(trend="add")
    model2.load(str(save_path))
    preds2 = model2.predict(4)
    assert np.allclose(preds2, preds)


def test_garch_model(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.standard_normal(50)
    model = GARCHModel(p=1, q=1)
    model.fit(data)
    preds = model.predict(2)
    assert len(preds) == 2

    save_path = tmp_path / "garch.pkl"
    model.save(str(save_path))
    model2 = GARCHModel(p=1, q=1)
    model2.load(str(save_path))
    preds2 = model2.predict(2)
    assert np.allclose(preds2, preds)
