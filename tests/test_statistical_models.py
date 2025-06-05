import numpy as np
import pandas as pd

from tensorus.models.anova import AnovaModel
from tensorus.models.manova import ManovaModel


def test_anova_model(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = 0.5 * X[:, 0] + rng.normal(size=30)
    model = AnovaModel()
    model.fit(X, y)
    table = model.summary()
    assert table.shape[0] == 3
    assert table.shape[1] >= 4
    save_path = tmp_path / "anova.pkl"
    model.save(str(save_path))
    model2 = AnovaModel()
    model2.load(str(save_path))
    assert model2.summary().equals(table)


def test_manova_model(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    Y = np.column_stack([
        0.5 * X[:, 0] + rng.normal(size=40),
        0.2 * X[:, 1] + rng.normal(size=40),
    ])
    model = ManovaModel()
    model.fit(X, Y)
    frame = model.summary()
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape[1] == 5
    save_path = tmp_path / "manova.pkl"
    model.save(str(save_path))
    model2 = ManovaModel()
    model2.load(str(save_path))
    assert model2.summary().shape == frame.shape
