import numpy as np
import pandas as pd

from tensorus.models.anova import AnovaModel
from tensorus.models.manova import ManovaModel
from tensorus.models.structural_equation_model import StructuralEquationModel
from tensorus.models.mixed_effects_model import MixedEffectsModel


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


def test_structural_equation_model(tmp_path):
    np.random.seed(0)
    latent = np.random.normal(size=50)
    x1 = latent + np.random.normal(size=50)
    x2 = latent + np.random.normal(size=50)
    x3 = np.random.normal(size=50)
    data = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    desc = "L =~ x1 + x2\nL ~ x3"
    model = StructuralEquationModel(desc)
    model.fit(data)
    preds = model.predict(data)
    assert isinstance(preds, pd.DataFrame)
    save_path = tmp_path / "sem.pkl"
    model.save(str(save_path))
    model2 = StructuralEquationModel(desc)
    model2.load(str(save_path))
    model2.fit(data)
    preds2 = model2.predict(data)
    assert preds2.shape == preds.shape


def test_mixed_effects_model(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 1))
    groups = np.repeat(np.arange(4), 5)
    beta = 0.5
    group_effects = np.array([0.1, -0.2, 0.2, -0.1])
    y = beta * X[:, 0] + group_effects[groups] + rng.normal(scale=0.01, size=20)
    model = MixedEffectsModel()
    model.fit(X, y, groups)
    preds = model.predict(X)
    assert len(preds) == len(y)
    save_path = tmp_path / "mixed.pkl"
    model.save(str(save_path))
    model2 = MixedEffectsModel()
    model2.load(str(save_path))
    preds2 = model2.predict(X)
    assert np.allclose(preds2, preds)
