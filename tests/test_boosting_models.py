import numpy as np
import pytest

from tensorus.models.gradient_boosting_classifier import GradientBoostingClassifierModel
from tensorus.models.gradient_boosting_regressor import GradientBoostingRegressorModel
from tensorus.models.xgboost_classifier import XGBoostClassifierModel
from tensorus.models.xgboost_regressor import XGBoostRegressorModel
from tensorus.models.lightgbm_classifier import LightGBMClassifierModel
from tensorus.models.lightgbm_regressor import LightGBMRegressorModel
from tensorus.models.catboost_classifier import CatBoostClassifierModel
from tensorus.models.catboost_regressor import CatBoostRegressorModel


@pytest.mark.parametrize("Model", [
    GradientBoostingClassifierModel,
    XGBoostClassifierModel,
    LightGBMClassifierModel,
    CatBoostClassifierModel,
])
def test_boosting_classifiers(Model):
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = Model()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


@pytest.mark.parametrize("Model", [
    GradientBoostingRegressorModel,
    XGBoostRegressorModel,
    LightGBMRegressorModel,
    CatBoostRegressorModel,
])
def test_boosting_regressors(Model):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])
    model = Model()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
