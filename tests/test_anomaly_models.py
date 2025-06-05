import numpy as np
import pytest

pytest.importorskip("torch")

from tensorus.models.isolation_forest import IsolationForestModel
from tensorus.models.one_class_svm import OneClassSVMModel


def test_isolation_forest_outlier_detection():
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    X = np.vstack([X, [[5.0]]])
    model = IsolationForestModel(contamination=0.05, random_state=42)
    model.fit(X)
    preds = model.predict(X)
    assert preds[-1] == -1
    assert 1 in set(preds[:-1])


def test_one_class_svm_outlier_detection():
    X_train = np.linspace(0, 1, 20).reshape(-1, 1)
    X_test = np.vstack([X_train, [[5.0]]])
    model = OneClassSVMModel(nu=0.05, kernel="rbf", gamma="scale")
    model.fit(X_train)
    preds = model.predict(X_test)
    assert preds[-1] == -1
    assert 1 in set(preds[:-1])

