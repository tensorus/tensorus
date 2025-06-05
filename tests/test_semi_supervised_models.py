import numpy as np
from sklearn.svm import SVC

from tensorus.models.label_propagation import LabelPropagationModel
from tensorus.models.self_training_classifier import SelfTrainingClassifierModel


def test_label_propagation_model(tmp_path):
    X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([0, -1, -1, 1, -1, -1])
    model = LabelPropagationModel()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert -1 not in preds

    save_path = tmp_path / "lp.joblib"
    model.save(str(save_path))
    model2 = LabelPropagationModel()
    model2.load(str(save_path))
    preds2 = model2.predict(X)
    assert np.array_equal(preds2, preds)


def test_self_training_classifier_model(tmp_path):
    X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([0, -1, -1, 1, -1, -1])
    base = SVC(probability=True, kernel="linear")
    model = SelfTrainingClassifierModel(base_estimator=base)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert -1 not in preds

    save_path = tmp_path / "st.joblib"
    model.save(str(save_path))
    model2 = SelfTrainingClassifierModel()
    model2.load(str(save_path))
    preds2 = model2.predict(X)
    assert np.array_equal(preds2, preds)
