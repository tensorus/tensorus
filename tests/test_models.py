import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from tensorus.models.linear_regression import LinearRegressionModel
from tensorus.models.logistic_regression import LogisticRegressionModel
from tensorus.models.ridge_regression import RidgeRegressionModel
from tensorus.models.lasso_regression import LassoRegressionModel
from tensorus.models.decision_tree_classifier import DecisionTreeClassifierModel
from tensorus.models.svm_classifier import SVMClassifierModel
from tensorus.models.svr import SVRModel
from tensorus.models.kmeans import KMeansClusteringModel
from tensorus.models.dbscan import DBSCANClusteringModel
from tensorus.models.agglomerative import AgglomerativeClusteringModel
from tensorus.models.gaussian_mixture import GaussianMixtureModel
from tensorus.models.knn_classifier import KNNClassifierModel
from tensorus.models.gaussian_nb_classifier import GaussianNBClassifierModel
from tensorus.models.lda_classifier import LDAClassifierModel
from tensorus.models.qda_classifier import QDAClassifierModel
from tensorus.models.random_forest_classifier import RandomForestClassifierModel
from tensorus.models.random_forest_regressor import RandomForestRegressorModel
from tensorus.models.decision_tree_regressor import DecisionTreeRegressorModel
from tensorus.models.pca_decomposition import PCADecompositionModel
from tensorus.models.tsne_embedding import TSNEEmbeddingModel
from tensorus.models.mlp_classifier import MLPClassifierModel
from tensorus.models.elastic_net_regression import ElasticNetRegressionModel
from tensorus.models.poisson_regressor import PoissonRegressorModel
from tensorus.models.polynomial_regression import PolynomialRegressionModel
from tensorus.tensor_storage import TensorStorage
from tensorus.models.utils import load_xy_from_storage, store_predictions


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


def test_mlp_classifier(tmp_path):
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = MLPClassifierModel(
        input_size=1,
        hidden_layers=[8],
        output_size=2,
        lr=0.1,
        epochs=200,
    )
    model.fit(X, y)
    pred = model.predict(X)
    assert torch.equal(pred, torch.tensor(y))

    save_path = tmp_path / "mlp.pt"
    model.save(str(save_path))
    model2 = MLPClassifierModel(input_size=1, hidden_layers=[8], output_size=2)
    model2.load(str(save_path))
    pred2 = model2.predict(X)
    assert torch.equal(pred, pred2)


def test_linear_regression_with_tensor_storage(tmp_path):
    storage = TensorStorage(storage_path=tmp_path / "storage")
    ds_name = "train_ds"
    storage.create_dataset(ds_name)

    X_np = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_np = np.array([3.0, 5.0, 7.0, 9.0])
    for x, y in zip(X_np, y_np):
        storage.insert(ds_name, torch.tensor(x), {"label": float(y)})

    X, y = load_xy_from_storage(storage, ds_name, target_field="label")
    model = LinearRegressionModel()
    model.fit(X, y)
    preds = model.predict(X)

    preds_ds = "predictions_ds"
    rec_id = store_predictions(storage, preds_ds, preds, model_name="LinearRegressionModel")
    stored = storage.get_dataset_with_metadata(preds_ds)

    assert len(stored) == 1
    assert torch.allclose(stored[0]["tensor"], preds)
    assert stored[0]["metadata"]["record_id"] == rec_id


def test_sklearn_models(tmp_path):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])

    ridge = RidgeRegressionModel(alpha=0.1)
    ridge.fit(X, y)
    assert np.allclose(ridge.predict(X), y, atol=1e-1)

    lasso = LassoRegressionModel(alpha=0.01)
    lasso.fit(X, y)
    assert np.allclose(lasso.predict(X), y, atol=1e-1)

    tree = DecisionTreeClassifierModel()
    Xc = np.array([[0.0], [1.0], [2.0], [3.0]])
    yc = np.array([0, 0, 1, 1])
    tree.fit(Xc, yc)
    assert np.array_equal(tree.predict(Xc), yc)

    km = KMeansClusteringModel(n_clusters=2, random_state=42)
    Xk = np.array([[0.0], [0.1], [1.0], [1.1]])
    km.fit(Xk)
    labels = km.predict(Xk)
    assert set(labels) == {0, 1}

    svc = SVMClassifierModel(kernel="linear", C=1.0)
    svc.fit(Xc, yc)
    assert np.array_equal(svc.predict(Xc), yc)

    knn = KNNClassifierModel(n_neighbors=1)
    knn.fit(Xc, yc)
    assert np.array_equal(knn.predict(Xc), yc)

    svr = SVRModel(kernel="linear", C=1.0, epsilon=0.0)
    svr.fit(X, y)
    assert np.allclose(svr.predict(X), y, atol=1e-1)


def test_random_forest_models(tmp_path):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_reg = np.array([3.0, 5.0, 7.0, 9.0])
    y_clf = np.array([0, 0, 1, 1])

    clf = RandomForestClassifierModel(n_estimators=10, random_state=42)
    clf.fit(X, y_clf)
    preds_clf = clf.predict(X)
    assert len(preds_clf) == len(y_clf)
    save_clf = tmp_path / "rf_clf.joblib"
    clf.save(str(save_clf))
    clf2 = RandomForestClassifierModel()
    clf2.load(str(save_clf))
    assert np.array_equal(clf2.predict(X), preds_clf)

    reg = RandomForestRegressorModel(n_estimators=10, random_state=42)
    reg.fit(X, y_reg)
    preds_reg = reg.predict(X)
    assert len(preds_reg) == len(y_reg)
    save_reg = tmp_path / "rf_reg.joblib"
    reg.save(str(save_reg))
    reg2 = RandomForestRegressorModel()
    reg2.load(str(save_reg))
    assert np.allclose(reg2.predict(X), preds_reg)


def test_decision_tree_regressor_model(tmp_path):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])

    reg = DecisionTreeRegressorModel(random_state=42)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert len(preds) == len(y)

    save_path = tmp_path / "dt_reg.joblib"
    reg.save(str(save_path))
    reg2 = DecisionTreeRegressorModel()
    reg2.load(str(save_path))
    assert np.allclose(reg2.predict(X), preds)


def test_dimensionality_reduction_models(tmp_path):
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])

    pca = PCADecompositionModel(n_components=1)
    X_pca = pca.fit_transform(X)
    assert X_pca.shape == (4, 1)

    save_pca = tmp_path / "pca.joblib"
    pca.save(str(save_pca))
    pca2 = PCADecompositionModel(n_components=1)
    pca2.load(str(save_pca))
    assert np.allclose(pca2.transform(X), X_pca)

    tsne = TSNEEmbeddingModel(n_components=2, perplexity=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    assert X_tsne.shape == (4, 2)

    save_tsne = tmp_path / "tsne.joblib"
    tsne.save(str(save_tsne))
    tsne2 = TSNEEmbeddingModel(n_components=2)
    tsne2.load(str(save_tsne))
    if hasattr(tsne2.model, "transform"):
        assert np.allclose(tsne2.transform(X), tsne.transform(X))


def test_elastic_net_regression_model(tmp_path):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])

    enet = ElasticNetRegressionModel(alpha=0.1, l1_ratio=0.5)
    enet.fit(X, y)
    preds = enet.predict(X)
    assert np.allclose(preds, y, atol=2e-1)

    save_path = tmp_path / "enet.joblib"
    enet.save(str(save_path))
    enet2 = ElasticNetRegressionModel()
    enet2.load(str(save_path))
    preds2 = enet2.predict(X)
    assert np.allclose(preds2, preds)


def test_polynomial_regression_model(tmp_path):
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])
    model = PolynomialRegressionModel(degree=2, regularization=0.0)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.allclose(preds, y, atol=1e-1)

    save_path = tmp_path / "poly.joblib"
    model.save(str(save_path))
    model2 = PolynomialRegressionModel()
    model2.load(str(save_path))
    preds2 = model2.predict(X)
    assert np.allclose(preds2, preds)


def test_poisson_regressor_model(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(100, 1))
    coef = 0.8
    y = rng.poisson(np.exp(coef * X[:, 0]))

    model = PoissonRegressorModel(alpha=0.0, max_iter=1000)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape

    save_path = tmp_path / "poisson.joblib"
    model.save(str(save_path))
    model2 = PoissonRegressorModel()
    model2.load(str(save_path))
    preds2 = model2.predict(X)
    assert np.allclose(preds2, preds)


def test_gaussian_nb_classifier_model():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = GaussianNBClassifierModel()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.array_equal(preds, y)


def test_lda_classifier_model():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = LDAClassifierModel()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.array_equal(preds, y)


def test_qda_classifier_model():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = QDAClassifierModel()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.array_equal(preds, y)


def test_clustering_models():
    X = np.array([[0.0], [0.1], [1.0], [1.1]])

    db = DBSCANClusteringModel(eps=0.2, min_samples=1)
    db.fit(X)
    labels_db = db.predict(X)
    assert set(labels_db) == {0, 1}

    ag = AgglomerativeClusteringModel(n_clusters=2)
    ag.fit(X)
    labels_ag = ag.predict(X)
    assert set(labels_ag) == {0, 1}

    gm = GaussianMixtureModel(n_components=2, random_state=42)
    gm.fit(X)
    labels_gm = gm.predict(X)
    assert set(labels_gm) == {0, 1}

