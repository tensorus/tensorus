"""Model utilities for Tensorus."""

from .base import TensorusModel
from .registry import register_model, get_model
from .linear_regression import LinearRegressionModel
from .logistic_regression import LogisticRegressionModel
from .ridge_regression import RidgeRegressionModel
from .lasso_regression import LassoRegressionModel
from .elastic_net_regression import ElasticNetRegressionModel
from .decision_tree_classifier import DecisionTreeClassifierModel
from .svm_classifier import SVMClassifierModel
from .svr import SVRModel
from .kmeans import KMeansClusteringModel
from .knn_classifier import KNNClassifierModel
from .random_forest_classifier import RandomForestClassifierModel
from .random_forest_regressor import RandomForestRegressorModel
from .pca_decomposition import PCADecompositionModel
from .tsne_embedding import TSNEEmbeddingModel
from .mlp_classifier import MLPClassifierModel
from .utils import load_xy_from_storage, store_predictions

__all__ = [
    "TensorusModel",
    "register_model",
    "get_model",
    "LinearRegressionModel",
    "LogisticRegressionModel",
    "RidgeRegressionModel",
    "LassoRegressionModel",
    "ElasticNetRegressionModel",
    "DecisionTreeClassifierModel",
    "SVMClassifierModel",
    "SVRModel",
    "KMeansClusteringModel",
    "KNNClassifierModel",
    "RandomForestClassifierModel",
    "RandomForestRegressorModel",
    "PCADecompositionModel",
    "TSNEEmbeddingModel",
    "MLPClassifierModel",
    "load_xy_from_storage",
    "store_predictions",
]
