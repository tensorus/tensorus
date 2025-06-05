"""Model utilities for Tensorus."""

from .base import TensorusModel
from .registry import register_model, get_model
from .linear_regression import LinearRegressionModel
from .logistic_regression import LogisticRegressionModel
from .ridge_regression import RidgeRegressionModel
from .lasso_regression import LassoRegressionModel
from .elastic_net_regression import ElasticNetRegressionModel
from .decision_tree_classifier import DecisionTreeClassifierModel
from .decision_tree_regressor import DecisionTreeRegressorModel
from .svm_classifier import SVMClassifierModel
from .svr import SVRModel
from .kmeans import KMeansClusteringModel
from .dbscan import DBSCANClusteringModel
from .agglomerative import AgglomerativeClusteringModel
from .gaussian_mixture import GaussianMixtureModel
from .knn_classifier import KNNClassifierModel
from .gaussian_nb_classifier import GaussianNBClassifierModel
from .lda_classifier import LDAClassifierModel
from .qda_classifier import QDAClassifierModel
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
    "DecisionTreeRegressorModel",
    "SVMClassifierModel",
    "SVRModel",
    "KMeansClusteringModel",
    "DBSCANClusteringModel",
    "AgglomerativeClusteringModel",
    "GaussianMixtureModel",
    "KNNClassifierModel",
    "GaussianNBClassifierModel",
    "LDAClassifierModel",
    "QDAClassifierModel",
    "RandomForestClassifierModel",
    "RandomForestRegressorModel",
    "PCADecompositionModel",
    "TSNEEmbeddingModel",
    "MLPClassifierModel",
    "load_xy_from_storage",
    "store_predictions",
]
