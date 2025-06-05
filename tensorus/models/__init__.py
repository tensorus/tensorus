"""Model utilities for Tensorus."""

from .base import TensorusModel
from .registry import register_model, get_model
from .linear_regression import LinearRegressionModel
from .logistic_regression import LogisticRegressionModel
from .ridge_regression import RidgeRegressionModel
from .lasso_regression import LassoRegressionModel
from .elastic_net_regression import ElasticNetRegressionModel
from .poisson_regressor import PoissonRegressorModel
from .polynomial_regression import PolynomialRegressionModel
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
from .gradient_boosting_classifier import GradientBoostingClassifierModel
from .gradient_boosting_regressor import GradientBoostingRegressorModel
from .xgboost_classifier import XGBoostClassifierModel
from .xgboost_regressor import XGBoostRegressorModel
from .lightgbm_classifier import LightGBMClassifierModel
from .lightgbm_regressor import LightGBMRegressorModel
from .catboost_classifier import CatBoostClassifierModel
from .catboost_regressor import CatBoostRegressorModel
from .pca_decomposition import PCADecompositionModel
from .factor_analysis import FactorAnalysisModel
from .cca import CCAModel
from .tsne_embedding import TSNEEmbeddingModel
from .umap_embedding import UMAPEmbeddingModel
from .isolation_forest import IsolationForestModel
from .one_class_svm import OneClassSVMModel
from .mlp_classifier import MLPClassifierModel
from .stacked_rbm_classifier import StackedRBMClassifierModel
from .label_propagation import LabelPropagationModel
from .self_training_classifier import SelfTrainingClassifierModel
from .arima_model import ARIMAModel
from .sarima_model import SARIMAModel
from .exponential_smoothing_model import ExponentialSmoothingModel
from .garch_model import GARCHModel
from .cox_ph_model import CoxPHModel
from .anova import AnovaModel
from .manova import ManovaModel
from .mixed_effects_model import MixedEffectsModel
from .structural_equation_model import StructuralEquationModel
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
    "PoissonRegressorModel",
    "PolynomialRegressionModel",
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
    "GradientBoostingClassifierModel",
    "GradientBoostingRegressorModel",
    "XGBoostClassifierModel",
    "XGBoostRegressorModel",
    "LightGBMClassifierModel",
    "LightGBMRegressorModel",
    "CatBoostClassifierModel",
    "CatBoostRegressorModel",
    "PCADecompositionModel",
    "FactorAnalysisModel",
    "CCAModel",
    "TSNEEmbeddingModel",
    "UMAPEmbeddingModel",
    "IsolationForestModel",
    "OneClassSVMModel",
    "LabelPropagationModel",
    "SelfTrainingClassifierModel",
    "MLPClassifierModel",
    "StackedRBMClassifierModel",
    "ARIMAModel",
    "SARIMAModel",
    "ExponentialSmoothingModel",
    "GARCHModel",
    "CoxPHModel",
    "AnovaModel",
    "ManovaModel",
    "MixedEffectsModel",
    "StructuralEquationModel",
    "load_xy_from_storage",
    "store_predictions",
]

# Register models in the simple registry for convenience
register_model("GradientBoostingClassifier", GradientBoostingClassifierModel)
register_model("GradientBoostingRegressor", GradientBoostingRegressorModel)
register_model("XGBoostClassifier", XGBoostClassifierModel)
register_model("XGBoostRegressor", XGBoostRegressorModel)
register_model("LightGBMClassifier", LightGBMClassifierModel)
register_model("LightGBMRegressor", LightGBMRegressorModel)
register_model("CatBoostClassifier", CatBoostClassifierModel)
register_model("CatBoostRegressor", CatBoostRegressorModel)
