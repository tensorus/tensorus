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
from .word2vec_model import Word2VecModel
from .glove_model import GloVeModel
from .isolation_forest import IsolationForestModel
from .one_class_svm import OneClassSVMModel
from .mlp_classifier import MLPClassifierModel
from .stacked_rbm_classifier import StackedRBMClassifierModel
from .lenet import LeNetModel
from .alexnet import AlexNetModel
from .vgg import VGGModel
from .resnet import ResNetModel
from .mobilenet import MobileNetModel
from .efficientnet import EfficientNetModel
from .rnn_models import (
    LSTMModule,
    GRUModule,
    BidirectionalWrapper,
    LSTMClassifierModel,
    GRUClassifierModel,
)
from .transformer_models import (
    TransformerModel,
    BERTModel,
    GPTModel,
    T5Model,
    VisionTransformerModel,
)
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
from .vae_model import VAEModel
from .gan_model import GANModel
from .diffusion_model import DiffusionModel
from .flow_based_model import FlowBasedModel
from .gcn_classifier import GCNClassifierModel
from .gat_classifier import GATClassifierModel
from .named_entity_recognition import NamedEntityRecognitionModel
from .faster_rcnn import FasterRCNNModel
from .unet_segmentation import UNetSegmentationModel
from .collaborative_filtering import CollaborativeFilteringModel
from .matrix_factorization import MatrixFactorizationModel
from .neural_cf import NeuralCollaborativeFilteringModel
from .neuro_symbolic_model import NeuroSymbolicModel
from .physics_informed_nn import PhysicsInformedNNModel
from .stacked_generalization import StackedGeneralizationModel
from .mixture_of_experts import MixtureOfExpertsModel
from .large_language_model import LargeLanguageModelWrapper
from .multimodal_foundation import MultimodalFoundationModel
from .fedavg_model import FedAvgModel
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
    "Word2VecModel",
    "GloVeModel",
    "IsolationForestModel",
    "OneClassSVMModel",
    "LabelPropagationModel",
    "SelfTrainingClassifierModel",
    "MLPClassifierModel",
    "StackedRBMClassifierModel",
    "VAEModel",
    "GANModel",
    "DiffusionModel",
    "FlowBasedModel",
    "GCNClassifierModel",
    "GATClassifierModel",
    "NamedEntityRecognitionModel",
    "LeNetModel",
    "AlexNetModel",
    "VGGModel",
    "ResNetModel",
    "MobileNetModel",
    "EfficientNetModel",
    "FasterRCNNModel",
    "UNetSegmentationModel",
    "CollaborativeFilteringModel",
    "MatrixFactorizationModel",
    "NeuralCollaborativeFilteringModel",
    "NeuroSymbolicModel",
    "PhysicsInformedNNModel",
    "StackedGeneralizationModel",
    "MixtureOfExpertsModel",
    "LSTMModule",
    "GRUModule",
    "BidirectionalWrapper",
    "LSTMClassifierModel",
    "GRUClassifierModel",
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
    "TransformerModel",
    "BERTModel",
    "GPTModel",
    "T5Model",
    "VisionTransformerModel",
    "LargeLanguageModelWrapper",
    "MultimodalFoundationModel",
    "FedAvgModel",
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
register_model("LeNet", LeNetModel)
register_model("AlexNet", AlexNetModel)
register_model("VGG", VGGModel)
register_model("ResNet", ResNetModel)
register_model("MobileNet", MobileNetModel)
register_model("EfficientNet", EfficientNetModel)
register_model("LSTMClassifier", LSTMClassifierModel)
register_model("GRUClassifier", GRUClassifierModel)
register_model("Transformer", TransformerModel)
register_model("BERT", BERTModel)
register_model("GPT", GPTModel)
register_model("T5", T5Model)
register_model("VisionTransformer", VisionTransformerModel)
register_model("VAE", VAEModel)
register_model("GAN", GANModel)
register_model("Diffusion", DiffusionModel)
register_model("FlowBased", FlowBasedModel)
register_model("GCNClassifier", GCNClassifierModel)
register_model("GATClassifier", GATClassifierModel)
register_model("Word2Vec", Word2VecModel)
register_model("GloVe", GloVeModel)
register_model("NamedEntityRecognition", NamedEntityRecognitionModel)
register_model("FasterRCNN", FasterRCNNModel)
register_model("UNetSegmentation", UNetSegmentationModel)
register_model("CollaborativeFiltering", CollaborativeFilteringModel)
register_model("MatrixFactorization", MatrixFactorizationModel)
register_model("NeuralCollaborativeFiltering", NeuralCollaborativeFilteringModel)
register_model("NeuroSymbolic", NeuroSymbolicModel)
register_model("PhysicsInformedNN", PhysicsInformedNNModel)
register_model("StackedGeneralization", StackedGeneralizationModel)
register_model("MixtureOfExperts", MixtureOfExpertsModel)
register_model("LargeLanguageModel", LargeLanguageModelWrapper)
register_model("MultimodalFoundation", MultimodalFoundationModel)
register_model("FedAvg", FedAvgModel)
