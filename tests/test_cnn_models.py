import pytest
import torch

from tensorus.models.lenet import LeNetModel
from tensorus.models.alexnet import AlexNetModel
from tensorus.models.vgg import VGGModel
from tensorus.models.resnet import ResNetModel
from tensorus.models.mobilenet import MobileNetModel
from tensorus.models.efficientnet import EfficientNetModel


@pytest.mark.parametrize(
    "Model,input_shape",
    [
        (LeNetModel, (1, 1, 32, 32)),
        (AlexNetModel, (1, 3, 224, 224)),
        (VGGModel, (1, 3, 224, 224)),
        (ResNetModel, (1, 3, 224, 224)),
        (MobileNetModel, (1, 3, 224, 224)),
        (EfficientNetModel, (1, 3, 224, 224)),
    ],
)
def test_cnn_forward(Model, input_shape):
    if Model is LeNetModel:
        model = Model()
    else:
        model = Model(pretrained=False)
    x = torch.randn(*input_shape)
    preds = model.predict(x)
    assert preds.shape[0] == input_shape[0]
