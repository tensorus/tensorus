import torch
import importlib.util
from pathlib import Path
import sys
import types

tensorus_pkg = types.ModuleType("tensorus")
models_pkg = types.ModuleType("tensorus.models")
sys.modules.setdefault("tensorus", tensorus_pkg)
sys.modules.setdefault("tensorus.models", models_pkg)

ts_spec = importlib.util.spec_from_file_location(
    "tensorus.tensor_storage",
    Path(__file__).resolve().parents[1] / "tensorus" / "tensor_storage.py",
)
ts_mod = importlib.util.module_from_spec(ts_spec)
ts_spec.loader.exec_module(ts_mod)  # type: ignore
sys.modules["tensorus.tensor_storage"] = ts_mod

base_spec = importlib.util.spec_from_file_location(
    "tensorus.models.base",
    Path(__file__).resolve().parents[1] / "tensorus" / "models" / "base.py",
)
base_mod = importlib.util.module_from_spec(base_spec)
base_spec.loader.exec_module(base_mod)  # type: ignore
sys.modules["tensorus.models.base"] = base_mod

spec_frcnn = importlib.util.spec_from_file_location(
    "tensorus.models.faster_rcnn",
    Path(__file__).resolve().parents[1] / "tensorus" / "models" / "faster_rcnn.py",
)
faster_rcnn = importlib.util.module_from_spec(spec_frcnn)
spec_frcnn.loader.exec_module(faster_rcnn)  # type: ignore
sys.modules["tensorus.models.faster_rcnn"] = faster_rcnn

spec_unet = importlib.util.spec_from_file_location(
    "tensorus.models.unet_segmentation",
    Path(__file__).resolve().parents[1] / "tensorus" / "models" / "unet_segmentation.py",
)
unet_segmentation = importlib.util.module_from_spec(spec_unet)
spec_unet.loader.exec_module(unet_segmentation)  # type: ignore
sys.modules["tensorus.models.unet_segmentation"] = unet_segmentation

FasterRCNNModel = faster_rcnn.FasterRCNNModel
UNetSegmentationModel = unet_segmentation.UNetSegmentationModel


def test_faster_rcnn_forward():
    model = FasterRCNNModel(num_classes=2, pretrained=False)
    x = torch.randn(1, 3, 128, 128)
    preds = model.predict(x)
    assert isinstance(preds, list)
    assert "boxes" in preds[0]
    assert preds[0]["boxes"].shape[1] == 4


def test_unet_forward():
    model = UNetSegmentationModel(in_channels=3, num_classes=2, pretrained=False)
    x = torch.randn(1, 3, 64, 64)
    out = model.predict(x)
    assert out.shape == (1, 2, 64, 64)
