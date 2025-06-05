import torch
import importlib.util
from pathlib import Path
import sys
import types

# Dynamic import to avoid loading full package on test start
pkg = types.ModuleType("tensorus")
models_pkg = types.ModuleType("tensorus.models")
sys.modules.setdefault("tensorus", pkg)
sys.modules.setdefault("tensorus.models", models_pkg)

storage_spec = importlib.util.spec_from_file_location(
    "tensorus.tensor_storage",
    Path(__file__).resolve().parents[1] / "tensorus" / "tensor_storage.py",
)
storage_mod = importlib.util.module_from_spec(storage_spec)
storage_spec.loader.exec_module(storage_mod)  # type: ignore
sys.modules["tensorus.tensor_storage"] = storage_mod

# Provide a dummy OpenCV module to satisfy YOLOv5 imports
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
# Additional lightweight stubs for optional dependencies
sys.modules.setdefault("pandas", types.ModuleType("pandas"))
sys.modules.setdefault("requests", types.ModuleType("requests"))
sys.modules.setdefault("ultralytics", types.ModuleType("ultralytics"))

base_spec = importlib.util.spec_from_file_location(
    "tensorus.models.base",
    Path(__file__).resolve().parents[1] / "tensorus" / "models" / "base.py",
)
base_mod = importlib.util.module_from_spec(base_spec)
base_spec.loader.exec_module(base_mod)  # type: ignore
sys.modules["tensorus.models.base"] = base_mod

spec_yolo = importlib.util.spec_from_file_location(
    "tensorus.models.yolov5_detector",
    Path(__file__).resolve().parents[1] / "tensorus" / "models" / "yolov5_detector.py",
)
yolov5_mod = importlib.util.module_from_spec(spec_yolo)
spec_yolo.loader.exec_module(yolov5_mod)  # type: ignore
sys.modules["tensorus.models.yolov5_detector"] = yolov5_mod

YOLOv5Detector = yolov5_mod.YOLOv5Detector


def test_yolov5_predict(tmp_path):
    model = YOLOv5Detector(model_name="yolov5n", pretrained=False)
    img = torch.zeros(3, 32, 32)
    model.fit(None)
    results = model.predict(img)
    assert hasattr(results, "xyxy")

    save_path = tmp_path / "yolo.pt"
    model.save(save_path)
    model2 = YOLOv5Detector(model_name="yolov5n", pretrained=False)
    model2.load(save_path)
    preds2 = model2.predict(img)
    assert hasattr(preds2, "xyxy")
