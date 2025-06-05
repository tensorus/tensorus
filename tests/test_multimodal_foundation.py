import importlib.util
import sys
import types
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _get_model(monkeypatch):
    tensorus_pkg = types.ModuleType("tensorus")
    models_pkg = types.ModuleType("tensorus.models")
    monkeypatch.setitem(sys.modules, "tensorus", tensorus_pkg)
    monkeypatch.setitem(sys.modules, "tensorus.models", models_pkg)

    spec_ts = importlib.util.spec_from_file_location(
        "tensorus.tensor_storage", ROOT / "tensorus" / "tensor_storage.py"
    )
    ts_mod = importlib.util.module_from_spec(spec_ts)
    spec_ts.loader.exec_module(ts_mod)  # type: ignore
    monkeypatch.setitem(sys.modules, "tensorus.tensor_storage", ts_mod)

    spec_base = importlib.util.spec_from_file_location(
        "tensorus.models.base", ROOT / "tensorus" / "models" / "base.py"
    )
    base_mod = importlib.util.module_from_spec(spec_base)
    spec_base.loader.exec_module(base_mod)  # type: ignore
    monkeypatch.setitem(sys.modules, "tensorus.models.base", base_mod)

    spec_mm = importlib.util.spec_from_file_location(
        "tensorus.models.multimodal_foundation",
        ROOT / "tensorus" / "models" / "multimodal_foundation.py",
    )
    mm_mod = importlib.util.module_from_spec(spec_mm)
    spec_mm.loader.exec_module(mm_mod)  # type: ignore
    return mm_mod.MultimodalFoundationModel


def test_multimodal_foundation(monkeypatch, tmp_path):
    Model = _get_model(monkeypatch)
    img = Image.new("RGB", (224, 224))
    text = ["a blank image"]
    model = Model(model_name="openai/clip-vit-base-patch32")

    sim = model.predict([img], text)
    assert isinstance(sim, torch.Tensor)
    assert sim.shape == (1, 1)

    t_feat = model.encode_text(text)
    i_feat = model.encode_image([img])
    assert t_feat.shape == (1, 512)
    assert i_feat.shape == (1, 512)

    save_dir = tmp_path / "mmf"
    model.save(str(save_dir))
    model2 = Model(model_name="openai/clip-vit-base-patch32")
    model2.load(str(save_dir))
    sim2 = model2.predict([img], text)
    assert torch.allclose(sim, sim2)
