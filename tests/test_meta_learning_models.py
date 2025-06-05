import importlib.util
import types
import sys
from pathlib import Path
import numpy as np
import torch

# Create lightweight package structure to avoid heavy dependencies
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

base_path = Path(__file__).resolve().parents[1] / "tensorus" / "models"

for mod_name in [
    "base",
    "linear_regression",
    "logistic_regression",
    "neuro_symbolic_model",
    "physics_informed_nn",
    "stacked_generalization",
    "mixture_of_experts",
]:
    spec = importlib.util.spec_from_file_location(
        f"tensorus.models.{mod_name}", base_path / f"{mod_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    sys.modules[f"tensorus.models.{mod_name}"] = module

from tensorus.models.linear_regression import LinearRegressionModel
from tensorus.models.logistic_regression import LogisticRegressionModel
from tensorus.models.neuro_symbolic_model import NeuroSymbolicModel
from tensorus.models.physics_informed_nn import PhysicsInformedNNModel
from tensorus.models.stacked_generalization import StackedGeneralizationModel
from tensorus.models.mixture_of_experts import MixtureOfExpertsModel


def test_stacked_generalization_model():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])
    base1 = LinearRegressionModel()
    base2 = LinearRegressionModel()
    meta = LinearRegressionModel()
    model = StackedGeneralizationModel([base1, base2], meta)
    model.fit(X, y)
    preds = model.predict(X)
    assert torch.allclose(preds, torch.tensor(y, dtype=preds.dtype), atol=1e-4)


def test_mixture_of_experts_model():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])
    m1 = LinearRegressionModel()
    m2 = LinearRegressionModel()
    model = MixtureOfExpertsModel([m1, m2])
    model.fit(X, y)
    preds = model.predict(X)
    assert torch.allclose(preds, torch.tensor(y, dtype=preds.dtype), atol=1e-4)


def test_neuro_symbolic_model():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    base = LogisticRegressionModel(lr=0.5, epochs=1000)

    def rule(X_in, preds_in):
        x_np = np.asarray(X_in)
        override = (x_np[:, 0] >= 1.5).astype(float)
        if isinstance(preds_in, torch.Tensor):
            return torch.tensor(override, dtype=preds_in.dtype)
        return override

    model = NeuroSymbolicModel(base, symbolic_fn=rule)
    model.fit(X, y)
    preds = model.predict(X)
    assert torch.equal(preds, torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_physics_informed_nn_model():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = 2 * X[:, 0]

    def phys_loss(xb, pred):
        grad = torch.autograd.grad(pred.sum(), xb, create_graph=True)[0]
        return ((grad.squeeze() - 2.0) ** 2).mean()

    model = PhysicsInformedNNModel(input_size=1, epochs=200, physics_loss_fn=phys_loss)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
