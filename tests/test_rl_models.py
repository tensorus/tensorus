import importlib.util
import types
import sys
from pathlib import Path

# Dynamically load module to avoid heavy package imports via tensorus.models
pkg = types.ModuleType("tensorus")
models_pkg = types.ModuleType("tensorus.models")
pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "tensorus")]
sys.modules.setdefault("tensorus", pkg)
sys.modules.setdefault("tensorus.models", models_pkg)

# Load required base module for rl_models
base_spec = importlib.util.spec_from_file_location(
    "tensorus.models.base",
    Path(__file__).resolve().parents[1] / "tensorus" / "models" / "base.py",
)
base_mod = importlib.util.module_from_spec(base_spec)
base_spec.loader.exec_module(base_mod)  # type: ignore
sys.modules["tensorus.models.base"] = base_mod

spec = importlib.util.spec_from_file_location(
    "tensorus.models.rl_models",
    Path(__file__).resolve().parents[1] / "tensorus" / "models" / "rl_models.py",
)
rl_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl_mod)  # type: ignore

QLearningModel = rl_mod.QLearningModel
DQNModel = rl_mod.DQNModel
A2CModel = rl_mod.A2CModel
PPOModel = rl_mod.PPOModel
TRPOModel = rl_mod.TRPOModel
SACModel = rl_mod.SACModel


class SimpleEnv:
    """Tiny deterministic environment with discrete states."""

    def __init__(self):
        self.state = 0

    @property
    def action_dim(self):
        return 2

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        # action 1 moves right toward goal, 0 moves left but clipped at 0
        if action == 1 and self.state < 1:
            self.state += 1
        elif action == 0 and self.state > 0:
            self.state -= 1
        done = self.state >= 1
        reward = 1.0 if done else -0.1
        return self.state, reward, done, {}


def _run_model(model):
    env = SimpleEnv()
    model.fit(env, episodes=2)
    action = model.predict(0)
    assert isinstance(action, int)


def test_qlearning_model():
    model = QLearningModel(n_states=5, n_actions=2, alpha=0.5)
    _run_model(model)


def test_dqn_model():
    model = DQNModel(state_dim=1, action_dim=2, hidden_size=8, batch_size=1, buffer_size=10)
    _run_model(model)


def test_a2c_model():
    model = A2CModel(state_dim=1, action_dim=2, hidden_size=8)
    _run_model(model)


def test_ppo_model():
    model = PPOModel(state_dim=1, action_dim=2, hidden_size=8)
    _run_model(model)


def test_trpo_model():
    model = TRPOModel(state_dim=1, action_dim=2, hidden_size=8)
    _run_model(model)


def test_sac_model():
    model = SACModel(state_dim=1, action_dim=2, hidden_size=8, batch_size=1, buffer_size=10)
    _run_model(model)
