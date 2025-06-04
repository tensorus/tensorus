"""Simple registry for Tensorus models."""

from typing import Dict, Type

from .base import TensorusModel


_MODEL_REGISTRY: Dict[str, Type[TensorusModel]] = {}


def register_model(name: str, cls: Type[TensorusModel]) -> None:
    """Register a model class under a given name."""
    if not issubclass(cls, TensorusModel):
        raise TypeError("cls must be a subclass of TensorusModel")
    _MODEL_REGISTRY[name] = cls


def get_model(name: str) -> Type[TensorusModel]:
    """Retrieve a registered model class by name."""
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' is not registered")
    return _MODEL_REGISTRY[name]
