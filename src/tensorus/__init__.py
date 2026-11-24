"""Tensorus — Official Python SDK, Xarray integration, and MCP bridge."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorus.client import Client as Client

__version__ = "0.2.0"
__all__ = ["Client", "__version__"]


def __getattr__(name: str) -> object:
    if name == "Client":
        from tensorus.client import Client

        return Client
    if name == "Tensorus":
        raise ImportError(
            "Embedded runtime is part of Tensorus Cloud; use `tensorus.Client(url, api_key)` "
            "to connect to a Tensorus deployment."
        )
    raise AttributeError(name)
