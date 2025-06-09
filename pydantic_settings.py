"""Shim for pydantic-settings used during testing."""

import importlib.util
import site
import sys
from pathlib import Path
import types

_real_pkg = None
for p in site.getsitepackages():
    candidate = Path(p) / 'pydantic_settings' / '__init__.py'
    if candidate.is_file() and candidate.resolve() != Path(__file__).resolve():
        _real_pkg = candidate
        break

if _real_pkg:
    spec = importlib.util.spec_from_file_location(
        'pydantic_settings', str(_real_pkg),
        submodule_search_locations=[str(Path(_real_pkg).parent)]
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules['pydantic_settings'] = module
    spec.loader.exec_module(module)  # type: ignore
    BaseSettings = module.BaseSettings  # type: ignore
    SettingsConfigDict = getattr(module, 'SettingsConfigDict', dict)
else:
    from pydantic import BaseSettings

    class SettingsConfigDict(dict):
        pass

