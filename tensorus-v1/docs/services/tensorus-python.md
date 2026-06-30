# Service: `tensorus-python`

**Python SDK.** PyO3 bindings exposing the Rust engine to Python, plus a thin,
ergonomic high-level package.

- **Rust crate:** `tensorus-python` (lib name `_native`), behind a non-default
  `python` feature so `cargo build --workspace` stays green without a CPython
  link target.
- **Depends on (with `python` feature):** `pyo3` 0.22, `numpy` 0.22,
  `tensorus-core`, `tensorus-storage`, `tensorus-compute`, `tokio`.
- **Python package:** `python/tensorus/` (mixed maturin layout).

---

## Layout

```
python/
  pyproject.toml            # maturin build config (module-name = tensorus._native)
  tensorus/__init__.py      # high-level `Tensorus` wrapper (numpy convenience)
  tensorus/_native.pyi      # type stubs for the native module
  tests/test_sdk.py         # pytest end-to-end SDK tests
```

The native extension module is `tensorus._native`; the pure-Python `tensorus`
package wraps it.

---

## Native module (`tensorus._native`)

A `Tensorus` class backed by the Rust `FileStorage` + descriptor compute, with a
small Tokio runtime bridging async storage into synchronous Python calls.

```python
class Tensorus:
    @staticmethod
    def memory() -> "Tensorus": ...                 # ephemeral temp dir
    @staticmethod
    def local(data_dir: str | None = None) -> "Tensorus": ...   # default ./data
    @staticmethod
    def connect(uri: str) -> "Tensorus": ...         # local path / file:// (grpc:// rejected)
    @staticmethod
    def version() -> str: ...

    def create_dataset(self, name: str) -> None: ...
    def list_datasets(self) -> list[str]: ...
    def insert(self, dataset: str, array: np.ndarray, metadata: str | None = None) -> str: ...
    def get(self, dataset: str, id: str) -> np.ndarray: ...
    def delete(self, dataset: str, id: str) -> None: ...

__version__: str
```

- `insert` reads a numpy `float32` array (logical C-order), computes its
  descriptor, stores it, and returns the tensor id. `metadata` is a JSON string.
- `get` returns a numpy `float32` array reshaped to the stored shape.

---

## High-level SDK (`tensorus`)

A friendlier wrapper that handles numpy/JSON conversion.

```python
import numpy as np
from tensorus import Tensorus, __version__

ts = Tensorus.memory()
ts.create_dataset("weights")
tid = ts.insert("weights", np.eye(3, dtype=np.float32), metadata={"name": "I3"})
restored = ts.get("weights", tid)
assert np.allclose(restored, np.eye(3))
ts.delete("weights", tid)
```

| Method | Notes |
|--------|-------|
| `Tensorus.memory() / .local(dir) / .connect(uri)` | construction |
| `create_dataset(name)` / `list_datasets()` | datasets |
| `insert(dataset, ndarray, metadata=dict)` | coerces to `float32`; `json.dumps` the metadata dict |
| `get(dataset, id) -> ndarray` | returns the numpy array |
| `delete(dataset, id)` | remove |

---

## Building the extension

```bash
cd python
pip install maturin
maturin develop --features python      # build + install into the active venv
pytest -q                              # run the SDK tests
```

> **Toolchain note (important).** Building the native extension requires a Rust
> target whose ABI matches your CPython. On **Windows**, CPython is built with
> **MSVC**, so build the wheel with the `x86_64-pc-windows-msvc` toolchain (plus
> the Visual Studio Build Tools). The rest of the workspace builds with any
> toolchain (this repo's default is `x86_64-pc-windows-gnu`).
>
> In this development environment the wheel was **not** built (gnu toolchain vs
> MSVC-built CPython); the binding code is compile-validated via
> `cargo check -p tensorus-python --features python`, and CI builds + tests the
> wheel on Linux where the toolchain matches.

## CI

The `python` job in `.github/workflows/ci.yml` runs `maturin build`, installs the
wheel, runs `pytest`, and smoke-tests `import tensorus`.
