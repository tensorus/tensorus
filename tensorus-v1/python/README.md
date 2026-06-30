# Tensorus Python SDK

Python bindings for the Tensorus v1.0 tensor-native database, built with PyO3.

## Install

```bash
pip install maturin
maturin develop --features python   # from this directory, builds tensorus._native
```

> **Build note:** building the native extension requires a Rust toolchain whose
> target ABI matches your CPython. On Windows, CPython is built with MSVC, so use
> the `x86_64-pc-windows-msvc` Rust toolchain (with the Visual Studio Build
> Tools) to build the wheel. The rest of the workspace builds with any toolchain.

## Usage

```python
import numpy as np
from tensorus import Tensorus

ts = Tensorus.memory()
ts.create_dataset("weights")
tid = ts.insert("weights", np.eye(3, dtype=np.float32), metadata={"name": "I3"})
restored = ts.get("weights", tid)
assert np.allclose(restored, np.eye(3))
```
