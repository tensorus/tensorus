"""End-to-end tests for the Tensorus Python SDK.

Run after building the native extension: ``maturin develop --features python``.
"""

import numpy as np
import pytest

from tensorus import Tensorus, __version__


def test_version_is_string():
    assert isinstance(__version__, str)
    assert __version__.count(".") >= 2


def test_insert_get_roundtrip():
    ts = Tensorus.memory()
    ts.create_dataset("weights")
    a = np.eye(3, dtype=np.float32)
    tid = ts.insert("weights", a, metadata={"name": "I3"})
    b = ts.get("weights", tid)
    assert b.shape == (3, 3)
    assert np.allclose(a, b)
    assert ts.list_datasets() == ["weights"]


def test_non_float_input_is_coerced():
    ts = Tensorus.memory()
    ts.create_dataset("d")
    a = np.arange(6, dtype=np.int64).reshape(2, 3)
    tid = ts.insert("d", a)
    b = ts.get("d", tid)
    assert b.shape == (2, 3)
    assert np.allclose(b, a.astype(np.float32))


def test_delete_then_missing():
    ts = Tensorus.memory()
    ts.create_dataset("d")
    tid = ts.insert("d", np.ones(4, dtype=np.float32))
    ts.delete("d", tid)
    with pytest.raises(Exception):
        ts.get("d", tid)


def test_local_persistence(tmp_path):
    data_dir = str(tmp_path / "store")
    ts = Tensorus.local(data_dir)
    ts.create_dataset("d")
    tid = ts.insert("d", np.array([1.0, 2.0, 3.0], dtype=np.float32))
    # Re-open the same directory.
    ts2 = Tensorus.local(data_dir)
    got = ts2.get("d", tid)
    assert np.allclose(got, [1.0, 2.0, 3.0])
