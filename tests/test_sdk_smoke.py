"""Standalone smoke tests for the public `tensorus` Python SDK & CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import tensorus
from tensorus import Client, sdk_cli
from tensorus.arrays import array_checksum
from tensorus.serialization import dump_npy, load_npy


def test_sdk_package_exports_and_npy_roundtrip(tmp_path: Path) -> None:
    assert tensorus.__version__ == "0.2.0"
    assert len([m for m in dir(Client) if not m.startswith("_")]) >= 380

    arr = np.asarray([[1.25, -3.5], [0.0, 8.75]], dtype="float32")
    raw = dump_npy(arr)
    npy_path = tmp_path / "sample.npy"
    npy_path.write_bytes(raw)

    assert sdk_cli.main(["--version"]) == 0
    assert sdk_cli.main(["inspect-npy", str(npy_path)]) == 0
    restored = load_npy(raw, max_bytes=len(raw))
    np.testing.assert_array_equal(restored, arr)
    assert len(array_checksum(restored)) == 64
