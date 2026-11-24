"""Minimal remote Tensorus SDK quickstart example."""

import os

import numpy as np

from tensorus import Client


def main() -> None:
    url = os.environ.get("TENSORUS_URL", "http://127.0.0.1:8000")
    api_key = os.environ["TENSORUS_API_KEY"]

    with Client(url, api_key) as client:
        client.create_dataset("demo_sensors", description="Example sensor array dataset")
        arr = np.linspace(0.0, 1.0, 16, dtype="float32").reshape(4, 4)
        version = client.put("demo_sensors", "calibration_matrix", arr, expected_version=0)
        loaded = client.get(version["id"])
        assert np.array_equal(loaded, arr)
        print(f"Stored and verified version {version['id']} (sha256={version['checksum']})")


if __name__ == "__main__":
    main()
