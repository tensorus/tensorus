"""Xarray labeled-array round-trip example (`pip install "tensorus[xarray]"`)."""

import os

import numpy as np
import xarray as xr

from tensorus import Client
from tensorus.xarray import prepare_xarray, read_xarray


def main() -> None:
    url = os.environ.get("TENSORUS_URL", "https://api.tensorus.com")
    api_key = os.environ["TENSORUS_API_KEY"]

    grid = xr.DataArray(
        np.linspace(12.0, 23.0, 12, dtype="float64").reshape(3, 4),
        dims=("lat", "lon"),
        coords={"lat": [-10.0, 0.0, 10.0], "lon": [120.0, 130.0, 140.0, 150.0]},
        attrs={"units": "degC"},
        name="sea_surface_temp",
    )

    with Client(url, api_key) as client:
        client.create_dataset("ocean_xarray_demo", description="Labeled coordinate grid example")
        prepared = prepare_xarray(grid)
        version = client.put(
            "ocean_xarray_demo",
            "sea_surface_temp",
            prepared.values,
            expected_version=0,
            metadata=prepared.metadata,
        )
        restored = read_xarray(client, version["id"])
        assert list(restored.data.dims) == ["lat", "lon"]
        assert np.allclose(restored.data.values, grid.values)
        print(f"Round-tripped Xarray {restored.data.dims} (version={version['id']})")


if __name__ == "__main__":
    main()
