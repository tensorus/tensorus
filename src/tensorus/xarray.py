"""Bounded in-memory xarray projection with explicit observation validity."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

import numpy as np
from pydantic import ValidationError

try:
    import xarray as xr
except ModuleNotFoundError as exc:
    if exc.name == "xarray":
        raise ModuleNotFoundError(
            "xarray interoperability requires the optional tensorus[xarray] extra"
        ) from exc
    raise

import pandas as pd
from xarray.core.indexes import PandasIndex
from xarray.core.indexing import PandasIndexingAdapter

from tensorus.arrays import (
    Array,
    array_checksum,
    canonical_json,
    descriptor_nbytes,
    validated_array,
)
from tensorus.errors import Conflict, LimitExceeded, StorageError, TensorusError
from tensorus.responses import TensorVersion
from tensorus.semantics import (
    MAX_AXIS_COORDINATES,
    SEMANTICS_KEY,
    SemanticAxis,
    TensorSemantics,
    read_semantics,
    require_matching_axes,
)
from tensorus.validity import ValidityBinding, _descriptor_hash
from tensorus.validity_similarity import ValiditySelection

if TYPE_CHECKING:
    from tensorus.client import Client


DEFAULT_MAX_TENSOR_BYTES = 64 * 1024**2
REFERENCE_ATTRIBUTES = {
    "tensorus_time_reference": "time_reference",
    "tensorus_coordinate_system": "coordinate_system",
}
INPUT_LOSSES = (
    "DataArray names are not stored in tensor metadata; choose Tensorus names explicitly.",
    "Coordinate dtypes and xarray index types or index presence are not stored.",
)
READ_LOSSES = (
    "DataArray names are Tensorus names, not necessarily the original xarray names.",
    "Original coordinate dtypes and xarray index types or index presence are not stored.",
    "Other metadata, version provenance and authority are not DataArray attributes; "
    "inspect the returned descriptors. Quality evidence is not returned.",
)
V2_ATTRIBUTE_WARNING = (
    "xarray operations can discard or change v2 semantic attributes; inspect them before reuse."
)
PREPROCESSING_ATTRIBUTE_WARNING = (
    "xarray can retain, discard or change preprocessing attributes during local operations; "
    "review any new declaration explicitly."
)


@dataclass(frozen=True)
class PreparedXarray:
    values: Array
    metadata: dict[str, Any]
    source_name: str | None
    validity_values: Array | None
    validity_metadata: dict[str, Any] | None
    validity_name: str | None
    losses: tuple[str, ...] = INPUT_LOSSES


@dataclass(frozen=True)
class XarrayRead:
    data: xr.DataArray
    validity: xr.DataArray | None
    version: dict[str, Any]
    mask_version: dict[str, Any] | None
    binding: dict[str, Any] | None
    value_scope: Literal["raw_stored_values"] = "raw_stored_values"
    losses: tuple[str, ...] = READ_LOSSES


def _attributes(variable: xr.DataArray, allowed: set[str], context: str) -> None:
    extra = set(variable.attrs) - allowed
    if extra:
        names = ", ".join(sorted(repr(key) for key in extra))
        raise TensorusError(f"Unsupported {context} attributes: {names}")
    if variable.encoding:
        raise TensorusError(f"Unsupported {context} encoding; preserve it separately")
    if "units" in variable.attrs and not isinstance(variable.attrs["units"], str):
        raise TensorusError(f"{context} units must be an explicit string")


def _eager_data(variable: xr.DataArray, context: str, *, coordinate: bool = False) -> Array:
    # Accessing Variable.data can materialize lazy backends, so inspect its storage first.
    storage = variable.variable._data
    if coordinate and isinstance(storage, PandasIndexingAdapter):
        if type(storage.array) not in (pd.Index, pd.RangeIndex):
            raise TensorusError(f"{context} has an unsupported pandas index")
        if not isinstance(storage.array.dtype, np.dtype):
            raise TensorusError(f"{context} has an unsupported pandas coordinate dtype")
        storage = variable.data
    if not isinstance(storage, np.ndarray) or isinstance(storage, np.ma.MaskedArray):
        raise TensorusError(f"{context} must already be an eager NumPy array")
    return storage


def _coordinates(variable: xr.DataArray, context: str) -> list[int | float]:
    if variable.size > MAX_AXIS_COORDINATES:
        raise LimitExceeded(f"{context} exceeds {MAX_AXIS_COORDINATES} coordinates")
    values = _eager_data(variable, context, coordinate=True)
    if values.ndim != 1 or values.dtype.kind not in "iuf" or values.dtype.itemsize > 8:
        raise TensorusError(f"{context} must use one-dimensional integer or float coordinates")
    return values.tolist()


def _axes_identity(semantics: TensorSemantics) -> str:
    return canonical_json([axis.model_dump(mode="json") for axis in semantics.axes])


def _losses(semantics: TensorSemantics, base: tuple[str, ...]) -> tuple[str, ...]:
    result = base
    if semantics.version == 2 and V2_ATTRIBUTE_WARNING not in result:
        result = (*result, V2_ATTRIBUTE_WARNING)
    if semantics.preprocessing is not None and PREPROCESSING_ATTRIBUTE_WARNING not in result:
        result = (*result, PREPROCESSING_ATTRIBUTE_WARNING)
    return result


def _require_matching_index(
    source: xr.DataArray, dimension: str, coordinate_values: list[int | float]
) -> None:
    index = source.xindexes.get(dimension)
    if index is None:
        return
    if not isinstance(index, PandasIndex) or type(index) is not PandasIndex:
        raise TensorusError(f"Coordinate {dimension!r} has an unsupported xarray index")
    pandas_index = index.index
    if (
        type(pandas_index) not in (pd.Index, pd.RangeIndex)
        or not isinstance(pandas_index.dtype, np.dtype)
        or pandas_index.dtype.kind not in "iuf"
        or pandas_index.dtype.itemsize > 8
    ):
        raise TensorusError(f"Coordinate {dimension!r} has an unsupported pandas index")
    if len(pandas_index) != len(coordinate_values) or canonical_json(
        pandas_index.to_numpy().tolist()
    ) != canonical_json(coordinate_values):
        raise TensorusError(f"Coordinate {dimension!r} disagrees with its index")


def _prepare_one(
    source: xr.DataArray, max_tensor_bytes: int, context: str
) -> tuple[Array, dict[str, Any], TensorSemantics, str | None]:
    if not isinstance(source, xr.DataArray):
        raise TensorusError(f"{context} must be an xarray.DataArray")
    source_name = source.name
    if source_name is not None and not isinstance(source_name, str):
        raise TensorusError(f"{context} name must be a string or None")
    _attributes(
        source,
        {"units", "tensorus_model", "tensorus_preprocessing", "tensorus_semantics_version"},
        context,
    )
    if len(source.dims) > 16:
        raise LimitExceeded("Tensors may have at most 16 dimensions")
    dimensions: list[str] = []
    for dimension in source.dims:
        if not isinstance(dimension, str):
            raise TensorusError(f"{context} dimensions must have string names")
        dimensions.append(dimension)
    if len(set(dimensions)) != len(dimensions):
        raise TensorusError(f"{context} dimensions must be unique")
    extra = set(source.coords) - set(dimensions)
    if extra:
        raise TensorusError(f"{context} has unsupported auxiliary or scalar coordinates")
    if any(type(index) is not PandasIndex for index in source.xindexes.values()):
        raise TensorusError(f"{context} has an unsupported xarray index")

    values = validated_array(_eager_data(source, context), max_tensor_bytes)
    axes: list[dict[str, Any]] = []
    for dimension in dimensions:
        axis: dict[str, Any] = {"label": dimension}
        if dimension in source.coords:
            coordinate = source.coords[dimension]
            if coordinate.dims != (dimension,):
                raise TensorusError(f"Coordinate {dimension!r} must index its own dimension")
            _attributes(coordinate, {"units", *REFERENCE_ATTRIBUTES}, f"Coordinate {dimension!r}")
            coordinate_values = _coordinates(coordinate, f"Coordinate {dimension!r}")
            _require_matching_index(source, dimension, coordinate_values)
            axis["coordinates"] = coordinate_values
            if "units" in coordinate.attrs:
                axis["unit"] = coordinate.attrs["units"]
            for attribute, field in REFERENCE_ATTRIBUTES.items():
                if attribute in coordinate.attrs:
                    reference = coordinate.attrs[attribute]
                    if not isinstance(reference, dict):
                        raise TensorusError(
                            f"Coordinate {dimension!r} {attribute} must be an explicit mapping"
                        )
                    axis[field] = reference
        axes.append(axis)
    declaration: dict[str, Any] = {
        "version": source.attrs.get("tensorus_semantics_version", 1),
        "axes": axes,
    }
    if "units" in source.attrs:
        declaration["value_unit"] = source.attrs["units"]
    if "tensorus_model" in source.attrs:
        model = source.attrs["tensorus_model"]
        if not isinstance(model, dict):
            raise TensorusError(f"{context} tensorus_model must be an explicit mapping")
        declaration["model"] = model
    if "tensorus_preprocessing" in source.attrs:
        preprocessing = source.attrs["tensorus_preprocessing"]
        if not isinstance(preprocessing, dict):
            raise TensorusError(f"{context} tensorus_preprocessing must be an explicit mapping")
        declaration["preprocessing"] = preprocessing
    canonical_json(declaration)
    semantics = read_semantics({SEMANTICS_KEY: declaration}, values.shape)
    if semantics is None:
        raise TensorusError("Tensor semantics are required for xarray interoperability")
    return (
        values,
        {SEMANTICS_KEY: semantics.model_dump(mode="json", exclude_none=True)},
        semantics,
        source_name,
    )


def prepare_xarray(
    data: xr.DataArray,
    *,
    validity: xr.DataArray | None = None,
    max_tensor_bytes: int = DEFAULT_MAX_TENSOR_BYTES,
) -> PreparedXarray:
    """Prepare separate data and true-includes mask writes without performing I/O."""
    if type(max_tensor_bytes) is not int or max_tensor_bytes <= 0:
        raise ValueError("max_tensor_bytes must be a positive integer")
    values, metadata, data_semantics, data_name = _prepare_one(data, max_tensor_bytes, "Data")
    if validity is None:
        return PreparedXarray(
            values,
            metadata,
            data_name,
            None,
            None,
            None,
            losses=_losses(data_semantics, INPUT_LOSSES),
        )

    mask, mask_metadata, mask_semantics, mask_name = _prepare_one(
        validity, max_tensor_bytes, "Validity"
    )
    if mask.dtype != np.dtype("bool") or mask.shape != values.shape:
        raise TensorusError("Validity must be boolean and have the exact data shape")
    if _axes_identity(data_semantics) != _axes_identity(mask_semantics):
        raise TensorusError(
            "Data and validity dimensions, units, coordinates and references must match exactly"
        )
    require_matching_axes(data_semantics, mask_semantics, values.shape)
    return PreparedXarray(
        values,
        metadata,
        data_name,
        mask,
        mask_metadata,
        mask_name,
        losses=_losses(mask_semantics, _losses(data_semantics, INPUT_LOSSES)),
    )


def _coordinate_array(values: list[int | float], label: str) -> Array:
    if not values:
        return np.array([], dtype="float64")
    if all(type(value) is int for value in values):
        if min(values) >= -(2**63) and max(values) <= 2**63 - 1:
            dtype = "int64"
        elif min(values) >= 0 and max(values) <= 2**64 - 1:
            dtype = "uint64"
        else:
            raise TensorusError(f"Coordinate {label!r} is outside supported integer ranges")
    elif all(type(value) is float for value in values):
        dtype = "float64"
    else:
        raise TensorusError(f"Coordinate {label!r} mixes integer and floating-point values")
    result = np.array(values, dtype=dtype)
    if canonical_json(result.tolist()) != canonical_json(values):
        raise TensorusError(f"Coordinate {label!r} cannot be represented exactly")
    return result


def _coordinate_attributes(axis: SemanticAxis) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    if axis.unit is not None:
        attributes["units"] = axis.unit
    if axis.time_reference is not None:
        attributes["tensorus_time_reference"] = axis.time_reference.model_dump(mode="json")
    if axis.coordinate_system is not None:
        attributes["tensorus_coordinate_system"] = axis.coordinate_system.model_dump(mode="json")
    return attributes


def _data_attributes(semantics: TensorSemantics) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    if semantics.version == 2:
        attributes["tensorus_semantics_version"] = 2
    if semantics.value_unit is not None:
        attributes["units"] = semantics.value_unit
    if semantics.model is not None:
        attributes["tensorus_model"] = semantics.model.model_dump(mode="json")
    if semantics.preprocessing is not None:
        attributes["tensorus_preprocessing"] = semantics.preprocessing.model_dump(mode="json")
    return attributes


def _data_array(values: Array, semantics: TensorSemantics, name: str) -> xr.DataArray:
    coordinates: dict[str, xr.DataArray] = {}
    for axis in semantics.axes:
        if axis.coordinates is None:
            if axis.unit is not None:
                raise TensorusError(
                    f"Axis {axis.label!r} has a unit but no xarray coordinate to carry it"
                )
            continue
        coordinates[axis.label] = xr.DataArray(
            _coordinate_array(axis.coordinates, axis.label),
            dims=(axis.label,),
            attrs=_coordinate_attributes(axis),
        )
    result = xr.DataArray(
        values,
        dims=[axis.label for axis in semantics.axes],
        coords=coordinates,
        name=name,
        attrs=_data_attributes(semantics),
    )
    if canonical_json(result.attrs) != canonical_json(_data_attributes(semantics)):
        raise TensorusError("xarray changed tensor semantic attributes")
    for axis in semantics.axes:
        if axis.coordinates is not None:
            if canonical_json(result.coords[axis.label].attrs) != canonical_json(
                _coordinate_attributes(axis)
            ):
                raise TensorusError(f"xarray changed coordinate {axis.label!r} attributes")
            coordinate_values = _coordinates(
                result.coords[axis.label], f"Coordinate {axis.label!r}"
            )
            if canonical_json(coordinate_values) != canonical_json(axis.coordinates):
                raise TensorusError(f"xarray changed coordinate {axis.label!r}")
            _require_matching_index(result, axis.label, coordinate_values)
    return result


def _version(client: "Client", version_id: UUID) -> tuple[dict[str, Any], TensorSemantics]:
    record = client.version(str(version_id))
    try:
        version = TensorVersion.model_validate(record)
    except ValidationError as exc:
        raise StorageError("Tensor version response is invalid") from exc
    if version.id != version_id:
        raise StorageError("Tensor version response does not match the requested version")
    if descriptor_nbytes(record) > client.max_tensor_bytes:
        raise LimitExceeded("Tensor exceeds the configured xarray read byte limit")
    record = version.model_dump(mode="json")
    semantics = read_semantics(record["metadata"], tuple(record["shape"]), record["operation"])
    if semantics is None:
        raise TensorusError("xarray reads require versioned tensor semantics")
    return record, semantics


def _verified_values(client: "Client", record: dict[str, Any]) -> Array:
    values = client._get_verified_npy(record)
    if (
        list(values.shape) != record["shape"]
        or values.dtype.name != record["dtype"]
        or array_checksum(values) != record["checksum"]
    ):
        raise StorageError("Downloaded tensor does not match its version descriptor")
    return values


def _binding(client: "Client", binding_id: UUID) -> ValidityBinding:
    try:
        return ValidityBinding.model_validate(client.validity_binding(str(binding_id)))
    except ValidationError as exc:
        raise StorageError("Validity binding response is invalid") from exc


def read_xarray(
    client: "Client",
    version_id: str | UUID,
    *,
    validity: ValiditySelection | None = None,
) -> XarrayRead:
    """Read verified raw values and, only when pinned, a separate true-includes mask."""
    if type(client.max_tensor_bytes) is not int or client.max_tensor_bytes <= 0:
        raise ValueError("client.max_tensor_bytes must be a positive integer")
    try:
        requested = UUID(str(version_id))
    except ValueError as exc:
        raise TensorusError("version_id must be a UUID") from exc
    if validity is not None and not isinstance(validity, ValiditySelection):
        raise TensorusError("Validity must be an explicit ValiditySelection")
    data_record, data_semantics = _version(client, requested)
    if validity is None:
        values = _verified_values(client, data_record)
        return XarrayRead(
            _data_array(values, data_semantics, data_record["name"]),
            None,
            data_record,
            None,
            None,
            losses=_losses(data_semantics, READ_LOSSES),
        )

    binding = _binding(client, validity.binding_id)
    if binding.id != validity.binding_id or binding.descriptor_hash != validity.descriptor_hash:
        raise Conflict("Validity binding does not match the pinned selection")
    mask_record, mask_semantics = _version(client, binding.mask_version_id)
    if (
        binding.data_version_id != requested
        or binding.data_checksum != data_record["checksum"]
        or binding.data_dtype != data_record["dtype"]
        or binding.mask_checksum != mask_record["checksum"]
        or binding.tenant_id != UUID(data_record["tenant_id"])
        or binding.tenant_id != UUID(mask_record["tenant_id"])
        or binding.shape != data_record["shape"]
        or binding.shape != mask_record["shape"]
        or mask_record["dtype"] != "bool"
        or _descriptor_hash(
            data_record,
            mask_record,
            binding.origin.model_dump(mode="json"),
            str(binding.id),
            binding.created_at,
            binding.created_by,
        )
        != binding.descriptor_hash
    ):
        raise StorageError("Validity binding does not match its version descriptors")
    if _axes_identity(data_semantics) != _axes_identity(mask_semantics):
        raise TensorusError(
            "Data and validity dimensions, units, coordinates and references must match exactly"
        )
    require_matching_axes(data_semantics, mask_semantics, tuple(binding.shape))
    values = _verified_values(client, data_record)
    mask = _verified_values(client, mask_record)
    data_array = _data_array(values, data_semantics, data_record["name"])
    mask_array = _data_array(mask, mask_semantics, mask_record["name"])
    if _binding(client, binding.id) != binding:
        raise Conflict("Validity binding changed during the xarray read")
    return XarrayRead(
        data_array,
        mask_array,
        data_record,
        mask_record,
        binding.model_dump(mode="json"),
        losses=_losses(mask_semantics, _losses(data_semantics, READ_LOSSES)),
    )
