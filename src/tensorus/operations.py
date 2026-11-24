import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from types import EllipsisType
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self
from uuid import UUID

import numpy as np
from pydantic import (
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - standalone public SDK without threadpoolctl

    def threadpool_limits(*args: Any, **kwargs: Any) -> Any:
        return nullcontext()


from tensorus.arrays import (
    CHUNKED_DISPERSION_ITEMSIZE,
    CHUNKED_DISPERSION_PASSES,
    CHUNKED_DISPERSION_WORK_ARRAYS,
    CHUNKED_MAX_BLOCKS,
    Array,
    accumulate_chunked_any,
    accumulate_chunked_extrema,
    accumulate_chunked_included_count,
    accumulate_chunked_sum,
    chunked_block_layout,
    chunked_c_order_selections,
    chunked_dispersion,
    extrema_identity,
    mean_from_chunked_sums,
    reduced_shape,
    require_float64_integer_precision,
    validated_array,
)

if TYPE_CHECKING:
    from tensorus.config import Settings
from tensorus.errors import LimitExceeded, TensorusError
from tensorus.models import (
    MASKED_BINARY_OPERATIONS,
    DType,
    Model,
    Name,
    Operation,
    ReindexFill,
    SliceRequest,
    Step,
    _check_coordinate_float,
)
from tensorus.operation_contracts import OPERATION_CONTRACTS, operation_input_count


class EmptyParameters(Model):
    pass


def is_ambiguous_arithmetic_scalar(value: object) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**53) or (
        isinstance(value, float) and abs(value) >= 2**53
    )


class ArithmeticParameters(Model):
    scalar: Annotated[float, Field(strict=True)] | None = None

    @field_validator("scalar", mode="before")
    @classmethod
    def reject_ambiguous_scalar(cls, value: object) -> object:
        # Integers are converted to binary64 by Pydantic before calculation.
        # Reject values that could already have rounded during JSON decoding.
        if is_ambiguous_arithmetic_scalar(value):
            raise ValueError("Scalar cannot be represented unambiguously as float64")
        return value


class SelectedScalarParameters(ArithmeticParameters):
    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "required": ["operator"],
                        "properties": {"operator": {"const": "divide"}},
                    },
                    "then": {"properties": {"scalar": {"not": {"const": 0}}}},
                }
            ]
        }
    )
    scalar: Annotated[float, Field(strict=True, gt=-(2**53), lt=2**53)]
    operator: Literal["add", "subtract", "multiply", "divide"]
    algorithm_version: Annotated[StrictInt, Field(ge=1, le=1)]

    @model_validator(mode="after")
    def nonzero_divisor(self) -> Self:
        if self.operator == "divide" and self.scalar == 0:
            raise ValueError("Selected scalar division requires a nonzero divisor")
        return self


BOUND_SELECTED_PREPROCESSING = "bound_selected_v1"


class MaskedBinaryAlgorithmParameters(Model):
    algorithm_version: Annotated[StrictInt, Field(ge=1, le=1)]


class MaskedAddParameters(MaskedBinaryAlgorithmParameters):
    selected_preprocessing_policy: Literal["bound_selected_v1"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "Plan-step evidence for matching source preprocessing only at positions selected "
            "by a reviewed v2 filled paired-validity graph. For guided review, request this "
            "as the top-level selected_preprocessing_policy instead."
        ),
    )


class MaskedArithmeticParameters(MaskedAddParameters):
    pass


class NormalizeParameters(Model):
    epsilon: Annotated[float, Field(gt=0, le=1, strict=True)] = 1e-12


class MaskedScalingParameters(NormalizeParameters):
    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "required": ["algorithm_version"],
                        "properties": {"algorithm_version": {"const": 1}},
                    },
                    "then": {"not": {"required": ["axis"]}},
                }
            ]
        }
    )
    algorithm_version: Annotated[StrictInt, Field(ge=1, le=2)]
    axis: Annotated[list[StrictInt], Field(max_length=16)] | StrictInt | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Version 2 reduces these axes separately for each remaining-axis position.",
    )

    @model_validator(mode="after")
    def axis_requires_version_two(self) -> Self:
        if self.algorithm_version == 1 and "axis" in self.model_fields_set:
            raise ValueError("Selected scaling axes require algorithm_version 2")
        return self


class FittedScalingParameters(NormalizeParameters):
    algorithm: Literal["l2_normalize", "standardize"]
    algorithm_version: Annotated[StrictInt, Field(ge=1, le=2)]
    axis: Annotated[list[StrictInt], Field(max_length=16)] | StrictInt | None = None


class ImportedFittedStateSelection(Model):
    import_id: UUID
    source_hash: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
    state_role: Name
    local_release_id: UUID
    local_release_report_hash: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]


class ApplyFittedScalingParameters(Model):
    algorithm: Literal["l2_normalize", "standardize"]
    algorithm_version: Annotated[StrictInt, Field(ge=1, le=2)]
    axis: Annotated[list[StrictInt], Field(max_length=16)] | StrictInt | None = None
    state_checksum: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
    imported_state: ImportedFittedStateSelection | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class TransposeParameters(Model):
    axes: list[StrictInt] | None = Field(default=None, max_length=16)


class ReshapeParameters(Model):
    shape: list[Annotated[StrictInt, Field(ge=-1)]] = Field(max_length=16)


MAX_SELECTION_INDICES = 16_384
MAX_NEAREST_MAPPING_POSITIONS = 32_768
NEAREST_INTEGER_FORMAT = "tensorus-nearest-integer-coordinate-v1"


class IndexSelectionParameters(Model):
    indices: list[
        Annotated[list[Annotated[StrictInt, Field(ge=0)]], Field(max_length=8192)] | None
    ] = Field(
        max_length=16,
        description=(
            "One entry per axis: null keeps the full axis; otherwise use strictly increasing, "
            "in-range positions. At most 16384 explicit indices are allowed in total."
        ),
    )

    @model_validator(mode="after")
    def bounded_increasing_indices(self) -> Self:
        selected = [axis for axis in self.indices if axis is not None]
        if any(len(axis) > 8192 for axis in selected) or sum(map(len, selected)) > (
            MAX_SELECTION_INDICES
        ):
            raise ValueError("Index selection exceeds the per-axis or total index limit")
        if any(left >= right for axis in selected for left, right in pairwise(axis)):
            raise ValueError("Indices on each axis must be strictly increasing")
        return self


def _validate_reindex_axis(
    coordinates: Sequence[int | float], indices: Sequence[int | None], *, repeated: bool
) -> None:
    if len(coordinates) != len(indices):
        raise ValueError("Reindex coordinates and indices must have equal lengths")
    for coordinate in coordinates:
        _check_coordinate_float(coordinate)
        if isinstance(coordinate, float) and not math.isfinite(coordinate):
            raise ValueError("Reindex coordinates must be finite")
    if any(left >= right for left, right in pairwise(coordinates)):
        raise ValueError("Reindex coordinates must be strictly increasing")
    selected = [index for index in indices if index is not None]
    if any(left > right if repeated else left >= right for left, right in pairwise(selected)):
        raise ValueError(
            "Reindex source positions must be nondecreasing"
            if repeated
            else "Reindex source positions must be strictly increasing"
        )


class ReindexAxis(Model):
    coordinates: list[StrictInt | StrictFloat] = Field(max_length=8192)
    indices: list[Annotated[StrictInt, Field(ge=0)] | None] = Field(max_length=8192)

    @model_validator(mode="after")
    def validate_axis(self) -> Self:
        _validate_reindex_axis(self.coordinates, self.indices, repeated=False)
        return self


class NearestReindexAxis(Model):
    coordinates: list[Annotated[StrictInt, Field(ge=-(2**63), le=2**63 - 1)]] = Field(
        max_length=8192
    )
    indices: list[Annotated[StrictInt, Field(ge=0)] | None] = Field(max_length=8192)

    @model_validator(mode="after")
    def validate_axis(self) -> Self:
        _validate_reindex_axis(self.coordinates, self.indices, repeated=True)
        return self


class NearestIntegerPolicy(Model):
    algorithm: Literal["nearest_integer_v1"]
    max_distance: Annotated[StrictInt, Field(ge=0, le=2**53 - 1)]


def _validate_reindex_limits(axes: Sequence[ReindexAxis | NearestReindexAxis | None]) -> None:
    if sum(len(axis.indices) for axis in axes if axis is not None) > MAX_SELECTION_INDICES:
        raise ValueError("Reindex coordinates exceed the total position limit")


class CoordinateMappingParameters(Model):
    axes: list[ReindexAxis | None] = Field(max_length=16)

    @model_validator(mode="after")
    def bounded_axes(self) -> Self:
        _validate_reindex_limits(self.axes)
        return self


PRESERVE_NO_FILL_PREPROCESSING = "preserve_no_fill_v1"


def reindex_without_fill(axes: Sequence[ReindexAxis | NearestReindexAxis | None]) -> bool:
    return all(axis is None or all(index is not None for index in axis.indices) for axis in axes)


class PreprocessingReindexParameters(Model):
    axes: list[ReindexAxis | NearestReindexAxis | None] = Field(max_length=16)
    nearest: NearestIntegerPolicy | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    preprocessing_policy: Literal["preserve_no_fill_v1"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "Optionally preserve the source's declared preprocessing identity when every "
            "target position maps to the source. Gaps are rejected. Omission clears it."
        ),
    )

    @model_validator(mode="after")
    def preserve_only_without_fill(self) -> Self:
        _validate_reindex_limits(self.axes)
        if self.nearest is None:
            for axis in self.axes:
                if axis is not None:
                    _validate_reindex_axis(axis.coordinates, axis.indices, repeated=False)
        elif self.preprocessing_policy is not None:
            raise ValueError("Nearest-coordinate mapping cannot preserve preprocessing identity")
        else:
            for axis in self.axes:
                if axis is not None and any(
                    type(value) is not int or not -(2**63) <= value <= 2**63 - 1
                    for value in axis.coordinates
                ):
                    raise ValueError("Nearest-coordinate targets must be signed 64-bit integers")
        if self.preprocessing_policy is not None and not reindex_without_fill(self.axes):
            raise ValueError("Preprocessing preservation requires every target position to map")
        return self


class ReindexCoordinatesParameters(PreprocessingReindexParameters):
    fill_value: ReindexFill


def reindex_fill(dtype: str, value: int | bool) -> Any:
    target = np.dtype(dtype)
    if np.issubdtype(target, np.bool_):
        if type(value) is not bool:
            raise TensorusError(
                "Coordinate reindexing requires a non-boolean numeric dtype for integer fills; "
                "boolean data requires an explicit boolean fill"
            )
        return np.bool_(value)
    if type(value) is bool:
        raise TensorusError("A boolean fill requires boolean input data")
    if np.issubdtype(target, np.integer):
        limits = np.iinfo(target)
        if not limits.min <= value <= limits.max:
            raise TensorusError("Fill value is outside the input dtype range")
        return target.type(value)
    try:
        with np.errstate(over="raise", invalid="raise"):
            converted = target.type(value)
    except (FloatingPointError, OverflowError) as exc:
        raise TensorusError("Fill value is outside the input dtype range") from exc
    if not np.isfinite(converted) or int(converted) != value:
        raise TensorusError("Fill value must be exactly representable in the input dtype")
    return converted


class CastParameters(Model):
    dtype: DType


class SelectedCastParameters(Model):
    dtype: Annotated[DType, Field(json_schema_extra={"not": {"const": "bool"}})]
    algorithm_version: Annotated[StrictInt, Field(ge=1, le=1)]

    @field_validator("dtype")
    @classmethod
    def numeric_target(cls, value: DType) -> DType:
        if value == DType.bool:
            raise ValueError("Selected casting requires a non-boolean numeric target")
        return value


ValueUnit = Literal["m", "km", "cm", "mm", "um", "nm", "s", "ms", "us", "ns", "g", "kg", "mg", "ug"]
VALUE_UNITS_V1: dict[str, tuple[str, int]] = {
    "m": ("length", 0),
    "km": ("length", 3),
    "cm": ("length", -2),
    "mm": ("length", -3),
    "um": ("length", -6),
    "nm": ("length", -9),
    "s": ("time", 0),
    "ms": ("time", -3),
    "us": ("time", -6),
    "ns": ("time", -9),
    "g": ("mass", 0),
    "kg": ("mass", 3),
    "mg": ("mass", -3),
    "ug": ("mass", -6),
}


class ConvertValueUnitParameters(Model):
    conversion_version: Annotated[StrictInt, Field(ge=1, le=1)]
    source_unit: ValueUnit
    target_unit: ValueUnit

    @model_validator(mode="after")
    def compatible_units(self) -> Self:
        if self.source_unit == self.target_unit:
            raise ValueError("Source and target units must differ")
        if VALUE_UNITS_V1[self.source_unit][0] != VALUE_UNITS_V1[self.target_unit][0]:
            raise ValueError("Source and target units must have the same dimension")
        return self


class ConvertAxisUnitParameters(ConvertValueUnitParameters):
    axis: Name


TemperatureUnit = Literal["K", "Cel", "[degF]"]
MAX_TEMPERATURE_ELEMENTS = 100_000
TEMPERATURE_UNITS_V1: dict[str, tuple[Fraction, Fraction]] = {
    "K": (Fraction(1), Fraction(0)),
    "Cel": (Fraction(1), Fraction(27315, 100)),
    "[degF]": (Fraction(5, 9), Fraction(45967, 180)),
}


class ConvertTemperatureUnitParameters(Model):
    conversion_version: Annotated[StrictInt, Field(ge=1, le=1)]
    source_unit: TemperatureUnit
    target_unit: TemperatureUnit

    @model_validator(mode="after")
    def different_units(self) -> Self:
        if self.source_unit == self.target_unit:
            raise ValueError("Source and target units must differ")
        return self


def value_unit_factor(params: dict[str, Any]) -> float:
    source = VALUE_UNITS_V1[params["source_unit"]]
    target = VALUE_UNITS_V1[params["target_unit"]]
    return float(f"1e{source[1] - target[1]}")


class ReduceParameters(Model):
    axis: list[StrictInt] | StrictInt | None = None
    keepdims: bool = Field(default=False, strict=True)


class ChunkedCountParameters(Model):
    algorithm_version: Annotated[StrictInt, Field(ge=1, le=1)]


class AxisChunkedCountParameters(ReduceParameters):
    algorithm_version: Annotated[StrictInt, Field(ge=1, le=1)]


class AxisChunkedSumParameters(ReduceParameters):
    algorithm_version: Annotated[StrictInt, Field(ge=1, le=1)]


class AxisChunkedMeanParameters(ReduceParameters):
    algorithm_version: Annotated[StrictInt, Field(ge=1, le=1)]


class AxisChunkedDispersionParameters(ReduceParameters):
    algorithm_version: Annotated[StrictInt, Field(ge=1, le=1)]


class AxisChunkedExtremaParameters(ReduceParameters):
    algorithm_version: Annotated[StrictInt, Field(ge=1, le=1)]


class ValidityReductionParameters(ReduceParameters):
    algorithm_version: Annotated[StrictInt, Field(ge=1, le=1)]


class SelectedMeanParameters(ReduceParameters):
    algorithm_version: Annotated[StrictInt, Field(ge=1, le=2)]


class ValidityAnyParameters(ReduceParameters):
    algorithm_version: Annotated[StrictInt, Field(ge=1, le=2)]


PARAMETERS: dict[Operation, type[Model]] = {
    Operation.profile: EmptyParameters,
    Operation.normalize: NormalizeParameters,
    Operation.standardize: NormalizeParameters,
    Operation.masked_normalize_with_placeholders: MaskedScalingParameters,
    Operation.masked_standardize_with_placeholders: MaskedScalingParameters,
    Operation.fit_selected_preprocessing: FittedScalingParameters,
    Operation.apply_fitted_preprocessing: ApplyFittedScalingParameters,
    Operation.add: ArithmeticParameters,
    Operation.masked_add_with_placeholders: MaskedAddParameters,
    Operation.subtract: ArithmeticParameters,
    Operation.masked_subtract_with_placeholders: MaskedArithmeticParameters,
    Operation.multiply: ArithmeticParameters,
    Operation.masked_multiply_with_placeholders: MaskedArithmeticParameters,
    Operation.divide: ArithmeticParameters,
    Operation.masked_divide_with_placeholders: MaskedArithmeticParameters,
    Operation.masked_scalar_arithmetic_with_placeholders: SelectedScalarParameters,
    Operation.matmul: EmptyParameters,
    Operation.batched_matmul: EmptyParameters,
    Operation.transpose: TransposeParameters,
    Operation.slice: SliceRequest,
    Operation.chunked_slice: SliceRequest,
    Operation.select_indices: IndexSelectionParameters,
    Operation.reindex_coordinates: ReindexCoordinatesParameters,
    Operation.reindex_validity: PreprocessingReindexParameters,
    Operation.coordinate_coverage: CoordinateMappingParameters,
    Operation.logical_and: EmptyParameters,
    Operation.reshape: ReshapeParameters,
    Operation.cast: CastParameters,
    Operation.masked_cast_with_placeholders: SelectedCastParameters,
    Operation.convert_value_unit: ConvertValueUnitParameters,
    Operation.masked_convert_value_unit: ConvertValueUnitParameters,
    Operation.convert_axis_unit: ConvertAxisUnitParameters,
    Operation.convert_temperature_unit: ConvertTemperatureUnitParameters,
    Operation.masked_convert_temperature_unit: ConvertTemperatureUnitParameters,
    Operation.chunked_count_nonzero: ChunkedCountParameters,
    Operation.chunked_masked_count_nonzero: ChunkedCountParameters,
    Operation.chunked_count_nonzero_by_axis: AxisChunkedCountParameters,
    Operation.chunked_masked_count_nonzero_by_axis: AxisChunkedCountParameters,
    Operation.chunked_masked_count_nonzero_with_placeholders: AxisChunkedCountParameters,
    Operation.chunked_sum_by_axis: AxisChunkedSumParameters,
    Operation.chunked_masked_sum_with_placeholders: AxisChunkedSumParameters,
    Operation.chunked_mean_by_axis: AxisChunkedMeanParameters,
    Operation.chunked_masked_mean_with_placeholders: AxisChunkedMeanParameters,
    Operation.chunked_variance_by_axis: AxisChunkedDispersionParameters,
    Operation.chunked_std_by_axis: AxisChunkedDispersionParameters,
    Operation.chunked_masked_variance_with_placeholders: AxisChunkedDispersionParameters,
    Operation.chunked_masked_std_with_placeholders: AxisChunkedDispersionParameters,
    Operation.chunked_min_by_axis: AxisChunkedExtremaParameters,
    Operation.chunked_max_by_axis: AxisChunkedExtremaParameters,
    Operation.chunked_masked_min_with_placeholders: AxisChunkedExtremaParameters,
    Operation.chunked_masked_max_with_placeholders: AxisChunkedExtremaParameters,
    Operation.chunked_reduce_validity_any: ValidityReductionParameters,
    Operation.sum: ReduceParameters,
    Operation.mean: ReduceParameters,
    Operation.masked_sum: ReduceParameters,
    Operation.masked_mean: ReduceParameters,
    Operation.masked_sum_with_placeholders: ValidityReductionParameters,
    Operation.masked_mean_with_placeholders: SelectedMeanParameters,
    Operation.masked_min_with_placeholders: ValidityReductionParameters,
    Operation.masked_max_with_placeholders: ValidityReductionParameters,
    Operation.masked_variance_with_placeholders: ValidityReductionParameters,
    Operation.masked_std_with_placeholders: ValidityReductionParameters,
    Operation.reduce_validity_any: ValidityAnyParameters,
    Operation.variance: ReduceParameters,
    Operation.std: ReduceParameters,
    Operation.min: ReduceParameters,
    Operation.max: ReduceParameters,
}
ARITHMETIC = {Operation.add, Operation.subtract, Operation.multiply, Operation.divide}
MATRIX_MULTIPLICATION = {Operation.matmul, Operation.batched_matmul}
PAIRED_SCALING_OPERATIONS = {
    Operation.normalize: Operation.masked_normalize_with_placeholders,
    Operation.standardize: Operation.masked_standardize_with_placeholders,
}
PAIRED_SCALING_VALUES = frozenset(PAIRED_SCALING_OPERATIONS.values())
MASKED_SCALING_WORK_FLOAT_ARRAYS = 10
MAX_SELECTED_SCALING_GROUPS = 16_384
SELECTED_AXIS_SCALING_ALGORITHM: Literal["slice_wise_masked_scale_v1"] = (
    "slice_wise_masked_scale_v1"
)
PAIRED_REDUCTION_OPERATIONS = {
    Operation.sum: Operation.masked_sum_with_placeholders,
    Operation.mean: Operation.masked_mean_with_placeholders,
    Operation.min: Operation.masked_min_with_placeholders,
    Operation.max: Operation.masked_max_with_placeholders,
    Operation.variance: Operation.masked_variance_with_placeholders,
    Operation.std: Operation.masked_std_with_placeholders,
    Operation.chunked_count_nonzero_by_axis: (
        Operation.chunked_masked_count_nonzero_with_placeholders
    ),
    Operation.chunked_sum_by_axis: Operation.chunked_masked_sum_with_placeholders,
    Operation.chunked_mean_by_axis: Operation.chunked_masked_mean_with_placeholders,
    Operation.chunked_variance_by_axis: Operation.chunked_masked_variance_with_placeholders,
    Operation.chunked_std_by_axis: Operation.chunked_masked_std_with_placeholders,
    Operation.chunked_min_by_axis: Operation.chunked_masked_min_with_placeholders,
    Operation.chunked_max_by_axis: Operation.chunked_masked_max_with_placeholders,
}
PAIRED_REDUCTION_VALUES = frozenset(PAIRED_REDUCTION_OPERATIONS.values())
CHUNKED_PAIRED_REDUCTION_VALUES = frozenset(
    value
    for operation, value in PAIRED_REDUCTION_OPERATIONS.items()
    if operation.startswith("chunked_")
)
REDUCTIONS = {
    Operation.sum,
    Operation.chunked_sum_by_axis,
    Operation.chunked_masked_sum_with_placeholders,
    Operation.chunked_mean_by_axis,
    Operation.chunked_masked_mean_with_placeholders,
    Operation.chunked_variance_by_axis,
    Operation.chunked_std_by_axis,
    Operation.chunked_masked_variance_with_placeholders,
    Operation.chunked_masked_std_with_placeholders,
    Operation.chunked_min_by_axis,
    Operation.chunked_max_by_axis,
    Operation.chunked_masked_min_with_placeholders,
    Operation.chunked_masked_max_with_placeholders,
    Operation.chunked_reduce_validity_any,
    Operation.mean,
    Operation.masked_sum,
    Operation.masked_mean,
    Operation.masked_sum_with_placeholders,
    Operation.masked_mean_with_placeholders,
    Operation.masked_min_with_placeholders,
    Operation.masked_max_with_placeholders,
    Operation.masked_variance_with_placeholders,
    Operation.masked_std_with_placeholders,
    Operation.reduce_validity_any,
    Operation.variance,
    Operation.std,
    Operation.min,
    Operation.max,
}
MASKED_REDUCTIONS = {
    Operation.masked_sum,
    Operation.masked_mean,
    Operation.masked_sum_with_placeholders,
    Operation.masked_mean_with_placeholders,
    Operation.masked_min_with_placeholders,
    Operation.masked_max_with_placeholders,
    Operation.masked_variance_with_placeholders,
    Operation.masked_std_with_placeholders,
    Operation.chunked_masked_sum_with_placeholders,
    Operation.chunked_masked_mean_with_placeholders,
    Operation.chunked_masked_variance_with_placeholders,
    Operation.chunked_masked_std_with_placeholders,
    Operation.chunked_masked_min_with_placeholders,
    Operation.chunked_masked_max_with_placeholders,
}
CHUNKED_COUNTS = {
    Operation.chunked_count_nonzero,
    Operation.chunked_masked_count_nonzero,
    Operation.chunked_count_nonzero_by_axis,
    Operation.chunked_masked_count_nonzero_by_axis,
    Operation.chunked_masked_count_nonzero_with_placeholders,
}
CHUNKED_SUMS = {
    Operation.chunked_sum_by_axis,
    Operation.chunked_masked_sum_with_placeholders,
}
CHUNKED_MEANS = {
    Operation.chunked_mean_by_axis,
    Operation.chunked_masked_mean_with_placeholders,
}
CHUNKED_DISPERSIONS = {
    Operation.chunked_variance_by_axis,
    Operation.chunked_std_by_axis,
    Operation.chunked_masked_variance_with_placeholders,
    Operation.chunked_masked_std_with_placeholders,
}
CHUNKED_VARIANCES = {
    Operation.chunked_variance_by_axis,
    Operation.chunked_masked_variance_with_placeholders,
}
CHUNKED_EXTREMA = {
    Operation.chunked_min_by_axis,
    Operation.chunked_max_by_axis,
    Operation.chunked_masked_min_with_placeholders,
    Operation.chunked_masked_max_with_placeholders,
}
CHUNKED_MINIMA = {
    Operation.chunked_min_by_axis,
    Operation.chunked_masked_min_with_placeholders,
}
CHUNKED_OPERATIONS = (
    CHUNKED_COUNTS
    | CHUNKED_SUMS
    | CHUNKED_MEANS
    | CHUNKED_DISPERSIONS
    | CHUNKED_EXTREMA
    | {Operation.chunked_reduce_validity_any, Operation.chunked_slice}
)
CHUNKED_MASKED_COUNTS = {
    Operation.chunked_masked_count_nonzero,
    Operation.chunked_masked_count_nonzero_by_axis,
    Operation.chunked_masked_count_nonzero_with_placeholders,
}
CHUNKED_AXIS_COUNTS = {
    Operation.chunked_count_nonzero_by_axis,
    Operation.chunked_masked_count_nonzero_by_axis,
    Operation.chunked_masked_count_nonzero_with_placeholders,
}
NO_FLOAT64_PROMOTION = {
    Operation.transpose,
    Operation.slice,
    Operation.chunked_slice,
    Operation.select_indices,
    Operation.reindex_coordinates,
    Operation.reindex_validity,
    Operation.coordinate_coverage,
    Operation.logical_and,
    Operation.reshape,
    Operation.convert_axis_unit,
    Operation.cast,
    Operation.masked_cast_with_placeholders,
    Operation.reduce_validity_any,
    Operation.chunked_reduce_validity_any,
    Operation.chunked_count_nonzero,
    Operation.chunked_masked_count_nonzero,
    Operation.chunked_count_nonzero_by_axis,
    Operation.chunked_masked_count_nonzero_by_axis,
    Operation.chunked_masked_count_nonzero_with_placeholders,
    Operation.min,
    Operation.max,
    Operation.masked_min_with_placeholders,
    Operation.masked_max_with_placeholders,
    *CHUNKED_EXTREMA,
}


@dataclass(frozen=True)
class TensorShape:
    shape: tuple[int, ...]
    dtype: str
    cost: int = 0

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.size * np.dtype(self.dtype).itemsize


def parameters(step: Step) -> dict[str, Any]:
    try:
        return PARAMETERS[step.operation].model_validate(step.parameters).model_dump(mode="json")
    except ValidationError as exc:
        message = f"Invalid parameters for {step.operation}"
        if step.operation in ARITHMETIC and is_ambiguous_arithmetic_scalar(
            step.parameters.get("scalar")
        ):
            message += (
                ": scalar cannot be represented unambiguously as float64; "
                "review and submit a new plan or workflow revision"
            )
        raise TensorusError(
            message,
            details=exc.errors(include_input=False, include_context=False),
        ) from exc


def reduction_axes(axis: int | list[int] | None, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    axes = [axis] if isinstance(axis, int) else axis
    if any(not -ndim <= value < ndim for value in axes):
        raise TensorusError("Reduction axis is out of bounds")
    normalized = tuple(value % ndim for value in axes)
    if len(set(normalized)) != len(normalized):
        raise TensorusError("Reduction axes must be unique")
    return normalized


def paired_reduction_dtype(operation: Operation, input_dtype: str) -> str:
    if operation in (Operation.reduce_validity_any, Operation.chunked_reduce_validity_any):
        return "bool"
    if operation == Operation.chunked_masked_count_nonzero_with_placeholders:
        return "int64"
    if operation in (
        Operation.masked_min_with_placeholders,
        Operation.masked_max_with_placeholders,
        Operation.chunked_masked_min_with_placeholders,
        Operation.chunked_masked_max_with_placeholders,
    ):
        return input_dtype
    if operation in PAIRED_REDUCTION_VALUES:
        return "float64"
    raise TensorusError("Unsupported paired reduction operation")


def fitted_state_shape(
    shape: tuple[int, ...], axis: int | list[int] | None, algorithm_version: int = 1
) -> tuple[int, ...]:
    reduced = reduction_axes(axis, len(shape))
    count = 2 if algorithm_version == 1 else 3
    return (count, *(dimension for index, dimension in enumerate(shape) if index not in reduced))


def predict(step: Step, inputs: list[TensorShape], settings: "Settings") -> TensorShape | None:
    params = parameters(step)
    operation = step.operation
    if len(inputs) != len(step.inputs):
        raise TensorusError("Input shape count does not match the step")
    required = operation_input_count(operation, params, len(inputs))
    if len(inputs) != required:
        raise TensorusError(f"{operation} requires {required} tensor input(s)")
    first = inputs[0]
    if operation == Operation.profile:
        return None
    shape, dtype, cost = first.shape, "float64", first.size * 6
    try:
        if operation in ARITHMETIC:
            shape = np.broadcast_shapes(*(item.shape for item in inputs))
            cost = math.prod(shape) * 2
        elif operation in MASKED_BINARY_OPERATIONS:
            kind = (
                "addition" if operation == Operation.masked_add_with_placeholders else "arithmetic"
            )
            if inputs[1].shape != first.shape or inputs[2].shape != first.shape:
                raise TensorusError(
                    f"Masked {kind} requires three inputs with the exact same shape"
                )
            if inputs[2].dtype != "bool":
                raise TensorusError(f"Masked {kind} requires a boolean inclusion mask")
            cost = first.size * (
                14
                if operation
                in (
                    Operation.masked_multiply_with_placeholders,
                    Operation.masked_divide_with_placeholders,
                )
                else 10
            )
        elif operation in PAIRED_SCALING_VALUES:
            if inputs[1].dtype != "bool" or inputs[1].shape != first.shape:
                raise TensorusError(
                    "Masked scaling requires a boolean mask with the exact input shape"
                )
            estimated_bytes = (
                first.nbytes
                + inputs[1].nbytes
                + first.size * np.dtype("float64").itemsize * MASKED_SCALING_WORK_FLOAT_ARRAYS
            )
            if estimated_bytes > settings.max_tensor_bytes:
                raise LimitExceeded("Masked scaling exceeds the working-memory estimate limit")
            cost = first.size * 40
            if params["algorithm_version"] == 2:
                scaling_axes = reduction_axes(params.get("axis"), len(first.shape))
                groups = (
                    math.prod(
                        dimension
                        for axis, dimension in enumerate(first.shape)
                        if axis not in scaling_axes
                    )
                    if first.size
                    else 0
                )
                if groups > MAX_SELECTED_SCALING_GROUPS:
                    raise LimitExceeded("Selected axis scaling exceeds the slice-group limit")
                cost += groups * 64
        elif operation in (
            Operation.fit_selected_preprocessing,
            Operation.apply_fitted_preprocessing,
        ):
            mask = inputs[1] if operation == Operation.fit_selected_preprocessing else inputs[2]
            if mask.dtype != "bool" or mask.shape != first.shape:
                raise TensorusError(
                    "Fitted preprocessing requires an exactly matching boolean mask"
                )
            state_shape = fitted_state_shape(
                first.shape, params["axis"], params["algorithm_version"]
            )
            groups = math.prod(state_shape[1:])
            if len(state_shape) > 16 or groups > MAX_SELECTED_SCALING_GROUPS:
                raise LimitExceeded("Fitted preprocessing exceeds the rank or slice-group limit")
            if operation == Operation.fit_selected_preprocessing:
                if first.size == 0:
                    raise TensorusError("Fitting requires a nonempty training tensor")
                shape = state_shape
            elif inputs[1].dtype != "float64" or inputs[1].shape != state_shape:
                raise TensorusError(
                    "Fitted state shape and dtype do not match the transform domain"
                )
            estimated_bytes = (
                sum(item.nbytes for item in inputs)
                + first.size * np.dtype("float64").itemsize * MASKED_SCALING_WORK_FLOAT_ARRAYS
                + math.prod(state_shape) * np.dtype("float64").itemsize
            )
            if estimated_bytes > settings.max_tensor_bytes:
                raise LimitExceeded(
                    "Fitted preprocessing exceeds the working-memory estimate limit"
                )
            cost = first.size * 40 + groups * 64
        elif operation == Operation.masked_cast_with_placeholders:
            if first.dtype == "bool":
                raise TensorusError("Selected casting requires non-boolean numeric values")
            if inputs[1].dtype != "bool" or inputs[1].shape != first.shape:
                raise TensorusError(
                    "Selected casting requires a boolean mask with the exact input shape"
                )
            dtype = params["dtype"]
            estimated_bytes = (
                first.nbytes
                + inputs[1].nbytes
                + first.size
                * (2 * np.dtype(first.dtype).itemsize + 2 * np.dtype(dtype).itemsize + 3)
            )
            if estimated_bytes > settings.max_tensor_bytes:
                raise LimitExceeded("Selected casting exceeds the working-memory estimate limit")
            cost = first.size * 8
        elif operation == Operation.masked_scalar_arithmetic_with_placeholders:
            if first.dtype == "bool":
                raise TensorusError(
                    "Selected scalar arithmetic requires non-boolean numeric values"
                )
            if inputs[1].dtype != "bool" or inputs[1].shape != first.shape:
                raise TensorusError(
                    "Selected scalar arithmetic requires a boolean mask with the exact input shape"
                )
            estimated_bytes = (
                first.nbytes
                + inputs[1].nbytes
                + first.size * (np.dtype(first.dtype).itemsize + np.dtype("float64").itemsize + 4)
            )
            if estimated_bytes > settings.max_tensor_bytes:
                raise LimitExceeded(
                    "Selected scalar arithmetic exceeds the working-memory estimate limit"
                )
            cost = first.size * 10
        elif operation == Operation.logical_and:
            if any(item.dtype != "bool" for item in inputs):
                raise TensorusError("Logical AND requires boolean tensor inputs")
            shape = np.broadcast_shapes(*(item.shape for item in inputs))
            dtype, cost = "bool", math.prod(shape) * 2
        elif operation == Operation.matmul:
            second = inputs[1]
            if len(first.shape) != 2 or len(second.shape) != 2:
                raise TensorusError(
                    "This release supports matrix multiplication of two 2-D tensors"
                )
            if first.shape[1] != second.shape[0]:
                raise TensorusError("Matrix multiplication inner dimensions do not match")
            shape = (first.shape[0], second.shape[1])
            cost = 2 * first.shape[0] * first.shape[1] * second.shape[1]
        elif operation == Operation.batched_matmul:
            second = inputs[1]
            if len(first.shape) < 2 or len(second.shape) < 2:
                raise TensorusError(
                    "Batched matrix multiplication requires rank-two or higher inputs"
                )
            if len(first.shape) == len(second.shape) == 2:
                raise TensorusError("Batched matrix multiplication requires a batch dimension")
            if first.shape[-1] != second.shape[-2]:
                raise TensorusError("Matrix multiplication inner dimensions do not match")
            try:
                batch = np.broadcast_shapes(first.shape[:-2], second.shape[:-2])
            except ValueError as exc:
                raise TensorusError("Matrix batch dimensions cannot be broadcast together") from exc
            rows, inner, columns = first.shape[-2], first.shape[-1], second.shape[-1]
            shape = (*batch, rows, columns)
            cost = max(math.prod(shape), 2 * math.prod(batch) * rows * inner * columns)
        elif operation == Operation.transpose:
            axes = params["axes"]
            if axes is None:
                axes = list(reversed(range(len(first.shape))))
            if sorted(axes) != list(range(len(first.shape))):
                raise TensorusError("Transpose axes must be a permutation of all dimensions")
            shape = tuple(first.shape[axis] for axis in axes)
            dtype = first.dtype
        elif operation in (Operation.slice, Operation.chunked_slice):
            selection = params["selection"]
            if len(selection) > len(first.shape):
                raise TensorusError("Slice selection has more axes than the input")
            shape = tuple(
                len(range(*slice(*selection[axis]).indices(dimension)))
                if axis < len(selection)
                else dimension
                for axis, dimension in enumerate(first.shape)
            )
            dtype = first.dtype
            if operation == Operation.chunked_slice:
                if first.nbytes > settings.chunked_input_bytes:
                    raise LimitExceeded(
                        "A chunked input exceeds the configured chunked-input limit"
                    )
                _, _, count = chunked_block_layout(first.shape, np.dtype(first.dtype).itemsize)
                if count > CHUNKED_MAX_BLOCKS:
                    raise LimitExceeded("Tensor exceeds the chunked block-count limit")
                cost = first.size + math.prod(shape) + count * max(1, len(first.shape))
        elif operation == Operation.select_indices:
            indices = params["indices"]
            if len(indices) != len(first.shape):
                raise TensorusError("Index selection requires one entry per input axis")
            dimensions = list(first.shape)
            cost = first.size
            for axis, selected in enumerate(indices):
                if selected is not None:
                    if selected and selected[-1] >= first.shape[axis]:
                        raise TensorusError("Index selection is out of bounds")
                    dimensions[axis] = len(selected)
                    cost += math.prod(dimensions) + len(selected)
            shape, dtype = tuple(dimensions), first.dtype
        elif operation in (
            Operation.reindex_coordinates,
            Operation.reindex_validity,
            Operation.coordinate_coverage,
        ):
            axes = params["axes"]
            if len(axes) != len(first.shape):
                raise TensorusError("Coordinate reindexing requires one entry per input axis")
            if operation == Operation.reindex_coordinates:
                reindex_fill(first.dtype, params["fill_value"])
            elif operation == Operation.reindex_validity and first.dtype != "bool":
                raise TensorusError("Validity reindexing requires a boolean tensor input")
            dimensions = list(first.shape)
            selected = []
            for axis, mapping in enumerate(axes):
                if mapping is None:
                    selected.append(first.shape[axis])
                    continue
                indices = [index for index in mapping["indices"] if index is not None]
                if indices and indices[-1] >= first.shape[axis]:
                    raise TensorusError("Coordinate reindexing is out of bounds")
                dimensions[axis] = len(mapping["coordinates"])
                selected.append(len(indices))
            shape = tuple(dimensions)
            dtype = first.dtype if operation == Operation.reindex_coordinates else "bool"
            if len(inputs) == 2 and inputs[1].shape != shape:
                raise TensorusError("Coordinate anchor shape must match the target shape")
            if sum(dimensions) > MAX_SELECTION_INDICES:
                raise LimitExceeded("Coordinate reindexing exceeds the total position limit")
            cost = (
                math.prod(shape) + math.prod(selected) + sum(dimensions)
                if operation in (Operation.reindex_coordinates, Operation.reindex_validity)
                else math.prod(shape) * (1 + sum(axis is not None for axis in axes))
                + sum(dimensions)
            )
        elif operation == Operation.reshape:
            requested = params["shape"]
            if requested.count(-1) > 1:
                raise TensorusError("Reshape permits at most one inferred dimension")
            known = math.prod(value for value in requested if value != -1)
            if -1 in requested:
                if known == 0 or first.size % known:
                    raise TensorusError("Cannot infer the requested reshape dimension")
                requested = [first.size // known if value == -1 else value for value in requested]
            if math.prod(requested) != first.size:
                raise TensorusError("Reshape must preserve the element count")
            shape, dtype = tuple(requested), first.dtype
        elif operation == Operation.cast:
            dtype = params["dtype"]
        elif operation == Operation.convert_axis_unit:
            dtype = first.dtype
            cost = first.size + min(max(first.shape, default=0), 8192)
        elif operation in (Operation.convert_value_unit, Operation.masked_convert_value_unit):
            if first.dtype == "bool":
                raise TensorusError("Value-unit conversion requires non-boolean numeric values")
            if operation == Operation.masked_convert_value_unit and (
                inputs[1].dtype != "bool" or inputs[1].shape != first.shape
            ):
                raise TensorusError(
                    "Masked value-unit conversion requires a boolean mask "
                    "with the exact input shape"
                )
            cost = first.size * (4 if operation == Operation.masked_convert_value_unit else 2)
        elif operation in (
            Operation.convert_temperature_unit,
            Operation.masked_convert_temperature_unit,
        ):
            if first.dtype == "bool":
                raise TensorusError("Temperature conversion requires non-boolean numeric values")
            if operation == Operation.masked_convert_temperature_unit and (
                inputs[1].dtype != "bool" or inputs[1].shape != first.shape
            ):
                raise TensorusError(
                    "Masked temperature conversion requires a boolean mask "
                    "with the exact input shape"
                )
            if first.size > MAX_TEMPERATURE_ELEMENTS:
                raise LimitExceeded("Temperature conversion exceeds the 100000-element limit")
            cost = first.size * (
                32 if operation == Operation.masked_convert_temperature_unit else 30
            )
        elif operation in (
            CHUNKED_SUMS
            | CHUNKED_MEANS
            | CHUNKED_DISPERSIONS
            | CHUNKED_EXTREMA
            | {Operation.chunked_reduce_validity_any}
        ):
            if any(item.nbytes > settings.chunked_input_bytes for item in inputs):
                raise LimitExceeded("A chunked input exceeds the configured chunked-input limit")
            if operation in MASKED_REDUCTIONS and (
                inputs[1].dtype != "bool" or inputs[1].shape != first.shape
            ):
                raise TensorusError(
                    "Masked reductions require a boolean mask with the exact input shape"
                )
            if operation == Operation.chunked_reduce_validity_any and first.dtype != "bool":
                raise TensorusError("Validity reduction requires a boolean tensor input")
            itemsize = (
                max(np.dtype(first.dtype).itemsize, CHUNKED_DISPERSION_ITEMSIZE)
                if operation in CHUNKED_DISPERSIONS
                else np.dtype(first.dtype).itemsize
            )
            _, _, count = chunked_block_layout(first.shape, itemsize)
            passes = CHUNKED_DISPERSION_PASSES if operation in CHUNKED_DISPERSIONS else 1
            if count * passes > CHUNKED_MAX_BLOCKS:
                raise LimitExceeded("Tensor exceeds the chunked block-count limit")
            axes = reduction_axes(params["axis"], len(first.shape))
            if operation == Operation.chunked_mean_by_axis and any(
                first.shape[axis] == 0 for axis in axes
            ):
                raise TensorusError("Chunked mean of an empty reduction is undefined")
            if operation in CHUNKED_DISPERSIONS - MASKED_REDUCTIONS and any(
                first.shape[axis] == 0 for axis in axes
            ):
                raise TensorusError("Chunked dispersion of an empty reduction is undefined")
            if operation in (Operation.chunked_min_by_axis, Operation.chunked_max_by_axis) and any(
                first.shape[axis] == 0 for axis in axes
            ):
                raise TensorusError("Chunked extremum of an empty reduction is undefined")
            shape = reduced_shape(first.shape, axes, params["keepdims"])
            if operation in CHUNKED_DISPERSIONS and (
                math.prod(shape) * CHUNKED_DISPERSION_ITEMSIZE * CHUNKED_DISPERSION_WORK_ARRAYS
                > settings.max_tensor_bytes
            ):
                raise LimitExceeded("Chunked dispersion exceeds the output working-memory limit")
            multiplier = (
                48
                if operation in CHUNKED_DISPERSIONS & MASKED_REDUCTIONS
                else 40
                if operation in CHUNKED_DISPERSIONS
                else 12
                if operation == Operation.chunked_masked_mean_with_placeholders
                else 10
                if operation in MASKED_REDUCTIONS
                else 8
            )
            if operation == Operation.chunked_reduce_validity_any:
                dtype, multiplier = "bool", 1
            elif operation in CHUNKED_EXTREMA:
                dtype = first.dtype
            cost = first.size * multiplier + math.prod(shape) * (
                CHUNKED_DISPERSION_WORK_ARRAYS
                if operation in CHUNKED_DISPERSIONS
                else 3
                if operation in CHUNKED_MEANS
                else 2
                if operation in CHUNKED_EXTREMA & MASKED_REDUCTIONS
                else 1
            )
        elif operation in CHUNKED_COUNTS:
            if any(item.nbytes > settings.chunked_input_bytes for item in inputs):
                raise LimitExceeded("A chunked input exceeds the configured chunked-input limit")
            if operation in CHUNKED_MASKED_COUNTS:
                if inputs[1].dtype != "bool" or inputs[1].shape != first.shape:
                    raise TensorusError(
                        "Masked chunked counting requires a boolean mask with the exact input shape"
                    )
                if operation == Operation.chunked_masked_count_nonzero and not first.size:
                    raise TensorusError(
                        "Masked chunked counting requires at least one included value"
                    )
                cost = first.size * 10
            _, _, count = chunked_block_layout(first.shape, np.dtype(first.dtype).itemsize)
            if count > CHUNKED_MAX_BLOCKS:
                raise LimitExceeded("Tensor exceeds the chunked block-count limit")
            if operation in CHUNKED_AXIS_COUNTS:
                axes = reduction_axes(params["axis"], len(first.shape))
                if operation == Operation.chunked_masked_count_nonzero_by_axis and any(
                    first.shape[axis] == 0 for axis in axes
                ):
                    raise TensorusError(
                        "Masked chunked counting of an empty reduction is undefined"
                    )
                shape = reduced_shape(first.shape, axes, params["keepdims"])
                cost += math.prod(shape) * (
                    2 if operation == Operation.chunked_masked_count_nonzero_by_axis else 1
                )
            else:
                shape = ()
            dtype = "int64"
        elif operation in REDUCTIONS:
            axes = reduction_axes(params["axis"], len(first.shape))
            if operation in MASKED_REDUCTIONS and (
                inputs[1].dtype != "bool" or inputs[1].shape != first.shape
            ):
                raise TensorusError(
                    "Masked reductions require a boolean mask with the exact input shape"
                )
            if operation == Operation.reduce_validity_any and first.dtype != "bool":
                raise TensorusError("Validity reduction requires a boolean tensor input")
            if operation in (
                Operation.mean,
                Operation.masked_sum,
                Operation.masked_mean,
                Operation.variance,
                Operation.std,
                Operation.min,
                Operation.max,
            ) and any(first.shape[axis] == 0 for axis in axes):
                raise TensorusError(f"{operation} of an empty reduction is undefined")
            shape = reduced_shape(first.shape, axes, params["keepdims"])
            if operation in (Operation.variance, Operation.std):
                # Several float64 passes and temporary arrays are used for stable dispersion.
                cost = first.size * 20
            if operation in MASKED_REDUCTIONS:
                cost = first.size * 8 + math.prod(shape)
            if operation == Operation.masked_mean_with_placeholders:
                if params["algorithm_version"] == 2:
                    estimated_bytes = (
                        first.nbytes
                        + inputs[1].nbytes
                        + first.size * (np.dtype(first.dtype).itemsize + 3 * 8 + 5)
                        + math.prod(shape) * 64
                    )
                    if estimated_bytes > settings.max_tensor_bytes:
                        raise LimitExceeded(
                            "Scaled masked mean exceeds the working-memory estimate limit"
                        )
                    cost = first.size * 30 + math.prod(shape) * 8
                else:
                    cost = first.size * 10 + math.prod(shape) * 2
            if operation in (
                Operation.masked_variance_with_placeholders,
                Operation.masked_std_with_placeholders,
            ):
                cost = first.size * 30 + math.prod(shape) * 8
            if operation == Operation.reduce_validity_any:
                dtype, cost = "bool", first.size + math.prod(shape)
            if operation in (
                Operation.min,
                Operation.max,
                Operation.masked_min_with_placeholders,
                Operation.masked_max_with_placeholders,
            ):
                dtype = first.dtype
    except ValueError as exc:
        raise TensorusError("Input shapes cannot be broadcast together") from exc
    if operation in PAIRED_REDUCTION_VALUES or operation in (
        Operation.reduce_validity_any,
        Operation.chunked_reduce_validity_any,
    ):
        dtype = paired_reduction_dtype(operation, first.dtype)
    declared_dtype = OPERATION_CONTRACTS[operation].dtype
    expected_dtype = (
        first.dtype
        if declared_dtype == "input"
        else params["dtype"]
        if declared_dtype == "target"
        else declared_dtype
    )
    if dtype != expected_dtype:
        raise RuntimeError(f"Predicted dtype does not match the {operation} contract")
    output = TensorShape(shape, dtype, cost)
    if (
        len(shape) > 16
        or output.nbytes > settings.max_tensor_bytes
        or (
            output.size == 0
            and math.prod(max(1, dimension) for dimension in shape)
            > np.iinfo(np.intp).max // np.dtype(dtype).itemsize
        )
    ):
        raise LimitExceeded("Predicted operation output exceeds the tensor byte/dimension limit")
    if output.cost > settings.max_compute_cost:
        raise LimitExceeded("Predicted operation exceeds the compute budget")
    return output


def stable_std(first: Array, axes: tuple[int, ...], keepdims: bool) -> np.ndarray:
    """Population standard deviation using per-slice scaling in float64.

    This avoids overflow in the mean and squared deviations; as with other
    floating-point reductions it is not an exact or correctly rounded result.
    """
    values = first.astype(np.float64)
    low = np.min(values, axis=axes, keepdims=True)
    high = np.max(values, axis=axes, keepdims=True)
    midpoint = low / 2 + high / 2
    deviations = values - midpoint
    scale = np.max(np.abs(deviations), axis=axes, keepdims=True)
    normalized = np.divide(deviations, scale, out=np.zeros_like(deviations), where=scale != 0)
    centered = normalized - np.mean(normalized, axis=axes, keepdims=True)
    centered_scale = np.max(np.abs(centered), axis=axes, keepdims=True)
    normalized_centered = np.divide(
        centered, centered_scale, out=np.zeros_like(centered), where=centered_scale != 0
    )
    spread = scale * (
        centered_scale * np.sqrt(np.mean(normalized_centered**2, axis=axes, keepdims=True))
    )
    if np.any((low != high) & (spread == 0)):
        raise TensorusError("Standard deviation underflowed float64")
    return np.squeeze(spread, axis=axes) if not keepdims else spread


def stable_masked_std(
    first: Array, mask: Array, axes: tuple[int, ...], keepdims: bool
) -> np.ndarray:
    """Scale included population deviations; return zero for empty slices."""
    values = first.astype(np.float64)
    included = np.any(mask, axis=axes, keepdims=True)
    counts = np.sum(mask, axis=axes, keepdims=True, dtype=np.int64)
    low = np.min(values, axis=axes, keepdims=True, where=mask, initial=np.inf)
    high = np.max(values, axis=axes, keepdims=True, where=mask, initial=-np.inf)
    low = np.where(included, low, 0.0)
    high = np.where(included, high, 0.0)
    midpoint = low / 2 + high / 2
    deviations = np.subtract(values, midpoint, out=np.zeros_like(values), where=mask)
    scale = np.max(np.abs(deviations), axis=axes, keepdims=True, initial=0)
    normalized = np.divide(deviations, scale, out=np.zeros_like(deviations), where=scale != 0)
    normalized_sum = np.sum(normalized, axis=axes, keepdims=True, dtype=np.float64)
    mean = np.divide(normalized_sum, counts, out=np.zeros_like(normalized_sum), where=included)
    centered = np.subtract(normalized, mean, out=np.zeros_like(normalized), where=mask)
    centered_scale = np.max(np.abs(centered), axis=axes, keepdims=True, initial=0)
    normalized_centered = np.divide(
        centered, centered_scale, out=np.zeros_like(centered), where=centered_scale != 0
    )
    squares = np.sum(normalized_centered**2, axis=axes, keepdims=True, dtype=np.float64)
    mean_square = np.divide(squares, counts, out=np.zeros_like(squares), where=included)
    spread = scale * (centered_scale * np.sqrt(mean_square))
    if np.any((low != high) & (spread == 0)):
        raise TensorusError("Standard deviation underflowed float64")
    return np.squeeze(spread, axis=axes) if not keepdims else spread


def scaled_masked_mean(
    first: Array, mask: Array, axes: tuple[int, ...], keepdims: bool
) -> np.ndarray:
    values = first.astype(np.float64)
    scale = np.max(np.abs(values), axis=axes, keepdims=True, where=mask, initial=0.0)
    with np.errstate(under="ignore"):
        normalized = np.divide(
            values,
            scale,
            out=np.zeros_like(values),
            where=mask & (scale != 0),
        )
        if np.any(mask & (values != 0) & (normalized == 0)):
            raise TensorusError("Scaled masked mean normalization underflowed float64 to zero")
        totals = np.sum(normalized, axis=axes, keepdims=True, dtype=np.float64)
        counts = np.sum(mask, axis=axes, keepdims=True, dtype=np.int64)
        means = np.divide(totals, counts, out=np.zeros_like(totals), where=counts != 0)
        if np.any((totals != 0) & (means == 0)):
            raise TensorusError("Scaled masked mean division underflowed float64 to zero")
        result = means * scale
        if np.any((means != 0) & (result == 0)):
            raise TensorusError("Scaled masked mean result underflowed float64 to zero")
    return np.squeeze(result, axis=axes) if not keepdims else result


def masked_scale(first: Array, mask: Array, epsilon: float, *, standardize: bool) -> Array:
    selected = first[mask].astype(np.float64)
    result = np.zeros(first.shape, dtype=np.float64)
    if not selected.size:
        return result
    if standardize:
        low, high = selected.min(), selected.max()
        midpoint = low / 2 + high / 2
        deviations = selected - midpoint
        scale = np.max(np.abs(deviations))
        if scale == 0:
            result[mask] = 0.0
            return result
        normalized = deviations / scale
        if np.any((deviations != 0) & (normalized == 0)):
            raise TensorusError("Masked standardization underflowed float64 to zero")
        centered = normalized - math.fsum(normalized) / normalized.size
        spread = np.sqrt(np.mean(np.square(centered), dtype=np.float64))
        if spread == 0:
            if low != high:
                raise TensorusError("Masked standardization dispersion underflowed float64")
            result[mask] = 0.0
            return result
        transformed = (
            centered / spread if scale >= epsilon / spread else centered * (scale / epsilon)
        )
        if np.any((centered != 0) & (transformed == 0)):
            raise TensorusError("Masked standardization underflowed float64 to zero")
    else:
        scale = np.max(np.abs(selected))
        if scale == 0:
            result[mask] = selected / epsilon
            return result
        normalized = selected / scale
        if np.any((selected != 0) & (normalized == 0)):
            raise TensorusError("Masked normalization underflowed float64 to zero")
        norm = np.sqrt(np.sum(np.square(normalized), dtype=np.float64))
        transformed = normalized / norm if scale >= epsilon / norm else selected / epsilon
        if np.any((selected != 0) & (transformed == 0)):
            raise TensorusError("Masked normalization underflowed float64 to zero")
    result[mask] = transformed
    return result


def axis_masked_scale(
    first: Array,
    mask: Array,
    epsilon: float,
    axes: tuple[int, ...],
    *,
    standardize: bool,
) -> Array:
    result = np.zeros(first.shape, dtype=np.float64)
    if not first.size:
        return result
    remaining = tuple(axis for axis in range(first.ndim) if axis not in axes)
    for position in np.ndindex(*(first.shape[axis] for axis in remaining)):
        selection = [slice(None)] * first.ndim
        for axis, index in zip(remaining, position, strict=True):
            selection[axis] = slice(index, index + 1)
        key: tuple[slice | EllipsisType, ...] = (*selection, Ellipsis)
        result[key] = masked_scale(first[key], mask[key], epsilon, standardize=standardize)
    return result


def _fitted_groups(
    shape: tuple[int, ...], axes: tuple[int, ...]
) -> Iterator[tuple[tuple[int, ...], tuple[slice | EllipsisType, ...]]]:
    remaining = tuple(axis for axis in range(len(shape)) if axis not in axes)
    for position in np.ndindex(*(shape[axis] for axis in remaining)):
        selection = [slice(None)] * len(shape)
        for axis, index in zip(remaining, position, strict=True):
            selection[axis] = slice(index, index + 1)
        yield position, (*selection, Ellipsis)


def fit_selected_scaling(first: Array, mask: Array, params: dict[str, Any]) -> Array:
    axes = reduction_axes(params["axis"], first.ndim)
    version = params["algorithm_version"]
    result = np.empty(
        fitted_state_shape(tuple(first.shape), params["axis"], version), dtype=np.float64
    )
    for position, selection in _fitted_groups(tuple(first.shape), axes):
        selected = first[selection][mask[selection]].astype(np.float64)
        if not selected.size:
            raise TensorusError("Fitting requires at least one included value in every slice")
        if params["algorithm"] == "standardize":
            selected_mask = np.ones(selected.shape, dtype=bool)
            if version == 2:
                anchor = float(selected.min() / 2 + selected.max() / 2)
                deviations = selected - anchor
                center = float(scaled_masked_mean(deviations, selected_mask, (0,), False))
            else:
                center = float(scaled_masked_mean(selected, selected_mask, (0,), False))
            scale = float(stable_masked_std(selected, selected_mask, (0,), False))
        else:
            center = 0.0
            anchor = 0.0
            largest = float(np.max(np.abs(selected)))
            if largest == 0:
                scale = 0.0
            else:
                normalized = selected / largest
                if np.any((selected != 0) & (normalized == 0)):
                    raise TensorusError("Fitted normalization underflowed float64 to zero")
                scale = largest * float(np.sqrt(np.sum(normalized**2, dtype=np.float64)))
        if version == 2:
            result[(0, *position)] = anchor
            result[(1, *position)] = center
            result[(2, *position)] = max(scale, params["epsilon"])
        else:
            result[(0, *position)] = center
            result[(1, *position)] = max(scale, params["epsilon"])
    if not np.isfinite(result).all():
        raise TensorusError("Fitted preprocessing state exceeds float64 range")
    validate_fitted_scaling_state(result, params)
    return result


def validate_fitted_scaling_state(state: Array, params: dict[str, Any]) -> None:
    version = params["algorithm_version"]
    scale_index = 1 if version == 1 else 2
    if not np.isfinite(state).all() or np.any(state[scale_index] <= 0):
        raise TensorusError("Fitted preprocessing state must have finite positive scales")
    if "epsilon" in params and np.any(state[scale_index] < params["epsilon"]):
        raise TensorusError("Fitted preprocessing state has a scale below its claimed epsilon")
    if params["algorithm"] == "l2_normalize":
        if np.any(state[0] != 0) or (version == 2 and np.any(state[1] != 0)):
            raise TensorusError("Fitted normalization state must have zero centers")
    elif version == 1:
        for center, scale in zip(state[0].flat, state[1].flat, strict=True):
            if center != 0 and math.ulp(float(center)) >= float(scale):
                raise TensorusError(
                    "Legacy fitted standardization center is too imprecise for its scale; "
                    "fit a version 2 state"
                )


def apply_fitted_scaling(first: Array, state: Array, mask: Array, params: dict[str, Any]) -> Array:
    validate_fitted_scaling_state(state, params)
    result = np.zeros(first.shape, dtype=np.float64)
    axes = reduction_axes(params["axis"], first.ndim)
    for position, selection in _fitted_groups(tuple(first.shape), axes):
        included = mask[selection]
        selected = first[selection][included].astype(np.float64)
        centered = selected - state[(0, *position)]
        if params["algorithm_version"] == 2:
            centered = centered - state[(1, *position)]
        scale_index = 1 if params["algorithm_version"] == 1 else 2
        with np.errstate(under="ignore"):
            transformed = centered / state[(scale_index, *position)]
        if np.any((centered != 0) & (transformed == 0)):
            raise TensorusError("Fitted preprocessing underflowed float64 to zero")
        group = np.zeros(first[selection].shape, dtype=np.float64)
        group[included] = transformed
        result[selection] = group
    return result


def execute(step: Step, inputs: list[Array], settings: "Settings") -> Array:
    specs = [TensorShape(tuple(item.shape), item.dtype.name) for item in inputs]
    predicted = predict(step, specs, settings)
    if predicted is None:
        raise TensorusError("Profile uses catalog statistics, not an array transformation")
    params = parameters(step)
    operation = step.operation
    if operation == Operation.masked_convert_temperature_unit and not np.isfinite(inputs[0]).all():
        raise TensorusError("Temperature conversion requires finite input values")
    if operation in MASKED_BINARY_OPERATIONS:
        if any(not np.isfinite(item).all() for item in inputs[:2]):
            kind = (
                "addition" if operation == Operation.masked_add_with_placeholders else "arithmetic"
            )
            raise TensorusError(f"Masked {kind} requires finite input values")
        for item in inputs[:2]:
            require_float64_integer_precision(item[inputs[2]])
    elif operation in (
        Operation.masked_scalar_arithmetic_with_placeholders,
        Operation.fit_selected_preprocessing,
        Operation.apply_fitted_preprocessing,
    ):
        selected = inputs[0][
            inputs[2] if operation == Operation.apply_fitted_preprocessing else inputs[1]
        ]
        if not np.isfinite(selected).all():
            raise TensorusError("Selected computation requires finite included values")
        require_float64_integer_precision(selected)
    elif operation not in NO_FLOAT64_PROMOTION and operation not in CHUNKED_DISPERSIONS:
        for index, item in enumerate(inputs):
            if (
                operation
                in (
                    Operation.masked_convert_value_unit,
                    Operation.masked_convert_temperature_unit,
                    Operation.masked_normalize_with_placeholders,
                    Operation.masked_standardize_with_placeholders,
                    Operation.masked_sum_with_placeholders,
                    Operation.masked_mean_with_placeholders,
                    Operation.masked_variance_with_placeholders,
                    Operation.masked_std_with_placeholders,
                    Operation.chunked_masked_sum_with_placeholders,
                    Operation.chunked_masked_mean_with_placeholders,
                )
                and index == 0
            ):
                item = item[inputs[1]]
            require_float64_integer_precision(item)
    try:
        with (
            threadpool_limits(limits=1),
            np.errstate(over="raise", invalid="raise", divide="raise"),
        ):
            first = inputs[0]
            if operation in ARITHMETIC:
                left = first.astype(np.float64)
                right = params["scalar"] if len(inputs) == 1 else inputs[1].astype(np.float64)
                functions = {
                    Operation.add: np.add,
                    Operation.subtract: np.subtract,
                    Operation.multiply: np.multiply,
                    Operation.divide: np.divide,
                }
                result = functions[operation](left, right)
            elif operation in MASKED_BINARY_OPERATIONS:
                result = np.zeros(predicted.shape, dtype=np.float64)
                functions = {
                    Operation.masked_add_with_placeholders: np.add,
                    Operation.masked_subtract_with_placeholders: np.subtract,
                    Operation.masked_multiply_with_placeholders: np.multiply,
                    Operation.masked_divide_with_placeholders: np.divide,
                }
                if operation in (
                    Operation.masked_multiply_with_placeholders,
                    Operation.masked_divide_with_placeholders,
                ):
                    with np.errstate(under="ignore"):
                        functions[operation](
                            first, inputs[1], out=result, where=inputs[2], dtype=np.float64
                        )
                    if np.any(inputs[2] & (first != 0) & (inputs[1] != 0) & (result == 0)):
                        raise TensorusError("Masked arithmetic underflowed float64 to zero")
                else:
                    functions[operation](
                        first, inputs[1], out=result, where=inputs[2], dtype=np.float64
                    )
            elif operation == Operation.masked_scalar_arithmetic_with_placeholders:
                result = np.zeros(predicted.shape, dtype=np.float64)
                scalar_functions = {
                    "add": np.add,
                    "subtract": np.subtract,
                    "multiply": np.multiply,
                    "divide": np.divide,
                }
                with np.errstate(under="ignore"):
                    scalar_functions[params["operator"]](
                        first, params["scalar"], out=result, where=inputs[1], dtype=np.float64
                    )
                if (
                    params["operator"] in ("multiply", "divide")
                    and params["scalar"] != 0
                    and np.any(inputs[1] & (first != 0) & (result == 0))
                ):
                    raise TensorusError("Selected scalar arithmetic underflowed float64 to zero")
            elif operation == Operation.logical_and:
                result = np.logical_and(first, inputs[1])
            elif operation in MATRIX_MULTIPLICATION:
                result = (
                    np.empty(predicted.shape, dtype=np.float64)
                    if predicted.size == 0
                    else np.matmul(first.astype(np.float64), inputs[1].astype(np.float64))
                )
            elif operation in (Operation.normalize, Operation.standardize):
                extended = first.astype(np.longdouble)
                if operation == Operation.normalize:
                    denominator = np.sqrt(np.sum(extended**2))
                    numerator = extended
                else:
                    numerator = extended - np.mean(extended) if first.size else extended
                    denominator = np.sqrt(np.mean(numerator**2)) if first.size else 0
                result = np.asarray(
                    numerator / max(denominator, params["epsilon"]), dtype=np.float64
                )
            elif operation in PAIRED_SCALING_VALUES:
                standardize = operation == Operation.masked_standardize_with_placeholders
                if params["algorithm_version"] == 1:
                    result = masked_scale(
                        first, inputs[1], params["epsilon"], standardize=standardize
                    )
                else:
                    result = axis_masked_scale(
                        first,
                        inputs[1],
                        params["epsilon"],
                        reduction_axes(params.get("axis"), first.ndim),
                        standardize=standardize,
                    )
            elif operation == Operation.fit_selected_preprocessing:
                result = fit_selected_scaling(first, inputs[1], params)
            elif operation == Operation.apply_fitted_preprocessing:
                result = apply_fitted_scaling(first, inputs[1], inputs[2], params)
            elif operation == Operation.transpose:
                result = np.transpose(first, axes=params["axes"])
            elif operation in (Operation.slice, Operation.chunked_slice):
                result = first[tuple(slice(*item) for item in params["selection"])]
            elif operation == Operation.select_indices:
                result = first
                for axis, indices in enumerate(params["indices"]):
                    if indices is not None:
                        result = np.take(result, indices, axis=axis, mode="raise")
            elif operation in (Operation.reindex_coordinates, Operation.reindex_validity):
                result = np.full(
                    predicted.shape,
                    (
                        reindex_fill(first.dtype.name, params["fill_value"])
                        if operation == Operation.reindex_coordinates
                        else False
                    ),
                    dtype=first.dtype,
                )
                source_positions: list[np.ndarray] = []
                target_positions: list[np.ndarray] = []
                for axis, mapping in enumerate(params["axes"]):
                    if mapping is None:
                        positions = np.arange(first.shape[axis], dtype=np.intp)
                        source_positions.append(positions)
                        target_positions.append(positions)
                    else:
                        source_positions.append(
                            np.asarray(
                                [index for index in mapping["indices"] if index is not None],
                                dtype=np.intp,
                            )
                        )
                        target_positions.append(
                            np.asarray(
                                [
                                    index
                                    for index, value in enumerate(mapping["indices"])
                                    if value is not None
                                ],
                                dtype=np.intp,
                            )
                        )
                result[np.ix_(*target_positions)] = first[np.ix_(*source_positions)]
            elif operation == Operation.coordinate_coverage:
                result = np.ones(predicted.shape, dtype=np.bool_)
                for axis, mapping in enumerate(params["axes"]):
                    if mapping is not None:
                        covered = np.asarray(
                            [index is not None for index in mapping["indices"]], dtype=np.bool_
                        )
                        shape = [1] * first.ndim
                        shape[axis] = len(covered)
                        np.logical_and(result, covered.reshape(shape), out=result)
            elif operation == Operation.reshape:
                result = np.reshape(first, predicted.shape)
            elif operation == Operation.cast:
                target = np.dtype(params["dtype"])
                if np.issubdtype(target, np.integer) and first.size:
                    limits = np.iinfo(target)
                    if first.min().item() < limits.min or first.max().item() > limits.max:
                        raise TensorusError("Cast would exceed the target integer range")
                result = first.astype(target)
            elif operation == Operation.masked_cast_with_placeholders:
                target = np.dtype(params["dtype"])
                selected = first[inputs[1]]
                if not np.isfinite(selected).all():
                    raise TensorusError("Selected casting requires finite included values")
                if (
                    np.issubdtype(first.dtype, np.floating)
                    and np.issubdtype(target, np.integer)
                    and np.any((selected == 0) & np.signbit(selected))
                ):
                    raise TensorusError("Selected casting would lose the sign of zero")
                try:
                    converted = selected.astype(target, casting="same_value")
                except ValueError as exc:
                    raise TensorusError("Selected casting would change an included value") from exc
                result = np.zeros(first.shape, dtype=target)
                result[inputs[1]] = converted
            elif operation == Operation.convert_axis_unit:
                result = first
            elif operation in (Operation.convert_value_unit, Operation.masked_convert_value_unit):
                mask = inputs[1] if operation == Operation.masked_convert_value_unit else None
                converted = (first[mask] if mask is not None else first).astype(np.float64)
                with np.errstate(under="ignore"):
                    transformed = np.multiply(converted, value_unit_factor(params))
                if np.any((converted != 0) & (transformed == 0)):
                    raise TensorusError("Value-unit conversion underflowed float64 to zero")
                if mask is None:
                    result = transformed
                else:
                    result = np.zeros(first.shape, dtype=np.float64)
                    result[mask] = transformed
            elif operation in (
                Operation.convert_temperature_unit,
                Operation.masked_convert_temperature_unit,
            ):
                mask = inputs[1] if operation == Operation.masked_convert_temperature_unit else None
                values = first[mask] if mask is not None else first
                source_scale, source_offset = TEMPERATURE_UNITS_V1[params["source_unit"]]
                target_scale, target_offset = TEMPERATURE_UNITS_V1[params["target_unit"]]
                scale = source_scale / target_scale
                offset = (source_offset - target_offset) / target_scale

                def convert_temperature(value: Any) -> float:
                    exact = Fraction(value.item()) * scale + offset
                    converted = float(exact)
                    if exact and converted == 0:
                        raise TensorusError("Temperature conversion underflowed float64 to zero")
                    return converted

                transformed = np.fromiter(
                    (convert_temperature(value) for value in values.flat),
                    dtype=np.float64,
                    count=values.size,
                )
                if mask is None:
                    result = transformed.reshape(first.shape)
                else:
                    result = np.zeros(first.shape, dtype=np.float64)
                    result[mask] = transformed
            elif operation in CHUNKED_DISPERSIONS:
                axes = reduction_axes(params["axis"], first.ndim)
                masked = operation in MASKED_REDUCTIONS

                def dispersion_blocks() -> Iterator[tuple[tuple[slice, ...], list[Array]]]:
                    itemsize = max(first.dtype.itemsize, CHUNKED_DISPERSION_ITEMSIZE)
                    for selection in chunked_c_order_selections(
                        first.shape, (1,) * first.ndim, itemsize, align_chunks=False
                    ):
                        yield (
                            selection,
                            (
                                [first[selection], inputs[1][selection]]
                                if masked
                                else [first[selection]]
                            ),
                        )

                result = chunked_dispersion(
                    first.shape,
                    axes,
                    params["keepdims"],
                    dispersion_blocks,
                    masked=masked,
                    variance=operation in CHUNKED_VARIANCES,
                    max_working_bytes=settings.max_tensor_bytes,
                )
            elif operation in (
                CHUNKED_SUMS
                | CHUNKED_MEANS
                | CHUNKED_EXTREMA
                | {Operation.chunked_reduce_validity_any}
            ):
                if not np.isfinite(first).all():
                    message = (
                        "Chunked summation requires finite input values"
                        if operation == Operation.chunked_sum_by_axis
                        else "Chunked reduction requires finite input values"
                    )
                    raise TensorusError(message)
                axes = reduction_axes(params["axis"], first.ndim)
                result = np.zeros(predicted.shape, dtype=predicted.dtype)
                if operation in CHUNKED_EXTREMA:
                    result.fill(extrema_identity(first.dtype, operation in CHUNKED_MINIMA))
                included_extrema = (
                    np.zeros(predicted.shape, dtype="bool")
                    if operation in CHUNKED_EXTREMA & MASKED_REDUCTIONS
                    else None
                )
                included_counts = (
                    np.zeros(predicted.shape, dtype=np.int64)
                    if operation == Operation.chunked_masked_mean_with_placeholders
                    else None
                )
                for selection in chunked_c_order_selections(
                    first.shape, (1,) * first.ndim, first.dtype.itemsize, align_chunks=False
                ):
                    if operation == Operation.chunked_reduce_validity_any:
                        accumulate_chunked_any(
                            result, first[selection], selection, axes, params["keepdims"]
                        )
                    elif operation in CHUNKED_EXTREMA:
                        accumulate_chunked_extrema(
                            result,
                            first[selection],
                            selection,
                            axes,
                            params["keepdims"],
                            operation in CHUNKED_MINIMA,
                            inputs[1][selection] if included_extrema is not None else None,
                            included_extrema,
                        )
                    else:
                        mask = inputs[1][selection] if operation in MASKED_REDUCTIONS else None
                        accumulate_chunked_sum(
                            result, first[selection], selection, axes, params["keepdims"], mask
                        )
                        if included_counts is not None:
                            accumulate_chunked_included_count(
                                included_counts,
                                inputs[1][selection],
                                selection,
                                axes,
                                params["keepdims"],
                            )
                if operation in CHUNKED_MEANS:
                    denominator = (
                        included_counts
                        if included_counts is not None
                        else math.prod(first.shape[axis] for axis in axes)
                    )
                    result = mean_from_chunked_sums(result, denominator)
                if included_extrema is not None:
                    result[~included_extrema] = 0
            elif operation in CHUNKED_COUNTS:
                if not np.isfinite(first).all():
                    raise TensorusError("Chunked counting requires finite input values")
                if operation in CHUNKED_MASKED_COUNTS:
                    mask = inputs[1]
                    if operation == Operation.chunked_masked_count_nonzero and not np.any(mask):
                        raise TensorusError(
                            "Masked chunked counting requires at least one included value"
                        )
                    if operation == Operation.chunked_masked_count_nonzero_by_axis:
                        axes = reduction_axes(params["axis"], first.ndim)
                        included = np.count_nonzero(mask, axis=axes, keepdims=params["keepdims"])
                        if np.any(included == 0):
                            raise TensorusError(
                                "Masked chunked counting has an output with no included values"
                            )
                    first = np.logical_and(first, mask)
                if operation in CHUNKED_AXIS_COUNTS:
                    axes = reduction_axes(params["axis"], first.ndim)
                    result = np.asarray(
                        np.count_nonzero(first, axis=axes, keepdims=params["keepdims"]),
                        dtype=np.int64,
                    )
                else:
                    result = np.asarray(np.count_nonzero(first), dtype=np.int64)
            elif operation in REDUCTIONS:
                axes = reduction_axes(params["axis"], first.ndim)
                if operation in MASKED_REDUCTIONS:
                    mask = inputs[1]
                    if operation in (
                        Operation.masked_variance_with_placeholders,
                        Operation.masked_std_with_placeholders,
                    ):
                        if not np.isfinite(first).all():
                            raise TensorusError("Masked reduction requires finite input values")
                        result = stable_masked_std(first, mask, axes, params["keepdims"])
                        if operation == Operation.masked_variance_with_placeholders:
                            variance = np.square(result)
                            if np.any((result != 0) & (variance == 0)):
                                raise TensorusError("Variance underflowed float64")
                            result = variance
                    elif operation in (
                        Operation.masked_min_with_placeholders,
                        Operation.masked_max_with_placeholders,
                    ):
                        if not np.isfinite(first).all():
                            raise TensorusError("Masked reduction requires finite input values")
                        minimum = operation == Operation.masked_min_with_placeholders
                        initial = extrema_identity(first.dtype, minimum)
                        extremum_fn: Callable[..., Any] = np.min if minimum else np.max
                        reduced = extremum_fn(
                            first,
                            axis=axes,
                            keepdims=params["keepdims"],
                            where=mask,
                            initial=initial,
                        )
                        included = np.any(mask, axis=axes, keepdims=params["keepdims"])
                        result = np.where(included, reduced, np.zeros((), dtype=first.dtype))
                    elif operation == Operation.masked_sum_with_placeholders:
                        if not np.isfinite(first).all():
                            raise TensorusError("Masked reduction requires finite input values")
                        result = np.sum(
                            first,
                            axis=axes,
                            keepdims=params["keepdims"],
                            dtype=np.float64,
                            where=mask,
                        )
                    elif operation == Operation.masked_mean_with_placeholders:
                        if not np.isfinite(first).all():
                            raise TensorusError("Masked reduction requires finite input values")
                        if params["algorithm_version"] == 2:
                            result = scaled_masked_mean(first, mask, axes, params["keepdims"])
                        else:
                            counts = np.sum(
                                mask, axis=axes, keepdims=params["keepdims"], dtype=np.int64
                            )
                            totals = np.sum(
                                first,
                                axis=axes,
                                keepdims=params["keepdims"],
                                dtype=np.float64,
                                where=mask,
                            )
                            result = np.divide(
                                totals,
                                counts,
                                out=np.zeros_like(totals, dtype=np.float64),
                                where=counts != 0,
                            )
                            if np.any((totals != 0) & (result == 0)):
                                raise TensorusError("Masked mean underflowed float64 to zero")
                    else:
                        counts = np.sum(
                            mask, axis=axes, keepdims=params["keepdims"], dtype=np.int64
                        )
                        if np.any(counts == 0):
                            raise TensorusError(
                                "Masked reduction has an output with no valid inputs"
                            )
                        result = np.sum(
                            first,
                            axis=axes,
                            keepdims=params["keepdims"],
                            dtype=np.float64,
                            where=mask,
                        )
                        if operation == Operation.masked_mean:
                            mean = np.divide(result, counts)
                            if np.any((result != 0) & (mean == 0)):
                                raise TensorusError("Masked mean underflowed float64 to zero")
                            result = mean
                elif operation == Operation.reduce_validity_any:
                    result = np.any(first, axis=axes, keepdims=params["keepdims"])
                elif operation in (Operation.std, Operation.variance):
                    result = stable_std(first, axes, params["keepdims"])
                    if operation == Operation.variance:
                        variance = np.square(result)
                        if np.any((result != 0) & (variance == 0)):
                            raise TensorusError("Variance underflowed float64")
                        result = variance
                elif operation in (Operation.sum, Operation.mean):
                    reduction_functions: dict[Operation, Callable[..., Any]] = {
                        Operation.sum: np.sum,
                        Operation.mean: np.mean,
                    }
                    result = reduction_functions[operation](
                        first, axis=axes, keepdims=params["keepdims"], dtype=np.float64
                    )
                else:
                    extremum = np.min if operation == Operation.min else np.max
                    result = extremum(first, axis=axes, keepdims=params["keepdims"])
            else:
                raise TensorusError(f"Unsupported operation: {operation}")
    except (ValueError, FloatingPointError, OverflowError) as exc:
        raise TensorusError(f"{operation} failed numerically: {exc}") from exc
    output = validated_array(result, settings.max_tensor_bytes)
    if tuple(output.shape) != predicted.shape or output.dtype.name != predicted.dtype:
        raise TensorusError("Operation result does not match its validated shape/dtype contract")
    return output
