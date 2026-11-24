"""Versioned, caller-declared tensor semantics and conservative propagation."""

import math
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Annotated, Any, Literal, Self

from pydantic import (
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from tensorus.arrays import canonical_json, descriptor_nbytes
from tensorus.errors import LimitExceeded, StorageError, TensorusError
from tensorus.models import (
    MASKED_BINARY_OPERATIONS,
    MAX_COORDINATE_QUERY_COUNT,
    AxisDescriptor,
    AxisSliceRequest,
    CoordinateSliceRequest,
    CoordinateSystemReference,
    CoordinateValues,
    Model,
    Operation,
    PreprocessingIdentityText,
    SemanticFilter,
    Step,
    TimeReference,
    _check_coordinate_float,
)
from tensorus.operations import (
    ARITHMETIC,
    BOUND_SELECTED_PREPROCESSING,
    CHUNKED_AXIS_COUNTS,
    CHUNKED_COUNTS,
    CHUNKED_MASKED_COUNTS,
    MASKED_REDUCTIONS,
    MATRIX_MULTIPLICATION,
    MAX_NEAREST_MAPPING_POSITIONS,
    MAX_SELECTION_INDICES,
    PAIRED_REDUCTION_VALUES,
    PAIRED_SCALING_VALUES,
    PRESERVE_NO_FILL_PREPROCESSING,
    REDUCTIONS,
    VALUE_UNITS_V1,
    NearestReindexAxis,
    ReindexAxis,
    TensorShape,
    parameters,
    reduction_axes,
)

SEMANTICS_KEY = "tensorus_semantics"
SEMANTICS_VERSION: Literal[1] = 1
REFERENCE_SEMANTICS_VERSION: Literal[2] = 2
MAX_AXIS_COORDINATES = MAX_COORDINATE_QUERY_COUNT
MAX_SHARED_AXIS_QUERY_COORDINATE_VALUES = 1_000_000
SHARED_AXIS_QUERY_FORMAT = "tensorus-exact-shared-axis-query-v1"
OVERLAP_SHARED_AXIS_QUERY_FORMAT = "tensorus-overlap-shared-axis-query-v1"
OVERLAP_CROSS_SNAPSHOT_QUERY_FORMAT = "tensorus-overlap-cross-snapshot-shared-axis-query-v1"

_REFERENCE_AXIS_SCHEMA: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "required": [name],
                "properties": {name: {"type": "object"}},
            },
            "then": {"required": ["unit"], "properties": {"unit": {"type": "string"}}},
        }
        for name in ("time_reference", "coordinate_system")
    ]
    + [
        {
            "not": {
                "required": ["time_reference", "coordinate_system"],
                "properties": {
                    "time_reference": {"type": "object"},
                    "coordinate_system": {"type": "object"},
                },
            }
        }
    ]
}


class SemanticAxis(AxisDescriptor):
    model_config = ConfigDict(json_schema_extra=_REFERENCE_AXIS_SCHEMA)
    coordinates: list[StrictInt | StrictFloat] | None = Field(
        default=None, max_length=MAX_AXIS_COORDINATES
    )
    time_reference: TimeReference | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    coordinate_system: CoordinateSystemReference | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, values: list[int | float] | None) -> list[int | float] | None:
        if values is not None:
            return [_check_coordinate_float(value) for value in values]
        return None

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if self.time_reference is not None and self.coordinate_system is not None:
            raise ValueError("An axis cannot specify both reference kinds")
        if (
            self.time_reference is not None or self.coordinate_system is not None
        ) and self.unit is None:
            raise ValueError("A referenced axis requires an explicit unit")
        return self


class ModelIdentity(Model):
    id: Annotated[str, Field(strict=True, min_length=1, max_length=256)]
    version: Annotated[str, Field(strict=True, min_length=1, max_length=256)]

    @field_validator("id", "version")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("Model identity must not have leading or trailing whitespace")
        return value


class PreprocessingIdentity(ModelIdentity):
    """Caller-declared identity of the preprocessing represented by the values."""

    id: PreprocessingIdentityText
    version: PreprocessingIdentityText


class TensorSemantics(Model):
    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "if": {"required": ["version"], "properties": {"version": {"const": 1}}},
                    "then": {
                        "properties": {
                            "axes": {
                                "items": {
                                    "not": {
                                        "anyOf": [
                                            {"required": ["time_reference"]},
                                            {"required": ["coordinate_system"]},
                                        ]
                                    }
                                }
                            }
                        }
                    },
                }
            ]
        }
    )
    version: Literal[1, 2]
    axes: list[SemanticAxis] = Field(max_length=16)
    value_unit: Annotated[str, Field(strict=True, min_length=1, max_length=128)] | None = None
    model: ModelIdentity | None = None
    preprocessing: PreprocessingIdentity | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "Caller-declared preprocessing identity; it is not verified against a pipeline."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def reject_v1_reference_fields(cls, value: object) -> object:
        if isinstance(value, dict) and type(value.get("version")) is int and value["version"] == 1:
            axes = value.get("axes")
            if isinstance(axes, list) and any(
                isinstance(axis, dict) and ("time_reference" in axis or "coordinate_system" in axis)
                for axis in axes
            ):
                raise ValueError("Reference declarations require tensor semantics version 2")
        return value

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        if type(value) is not int or value not in (SEMANTICS_VERSION, REFERENCE_SEMANTICS_VERSION):
            raise ValueError("Unsupported tensor semantics version")
        return value

    @model_validator(mode="after")
    def validate_reference_version(self) -> Self:
        if self.version == SEMANTICS_VERSION and any(
            axis.time_reference is not None or axis.coordinate_system is not None
            for axis in self.axes
        ):
            raise ValueError("Reference declarations require tensor semantics version 2")
        return self

    @field_validator("value_unit")
    @classmethod
    def validate_unit(cls, value: str | None) -> str | None:
        return AxisDescriptor.validate_unit(value)


class CoordinateOverlap(Model):
    label: str
    left_count: int
    right_count: int
    retained_count: int
    left_dropped: int
    right_dropped: int


def read_semantics(
    metadata: dict[str, Any], shape: tuple[int, ...], operation: dict[str, Any] | None = None
) -> TensorSemantics | None:
    if SEMANTICS_KEY not in metadata:
        return None
    if "tensorus_axes" in metadata or "tensorus_coordinates" in metadata:
        raise TensorusError("Versioned semantics cannot be combined with legacy axis metadata")
    try:
        result = TensorSemantics.model_validate(metadata[SEMANTICS_KEY], strict=True)
    except ValidationError as exc:
        raise TensorusError(
            "Invalid tensor semantics descriptor",
            details=exc.errors(include_input=False, include_context=False),
        ) from exc
    if operation is not None:
        version = operation.get("semantics_version")
        if type(version) is not int or version != result.version:
            raise TensorusError(
                "Derived tensor semantics were not validated; "
                "explicitly redeclare them in a new version"
            )
    labels = [axis.label for axis in result.axes]
    if len(labels) != len(shape) or len(set(labels)) != len(labels):
        raise TensorusError("Tensor semantics require one unique axis label per dimension")
    for axis, dimension in zip(result.axes, shape, strict=True):
        values = axis.coordinates
        if values is not None and (
            len(values) != dimension
            or any(isinstance(value, float) and not math.isfinite(value) for value in values)
            or any(left >= right for left, right in pairwise(values))
        ):
            raise TensorusError(
                "Axis coordinates must match their dimension and be finite and strictly increasing"
            )
    return result


def require_policy(
    policy: Literal["v1", "v2"] | None, inputs: list[TensorSemantics | None]
) -> None:
    if policy is None and any(item is not None for item in inputs):
        raise TensorusError("Versioned tensor semantics require semantic_policy='v1' or 'v2'")
    if policy is not None:
        version = SEMANTICS_VERSION if policy == "v1" else REFERENCE_SEMANTICS_VERSION
        if any(item is None for item in inputs):
            raise TensorusError(
                f"semantic_policy='{policy}' requires declared semantics for every input"
            )
        if any(item is not None and item.version != version for item in inputs):
            raise TensorusError(f"semantic_policy='{policy}' requires version {version} semantics")


def _require_same_version(left: TensorSemantics, right: TensorSemantics) -> None:
    if left.version != right.version:
        raise TensorusError("Semantic descriptor versions must match; conversion is not inferred")


def _aligned_axis(
    left: SemanticAxis, right: SemanticAxis, left_size: int, right_size: int, *, exact: bool = False
) -> None:
    if left_size != right_size:
        raise TensorusError("Explicit singleton-axis broadcasting is semantically ambiguous")
    if left.label != right.label or left.unit != right.unit:
        raise TensorusError("Shared axis labels and units must match exactly")
    if left.unit is None:
        raise TensorusError("Shared axes require explicit units; use '1' for dimensionless axes")
    if left.coordinates is None or right.coordinates is None:
        raise TensorusError("Shared axes require explicit coordinates for semantic alignment")
    if (
        left.time_reference != right.time_reference
        or left.coordinate_system != right.coordinate_system
    ):
        raise TensorusError("Shared axis reference identities must match exactly")
    if (
        canonical_json(left.coordinates) != canonical_json(right.coordinates)
        if exact
        else left.coordinates != right.coordinates
    ):
        raise TensorusError("Shared axis coordinates do not match")


def require_exact_shared_axis(
    left: TensorSemantics,
    right: TensorSemantics,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    left_label: str,
    right_label: str,
) -> None:
    if left.version != 2 or right.version != 2:
        raise TensorusError("Shared-axis relationships require version 2 semantics")
    left_index = next((i for i, axis in enumerate(left.axes) if axis.label == left_label), None)
    right_index = next((i for i, axis in enumerate(right.axes) if axis.label == right_label), None)
    if left_index is None or right_index is None:
        raise TensorusError("Shared-axis relationship names an unavailable axis")
    _aligned_axis(
        left.axes[left_index],
        right.axes[right_index],
        left_shape[left_index],
        right_shape[right_index],
        exact=True,
    )


@dataclass
class SharedAxisQueryMatcher:
    anchor: TensorSemantics
    anchor_shape: tuple[int, ...]
    axes: tuple[str, ...]
    anchor_fingerprint: str | None = None
    anchor_scope: tuple[str, str] | None = None
    across_datasets: bool = False
    overlap: bool = False
    compared_coordinate_values: int = 0

    def __post_init__(self) -> None:
        for label in self.axes:
            require_exact_shared_axis(
                self.anchor, self.anchor, self.anchor_shape, self.anchor_shape, label, label
            )

    def matches(self, candidate: TensorSemantics, candidate_shape: tuple[int, ...]) -> bool:
        if candidate.version != 2:
            return False
        for label in self.axes:
            left = next(axis for axis in self.anchor.axes if axis.label == label)
            right = next((axis for axis in candidate.axes if axis.label == label), None)
            if right is None:
                return False
            self.compared_coordinate_values += len(left.coordinates or ()) + len(
                right.coordinates or ()
            )
            if self.compared_coordinate_values > MAX_SHARED_AXIS_QUERY_COORDINATE_VALUES:
                raise LimitExceeded("Shared-axis discovery exceeds its coordinate comparison limit")
            if self.overlap:
                if (
                    left.unit != right.unit
                    or left.time_reference != right.time_reference
                    or left.coordinate_system != right.coordinate_system
                    or left.coordinates is None
                    or right.coordinates is None
                    or not any(
                        position is not None
                        for position in coordinate_reindex_positions(
                            left.coordinates, right.coordinates, exact=True
                        )
                    )
                ):
                    return False
                continue
            try:
                require_exact_shared_axis(
                    self.anchor,
                    candidate,
                    self.anchor_shape,
                    candidate_shape,
                    label,
                    label,
                )
            except TensorusError:
                return False
        return True


def require_matching_axes(
    left: TensorSemantics, right: TensorSemantics, shape: tuple[int, ...]
) -> None:
    _require_same_version(left, right)
    for left_axis, right_axis, size in zip(left.axes, right.axes, shape, strict=True):
        _aligned_axis(left_axis, right_axis, size, size, exact=left.version == 2)


def _broadcast_semantic_axes(
    left_axes: list[SemanticAxis],
    right_axes: list[SemanticAxis],
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    *,
    exact: bool = False,
) -> list[SemanticAxis]:
    rank = max(len(left_shape), len(right_shape))
    axes: list[SemanticAxis] = []
    for index in range(rank):
        left_index = index - (rank - len(left_shape))
        right_index = index - (rank - len(right_shape))
        if left_index < 0:
            axes.append(right_axes[right_index].model_copy(deep=True))
        elif right_index < 0:
            axes.append(left_axes[left_index].model_copy(deep=True))
        else:
            _aligned_axis(
                left_axes[left_index],
                right_axes[right_index],
                left_shape[left_index],
                right_shape[right_index],
                exact=exact,
            )
            axes.append(left_axes[left_index].model_copy(deep=True))
    return axes


def _named_axis_permutation(
    left: TensorSemantics,
    right: TensorSemantics,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> list[int]:
    _require_same_version(left, right)
    if len(left.axes) != len(left_shape) or len(right.axes) != len(right_shape):
        raise TensorusError("Tensor axes do not match their dimensions")
    right_positions = {axis.label: index for index, axis in enumerate(right.axes)}
    if len(left.axes) != len(right.axes) or {axis.label for axis in left.axes} != set(
        right_positions
    ):
        raise TensorusError("Alignment requires the same named axes on both tensors")
    return [right_positions[axis.label] for axis in left.axes]


def alignment_permutation(
    left: TensorSemantics,
    right: TensorSemantics,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> list[int]:
    permutation = _named_axis_permutation(left, right, left_shape, right_shape)
    for index, right_index in enumerate(permutation):
        _aligned_axis(
            left.axes[index],
            right.axes[right_index],
            left_shape[index],
            right_shape[right_index],
            exact=left.version == 2,
        )
    return permutation


def _coordinate_slice(indices: list[int]) -> list[int]:
    if not indices:
        raise TensorusError("Coordinate intersection is empty")
    stride = indices[1] - indices[0] if len(indices) > 1 else 1
    if any(right - left != stride for left, right in pairwise(indices)):
        raise TensorusError("Irregular coordinate positions cannot use a single strided slice")
    return [indices[0], indices[-1] + 1, stride]


def _coordinate_intersection_positions(
    left: TensorSemantics,
    right: TensorSemantics,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    *,
    exact_references: bool = False,
) -> tuple[list[int], list[list[int]], list[list[int]], list[CoordinateOverlap]]:
    if exact_references:
        if left.version != REFERENCE_SEMANTICS_VERSION or right.version != (
            REFERENCE_SEMANTICS_VERSION
        ):
            raise TensorusError("Typed coordinate intersection requires semantics version 2")
    elif left.version == REFERENCE_SEMANTICS_VERSION or right.version == (
        REFERENCE_SEMANTICS_VERSION
    ):
        raise TensorusError("Coordinate intersection is not supported for semantics version 2")
    permutation = _named_axis_permutation(left, right, left_shape, right_shape)
    left_selection: list[list[int]] = []
    right_selection: list[list[int]] = []
    overlaps: list[CoordinateOverlap] = []
    for index, right_index in enumerate(permutation):
        first, second = left.axes[index], right.axes[right_index]
        if first.unit is None or second.unit is None:
            raise TensorusError(
                "Shared axes require explicit units; use '1' for dimensionless axes"
            )
        if first.unit != second.unit:
            raise TensorusError("Shared axis labels and units must match exactly")
        if exact_references and (
            first.time_reference != second.time_reference
            or first.coordinate_system != second.coordinate_system
        ):
            raise TensorusError("Shared axis reference identities must match exactly")
        if first.coordinates is None or second.coordinates is None:
            raise TensorusError("Shared axes require explicit coordinates for semantic alignment")
        left_indices, right_indices = [], []
        left_index, right_position = 0, 0
        while left_index < len(first.coordinates) and right_position < len(second.coordinates):
            left_value, right_value = (
                first.coordinates[left_index],
                second.coordinates[right_position],
            )
            if left_value == right_value:
                if not exact_references or canonical_json(left_value) == canonical_json(
                    right_value
                ):
                    left_indices.append(left_index)
                    right_indices.append(right_position)
                left_index += 1
                right_position += 1
            elif left_value < right_value:
                left_index += 1
            else:
                right_position += 1
        if not left_indices:
            raise TensorusError("Coordinate intersection is empty")
        left_selection.append(left_indices)
        right_selection.append(right_indices)
        overlaps.append(
            CoordinateOverlap(
                label=first.label,
                left_count=left_shape[index],
                right_count=right_shape[right_index],
                retained_count=len(left_indices),
                left_dropped=left_shape[index] - len(left_indices),
                right_dropped=right_shape[right_index] - len(right_indices),
            )
        )
    return permutation, left_selection, right_selection, overlaps


def coordinate_intersection(
    left: TensorSemantics,
    right: TensorSemantics,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[list[int], list[list[int]], list[list[int]], list[CoordinateOverlap]]:
    return _strided_coordinate_intersection(left, right, left_shape, right_shape)


def referenced_coordinate_intersection(
    left: TensorSemantics,
    right: TensorSemantics,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[list[int], list[list[int]], list[list[int]], list[CoordinateOverlap]]:
    if not left_shape or not right_shape:
        raise TensorusError("Typed coordinate intersection requires at least one axis")
    return _strided_coordinate_intersection(
        left, right, left_shape, right_shape, exact_references=True
    )


def _strided_coordinate_intersection(
    left: TensorSemantics,
    right: TensorSemantics,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    *,
    exact_references: bool = False,
) -> tuple[list[int], list[list[int]], list[list[int]], list[CoordinateOverlap]]:
    permutation, left_positions, right_positions, overlaps = _coordinate_intersection_positions(
        left, right, left_shape, right_shape, exact_references=exact_references
    )
    return (
        permutation,
        [_coordinate_slice(indices) for indices in left_positions],
        [_coordinate_slice(indices) for indices in right_positions],
        overlaps,
    )


def coordinate_index_intersection(
    left: TensorSemantics,
    right: TensorSemantics,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[list[int], list[list[int] | None], list[list[int] | None], list[CoordinateOverlap]]:
    permutation, left_positions, right_positions, overlaps = _coordinate_intersection_positions(
        left, right, left_shape, right_shape
    )
    return _bounded_intersection_selections(
        permutation, left_positions, right_positions, overlaps, left_shape, right_shape
    )


def referenced_coordinate_index_intersection(
    left: TensorSemantics,
    right: TensorSemantics,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[list[list[int] | None], list[list[int] | None], list[CoordinateOverlap]]:
    permutation, left_selection, right_selection, overlaps = referenced_coordinate_index_alignment(
        left, right, left_shape, right_shape
    )
    if permutation != list(range(len(permutation))):
        raise TensorusError("Typed coordinate intersection requires the same axis order")
    return left_selection, right_selection, overlaps


def referenced_coordinate_index_alignment(
    left: TensorSemantics,
    right: TensorSemantics,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[list[int], list[list[int] | None], list[list[int] | None], list[CoordinateOverlap]]:
    if not left_shape or not right_shape:
        raise TensorusError("Typed coordinate intersection requires at least one axis")
    permutation, left_positions, right_positions, overlaps = _coordinate_intersection_positions(
        left, right, left_shape, right_shape, exact_references=True
    )
    return _bounded_intersection_selections(
        permutation, left_positions, right_positions, overlaps, left_shape, right_shape
    )


def _bounded_intersection_selections(
    permutation: list[int],
    left_positions: list[list[int]],
    right_positions: list[list[int]],
    overlaps: list[CoordinateOverlap],
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[list[int], list[list[int] | None], list[list[int] | None], list[CoordinateOverlap]]:
    left_selection = [
        None if len(indices) == dimension else indices
        for indices, dimension in zip(left_positions, left_shape, strict=True)
    ]
    right_selection = [
        None if len(indices) == right_shape[right_index] else indices
        for indices, right_index in zip(right_positions, permutation, strict=True)
    ]
    for selection in (left_selection, right_selection):
        if (
            sum(len(indices) for indices in selection if indices is not None)
            > MAX_SELECTION_INDICES
        ):
            raise LimitExceeded("Indexed coordinate intersection exceeds the index limit")
    return permutation, left_selection, right_selection, overlaps


def coordinate_reindex_positions(
    source: list[int | float], target: list[int | float], *, exact: bool = False
) -> list[int | None]:
    positions: list[int | None] = []
    current = 0
    for coordinate in target:
        while current < len(source) and source[current] < coordinate:
            current += 1
        if (
            current < len(source)
            and source[current] == coordinate
            and (not exact or canonical_json(source[current]) == canonical_json(coordinate))
        ):
            positions.append(current)
            current += 1
        else:
            positions.append(None)
    return positions


def coordinate_nearest_integer_positions(
    source: Sequence[int | float], target: Sequence[int | float], max_distance: int
) -> list[int | None]:
    if type(max_distance) is not int or not 0 <= max_distance <= 2**53 - 1:
        raise TensorusError("Nearest-coordinate distance must be a bounded nonnegative integer")
    if any(
        type(value) is not int or not -(2**63) <= value <= 2**63 - 1 for value in (*source, *target)
    ):
        raise TensorusError("Nearest-coordinate mapping requires signed 64-bit integer coordinates")
    positions: list[int | None] = []
    current = 0
    for coordinate in target:
        if not source:
            positions.append(None)
            continue
        while current + 1 < len(source) and source[current + 1] <= coordinate:
            current += 1
        candidate = current
        if current + 1 < len(source) and abs(source[current + 1] - coordinate) < abs(
            source[current] - coordinate
        ):
            candidate += 1
        positions.append(candidate if abs(source[candidate] - coordinate) <= max_distance else None)
    return positions


class CoordinateFillAxis(Model):
    label: str
    target_count: int
    matched_count: int
    missing_count: int
    source_dropped: int


def coordinate_filled_alignment(
    left: TensorSemantics,
    right: TensorSemantics,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    anchor: Literal["left", "right"],
) -> tuple[list[int], list[ReindexAxis | None], list[CoordinateFillAxis]]:
    if left.version == REFERENCE_SEMANTICS_VERSION or right.version == REFERENCE_SEMANTICS_VERSION:
        raise TensorusError("Coordinate reindexing is not supported for semantics version 2")
    return _coordinate_filled_alignment(left, right, left_shape, right_shape, anchor, exact=False)


def referenced_coordinate_filled_alignment(
    left: TensorSemantics,
    right: TensorSemantics,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    anchor: Literal["left", "right"],
) -> tuple[list[int], list[ReindexAxis | None], list[CoordinateFillAxis]]:
    if left.version != REFERENCE_SEMANTICS_VERSION or right.version != REFERENCE_SEMANTICS_VERSION:
        raise TensorusError("Typed filled alignment requires semantics version 2")
    return _coordinate_filled_alignment(left, right, left_shape, right_shape, anchor, exact=True)


def referenced_nearest_coordinate_alignment(
    source: TensorSemantics,
    anchor: TensorSemantics,
    source_shape: tuple[int, ...],
    anchor_shape: tuple[int, ...],
    max_distance: int,
) -> tuple[list[int], list[NearestReindexAxis | None]]:
    if (
        source.version != REFERENCE_SEMANTICS_VERSION
        or anchor.version != REFERENCE_SEMANTICS_VERSION
    ):
        raise TensorusError("Nearest-coordinate mapping requires semantics version 2")
    permutation = _named_axis_permutation(source, anchor, source_shape, anchor_shape)
    mappings: list[NearestReindexAxis | None] = []
    mapping_positions = 0
    target_positions = 0
    for source_index, anchor_index in enumerate(permutation):
        first, second = source.axes[source_index], anchor.axes[anchor_index]
        _require_filled_axis_compatibility(first, second, exact=True)
        assert first.coordinates is not None and second.coordinates is not None
        target_positions += len(second.coordinates)
        if target_positions > MAX_SELECTION_INDICES:
            raise LimitExceeded("Coordinate reindexing exceeds the total position limit")
        if canonical_json(first.coordinates) == canonical_json(second.coordinates):
            mappings.append(None)
            continue
        mapping_positions += len(first.coordinates) + len(second.coordinates)
        if mapping_positions > MAX_NEAREST_MAPPING_POSITIONS:
            raise LimitExceeded("Nearest-coordinate mapping exceeds the position-work limit")
        indices = coordinate_nearest_integer_positions(
            first.coordinates, second.coordinates, max_distance
        )
        mappings.append(
            NearestReindexAxis.model_validate(
                {"coordinates": second.coordinates, "indices": indices}
            )
        )
    return permutation, mappings


def _require_filled_axis_compatibility(
    first: SemanticAxis, second: SemanticAxis, *, exact: bool
) -> None:
    if first.unit is None or second.unit is None:
        raise TensorusError("Shared axes require explicit units; use '1' for dimensionless axes")
    if first.unit != second.unit:
        raise TensorusError("Shared axis labels and units must match exactly")
    if exact and (
        first.time_reference != second.time_reference
        or first.coordinate_system != second.coordinate_system
    ):
        raise TensorusError("Shared axis reference identities must match exactly")
    if first.coordinates is None or second.coordinates is None:
        raise TensorusError("Shared axes require explicit coordinates for semantic alignment")


def _coordinate_filled_alignment(
    left: TensorSemantics,
    right: TensorSemantics,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    anchor: Literal["left", "right"],
    *,
    exact: bool,
) -> tuple[list[int], list[ReindexAxis | None], list[CoordinateFillAxis]]:
    permutation = _named_axis_permutation(left, right, left_shape, right_shape)
    mappings: list[ReindexAxis | None] = []
    summaries: list[CoordinateFillAxis] = []
    target_positions = 0
    for left_index, right_index in enumerate(permutation):
        first, second = left.axes[left_index], right.axes[right_index]
        _require_filled_axis_compatibility(first, second, exact=exact)
        assert first.coordinates is not None and second.coordinates is not None
        target = first.coordinates if anchor == "left" else second.coordinates
        source = second.coordinates if anchor == "left" else first.coordinates
        target_positions += len(target)
        if target_positions > MAX_SELECTION_INDICES:
            raise LimitExceeded("Coordinate reindexing exceeds the total position limit")
        indices = coordinate_reindex_positions(source, target, exact=exact)
        matched = sum(index is not None for index in indices)
        summaries.append(
            CoordinateFillAxis(
                label=first.label,
                target_count=len(target),
                matched_count=matched,
                missing_count=len(target) - matched,
                source_dropped=len(source) - matched,
            )
        )
        same = canonical_json(source) == canonical_json(target) if exact else source == target
        mappings.append(None if same else ReindexAxis(coordinates=list(target), indices=indices))
    return permutation, mappings, summaries


def _model_identity(
    inputs: list[TensorSemantics], shapes: list[TensorShape]
) -> ModelIdentity | None:
    identities = [
        item.model
        for item, shape in zip(inputs, shapes, strict=True)
        if shape.shape or item.model is not None
    ]
    if identities and any(identity != identities[0] for identity in identities[1:]):
        raise TensorusError("Tensor model identities or versions do not match")
    return identities[0] if identities else None


def _require_preprocessing_identity(
    inputs: list[TensorSemantics], shapes: list[TensorShape]
) -> None:
    identities = [
        item.preprocessing
        for item, shape in zip(inputs, shapes, strict=True)
        if shape.shape or item.preprocessing is not None
    ]
    if identities and any(identity != identities[0] for identity in identities[1:]):
        raise TensorusError("Tensor preprocessing identities or versions do not match")


def _value_units(operation: Operation, left: TensorSemantics, right: TensorSemantics) -> str | None:
    first, second = left.value_unit, right.value_unit
    if first is None or second is None:
        raise TensorusError("Binary tensor operations require explicit value units")
    if operation in (Operation.add, Operation.subtract):
        if first != second:
            raise TensorusError("Value units must match exactly for addition and subtraction")
        return first
    if second == "1":
        return first
    if operation == Operation.multiply and first == "1":
        return second
    return None


REFERENCE_OPERATIONS = frozenset(
    {
        Operation.profile,
        Operation.transpose,
        Operation.slice,
        Operation.chunked_slice,
        Operation.select_indices,
        Operation.reindex_coordinates,
        Operation.reindex_validity,
        Operation.coordinate_coverage,
        Operation.cast,
        Operation.masked_cast_with_placeholders,
        Operation.masked_scalar_arithmetic_with_placeholders,
        Operation.fit_selected_preprocessing,
        Operation.apply_fitted_preprocessing,
        Operation.convert_axis_unit,
        Operation.convert_value_unit,
        Operation.masked_convert_value_unit,
        Operation.convert_temperature_unit,
        Operation.masked_convert_temperature_unit,
        Operation.reshape,
        Operation.add,
        Operation.subtract,
        Operation.multiply,
        Operation.divide,
        Operation.logical_and,
        Operation.normalize,
        Operation.standardize,
        Operation.sum,
        Operation.mean,
        Operation.min,
        Operation.max,
        Operation.variance,
        Operation.std,
        Operation.masked_sum,
        Operation.masked_mean,
        Operation.chunked_sum_by_axis,
        Operation.chunked_mean_by_axis,
        Operation.chunked_min_by_axis,
        Operation.chunked_max_by_axis,
        Operation.chunked_variance_by_axis,
        Operation.chunked_std_by_axis,
    }
    | PAIRED_REDUCTION_VALUES
    | PAIRED_SCALING_VALUES
    | {Operation.reduce_validity_any, Operation.chunked_reduce_validity_any}
    | MASKED_BINARY_OPERATIONS.keys()
)

PREPROCESSING_PRESERVING_OPERATIONS = frozenset(
    {
        Operation.transpose,
        Operation.slice,
        Operation.chunked_slice,
        Operation.select_indices,
        Operation.reshape,
        Operation.convert_axis_unit,
    }
)


def _invalidate_reduced_axis(axis: SemanticAxis, version: int) -> None:
    axis.coordinates = None
    if version == REFERENCE_SEMANTICS_VERSION:
        axis.time_reference = None
        axis.coordinate_system = None


def require_fitted_domain(
    training: TensorSemantics,
    target: TensorSemantics,
    training_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    axis: int | list[int] | None,
) -> None:
    if training.version != 2 or target.version != 2 or len(training_shape) != len(target_shape):
        raise TensorusError("Fitted preprocessing requires matching v2 tensor ranks")
    if (
        training.value_unit != target.value_unit
        or training.model != target.model
        or training.preprocessing != target.preprocessing
    ):
        raise TensorusError("Fitted preprocessing value units or declared identities do not match")
    reduced = reduction_axes(axis, len(training_shape))
    for index, (left, right) in enumerate(zip(training.axes, target.axes, strict=True)):
        if index not in reduced:
            _aligned_axis(left, right, training_shape[index], target_shape[index], exact=True)
        elif (
            left.label != right.label
            or left.unit is None
            or left.unit != right.unit
            or left.coordinates is None
            or right.coordinates is None
            or left.time_reference != right.time_reference
            or left.coordinate_system != right.coordinate_system
        ):
            raise TensorusError("Fitted preprocessing reduced-axis declarations do not match")


def derive_semantics(
    step: Step,
    shapes: list[TensorShape],
    semantics: list[TensorSemantics | None],
    output: TensorShape | None,
    policy: Literal["v1", "v2"] | None,
    *,
    selected_preprocessing: tuple[PreprocessingIdentity, PreprocessingIdentity] | None = None,
) -> TensorSemantics | None:
    if policy == "v2" and step.operation not in REFERENCE_OPERATIONS:
        raise TensorusError(
            f"No semantic_policy='v2' propagation rule is available for {step.operation}"
        )
    if step.operation in (
        Operation.convert_value_unit,
        Operation.masked_convert_value_unit,
        Operation.convert_temperature_unit,
        Operation.masked_convert_temperature_unit,
    ) and policy not in ("v1", "v2"):
        raise TensorusError("Value-unit conversion requires semantic_policy='v1' or 'v2'")
    if step.operation == Operation.convert_axis_unit and policy not in ("v1", "v2"):
        raise TensorusError("Axis-unit conversion requires semantic_policy='v1' or 'v2'")
    if step.operation in (Operation.reindex_coordinates, Operation.reindex_validity):
        if policy not in ("v1", "v2"):
            raise TensorusError("Coordinate reindexing requires semantic_policy='v1' or 'v2'")
        if policy == "v2" and len(shapes) != 2:
            raise TensorusError("V2 coordinate reindexing requires an explicit anchor")
    if step.operation == Operation.coordinate_coverage:
        if policy not in ("v1", "v2"):
            raise TensorusError("Coordinate coverage requires semantic_policy='v1' or 'v2'")
        if policy == "v2" and len(shapes) != 2:
            raise TensorusError("V2 coordinate coverage requires an explicit anchor")
    if step.operation in MASKED_REDUCTIONS and policy is None:
        raise TensorusError("Masked reductions require semantic_policy='v1' or 'v2'")
    if step.operation in PAIRED_SCALING_VALUES and policy != "v2":
        raise TensorusError("Selected scaling requires semantic_policy='v2'")
    if step.operation == Operation.masked_cast_with_placeholders and policy != "v2":
        raise TensorusError("Selected casting requires semantic_policy='v2'")
    if step.operation == Operation.masked_scalar_arithmetic_with_placeholders and policy != "v2":
        raise TensorusError("Selected scalar arithmetic requires semantic_policy='v2'")
    if (
        step.operation
        in (
            Operation.fit_selected_preprocessing,
            Operation.apply_fitted_preprocessing,
        )
        and policy != "v2"
    ):
        raise TensorusError("Fitted preprocessing requires semantic_policy='v2'")
    if step.operation in MASKED_BINARY_OPERATIONS and policy not in ("v1", "v2"):
        kind = (
            "addition" if step.operation == Operation.masked_add_with_placeholders else "arithmetic"
        )
        raise TensorusError(f"Masked {kind} requires semantic_policy='v1' or 'v2'")
    if step.operation in MASKED_BINARY_OPERATIONS:
        selected_policy = parameters(step).get("selected_preprocessing_policy")
        if selected_policy == BOUND_SELECTED_PREPROCESSING:
            if policy != "v2" or selected_preprocessing is None:
                raise TensorusError(
                    "Selected preprocessing requires a reviewed v2 paired-validity graph"
                )
        elif selected_preprocessing is not None:
            raise TensorusError("Selected preprocessing was not requested by the operation")
    if step.operation in (
        Operation.reduce_validity_any,
        Operation.chunked_reduce_validity_any,
    ) and policy not in ("v1", "v2"):
        raise TensorusError("Validity reduction requires semantic_policy='v1' or 'v2'")
    if step.operation in CHUNKED_MASKED_COUNTS and policy not in ("v1", "v2"):
        raise TensorusError("Masked chunked counting requires semantic_policy='v1' or 'v2'")
    require_policy(policy, semantics)
    if policy is None or output is None:
        return None
    typed = [item for item in semantics if item is not None]
    first = typed[0]
    axes = [axis.model_copy(deep=True) for axis in first.axes]
    unit, model = first.value_unit, first.model
    params = parameters(step)
    operation = step.operation
    preprocessing = (
        first.preprocessing if operation in PREPROCESSING_PRESERVING_OPERATIONS else None
    )

    if operation in MASKED_BINARY_OPERATIONS:
        require_matching_axes(first, typed[1], shapes[0].shape)
        require_matching_axes(first, typed[2], shapes[0].shape)
        model = _model_identity(typed[:2], shapes[:2])
        if selected_preprocessing is None:
            _require_preprocessing_identity(typed[:2], shapes[:2])
        elif selected_preprocessing[0] != selected_preprocessing[1] or any(
            item.preprocessing is not None and item.preprocessing != selected_preprocessing[index]
            for index, item in enumerate(typed[:2])
        ):
            raise TensorusError("Selected source preprocessing identities do not match")
        unit = _value_units(MASKED_BINARY_OPERATIONS[operation], first, typed[1])
    elif operation in ARITHMETIC and len(typed) == 1:
        if operation in (Operation.add, Operation.subtract) and unit != "1":
            raise TensorusError(
                "Scalar addition or subtraction requires dimensionless values; "
                "use a tensor with an explicitly matching unit"
            )
    elif operation in ARITHMETIC:
        second = typed[1]
        model = _model_identity(typed, shapes)
        _require_preprocessing_identity(typed, shapes)
        unit = _value_units(operation, first, second)
        left_shape, right_shape = shapes[0].shape, shapes[1].shape
        axes = _broadcast_semantic_axes(
            first.axes, second.axes, left_shape, right_shape, exact=first.version == 2
        )
    elif operation == Operation.logical_and:
        second = typed[1]
        axes = _broadcast_semantic_axes(
            first.axes, second.axes, shapes[0].shape, shapes[1].shape, exact=first.version == 2
        )
        unit, model = None, None
    elif operation in MATRIX_MULTIPLICATION:
        second = typed[1]
        model = _model_identity(typed, shapes)
        _require_preprocessing_identity(typed, shapes)
        unit = _value_units(Operation.multiply, first, second)
        left_shape, right_shape = shapes[0].shape, shapes[1].shape
        _aligned_axis(
            first.axes[-1],
            second.axes[-2],
            left_shape[-1],
            right_shape[-2],
            exact=first.version == 2,
        )
        axes = _broadcast_semantic_axes(
            first.axes[:-2],
            second.axes[:-2],
            left_shape[:-2],
            right_shape[:-2],
            exact=first.version == 2,
        )
        axes.extend((first.axes[-2].model_copy(deep=True), second.axes[-1].model_copy(deep=True)))
    elif operation == Operation.transpose:
        order = params["axes"]
        if order is None:
            order = list(reversed(range(len(axes))))
        axes = [axes[index] for index in order]
    elif operation in (Operation.slice, Operation.chunked_slice):
        for axis, selection in zip(axes, params["selection"], strict=False):
            if axis.coordinates is not None:
                axis.coordinates = axis.coordinates[slice(*selection)]
    elif operation == Operation.select_indices:
        for axis, indices in zip(axes, params["indices"], strict=True):
            if axis.coordinates is not None and indices is not None:
                axis.coordinates = [axis.coordinates[index] for index in indices]
    elif operation in (
        Operation.reindex_coordinates,
        Operation.reindex_validity,
        Operation.coordinate_coverage,
    ):
        if params.get("preprocessing_policy") == PRESERVE_NO_FILL_PREPROCESSING:
            preprocessing = first.preprocessing
        nearest = params.get("nearest")
        if nearest is not None and (first.version != 2 or len(typed) != 2):
            raise TensorusError("Nearest-coordinate mapping requires v2 semantics and an anchor")
        mapping_positions = 0
        for axis, mapping in zip(axes, params["axes"], strict=True):
            if mapping is not None:
                if axis.coordinates is None or axis.unit is None:
                    raise TensorusError(
                        "Coordinate reindexing requires declared coordinates and units"
                    )
                if nearest is None:
                    expected = coordinate_reindex_positions(
                        axis.coordinates, mapping["coordinates"], exact=first.version == 2
                    )
                else:
                    mapping_positions += len(axis.coordinates) + len(mapping["coordinates"])
                    if mapping_positions > MAX_NEAREST_MAPPING_POSITIONS:
                        raise LimitExceeded(
                            "Nearest-coordinate mapping exceeds the position-work limit"
                        )
                    expected = coordinate_nearest_integer_positions(
                        axis.coordinates, mapping["coordinates"], nearest["max_distance"]
                    )
                if mapping["indices"] != expected:
                    raise TensorusError("Reindex positions do not match the declared coordinates")
                axis.coordinates = mapping["coordinates"]
        if len(typed) == 2:
            for axis, anchor, dimension in zip(axes, typed[1].axes, output.shape, strict=True):
                _aligned_axis(axis, anchor, dimension, dimension, exact=first.version == 2)
        if operation == Operation.coordinate_coverage:
            unit, model = None, None
    elif operation == Operation.reshape:
        if output.shape != shapes[0].shape:
            raise TensorusError("Shape-changing reshape needs an explicit semantic mapping")
    elif operation in CHUNKED_COUNTS:
        if operation in CHUNKED_MASKED_COUNTS:
            require_matching_axes(first, typed[1], shapes[0].shape)
        if operation in CHUNKED_AXIS_COUNTS:
            reduced = set(reduction_axes(params["axis"], len(axes)))
            if params["keepdims"]:
                for index in reduced:
                    _invalidate_reduced_axis(axes[index], first.version)
            else:
                axes = [axis for index, axis in enumerate(axes) if index not in reduced]
        else:
            axes = []
        unit, model = "1", None
    elif operation in REDUCTIONS:
        if operation in MASKED_REDUCTIONS:
            require_matching_axes(first, typed[1], shapes[0].shape)
        reduced = set(reduction_axes(params["axis"], len(axes)))
        if params["keepdims"]:
            for index in reduced:
                _invalidate_reduced_axis(axes[index], first.version)
        else:
            axes = [axis for index, axis in enumerate(axes) if index not in reduced]
        if (
            operation
            in (
                Operation.variance,
                Operation.masked_variance_with_placeholders,
                Operation.chunked_variance_by_axis,
                Operation.chunked_masked_variance_with_placeholders,
            )
            and unit != "1"
        ):
            unit = None
        if operation in (Operation.reduce_validity_any, Operation.chunked_reduce_validity_any):
            unit, model = None, None
    elif operation in {Operation.normalize, Operation.standardize} | PAIRED_SCALING_VALUES:
        if operation in PAIRED_SCALING_VALUES:
            require_matching_axes(first, typed[1], shapes[0].shape)
        unit = "1"
    elif operation == Operation.fit_selected_preprocessing:
        require_matching_axes(first, typed[1], shapes[0].shape)
        reduced = set(reduction_axes(params["axis"], len(axes)))
        retained_axes = [axis for index, axis in enumerate(axes) if index not in reduced]
        labels = {axis.label for axis in retained_axes}
        label = "fitted_parameter"
        while label in labels:
            label += "_"
        coordinates: list[int | float] = list(range(2 if params["algorithm_version"] == 1 else 3))
        axes = [SemanticAxis(label=label, unit="1", coordinates=coordinates), *retained_axes]
    elif operation == Operation.apply_fitted_preprocessing:
        state = typed[1]
        require_matching_axes(first, typed[2], shapes[0].shape)
        if state.value_unit != first.value_unit or state.model != first.model:
            raise TensorusError("Fitted state value unit or model does not match the input")
        reduced = set(reduction_axes(params["axis"], len(axes)))
        retained_with_size = [
            (axis, shapes[0].shape[index])
            for index, axis in enumerate(axes)
            if index not in reduced
        ]
        if len(state.axes) != len(retained_with_size) + 1:
            raise TensorusError("Fitted state axes do not match the transform domain")
        for (target_axis, target_dimension), state_axis, state_dimension in zip(
            retained_with_size, state.axes[1:], shapes[1].shape[1:], strict=True
        ):
            _aligned_axis(target_axis, state_axis, target_dimension, state_dimension, exact=True)
        unit = "1"
    elif operation == Operation.masked_cast_with_placeholders:
        require_matching_axes(first, typed[1], shapes[0].shape)
    elif operation == Operation.masked_scalar_arithmetic_with_placeholders:
        require_matching_axes(first, typed[1], shapes[0].shape)
        if params["operator"] in ("add", "subtract") and unit != "1":
            raise TensorusError(
                "Scalar addition or subtraction requires dimensionless values; "
                "use a tensor with an explicitly matching unit"
            )
    elif operation in (
        Operation.convert_value_unit,
        Operation.masked_convert_value_unit,
        Operation.convert_temperature_unit,
        Operation.masked_convert_temperature_unit,
    ):
        if operation in (
            Operation.masked_convert_value_unit,
            Operation.masked_convert_temperature_unit,
        ):
            require_matching_axes(first, typed[1], shapes[0].shape)
        if unit != params["source_unit"]:
            raise TensorusError("Declared value unit does not match the conversion source unit")
        unit = params["target_unit"]
    elif operation == Operation.convert_axis_unit:
        selected = next((axis for axis in axes if axis.label == params["axis"]), None)
        if selected is None:
            raise TensorusError("The conversion axis is not declared")
        if selected.unit != params["source_unit"]:
            raise TensorusError("Declared axis unit does not match the conversion source unit")
        if selected.coordinates is None:
            raise TensorusError("Axis-unit conversion requires explicit integer coordinates")
        if first.version == 2:
            if selected.coordinate_system is not None:
                raise TensorusError(
                    "V2 axis-unit conversion cannot change a coordinate-system-referenced axis"
                )
            if (
                selected.time_reference is not None
                and VALUE_UNITS_V1[params["source_unit"]][0] != "time"
            ):
                raise TensorusError("V2 referenced time axes require fixed-duration time units")
        exponent = (
            VALUE_UNITS_V1[params["source_unit"]][1] - VALUE_UNITS_V1[params["target_unit"]][1]
        )
        factor = 10 ** abs(exponent)
        converted = []
        for coordinate in selected.coordinates:
            if type(coordinate) is not int:
                raise TensorusError("Axis-unit conversion requires integer coordinates")
            if exponent < 0:
                if coordinate % factor:
                    raise TensorusError("Axis-unit conversion would require fractional coordinates")
                value = coordinate // factor
            else:
                value = coordinate * factor
            if not -(2**63) <= value < 2**63:
                raise TensorusError("Converted axis coordinate exceeds the signed 64-bit range")
            converted.append(value)
        selected.unit = params["target_unit"]
        selected.coordinates = converted
    elif operation != Operation.cast:
        raise TensorusError(f"No semantic propagation policy is available for {operation}")

    result = TensorSemantics(
        version=first.version,
        axes=axes,
        value_unit=unit,
        model=model,
        preprocessing=preprocessing,
    )
    return read_semantics({SEMANTICS_KEY: result.model_dump(mode="json")}, output.shape)


def require_comparable(
    left: TensorSemantics, right: TensorSemantics, shape: tuple[int, ...]
) -> None:
    _value_units(Operation.subtract, left, right)
    _model_identity([left, right], [TensorShape(shape, "float64")] * 2)
    if left.preprocessing != right.preprocessing:
        raise TensorusError("Tensor preprocessing identities or versions do not match")
    require_matching_axes(left, right, shape)


def matches_semantics(row: Mapping[str, Any], selected: SemanticFilter) -> bool:
    attributes, operation = row["attributes"], row["operation"]
    if not isinstance(attributes, dict) or (
        operation is not None and not isinstance(operation, dict)
    ):
        raise StorageError("Stored tensor descriptor is invalid for semantic discovery")
    version = operation.get("semantics_version") if operation is not None else None
    declared = attributes.get(SEMANTICS_KEY)
    if SEMANTICS_KEY not in attributes or (
        operation is not None
        and (
            type(version) is not int
            or not isinstance(declared, dict)
            or type(declared.get("version")) is not int
            or version != declared["version"]
        )
    ):
        return False
    descriptor_nbytes(dict(row))
    try:
        semantics = read_semantics(attributes, tuple(row["shape"]), operation)
    except TensorusError as exc:
        raise StorageError("Stored tensor semantics are invalid for discovery") from exc
    if semantics is None:
        return False
    if selected.value_unit is not None and semantics.value_unit != selected.value_unit:
        return False
    if selected.model_id is not None and (
        semantics.model is None or semantics.model.id != selected.model_id
    ):
        return False
    if selected.model_version is not None and (
        semantics.model is None or semantics.model.version != selected.model_version
    ):
        return False
    if selected.preprocessing_id is not None and (
        semantics.preprocessing is None or semantics.preprocessing.id != selected.preprocessing_id
    ):
        return False
    if selected.preprocessing_version is not None and (
        semantics.preprocessing is None
        or semantics.preprocessing.version != selected.preprocessing_version
    ):
        return False
    axes = {axis.label: axis for axis in semantics.axes}
    for axis in selected.axes:
        actual = axes.get(axis.label)
        if actual is None or (axis.unit is not None and actual.unit != axis.unit):
            return False
        if axis.time_reference is not None and actual.time_reference != axis.time_reference:
            return False
        if (
            axis.coordinate_system is not None
            and actual.coordinate_system != axis.coordinate_system
        ):
            return False
        coordinates = actual.coordinates
        if axis.coordinates is not None and (coordinates is not None) != (
            axis.coordinates == "present"
        ):
            return False
        for requirement in (
            axis.coordinate_span,
            axis.coordinate_points,
            axis.coordinate_count,
        ):
            if requirement is None:
                continue
            time_reference = axis.time_reference or requirement.time_reference
            coordinate_system = axis.coordinate_system or requirement.coordinate_system
            if (
                actual.time_reference != time_reference
                or actual.coordinate_system != coordinate_system
            ):
                return False
            if actual.unit != requirement.unit or coordinates is None:
                return False
        span = axis.coordinate_span
        if span is not None:
            if not coordinates:
                return False
            first, last = coordinates[0], coordinates[-1]
            if span.relation == "covers" and (first > span.min or last < span.max):
                return False
            if span.relation == "overlaps" and (first > span.max or last < span.min):
                return False
            if span.relation == "within" and (first < span.min or last > span.max):
                return False
        points = axis.coordinate_points
        if points is not None:
            if not coordinates or (
                points.comparison == "typed_v2" and semantics.version != REFERENCE_SEMANTICS_VERSION
            ):
                return False
            matches = []
            for value in points.values:
                position = bisect_left(coordinates, value)
                matches.append(
                    position < len(coordinates)
                    and coordinates[position] == value
                    and (
                        points.comparison is None
                        or canonical_json(coordinates[position]) == canonical_json(value)
                    )
                )
            if points.relation == "all" and not all(matches):
                return False
            if points.relation == "any" and not any(matches):
                return False
        count = axis.coordinate_count
        if count is not None:
            if coordinates is None or semantics.version != REFERENCE_SEMANTICS_VERSION:
                return False
            matches_count = bisect_right(coordinates, count.max) - bisect_left(
                coordinates, count.min
            )
            if matches_count < count.min_count or (
                count.max_count is not None and matches_count > count.max_count
            ):
                return False
    return True


def matches_shared_axis(row: Mapping[str, Any], matcher: SharedAxisQueryMatcher | None) -> bool:
    if matcher is None:
        return True
    if (
        matcher.anchor_scope is None
        or matcher.anchor_scope[0] != row["tenant_id"]
        or (not matcher.across_datasets and matcher.anchor_scope[1] != row["dataset_id"])
    ):
        return False
    try:
        semantics = read_semantics(row["attributes"], tuple(row["shape"]), row["operation"])
    except (TensorusError, TypeError) as exc:
        raise StorageError("Stored shared-axis candidate semantics are invalid") from exc
    return semantics is not None and matcher.matches(semantics, tuple(row["shape"]))


def resolve_coordinate_axis_selection(
    version: Mapping[str, Any], request: CoordinateSliceRequest
) -> AxisSliceRequest:
    shape = version["shape"]
    metadata = version["metadata"]
    semantics = read_semantics(metadata, tuple(shape), version["operation"])
    if semantics is None and version["operation"] is not None:
        raise TensorusError("Coordinates on derived tensors are not verified")
    if semantics is not None:
        axes = [AxisDescriptor(label=axis.label, unit=axis.unit) for axis in semantics.axes]
        coordinates = {
            axis.label: CoordinateValues(values=axis.coordinates, unit=axis.unit)
            for axis in semantics.axes
            if axis.coordinates is not None
        }
    else:
        try:
            axes = TypeAdapter(list[AxisDescriptor]).validate_python(
                metadata.get("tensorus_axes"), strict=True
            )
            coordinates = TypeAdapter(dict[str, CoordinateValues]).validate_python(
                metadata.get("tensorus_coordinates"), strict=True
            )
        except ValidationError as exc:
            raise TensorusError("Version has invalid axis or coordinate metadata") from exc
    labels = [axis.label for axis in axes]
    if len(labels) != len(shape) or len(set(labels)) != len(labels):
        raise TensorusError("Axis descriptor must have one unique label per tensor dimension")
    if set(coordinates) - set(labels) or set(request.selection) - set(coordinates):
        raise TensorusError("Selection or coordinates contain an unknown axis")
    for axis, dimension in zip(axes, shape, strict=True):
        if axis.label not in coordinates:
            continue
        coord = coordinates[axis.label]
        values = coord.values
        if (
            coord.unit != axis.unit
            or len(values) != dimension
            or any(isinstance(v, float) and not math.isfinite(v) for v in values)
            or any(a >= b for a, b in pairwise(values))
        ):
            raise TensorusError(
                "Coordinate metadata must match its axis and be strictly increasing"
            )
    positional: dict[str, list[int | None]] = {}
    references = {axis.label: axis for axis in semantics.axes} if semantics is not None else {}
    for label, bounds in request.selection.items():
        coord = coordinates[label]
        values = coord.values
        reference_axis = references.get(label)
        if bounds.time_reference != (
            reference_axis.time_reference if reference_axis is not None else None
        ) or bounds.coordinate_system != (
            reference_axis.coordinate_system if reference_axis is not None else None
        ):
            raise TensorusError("Coordinate range reference must match the declared axis")
        if (
            bounds.unit != coord.unit
            or bounds.min > bounds.max
            or not values
            or bounds.min < values[0]
            or bounds.max > values[-1]
        ):
            raise TensorusError("Coordinate range or unit does not match the axis")
        positional[label] = [bisect_left(values, bounds.min), bisect_right(values, bounds.max)]
    return AxisSliceRequest(selection=positional)
