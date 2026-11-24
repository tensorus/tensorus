"""Operation-level disclosure of direct observation selection, not data truth."""

import math
from typing import Annotated, Any, Literal

from pydantic import Field, StrictInt

from tensorus.models import MASKED_BINARY_OPERATIONS, Model, Operation
from tensorus.operations import (
    ARITHMETIC,
    CHUNKED_MASKED_COUNTS,
    MASKED_REDUCTIONS,
    PAIRED_SCALING_VALUES,
)

OBSERVATION_SCOPE_FORMAT: Literal["tensorus-step-observation-scope-v1"] = (
    "tensorus-step-observation-scope-v1"
)
OBSERVATION_SCOPE_V2_FORMAT: Literal["tensorus-step-observation-scope-v2"] = (
    "tensorus-step-observation-scope-v2"
)
OBSERVATION_SCOPE_V3_FORMAT: Literal["tensorus-step-observation-scope-v3"] = (
    "tensorus-step-observation-scope-v3"
)
OBSERVATION_SCOPE_V4_FORMAT: Literal["tensorus-step-observation-scope-v4"] = (
    "tensorus-step-observation-scope-v4"
)
ObservationScopeVersion = Literal["v1", "v2", "v3", "v4"]
ObservationScopeFormat = Literal[
    "tensorus-step-observation-scope-v1",
    "tensorus-step-observation-scope-v2",
    "tensorus-step-observation-scope-v3",
    "tensorus-step-observation-scope-v4",
]

V4_COORDINATE_OPERATIONS = frozenset(
    {Operation.reindex_coordinates, Operation.reindex_validity, Operation.coordinate_coverage}
)
V4_FITTED_OPERATIONS = frozenset(
    {Operation.fit_selected_preprocessing, Operation.apply_fitted_preprocessing}
)
V3_RAW_UNARY_OPERATIONS = frozenset(
    {
        Operation.transpose,
        Operation.slice,
        Operation.chunked_slice,
        Operation.select_indices,
        Operation.reshape,
        Operation.convert_axis_unit,
    }
)
V3_RAW_BINARY_OPERATIONS = frozenset(
    {Operation.logical_and, Operation.matmul, Operation.batched_matmul}
)
V3_RAW_OPERATIONS = V3_RAW_UNARY_OPERATIONS | V3_RAW_BINARY_OPERATIONS

RAW_STORED_OPERATIONS = frozenset(
    {
        Operation.profile,
        Operation.normalize,
        Operation.standardize,
        Operation.cast,
        Operation.convert_value_unit,
        Operation.convert_temperature_unit,
        Operation.sum,
        Operation.mean,
        Operation.variance,
        Operation.std,
        Operation.min,
        Operation.max,
        Operation.chunked_count_nonzero,
        Operation.chunked_count_nonzero_by_axis,
        Operation.chunked_sum_by_axis,
        Operation.chunked_mean_by_axis,
        Operation.chunked_variance_by_axis,
        Operation.chunked_std_by_axis,
        Operation.chunked_min_by_axis,
        Operation.chunked_max_by_axis,
    }
)
VALIDITY_BIT_OPERATIONS = frozenset(
    {Operation.reduce_validity_any, Operation.chunked_reduce_validity_any}
)
MASKED_VALUE_OPERATIONS = (
    frozenset(PAIRED_SCALING_VALUES)
    | frozenset(MASKED_BINARY_OPERATIONS)
    | {
        Operation.masked_cast_with_placeholders,
        Operation.masked_scalar_arithmetic_with_placeholders,
        Operation.masked_convert_value_unit,
        Operation.masked_convert_temperature_unit,
    }
)
MASKED_REDUCTION_OPERATIONS = frozenset(MASKED_REDUCTIONS | CHUNKED_MASKED_COUNTS)
REJECT_EMPTY_SELECTION = frozenset(
    {
        Operation.masked_sum,
        Operation.masked_mean,
        Operation.chunked_masked_count_nonzero,
        Operation.chunked_masked_count_nonzero_by_axis,
    }
)
ValueInputIndex = Annotated[StrictInt, Field(ge=0, le=1)]
MaskInputIndex = Annotated[StrictInt, Field(ge=1, le=2)]


class StepObservationScope(Model):
    kind: Literal[
        "raw_stored_values",
        "explicit_true_mask",
        "validity_mask_bits",
        "coordinate_mapped_values",
        "coordinate_mapped_bits",
        "coordinate_coverage",
        "not_classified",
    ]
    value_inputs: list[ValueInputIndex] = Field(
        max_length=2, description="Zero-based indices of direct value inputs."
    )
    descriptor_only_inputs: list[ValueInputIndex] | None = Field(
        default=None,
        max_length=2,
        exclude_if=lambda value: value is None,
        description=(
            "Inputs whose descriptors are used but whose payload values do not determine the "
            "coordinate result. Execution may still read and verify their complete payloads."
        ),
    )
    fitted_state_inputs: list[ValueInputIndex] | None = Field(
        default=None,
        max_length=1,
        exclude_if=lambda value: value is None,
        description="Direct fitted-state payload inputs; they are not masked observations.",
    )
    mask_input: MaskInputIndex | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Zero-based index of the explicit true-includes mask input.",
    )
    binding_relation: Literal["not_inferred"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    excluded_positions: Literal["zero_placeholder"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    empty_selected_slice: Literal["zero_placeholder", "reject"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    coordinate_gaps: Literal["declared_fill", "false"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    output_binding: Literal["not_created_by_step", "not_applicable"]


def step_observation_scope(operation: Operation) -> StepObservationScope:
    output_binding: Literal["not_created_by_step", "not_applicable"] = (
        "not_applicable" if operation == Operation.profile else "not_created_by_step"
    )
    if operation in RAW_STORED_OPERATIONS:
        return StepObservationScope(
            kind="raw_stored_values", value_inputs=[0], output_binding=output_binding
        )
    if operation in VALIDITY_BIT_OPERATIONS:
        return StepObservationScope(
            kind="validity_mask_bits", value_inputs=[0], output_binding=output_binding
        )
    if operation in MASKED_REDUCTION_OPERATIONS:
        return StepObservationScope(
            kind="explicit_true_mask",
            value_inputs=[0],
            mask_input=1,
            binding_relation="not_inferred",
            empty_selected_slice=(
                "reject" if operation in REJECT_EMPTY_SELECTION else "zero_placeholder"
            ),
            output_binding=output_binding,
        )
    if operation in MASKED_VALUE_OPERATIONS:
        binary = operation in MASKED_BINARY_OPERATIONS
        return StepObservationScope(
            kind="explicit_true_mask",
            value_inputs=[0, 1] if binary else [0],
            mask_input=2 if binary else 1,
            binding_relation="not_inferred",
            excluded_positions="zero_placeholder",
            output_binding=output_binding,
        )
    return StepObservationScope(
        kind="not_classified", value_inputs=[], output_binding=output_binding
    )


def step_observation_scope_v2(operation: Operation, input_count: int) -> StepObservationScope:
    if operation in ARITHMETIC:
        if input_count not in (1, 2):
            raise ValueError("Raw arithmetic requires one or two direct tensor inputs")
        return StepObservationScope(
            kind="raw_stored_values",
            value_inputs=list(range(input_count)),
            output_binding="not_created_by_step",
        )
    return step_observation_scope(operation)


def step_observation_scope_v3(operation: Operation, input_count: int) -> StepObservationScope:
    if operation in V3_RAW_OPERATIONS:
        expected = 1 if operation in V3_RAW_UNARY_OPERATIONS else 2
        if input_count != expected:
            raise ValueError(f"{operation} requires {expected} direct tensor input(s)")
        return StepObservationScope(
            kind="raw_stored_values",
            value_inputs=list(range(input_count)),
            output_binding="not_created_by_step",
        )
    return step_observation_scope_v2(operation, input_count)


def step_observation_scope_v4(operation: Operation, input_count: int) -> StepObservationScope:
    if operation in V4_FITTED_OPERATIONS:
        fitting = operation == Operation.fit_selected_preprocessing
        if input_count != (2 if fitting else 3):
            raise ValueError(f"{operation} has an invalid direct input count")
        return StepObservationScope(
            kind="explicit_true_mask",
            value_inputs=[0],
            fitted_state_inputs=None if fitting else [1],
            mask_input=1 if fitting else 2,
            binding_relation="not_inferred",
            empty_selected_slice="reject" if fitting else None,
            excluded_positions=None if fitting else "zero_placeholder",
            output_binding="not_created_by_step",
        )
    if operation in V4_COORDINATE_OPERATIONS:
        if input_count not in (1, 2):
            raise ValueError(f"{operation} requires one or two direct tensor inputs")
        coverage = operation == Operation.coordinate_coverage
        mapped_bits = operation == Operation.reindex_validity
        descriptor_only = list(range(input_count)) if coverage else list(range(1, input_count))
        return StepObservationScope(
            kind=(
                "coordinate_coverage"
                if coverage
                else "coordinate_mapped_bits"
                if mapped_bits
                else "coordinate_mapped_values"
            ),
            value_inputs=[] if coverage else [0],
            descriptor_only_inputs=descriptor_only or None,
            binding_relation="not_inferred",
            coordinate_gaps="false" if coverage or mapped_bits else "declared_fill",
            output_binding="not_created_by_step",
        )
    return step_observation_scope_v3(operation, input_count)


def validate_arithmetic_input_count(
    operation: Operation, input_count: int, parameters: dict[str, Any]
) -> None:
    if operation not in ARITHMETIC:
        return
    if set(parameters) != {"scalar"}:
        raise ValueError("Raw arithmetic parameters do not identify the scalar operand")
    scalar = parameters["scalar"]
    if scalar is not None and (
        type(scalar) not in (int, float)
        or (isinstance(scalar, float) and not math.isfinite(scalar))
    ):
        raise ValueError("Raw arithmetic scalar must be a finite number")
    if input_count != (1 if scalar is not None else 2):
        raise ValueError("Raw arithmetic inputs do not match the scalar operand")


def observation_scope_format(
    include: bool, version: ObservationScopeVersion
) -> ObservationScopeFormat | None:
    if type(include) is not bool:
        raise TypeError("include_observation_scope must be a boolean")
    if version not in ("v1", "v2", "v3", "v4"):
        raise ValueError("observation_scope_version must be v1, v2, v3 or v4")
    if not include:
        if version != "v1":
            raise ValueError("A scope version requires include_observation_scope=True")
        return None
    if version == "v4":
        return OBSERVATION_SCOPE_V4_FORMAT
    if version == "v3":
        return OBSERVATION_SCOPE_V3_FORMAT
    if version == "v2":
        return OBSERVATION_SCOPE_V2_FORMAT
    return OBSERVATION_SCOPE_FORMAT


def _validate_scope_indices(scope: StepObservationScope, input_count: int) -> None:
    indices = [*scope.value_inputs]
    if scope.mask_input is not None:
        indices.append(scope.mask_input)
    if scope.descriptor_only_inputs is not None:
        indices.extend(scope.descriptor_only_inputs)
    if scope.fitted_state_inputs is not None:
        indices.extend(scope.fitted_state_inputs)
    if indices and input_count != max(indices) + 1:
        raise ValueError("Observation scope does not match the operation inputs")


def validate_step_observation_scope(
    operation: Operation, scope: StepObservationScope, input_count: int
) -> None:
    if scope != step_observation_scope(operation):
        raise ValueError("Observation scope does not match the operation")
    _validate_scope_indices(scope, input_count)


def validate_step_observation_scope_v2(
    operation: Operation, scope: StepObservationScope, input_count: int
) -> None:
    if scope != step_observation_scope_v2(operation, input_count):
        raise ValueError("V2 observation scope does not match the operation inputs")
    _validate_scope_indices(scope, input_count)


def validate_step_observation_scope_v3(
    operation: Operation, scope: StepObservationScope, input_count: int
) -> None:
    if scope != step_observation_scope_v3(operation, input_count):
        raise ValueError("V3 observation scope does not match the operation inputs")
    _validate_scope_indices(scope, input_count)


def validate_step_observation_scope_v4(
    operation: Operation, scope: StepObservationScope, input_count: int
) -> None:
    if scope != step_observation_scope_v4(operation, input_count):
        raise ValueError("V4 observation scope does not match the operation inputs")
    _validate_scope_indices(scope, input_count)
