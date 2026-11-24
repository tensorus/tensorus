"""Exhaustive operation-family contracts used by prediction and discovery."""

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Self

from pydantic import Field, JsonValue, model_validator

from tensorus.errors import TensorusError
from tensorus.models import (
    MASKED_BINARY_OPERATIONS,
    PINNED_SELECTED_VALUE_OPERATIONS,
    Model,
    Operation,
)

ArityRule = Literal["one", "two", "three", "scalar_or_pair", "one_or_anchor"]
ResultDtype = Literal["none", "input", "float64", "bool", "int64", "target"]


@dataclass(frozen=True)
class OperationContract:
    family: str
    arity: ArityRule
    shape: str
    dtype: ResultDtype
    semantics: str
    validity: str
    numerical: str


def _contracts() -> dict[Operation, OperationContract]:
    result: dict[Operation, OperationContract] = {}

    def add(
        operations: set[Operation],
        *,
        family: str,
        arity: ArityRule,
        shape: str,
        dtype: ResultDtype,
        semantics: str,
        validity: str,
        numerical: str,
    ) -> None:
        contract = OperationContract(family, arity, shape, dtype, semantics, validity, numerical)
        for operation in operations:
            if operation in result:
                raise RuntimeError(f"Duplicate operation contract for {operation}")
            result[operation] = contract

    no_binding = "No observation-validity binding is created or inferred by this step."
    selected = (
        "The explicit true-includes mask selects direct inputs; excluded output positions "
        "are zero placeholders. A separately completed binding is required."
    )
    reduced = (
        "The output has no observation-validity binding. Reviewed paired reductions "
        "separately compute any-included mask bits and require explicit completion."
    )
    axes_preserved = (
        "Axes, units, model and preprocessing declarations are preserved or positionally "
        "transformed; v2 references must match exactly where applicable."
    )
    reduced_semantics = (
        "Removes or invalidates reduced axes, preserves unit/model and invalidates preprocessing."
    )
    float64 = (
        "Results use float64. Integer-to-float precision and finite-result checks apply, "
        "along with operation-specific overflow or underflow checks; arithmetic is not exact."
    )
    add(
        {Operation.profile},
        family="profile",
        arity="one",
        shape="no_tensor_output",
        dtype="none",
        semantics="Reports the input's recorded raw catalog statistics; no result semantics.",
        validity="No tensor output or binding; catalog statistics include all stored values.",
        numerical="No payload calculation is performed by this step.",
    )
    add(
        {Operation.normalize, Operation.standardize},
        family="raw_scaling",
        arity="one",
        shape="input_shape",
        dtype="float64",
        semantics="Preserves axes and model, sets value unit to 1 and invalidates preprocessing.",
        validity=no_binding,
        numerical=(
            float64
            + " Normalization uses the L2 norm; standardization uses the mean and population "
            "standard deviation. Epsilon bounds the global denominator."
        ),
    )
    add(
        {
            Operation.masked_normalize_with_placeholders,
            Operation.masked_standardize_with_placeholders,
        },
        family="selected_scaling",
        arity="two",
        shape="input_shape",
        dtype="float64",
        semantics=(
            "V2 requires matching data/mask axes; preserves model, sets unit to 1 and "
            "invalidates preprocessing."
        ),
        validity=selected,
        numerical=(
            float64
            + " Selected normalization uses an L2 norm; selected standardization uses the mean "
            "and population standard deviation. Version 1 reduces the current tensor globally; "
            "version 2 reduces each remaining-axis slice. Epsilon bounds each denominator. "
            "Neither produces reusable fitted state."
        ),
    )
    add(
        {Operation.fit_selected_preprocessing},
        family="fitted_selected_preprocessing_fit",
        arity="two",
        shape="versioned_parameters_per_remaining_axis_slice",
        dtype="float64",
        semantics="V2 data/mask axes match; state records a parameter axis and remaining axes.",
        validity=(
            "The explicit true-includes mask selects training values. Every fitted slice must "
            "contain an observation; the state is not a validity binding."
        ),
        numerical=(
            "Version 1 stores a float64 center and positive scale; version 2 stores an anchor, "
            "center offset and positive scale per remaining-axis slice. The scale is an "
            "epsilon-clamped L2 norm or population standard deviation. Unsafe legacy centers "
            "are rejected; integer precision, overflow and underflow limits apply."
        ),
    )
    add(
        {Operation.apply_fitted_preprocessing},
        family="fitted_selected_preprocessing_apply",
        arity="three",
        shape="input_shape",
        dtype="float64",
        semantics=(
            "V2 data/mask axes match; the exact fitted state and training domain are checked. "
            "Preserves axes/model, sets unit to 1 and clears the declared preprocessing identity."
        ),
        validity=selected,
        numerical=(
            "Applies the versioned immutable float64 center or anchor/offset and positive scale. "
            "Integer precision, overflow and underflow checks apply; excluded positions are zero "
            "placeholders."
        ),
    )
    add(
        {Operation.add, Operation.subtract, Operation.multiply, Operation.divide},
        family="raw_arithmetic",
        arity="scalar_or_pair",
        shape="broadcast_inputs",
        dtype="float64",
        semantics=(
            "Broadcast axes are reconciled; tensor operands require compatible model and "
            "preprocessing identities. Units are derived or rejected; preprocessing is invalidated."
        ),
        validity=no_binding,
        numerical=float64 + " Raw arithmetic also transforms any stored placeholders.",
    )
    add(
        set(MASKED_BINARY_OPERATIONS),
        family="selected_binary_arithmetic",
        arity="three",
        shape="exact_equal_input_shapes",
        dtype="float64",
        semantics=(
            "Value and mask axes must match; model, units and preprocessing compatibility "
            "are checked. The result preprocessing declaration is invalidated."
        ),
        validity=selected,
        numerical=float64,
    )
    add(
        {Operation.masked_scalar_arithmetic_with_placeholders},
        family="selected_scalar_arithmetic",
        arity="two",
        shape="input_shape",
        dtype="float64",
        semantics=(
            "V2 data/mask axes must match; scalar add/subtract requires unit 1. "
            "The preprocessing declaration is invalidated."
        ),
        validity=selected,
        numerical=float64 + " The selected scalar and algorithm version are explicit.",
    )
    add(
        {Operation.matmul, Operation.batched_matmul},
        family="matrix_multiplication",
        arity="two",
        shape="matrix_contraction_and_batch_broadcast",
        dtype="float64",
        semantics=(
            "Contracted axes must align; batch axes broadcast, units and model identities "
            "are checked, and preprocessing is invalidated."
        ),
        validity=no_binding,
        numerical=float64,
    )
    add(
        {Operation.transpose},
        family="transpose",
        arity="one",
        shape="permuted_axes",
        dtype="input",
        semantics=axes_preserved,
        validity=no_binding,
        numerical="Preserves stored values and dtype while permuting axes.",
    )
    add(
        {Operation.slice, Operation.chunked_slice},
        family="slice",
        arity="one",
        shape="positional_slice",
        dtype="input",
        semantics=axes_preserved,
        validity=no_binding,
        numerical="Preserves selected stored values and dtype; slicing is positional.",
    )
    add(
        {Operation.select_indices},
        family="index_selection",
        arity="one",
        shape="selected_axis_positions",
        dtype="input",
        semantics=axes_preserved,
        validity=no_binding,
        numerical="Preserves selected stored values and dtype; indices are strictly increasing.",
    )
    add(
        {Operation.reindex_coordinates},
        family="coordinate_values",
        arity="one_or_anchor",
        shape="coordinate_target_shape",
        dtype="input",
        semantics=(
            "Reconciles exact coordinate positions and any anchor axes; preprocessing is "
            "invalidated unless explicit gapless preservation is requested."
        ),
        validity=no_binding,
        numerical="Preserves mapped stored values; each gap uses the explicit representable fill.",
    )
    add(
        {Operation.reindex_validity},
        family="coordinate_bits",
        arity="one_or_anchor",
        shape="coordinate_target_shape",
        dtype="bool",
        semantics=(
            "Reconciles exact coordinate positions and any anchor axes; preprocessing is "
            "invalidated unless explicit gapless preservation is requested."
        ),
        validity="Produces mapped boolean bits with false gaps, not a binding by itself.",
        numerical="Requires boolean input and preserves mapped bits.",
    )
    add(
        {Operation.coordinate_coverage},
        family="coordinate_coverage",
        arity="one_or_anchor",
        shape="coordinate_target_shape",
        dtype="bool",
        semantics="Reconciles coordinate axes and clears value unit, model and preprocessing.",
        validity="Coordinate coverage is mapping presence, not observation validity.",
        numerical="Produces true for mapped positions and false for gaps.",
    )
    add(
        {Operation.logical_and},
        family="boolean_conjunction",
        arity="two",
        shape="broadcast_inputs",
        dtype="bool",
        semantics="Reconciles broadcast axes and clears unit, model and preprocessing.",
        validity="Produces boolean conjunction bits, not a binding by itself.",
        numerical="Requires boolean inputs and applies logical AND.",
    )
    add(
        {Operation.reshape},
        family="reshape",
        arity="one",
        shape="requested_element_count_preserving_shape",
        dtype="input",
        semantics=(
            "Under a semantic policy, shape-changing reshape is rejected without an explicit "
            "mapping; an unchanged shape preserves declarations."
        ),
        validity=no_binding,
        numerical="Preserves element count, values and dtype.",
    )
    add(
        {Operation.cast},
        family="cast",
        arity="one",
        shape="input_shape",
        dtype="target",
        semantics="Preserves axes, unit and model; invalidates preprocessing.",
        validity=no_binding,
        numerical="Checks integer range; ordinary casting may change values or lose precision.",
    )
    add(
        {Operation.masked_cast_with_placeholders},
        family="selected_cast",
        arity="two",
        shape="input_shape",
        dtype="target",
        semantics=(
            "V2 data/mask axes must match; preserves unit/model and invalidates preprocessing."
        ),
        validity=selected,
        numerical=(
            "Requires exact representability of selected numeric values; excluded values are zero."
        ),
    )
    add(
        {Operation.convert_axis_unit},
        family="axis_unit_conversion",
        arity="one",
        shape="input_shape",
        dtype="input",
        semantics=(
            "Converts one declared integer coordinate axis exactly; incompatible references, "
            "fractional results and out-of-range coordinates are rejected."
        ),
        validity=no_binding,
        numerical=(
            "Tensor values and dtype are unchanged; coordinate conversion uses exact integers."
        ),
    )
    add(
        {Operation.convert_value_unit, Operation.convert_temperature_unit},
        family="raw_value_unit_conversion",
        arity="one",
        shape="input_shape",
        dtype="float64",
        semantics=(
            "Requires the declared source unit and a supported conversion; changes the value "
            "unit, preserves axes/model and invalidates preprocessing."
        ),
        validity=no_binding,
        numerical=float64 + " Absolute-temperature conversion has a separate element limit.",
    )
    add(
        {Operation.masked_convert_value_unit, Operation.masked_convert_temperature_unit},
        family="selected_value_unit_conversion",
        arity="two",
        shape="input_shape",
        dtype="float64",
        semantics=(
            "Requires matching data/mask axes and the declared source unit; changes the value "
            "unit, preserves model and invalidates preprocessing."
        ),
        validity=selected,
        numerical=float64 + " Absolute-temperature conversion has a separate element limit.",
    )
    add(
        {Operation.chunked_count_nonzero},
        family="raw_scalar_count",
        arity="one",
        shape="scalar",
        dtype="int64",
        semantics="Produces unit 1 with no axes, model or preprocessing declaration.",
        validity="Counts stored nonzero values; this is not observation validity.",
        numerical="Counts nonzero stored values exactly within int64 and configured limits.",
    )
    add(
        {Operation.chunked_masked_count_nonzero},
        family="selected_scalar_count",
        arity="two",
        shape="scalar",
        dtype="int64",
        semantics=(
            "Requires matching data/mask axes; result unit is 1 with no model or preprocessing."
        ),
        validity="Counts only explicit true-mask positions; empty selection is rejected.",
        numerical="Counts nonzero selected values exactly within int64 and configured limits.",
    )
    add(
        {Operation.chunked_count_nonzero_by_axis},
        family="raw_axis_count",
        arity="one",
        shape="reduced_axes",
        dtype="int64",
        semantics=(
            "Removes or invalidates reduced axes; result unit is 1 with no model or preprocessing."
        ),
        validity="Counts stored nonzero values; this is not observation validity.",
        numerical="Counts nonzero stored values exactly within int64 and configured limits.",
    )
    add(
        {Operation.chunked_masked_count_nonzero_by_axis},
        family="selected_axis_count",
        arity="two",
        shape="reduced_axes",
        dtype="int64",
        semantics=(
            "Requires matching data/mask axes; result unit is 1 with no model or preprocessing."
        ),
        validity="Counts explicit true-mask positions; empty selected slices are rejected.",
        numerical="Counts nonzero selected values exactly within int64 and configured limits.",
    )
    add(
        {Operation.chunked_masked_count_nonzero_with_placeholders},
        family="selected_axis_count_placeholders",
        arity="two",
        shape="reduced_axes",
        dtype="int64",
        semantics=(
            "Requires matching data/mask axes; result unit is 1 with no model or preprocessing."
        ),
        validity=reduced + " Empty selected slices yield zero placeholders.",
        numerical="Counts nonzero selected values exactly within int64 and configured limits.",
    )
    add(
        {Operation.sum, Operation.chunked_sum_by_axis},
        family="raw_sum",
        arity="one",
        shape="reduced_axes",
        dtype="float64",
        semantics=reduced_semantics,
        validity=no_binding,
        numerical=float64 + " An empty sum is zero.",
    )
    add(
        {
            Operation.mean,
            Operation.std,
            Operation.chunked_mean_by_axis,
            Operation.chunked_std_by_axis,
        },
        family="raw_mean_or_std",
        arity="one",
        shape="reduced_axes",
        dtype="float64",
        semantics=reduced_semantics,
        validity=no_binding,
        numerical=float64 + " Empty reductions are rejected; dispersion is population dispersion.",
    )
    add(
        {Operation.variance, Operation.chunked_variance_by_axis},
        family="raw_variance",
        arity="one",
        shape="reduced_axes",
        dtype="float64",
        semantics=(
            "Removes or invalidates reduced axes; a non-dimensionless unit becomes unknown, "
            "model is preserved and preprocessing is invalidated."
        ),
        validity=no_binding,
        numerical=float64 + " Empty reductions are rejected; variance is population variance.",
    )
    add(
        {
            Operation.min,
            Operation.max,
            Operation.chunked_min_by_axis,
            Operation.chunked_max_by_axis,
        },
        family="raw_extremum",
        arity="one",
        shape="reduced_axes",
        dtype="input",
        semantics=reduced_semantics,
        validity=no_binding,
        numerical="Preserves input dtype; empty reductions are rejected.",
    )
    add(
        {Operation.masked_sum, Operation.masked_mean},
        family="selected_reduction",
        arity="two",
        shape="reduced_axes",
        dtype="float64",
        semantics="Requires matching data/mask axes; reduces axes and invalidates preprocessing.",
        validity="Uses an explicit true-includes mask; empty selected slices are rejected.",
        numerical=float64,
    )
    add(
        {
            Operation.masked_sum_with_placeholders,
            Operation.masked_mean_with_placeholders,
            Operation.masked_std_with_placeholders,
            Operation.chunked_masked_sum_with_placeholders,
            Operation.chunked_masked_mean_with_placeholders,
            Operation.chunked_masked_std_with_placeholders,
        },
        family="selected_reduction_placeholders",
        arity="two",
        shape="reduced_axes",
        dtype="float64",
        semantics="Requires matching data/mask axes; reduces axes and invalidates preprocessing.",
        validity=reduced + " Empty selected slices yield zero placeholders.",
        numerical=float64 + " Dispersion is population dispersion.",
    )
    add(
        {
            Operation.masked_variance_with_placeholders,
            Operation.chunked_masked_variance_with_placeholders,
        },
        family="selected_variance_placeholders",
        arity="two",
        shape="reduced_axes",
        dtype="float64",
        semantics=(
            "Requires matching data/mask axes; reduces axes, clears a non-dimensionless "
            "value unit and invalidates preprocessing."
        ),
        validity=reduced + " Empty selected slices yield zero placeholders.",
        numerical=float64 + " Variance is population variance.",
    )
    add(
        {
            Operation.masked_min_with_placeholders,
            Operation.masked_max_with_placeholders,
            Operation.chunked_masked_min_with_placeholders,
            Operation.chunked_masked_max_with_placeholders,
        },
        family="selected_extremum_placeholders",
        arity="two",
        shape="reduced_axes",
        dtype="input",
        semantics="Requires matching data/mask axes; reduces axes and invalidates preprocessing.",
        validity=reduced + " Empty selected slices yield zero placeholders.",
        numerical="Preserves input dtype and selected extrema.",
    )
    add(
        {Operation.reduce_validity_any, Operation.chunked_reduce_validity_any},
        family="validity_any",
        arity="one",
        shape="reduced_axes",
        dtype="bool",
        semantics="Reduces axes and clears unit, model and preprocessing.",
        validity="Produces any-included boolean bits, not a binding by itself.",
        numerical="Requires boolean input; empty selected slices produce false.",
    )
    if set(result) != set(Operation):
        missing = sorted(set(Operation) - set(result))
        extra = sorted(set(result) - set(Operation))
        raise RuntimeError(
            f"Operation contract coverage mismatch: missing={missing}, extra={extra}"
        )
    return result


OPERATION_CONTRACTS = _contracts()
OPERATION_CONTRACT_FORMAT: Literal["tensorus-operation-contract-matrix-v1"] = (
    "tensorus-operation-contract-matrix-v1"
)


class OperationContractDescription(Model):
    family: str
    arity: ArityRule
    input_counts: list[int] = Field(min_length=1, max_length=2)
    direct_input_scope: dict[str, dict[str, JsonValue]]
    parameters: dict[str, JsonValue]
    shape: str
    dtype: ResultDtype
    semantics: str
    supports_semantic_policy_v2: bool
    requires_pinned_selected_plan: bool
    validity: str
    numerical: str
    execution: Literal["catalog_profile", "bounded_verified_chunk_scan", "bounded_eager_full_read"]
    budget: str


class GuidedValidityOperations(Model):
    values_operation: str
    mask_operation: str


class GuidedValidityContracts(Model):
    request_schema: dict[str, JsonValue]
    requests: dict[str, GuidedValidityOperations]
    policy: str


class OperationContractMatrix(Model):
    format: Literal["tensorus-operation-contract-matrix-v1"]
    source_identity: str
    publication: str
    limits: str
    semantics_policy: str
    guided_validity: GuidedValidityContracts
    operations: dict[str, OperationContractDescription] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_contracts(self) -> Self:
        from tensorus.observation_scope import (
            StepObservationScope,
            validate_step_observation_scope_v4,
        )
        from tensorus.semantics import REFERENCE_OPERATIONS
        from tensorus.validity import validity_transform_operations

        counts = {
            "one": [1],
            "two": [2],
            "three": [3],
            "scalar_or_pair": [1, 2],
            "one_or_anchor": [1, 2],
        }
        for name, item in self.operations.items():
            try:
                operation = Operation(name)
            except ValueError as exc:
                raise ValueError(f"Unknown operation contract: {name}") from exc
            known = OPERATION_CONTRACTS[operation]
            if item.input_counts != counts[item.arity] or set(item.direct_input_scope) != {
                str(count) for count in item.input_counts
            }:
                raise ValueError("Operation arity and direct-input scopes disagree")
            if item.arity != known.arity or item.dtype != known.dtype:
                raise ValueError("Operation arity or dtype is unsupported by this client")
            if item.supports_semantic_policy_v2 != (
                operation in REFERENCE_OPERATIONS
            ) or item.requires_pinned_selected_plan != (
                operation in PINNED_SELECTED_VALUE_OPERATIONS
            ):
                raise ValueError("Operation execution flags are unsupported by this client")
            for count in item.input_counts:
                scope = StepObservationScope.model_validate(
                    item.direct_input_scope[str(count)], strict=True
                )
                validate_step_observation_scope_v4(operation, scope, count)
            if (item.dtype == "none") != (item.execution == "catalog_profile"):
                raise ValueError("Operation result and execution contracts disagree")
            if item.requires_pinned_selected_plan and not item.supports_semantic_policy_v2:
                raise ValueError("Pinned selected operations require a v2 semantic rule")
        for request, pair in self.guided_validity.requests.items():
            if pair.values_operation not in self.operations or pair.mask_operation not in (
                self.operations
            ):
                raise ValueError("Guided validity refers to an unknown operation")
            try:
                values, mask = validity_transform_operations(request)
            except TensorusError as exc:
                raise ValueError("Unsupported guided validity request") from exc
            if (pair.values_operation, pair.mask_operation) != (values, mask):
                raise ValueError("Guided validity operations are unsupported by this client")
        return self


def operation_input_count(operation: Operation, parameters: dict[str, Any], count: int) -> int:
    rule = OPERATION_CONTRACTS[operation].arity
    if rule == "scalar_or_pair":
        return 1 if parameters["scalar"] is not None else 2
    if rule == "one_or_anchor":
        return count if count in (1, 2) else 1
    return {"one": 1, "two": 2, "three": 3}[rule]


@lru_cache(maxsize=1)
def _operation_contract_matrix() -> dict[str, Any]:
    from tensorus.arrays import CHUNKED_BLOCK_BYTES, CHUNKED_MAX_BLOCKS
    from tensorus.observation_scope import step_observation_scope_v4
    from tensorus.operations import (
        CHUNKED_OPERATIONS,
        MAX_SELECTED_SCALING_GROUPS,
        MAX_SELECTION_INDICES,
        MAX_TEMPERATURE_ELEMENTS,
        PARAMETERS,
    )
    from tensorus.semantics import REFERENCE_OPERATIONS
    from tensorus.validity import (
        V2_PAIRED_TRANSFORM_REQUESTS,
        ValidityTransform,
        validity_transform_operations,
    )

    if set(PARAMETERS) != set(OPERATION_CONTRACTS):
        raise RuntimeError("Operation parameter schemas and contracts do not match")
    validity_schema = ValidityTransform.model_json_schema()
    if set(validity_schema["properties"]["operation"]["enum"]) != V2_PAIRED_TRANSFORM_REQUESTS:
        raise RuntimeError("Guided validity schema and request registry do not match")
    counts = {
        "one": (1,),
        "two": (2,),
        "three": (3,),
        "scalar_or_pair": (1, 2),
        "one_or_anchor": (1, 2),
    }
    guided: dict[str, dict[str, str]] = {}
    for request in sorted(V2_PAIRED_TRANSFORM_REQUESTS):
        values, mask = validity_transform_operations(request)
        guided[request] = {"values_operation": values.value, "mask_operation": mask.value}

    def budget(operation: Operation) -> str:
        result = (
            "Each output is bounded by limits.tensor_bytes; preflight reports the plan's "
            "output-byte and estimated-compute limits. Estimates are not peak memory or duration. "
        )
        if operation in CHUNKED_OPERATIONS:
            result += (
                "Inputs use the configured chunked-input limit; verified scans use at most "
                f"{CHUNKED_BLOCK_BYTES} bytes per logical block and {CHUNKED_MAX_BLOCKS} blocks "
                "across the operation's passes, plus layout-dependent chunk, scanned-position "
                "and decoded-byte limits."
            )
        else:
            result += "Each eager input is bounded by limits.tensor_bytes."
        if operation in (
            Operation.masked_normalize_with_placeholders,
            Operation.masked_standardize_with_placeholders,
            Operation.fit_selected_preprocessing,
            Operation.apply_fitted_preprocessing,
        ):
            result += (
                " Selected scaling and fitted preprocessing check a working-memory estimate "
                f"and at most {MAX_SELECTED_SCALING_GROUPS} remaining-axis groups where applicable."
            )
        if operation in (
            Operation.masked_cast_with_placeholders,
            Operation.masked_scalar_arithmetic_with_placeholders,
        ):
            result += " Admission includes an operation-specific working-memory estimate."
        if operation == Operation.masked_mean_with_placeholders:
            result += " Algorithm version 2 also checks a working-memory estimate."
        if operation in (
            Operation.convert_temperature_unit,
            Operation.masked_convert_temperature_unit,
        ):
            result += f" Temperature conversion is limited to {MAX_TEMPERATURE_ELEMENTS} elements."
        if operation in (
            Operation.select_indices,
            Operation.reindex_coordinates,
            Operation.reindex_validity,
            Operation.coordinate_coverage,
        ):
            result += (
                f" Explicit selection or target positions are limited to {MAX_SELECTION_INDICES} "
                "in total, with per-axis limits in the parameter schema."
            )
        return result

    return {
        "format": OPERATION_CONTRACT_FORMAT,
        "source_identity": (
            "Each executable-plan input is an immutable version UUID or an earlier materialized "
            "@step; profile checkpoints cannot be tensor inputs. Workflow tensor names resolve "
            "to exact versions when the plan is instantiated."
        ),
        "publication": (
            "Each materialized output and checkpoint commit together. Separate steps do not "
            "commit atomically; reviewed data/mask pairs require explicit binding completion. "
            "Pinned selected scaling, casting, scalar arithmetic and fitted preprocessing require "
            "their fixed reviewed graphs rather than arbitrary earlier steps; complete a result "
            "binding before reuse."
        ),
        "limits": (
            "Preflight and execution apply configured input, output and compute limits to every "
            "step; "
            "chunked operations additionally apply advertised block, scan and decoded-byte limits."
        ),
        "semantics_policy": (
            "The semantic rules below apply only with an explicitly selected compatible policy. "
            "Undeclared inputs without a policy receive no versioned semantic derivation; v2 "
            "rejects operations without a listed v2 rule. V2 coordinate operations require an "
            "anchor. Prediction and semantic derivation run on every intermediate."
        ),
        "guided_validity": {
            "request_schema": validity_schema,
            "requests": guided,
            "policy": (
                "Guided requests create reviewed plans, not automatic bindings. Binary graphs "
                "pin two sources and intersect masks; unary pins depend on the request. A paired "
                "result is usable as a binding only after explicit authorized completion. "
                "The graph may contain additional alignment steps."
            ),
        },
        "operations": {
            operation.value: {
                "family": contract.family,
                "arity": contract.arity,
                "input_counts": list(counts[contract.arity]),
                "direct_input_scope": {
                    str(count): step_observation_scope_v4(operation, count).model_dump(mode="json")
                    for count in counts[contract.arity]
                },
                "parameters": PARAMETERS[operation].model_json_schema(),
                "shape": contract.shape,
                "dtype": contract.dtype,
                "semantics": contract.semantics,
                "supports_semantic_policy_v2": operation in REFERENCE_OPERATIONS,
                "requires_pinned_selected_plan": operation in PINNED_SELECTED_VALUE_OPERATIONS,
                "validity": contract.validity,
                "numerical": contract.numerical,
                "budget": budget(operation),
                "execution": (
                    "catalog_profile"
                    if operation == Operation.profile
                    else "bounded_verified_chunk_scan"
                    if operation in CHUNKED_OPERATIONS
                    else "bounded_eager_full_read"
                ),
            }
            for operation, contract in OPERATION_CONTRACTS.items()
        },
    }


def operation_contract_matrix() -> dict[str, Any]:
    return deepcopy(_operation_contract_matrix())
