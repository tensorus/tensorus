"""Explicit, immutable assertions pairing data and observation-inclusion masks."""
import hashlib
import math
from collections.abc import Collection
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import ConfigDict
from pydantic import Field
from pydantic import JsonValue
from pydantic import StrictInt
from pydantic import ValidationError
from pydantic import model_validator
from tensorus.arrays import Record
from tensorus.arrays import canonical_json
from tensorus.arrays import descriptor_nbytes
from tensorus.arrays import reduced_shape
from tensorus.errors import TensorusError
from tensorus.models import MASKED_BINARY_OPERATIONS
from tensorus.models import CoordinateRange
from tensorus.models import DType
from tensorus.models import Model
from tensorus.models import Name
from tensorus.models import Operation
from tensorus.models import PlanDefinition
from tensorus.models import PlanValiditySource
from tensorus.models import Step
from tensorus.operations import BOUND_SELECTED_PREPROCESSING
from tensorus.operations import CHUNKED_PAIRED_REDUCTION_VALUES
from tensorus.operations import MAX_SELECTION_INDICES
from tensorus.operations import MAX_TEMPERATURE_ELEMENTS
from tensorus.operations import PAIRED_REDUCTION_OPERATIONS
from tensorus.operations import PAIRED_REDUCTION_VALUES
from tensorus.operations import PAIRED_SCALING_OPERATIONS
from tensorus.operations import PAIRED_SCALING_VALUES
from tensorus.operations import FittedScalingParameters
from tensorus.operations import ImportedFittedStateSelection
from tensorus.operations import TensorShape
from tensorus.operations import fitted_state_shape
from tensorus.operations import paired_reduction_dtype
from tensorus.operations import parameters
from tensorus.operations import reduction_axes
from tensorus.operations import reindex_fill
from tensorus.semantics import PreprocessingIdentity
from tensorus.semantics import TensorSemantics
from tensorus.semantics import derive_semantics
from tensorus.semantics import read_semantics
from tensorus.semantics import referenced_coordinate_filled_alignment
from tensorus.semantics import referenced_coordinate_index_alignment
from tensorus.semantics import referenced_coordinate_index_intersection
from tensorus.semantics import require_matching_axes
from tensorus.semantics import resolve_coordinate_axis_selection
VALIDITY_FORMAT: Literal["tensorus-observation-validity-v1"] = "tensorus-observation-validity-v1"
BINARY_VALIDITY_FORMAT: Literal["tensorus-observation-validity-v2"] = (
    "tensorus-observation-validity-v2"
)
MAX_BINDINGS_PER_VERSION = 32
POSITIONAL_TRANSFORMS = {
    Operation.transpose,
    Operation.slice,
    Operation.chunked_slice,
    Operation.select_indices,
}
TRANSFORMS = POSITIONAL_TRANSFORMS | {Operation.convert_axis_unit}
VALUE_CONVERSIONS = {
    Operation.convert_value_unit: Operation.masked_convert_value_unit,
    Operation.convert_temperature_unit: Operation.masked_convert_temperature_unit,
}
MASKED_VALUE_CONVERSIONS = frozenset(VALUE_CONVERSIONS.values())
CONVERSION_TRANSFORMS = VALUE_CONVERSIONS.keys() | {Operation.convert_axis_unit}
CONVERSION_VALUES = MASKED_VALUE_CONVERSIONS | {Operation.convert_axis_unit}
PAIRED_CAST_OPERATIONS = {Operation.cast: Operation.masked_cast_with_placeholders}
PAIRED_CAST_VALUES = frozenset(PAIRED_CAST_OPERATIONS.values())
PAIRED_SCALAR_OPERATION = Operation.masked_scalar_arithmetic_with_placeholders
PAIRED_SCALAR_REQUEST = "scalar_arithmetic"
_PINNED_TRANSFORM_SCHEMA_OPERATIONS: list[JsonValue] = [
    str(operation)
    for operation in sorted(
        PAIRED_REDUCTION_OPERATIONS.keys()
        | PAIRED_SCALING_OPERATIONS.keys()
        | PAIRED_CAST_OPERATIONS.keys()
        | POSITIONAL_TRANSFORMS
        | CONVERSION_TRANSFORMS
        | {PAIRED_SCALAR_REQUEST}
    )
]
BINARY_TRANSFORMS = {
    value.value: operation for operation, value in MASKED_BINARY_OPERATIONS.items()
}
FILLED_COORDINATE_MODES = frozenset(
    ("left_filled", "right_filled", "left_filled_permuted", "right_filled_permuted")
)
V2_PAIRED_TRANSFORMS = (
    TRANSFORMS
    | {Operation.reindex_coordinates}
    | PAIRED_REDUCTION_OPERATIONS.keys()
    | PAIRED_SCALING_OPERATIONS.keys()
    | PAIRED_CAST_OPERATIONS.keys()
    | {PAIRED_SCALAR_OPERATION}
    | set(MASKED_BINARY_OPERATIONS.values())
    | VALUE_CONVERSIONS.keys()
)
V2_PAIRED_TRANSFORM_REQUESTS = frozenset(
    str(operation) for operation in V2_PAIRED_TRANSFORMS if operation != PAIRED_SCALAR_OPERATION
) | {PAIRED_SCALAR_REQUEST}
Hash = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
def _transform_operation(operation: str) -> Operation:
    return PAIRED_SCALAR_OPERATION if operation == PAIRED_SCALAR_REQUEST else Operation(operation)
def validity_transform_operations(request: str) -> tuple[Operation, Operation]:
    if request not in V2_PAIRED_TRANSFORM_REQUESTS:
        raise TensorusError("Unsupported validity transformation")
    source = _transform_operation(request)
    if request in BINARY_TRANSFORMS:
        return BINARY_TRANSFORMS[request], Operation.logical_and
    if source in PAIRED_REDUCTION_OPERATIONS:
        return (
            PAIRED_REDUCTION_OPERATIONS[source],
            Operation.chunked_reduce_validity_any
            if request.startswith("chunked_")
            else Operation.reduce_validity_any,
        )
    if source in PAIRED_SCALING_OPERATIONS:
        return PAIRED_SCALING_OPERATIONS[source], Operation.cast
    if source in PAIRED_CAST_OPERATIONS:
        return PAIRED_CAST_OPERATIONS[source], Operation.cast
    if source in VALUE_CONVERSIONS:
        return VALUE_CONVERSIONS[source], Operation.cast
    if source == PAIRED_SCALAR_OPERATION:
        return source, Operation.cast
    if source == Operation.reindex_coordinates:
        return source, Operation.reindex_validity
    if source in POSITIONAL_TRANSFORMS or source == Operation.convert_axis_unit:
        return source, source
    raise TensorusError("Unsupported validity transformation")
class ValidityBindingCreate(Model):
    data_version_id: UUID
    mask_version_id: UUID
class AssertedValidityOrigin(Model):
    kind: Literal["asserted"] = "asserted"
class TransformedValidityOrigin(Model):
    kind: Literal["transformed"] = "transformed"
    binding_id: UUID
    binding_hash: Hash
    plan_id: UUID
    plan_hash: Hash
class BinaryTransformedValidityOrigin(Model):
    kind: Literal["binary_transformed"] = "binary_transformed"
    sources: list[PlanValiditySource] = Field(min_length=2, max_length=2)
    plan_id: UUID
    plan_hash: Hash

    @model_validator(mode="after")
    def distinct_sources(self) -> Self:
        if self.sources[0].binding_id == self.sources[1].binding_id:
            raise ValueError("Binary validity sources must be distinct")
        return self
ValidityOrigin = Annotated[
    AssertedValidityOrigin | TransformedValidityOrigin | BinaryTransformedValidityOrigin,
    Field(discriminator="kind"),
]
class ValidityBinding(Model):
    format: Literal["tensorus-observation-validity-v1", "tensorus-observation-validity-v2"] = (
        VALIDITY_FORMAT
    )
    id: UUID
    tenant_id: UUID
    data_version_id: UUID
    data_checksum: Hash
    data_dtype: DType
    mask_version_id: UUID
    mask_checksum: Hash
    shape: list[int]
    origin: ValidityOrigin
    descriptor_hash: Hash
    created_at: str
    created_by: str

    @model_validator(mode="after")
    def format_matches_origin(self) -> Self:
        if (self.format == BINARY_VALIDITY_FORMAT) != isinstance(
            self.origin, BinaryTransformedValidityOrigin
        ):
            raise ValueError("Validity format does not match its provenance")
        return self
class ValidityBindingList(Model):
    items: list[ValidityBinding]
class ValidityCoordinateWindow(Model):
    dataset: Name
    expected_binding_hash: Hash
    selection: dict[Name, CoordinateRange] = Field(min_length=1, max_length=16)
    data_output: Name
    mask_output: Name
    access_policy: Literal["source_datasets_v1"]
    execution: Literal["eager", "chunked_verified"] = Field(
        default="eager", exclude_if=lambda value: value == "eager"
    )

    @model_validator(mode="after")
    def distinct_outputs(self) -> Self:
        if self.data_output == self.mask_output:
            raise ValueError("Data and mask output names must differ")
        return self
class FittedPreprocessingFit(Model):
    dataset: Name
    expected_binding_hash: Hash
    state_output: Name
    parameters: FittedScalingParameters
    access_policy: Literal["source_datasets_v1"]
class FittedPreprocessingApply(Model):
    dataset: Name
    expected_binding_hash: Hash
    state_version_id: UUID
    expected_state_checksum: Hash
    data_output: Name
    mask_output: Name
    access_policy: Literal["source_datasets_v1"]
    imported_state: ImportedFittedStateSelection | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def distinct_outputs(self) -> Self:
        if self.data_output == self.mask_output:
            raise ValueError("Data and mask output names must differ")
        return self
class FittedPreprocessingState(Model):
    format: Literal[
        "tensorus-fitted-preprocessing-state-v1", "tensorus-fitted-preprocessing-state-v2"
    ]
    state_version_id: UUID
    state_checksum: Hash
    state_shape: list[Annotated[StrictInt, Field(ge=0)]] = Field(max_length=16)
    state_dtype: Literal["float64"] = "float64"
    algorithm: Literal["l2_normalize", "standardize"]
    algorithm_version: Annotated[StrictInt, Field(ge=1, le=2)]
    axis: Annotated[list[StrictInt], Field(max_length=16)] | StrictInt | None
    epsilon: Annotated[float, Field(strict=True, gt=0, le=1)]
    training_binding: ValidityBinding
    training_shape: list[Annotated[StrictInt, Field(ge=0)]] = Field(max_length=16)
    training_semantics: TensorSemantics
    fit_plan_id: UUID
    fit_plan_hash: Hash
    inverse_supported: Literal[False] = False
    independent_recomputation: Literal[False] = False
    limitations: list[str]

    @model_validator(mode="after")
    def consistent_domain(self) -> Self:
        try:
            expected_shape = fitted_state_shape(
                tuple(self.training_shape), self.axis, self.algorithm_version
            )
        except TensorusError as exc:
            raise ValueError("Fitted state reduction axes are invalid") from exc
        if (
            self.training_binding.shape != self.training_shape
            or self.training_semantics.version != 2
            or not math.prod(self.training_shape)
            or tuple(self.state_shape) != expected_shape
            or self.format != f"tensorus-fitted-preprocessing-state-v{self.algorithm_version}"
        ):
            raise ValueError("Fitted state shape and training domain are inconsistent")
        try:
            read_semantics(
                {"tensorus_semantics": self.training_semantics.model_dump(mode="json")},
                tuple(self.training_shape),
            )
        except TensorusError as exc:
            raise ValueError("Fitted state training semantics are invalid") from exc
        return self
class ReviewedImportedFittedState(Model):
    format: Literal["tensorus-reviewed-imported-fitted-state-v1"] = (
        "tensorus-reviewed-imported-fitted-state-v1"
    )
    state_version_id: UUID
    state_checksum: Hash
    state_shape: list[Annotated[StrictInt, Field(ge=0)]] = Field(max_length=16)
    state_dtype: Literal["float64"] = "float64"
    algorithm: Literal["l2_normalize", "standardize"]
    algorithm_version: Annotated[StrictInt, Field(ge=1, le=2)]
    axis: Annotated[list[StrictInt], Field(max_length=16)] | StrictInt | None
    epsilon: Annotated[float, Field(strict=True, gt=0, le=1)]
    training_binding: ValidityBinding
    training_shape: list[Annotated[StrictInt, Field(ge=0)]] = Field(max_length=16)
    training_semantics: TensorSemantics
    selection: ImportedFittedStateSelection
    claimed_fit_plan_id: UUID
    claimed_fit_plan_hash: Hash
    claimed_fit_outcome_hash: Hash
    local_reviewed_by: str
    local_reviewed_at: str
    source_claims_authenticated: Literal[False] = False
    independent_recomputation: Literal[False] = False
    inverse_supported: Literal[False] = False
    limitations: list[str]

    @model_validator(mode="after")
    def consistent_domain(self) -> Self:
        try:
            expected_shape = fitted_state_shape(
                tuple(self.training_shape), self.axis, self.algorithm_version
            )
            read_semantics(
                {"tensorus_semantics": self.training_semantics.model_dump(mode="json")},
                tuple(self.training_shape),
            )
        except TensorusError as exc:
            raise ValueError("Imported fitted state training domain is invalid") from exc
        if (
            self.training_binding.shape != self.training_shape
            or self.training_semantics.version != 2
            or not math.prod(self.training_shape)
            or tuple(self.state_shape) != expected_shape
        ):
            raise ValueError("Imported fitted state shape and training domain are inconsistent")
        return self
class ValidityTransform(Model):
    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "required": ["expected_binding_hash"],
                        "properties": {"expected_binding_hash": {"type": "string"}},
                    },
                    "then": {
                        "required": ["access_policy"],
                        "properties": {
                            "operation": {"enum": _PINNED_TRANSFORM_SCHEMA_OPERATIONS},
                            "access_policy": {"const": "source_datasets_v1"},
                        },
                    },
                },
                {
                    "if": {
                        "required": ["operation"],
                        "properties": {
                            "operation": {
                                "enum": ["normalize", "standardize", "cast", "scalar_arithmetic"]
                            }
                        },
                    },
                    "then": {
                        "required": ["expected_binding_hash", "access_policy"],
                        "properties": {
                            "expected_binding_hash": {"type": "string"},
                            "access_policy": {"const": "source_datasets_v1"},
                        },
                    },
                },
            ]
        }
    )
    dataset: Name
    operation: Literal[
        "transpose",
        "slice",
        "chunked_slice",
        "select_indices",
        "convert_axis_unit",
        "convert_value_unit",
        "convert_temperature_unit",
        "cast",
        "scalar_arithmetic",
        "normalize",
        "standardize",
        "reindex_coordinates",
        "sum",
        "mean",
        "min",
        "max",
        "variance",
        "std",
        "chunked_count_nonzero_by_axis",
        "chunked_sum_by_axis",
        "chunked_mean_by_axis",
        "chunked_variance_by_axis",
        "chunked_std_by_axis",
        "chunked_min_by_axis",
        "chunked_max_by_axis",
        "add",
        "subtract",
        "multiply",
        "divide",
    ]
    parameters: dict[str, JsonValue] = Field(default_factory=dict, max_length=8)
    expected_binding_hash: Hash | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "For an advertised v2 paired reduction, positional transform, unit conversion, "
            "selected scaling, exact selected casting or selected scalar arithmetic, "
            "pin this exact source binding."
        ),
    )
    selected_preprocessing_policy: Literal["bound_selected_v1"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "For v2 filled paired arithmetic, compare explicit matching source identities "
            "only at positions included by both bound masks."
        ),
    )
    anchor_version_id: UUID | None = None
    other_binding_id: UUID | None = Field(default=None, exclude_if=lambda value: value is None)
    coordinate_mode: Literal[
        "exact",
        "inner_indexed",
        "inner_indexed_permuted",
        "left_filled",
        "right_filled",
        "left_filled_permuted",
        "right_filled_permuted",
    ] = Field(default="exact", exclude_if=lambda value: value == "exact")
    aligned_data_output: Name | None = Field(default=None, exclude_if=lambda value: value is None)
    aligned_mask_output: Name | None = Field(default=None, exclude_if=lambda value: value is None)
    left_transposed_data_output: Name | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    left_transposed_mask_output: Name | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    right_transposed_data_output: Name | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    right_transposed_mask_output: Name | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    left_data_output: Name | None = Field(default=None, exclude_if=lambda value: value is None)
    left_mask_output: Name | None = Field(default=None, exclude_if=lambda value: value is None)
    right_data_output: Name | None = Field(default=None, exclude_if=lambda value: value is None)
    right_mask_output: Name | None = Field(default=None, exclude_if=lambda value: value is None)
    data_output: Name
    mask_output: Name
    access_policy: Literal["source_datasets_v1"] | None = None

    @model_validator(mode="after")
    def distinct_outputs(self) -> Self:
        outputs = [
            output
            for output in (
                self.data_output,
                self.mask_output,
                self.aligned_data_output,
                self.aligned_mask_output,
                self.left_transposed_data_output,
                self.left_transposed_mask_output,
                self.right_transposed_data_output,
                self.right_transposed_mask_output,
                self.left_data_output,
                self.left_mask_output,
                self.right_data_output,
                self.right_mask_output,
            )
            if output is not None
        ]
        if len(set(outputs)) != len(outputs):
            raise ValueError("Data, mask and aligned output names must differ")
        if self.operation != "reindex_coordinates" and self.anchor_version_id is not None:
            raise ValueError("An anchor is only supported for coordinate reindexing")
        if self.operation in BINARY_TRANSFORMS:
            if self.other_binding_id is None or self.access_policy != "source_datasets_v1":
                raise ValueError("Paired arithmetic requires a second binding and source access")
        elif self.other_binding_id is not None:
            raise ValueError("A second validity binding is only supported for paired arithmetic")
        if self.expected_binding_hash is not None and (
            _transform_operation(self.operation)
            not in PAIRED_REDUCTION_OPERATIONS.keys()
            | PAIRED_SCALING_OPERATIONS.keys()
            | PAIRED_CAST_OPERATIONS.keys()
            | {PAIRED_SCALAR_OPERATION}
            | POSITIONAL_TRANSFORMS
            | CONVERSION_TRANSFORMS
            or self.access_policy != "source_datasets_v1"
        ):
            raise ValueError("A pinned transform requires a supported operation and source access")
        if (
            _transform_operation(self.operation)
            in PAIRED_SCALING_OPERATIONS.keys()
            | PAIRED_CAST_OPERATIONS.keys()
            | {PAIRED_SCALAR_OPERATION}
            and self.expected_binding_hash is None
        ):
            raise ValueError("Selected operations require an exact source binding hash")
        filled = self.coordinate_mode in FILLED_COORDINATE_MODES
        if self.selected_preprocessing_policy is not None and (
            not filled or self.operation not in BINARY_TRANSFORMS
        ):
            raise ValueError("Selected preprocessing requires filled paired arithmetic")
        if self.operation in BINARY_TRANSFORMS and "selected_preprocessing_policy" in (
            self.parameters
        ):
            raise ValueError("Specify selected preprocessing as a top-level review policy")
        indexed = self.coordinate_mode in ("inner_indexed", "inner_indexed_permuted")
        right_permuted = self.coordinate_mode in (
            "inner_indexed_permuted",
            "left_filled_permuted",
        )
        left_permuted = self.coordinate_mode == "right_filled_permuted"
        if self.coordinate_mode != "exact" and self.operation not in BINARY_TRANSFORMS:
            raise ValueError("Coordinate alignment is only supported for paired arithmetic")
        if (filled and (self.aligned_data_output is None or self.aligned_mask_output is None)) or (
            not filled
            and (self.aligned_data_output is not None or self.aligned_mask_output is not None)
        ):
            raise ValueError(
                "Filled alignment requires both aligned output names, and other modes do not"
            )
        selected = (
            self.left_data_output,
            self.left_mask_output,
            self.right_data_output,
            self.right_mask_output,
        )
        if (indexed and any(output is None for output in selected)) or (
            not indexed and any(output is not None for output in selected)
        ):
            raise ValueError("Indexed intersection requires all four selected output names")
        for required, transposed in (
            (left_permuted, (self.left_transposed_data_output, self.left_transposed_mask_output)),
            (
                right_permuted,
                (self.right_transposed_data_output, self.right_transposed_mask_output),
            ),
        ):
            if (required and any(output is None for output in transposed)) or (
                not required and any(output is not None for output in transposed)
            ):
                raise ValueError("Permuted alignment requires both source transposed output names")
        return self
class ValidityTransformCompletion(Model):
    plan_id: UUID
    expected_plan_hash: Hash
def _hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()
def _stable_descriptor(record: Record) -> Record:
    fields = (
        "id",
        "tenant_id",
        "dataset_id",
        "tensor_id",
        "version",
        "dtype",
        "shape",
        "nbytes",
        "checksum",
        "metadata",
        "operation",
        "parents",
        "write_contract",
    )
    descriptor = {field: record[field] for field in fields}
    descriptor["access"] = record.get("access")
    return descriptor
def _descriptor_hash(
    data: Record,
    mask: Record,
    origin: Record,
    binding_id: str,
    created_at: str,
    created_by: str,
) -> str:
    return _hash(
        {
            "format": (
                BINARY_VALIDITY_FORMAT
                if origin.get("kind") == "binary_transformed"
                else VALIDITY_FORMAT
            ),
            "id": binding_id,
            "created_at": created_at,
            "created_by": created_by,
            "data": _stable_descriptor(data),
            "mask": _stable_descriptor(mask),
            "origin": origin,
        }
    )
def _check_pair_descriptors(data: Record, mask: Record) -> None:
    descriptor_nbytes(data)
    descriptor_nbytes(mask)
    if mask["dtype"] != "bool" or mask["shape"] != data["shape"]:
        raise TensorusError("A validity mask must be boolean and have the exact data shape")
    data_semantics = read_semantics(data["metadata"], tuple(data["shape"]), data["operation"])
    mask_semantics = read_semantics(mask["metadata"], tuple(mask["shape"]), mask["operation"])
    if data_semantics is None or mask_semantics is None:
        raise TensorusError("Validity bindings require versioned semantics for data and mask")
    require_matching_axes(data_semantics, mask_semantics, tuple(data["shape"]))
def _matching_reindex_parameters(values: Step, mask: Step) -> bool:
    try:
        values_parameters = parameters(values)
        mask_parameters = parameters(mask)
    except TensorusError:
        return False
    return canonical_json(values_parameters["axes"]) == canonical_json(
        mask_parameters["axes"]
    ) and canonical_json(values_parameters.get("nearest")) == canonical_json(
        mask_parameters.get("nearest")
    )
def _pinned_v2_reduction(definition: PlanDefinition) -> bool:
    return (
        definition.semantic_policy == "v2"
        and definition.validity_sources is not None
        and len(definition.validity_sources) == 1
        and len(definition.steps) == 2
        and definition.steps[0].operation in PAIRED_REDUCTION_VALUES
    )
def _pinned_v2_positional(definition: PlanDefinition) -> bool:
    return definition.paired_validity_policy == "pinned_positional_v1"
def _pinned_v2_conversion(definition: PlanDefinition) -> bool:
    return definition.paired_validity_policy == "pinned_conversion_v1"
def _pinned_v2_scaling(definition: PlanDefinition) -> bool:
    return definition.paired_validity_policy == "pinned_scaling_v1"
def _pinned_v2_cast(definition: PlanDefinition) -> bool:
    return definition.paired_validity_policy == "pinned_cast_v1"
def _pinned_v2_scalar(definition: PlanDefinition) -> bool:
    return definition.paired_validity_policy == "pinned_scalar_v1"
def _pinned_fitted_fit(definition: PlanDefinition) -> bool:
    return definition.paired_validity_policy == "pinned_fitted_fit_v1"
def _pinned_fitted_apply(definition: PlanDefinition) -> bool:
    return definition.paired_validity_policy == "pinned_fitted_apply_v1"
def _paired_definition(
    definition: PlanDefinition,
    data_id: str,
    mask_id: str,
    binding_id: str | None = None,
    binding_hash: str | None = None,
) -> bool:
    if definition.semantic_policy not in ("v1", "v2"):
        return False
    if _pinned_fitted_fit(definition):
        source = definition.validity_sources[0] if definition.validity_sources else None
        if len(definition.steps) != 1 or source is None:
            return False
        step = definition.steps[0]
        return (
            definition.semantic_policy == "v2"
            and definition.access_policy == "source_datasets_v1"
            and len(definition.validity_sources or []) == 1
            and definition.coordinate_window is None
            and str(source.binding_id) == binding_id
            and source.descriptor_hash == binding_hash
            and step.id == "state"
            and step.operation == Operation.fit_selected_preprocessing
            and step.inputs == [data_id, mask_id]
        )
    if len(definition.steps) != 2:
        return False
    if _pinned_v2_reduction(definition) and definition.access_policy != "source_datasets_v1":
        return False
    if definition.validity_sources is not None:
        if len(definition.validity_sources) != 1:
            return False
        source = definition.validity_sources[0]
        if str(source.binding_id) != binding_id or source.descriptor_hash != binding_hash:
            return False
    values, mask = definition.steps
    if values.id != "values" or mask.id != "validity":
        return False
    if _pinned_v2_positional(definition) and (
        definition.semantic_policy != "v2"
        or definition.access_policy != "source_datasets_v1"
        or definition.validity_sources is None
        or definition.coordinate_window is not None
        or values.operation not in POSITIONAL_TRANSFORMS
    ):
        return False
    if _pinned_v2_conversion(definition) and (
        definition.semantic_policy != "v2"
        or definition.access_policy != "source_datasets_v1"
        or definition.validity_sources is None
        or definition.coordinate_window is not None
        or values.operation not in CONVERSION_VALUES
    ):
        return False
    if _pinned_v2_scaling(definition) and (
        definition.semantic_policy != "v2"
        or definition.access_policy != "source_datasets_v1"
        or definition.validity_sources is None
        or definition.coordinate_window is not None
        or values.operation not in PAIRED_SCALING_VALUES
    ):
        return False
    if _pinned_v2_cast(definition) and (
        definition.semantic_policy != "v2"
        or definition.access_policy != "source_datasets_v1"
        or definition.validity_sources is None
        or definition.coordinate_window is not None
        or values.operation not in PAIRED_CAST_VALUES
    ):
        return False
    if _pinned_v2_scalar(definition) and (
        definition.semantic_policy != "v2"
        or definition.access_policy != "source_datasets_v1"
        or definition.validity_sources is None
        or definition.coordinate_window is not None
        or values.operation != PAIRED_SCALAR_OPERATION
    ):
        return False
    if _pinned_fitted_apply(definition) and (
        definition.semantic_policy != "v2"
        or definition.access_policy != "source_datasets_v1"
        or definition.validity_sources is None
        or definition.coordinate_window is not None
        or values.operation != Operation.apply_fitted_preprocessing
    ):
        return False
    if definition.coordinate_window is not None and values.operation not in (
        Operation.slice,
        Operation.chunked_slice,
    ):
        return False
    if definition.semantic_policy == "v2":
        if values.operation not in (
            TRANSFORMS
            | PAIRED_REDUCTION_VALUES
            | PAIRED_SCALING_VALUES
            | PAIRED_CAST_VALUES
            | {PAIRED_SCALAR_OPERATION}
            | {Operation.apply_fitted_preprocessing}
            | MASKED_VALUE_CONVERSIONS
            | {Operation.reindex_coordinates}
        ):
            return False
        if (
            values.operation
            in (
                MASKED_VALUE_CONVERSIONS
                | PAIRED_SCALING_VALUES
                | PAIRED_CAST_VALUES
                | {PAIRED_SCALAR_OPERATION}
                | {Operation.apply_fitted_preprocessing}
            )
            or values.operation in (Operation.convert_axis_unit, Operation.reindex_coordinates)
        ) and definition.access_policy != "source_datasets_v1":
            return False
        if values.operation == Operation.reindex_coordinates and (
            definition.validity_sources is None or len(values.inputs) != 2
        ):
            return False
    if values.operation == Operation.apply_fitted_preprocessing:
        return (
            _pinned_fitted_apply(definition)
            and len(values.inputs) == 3
            and values.inputs[0] == data_id
            and values.inputs[2] == mask_id
            and not values.inputs[1].startswith("@")
            and mask.operation == Operation.cast
            and mask.inputs == [mask_id]
            and mask.parameters == {"dtype": "bool"}
        )
    if values.operation in TRANSFORMS:
        return (
            mask.operation == values.operation
            and values.inputs == [data_id]
            and mask.inputs == [mask_id]
            and canonical_json(values.parameters) == canonical_json(mask.parameters)
        )
    if values.operation in (
        MASKED_VALUE_CONVERSIONS
        | PAIRED_SCALING_VALUES
        | PAIRED_CAST_VALUES
        | {PAIRED_SCALAR_OPERATION}
    ):
        return (
            mask.operation == Operation.cast
            and values.inputs == [data_id, mask_id]
            and mask.inputs == [mask_id]
            and mask.parameters == {"dtype": "bool"}
        )
    if values.operation in PAIRED_REDUCTION_VALUES:
        expected_mask = (
            Operation.chunked_reduce_validity_any
            if values.operation in CHUNKED_PAIRED_REDUCTION_VALUES
            else Operation.reduce_validity_any
        )
        return (
            mask.operation == expected_mask
            and values.inputs == [data_id, mask_id]
            and mask.inputs == [mask_id]
            and canonical_json(values.parameters) == canonical_json(mask.parameters)
        )
    return (
        values.operation == Operation.reindex_coordinates
        and mask.operation == Operation.reindex_validity
        and len(values.inputs) in (1, 2)
        and len(mask.inputs) == len(values.inputs)
        and values.inputs[0] == data_id
        and mask.inputs[0] == mask_id
        and values.inputs[1:] == mask.inputs[1:]
        and _matching_reindex_parameters(values, mask)
    )
def _v2_unary_definition(definition: PlanDefinition, operations: Collection[Operation]) -> bool:
    if (
        definition.semantic_policy != "v2"
        or len(definition.steps) != 2
        or definition.steps[0].operation not in operations
    ):
        return False
    values, mask = definition.steps
    if (
        not values.inputs
        or not mask.inputs
        or any(reference.startswith("@") for step in (values, mask) for reference in step.inputs)
    ):
        return False
    source = definition.validity_sources[0] if definition.validity_sources else None
    return _paired_definition(
        definition,
        values.inputs[0],
        mask.inputs[0],
        str(source.binding_id) if source else None,
        source.descriptor_hash if source else None,
    )
def _v2_positional_definition(definition: PlanDefinition) -> bool:
    return _v2_unary_definition(definition, POSITIONAL_TRANSFORMS)
def _v2_conversion_definition(definition: PlanDefinition) -> bool:
    return _v2_unary_definition(definition, CONVERSION_VALUES)
def _v2_scaling_definition(definition: PlanDefinition) -> bool:
    return _pinned_v2_scaling(definition) and _v2_unary_definition(
        definition, PAIRED_SCALING_VALUES
    )
def _v2_cast_definition(definition: PlanDefinition) -> bool:
    return _pinned_v2_cast(definition) and _v2_unary_definition(definition, PAIRED_CAST_VALUES)
def _v2_scalar_definition(definition: PlanDefinition) -> bool:
    return _pinned_v2_scalar(definition) and _v2_unary_definition(
        definition, {PAIRED_SCALAR_OPERATION}
    )
def _v2_fitted_fit_definition(definition: PlanDefinition) -> bool:
    if not _pinned_fitted_fit(definition) or len(definition.steps) != 1:
        return False
    step = definition.steps[0]
    source = definition.validity_sources[0] if definition.validity_sources else None
    return (
        len(step.inputs) == 2
        and source is not None
        and not any(reference.startswith("@") for reference in step.inputs)
        and _paired_definition(
            definition,
            step.inputs[0],
            step.inputs[1],
            str(source.binding_id),
            source.descriptor_hash,
        )
    )
def _v2_fitted_apply_definition(definition: PlanDefinition) -> bool:
    return _pinned_fitted_apply(definition) and _v2_unary_definition(
        definition, {Operation.apply_fitted_preprocessing}
    )
def _paired_binary_definition(definition: PlanDefinition, bindings: list[Record]) -> bool:
    if (
        len(bindings) != 2
        or definition.semantic_policy not in ("v1", "v2")
        or definition.access_policy != "source_datasets_v1"
        or definition.validity_sources is None
        or len(definition.steps) not in (2, 4, 6, 8)
        or (definition.semantic_policy == "v1" and len(definition.steps) in (6, 8))
    ):
        return False
    expected_sources = [
        {"binding_id": binding["id"], "descriptor_hash": binding["descriptor_hash"]}
        for binding in bindings
    ]
    if [
        source.model_dump(mode="json") for source in definition.validity_sources
    ] != expected_sources:
        return False
    data_inputs = [binding["data_version_id"] for binding in bindings]
    mask_inputs = [binding["mask_version_id"] for binding in bindings]
    mask, values = definition.steps[-2:]
    selected_policy = values.parameters.get("selected_preprocessing_policy")
    expected_parameters: dict[str, JsonValue] = {"algorithm_version": 1}
    if selected_policy == BOUND_SELECTED_PREPROCESSING:
        if definition.semantic_policy != "v2" or not (
            len(definition.steps) == 4
            or (len(definition.steps) == 6 and definition.steps[0].operation == Operation.transpose)
        ):
            return False
        expected_parameters["selected_preprocessing_policy"] = BOUND_SELECTED_PREPROCESSING

    def matches_outputs(data_refs: list[str], mask_refs: list[str]) -> bool:
        return (
            mask.id == "validity"
            and mask.operation == Operation.logical_and
            and mask.inputs == mask_refs
            and mask.parameters == {}
            and values.id == "values"
            and values.operation in MASKED_BINARY_OPERATIONS
            and values.inputs == [*data_refs, "@validity"]
            and canonical_json(values.parameters) == canonical_json(expected_parameters)
        )

    if len(definition.steps) == 2:
        return matches_outputs(data_inputs, mask_inputs)
    if len(definition.steps) == 4 or (
        len(definition.steps) == 6 and definition.steps[0].operation == Operation.transpose
    ):
        offset = len(definition.steps) - 4
        aligned_data, aligned_mask = definition.steps[offset : offset + 2]
        if (
            aligned_data.id != "aligned_data"
            or aligned_data.operation != Operation.reindex_coordinates
            or aligned_mask.id != "aligned_mask"
            or aligned_mask.operation != Operation.reindex_validity
            or not _matching_reindex_parameters(aligned_data, aligned_mask)
            or aligned_data.parameters.get("nearest") is not None
            or type(aligned_data.parameters["fill_value"]) not in (int, bool)
            or aligned_data.parameters["fill_value"] not in (0, False)
        ):
            return False
        for source in (0, 1):
            anchor = 1 - source
            source_data, source_mask = data_inputs[source], mask_inputs[source]
            if offset:
                side = "left" if source == 0 else "right"
                transposed_data, transposed_mask = definition.steps[:2]
                if (
                    transposed_data.id != f"transpose_{side}_data"
                    or transposed_mask.id != f"transpose_{side}_mask"
                    or transposed_data.operation != Operation.transpose
                    or transposed_mask.operation != Operation.transpose
                    or transposed_data.inputs != [source_data]
                    or transposed_mask.inputs != [source_mask]
                    or set(transposed_data.parameters) != {"axes"}
                    or canonical_json(transposed_data.parameters)
                    != canonical_json(transposed_mask.parameters)
                ):
                    continue
                source_data, source_mask = f"@{transposed_data.id}", f"@{transposed_mask.id}"
            if aligned_data.inputs == [
                source_data,
                data_inputs[anchor],
            ] and aligned_mask.inputs == [source_mask, data_inputs[anchor]]:
                aligned_data_inputs = data_inputs.copy()
                aligned_mask_inputs = mask_inputs.copy()
                aligned_data_inputs[source] = "@aligned_data"
                aligned_mask_inputs[source] = "@aligned_mask"
                if matches_outputs(aligned_data_inputs, aligned_mask_inputs):
                    return True
        return False
    if len(definition.steps) in (6, 8):
        offset = 0
        if len(definition.steps) == 8:
            transposed_data, transposed_mask = definition.steps[:2]
            if (
                transposed_data.id != "transpose_right_data"
                or transposed_mask.id != "transpose_right_mask"
                or transposed_data.operation != Operation.transpose
                or transposed_mask.operation != Operation.transpose
                or transposed_data.inputs != [data_inputs[1]]
                or transposed_mask.inputs != [mask_inputs[1]]
                or set(transposed_data.parameters) != {"axes"}
                or canonical_json(transposed_data.parameters)
                != canonical_json(transposed_mask.parameters)
            ):
                return False
            data_inputs[1] = "@transpose_right_data"
            mask_inputs[1] = "@transpose_right_mask"
            offset = 2
        selected = definition.steps[offset : offset + 4]
        ids = ["select_left_data", "select_left_mask", "select_right_data", "select_right_mask"]
        refs = [data_inputs[0], mask_inputs[0], data_inputs[1], mask_inputs[1]]
        if any(
            step.id != step_id
            or step.operation != Operation.select_indices
            or step.inputs != [ref]
            or set(step.parameters) != {"indices"}
            for step, step_id, ref in zip(selected, ids, refs, strict=True)
        ):
            return False
        if canonical_json(selected[0].parameters) != canonical_json(
            selected[1].parameters
        ) or canonical_json(selected[2].parameters) != canonical_json(selected[3].parameters):
            return False
        return matches_outputs(
            ["@select_left_data", "@select_right_data"],
            ["@select_left_mask", "@select_right_mask"],
        )
    return False
def _v2_window_plan_matches(definition: PlanDefinition, source: Record) -> bool:
    if definition.coordinate_window is None:
        return True
    if (
        definition.semantic_policy != "v2"
        or len(definition.steps) != 2
        or definition.steps[0].operation not in (Operation.slice, Operation.chunked_slice)
    ):
        return False
    semantics = read_semantics(source["metadata"], tuple(source["shape"]), source["operation"])
    if semantics is None or semantics.version != 2:
        return False
    named = resolve_coordinate_axis_selection(source, definition.coordinate_window)
    positional = [named.selection.get(axis.label, [None, None]) for axis in semantics.axes]
    return canonical_json(definition.steps[0].parameters) == canonical_json(
        {"selection": positional}
    )
def _v2_alignment_plan_matches(definition: PlanDefinition, sources: list[Record]) -> bool:
    if definition.semantic_policy != "v2" or len(definition.steps) == 2:
        return True
    if len(sources) != 2 or len(definition.steps) not in (4, 6, 8):
        return False
    shapes = [tuple(source["shape"]) for source in sources]
    semantics = [
        read_semantics(source["metadata"], shape, source["operation"])
        for source, shape in zip(sources, shapes, strict=True)
    ]
    left_semantics, right_semantics = semantics
    if left_semantics is None or right_semantics is None:
        return False
    typed_semantics = (left_semantics, right_semantics)
    if len(definition.steps) == 4:
        for source_index in (0, 1):
            anchor_index = 1 - source_index
            if definition.steps[0].inputs != [
                sources[source_index]["id"],
                sources[anchor_index]["id"],
            ]:
                continue
            permutation, mapping, _ = referenced_coordinate_filled_alignment(
                left_semantics,
                right_semantics,
                shapes[0],
                shapes[1],
                "left" if anchor_index == 0 else "right",
            )
            axes = [item.model_dump(mode="json") if item is not None else None for item in mapping]
            return permutation == list(range(len(permutation))) and canonical_json(
                definition.steps[0].parameters["axes"]
            ) == canonical_json(axes)
        return False
    if len(definition.steps) == 6 and definition.steps[0].operation == Operation.transpose:
        for source_index in (0, 1):
            side = "left" if source_index == 0 else "right"
            if definition.steps[0].id != f"transpose_{side}_data":
                continue
            anchor_index = 1 - source_index
            permutation, mapping, _ = referenced_coordinate_filled_alignment(
                typed_semantics[anchor_index],
                typed_semantics[source_index],
                shapes[anchor_index],
                shapes[source_index],
                "left",
            )
            axes = [item.model_dump(mode="json") if item is not None else None for item in mapping]
            return canonical_json(definition.steps[0].parameters) == canonical_json(
                {"axes": permutation}
            ) and canonical_json(definition.steps[2].parameters["axes"]) == canonical_json(axes)
        return False
    offset = 0
    if len(definition.steps) == 8:
        permutation, left, right, _ = referenced_coordinate_index_alignment(
            left_semantics, right_semantics, shapes[0], shapes[1]
        )
        if canonical_json(definition.steps[0].parameters) != canonical_json({"axes": permutation}):
            return False
        offset = 2
    else:
        left, right, _ = referenced_coordinate_index_intersection(
            left_semantics, right_semantics, shapes[0], shapes[1]
        )
    return canonical_json(definition.steps[offset].parameters) == canonical_json(
        {"indices": left}
    ) and canonical_json(definition.steps[offset + 2].parameters) == canonical_json(
        {"indices": right}
    )
def _has_selected_preprocessing(definition: PlanDefinition) -> bool:
    return any(
        step.parameters.get("selected_preprocessing_policy") == BOUND_SELECTED_PREPROCESSING
        for step in definition.steps
    )
def _selected_preprocessing(
    definition: PlanDefinition, bindings: list[Record], sources: list[Record]
) -> tuple[PreprocessingIdentity, PreprocessingIdentity] | None:
    if not _has_selected_preprocessing(definition):
        return None
    if (
        len(bindings) != 2
        or len(sources) != 2
        or [str(source["id"]) for source in sources]
        != [binding["data_version_id"] for binding in bindings]
        or not _paired_binary_definition(definition, bindings)
        or not _v2_alignment_plan_matches(definition, sources)
    ):
        raise TensorusError("Selected preprocessing requires the exact reviewed filled graph")
    declared = [
        read_semantics(source["metadata"], tuple(source["shape"]), source["operation"])
        for source in sources
    ]
    left, right = declared
    if (
        left is None
        or right is None
        or left.version != 2
        or right.version != 2
        or left.preprocessing is None
        or right.preprocessing is None
        or left.preprocessing != right.preprocessing
    ):
        raise TensorusError("Selected preprocessing requires matching explicit source identities")
    return left.preprocessing, right.preprocessing
def _coordinate_step_matches(
    step: Step,
    sources: list[Record],
    output: Record,
    policy: Literal["v1", "v2"] | None,
) -> bool:
    if step.operation not in (
        Operation.reindex_coordinates,
        Operation.reindex_validity,
        Operation.coordinate_coverage,
    ):
        return False
    if len(sources) not in (1, 2) or (policy == "v2" and len(sources) != 2):
        return False
    try:
        for descriptor in [*sources, output]:
            descriptor_nbytes(descriptor)
            if not isinstance(descriptor.get("metadata"), dict) or (
                descriptor.get("operation") is not None
                and not isinstance(descriptor["operation"], dict)
            ):
                return False
        shapes = [TensorShape(tuple(source["shape"]), source["dtype"]) for source in sources]
        result = TensorShape(tuple(output["shape"]), output["dtype"])
        source = shapes[0]
        params = parameters(step)
        axes = params["axes"]
        if len(axes) != len(source.shape):
            return False
        expected = tuple(
            dimension if mapping is None else len(mapping["coordinates"])
            for dimension, mapping in zip(source.shape, axes, strict=True)
        )
        if (
            result.shape != expected
            or (len(shapes) == 2 and result.shape != shapes[1].shape)
            or result.dtype
            != ("bool" if step.operation == Operation.coordinate_coverage else source.dtype)
            or sum(expected) > MAX_SELECTION_INDICES
            or (step.operation == Operation.reindex_validity and result.dtype != "bool")
            or any(
                mapping is not None
                and any(index is not None and index >= dimension for index in mapping["indices"])
                for dimension, mapping in zip(source.shape, axes, strict=True)
            )
        ):
            return False
        if step.operation == Operation.reindex_coordinates:
            reindex_fill(source.dtype, params["fill_value"])
        semantics = [
            read_semantics(item["metadata"], shape.shape, item["operation"])
            for item, shape in zip(sources, shapes, strict=True)
        ]
        derived = derive_semantics(step, shapes, semantics, result, policy)
        actual = read_semantics(output["metadata"], result.shape, output["operation"])
        return (
            derived is not None
            and actual is not None
            and canonical_json(derived.model_dump(mode="json"))
            == canonical_json(actual.model_dump(mode="json"))
        )
    except TensorusError:
        return False
def _v2_reindex_step_matches(step: Step, sources: list[Record], output: Record) -> bool:
    return (
        step.operation in (Operation.reindex_coordinates, Operation.reindex_validity)
        and len(sources) == 2
        and _coordinate_step_matches(step, sources, output, "v2")
    )
def _v2_reduction_step_matches(step: Step, sources: list[Record], output: Record) -> bool:
    validity_step = step.operation in (
        Operation.reduce_validity_any,
        Operation.chunked_reduce_validity_any,
    )
    if (not validity_step and step.operation not in PAIRED_REDUCTION_VALUES) or len(sources) != (
        1 if validity_step else 2
    ):
        return False
    try:
        for descriptor in [*sources, output]:
            descriptor_nbytes(
                {
                    "shape": descriptor["shape"],
                    "dtype": descriptor["dtype"],
                    "nbytes": descriptor["nbytes"],
                }
            )
        shapes = [TensorShape(tuple(source["shape"]), source["dtype"]) for source in sources]
        first = shapes[0]
        if validity_step:
            if first.dtype != "bool":
                return False
        elif shapes[1].dtype != "bool" or shapes[1].shape != first.shape:
            return False
        params = parameters(step)
        axes = reduction_axes(params["axis"], len(first.shape))
        if params["keepdims"] and axes:
            return False
        result = TensorShape(tuple(output["shape"]), output["dtype"])
        if result.shape != reduced_shape(first.shape, axes, params["keepdims"]) or (
            result.dtype != paired_reduction_dtype(step.operation, first.dtype)
        ):
            return False
        source_semantics = [
            read_semantics(source["metadata"], shape.shape, source["operation"])
            for source, shape in zip(sources, shapes, strict=True)
        ]
        actual = read_semantics(output["metadata"], result.shape, output["operation"])
        derived = derive_semantics(step, shapes, source_semantics, result, "v2")
        return (
            actual is not None
            and derived is not None
            and canonical_json(actual.model_dump(mode="json"))
            == canonical_json(derived.model_dump(mode="json"))
        )
    except TensorusError:
        return False
def _v2_positional_step_matches(step: Step, sources: list[Record], output: Record) -> bool:
    if step.operation not in POSITIONAL_TRANSFORMS or len(sources) != 1:
        return False
    try:
        source = sources[0]
        for descriptor in (source, output):
            descriptor_nbytes(descriptor)
        first = TensorShape(tuple(source["shape"]), source["dtype"])
        result = TensorShape(tuple(output["shape"]), output["dtype"])
        params = parameters(step)
        if step.operation == Operation.transpose:
            order = params["axes"]
            if order is None:
                order = list(reversed(range(len(first.shape))))
            if sorted(order) != list(range(len(first.shape))):
                return False
            expected = tuple(first.shape[axis] for axis in order)
        elif step.operation in (Operation.slice, Operation.chunked_slice):
            selection = params["selection"]
            if len(selection) > len(first.shape):
                return False
            expected = tuple(
                len(range(*slice(*selection[index]).indices(dimension)))
                if index < len(selection)
                else dimension
                for index, dimension in enumerate(first.shape)
            )
        else:
            indices = params["indices"]
            if len(indices) != len(first.shape) or any(
                selected is not None and selected and selected[-1] >= dimension
                for dimension, selected in zip(first.shape, indices, strict=True)
            ):
                return False
            expected = tuple(
                dimension if selected is None else len(selected)
                for dimension, selected in zip(first.shape, indices, strict=True)
            )
        if result.shape != expected or result.dtype != first.dtype:
            return False
        source_semantics = read_semantics(source["metadata"], first.shape, source["operation"])
        actual = read_semantics(output["metadata"], result.shape, output["operation"])
        derived = derive_semantics(step, [first], [source_semantics], result, "v2")
        return (
            actual is not None
            and derived is not None
            and canonical_json(actual.model_dump(mode="json"))
            == canonical_json(derived.model_dump(mode="json"))
        )
    except TensorusError:
        return False
def _v2_conversion_step_matches(step: Step, sources: list[Record], output: Record) -> bool:
    if step.operation not in CONVERSION_VALUES | {Operation.cast}:
        return False
    expected_inputs = 2 if step.operation in MASKED_VALUE_CONVERSIONS else 1
    if len(sources) != expected_inputs:
        return False
    try:
        for descriptor in [*sources, output]:
            descriptor_nbytes(descriptor)
        shapes = [TensorShape(tuple(source["shape"]), source["dtype"]) for source in sources]
        first = shapes[0]
        result = TensorShape(tuple(output["shape"]), output["dtype"])
        if result.shape != first.shape:
            return False
        if step.operation in MASKED_VALUE_CONVERSIONS:
            if (
                first.dtype == "bool"
                or shapes[1].dtype != "bool"
                or shapes[1].shape != first.shape
                or result.dtype != "float64"
                or (
                    step.operation == Operation.masked_convert_temperature_unit
                    and first.size > MAX_TEMPERATURE_ELEMENTS
                )
            ):
                return False
        elif (
            result.dtype != first.dtype
            or output["checksum"] != sources[0]["checksum"]
            or (step.operation == Operation.cast and first.dtype != "bool")
        ):
            return False
        semantics = [
            read_semantics(source["metadata"], shape.shape, source["operation"])
            for source, shape in zip(sources, shapes, strict=True)
        ]
        actual = read_semantics(output["metadata"], result.shape, output["operation"])
        derived = derive_semantics(step, shapes, semantics, result, "v2")
        return (
            actual is not None
            and derived is not None
            and canonical_json(actual.model_dump(mode="json"))
            == canonical_json(derived.model_dump(mode="json"))
        )
    except TensorusError:
        return False
def _v2_selected_float_step_matches(
    step: Step,
    sources: list[Record],
    output: Record,
    operations: Collection[Operation],
    *,
    allow_boolean: bool,
) -> bool:
    if step.operation == Operation.cast:
        return _v2_conversion_step_matches(step, sources, output)
    if step.operation not in operations or len(sources) != 2:
        return False
    try:
        for descriptor in [*sources, output]:
            descriptor_nbytes(descriptor)
        shapes = [TensorShape(tuple(source["shape"]), source["dtype"]) for source in sources]
        result = TensorShape(tuple(output["shape"]), output["dtype"])
        if (
            (not allow_boolean and shapes[0].dtype == "bool")
            or shapes[1].dtype != "bool"
            or shapes[1].shape != shapes[0].shape
            or result.shape != shapes[0].shape
            or result.dtype != "float64"
        ):
            return False
        declared = [
            read_semantics(source["metadata"], shape.shape, source["operation"])
            for source, shape in zip(sources, shapes, strict=True)
        ]
        actual = read_semantics(output["metadata"], result.shape, output["operation"])
        derived = derive_semantics(step, shapes, declared, result, "v2")
        return (
            actual is not None
            and derived is not None
            and canonical_json(actual.model_dump(mode="json"))
            == canonical_json(derived.model_dump(mode="json"))
        )
    except TensorusError:
        return False
def _v2_scaling_step_matches(step: Step, sources: list[Record], output: Record) -> bool:
    return _v2_selected_float_step_matches(
        step, sources, output, PAIRED_SCALING_VALUES, allow_boolean=True
    )
def _v2_cast_step_matches(step: Step, sources: list[Record], output: Record) -> bool:
    if step.operation == Operation.cast:
        return _v2_conversion_step_matches(step, sources, output)
    if step.operation not in PAIRED_CAST_VALUES or len(sources) != 2:
        return False
    try:
        for descriptor in [*sources, output]:
            descriptor_nbytes(descriptor)
        shapes = [TensorShape(tuple(source["shape"]), source["dtype"]) for source in sources]
        result = TensorShape(tuple(output["shape"]), output["dtype"])
        if (
            shapes[0].dtype == "bool"
            or shapes[1].dtype != "bool"
            or shapes[1].shape != shapes[0].shape
            or result.shape != shapes[0].shape
            or result.dtype != parameters(step)["dtype"]
        ):
            return False
        declared = [
            read_semantics(source["metadata"], shape.shape, source["operation"])
            for source, shape in zip(sources, shapes, strict=True)
        ]
        actual = read_semantics(output["metadata"], result.shape, output["operation"])
        derived = derive_semantics(step, shapes, declared, result, "v2")
        return (
            actual is not None
            and derived is not None
            and canonical_json(actual.model_dump(mode="json"))
            == canonical_json(derived.model_dump(mode="json"))
        )
    except TensorusError:
        return False
def _v2_scalar_step_matches(step: Step, sources: list[Record], output: Record) -> bool:
    return _v2_selected_float_step_matches(
        step, sources, output, {PAIRED_SCALAR_OPERATION}, allow_boolean=False
    )
def _v2_fitted_step_matches(step: Step, sources: list[Record], output: Record) -> bool:
    if step.operation == Operation.cast:
        return _v2_conversion_step_matches(step, sources, output)
    if step.operation not in (
        Operation.fit_selected_preprocessing,
        Operation.apply_fitted_preprocessing,
    ):
        return False
    try:
        for descriptor in [*sources, output]:
            descriptor_nbytes(descriptor)
        shapes = [TensorShape(tuple(source["shape"]), source["dtype"]) for source in sources]
        result = TensorShape(tuple(output["shape"]), output["dtype"])
        params = parameters(step)
        fitting = step.operation == Operation.fit_selected_preprocessing
        if len(shapes) != (2 if fitting else 3) or result.dtype != "float64":
            return False
        mask = shapes[1] if fitting else shapes[2]
        expected_state = fitted_state_shape(
            shapes[0].shape, params["axis"], params["algorithm_version"]
        )
        if mask.dtype != "bool" or mask.shape != shapes[0].shape:
            return False
        if fitting:
            if not shapes[0].size or result.shape != expected_state:
                return False
        else:
            state = sources[1]
            state_operation = state["operation"]
            if (
                shapes[1].dtype != "float64"
                or shapes[1].shape != expected_state
                or result.shape != shapes[0].shape
                or state["checksum"] != params["state_checksum"]
            ):
                return False
            if params.get("imported_state") is not None:
                if state_operation is not None:
                    return False
            else:
                if (
                    not isinstance(state_operation, dict)
                    or state_operation.get("name") != Operation.fit_selected_preprocessing
                    or not isinstance(state_operation.get("parameters"), dict)
                ):
                    return False
                fit = FittedScalingParameters.model_validate(state_operation["parameters"])
                if (
                    fit.algorithm != params["algorithm"]
                    or fit.algorithm_version != params["algorithm_version"]
                    or reduction_axes(fit.axis, len(shapes[0].shape))
                    != reduction_axes(params["axis"], len(shapes[0].shape))
                ):
                    return False
        declared = [
            read_semantics(source["metadata"], shape.shape, source["operation"])
            for source, shape in zip(sources, shapes, strict=True)
        ]
        actual = read_semantics(output["metadata"], result.shape, output["operation"])
        derived = derive_semantics(step, shapes, declared, result, "v2")
        return (
            actual is not None
            and derived is not None
            and canonical_json(actual.model_dump(mode="json"))
            == canonical_json(derived.model_dump(mode="json"))
        )
    except (TensorusError, ValidationError, ValueError, KeyError, TypeError, IndexError):
        return False
def _v2_paired_step_matches(
    definition: PlanDefinition,
    step: Step,
    sources: list[Record],
    output: Record,
    selected_preprocessing: tuple[PreprocessingIdentity, PreprocessingIdentity] | None,
) -> bool:
    if _pinned_v2_reduction(definition):
        return _v2_reduction_step_matches(step, sources, output)
    if _v2_positional_definition(definition):
        return _v2_positional_step_matches(step, sources, output)
    if _v2_conversion_definition(definition):
        return _v2_conversion_step_matches(step, sources, output)
    if _v2_scaling_definition(definition):
        return _v2_scaling_step_matches(step, sources, output)
    if _v2_cast_definition(definition):
        return _v2_cast_step_matches(step, sources, output)
    if _v2_scalar_definition(definition):
        return _v2_scalar_step_matches(step, sources, output)
    if _v2_fitted_fit_definition(definition) or _v2_fitted_apply_definition(definition):
        return _v2_fitted_step_matches(step, sources, output)
    if step.operation in (Operation.reindex_coordinates, Operation.reindex_validity):
        return _v2_reindex_step_matches(step, sources, output)
    try:
        for descriptor in [*sources, output]:
            descriptor_nbytes(descriptor)
        shapes = [TensorShape(tuple(source["shape"]), source["dtype"]) for source in sources]
        result = TensorShape(tuple(output["shape"]), output["dtype"])
        if step.operation == Operation.transpose:
            if len(shapes) != 1:
                return False
            axes = parameters(step)["axes"]
            if (
                axes is None
                or sorted(axes) != list(range(len(shapes[0].shape)))
                or result.shape != tuple(shapes[0].shape[axis] for axis in axes)
                or result.dtype != shapes[0].dtype
            ):
                return False
        elif step.operation == Operation.select_indices:
            if len(shapes) != 1:
                return False
            indices = parameters(step)["indices"]
            if len(indices) != len(shapes[0].shape):
                return False
            expected_shape = tuple(
                dimension if selected is None else len(selected)
                for dimension, selected in zip(shapes[0].shape, indices, strict=True)
            )
            if (
                result.shape != expected_shape
                or result.dtype != shapes[0].dtype
                or any(
                    selected is not None and selected and selected[-1] >= dimension
                    for dimension, selected in zip(shapes[0].shape, indices, strict=True)
                )
            ):
                return False
        elif step.operation == Operation.logical_and:
            if (
                len(shapes) != 2
                or result.dtype != "bool"
                or any(shape.dtype != "bool" or shape.shape != result.shape for shape in shapes)
            ):
                return False
        elif step.operation in MASKED_BINARY_OPERATIONS:
            if (
                len(shapes) != 3
                or result.dtype != "float64"
                or shapes[2].dtype != "bool"
                or any(shape.shape != result.shape for shape in shapes)
            ):
                return False
        else:
            return False
        declared = [
            read_semantics(source["metadata"], shape.shape, source["operation"])
            for source, shape in zip(sources, shapes, strict=True)
        ]
        actual = read_semantics(output["metadata"], result.shape, output["operation"])
        derived = derive_semantics(
            step,
            shapes,
            declared,
            result,
            "v2",
            selected_preprocessing=selected_preprocessing if step.id == "values" else None,
        )
        return (
            derived is not None
            and actual is not None
            and canonical_json(derived.model_dump(mode="json"))
            == canonical_json(actual.model_dump(mode="json"))
        )
    except (TensorusError, ValueError, KeyError, TypeError, IndexError):
        return False
