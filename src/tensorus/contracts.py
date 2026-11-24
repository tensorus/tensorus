"""Versioned release contracts and bounded quality evidence."""

import math
import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from itertools import pairwise
from typing import Annotated, Any, Literal, Self
from uuid import UUID

import numpy as np
from pydantic import (
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    StrictBool,
    StrictInt,
    field_validator,
    model_serializer,
    model_validator,
)

from tensorus.arrays import canonical_json
from tensorus.errors import TensorusError
from tensorus.models import AxisDescriptor, DType, Model, Name, TensorSpec
from tensorus.semantics import ModelIdentity, PreprocessingIdentity, SemanticAxis, read_semantics

DecimalBound = Annotated[
    str,
    Field(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]{1,4})?$",
    ),
]
MAX_NONZERO_ELEMENTS = 100_000
MAX_MEAN_ELEMENTS = 100_000
MAX_RELEASE_MASK_ELEMENTS = 1_000_000
MAX_RELEASE_MASK_BINDINGS = 32
MAX_RELEASE_QUALITY_OVERRIDES = 32
MAX_RELEASE_CONTRACT_OVERRIDES = 32
ReleaseHash = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]


class SemanticRequirements(Model):
    axes: list[SemanticAxis] | None = Field(default=None, max_length=16)
    value_unit: Annotated[str, Field(strict=True, min_length=1, max_length=128)] | None = None
    model: ModelIdentity | None = None
    preprocessing: PreprocessingIdentity | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    require_coordinates: StrictBool = False

    @field_validator("value_unit")
    @classmethod
    def validate_unit(cls, value: str | None) -> str | None:
        return AxisDescriptor.validate_unit(value)

    @field_validator("axes")
    @classmethod
    def validate_axes(cls, axes: list[SemanticAxis] | None) -> list[SemanticAxis] | None:
        if axes is not None:
            if len({axis.label for axis in axes}) != len(axes):
                raise ValueError("Contract axis labels must be unique")
            for axis in axes:
                coordinates = axis.coordinates
                if coordinates is not None and (
                    any(
                        isinstance(value, float) and not math.isfinite(value)
                        for value in coordinates
                    )
                    or any(left >= right for left, right in pairwise(coordinates))
                ):
                    raise ValueError("Contract coordinates must be finite and strictly increasing")
        return axes


class ContractDefinition(Model):
    spec: TensorSpec = Field(default_factory=TensorSpec)
    semantics: SemanticRequirements | None = None

    @model_validator(mode="after")
    def validate_shape_and_axes(self) -> Self:
        if self.semantics and self.semantics.axes is not None and self.spec.shape is not None:
            if len(self.semantics.axes) != len(self.spec.shape):
                raise ValueError("Contract axes must match the specified rank")
            for axis, dimension in zip(self.semantics.axes, self.spec.shape, strict=True):
                if (
                    dimension is not None
                    and axis.coordinates is not None
                    and len(axis.coordinates) != dimension
                ):
                    raise ValueError("Contract coordinates must match fixed dimensions")
        return self


class ContractWrite(Model):
    expected_revision: Annotated[StrictInt, Field(ge=1)]
    definition: ContractDefinition


def semantic_errors(
    requirements: SemanticRequirements | None, record: Mapping[str, Any]
) -> tuple[Literal["declared", "not_declared", "invalid"], list[str]]:
    try:
        actual = read_semantics(
            record["metadata"], tuple(record["shape"]), operation=record["operation"]
        )
    except TensorusError as exc:
        return "invalid", [str(exc)]
    if requirements is None:
        return ("declared" if actual is not None else "not_declared"), []
    if actual is None:
        return "not_declared", ["The contract requires a versioned tensor semantics declaration"]
    errors = []
    if requirements.value_unit is not None and actual.value_unit != requirements.value_unit:
        errors.append("Declared value unit does not match the contract")
    if requirements.model is not None and actual.model != requirements.model:
        errors.append("Declared model identity or version does not match the contract")
    if (
        requirements.preprocessing is not None
        and actual.preprocessing != requirements.preprocessing
    ):
        errors.append("Declared preprocessing identity or version does not match the contract")
    if requirements.require_coordinates and any(axis.coordinates is None for axis in actual.axes):
        errors.append("The contract requires coordinates for every axis")
    if requirements.axes is not None:
        if len(requirements.axes) != len(actual.axes):
            errors.append("Declared axis count does not match the contract")
        else:
            for expected, declared in zip(requirements.axes, actual.axes, strict=True):
                if expected.label != declared.label or expected.unit != declared.unit:
                    errors.append("Declared axis labels or units do not match the contract")
                    break
                if (
                    expected.time_reference is not None
                    and expected.time_reference != declared.time_reference
                ) or (
                    expected.coordinate_system is not None
                    and expected.coordinate_system != declared.coordinate_system
                ):
                    errors.append(f"Declared reference for axis {expected.label} does not match")
                    break
                if expected.coordinates is not None and (
                    canonical_json(expected.coordinates) != canonical_json(declared.coordinates)
                    if actual.version == 2
                    else expected.coordinates != declared.coordinates
                ):
                    errors.append(f"Declared coordinates for axis {expected.label} do not match")
    return "declared", errors


class MeanBounds(Model):
    """Select at least one bound; nonzero decimal adjusted exponents must be -400..400."""

    minimum: DecimalBound | None = Field(
        default=None, description="Inclusive decimal-string lower bound on the exact mean."
    )
    maximum: DecimalBound | None = Field(
        default=None, description="Inclusive decimal-string upper bound on the exact mean."
    )

    @field_validator("minimum", "maximum")
    @classmethod
    def validate_magnitude(cls, value: str | None) -> str | None:
        if value is not None:
            decimal = Decimal(value)
            if decimal and abs(decimal.adjusted()) > 400:
                raise ValueError("Mean bound magnitude exceeds the supported exponent range")
        return value

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.minimum is None and self.maximum is None:
            raise ValueError("Select at least one mean bound")
        if (
            self.minimum is not None
            and self.maximum is not None
            and Decimal(self.minimum) > Decimal(self.maximum)
        ):
            raise ValueError("Mean minimum cannot exceed maximum")
        return self


class NonzeroFractionBounds(Model):
    minimum: DecimalBound | None = Field(
        default=None, description="Inclusive decimal-string lower bound between zero and one."
    )
    maximum: DecimalBound | None = Field(
        default=None, description="Inclusive decimal-string upper bound between zero and one."
    )

    @field_validator("minimum", "maximum")
    @classmethod
    def validate_fraction(cls, value: str | None) -> str | None:
        if value is not None:
            decimal = Decimal(value)
            if not 0 <= decimal <= 1:
                raise ValueError("Nonzero fraction bounds must be between zero and one")
            if decimal and abs(decimal.adjusted()) > 400:
                raise ValueError("Nonzero fraction bound magnitude exceeds the supported range")
        return value

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.minimum is None and self.maximum is None:
            raise ValueError("Select at least one nonzero fraction bound")
        if (
            self.minimum is not None
            and self.maximum is not None
            and Decimal(self.minimum) > Decimal(self.maximum)
        ):
            raise ValueError("Nonzero fraction minimum cannot exceed maximum")
        return self


def exact_bound_results(
    bounds: MeanBounds | NonzeroFractionBounds, observed: Fraction | None
) -> tuple[bool | None, bool | None]:
    minimum = (
        observed is not None and observed >= Fraction(Decimal(bounds.minimum))
        if bounds.minimum is not None
        else None
    )
    maximum = (
        observed is not None and observed <= Fraction(Decimal(bounds.maximum))
        if bounds.maximum is not None
        else None
    )
    return minimum, maximum


def mean_bound_results(
    bounds: MeanBounds, observed: Fraction | None
) -> tuple[bool | None, bool | None]:
    return exact_bound_results(bounds, observed)


class _OptionalQualityModel(Model):
    # A return annotation would replace Pydantic's typed output schema with a generic dict.
    @model_serializer(mode="wrap")
    def serialize_optional_quality(self, handler: SerializerFunctionWrapHandler):  # type: ignore[no-untyped-def]
        result: dict[str, Any] = handler(self)
        for field in ("mean", "nonzero_fraction"):
            if result.get(field) is None:
                result.pop(field, None)
        return result


class QualityPolicy(_OptionalQualityModel):
    require_nonempty: StrictBool = True
    minimum: DecimalBound | None = None
    maximum: DecimalBound | None = None
    mean: MeanBounds | None = Field(
        default=None,
        description="Optional exact arithmetic mean of all stored tensor values.",
    )
    nonzero_fraction: NonzeroFractionBounds | None = Field(
        default=None,
        description="Optional fraction of stored numeric values that are not zero.",
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if (
            self.minimum is not None
            and self.maximum is not None
            and Decimal(self.minimum) > Decimal(self.maximum)
        ):
            raise ValueError("Minimum cannot exceed maximum")
        return self


class ReleaseQualityBinding(Model):
    data_version_id: UUID
    binding_id: UUID
    descriptor_hash: ReleaseHash
    minimum_included_fraction: DecimalBound

    @field_validator("minimum_included_fraction")
    @classmethod
    def validate_minimum(cls, value: str) -> str:
        decimal = Decimal(value)
        if not 0 <= decimal <= 1:
            raise ValueError("Minimum included fraction must be between zero and one")
        if decimal and abs(decimal.adjusted()) > 400:
            raise ValueError("Minimum included fraction magnitude exceeds the supported range")
        return value


class ReleaseQualityOverride(Model):
    version_id: UUID
    quality: QualityPolicy


class ReleaseMemberContract(Model):
    revision: Annotated[StrictInt, Field(ge=1, le=2**31 - 1)]
    definition_hash: ReleaseHash


class ReleaseContractOverride(ReleaseMemberContract):
    version_id: UUID


class ReleaseRequest(Model):
    name: Name
    snapshot_id: UUID
    contract_revision: Annotated[StrictInt, Field(ge=1)]
    quality: QualityPolicy = Field(default_factory=QualityPolicy)
    quality_bindings: list[ReleaseQualityBinding] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_RELEASE_MASK_BINDINGS,
        exclude_if=lambda value: value is None,
    )
    quality_overrides: list[ReleaseQualityOverride] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_RELEASE_QUALITY_OVERRIDES,
        exclude_if=lambda value: value is None,
    )
    contract_overrides: list[ReleaseContractOverride] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_RELEASE_CONTRACT_OVERRIDES,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def distinct_release_selections(self) -> Self:
        if self.quality_bindings is not None:
            if len({item.data_version_id for item in self.quality_bindings}) != len(
                self.quality_bindings
            ):
                raise ValueError("Each release member may select at most one validity binding")
            if len({item.binding_id for item in self.quality_bindings}) != len(
                self.quality_bindings
            ):
                raise ValueError("Release validity binding selections must be unique")
        if self.quality_overrides is not None and len(
            {item.version_id for item in self.quality_overrides}
        ) != len(self.quality_overrides):
            raise ValueError("Release quality overrides must name distinct versions")
        if self.contract_overrides is not None and len(
            {item.version_id for item in self.contract_overrides}
        ) != len(self.contract_overrides):
            raise ValueError("Release contract overrides must name distinct versions")
        return self


class ReleaseApproval(Model):
    report_hash: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]


class MeanEvidence(Model):
    numerator: (
        Annotated[str, Field(strict=True, max_length=1024, pattern=r"^(?:0|-?[1-9][0-9]*)$")] | None
    ) = Field(description="Exact mean numerator, or null for an empty tensor.")
    denominator: (
        Annotated[str, Field(strict=True, max_length=1024, pattern=r"^[1-9][0-9]*$")] | None
    ) = Field(description="Positive exact mean denominator, or null for an empty tensor.")
    minimum: StrictBool | None
    maximum: StrictBool | None

    @model_validator(mode="after")
    def validate_rational(self) -> Self:
        if (self.numerator is None) != (self.denominator is None):
            raise ValueError("Mean numerator and denominator must both be present or null")
        if self.minimum is None and self.maximum is None:
            raise ValueError("Mean evidence must contain at least one bound result")
        if self.numerator is None:
            if self.minimum is True or self.maximum is True:
                raise ValueError("An empty tensor cannot pass a mean bound")
        elif (
            self.denominator is not None
            and math.gcd(int(self.numerator), int(self.denominator)) != 1
        ):
            raise ValueError("Mean evidence must be a reduced rational")
        return self


class NonzeroFractionEvidence(Model):
    nonzero_count: Annotated[StrictInt, Field(ge=0, le=MAX_NONZERO_ELEMENTS)]
    minimum: StrictBool | None
    maximum: StrictBool | None

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.minimum is None and self.maximum is None:
            raise ValueError("Nonzero fraction evidence must contain a bound result")
        return self


class QualityEvidence(_OptionalQualityModel):
    count: int
    observed_minimum: str | None
    observed_maximum: str | None
    finite: Literal[True]
    nonempty: bool | None
    minimum: bool | None
    maximum: bool | None
    mean: MeanEvidence | None = None
    nonzero_fraction: NonzeroFractionEvidence | None = None

    @model_validator(mode="after")
    def validate_quality_counts(self) -> Self:
        if self.mean is not None and (
            self.count < 0 or (self.count == 0) != (self.mean.numerator is None)
        ):
            raise ValueError("Mean evidence must match the tensor element count")
        nonzero = self.nonzero_fraction
        if nonzero is not None:
            if not 0 <= self.count <= MAX_NONZERO_ELEMENTS or nonzero.nonzero_count > self.count:
                raise ValueError("Nonzero fraction evidence must match the tensor element count")
            if self.count == 0 and (nonzero.minimum is True or nonzero.maximum is True):
                raise ValueError("An empty tensor cannot pass a nonzero fraction bound")
        return self


def quality_passed(evidence: QualityEvidence) -> bool:
    checks = [evidence.nonempty, evidence.minimum, evidence.maximum]
    if evidence.mean is not None:
        checks.extend((evidence.mean.minimum, evidence.mean.maximum))
    if evidence.nonzero_fraction is not None:
        checks.extend((evidence.nonzero_fraction.minimum, evidence.nonzero_fraction.maximum))
    return all(check is not False for check in checks)


def _shape_matches_count(shape: list[int], count: int) -> bool:
    if len(shape) > 16 or count < 0 or any(dimension < 0 for dimension in shape):
        return False
    if 0 in shape:
        return count == 0
    size = 1
    for dimension in shape:
        if size > count // dimension:
            return False
        size *= dimension
    return size == count


def _quality_extrema_match(policy: QualityPolicy, evidence: QualityEvidence) -> bool:
    if (evidence.nonempty is not None) != policy.require_nonempty or (
        policy.require_nonempty and evidence.nonempty != (evidence.count > 0)
    ):
        return False
    if evidence.count == 0:
        minimum = maximum = None
        if evidence.observed_minimum is not None or evidence.observed_maximum is not None:
            return False
    else:
        if evidence.observed_minimum is None or evidence.observed_maximum is None:
            return False
        try:
            minimum = Decimal(evidence.observed_minimum)
            maximum = Decimal(evidence.observed_maximum)
        except InvalidOperation:
            return False
        if not minimum.is_finite() or not maximum.is_finite() or minimum > maximum:
            return False
    expected_minimum = (
        minimum is not None and minimum >= Decimal(policy.minimum)
        if policy.minimum is not None
        else None
    )
    expected_maximum = (
        maximum is not None and maximum <= Decimal(policy.maximum)
        if policy.maximum is not None
        else None
    )
    return evidence.minimum == expected_minimum and evidence.maximum == expected_maximum


class ReleaseQualityMask(Model):
    version_id: UUID
    checksum: ReleaseHash
    binding_id: UUID
    descriptor_hash: ReleaseHash
    origin_kind: Literal["asserted", "transformed", "binary_transformed"]
    included_count: Annotated[StrictInt, Field(ge=0, le=MAX_RELEASE_MASK_ELEMENTS)]
    excluded_count: Annotated[StrictInt, Field(ge=0, le=MAX_RELEASE_MASK_ELEMENTS)]
    minimum_included_fraction: DecimalBound
    minimum_passed: StrictBool

    @field_validator("minimum_included_fraction")
    @classmethod
    def validate_minimum(cls, value: str) -> str:
        return ReleaseQualityBinding.validate_minimum(value)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        total = self.included_count + self.excluded_count
        if total > MAX_RELEASE_MASK_ELEMENTS:
            raise ValueError("Release mask counts exceed the element limit")
        expected = (
            total > 0
            and self.included_count > 0
            and Fraction(self.included_count, total)
            >= Fraction(Decimal(self.minimum_included_fraction))
        )
        if self.minimum_passed != expected:
            raise ValueError("Release mask minimum result does not match its counts")
        return self


class ReleaseMember(Model):
    name: Name
    version_id: UUID
    dtype: DType
    shape: list[int]
    nbytes: int
    checksum: str
    metadata_hash: str
    operation: dict[str, JsonValue] | None
    parents: list[str]
    semantics: Literal["declared", "not_declared", "invalid"]
    contract_errors: list[str]
    quality: QualityEvidence
    passed: bool
    quality_mask: ReleaseQualityMask | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    quality_policy: QualityPolicy | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    contract_override: ReleaseMemberContract | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ReleaseReport(Model):
    format: Literal[
        "tensorus-release-report-v1",
        "tensorus-release-report-v2",
        "tensorus-release-report-v3",
        "tensorus-release-report-v4",
    ] = "tensorus-release-report-v1"
    tenant_id: UUID
    dataset_id: UUID
    dataset: Name
    release_name: Name
    proposed_by: str
    snapshot_id: UUID
    contract_revision: int
    contract_hash: str
    write_spec_hash: str
    quality: QualityPolicy
    members: list[ReleaseMember]
    passed: bool
    checked_bytes: Annotated[StrictInt, Field(ge=0)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_quality_evidence(self) -> Self:
        masked = [member for member in self.members if member.quality_mask is not None]
        overrides = [member for member in self.members if member.quality_policy is not None]
        contracts = [member for member in self.members if member.contract_override is not None]
        if self.format == "tensorus-release-report-v1":
            if masked or overrides or contracts or self.checked_bytes is not None:
                raise ValueError("Release report v1 cannot contain selected policies or masks")
        else:
            if self.checked_bytes is None or len(masked) > MAX_RELEASE_MASK_BINDINGS:
                raise ValueError("Release report requires bounded selections and checked bytes")
            if self.format == "tensorus-release-report-v2" and (
                not masked or overrides or contracts
            ):
                raise ValueError("Release report v2 requires masks and no quality overrides")
            if self.format == "tensorus-release-report-v3" and (
                contracts or not 1 <= len(overrides) <= MAX_RELEASE_QUALITY_OVERRIDES
            ):
                raise ValueError("Release report v3 requires bounded quality overrides")
            if self.format == "tensorus-release-report-v4" and (
                not 1 <= len(contracts) <= MAX_RELEASE_CONTRACT_OVERRIDES
                or len(overrides) > MAX_RELEASE_QUALITY_OVERRIDES
            ):
                raise ValueError("Release report v4 requires bounded contract overrides")
            members = {member.version_id: member for member in self.members}
            if (
                len(members) != len(self.members)
                or len({member.name for member in self.members}) != len(self.members)
                or len({member.quality_mask.binding_id for member in masked if member.quality_mask})
                != len(masked)
            ):
                raise ValueError("Release report members must have unique versions")
            if any(
                len(member.shape) > 16
                or any(not 0 <= dimension <= 2**63 - 1 for dimension in member.shape)
                or not 0 <= member.nbytes <= 2**63 - 1
                or member.nbytes % np.dtype(member.dtype).itemsize
                or not _shape_matches_count(
                    member.shape, member.nbytes // np.dtype(member.dtype).itemsize
                )
                or re.fullmatch(r"[0-9a-f]{64}", member.checksum) is None
                or re.fullmatch(r"[0-9a-f]{64}", member.metadata_hash) is None
                for member in self.members
            ):
                raise ValueError("Release report v2 contains an invalid member descriptor")
            total_mask_elements = 0
            mask_bytes = 0
            for member in masked:
                mask = member.quality_mask
                assert mask is not None
                selected = members.get(mask.version_id)
                total = mask.included_count + mask.excluded_count
                if (
                    selected is None
                    or selected.dtype != DType.bool
                    or selected.shape != member.shape
                    or selected.checksum != mask.checksum
                    or not _shape_matches_count(member.shape, total)
                    or member.quality.count != mask.included_count
                    or (member.passed and not mask.minimum_passed)
                ):
                    raise ValueError("Release mask evidence does not match snapshot membership")
                total_mask_elements += total
                mask_bytes += selected.nbytes
            if total_mask_elements > MAX_RELEASE_MASK_ELEMENTS:
                raise ValueError("Release masks exceed the total element limit")
            if self.checked_bytes != sum(member.nbytes for member in self.members) + mask_bytes:
                raise ValueError("Release checked-byte count does not match its reads")
            if self.passed != all(member.passed for member in self.members) or any(
                not _quality_extrema_match(member.quality_policy or self.quality, member.quality)
                or (
                    member.quality_mask is None
                    and not _shape_matches_count(member.shape, member.quality.count)
                )
                or member.passed
                != (
                    not member.contract_errors
                    and quality_passed(member.quality)
                    and (member.quality_mask is None or member.quality_mask.minimum_passed)
                )
                for member in self.members
            ):
                raise ValueError("Release member outcomes do not match selected checks")
            for field, work_limit in (
                ("mean", MAX_MEAN_ELEMENTS),
                ("nonzero_fraction", MAX_NONZERO_ELEMENTS),
            ):
                total_elements = sum(
                    member.nbytes // np.dtype(member.dtype).itemsize
                    for member in self.members
                    if getattr(member.quality_policy or self.quality, field) is not None
                )
                if total_elements > work_limit:
                    raise ValueError("Release quality checks exceed the total element limit")
        if (
            sum(
                member.quality.count
                for member in self.members
                if (member.quality_policy or self.quality).nonzero_fraction is not None
            )
            > MAX_NONZERO_ELEMENTS
        ):
            raise ValueError("Nonzero fraction evidence exceeds the total element limit")
        for member in self.members:
            policy = member.quality_policy or self.quality
            bounds = policy.mean
            nonzero_bounds = policy.nonzero_fraction
            mean = member.quality.mean
            nonzero = member.quality.nonzero_fraction
            if (bounds is None) != (mean is None):
                raise ValueError("Mean evidence must match the release policy")
            if (nonzero_bounds is None) != (nonzero is None):
                raise ValueError("Nonzero fraction evidence must match the release policy")
            if mean is not None or nonzero is not None:
                label = "Mean" if mean is not None else "Nonzero fraction"
                if len(member.shape) > 16 or any(dimension < 0 for dimension in member.shape):
                    raise ValueError(f"{label} evidence has an invalid tensor shape")
                count = (
                    member.quality_mask.included_count + member.quality_mask.excluded_count
                    if member.quality_mask is not None
                    else member.quality.count
                )
                expected_count = 0 if 0 in member.shape else 1
                for dimension in member.shape:
                    if expected_count == 0:
                        break
                    # Check before multiplication because imported shapes are untrusted.
                    if expected_count > count // dimension:
                        raise ValueError(f"{label} evidence count does not match the tensor shape")
                    expected_count *= dimension
                if expected_count != count:
                    raise ValueError(f"{label} evidence count does not match the tensor shape")
            if bounds is not None and mean is not None:
                observed = (
                    Fraction(int(mean.numerator), int(mean.denominator))
                    if mean.numerator is not None and mean.denominator is not None
                    else None
                )
                if (mean.minimum, mean.maximum) != exact_bound_results(bounds, observed):
                    raise ValueError("Mean bound results do not match the reported rational")
                if member.passed and (mean.minimum is False or mean.maximum is False):
                    raise ValueError("A passing member cannot fail a mean bound")
            if nonzero_bounds is not None and nonzero is not None:
                count = member.quality.count
                observed = Fraction(nonzero.nonzero_count, count) if count else None
                if (nonzero.minimum, nonzero.maximum) != exact_bound_results(
                    nonzero_bounds, observed
                ):
                    raise ValueError("Nonzero fraction bound results do not match the counts")
                if member.passed and (nonzero.minimum is False or nonzero.maximum is False):
                    raise ValueError("A passing member cannot fail a nonzero fraction bound")
        return self


def release_binding_selections_match(request: ReleaseRequest, report: ReleaseReport) -> bool:
    expected = {
        item.data_version_id: (
            item.binding_id,
            item.descriptor_hash,
            item.minimum_included_fraction,
        )
        for item in request.quality_bindings or []
    }
    recorded = {
        member.version_id: (
            member.quality_mask.binding_id,
            member.quality_mask.descriptor_hash,
            member.quality_mask.minimum_included_fraction,
        )
        for member in report.members
        if member.quality_mask is not None
    }
    return expected == recorded


def release_quality_overrides_match(request: ReleaseRequest, report: ReleaseReport) -> bool:
    expected = {item.version_id: item.quality for item in request.quality_overrides or []}
    recorded = {
        member.version_id: member.quality_policy
        for member in report.members
        if member.quality_policy is not None
    }
    return expected == recorded


def release_contract_overrides_match(request: ReleaseRequest, report: ReleaseReport) -> bool:
    expected = {
        item.version_id: (item.revision, item.definition_hash)
        for item in request.contract_overrides or []
    }
    recorded = {
        member.version_id: (
            member.contract_override.revision,
            member.contract_override.definition_hash,
        )
        for member in report.members
        if member.contract_override is not None
    }
    return expected == recorded
