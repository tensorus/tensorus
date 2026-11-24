"""Read-only evaluation of explicitly selected tensor and plan-output criteria."""
import math
from decimal import Decimal
from fractions import Fraction
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import SerializerFunctionWrapHandler
from pydantic import StrictBool
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_serializer
from pydantic import model_validator
from tensorus.contracts import DecimalBound
from tensorus.contracts import QualityEvidence
from tensorus.contracts import QualityPolicy
from tensorus.models import DType
from tensorus.models import Model
from tensorus.models import StepName
from tensorus.semantics import TensorSemantics
from tensorus.validity import Hash
MAX_MEAN_ABSOLUTE_ERROR_ELEMENTS = 100_000
MAX_QUALITY_MASK_ELEMENTS = 1_000_000
TOLERANCE_EXPONENT_LIMIT = 400
def _validate_reference_bound(value: str, name: str) -> str:
    decimal = Decimal(value)
    if decimal < 0:
        raise ValueError(f"{name} must be nonnegative")
    if decimal and abs(decimal.adjusted()) > TOLERANCE_EXPONENT_LIMIT:
        raise ValueError(f"{name} magnitude exceeds the supported exponent range")
    return value
class ReferenceTolerance(Model):
    absolute: DecimalBound = Field(
        description="Nonnegative decimal string; nonzero adjusted exponent is limited to -400..400."
    )
    relative: DecimalBound = Field(
        description=(
            "Nonnegative decimal string applied to the absolute reference value; "
            "nonzero adjusted exponent is limited to -400..400."
        )
    )

    @field_validator("absolute", "relative")
    @classmethod
    def validate_tolerance(cls, value: str) -> str:
        return _validate_reference_bound(value, "Reference tolerance")
class ReferenceMeanAbsoluteError(Model):
    maximum: DecimalBound = Field(
        description=(
            "Inclusive nonnegative decimal-string maximum for the exact mean absolute error; "
            "nonzero adjusted exponent is limited to -400..400."
        )
    )

    @field_validator("maximum")
    @classmethod
    def validate_maximum(cls, value: str) -> str:
        return _validate_reference_bound(value, "Mean absolute error maximum")
class _OptionalReferenceComparisonsModel(Model):
    @model_serializer(mode="wrap")
    def serialize_comparisons(self, handler: SerializerFunctionWrapHandler):  # type: ignore[no-untyped-def]
        result: dict[str, Any] = handler(self)
        for field in ("tolerance", "mean_absolute_error"):
            if result.get(field) is None:
                result.pop(field, None)
        return result
class _OptionalReferenceModel(Model):
    # A return annotation makes Pydantic replace the typed output schema with a generic dict.
    @model_serializer(mode="wrap")
    def serialize_reference(self, handler: SerializerFunctionWrapHandler):  # type: ignore[no-untyped-def]
        result: dict[str, Any] = handler(self)
        if result.get("reference") is None:
            result.pop("reference", None)
        return result
class QualityMaskCriterion(Model):
    version_id: UUID
    minimum_included_fraction: DecimalBound | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Optional inclusive minimum fraction of positions selected by the mask.",
    )

    @field_validator("minimum_included_fraction")
    @classmethod
    def validate_minimum(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_reference_bound(value, "Minimum included fraction")
            if Decimal(value) > 1:
                raise ValueError("Minimum included fraction must be between zero and one")
        return value
class ReferenceMaskCriterion(QualityMaskCriterion):
    pass
class ValidityBindingPin(Model):
    binding_id: UUID
    descriptor_hash: Hash
class ValidityMaskCriterion(ValidityBindingPin):
    minimum_included_fraction: DecimalBound | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Optional inclusive minimum fraction of positions selected by the binding.",
    )

    @field_validator("minimum_included_fraction")
    @classmethod
    def validate_minimum(cls, value: str | None) -> str | None:
        return QualityMaskCriterion.validate_minimum(value)
class ReferenceValidityIntersection(Model):
    output_binding: ValidityBindingPin
    reference_binding: ValidityBindingPin
    minimum_included_fraction: DecimalBound | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Inclusive minimum fraction of positions selected by both bindings.",
    )

    @field_validator("minimum_included_fraction")
    @classmethod
    def validate_minimum(cls, value: str | None) -> str | None:
        return QualityMaskCriterion.validate_minimum(value)
class ReferenceCriterion(_OptionalReferenceComparisonsModel):
    version_id: UUID
    allow_legacy: StrictBool = False
    tolerance: ReferenceTolerance | None = None
    mean_absolute_error: ReferenceMeanAbsoluteError | None = None
    mask: ReferenceMaskCriterion | ValidityMaskCriterion | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    validity_intersection: ReferenceValidityIntersection | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def exclusive_selection(self) -> Self:
        if self.mask is not None and self.validity_intersection is not None:
            raise ValueError("Choose either a reference mask or a validity intersection")
        return self
def _has_quality_checks(quality: QualityPolicy | None) -> bool:
    return quality is not None and (
        quality.require_nonempty
        or quality.minimum is not None
        or quality.maximum is not None
        or quality.mean is not None
        or quality.nonzero_fraction is not None
    )
class EvaluationCriterion(_OptionalReferenceModel):
    contract_revision: Annotated[StrictInt, Field(ge=1, le=2**31 - 1)] | None = None
    quality: QualityPolicy | None = None
    quality_mask: QualityMaskCriterion | ValidityMaskCriterion | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    reference: ReferenceCriterion | None = None

    @model_validator(mode="after")
    def require_criterion(self) -> Self:
        if self.quality_mask is not None and not _has_quality_checks(self.quality):
            raise ValueError("A quality mask requires at least one quality criterion")
        if (
            self.contract_revision is None
            and self.reference is None
            and not _has_quality_checks(self.quality)
        ):
            raise ValueError("Select a contract revision, reference, or quality criterion")
        return self
class _StepIdentity(Model):
    step_id: StepName
class StepCriterion(EvaluationCriterion, _StepIdentity):
    pass
class TensorEvaluationRequest(EvaluationCriterion):
    expected_checksum: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
class ValidityStatisticsRequest(Model):
    expected_checksum: Hash = Field(description="Expected checksum of the exact data version.")
    binding_id: UUID = Field(description="Explicit validity assertion for that data version.")
    descriptor_hash: Hash = Field(description="Expected descriptor hash of the selected binding.")
class EvaluationRequest(Model):
    expected_plan_hash: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
    criteria: list[StepCriterion] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def unique_steps(self) -> Self:
        if len({item.step_id for item in self.criteria}) != len(self.criteria):
            raise ValueError("Each step may be evaluated only once")
        return self
class ReferenceToleranceEvidence(Model):
    absolute: DecimalBound
    relative: DecimalBound
    compared_elements: int | None
    violations: int | None
    passed: bool | None
class ReferenceMeanAbsoluteErrorEvidence(ReferenceMeanAbsoluteError):
    compared_elements: Annotated[StrictInt, Field(ge=0, le=MAX_MEAN_ABSOLUTE_ERROR_ELEMENTS)] | None
    numerator: (
        Annotated[str, Field(strict=True, max_length=1024, pattern=r"^(?:0|[1-9][0-9]*)$")] | None
    )
    denominator: (
        Annotated[str, Field(strict=True, max_length=1024, pattern=r"^[1-9][0-9]*$")] | None
    )
    passed: StrictBool | None

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.compared_elements is None:
            if (
                self.numerator is not None
                or self.denominator is not None
                or self.passed is not None
            ):
                raise ValueError("Uncompared mean absolute error cannot contain a result")
        elif self.compared_elements == 0:
            if (
                self.numerator is not None
                or self.denominator is not None
                or self.passed is not False
            ):
                raise ValueError("Empty tensors cannot pass a mean absolute error bound")
        else:
            if self.numerator is None or self.denominator is None or self.passed is None:
                raise ValueError("Mean absolute error evidence requires a complete result")
            numerator, denominator = int(self.numerator), int(self.denominator)
            if math.gcd(numerator, denominator) != 1:
                raise ValueError("Mean absolute error must be a reduced rational")
            if self.passed != (Fraction(numerator, denominator) <= Fraction(Decimal(self.maximum))):
                raise ValueError("Mean absolute error result does not match the selected maximum")
        return self
class SelectedValidityBinding(Model):
    id: UUID
    descriptor_hash: Hash
    data_version_id: UUID
    data_checksum: Hash
    origin_kind: Literal["asserted", "transformed", "binary_transformed"]
class QualityMaskEvidence(QualityMaskCriterion):
    checksum: str
    dtype: Literal["bool"] = "bool"
    shape: list[int]
    nbytes: int
    included_count: Annotated[StrictInt, Field(ge=0, le=MAX_QUALITY_MASK_ELEMENTS)]
    excluded_count: Annotated[StrictInt, Field(ge=0, le=MAX_QUALITY_MASK_ELEMENTS)]
    minimum_passed: StrictBool | None = Field(default=None, exclude_if=lambda value: value is None)
    validity_binding: SelectedValidityBinding | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        total = self.included_count + self.excluded_count
        if total != math.prod(self.shape) or total > MAX_QUALITY_MASK_ELEMENTS:
            raise ValueError("Quality mask counts must match its shape and element limit")
        if self.minimum_included_fraction is None:
            if self.minimum_passed is not None:
                raise ValueError("A quality mask minimum result requires a selected minimum")
        else:
            expected = total > 0 and Fraction(self.included_count, total) >= Fraction(
                Decimal(self.minimum_included_fraction)
            )
            if self.minimum_passed != expected:
                raise ValueError("Quality mask minimum result does not match its counts")
        return self
class ReferenceMaskEvidence(QualityMaskEvidence):
    selected_exact_bytes_equal: StrictBool | None

    @model_validator(mode="after")
    def validate_selected_equality(self) -> Self:
        if self.included_count == 0 and self.selected_exact_bytes_equal is not None:
            raise ValueError("An empty reference selection cannot have a byte comparison")
        return self
class PairwiseSelectedValidityBinding(SelectedValidityBinding):
    mask_version_id: UUID
    mask_checksum: Hash
class PairwiseValidityMaskEvidence(QualityMaskEvidence):
    validity_binding: PairwiseSelectedValidityBinding

    @model_validator(mode="after")
    def binding_matches_mask(self) -> Self:
        if (
            self.validity_binding.mask_version_id != self.version_id
            or self.validity_binding.mask_checksum != self.checksum
        ):
            raise ValueError("Selected validity binding evidence does not match its mask")
        return self
class ReferenceValidityIntersectionEvidence(Model):
    output_mask: PairwiseValidityMaskEvidence
    reference_mask: PairwiseValidityMaskEvidence
    included_count: Annotated[StrictInt, Field(ge=0, le=MAX_QUALITY_MASK_ELEMENTS)]
    excluded_count: Annotated[StrictInt, Field(ge=0, le=MAX_QUALITY_MASK_ELEMENTS)]
    minimum_included_fraction: DecimalBound | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    minimum_passed: StrictBool | None = Field(default=None, exclude_if=lambda value: value is None)
    selected_exact_bytes_equal: StrictBool | None

    @model_validator(mode="after")
    def validate_intersection(self) -> Self:
        output, reference = self.output_mask, self.reference_mask
        total = self.included_count + self.excluded_count
        if (
            output.shape != reference.shape
            or total != math.prod(output.shape)
            or total > MAX_QUALITY_MASK_ELEMENTS
            or output.minimum_included_fraction is not None
            or reference.minimum_included_fraction is not None
            or not max(0, output.included_count + reference.included_count - total)
            <= self.included_count
            <= min(output.included_count, reference.included_count)
        ):
            raise ValueError("Validity intersection evidence does not match its masks")
        minimum = QualityMaskCriterion.validate_minimum(self.minimum_included_fraction)
        expected = (
            total > 0 and Fraction(self.included_count, total) >= Fraction(Decimal(minimum))
            if minimum is not None
            else None
        )
        if self.minimum_passed != expected:
            raise ValueError("Validity intersection minimum does not match its counts")
        if self.included_count == 0 and self.selected_exact_bytes_equal is not None:
            raise ValueError("An empty validity intersection cannot have a byte comparison")
        return self
class ReferenceEvidence(_OptionalReferenceComparisonsModel):
    version_id: UUID
    checksum: str
    dtype: DType
    shape: list[int]
    nbytes: int
    shape_matches: bool
    dtype_matches: bool
    semantic_comparison: Literal["v1", "v2", "legacy_unverified", "not_compared"]
    output_semantics: TensorSemantics | None
    reference_semantics: TensorSemantics | None
    exact_bytes_equal: bool | None
    tolerance: ReferenceToleranceEvidence | None = None
    mean_absolute_error: ReferenceMeanAbsoluteErrorEvidence | None = None
    mask: ReferenceMaskEvidence | None = Field(default=None, exclude_if=lambda value: value is None)
    validity_intersection: ReferenceValidityIntersectionEvidence | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    passed: bool

    @model_validator(mode="after")
    def validate_mask_comparison(self) -> Self:
        if self.mask is not None and self.validity_intersection is not None:
            raise ValueError("Reference evidence cannot contain two selection modes")
        selection = self.mask or self.validity_intersection
        if selection is not None:
            comparable = self.shape_matches and self.dtype_matches and selection.included_count > 0
            if comparable != (selection.selected_exact_bytes_equal is not None):
                raise ValueError("Selected byte comparison does not match reference availability")
            if self.passed and (not comparable or selection.minimum_passed is False):
                raise ValueError("A passing reference must include enough comparable positions")
        if self.validity_intersection is not None:
            reference = self.validity_intersection.reference_mask
            binding = reference.validity_binding
            if (
                binding is None
                or binding.data_version_id != self.version_id
                or binding.data_checksum != self.checksum
                or reference.shape != self.shape
            ):
                raise ValueError("Reference validity evidence does not match the reference")
            compared = (
                self.validity_intersection.included_count
                if self.shape_matches and self.dtype_matches
                else None
            )
            for evidence in (self.tolerance, self.mean_absolute_error):
                if evidence is not None and evidence.compared_elements != compared:
                    raise ValueError("Reference comparison count does not match the intersection")
        return self
class TensorEvaluationDetail(_OptionalReferenceModel):
    version_id: UUID
    checksum: str
    dtype: DType
    shape: list[int]
    nbytes: int
    contract_revision: int | None
    contract_hash: str | None
    semantics: Literal["declared", "not_declared", "invalid"] | None
    contract_errors: list[str]
    quality: QualityEvidence | None
    quality_mask: QualityMaskEvidence | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    reference: ReferenceEvidence | None = None
    passed: bool

    @model_validator(mode="after")
    def binding_matches_output(self) -> Self:
        masks = [self.quality_mask, self.reference.mask if self.reference is not None else None]
        if self.reference is not None and self.reference.validity_intersection is not None:
            output = self.reference.validity_intersection.output_mask
            if output.shape != self.shape:
                raise ValueError("Output validity evidence does not match the output shape")
            masks.append(output)
        for mask in masks:
            binding = mask.validity_binding if mask is not None else None
            if binding is not None and (
                binding.data_version_id != self.version_id or binding.data_checksum != self.checksum
            ):
                raise ValueError("Selected validity binding evidence does not match the output")
        return self
class TensorEvaluation(Model):
    format: Literal["tensorus-tensor-evaluation-v1"] = "tensorus-tensor-evaluation-v1"
    status: Literal["evaluated"] = "evaluated"
    dataset: str
    name: str
    criteria: TensorEvaluationRequest
    result: TensorEvaluationDetail
    passed: bool
    goal_verified: Literal[False] = False
    execution_verified: Literal[False] = False
    checked_bytes: int
    limitations: list[str]
