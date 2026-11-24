"""Retained, separately reviewed pairwise tensor reference comparisons."""
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import model_validator
from tensorus.arrays import canonical_hash
from tensorus.arrays import canonical_json
from tensorus.evaluation import ReferenceCriterion
from tensorus.evaluation import ReferenceValidityIntersection
from tensorus.evaluation import TensorEvaluation
from tensorus.evaluation import TensorEvaluationRequest
from tensorus.models import Model
from tensorus.models import Name
from tensorus.models import WithdrawalReason
from tensorus.releases import MAX_REPORT_BYTES
from tensorus.validity import Hash
REFERENCE_EVALUATION_FORMAT: Literal["tensorus-reference-evaluation-v1"] = (
    "tensorus-reference-evaluation-v1"
)
REFERENCE_RETIREMENT_REVIEW_FORMAT: Literal["tensorus-reference-retirement-review-v1"] = (
    "tensorus-reference-retirement-review-v1"
)
REFERENCE_ARCHIVE_FORMAT: Literal["tensorus-reference-evaluation-archive-v1"] = (
    "tensorus-reference-evaluation-archive-v1"
)
MAX_REFERENCE_REQUEST_BYTES = 128 * 1024
MAX_REFERENCE_ARCHIVE_BYTES = 2 * MAX_REPORT_BYTES + MAX_REFERENCE_REQUEST_BYTES + 8192
MAX_REFERENCE_RECEIPT_BYTES = 8192
class RetainedReferenceCriterion(ReferenceCriterion):
    mask: None = Field(default=None, exclude_if=lambda value: value is None)
    validity_intersection: ReferenceValidityIntersection
class ReferenceEvaluationRequest(Model):
    name: Name
    output_version_id: UUID
    expected_checksum: Hash
    reference: RetainedReferenceCriterion

    def tensor_request(self) -> TensorEvaluationRequest:
        return TensorEvaluationRequest(
            expected_checksum=self.expected_checksum,
            reference=ReferenceCriterion.model_validate(self.reference.model_dump(mode="json")),
        )
class ReferenceEvaluation(Model):
    format: Literal["tensorus-reference-evaluation-v1"] = REFERENCE_EVALUATION_FORMAT
    id: UUID
    tenant_id: UUID
    name: Name
    request: ReferenceEvaluationRequest
    report: TensorEvaluation
    report_hash: Hash
    status: Literal["proposed", "approved"]
    created_at: str
    created_by: str
    approved_at: str | None
    approved_by: str | None

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        request, report = self.request, self.report
        reference = report.result.reference
        selection = request.reference.validity_intersection
        evidence = reference.validity_intersection if reference is not None else None
        if selection is None or evidence is None or reference is None:
            raise ValueError("Retained reference evidence requires both validity selections")
        output_binding = evidence.output_mask.validity_binding
        reference_binding = evidence.reference_mask.validity_binding
        if (
            self.name != request.name
            or report.criteria != request.tensor_request()
            or report.result.version_id != request.output_version_id
            or report.result.checksum != request.expected_checksum
            or reference.version_id != request.reference.version_id
            or output_binding.id != selection.output_binding.binding_id
            or output_binding.descriptor_hash != selection.output_binding.descriptor_hash
            or reference_binding.id != selection.reference_binding.binding_id
            or reference_binding.descriptor_hash != selection.reference_binding.descriptor_hash
            or evidence.minimum_included_fraction != selection.minimum_included_fraction
            or report.passed != report.result.passed
            or report.passed != reference.passed
            or self.report_hash != canonical_hash(report.model_dump(mode="json"))
            or len(canonical_json(request.model_dump(mode="json")).encode())
            > MAX_REFERENCE_REQUEST_BYTES
            or len(canonical_json(report.model_dump(mode="json")).encode()) > MAX_REPORT_BYTES
            or (
                self.status == "proposed"
                and (self.approved_at is not None or self.approved_by is not None)
            )
            or (
                self.status == "approved"
                and (
                    self.approved_at is None
                    or self.approved_by is None
                    or self.approved_by == self.created_by
                )
            )
        ):
            raise ValueError("Retained reference evaluation evidence is inconsistent")
        return self
class ReferenceEvaluationSummary(Model):
    id: UUID
    tenant_id: UUID
    name: Name
    report_hash: Hash
    status: Literal["proposed", "approved"]
    created_at: str
    created_by: str
    approved_at: str | None
    approved_by: str | None
class ReferenceDispositionRequest(Model):
    report_hash: Hash
    reason: WithdrawalReason
class ReferenceRetirementRequest(ReferenceDispositionRequest):
    review_hash: Hash
class ReferenceRetirementReview(Model):
    format: Literal["tensorus-reference-retirement-review-v1"] = REFERENCE_RETIREMENT_REVIEW_FORMAT
    tenant_id: UUID
    evaluation_id: UUID
    name: Name
    evaluation_hash: Hash
    report_hash: Hash
    proposed_by: str
    approved_by: str
    approved_at: str
    reason: WithdrawalReason
    payloads_reverified: Literal[False] = False
    other_holds_checked: Literal[False] = False
    tensor_data_deleted: Literal[False] = False
class ReferenceRetirementReviewResult(Model):
    review: ReferenceRetirementReview
    review_hash: Hash

    @model_validator(mode="after")
    def consistent_hash(self) -> Self:
        if self.review_hash != canonical_hash(self.review.model_dump(mode="json")):
            raise ValueError("Reference retirement review hash is inconsistent")
        return self
class ReferenceArchiveReceipt(Model):
    format: Literal["tensorus-reference-evaluation-archive-v1"] = REFERENCE_ARCHIVE_FORMAT
    action: Literal["withdrawn", "retired"]
    tenant_id: UUID
    evaluation_id: UUID
    name: Name
    evaluation_hash: Hash
    report_hash: Hash
    reason: WithdrawalReason
    created_at: str
    created_by: str
    review: ReferenceRetirementReview | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    review_hash: Hash | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def consistent_review(self) -> Self:
        if self.action == "withdrawn":
            if self.review is not None or self.review_hash is not None:
                raise ValueError("A withdrawal cannot contain a retirement review")
        elif (
            self.review is None
            or self.review_hash != canonical_hash(self.review.model_dump(mode="json"))
            or self.review.tenant_id != self.tenant_id
            or self.review.evaluation_id != self.evaluation_id
            or self.review.name != self.name
            or self.review.evaluation_hash != self.evaluation_hash
            or self.review.report_hash != self.report_hash
            or self.review.reason != self.reason
            or self.created_by == self.review.approved_by
        ):
            raise ValueError("Reference retirement receipt is inconsistent")
        return self
class ReferenceEvaluationArchive(Model):
    evaluation: ReferenceEvaluation
    receipt: ReferenceArchiveReceipt
    receipt_hash: Hash

    @model_validator(mode="after")
    def consistent_archive(self) -> Self:
        evaluation, receipt = self.evaluation, self.receipt
        review = receipt.review
        if (
            receipt.tenant_id != evaluation.tenant_id
            or receipt.evaluation_id != evaluation.id
            or receipt.name != evaluation.name
            or receipt.report_hash != evaluation.report_hash
            or receipt.evaluation_hash != canonical_hash(evaluation.model_dump(mode="json"))
            or self.receipt_hash != canonical_hash(receipt.model_dump(mode="json"))
            or len(canonical_json(evaluation.model_dump(mode="json")).encode())
            > MAX_REFERENCE_ARCHIVE_BYTES
            or len(canonical_json(receipt.model_dump(mode="json")).encode())
            > MAX_REFERENCE_RECEIPT_BYTES
            or (receipt.action == "withdrawn" and evaluation.status != "proposed")
            or (receipt.action == "retired" and evaluation.status != "approved")
            or (
                review is not None
                and (
                    review.proposed_by != evaluation.created_by
                    or review.approved_by != evaluation.approved_by
                    or review.approved_at != evaluation.approved_at
                )
            )
        ):
            raise ValueError("Reference evaluation archive is inconsistent")
        return self
class ReferenceArchiveSummary(Model):
    receipt: ReferenceArchiveReceipt
    receipt_hash: Hash

    @model_validator(mode="after")
    def consistent_hash(self) -> Self:
        if self.receipt_hash != canonical_hash(self.receipt.model_dump(mode="json")):
            raise ValueError("Reference archive receipt hash is inconsistent")
        return self
