"""Reviewed retirement of quality releases with retained portable evidence."""
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictBool
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_validator
from tensorus.arrays import canonical_hash
from tensorus.contracts import ReleaseHash
from tensorus.models import Model
from tensorus.models import Name
from tensorus.models import WithdrawalReason
from tensorus.release_portability import MAX_PORTABLE_BYTES
from tensorus.release_portability import PortableRelease
from tensorus.releases import MAX_QUALITY_RELEASE_RETIREMENTS_PER_TENANT
from tensorus.retention import MAX_HOLDS_PER_RESOURCE
MAX_QUALITY_RELEASE_RETIREMENT_PAGE = 16
MAX_QUALITY_RELEASE_RETIREMENT_RECEIPT_BYTES = 64 * 1024
MAX_QUALITY_RELEASE_RETIREMENT_BYTES = (
    MAX_PORTABLE_BYTES + MAX_QUALITY_RELEASE_RETIREMENT_RECEIPT_BYTES + 4096
)
class QualityReleaseRetirementReviewRequest(Model):
    report_hash: ReleaseHash
    reason: WithdrawalReason

    @field_validator("reason")
    @classmethod
    def trimmed_reason(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("A trimmed reason is required")
        return value
class QualityReleaseRetirementRequest(QualityReleaseRetirementReviewRequest):
    review_hash: ReleaseHash
class QualityReleaseRetirementReview(Model):
    format: Literal["tensorus-quality-release-retirement-review-v1"] = (
        "tensorus-quality-release-retirement-review-v1"
    )
    tenant_id: UUID
    release_id: UUID
    dataset_id: UUID
    dataset: Name
    release_name: Name
    snapshot_id: UUID
    status: Literal["failed", "proposed", "approved"]
    report_hash: ReleaseHash
    source_hash: ReleaseHash
    proposed_by: str
    approved_by: str | None
    reviewed_by: str
    reason: WithdrawalReason
    product_referenced: StrictBool
    active_snapshot_hold_count: Annotated[StrictInt, Field(ge=0, le=MAX_HOLDS_PER_RESOURCE)]
    snapshot_hold_history_hash: ReleaseHash
    receipt_capacity_available: StrictBool
    different_from_approver: StrictBool
    can_retire: StrictBool
    payloads_reverified: Literal[False] = False
    tensor_data_deleted: Literal[False] = False
    snapshot_deleted: Literal[False] = False

    @model_validator(mode="after")
    def consistent_blockers(self) -> Self:
        expected = not (
            self.product_referenced
            or self.active_snapshot_hold_count
            or not self.receipt_capacity_available
            or not self.different_from_approver
        )
        if (
            self.can_retire != expected
            or self.different_from_approver != (self.approved_by != self.reviewed_by)
            or (self.status == "approved") != (self.approved_by is not None)
            or self.reason != self.reason.strip()
        ):
            raise ValueError("Quality-release retirement blockers are inconsistent")
        return self
class QualityReleaseRetirementReviewResult(Model):
    review: QualityReleaseRetirementReview
    review_hash: ReleaseHash

    @model_validator(mode="after")
    def consistent_hash(self) -> Self:
        if self.review_hash != canonical_hash(self.review.model_dump(mode="json")):
            raise ValueError("Quality-release retirement review hash is inconsistent")
        return self
class QualityReleaseRetirementReceipt(Model):
    format: Literal["tensorus-quality-release-retirement-v1"] = (
        "tensorus-quality-release-retirement-v1"
    )
    review: QualityReleaseRetirementReview
    review_hash: ReleaseHash
    retired_by: str
    retired_at: str

    @model_validator(mode="after")
    def consistent_review(self) -> Self:
        if (
            self.review_hash != canonical_hash(self.review.model_dump(mode="json"))
            or not self.review.can_retire
            or self.retired_by != self.review.reviewed_by
            or self.retired_by == self.review.approved_by
        ):
            raise ValueError("Quality-release retirement receipt is inconsistent")
        return self
class QualityReleaseRetirement(Model):
    source: PortableRelease
    receipt: QualityReleaseRetirementReceipt
    receipt_hash: ReleaseHash

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        source, review = self.source, self.receipt.review
        release = source.release
        if (
            source.lineage is not None
            or source.validity is not None
            or source.plans is not None
            or review.tenant_id != release.tenant_id
            or review.release_id != release.id
            or review.dataset_id != release.dataset_id
            or review.dataset != release.dataset
            or review.release_name != release.name
            or review.snapshot_id != release.snapshot_id
            or review.status != release.status
            or review.report_hash != release.report_hash
            or review.source_hash != source.content_hash
            or review.proposed_by != release.created_by
            or review.approved_by != release.approved_by
            or self.receipt_hash != canonical_hash(self.receipt.model_dump(mode="json"))
        ):
            raise ValueError("Quality-release retirement evidence is inconsistent")
        return self
class QualityReleaseRetirementSummary(Model):
    receipt: QualityReleaseRetirementReceipt
    receipt_hash: ReleaseHash

    @model_validator(mode="after")
    def consistent_hash(self) -> Self:
        if self.receipt_hash != canonical_hash(self.receipt.model_dump(mode="json")):
            raise ValueError("Quality-release retirement receipt hash is inconsistent")
        return self
