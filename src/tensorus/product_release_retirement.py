"""Reviewed retirement of product-release metadata with retained immutable evidence."""
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictInt
from pydantic import model_validator
from tensorus.arrays import canonical_hash
from tensorus.contracts import ReleaseHash
from tensorus.models import Model
from tensorus.models import Name
from tensorus.models import WithdrawalReason
from tensorus.product_releases import ProductRelease
from tensorus.products import Revision
from tensorus.retention import MAX_HOLDS_PER_RESOURCE
MAX_PRODUCT_RELEASE_RETIREMENT_PAGE = 16
MAX_RETIRED_PRODUCT_RELEASE_BYTES = 8 * 1024**2
MAX_PRODUCT_RELEASE_RETIREMENT_RECEIPT_BYTES = 64 * 1024
class ProductReleaseRetirementReviewRequest(Model):
    report_hash: ReleaseHash
    reason: WithdrawalReason
class ProductReleaseRetirementRequest(ProductReleaseRetirementReviewRequest):
    review_hash: ReleaseHash
class ProductReleaseRetirementReview(Model):
    format: Literal["tensorus-product-release-retirement-review-v1"] = (
        "tensorus-product-release-retirement-review-v1"
    )
    tenant_id: UUID
    release_id: UUID
    product_id: UUID
    product_name: Name
    product_revision: Revision
    release_name: Name
    status: Literal["proposed", "approved"]
    release_hash: ReleaseHash
    report_hash: ReleaseHash
    proposed_by: str
    approved_by: str | None
    reviewed_by: str
    reason: WithdrawalReason
    active_similarity_referenced: bool
    withdrawn_similarity_referenced: bool
    retired_similarity_referenced: bool
    source_binding_referenced: bool
    imported_revision_referenced: bool
    active_hold_count: Annotated[StrictInt, Field(ge=0, le=MAX_HOLDS_PER_RESOURCE)]
    hold_history_hash: ReleaseHash
    receipt_capacity_available: bool
    different_from_approver: bool
    can_retire: bool
    payloads_reverified: Literal[False] = False
    tensor_data_deleted: Literal[False] = False

    @model_validator(mode="after")
    def consistent_blockers(self) -> Self:
        expected = not (
            self.active_similarity_referenced
            or self.withdrawn_similarity_referenced
            or self.retired_similarity_referenced
            or self.source_binding_referenced
            or self.imported_revision_referenced
            or self.active_hold_count
            or not self.receipt_capacity_available
            or not self.different_from_approver
        )
        if (
            self.can_retire != expected
            or self.different_from_approver != (self.approved_by != self.reviewed_by)
            or (self.status == "approved") != (self.approved_by is not None)
        ):
            raise ValueError("Product-release retirement blockers are inconsistent")
        return self
class ProductReleaseRetirementReviewResult(Model):
    review: ProductReleaseRetirementReview
    review_hash: ReleaseHash

    @model_validator(mode="after")
    def consistent_hash(self) -> Self:
        if self.review_hash != canonical_hash(self.review.model_dump(mode="json")):
            raise ValueError("Product-release retirement review hash is inconsistent")
        return self
class ProductReleaseRetirementReceipt(Model):
    format: Literal["tensorus-product-release-retirement-v1"] = (
        "tensorus-product-release-retirement-v1"
    )
    review: ProductReleaseRetirementReview
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
            raise ValueError("Product-release retirement receipt is inconsistent")
        return self
class ProductReleaseRetirement(Model):
    release: ProductRelease
    receipt: ProductReleaseRetirementReceipt
    receipt_hash: ReleaseHash

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        release, review = self.release, self.receipt.review
        if (
            review.tenant_id != release.tenant_id
            or review.release_id != release.id
            or review.product_id != release.product_id
            or review.product_name != release.product_name
            or review.product_revision != release.product_revision
            or review.release_name != release.name
            or review.status != release.status
            or review.report_hash != release.report_hash
            or review.proposed_by != release.created_by
            or review.approved_by != release.approved_by
            or release.request.name != release.name
            or release.request.product_revision != release.product_revision
            or release.report.tenant_id != release.tenant_id
            or release.report.product_id != release.product_id
            or release.report.product_name != release.product_name
            or release.report.product_revision != release.product_revision
            or release.report.descriptor_hash != release.request.descriptor_hash
            or release.report.release_name != release.name
            or release.report.proposed_by != release.created_by
            or (
                release.status == "approved"
                and (
                    release.approved_at is None
                    or release.approved_by is None
                    or release.approved_by == release.created_by
                )
            )
            or (
                release.status == "proposed"
                and (release.approved_at is not None or release.approved_by is not None)
            )
            or review.release_hash != canonical_hash(release.model_dump(mode="json"))
            or release.report_hash != canonical_hash(release.report.model_dump(mode="json"))
            or self.receipt_hash != canonical_hash(self.receipt.model_dump(mode="json"))
        ):
            raise ValueError("Product-release retirement evidence is inconsistent")
        return self
class ProductReleaseRetirementSummary(Model):
    receipt: ProductReleaseRetirementReceipt
    receipt_hash: ReleaseHash

    @model_validator(mode="after")
    def consistent_receipt(self) -> Self:
        if self.receipt_hash != canonical_hash(self.receipt.model_dump(mode="json")):
            raise ValueError("Product-release retirement receipt hash is inconsistent")
        return self
