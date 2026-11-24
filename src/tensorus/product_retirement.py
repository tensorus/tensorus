"""Reviewed tombstones for complete data products with retired historical revisions."""
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictBool
from pydantic import StrictInt
from pydantic import model_validator
from tensorus.arrays import canonical_hash
from tensorus.contracts import ReleaseHash
from tensorus.models import Model
from tensorus.models import Name
from tensorus.models import WithdrawalReason
from tensorus.product_revision_retirement import _REFERENCE_NAMES
from tensorus.products import MAX_PRODUCT_BYTES
from tensorus.products import MAX_REVISIONS_PER_PRODUCT
from tensorus.products import DataProductRevision
from tensorus.products import Revision
from tensorus.retention import MAX_HOLDS_PER_RESOURCE
MAX_PRODUCT_RETIREMENT_PAGE = 16
MAX_PRODUCT_RETIREMENT_RECEIPT_BYTES = 64 * 1024
MAX_PRODUCT_RETIREMENT_BYTES = 2 * MAX_PRODUCT_BYTES + MAX_PRODUCT_RETIREMENT_RECEIPT_BYTES
class ProductRetirementReviewRequest(Model):
    descriptor_hash: ReleaseHash
    reason: WithdrawalReason
class ProductRetirementRequest(ProductRetirementReviewRequest):
    review_hash: ReleaseHash
class HistoricalRevisionEvidence(Model):
    revision: Revision
    descriptor_hash: ReleaseHash
    retirement_receipt_hash: ReleaseHash | None
class ProductRetirementReview(Model):
    format: Literal["tensorus-product-retirement-review-v1"] = (
        "tensorus-product-retirement-review-v1"
    )
    tenant_id: UUID
    product_id: UUID
    product_name: Name
    head_revision: Revision
    descriptor_hash: ReleaseHash
    reviewed_by: Annotated[str, Field(strict=True, min_length=1, max_length=36)]
    reason: WithdrawalReason
    historical_revisions: Annotated[
        list[HistoricalRevisionEvidence], Field(max_length=MAX_REVISIONS_PER_PRODUCT - 1)
    ]
    active_release_referenced: StrictBool
    retired_release_referenced: StrictBool
    imported_revision_referenced: StrictBool
    active_similarity_referenced: StrictBool
    withdrawn_similarity_referenced: StrictBool
    retired_similarity_referenced: StrictBool
    source_binding_referenced: StrictBool
    active_hold_count: Annotated[StrictInt, Field(ge=0, le=MAX_HOLDS_PER_RESOURCE)]
    hold_history_hash: ReleaseHash
    receipt_capacity_available: StrictBool
    can_retire: StrictBool
    payloads_reverified: Literal[False] = False
    tensor_data_deleted: Literal[False] = False
    descriptors_retained: Literal[True] = True
    reads_denied: Literal[True] = True
    quota_released: Literal[False] = False

    @model_validator(mode="after")
    def consistent_blockers(self) -> Self:
        if self.head_revision > MAX_REVISIONS_PER_PRODUCT or [
            item.revision for item in self.historical_revisions
        ] != list(range(1, self.head_revision)):
            raise ValueError("Product retirement revision evidence is incomplete")
        expected = (
            all(item.retirement_receipt_hash is not None for item in self.historical_revisions)
            and not any(getattr(self, name) for name in _REFERENCE_NAMES)
            and self.active_hold_count == 0
            and self.receipt_capacity_available
        )
        if self.can_retire != expected:
            raise ValueError("Product retirement blockers are inconsistent")
        return self
class ProductRetirementReviewResult(Model):
    review: ProductRetirementReview
    review_hash: ReleaseHash

    @model_validator(mode="after")
    def consistent_hash(self) -> Self:
        if self.review_hash != canonical_hash(self.review.model_dump(mode="json")):
            raise ValueError("Product retirement review hash is inconsistent")
        return self
class ProductRetirementReceipt(Model):
    format: Literal["tensorus-product-retirement-v1"] = "tensorus-product-retirement-v1"
    review: ProductRetirementReview
    review_hash: ReleaseHash
    retired_by: Annotated[str, Field(strict=True, min_length=1, max_length=36)]
    retired_at: Annotated[str, Field(strict=True, min_length=1, max_length=40)]

    @model_validator(mode="after")
    def consistent_review(self) -> Self:
        if (
            self.review_hash != canonical_hash(self.review.model_dump(mode="json"))
            or not self.review.can_retire
            or self.retired_by != self.review.reviewed_by
        ):
            raise ValueError("Product retirement receipt is inconsistent")
        return self
class ProductRetirementSummary(Model):
    receipt: ProductRetirementReceipt
    receipt_hash: ReleaseHash

    @model_validator(mode="after")
    def consistent_hash(self) -> Self:
        if self.receipt_hash != canonical_hash(self.receipt.model_dump(mode="json")):
            raise ValueError("Product retirement receipt hash is inconsistent")
        return self
class ProductRetirement(Model):
    product: DataProductRevision
    receipt: ProductRetirementReceipt
    receipt_hash: ReleaseHash

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        from tensorus.product_portability import _validate_source_product

        product, review = self.product, self.receipt.review
        _validate_source_product(product)
        if (
            product.id != review.product_id
            or product.tenant_id != review.tenant_id
            or product.name != review.product_name
            or product.revision != review.head_revision
            or product.descriptor_hash != review.descriptor_hash
            or self.receipt_hash != canonical_hash(self.receipt.model_dump(mode="json"))
        ):
            raise ValueError("Product retirement evidence is inconsistent")
        return self
