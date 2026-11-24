"""Reviewed tombstones for unreferenced historical data-product revisions."""
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
from tensorus.products import MAX_PRODUCT_BYTES
from tensorus.products import DataProductRevision
from tensorus.products import Revision
from tensorus.retention import MAX_HOLDS_PER_RESOURCE
MAX_REVISION_RETIREMENT_PAGE = 16
MAX_REVISION_RETIREMENT_RECEIPT_BYTES = 64 * 1024
MAX_REVISION_RETIREMENT_BYTES = 2 * MAX_PRODUCT_BYTES + MAX_REVISION_RETIREMENT_RECEIPT_BYTES
_REFERENCE_NAMES = (
    "active_release_referenced",
    "retired_release_referenced",
    "imported_revision_referenced",
    "active_similarity_referenced",
    "withdrawn_similarity_referenced",
    "retired_similarity_referenced",
    "source_binding_referenced",
)
class ProductRevisionRetirementReviewRequest(Model):
    descriptor_hash: ReleaseHash
    reason: WithdrawalReason
class ProductRevisionRetirementRequest(ProductRevisionRetirementReviewRequest):
    review_hash: ReleaseHash
class ProductRevisionRetirementReview(Model):
    format: Literal[
        "tensorus-product-revision-retirement-review-v1",
        "tensorus-product-revision-retirement-review-v2",
    ] = "tensorus-product-revision-retirement-review-v1"
    tenant_id: UUID
    product_id: UUID
    product_name: Name
    product_revision: Revision
    descriptor_hash: ReleaseHash
    head_revision: Revision
    reviewed_by: Annotated[str, Field(strict=True, min_length=1, max_length=36)]
    reason: WithdrawalReason
    active_release_referenced: StrictBool
    retired_release_referenced: StrictBool
    imported_revision_referenced: StrictBool
    active_similarity_referenced: StrictBool
    withdrawn_similarity_referenced: StrictBool
    retired_similarity_referenced: StrictBool
    source_binding_referenced: StrictBool
    receipt_capacity_available: StrictBool
    active_hold_count: Annotated[StrictInt, Field(ge=0, le=MAX_HOLDS_PER_RESOURCE)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    hold_history_hash: ReleaseHash | None = Field(default=None, exclude_if=lambda v: v is None)
    can_retire: StrictBool
    payloads_reverified: Literal[False] = False
    tensor_data_deleted: Literal[False] = False
    descriptor_retained: Literal[True] = True
    reads_denied: Literal[True] = True
    quota_released: Literal[False] = False

    @model_validator(mode="after")
    def consistent_blockers(self) -> Self:
        if self.format == "tensorus-product-revision-retirement-review-v1":
            if self.active_hold_count is not None or self.hold_history_hash is not None:
                raise ValueError("Legacy product-revision review cannot contain hold evidence")
            hold_clear = True
        else:
            if (
                type(self.active_hold_count) is not int
                or not 0 <= self.active_hold_count <= MAX_HOLDS_PER_RESOURCE
                or self.hold_history_hash is None
            ):
                raise ValueError("Product-revision hold evidence is invalid")
            hold_clear = self.active_hold_count == 0
        expected = (
            self.product_revision < self.head_revision
            and self.receipt_capacity_available
            and hold_clear
            and not any(getattr(self, name) for name in _REFERENCE_NAMES)
        )
        if self.product_revision > self.head_revision or self.can_retire != expected:
            raise ValueError("Product-revision retirement blockers are inconsistent")
        return self
class ProductRevisionRetirementReviewResult(Model):
    review: ProductRevisionRetirementReview
    review_hash: ReleaseHash

    @model_validator(mode="after")
    def consistent_hash(self) -> Self:
        if self.review_hash != canonical_hash(self.review.model_dump(mode="json")):
            raise ValueError("Product-revision retirement review hash is inconsistent")
        return self
class ProductRevisionRetirementReceipt(Model):
    format: Literal[
        "tensorus-product-revision-retirement-v1", "tensorus-product-revision-retirement-v2"
    ] = "tensorus-product-revision-retirement-v1"
    review: ProductRevisionRetirementReview
    review_hash: ReleaseHash
    retired_by: Annotated[str, Field(strict=True, min_length=1, max_length=36)]
    retired_at: Annotated[str, Field(strict=True, min_length=1, max_length=40)]

    @model_validator(mode="after")
    def consistent_review(self) -> Self:
        if (
            self.review_hash != canonical_hash(self.review.model_dump(mode="json"))
            or not self.review.can_retire
            or self.retired_by != self.review.reviewed_by
            or (self.format == "tensorus-product-revision-retirement-v2")
            != (self.review.format == "tensorus-product-revision-retirement-review-v2")
        ):
            raise ValueError("Product-revision retirement receipt is inconsistent")
        return self
class ProductRevisionRetirement(Model):
    product: DataProductRevision
    receipt: ProductRevisionRetirementReceipt
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
            or product.revision != review.product_revision
            or product.descriptor_hash != review.descriptor_hash
            or self.receipt_hash != canonical_hash(self.receipt.model_dump(mode="json"))
        ):
            raise ValueError("Product-revision retirement evidence is inconsistent")
        return self
class ProductRevisionRetirementSummary(Model):
    receipt: ProductRevisionRetirementReceipt
    receipt_hash: ReleaseHash

    @model_validator(mode="after")
    def consistent_hash(self) -> Self:
        if self.receipt_hash != canonical_hash(self.receipt.model_dump(mode="json")):
            raise ValueError("Product-revision retirement receipt hash is inconsistent")
        return self
