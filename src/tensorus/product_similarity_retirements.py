"""Administrator-reviewed retirement of approved product similarity evidence."""
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import model_validator
from tensorus.arrays import canonical_hash
from tensorus.arrays import canonical_json
from tensorus.contracts import ReleaseHash
from tensorus.models import Model
from tensorus.models import Name
from tensorus.models import WithdrawalReason
from tensorus.product_similarity_evaluations import MAX_WITHDRAWN_EVALUATION_BYTES
from tensorus.product_similarity_evaluations import ProductSimilarityEvaluation
from tensorus.products import Revision
class ProductSimilarityRetirementReviewRequest(Model):
    report_hash: ReleaseHash
    reason: WithdrawalReason
class ProductSimilarityRetirementRequest(ProductSimilarityRetirementReviewRequest):
    review_hash: ReleaseHash
class ProductSimilarityRetirementReview(Model):
    format: Literal["tensorus-product-similarity-retirement-review-v1"] = (
        "tensorus-product-similarity-retirement-review-v1"
    )
    tenant_id: UUID
    evaluation_id: UUID
    product_id: UUID
    product_name: Name
    product_revision: Revision
    release_id: UUID
    snapshot_id: UUID
    evaluation_name: Name
    evaluation_hash: ReleaseHash
    report_hash: ReleaseHash
    proposed_by: str
    approved_by: str
    approved_at: str
    reason: WithdrawalReason
    payloads_reverified: Literal[False] = False
    other_snapshot_holds_checked: Literal[False] = False
    tensor_data_deleted: Literal[False] = False
class ProductSimilarityRetirementReviewResult(Model):
    review: ProductSimilarityRetirementReview
    review_hash: ReleaseHash

    @model_validator(mode="after")
    def consistent_review(self) -> Self:
        if self.review_hash != canonical_hash(self.review.model_dump(mode="json")):
            raise ValueError("Product similarity retirement review hash is inconsistent")
        return self
class ProductSimilarityRetirementReceipt(Model):
    format: Literal["tensorus-product-similarity-retirement-v1"] = (
        "tensorus-product-similarity-retirement-v1"
    )
    review: ProductSimilarityRetirementReview
    review_hash: ReleaseHash
    retired_by: str
    retired_at: str

    @model_validator(mode="after")
    def consistent_review(self) -> Self:
        if (
            self.review_hash != canonical_hash(self.review.model_dump(mode="json"))
            or self.retired_by == self.review.approved_by
        ):
            raise ValueError("Product similarity retirement review is inconsistent")
        return self
class ProductSimilarityRetirement(Model):
    evaluation: ProductSimilarityEvaluation
    receipt: ProductSimilarityRetirementReceipt
    receipt_hash: ReleaseHash

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        evaluation, review = self.evaluation, self.receipt.review
        if (
            evaluation.status != "approved"
            or review.tenant_id != evaluation.tenant_id
            or review.evaluation_id != evaluation.id
            or review.product_id != evaluation.product_id
            or review.product_name != evaluation.product_name
            or review.product_revision != evaluation.product_revision
            or review.release_id != evaluation.release_id
            or review.snapshot_id != evaluation.snapshot_id
            or review.evaluation_name != evaluation.name
            or review.report_hash != evaluation.report_hash
            or review.proposed_by != evaluation.created_by
            or review.approved_by != evaluation.approved_by
            or review.approved_at != evaluation.approved_at
            or review.evaluation_hash != canonical_hash(evaluation.model_dump(mode="json"))
            or self.receipt_hash != canonical_hash(self.receipt.model_dump(mode="json"))
            or len(canonical_json(evaluation.model_dump(mode="json")).encode())
            > MAX_WITHDRAWN_EVALUATION_BYTES
        ):
            raise ValueError("Product similarity retirement evidence is inconsistent")
        return self
class ProductSimilarityRetirementSummary(Model):
    receipt: ProductSimilarityRetirementReceipt
    receipt_hash: ReleaseHash

    @model_validator(mode="after")
    def consistent_receipt(self) -> Self:
        if self.receipt_hash != canonical_hash(self.receipt.model_dump(mode="json")):
            raise ValueError("Product similarity retirement receipt hash is inconsistent")
        return self
