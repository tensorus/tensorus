"""Retained, separately reviewed validity-selected product comparisons."""
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
from tensorus.products import Revision
from tensorus.validity_similarity import ValiditySnapshotSimilarityReport
from tensorus.validity_similarity import ValiditySnapshotSimilarityRequest
PRODUCT_SIMILARITY_EVALUATION_FORMAT: Literal["tensorus-product-similarity-evaluation-v1"] = (
    "tensorus-product-similarity-evaluation-v1"
)
MAX_PRODUCT_SIMILARITY_EVALUATION_BYTES = 8 * 1024**2
PRODUCT_SIMILARITY_WITHDRAWAL_FORMAT: Literal["tensorus-product-similarity-withdrawal-v1"] = (
    "tensorus-product-similarity-withdrawal-v1"
)
MAX_WITHDRAWN_EVALUATION_BYTES = 2 * MAX_PRODUCT_SIMILARITY_EVALUATION_BYTES + 4096
class ProductSimilarityEvaluationRequest(Model):
    name: Name
    release_id: UUID
    release_report_hash: ReleaseHash
    reference_role: Name
    similarity: ValiditySnapshotSimilarityRequest
class ProductSimilarityEvaluationReport(Model):
    format: Literal["tensorus-product-similarity-evaluation-v1"] = (
        PRODUCT_SIMILARITY_EVALUATION_FORMAT
    )
    tenant_id: UUID
    product_id: UUID
    product_name: Name
    product_revision: Revision
    product_descriptor_hash: ReleaseHash
    release_id: UUID
    release_report_hash: ReleaseHash
    evaluation_name: Name
    proposed_by: str
    reference_role: Name
    similarity: ValiditySnapshotSimilarityReport
class ProductSimilarityEvaluation(Model):
    id: UUID
    tenant_id: UUID
    product_id: UUID
    product_name: Name
    product_revision: Revision
    release_id: UUID
    snapshot_id: UUID
    name: Name
    request: ProductSimilarityEvaluationRequest
    report: ProductSimilarityEvaluationReport
    report_hash: ReleaseHash
    status: Literal["proposed", "approved"]
    created_at: str
    created_by: str
    approved_at: str | None
    approved_by: str | None

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        report = self.report
        if (
            self.request.name != self.name
            or self.request.release_id != self.release_id
            or self.request.similarity.snapshot_id != self.snapshot_id
            or report.tenant_id != self.tenant_id
            or report.product_id != self.product_id
            or report.product_name != self.product_name
            or report.product_revision != self.product_revision
            or report.release_id != self.release_id
            or report.release_report_hash != self.request.release_report_hash
            or report.evaluation_name != self.name
            or report.proposed_by != self.created_by
            or report.reference_role != self.request.reference_role
            or canonical_json(report.similarity.request.model_dump(mode="json"))
            != canonical_json(self.request.similarity.model_dump(mode="json"))
            or canonical_hash(report.model_dump(mode="json")) != self.report_hash
            or len(canonical_json(self.request.model_dump(mode="json")).encode())
            > MAX_PRODUCT_SIMILARITY_EVALUATION_BYTES
            or len(canonical_json(report.model_dump(mode="json")).encode())
            > MAX_PRODUCT_SIMILARITY_EVALUATION_BYTES
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
            raise ValueError("Product similarity evaluation evidence is inconsistent")
        return self
class ProductSimilarityWithdrawalRequest(Model):
    report_hash: ReleaseHash
    reason: WithdrawalReason
class ProductSimilarityWithdrawalReceipt(Model):
    format: Literal["tensorus-product-similarity-withdrawal-v1"] = (
        PRODUCT_SIMILARITY_WITHDRAWAL_FORMAT
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
    proposed_at: str
    withdrawn_by: str
    withdrawn_at: str
    reason: WithdrawalReason
class ProductSimilarityWithdrawal(Model):
    evaluation: ProductSimilarityEvaluation
    receipt: ProductSimilarityWithdrawalReceipt
    receipt_hash: ReleaseHash

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        evaluation, receipt = self.evaluation, self.receipt
        if (
            evaluation.status != "proposed"
            or receipt.tenant_id != evaluation.tenant_id
            or receipt.evaluation_id != evaluation.id
            or receipt.product_id != evaluation.product_id
            or receipt.product_name != evaluation.product_name
            or receipt.product_revision != evaluation.product_revision
            or receipt.release_id != evaluation.release_id
            or receipt.snapshot_id != evaluation.snapshot_id
            or receipt.evaluation_name != evaluation.name
            or receipt.report_hash != evaluation.report_hash
            or receipt.proposed_by != evaluation.created_by
            or receipt.proposed_at != evaluation.created_at
            or receipt.evaluation_hash != canonical_hash(evaluation.model_dump(mode="json"))
            or self.receipt_hash != canonical_hash(receipt.model_dump(mode="json"))
            or len(canonical_json(evaluation.model_dump(mode="json")).encode())
            > MAX_WITHDRAWN_EVALUATION_BYTES
        ):
            raise ValueError("Product similarity withdrawal evidence is inconsistent")
        return self
class ProductSimilarityWithdrawalSummary(Model):
    receipt: ProductSimilarityWithdrawalReceipt
    receipt_hash: ReleaseHash

    @model_validator(mode="after")
    def consistent_receipt(self) -> Self:
        if self.receipt_hash != canonical_hash(self.receipt.model_dump(mode="json")):
            raise ValueError("Product similarity withdrawal receipt hash is inconsistent")
        return self
