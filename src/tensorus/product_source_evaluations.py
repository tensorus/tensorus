"""Reviewed links from retained source claims to local similarity evidence."""
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
from tensorus.products import MAX_PRODUCT_MEMBERS
from tensorus.products import Revision
SOURCE_EVALUATION_REVIEW_FORMAT: Literal["tensorus-product-source-evaluation-review-v1"] = (
    "tensorus-product-source-evaluation-review-v1"
)
QualityBasis = Literal["not_attached", "raw_values", "selected_validity", "other_validity"]
class ProductSourceEvaluationReviewRequest(Model):
    name: Name
    import_id: UUID
    import_request_hash: ReleaseHash
    source_hash: ReleaseHash
    evaluation_id: UUID
    evaluation_report_hash: ReleaseHash
class ProductSourceEvaluationBindingRequest(ProductSourceEvaluationReviewRequest):
    review_hash: ReleaseHash
class ProductSourceEvaluationRole(Model):
    role: Name
    source_version_id: UUID
    destination_version_id: UUID
    checksum: ReleaseHash
    nbytes: Annotated[StrictInt, Field(ge=0)]
    write_spec_hash: ReleaseHash
    contract_hash: ReleaseHash
    validity_mask_role: Name | None
    source_quality_basis: QualityBasis
    source_quality_report_hash: ReleaseHash | None
    destination_quality_basis: QualityBasis
    destination_quality_report_hash: ReleaseHash | None
class ProductSourceEvaluationReview(Model):
    format: Literal["tensorus-product-source-evaluation-review-v1"] = (
        SOURCE_EVALUATION_REVIEW_FORMAT
    )
    name: Name
    tenant_id: UUID
    product_id: UUID
    product_name: Name
    product_revision: Revision
    product_descriptor_hash: ReleaseHash
    import_id: UUID
    import_request_hash: ReleaseHash
    source_hash: ReleaseHash
    imported_by: str
    source_tenant_id: UUID
    source_product_id: UUID
    source_product_descriptor_hash: ReleaseHash
    source_release_id: UUID
    source_release_report_hash: ReleaseHash
    destination_release_id: UUID
    destination_release_report_hash: ReleaseHash
    evaluation_id: UUID
    evaluation_report_hash: ReleaseHash
    evaluation_hash: ReleaseHash
    evaluation_proposed_by: str
    evaluation_approved_by: str
    snapshot_id: UUID
    reference_role: Name
    roles: list[ProductSourceEvaluationRole] = Field(min_length=1, max_length=MAX_PRODUCT_MEMBERS)
    required_verification: Literal["destination_roles_and_exact_similarity_reproduction"] = (
        "destination_roles_and_exact_similarity_reproduction"
    )
    source_authenticated: Literal[False] = False
    source_similarity_evidence_included: Literal[False] = False
    attached_quality_recomputed: Literal[False] = False
    quality_equivalence_established: Literal[False] = False
    physical_truth_established: Literal[False] = False
class ProductSourceEvaluationReviewResult(Model):
    review: ProductSourceEvaluationReview
    review_hash: ReleaseHash

    @model_validator(mode="after")
    def consistent_hash(self) -> Self:
        if self.review_hash != canonical_hash(self.review.model_dump(mode="json")):
            raise ValueError("Source evaluation review hash is inconsistent")
        return self
class ProductSourceEvaluationBinding(Model):
    id: UUID
    tenant_id: UUID
    product_id: UUID
    product_name: Name
    product_revision: Revision
    import_id: UUID
    release_id: UUID
    evaluation_id: UUID
    name: Name
    request: ProductSourceEvaluationBindingRequest
    review: ProductSourceEvaluationReview
    review_hash: ReleaseHash
    created_at: str
    created_by: str

    @model_validator(mode="after")
    def consistent_binding(self) -> Self:
        if (
            self.review_hash != canonical_hash(self.review.model_dump(mode="json"))
            or self.request.review_hash != self.review_hash
            or self.name != self.request.name
            or self.name != self.review.name
            or self.tenant_id != self.review.tenant_id
            or self.product_id != self.review.product_id
            or self.product_name != self.review.product_name
            or self.product_revision != self.review.product_revision
            or self.import_id != self.request.import_id
            or self.import_id != self.review.import_id
            or self.request.import_request_hash != self.review.import_request_hash
            or self.request.source_hash != self.review.source_hash
            or self.release_id != self.review.destination_release_id
            or self.evaluation_id != self.request.evaluation_id
            or self.evaluation_id != self.review.evaluation_id
            or self.request.evaluation_report_hash != self.review.evaluation_report_hash
            or self.created_by == self.review.imported_by
        ):
            raise ValueError("Source evaluation binding is inconsistent")
        return self
class ProductSourceEvaluationBindingSummary(Model):
    id: UUID
    tenant_id: UUID
    product_id: UUID
    product_name: Name
    product_revision: Revision
    import_id: UUID
    release_id: UUID
    evaluation_id: UUID
    name: Name
    review_hash: ReleaseHash
    created_at: str
    created_by: str
