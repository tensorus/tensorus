"""Portable, explicitly mapped product-release evidence without transferred authority."""
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import model_validator
from tensorus.arrays import Record
from tensorus.arrays import canonical_json
from tensorus.arrays import descriptor_nbytes
from tensorus.contracts import ReleaseHash
from tensorus.errors import Conflict
from tensorus.errors import LimitExceeded
from tensorus.errors import TensorusError
from tensorus.models import DType
from tensorus.models import Model
from tensorus.models import Name
from tensorus.models import TensorSpec
from tensorus.product_releases import MAX_PORTABLE_PRODUCT_BYTES
from tensorus.product_releases import PRODUCT_RELEASE_COORDINATE_OUTCOME_FORMAT
from tensorus.product_releases import PRODUCT_RELEASE_RELATIONSHIP_FORMAT
from tensorus.product_releases import PRODUCT_RELEASE_SCOPED_OUTCOME_FORMAT
from tensorus.product_releases import ProductRelease
from tensorus.product_releases import product_release_report
from tensorus.products import MAX_PRODUCT_MEMBERS
from tensorus.products import DataProductRevision
from tensorus.products import ProductMember
from tensorus.products import Revision
from tensorus.products import _check_relationships
from tensorus.products import _envelope
from tensorus.products import _hash
from tensorus.reference_evaluations import ReferenceArchiveReceipt
from tensorus.reference_evaluations import ReferenceEvaluation
from tensorus.reference_evaluations import ReferenceEvaluationArchive
from tensorus.release_portability import PortableRelease
from tensorus.validity import _check_pair_descriptors
from tensorus.validity import _descriptor_hash
PORTABLE_PRODUCT_FORMAT: Literal["tensorus-portable-product-release-v1"] = (
    "tensorus-portable-product-release-v1"
)
PORTABLE_PRODUCT_REFERENCE_FORMAT: Literal["tensorus-portable-product-release-v2"] = (
    "tensorus-portable-product-release-v2"
)
PORTABLE_PRODUCT_OUTCOME_FORMAT: Literal["tensorus-portable-product-release-v3"] = (
    "tensorus-portable-product-release-v3"
)
PORTABLE_PRODUCT_SCOPED_OUTCOME_FORMAT: Literal["tensorus-portable-product-release-v4"] = (
    "tensorus-portable-product-release-v4"
)
PORTABLE_PRODUCT_COORDINATE_OUTCOME_FORMAT: Literal["tensorus-portable-product-release-v5"] = (
    "tensorus-portable-product-release-v5"
)
PORTABLE_PRODUCT_RELATIONSHIP_FORMAT: Literal["tensorus-portable-product-release-v6"] = (
    "tensorus-portable-product-release-v6"
)
def _validate_source_product(product: DataProductRevision) -> None:
    encoded = product.model_dump(mode="json")
    if (
        product.request.expected_revision != product.revision - 1
        or product.request.title != product.definition.title
        or product.request.description != product.definition.description
        or _hash(encoded["request"]) != product.request_hash
        or _hash(encoded["definition"]) != product.definition_hash
        or _hash(_envelope(encoded, product.definition.format)) != product.descriptor_hash
        or [item.role for item in product.request.members]
        != [item.role for item in product.definition.members]
        or product.request.relationships != product.definition.relationships
    ):
        raise ValueError("Portable product descriptor hashes or selections do not match")
    versions: dict[UUID, Record] = {}
    members = {member.role: member for member in product.definition.members}
    for selected, member in zip(product.request.members, product.definition.members, strict=True):
        version = member.version.model_dump(mode="json")
        try:
            descriptor_nbytes(version)
        except TensorusError as exc:
            raise ValueError("Portable product version descriptor is invalid") from exc
        if member.version.id in versions and versions[member.version.id] != version:
            raise ValueError("Portable product repeats a version with different descriptors")
        versions[member.version.id] = version
        if (
            member.version.tenant_id != product.tenant_id
            or selected.version_id != member.version.id
            or member.contract.dataset_id != member.version.dataset_id
            or selected.contract_revision != member.contract.revision
            or selected.contract_hash != member.contract.definition_hash
            or _hash(member.contract.definition.model_dump(mode="json"))
            != member.contract.definition_hash
            or (selected.validity is None) != (member.validity is None)
            or (selected.quality_release is None) != (member.quality_release is None)
        ):
            raise ValueError("Portable product member does not match its selection or contract")
        if selected.validity is not None and member.validity is not None:
            binding = member.validity.binding
            mask = members.get(selected.validity.mask_role)
            if mask is None:
                raise ValueError("Portable product validity mask role is missing")
            mask_version = mask.version.model_dump(mode="json")
            if (
                selected.validity.binding_id != binding.id
                or selected.validity.descriptor_hash != binding.descriptor_hash
                or selected.validity.mask_role != member.validity.mask_role
                or binding.tenant_id != product.tenant_id
                or binding.data_version_id != member.version.id
                or binding.data_checksum != member.version.checksum
                or binding.data_dtype != member.version.dtype
                or binding.mask_version_id != mask.version.id
                or binding.mask_checksum != mask.version.checksum
                or mask.version.dtype != DType.bool
                or binding.shape != member.version.shape
                or binding.shape != mask.version.shape
                or _descriptor_hash(
                    version,
                    mask_version,
                    binding.origin.model_dump(mode="json"),
                    str(binding.id),
                    binding.created_at,
                    binding.created_by,
                )
                != binding.descriptor_hash
            ):
                raise ValueError("Portable product validity binding does not match its roles")
            try:
                _check_pair_descriptors(version, mask_version)
            except TensorusError as exc:
                raise ValueError("Portable product validity axes are incompatible") from exc
        if (
            selected.quality_release is not None
            and member.quality_release is not None
            and (
                selected.quality_release.release_id != member.quality_release.release_id
                or selected.quality_release.report_hash != member.quality_release.report_hash
            )
        ):
            raise ValueError("Portable product quality selection does not match")
    try:
        _check_relationships(product.definition.members, product.definition.relationships)
    except TensorusError as exc:
        raise ValueError("Portable product relationship declarations are incompatible") from exc
def _validate_source_quality(member: ProductMember, portable: PortableRelease) -> None:
    evidence = member.quality_release
    if evidence is None:
        raise ValueError("Portable product contains unselected quality evidence")
    release, report = portable.release, portable.release.report
    selected = next((item for item in report.members if item.version_id == member.version.id), None)
    if selected is None:
        raise ValueError("Portable quality release does not contain its product member")
    effective = selected.contract_override
    revision = effective.revision if effective is not None else report.contract_revision
    contract_hash = effective.definition_hash if effective is not None else report.contract_hash
    if (
        release.id != evidence.release_id
        or release.status != "approved"
        or release.report_hash != evidence.report_hash
        or release.snapshot_id != evidence.snapshot_id
        or release.approved_at != evidence.approved_at
        or release.approved_by != evidence.approved_by
        or release.tenant_id != member.version.tenant_id
        or release.dataset_id != member.version.dataset_id
        or selected.name != member.version.name
        or selected.dtype != member.version.dtype
        or selected.shape != member.version.shape
        or selected.nbytes != member.version.nbytes
        or selected.checksum != member.version.checksum
        or selected.metadata_hash != _hash(member.version.metadata)
        or selected.operation != member.version.operation
        or selected.parents != [str(item) for item in member.version.parents]
        or revision != member.contract.revision
        or contract_hash != member.contract.definition_hash
        or selected.quality_mask != evidence.quality_mask
    ):
        raise ValueError("Portable quality release does not match its product member")
class PortableProductReference(Model):
    evaluation: ReferenceEvaluation
    retirement: ReferenceArchiveReceipt | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    retirement_hash: ReleaseHash | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def consistent_retirement(self) -> Self:
        if (self.retirement is None) != (self.retirement_hash is None):
            raise ValueError("Portable reference retirement requires its exact receipt and hash")
        if self.retirement is not None and self.retirement_hash is not None:
            archive = ReferenceEvaluationArchive(
                evaluation=self.evaluation,
                receipt=self.retirement,
                receipt_hash=self.retirement_hash,
            )
            if archive.receipt.action != "retired":
                raise ValueError("Portable reference evidence cannot contain a withdrawal")
        return self
class PortableProductRelease(Model):
    format: Literal[
        "tensorus-portable-product-release-v1",
        "tensorus-portable-product-release-v2",
        "tensorus-portable-product-release-v3",
        "tensorus-portable-product-release-v4",
        "tensorus-portable-product-release-v5",
        "tensorus-portable-product-release-v6",
    ] = PORTABLE_PRODUCT_FORMAT
    product: DataProductRevision
    release: ProductRelease
    quality_releases: list[PortableRelease] = Field(max_length=MAX_PRODUCT_MEMBERS)
    reference_evaluations: list[PortableProductReference] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_PRODUCT_MEMBERS,
        exclude_if=lambda value: value is None,
    )
    content_hash: ReleaseHash

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        encoded = self.model_dump(mode="json")
        if len(canonical_json(encoded).encode()) > MAX_PORTABLE_PRODUCT_BYTES:
            raise ValueError("Portable product release exceeds 8 MiB")
        if _hash(self.model_dump(mode="json", exclude={"content_hash"})) != self.content_hash:
            raise ValueError("Portable product release content hash does not match")
        _validate_source_product(self.product)
        release, product = self.release, self.product
        expected_format = (
            PORTABLE_PRODUCT_RELATIONSHIP_FORMAT
            if release.report.format == PRODUCT_RELEASE_RELATIONSHIP_FORMAT
            else PORTABLE_PRODUCT_COORDINATE_OUTCOME_FORMAT
            if release.report.format == PRODUCT_RELEASE_COORDINATE_OUTCOME_FORMAT
            else PORTABLE_PRODUCT_SCOPED_OUTCOME_FORMAT
            if release.report.format == PRODUCT_RELEASE_SCOPED_OUTCOME_FORMAT
            else PORTABLE_PRODUCT_OUTCOME_FORMAT
            if release.request.outcomes is not None
            else PORTABLE_PRODUCT_REFERENCE_FORMAT
            if self.reference_evaluations is not None
            else PORTABLE_PRODUCT_FORMAT
        )
        if self.format != expected_format or (release.request.references is not None) != (
            self.reference_evaluations is not None
        ):
            raise ValueError(
                "Portable format does not match selected outcome or reference evidence"
            )
        evaluations = {
            item.evaluation.id: item.evaluation for item in self.reference_evaluations or []
        }
        selected_evaluations = {item.evaluation_id for item in release.request.references or []}
        if (
            len(evaluations) != len(self.reference_evaluations or [])
            or set(evaluations) != selected_evaluations
        ):
            raise ValueError("Portable reference evaluations do not match product selections")
        if (
            release.status != "approved"
            or release.approved_at is None
            or release.approved_by is None
            or release.approved_by == release.created_by
            or release.tenant_id != product.tenant_id
            or release.product_id != product.id
            or release.product_name != product.name
            or release.product_revision != product.revision
            or release.name != release.request.name
            or _hash(release.report.model_dump(mode="json")) != release.report_hash
        ):
            raise ValueError("Portable product release status or references do not match")
        if len(release.report.members) != len(product.definition.members):
            raise ValueError("Portable product report membership does not match")
        specs: dict[str, TensorSpec] = {}
        for member, evidence in zip(
            product.definition.members, release.report.members, strict=True
        ):
            dataset_id = str(member.version.dataset_id)
            if (
                evidence.role != member.role
                or _hash(evidence.write_spec.model_dump(mode="json")) != evidence.write_spec_hash
                or (dataset_id in specs and specs[dataset_id] != evidence.write_spec)
            ):
                raise ValueError("Portable product write specifications do not match")
            specs[dataset_id] = evidence.write_spec
        try:
            snapshots = {
                UUID(item.evidence.outcome.receipt.plan_id): item
                for item in release.report.outcome_snapshots or []
            }
            if len(snapshots) != len(release.report.outcome_snapshots or []):
                raise ValueError("Portable product repeats a plan outcome")
            expected = product_release_report(
                product, release.request, release.created_by, specs, evaluations, snapshots
            )
        except (Conflict, KeyError, LimitExceeded) as exc:
            raise ValueError("Portable product release report is inconsistent") from exc
        if expected != release.report:
            raise ValueError("Portable product release report does not match its descriptor")
        selected_ids = {
            member.quality_release.release_id
            for member in product.definition.members
            if member.quality_release is not None
        }
        source_releases = {item.release.id: item for item in self.quality_releases}
        if (
            len(source_releases) != len(self.quality_releases)
            or set(source_releases) != selected_ids
        ):
            raise ValueError("Portable quality releases do not match product selections")
        for member, report_member in zip(
            product.definition.members, release.report.members, strict=True
        ):
            if member.quality_release is not None:
                source = source_releases[member.quality_release.release_id]
                _validate_source_quality(member, source)
                if (
                    _hash(source.write_spec.model_dump(mode="json"))
                    != report_member.write_spec_hash
                ):
                    raise ValueError("Portable quality and product write specifications differ")
        return self
class ProductReleaseImportRequest(Model):
    name: Name
    product_revision: Revision
    descriptor_hash: ReleaseHash
    source: PortableProductRelease
class ImportedProductRelease(Model):
    id: UUID
    tenant_id: UUID
    product_id: UUID
    product_name: Name
    product_revision: Revision
    name: Name
    request: ProductReleaseImportRequest
    request_hash: ReleaseHash
    source_hash: ReleaseHash
    created_at: str
    created_by: str
class ImportedProductReleaseSummary(Model):
    id: UUID
    tenant_id: UUID
    product_id: UUID
    product_name: Name
    product_revision: Revision
    name: Name
    source_hash: ReleaseHash
    created_at: str
    created_by: str
