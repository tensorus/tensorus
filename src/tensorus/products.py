"""Versioned, explicitly pinned tensor data-product descriptors."""
from collections.abc import Mapping
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import JsonValue
from pydantic import StrictInt
from pydantic import model_validator
from tensorus import responses as dto
from tensorus.arrays import Record
from tensorus.arrays import canonical_hash as _hash
from tensorus.contracts import ReleaseHash
from tensorus.contracts import ReleaseQualityMask
from tensorus.errors import TensorusError
from tensorus.models import DType
from tensorus.models import Model
from tensorus.models import Name
from tensorus.models import WriteContractBinding
from tensorus.semantics import TensorSemantics
from tensorus.semantics import read_semantics
from tensorus.semantics import require_exact_shared_axis
from tensorus.validity import ValidityBinding
PRODUCT_FORMAT: Literal["tensorus-data-product-v1"] = "tensorus-data-product-v1"
PRODUCT_RELATIONSHIP_FORMAT: Literal["tensorus-data-product-v2"] = "tensorus-data-product-v2"
MAX_PRODUCT_MEMBERS = 32
MAX_PRODUCT_RELATIONSHIPS = 32
MAX_REVISIONS_PER_PRODUCT = 64
MAX_PRODUCT_BYTES = 1024**2
Revision = Annotated[StrictInt, Field(ge=1, le=2**31 - 1)]
class ProductValiditySelection(Model):
    binding_id: UUID
    descriptor_hash: ReleaseHash
    mask_role: Name
class ProductReleaseSelection(Model):
    release_id: UUID
    report_hash: ReleaseHash
class ProductMemberSelection(Model):
    role: Name
    version_id: UUID
    contract_revision: Revision
    contract_hash: ReleaseHash
    validity: ProductValiditySelection | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    quality_release: ProductReleaseSelection | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
class ProductRelationshipEndpoint(Model):
    role: Name
    axis: Name
class ProductAxisRelationship(Model):
    id: Name
    kind: Literal["exact_shared_axis_v1"]
    left: ProductRelationshipEndpoint
    right: ProductRelationshipEndpoint

    @model_validator(mode="after")
    def distinct_roles(self) -> Self:
        if self.left.role == self.right.role:
            raise ValueError("A shared-axis relationship requires two distinct roles")
        return self
def _check_relationship_selections(
    relationships: list[ProductAxisRelationship], roles: set[str]
) -> None:
    identities: set[str] = set()
    pairs: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    for relation in relationships:
        if relation.left.role not in roles or relation.right.role not in roles:
            raise ValueError("Relationship endpoints must name product member roles")
        left = (relation.left.role, relation.left.axis)
        right = (relation.right.role, relation.right.axis)
        pair = (min(left, right), max(left, right))
        if relation.id in identities or pair in pairs:
            raise ValueError("Relationship identities and endpoint pairs must be distinct")
        identities.add(relation.id)
        pairs.add(pair)
class DataProductWrite(Model):
    expected_revision: Annotated[StrictInt, Field(ge=0, le=2**31 - 1)]
    title: Annotated[str, Field(max_length=256)] = ""
    description: Annotated[str, Field(max_length=4096)] = ""
    members: list[ProductMemberSelection] = Field(min_length=1, max_length=MAX_PRODUCT_MEMBERS)
    relationships: list[ProductAxisRelationship] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_PRODUCT_RELATIONSHIPS,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def distinct_roles(self) -> Self:
        roles = {member.role for member in self.members}
        if len(roles) != len(self.members):
            raise ValueError("Data-product member roles must be distinct")
        if any(
            member.validity and member.validity.mask_role not in roles for member in self.members
        ):
            raise ValueError("Each selected validity mask must name a member role")
        if self.relationships is not None:
            _check_relationship_selections(self.relationships, roles)
        return self
class ProductVersion(Model):
    id: UUID
    tenant_id: UUID
    tensor_id: UUID
    dataset_id: UUID
    version: Revision
    name: Name
    dataset: Name
    dtype: DType
    shape: list[Annotated[StrictInt, Field(ge=0, le=2**63 - 1)]] = Field(max_length=16)
    nbytes: Annotated[StrictInt, Field(ge=0, le=2**63 - 1)]
    checksum: ReleaseHash
    metadata: dict[str, JsonValue]
    statistics: dto.Statistics
    statistics_scope: Literal["raw_stored_values"] = "raw_stored_values"
    operation: dict[str, JsonValue] | None
    parents: list[UUID]
    write_contract: WriteContractBinding | None
    access: dto.VersionAccess | None
    created_at: str
    created_by: str
class ProductValidityEvidence(Model):
    mask_role: Name
    binding: ValidityBinding
class ProductReleaseEvidence(Model):
    release_id: UUID
    report_hash: ReleaseHash
    snapshot_id: UUID
    approved_at: str
    approved_by: str
    quality_mask: ReleaseQualityMask | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
class ProductMember(Model):
    role: Name
    version: ProductVersion
    contract: dto.ContractRevision
    validity: ProductValidityEvidence | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    quality_release: ProductReleaseEvidence | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
class DataProductDefinition(Model):
    format: Literal["tensorus-data-product-v1", "tensorus-data-product-v2"] = PRODUCT_FORMAT
    title: str
    description: str
    members: list[ProductMember] = Field(min_length=1, max_length=MAX_PRODUCT_MEMBERS)
    relationships: list[ProductAxisRelationship] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_PRODUCT_RELATIONSHIPS,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def consistent_relationships(self) -> Self:
        if (self.format == PRODUCT_RELATIONSHIP_FORMAT) != (self.relationships is not None):
            raise ValueError("Product descriptor format does not match its relationships")
        if self.relationships is not None:
            _check_relationship_selections(
                self.relationships, {member.role for member in self.members}
            )
        return self
class DataProductRevision(Model):
    id: UUID
    tenant_id: UUID
    name: Name
    revision: Revision
    request: DataProductWrite
    request_hash: ReleaseHash
    definition: DataProductDefinition
    definition_hash: ReleaseHash
    descriptor_hash: ReleaseHash
    created_at: str
    created_by: str
def _envelope(
    row: Mapping[Any, Any],
    product_format: Literal[
        "tensorus-data-product-v1", "tensorus-data-product-v2"
    ] = PRODUCT_FORMAT,
) -> Record:
    return {
        "format": product_format,
        "id": row["id"],
        "tenant_id": row["tenant_id"],
        "name": row["name"],
        "revision": row["revision"],
        "request_hash": row["request_hash"],
        "definition_hash": row["definition_hash"],
        "created_at": row["created_at"],
        "created_by": row["created_by"],
    }
def _check_relationships(
    members: list[ProductMember], relationships: list[ProductAxisRelationship] | None
) -> None:
    if relationships is None:
        return
    by_role = {member.role: member for member in members}
    _check_relationship_selections(relationships, set(by_role))
    semantics: dict[str, TensorSemantics] = {}
    for relation in relationships:
        left, right = by_role[relation.left.role], by_role[relation.right.role]
        if left.version.id == right.version.id:
            raise TensorusError("Shared-axis relationship requires distinct member versions")
        for member in (left, right):
            if member.role not in semantics:
                declaration = read_semantics(
                    member.version.metadata,
                    tuple(member.version.shape),
                    member.version.operation,
                )
                if declaration is None:
                    raise TensorusError(
                        "Shared-axis relationship requires declared version 2 semantics"
                    )
                semantics[member.role] = declaration
        require_exact_shared_axis(
            semantics[left.role],
            semantics[right.role],
            tuple(left.version.shape),
            tuple(right.version.shape),
            relation.left.axis,
            relation.right.axis,
        )
