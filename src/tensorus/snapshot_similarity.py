"""Exact similarity evidence over an explicitly pinned snapshot."""
from typing import Annotated
from typing import Literal
from uuid import UUID
from pydantic import Field
from pydantic import JsonValue
from pydantic import StrictInt
from tensorus.contracts import ReleaseHash
from tensorus.models import Model
from tensorus.models import Name
class SnapshotSimilarityRequest(Model):
    snapshot_id: UUID
    version_id: UUID
    metric: Literal["cosine", "l2"] = "cosine"
    k: Annotated[StrictInt, Field(ge=1, le=100)] = 10
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    expected_membership_hash: ReleaseHash | None = None
    expected_candidate_hash: ReleaseHash | None = None
class SnapshotSimilaritySnapshot(Model):
    id: UUID
    dataset_id: UUID
    dataset: Name
    name: Name
    created_at: str
    created_by: str
class SnapshotSimilarityMember(Model):
    name: Name
    version_id: UUID
class SnapshotSimilarityVersion(Model):
    id: UUID
    dataset_id: UUID
    dataset: Name
    name: Name
    version: Annotated[StrictInt, Field(ge=1)]
    nbytes: Annotated[StrictInt, Field(ge=0)]
    checksum: ReleaseHash
    metadata_hash: ReleaseHash
    descriptor_hash: ReleaseHash
