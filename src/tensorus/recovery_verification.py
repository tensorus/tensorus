"""Bounded, non-destructive verification of selected catalog object references."""
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_validator
from tensorus.arrays import canonical_hash
from tensorus.arrays import descriptor_nbytes
from tensorus.errors import TensorusError
from tensorus.middleware import MAX_RECOVERY_REQUEST_BYTES
from tensorus.models import DType
from tensorus.models import Model
MAX_REMOTE_RECOVERY_VERSIONS = 32
MAX_REMOTE_RECOVERY_LOGICAL_BYTES = 16 * 1024**2
MAX_REMOTE_RECOVERY_DECODED_BYTES = 64 * 1024**2
MAX_REMOTE_RECOVERY_RESPONSE_BYTES = 128 * 1024
RECOVERY_GUIDANCE = (
    "Isolate affected readers and writers and preserve catalog and object evidence. "
    "Restore a verified paired recovery point to an isolated service, reconcile "
    "authority and repeat this selection. Do not delete or overwrite a live "
    "referenced generation."
)
RecoveryHash = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
class RecoveryVerificationRequest(Model):
    version_ids: list[UUID] = Field(min_length=1, max_length=MAX_REMOTE_RECOVERY_VERSIONS)
    max_logical_bytes: StrictInt = Field(
        default=MAX_REMOTE_RECOVERY_LOGICAL_BYTES, ge=1, le=MAX_REMOTE_RECOVERY_LOGICAL_BYTES
    )
    max_decoded_work_bytes: StrictInt = Field(
        default=MAX_REMOTE_RECOVERY_DECODED_BYTES, ge=1, le=MAX_REMOTE_RECOVERY_DECODED_BYTES
    )
    expected_selection_hash: RecoveryHash | None = None

    @model_validator(mode="after")
    def distinct_versions(self) -> Self:
        if len(set(self.version_ids)) != len(self.version_ids):
            raise ValueError("Recovery version IDs must be distinct")
        return self
class RecoveryVerificationCapabilities(Model):
    format: Literal["tensorus-selected-recovery-objects-v1"]
    max_versions: StrictInt
    max_logical_bytes: StrictInt
    max_decoded_work_bytes: StrictInt
    max_request_bytes: StrictInt
    max_response_bytes: StrictInt
    authority: Literal["active_unscoped_tenant_admin"]
    whole_source_checksum: Literal[True]
    selected_versions_only: Literal[True]
    catalog_object_snapshot_atomic: Literal[False]
    source_authenticated: Literal[False]
    mutates_catalog_or_objects: Literal[False]
    hard_memory_limit: Literal[False]
    hard_wall_clock_limit: Literal[False]

    @field_validator(
        "whole_source_checksum",
        "selected_versions_only",
        "catalog_object_snapshot_atomic",
        "source_authenticated",
        "mutates_catalog_or_objects",
        "hard_memory_limit",
        "hard_wall_clock_limit",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Recovery verification flags must be booleans")
        return value

    @model_validator(mode="after")
    def supported_limits(self) -> Self:
        if (
            self.max_versions != MAX_REMOTE_RECOVERY_VERSIONS
            or self.max_logical_bytes != MAX_REMOTE_RECOVERY_LOGICAL_BYTES
            or self.max_decoded_work_bytes != MAX_REMOTE_RECOVERY_DECODED_BYTES
            or self.max_request_bytes != MAX_RECOVERY_REQUEST_BYTES
            or self.max_response_bytes != MAX_REMOTE_RECOVERY_RESPONSE_BYTES
        ):
            raise ValueError("Recovery verification limits do not match the advertised format")
        return self
class RecoveryObjectDescriptor(Model):
    version_id: UUID
    tenant_id: UUID
    dataset_id: UUID
    tensor_id: UUID
    version: StrictInt = Field(ge=1)
    shape: list[Annotated[StrictInt, Field(ge=0)]] = Field(max_length=16)
    dtype: DType
    nbytes: StrictInt = Field(ge=0)
    checksum: RecoveryHash
    object_key_hash: RecoveryHash

    @model_validator(mode="after")
    def validate_descriptor(self) -> Self:
        try:
            descriptor_nbytes(self.model_dump(mode="json"))
        except TensorusError as exc:
            raise ValueError("Recovery object descriptor is inconsistent") from exc
        return self
class RecoveryVerificationVersion(RecoveryObjectDescriptor):
    status: Literal["verified", "unverified"]
    error_code: Literal["storage_error", "limit_exceeded"] | None

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if (self.status == "verified") != (self.error_code is None):
            raise ValueError("Recovery object status and error code disagree")
        return self
class RecoveryVerificationReport(Model):
    format: Literal["tensorus-selected-recovery-objects-v1"]
    selection_hash: RecoveryHash
    status: Literal["all_selected_objects_verified", "incomplete"]
    versions: list[RecoveryVerificationVersion] = Field(
        min_length=1, max_length=MAX_REMOTE_RECOVERY_VERSIONS
    )
    selected_version_count: StrictInt = Field(ge=1, le=MAX_REMOTE_RECOVERY_VERSIONS)
    distinct_object_count: StrictInt = Field(ge=1, le=MAX_REMOTE_RECOVERY_VERSIONS)
    distinct_logical_bytes: StrictInt = Field(ge=0, le=MAX_REMOTE_RECOVERY_LOGICAL_BYTES)
    max_logical_bytes: StrictInt = Field(ge=1, le=MAX_REMOTE_RECOVERY_LOGICAL_BYTES)
    max_decoded_work_bytes: StrictInt = Field(ge=1, le=MAX_REMOTE_RECOVERY_DECODED_BYTES)
    per_object_decoded_work_bytes: StrictInt = Field(ge=1)
    effective_per_object_decoded_work_bytes: StrictInt = Field(ge=1)
    coverage: Literal["selected_versions_only"]
    catalog_object_snapshot_atomic: Literal[False]
    source_authenticated: Literal[False]
    mutated_catalog_or_objects: Literal[False]
    guidance: str = Field(strict=True)

    @field_validator(
        "catalog_object_snapshot_atomic",
        "source_authenticated",
        "mutated_catalog_or_objects",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Recovery verification flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        versions = [item.model_dump(mode="json") for item in self.versions]
        identifiers = [item["version_id"] for item in versions]
        if identifiers != sorted(set(identifiers)) or len(versions) != self.selected_version_count:
            raise ValueError("Recovery selection identities or count are inconsistent")
        projection = [
            {key: value for key, value in item.items() if key not in {"status", "error_code"}}
            for item in versions
        ]
        if self.selection_hash != canonical_hash({"format": self.format, "versions": projection}):
            raise ValueError("Recovery selection hash is inconsistent")
        objects: dict[str, dict[str, Any]] = {}
        for item in versions:
            previous = objects.setdefault(item["object_key_hash"], item)
            if any(
                previous[field] != item[field]
                for field in ("shape", "dtype", "nbytes", "checksum", "status", "error_code")
            ):
                raise ValueError("Recovery versions disagree about a referenced object")
        if (
            self.distinct_object_count != len(objects)
            or self.distinct_logical_bytes != sum(item["nbytes"] for item in objects.values())
            or self.distinct_logical_bytes > self.max_logical_bytes
            or self.per_object_decoded_work_bytes
            != self.max_decoded_work_bytes // self.distinct_object_count
            or self.effective_per_object_decoded_work_bytes > self.per_object_decoded_work_bytes
            or (self.status == "all_selected_objects_verified")
            != all(item["status"] == "verified" for item in versions)
            or self.guidance != RECOVERY_GUIDANCE
        ):
            raise ValueError("Recovery verification totals or coverage are inconsistent")
        return self
