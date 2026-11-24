"""Bounded catalog inventory and stateless selected-object recovery batches."""
import re
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import AfterValidator
from pydantic import Field
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_validator
from tensorus.arrays import Record
from tensorus.arrays import canonical_hash
from tensorus.arrays import canonical_json
from tensorus.errors import LimitExceeded
from tensorus.middleware import MAX_RECOVERY_REQUEST_BYTES
from tensorus.models import Model
from tensorus.recovery_verification import MAX_REMOTE_RECOVERY_DECODED_BYTES
from tensorus.recovery_verification import MAX_REMOTE_RECOVERY_LOGICAL_BYTES
from tensorus.recovery_verification import MAX_REMOTE_RECOVERY_VERSIONS
from tensorus.recovery_verification import RecoveryHash
from tensorus.recovery_verification import RecoveryObjectDescriptor
from tensorus.recovery_verification import RecoveryVerificationReport
RECOVERY_PARTITION_FORMAT = "tensorus-recovery-inventory-partition-v1"
MAX_INVENTORY_VERSIONS = 512
MAX_INVENTORY_RESPONSE_BYTES = 512 * 1024
MAX_INVENTORY_BATCH_RESPONSE_BYTES = 160 * 1024
MAX_INVENTORY_PROGRESS_RECEIPTS = 1024
MAX_INVENTORY_PROGRESS_BYTES = 4 * 1024 * 1024
def _validate_partition_prefix(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{1,8}", value) is None:
        raise ValueError("Recovery partition prefix must contain 1 to 8 lowercase hex digits")
    return value
PartitionPrefix = Annotated[
    str,
    Field(strict=True, min_length=1, max_length=8, pattern=r"^[0-9a-f]+$"),
    AfterValidator(_validate_partition_prefix),
]
class _RecoveryInventoryCapabilities(Model):
    max_versions: StrictInt
    max_batch_versions: StrictInt
    max_logical_bytes: StrictInt
    max_decoded_work_bytes: StrictInt
    max_request_bytes: StrictInt
    max_inventory_response_bytes: StrictInt
    max_batch_response_bytes: StrictInt
    authority: Literal["active_unscoped_tenant_admin"]
    consistent_catalog_snapshot: Literal[True]
    catalog_object_snapshot_atomic: Literal[False]
    source_authenticated: Literal[False]
    restored_authority_reconciled: Literal[False]
    earlier_batches_verified: Literal[False]
    database_work_hard_bounded: Literal[False]
    hard_memory_limit: Literal[False]
    hard_wall_clock_limit: Literal[False]

    @field_validator(
        "consistent_catalog_snapshot",
        "catalog_object_snapshot_atomic",
        "source_authenticated",
        "restored_authority_reconciled",
        "earlier_batches_verified",
        "database_work_hard_bounded",
        "hard_memory_limit",
        "hard_wall_clock_limit",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Recovery inventory flags must be booleans")
        return value

    @model_validator(mode="after")
    def supported_limits(self) -> Self:
        if (
            self.max_versions != MAX_INVENTORY_VERSIONS
            or self.max_batch_versions != MAX_REMOTE_RECOVERY_VERSIONS
            or self.max_logical_bytes != MAX_REMOTE_RECOVERY_LOGICAL_BYTES
            or self.max_decoded_work_bytes != MAX_REMOTE_RECOVERY_DECODED_BYTES
            or self.max_request_bytes != MAX_RECOVERY_REQUEST_BYTES
            or self.max_inventory_response_bytes != MAX_INVENTORY_RESPONSE_BYTES
            or self.max_batch_response_bytes != MAX_INVENTORY_BATCH_RESPONSE_BYTES
        ):
            raise ValueError("Recovery inventory limits do not match the advertised format")
        return self
class RecoveryInventoryCapabilities(_RecoveryInventoryCapabilities):
    format: Literal["tensorus-recovery-inventory-v1"]
    batch_format: Literal["tensorus-recovery-inventory-batch-v1"]
    coverage: Literal["retained_tenant_version_references"]
class RecoveryPartitionCapabilities(_RecoveryInventoryCapabilities):
    format: Literal["tensorus-recovery-inventory-partition-v1"]
    batch_format: Literal["tensorus-recovery-inventory-partition-batch-v1"]
    coverage: Literal["retained_tenant_version_references_matching_uuid_prefix"]
    max_prefix_chars: Literal[8]
    whole_tenant_snapshot: Literal[False]
    partitions_share_snapshot: Literal[False]

    @field_validator("max_prefix_chars", mode="before")
    @classmethod
    def strict_prefix_limit(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Recovery partition prefix limit must be an integer")
        return value

    @field_validator("whole_tenant_snapshot", "partitions_share_snapshot", mode="before")
    @classmethod
    def strict_partition_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Recovery partition flags must be booleans")
        return value
class _RecoveryInventoryEvidence(Model):
    tenant_id: UUID
    inventory_hash: RecoveryHash
    versions: list[RecoveryObjectDescriptor] = Field(max_length=MAX_INVENTORY_VERSIONS)
    version_count: StrictInt = Field(ge=0, le=MAX_INVENTORY_VERSIONS)
    distinct_object_count: StrictInt = Field(ge=0, le=MAX_INVENTORY_VERSIONS)
    distinct_logical_bytes: StrictInt = Field(ge=0)
    consistent_catalog_snapshot: Literal[True]
    object_payloads_verified: Literal[False]
    catalog_object_snapshot_atomic: Literal[False]
    source_authenticated: Literal[False]
    restored_authority_reconciled: Literal[False]
    mutated_catalog_or_objects: Literal[False]

    @field_validator(
        "consistent_catalog_snapshot",
        "object_payloads_verified",
        "catalog_object_snapshot_atomic",
        "source_authenticated",
        "restored_authority_reconciled",
        "mutated_catalog_or_objects",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Recovery inventory flags must be booleans")
        return value

    def _check_contents(self, expected_hash: str) -> None:
        versions = [item.model_dump(mode="json") for item in self.versions]
        identifiers = [item["version_id"] for item in versions]
        if (
            identifiers != sorted(set(identifiers))
            or len(versions) != self.version_count
            or any(item["tenant_id"] != str(self.tenant_id) for item in versions)
            or self.inventory_hash != expected_hash
        ):
            raise ValueError("Recovery inventory identities or hash are inconsistent")
        objects: dict[str, dict[str, Any]] = {}
        for item in versions:
            previous = objects.setdefault(item["object_key_hash"], item)
            if any(
                previous[field] != item[field] for field in ("shape", "dtype", "nbytes", "checksum")
            ):
                raise ValueError("Recovery versions disagree about a referenced object")
        if self.distinct_object_count != len(objects) or self.distinct_logical_bytes != sum(
            item["nbytes"] for item in objects.values()
        ):
            raise ValueError("Recovery inventory totals are inconsistent")
class RecoveryInventory(_RecoveryInventoryEvidence):
    format: Literal["tensorus-recovery-inventory-v1"]
    coverage: Literal["retained_tenant_version_references"]

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        versions = [item.model_dump(mode="json") for item in self.versions]
        self._check_contents(
            canonical_hash(
                {"format": self.format, "tenant_id": str(self.tenant_id), "versions": versions}
            )
        )
        return self
class RecoveryPartitionInventory(_RecoveryInventoryEvidence):
    format: Literal["tensorus-recovery-inventory-partition-v1"]
    coverage: Literal["retained_tenant_version_references_matching_uuid_prefix"]
    partition_prefix: PartitionPrefix
    whole_tenant_snapshot: Literal[False]
    partitions_share_snapshot: Literal[False]

    @field_validator("whole_tenant_snapshot", "partitions_share_snapshot", mode="before")
    @classmethod
    def strict_partition_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Recovery partition flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        versions = [item.model_dump(mode="json") for item in self.versions]
        if any(not item["version_id"].startswith(self.partition_prefix) for item in versions):
            raise ValueError("Recovery version is outside its UUID prefix")
        self._check_contents(
            canonical_hash(
                {
                    "format": self.format,
                    "tenant_id": str(self.tenant_id),
                    "partition_prefix": self.partition_prefix,
                    "versions": versions,
                }
            )
        )
        return self
class RecoveryInventoryBatchRequest(Model):
    expected_inventory_hash: RecoveryHash
    start_index: StrictInt = Field(ge=0, lt=MAX_INVENTORY_VERSIONS)
    limit: StrictInt = Field(default=1, ge=1, le=MAX_REMOTE_RECOVERY_VERSIONS)
    max_logical_bytes: StrictInt = Field(
        default=MAX_REMOTE_RECOVERY_LOGICAL_BYTES, ge=1, le=MAX_REMOTE_RECOVERY_LOGICAL_BYTES
    )
    max_decoded_work_bytes: StrictInt = Field(
        default=MAX_REMOTE_RECOVERY_DECODED_BYTES, ge=1, le=MAX_REMOTE_RECOVERY_DECODED_BYTES
    )
class RecoveryPartitionBatchRequest(RecoveryInventoryBatchRequest):
    partition_prefix: PartitionPrefix
class _RecoveryInventoryBatchEvidence(Model):
    tenant_id: UUID
    inventory_hash: RecoveryHash
    start_index: StrictInt = Field(ge=0, lt=MAX_INVENTORY_VERSIONS)
    end_index: StrictInt = Field(ge=1, le=MAX_INVENTORY_VERSIONS)
    total_versions: StrictInt = Field(ge=1, le=MAX_INVENTORY_VERSIONS)
    next_index: StrictInt | None = Field(ge=0, lt=MAX_INVENTORY_VERSIONS)
    verification: RecoveryVerificationReport
    earlier_batches_verified: Literal[False]
    catalog_object_snapshot_atomic: Literal[False]
    source_authenticated: Literal[False]
    restored_authority_reconciled: Literal[False]
    mutated_catalog_or_objects: Literal[False]

    @field_validator(
        "earlier_batches_verified",
        "catalog_object_snapshot_atomic",
        "source_authenticated",
        "restored_authority_reconciled",
        "mutated_catalog_or_objects",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Recovery batch flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        expected_next = (
            self.start_index
            if self.verification.status == "incomplete"
            else self.end_index
            if self.end_index < self.total_versions
            else None
        )
        if (
            not self.start_index < self.end_index <= self.total_versions
            or self.end_index - self.start_index != self.verification.selected_version_count
            or any(item.tenant_id != self.tenant_id for item in self.verification.versions)
            or self.next_index != expected_next
        ):
            raise ValueError("Recovery batch progress is inconsistent")
        return self
class RecoveryInventoryBatchReport(_RecoveryInventoryBatchEvidence):
    format: Literal["tensorus-recovery-inventory-batch-v1"]
class RecoveryPartitionBatchReport(_RecoveryInventoryBatchEvidence):
    format: Literal["tensorus-recovery-inventory-partition-batch-v1"]
    partition_prefix: PartitionPrefix
    whole_tenant_snapshot: Literal[False]
    partitions_share_snapshot: Literal[False]

    @field_validator("whole_tenant_snapshot", "partitions_share_snapshot", mode="before")
    @classmethod
    def strict_partition_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Recovery partition flags must be booleans")
        return value
def validate_batch_inventory(
    inventory: RecoveryInventory | RecoveryPartitionInventory,
    batch: RecoveryInventoryBatchReport | RecoveryPartitionBatchReport,
) -> None:
    """Compare one batch to a locally retained, checked inventory."""
    expected = [
        item.model_dump(mode="json")
        for item in inventory.versions[batch.start_index : batch.end_index]
    ]
    actual = [
        item.model_dump(mode="json", exclude={"status", "error_code"})
        for item in batch.verification.versions
    ]
    if (
        batch.inventory_hash != inventory.inventory_hash
        or batch.tenant_id != inventory.tenant_id
        or batch.total_versions != inventory.version_count
        or canonical_json(actual) != canonical_json(expected)
        or isinstance(inventory, RecoveryPartitionInventory)
        != isinstance(batch, RecoveryPartitionBatchReport)
        or (
            isinstance(inventory, RecoveryPartitionInventory)
            and isinstance(batch, RecoveryPartitionBatchReport)
            and batch.partition_prefix != inventory.partition_prefix
        )
    ):
        raise ValueError("Recovery batch does not match the retained inventory")
def review_recovery_inventory_progress(inventory: Record, receipts: list[Record]) -> Record:
    """Review retained, unauthenticated observations without reading current storage."""
    if not isinstance(inventory, dict):
        raise ValueError("Recovery inventory evidence must be an object")
    inventory_json = canonical_json(inventory)
    if len(inventory_json.encode()) > MAX_INVENTORY_RESPONSE_BYTES:
        raise LimitExceeded("Recovery inventory evidence exceeds its byte limit")
    if len(receipts) > MAX_INVENTORY_PROGRESS_RECEIPTS:
        raise LimitExceeded("Too many recovery batch receipts")
    partitioned = inventory.get("format") == RECOVERY_PARTITION_FORMAT
    checked_inventory = (
        RecoveryPartitionInventory.model_validate_json(inventory_json, strict=True)
        if partitioned
        else RecoveryInventory.model_validate_json(inventory_json, strict=True)
    )
    if canonical_json(checked_inventory.model_dump(mode="json")) != inventory_json:
        raise ValueError("Recovery inventory evidence is not canonical")
    cursor = 0
    incomplete = 0
    last_incomplete = False
    total_bytes = len(inventory_json.encode())
    for receipt in receipts:
        encoded = canonical_json(receipt)
        size = len(encoded.encode())
        total_bytes += size
        if size > MAX_INVENTORY_BATCH_RESPONSE_BYTES or total_bytes > MAX_INVENTORY_PROGRESS_BYTES:
            raise LimitExceeded("Recovery progress evidence exceeds its byte limit")
        batch = (
            RecoveryPartitionBatchReport.model_validate_json(encoded, strict=True)
            if partitioned
            else RecoveryInventoryBatchReport.model_validate_json(encoded, strict=True)
        )
        if canonical_json(batch.model_dump(mode="json")) != encoded:
            raise ValueError("Recovery batch evidence is not canonical")
        validate_batch_inventory(checked_inventory, batch)
        if batch.start_index != cursor:
            raise ValueError("Recovery batches are not contiguous from the beginning")
        last_incomplete = batch.verification.status == "incomplete"
        if last_incomplete:
            incomplete += 1
        else:
            cursor = batch.end_index
    total = checked_inventory.version_count
    if total == 0:
        status = "empty_inventory"
    elif cursor == total:
        status = "all_batches_observed"
    elif last_incomplete:
        status = "retry_required"
    else:
        status = "partial"
    result: Record = {
        "format": (
            "tensorus-recovery-inventory-partition-progress-v1"
            if partitioned
            else "tensorus-recovery-inventory-progress-v1"
        ),
        "tenant_id": str(checked_inventory.tenant_id),
        "inventory_hash": checked_inventory.inventory_hash,
        "status": status,
        "observed_verified_versions": cursor,
        "total_versions": total,
        "incomplete_batch_observations": incomplete,
        "next_index": cursor if cursor < total else None,
        "receipt_authentication": False,
        "current_objects_rechecked": False,
        "catalog_object_snapshot_atomic": False,
        "restored_authority_reconciled": False,
    }
    if isinstance(checked_inventory, RecoveryPartitionInventory):
        result.update(
            partition_prefix=checked_inventory.partition_prefix,
            whole_tenant_snapshot=False,
            partitions_share_snapshot=False,
        )
    return result
