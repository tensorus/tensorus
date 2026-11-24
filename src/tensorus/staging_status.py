"""Read-only catalog evidence for shared staging reservations and work-token policy."""
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictBool
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_validator
from tensorus.arrays import MAX_TENANT_STAGING_INPUT_BYTES
from tensorus.input_reservations import MAX_ACTIVE
from tensorus.input_reservations import MAX_NUMBER
from tensorus.models import Model
from tensorus.scientific_read_admission import TENANT_STAGING_STATUS_ROUTES
MAX_RESPONSE_BYTES = 8192
MAX_TOTAL_BYTES = MAX_NUMBER * MAX_ACTIVE
class TenantStagingReadAdmissionCapabilities(Model):
    format: Literal["tensorus-tenant-staging-read-admission-v1"] = (
        "tensorus-tenant-staging-read-admission-v1"
    )
    routes: list[str] = Field(default_factory=lambda: list(TENANT_STAGING_STATUS_ROUTES))
    after_authentication_and_unscoped_admin_check: Literal[True] = True
    shared_scientific_read_pool: Literal[True] = True
    single_api_process: Literal[True] = True
    active_until_response_and_handler_finish: Literal[True] = True
    caller_rechecked_after_serialization: Literal[True] = True
    response_bytes_bounded: Literal[True] = True
    response_delivery_atomic: Literal[False] = False
    fair_queue: Literal[False] = False
    dedicated_recovery_capacity: Literal[False] = False
    database_work_hard_bounded: Literal[False] = False
    physical_memory_reserved: Literal[False] = False

    @field_validator("routes")
    @classmethod
    def exact_routes(cls, value: list[str]) -> list[str]:
        if value != list(TENANT_STAGING_STATUS_ROUTES):
            raise ValueError("Tenant staging admission routes are inconsistent")
        return value

    @field_validator(
        "after_authentication_and_unscoped_admin_check",
        "shared_scientific_read_pool",
        "single_api_process",
        "active_until_response_and_handler_finish",
        "caller_rechecked_after_serialization",
        "response_bytes_bounded",
        "response_delivery_atomic",
        "fair_queue",
        "dedicated_recovery_capacity",
        "database_work_hard_bounded",
        "physical_memory_reserved",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Tenant staging admission flags must be booleans")
        return value
class TenantStagingCapabilities(Model):
    format: Literal["tensorus-tenant-staging-reservations-v1"]
    authority: Literal["active_unscoped_tenant_admin"]
    coverage: Literal["requesting_tenant_active_catalog_records"]
    max_active_reservations: StrictInt
    max_response_bytes: StrictInt
    catalog_snapshot: Literal[True]
    host_identity_checked: Literal[False]
    live_locks_checked: Literal[False]
    filesystem_capacity_checked: Literal[False]
    database_work_hard_bounded: Literal[False]
    other_tenant_usage_disclosed: Literal[False]
    mutates_catalog_or_objects: Literal[False]

    @field_validator(
        "catalog_snapshot",
        "host_identity_checked",
        "live_locks_checked",
        "filesystem_capacity_checked",
        "database_work_hard_bounded",
        "other_tenant_usage_disclosed",
        "mutates_catalog_or_objects",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Staging status flags must be booleans")
        return value

    @model_validator(mode="after")
    def supported_limits(self) -> Self:
        if (
            self.max_active_reservations != MAX_ACTIVE
            or self.max_response_bytes != MAX_RESPONSE_BYTES
        ):
            raise ValueError("Staging status limits do not match the advertised format")
        return self
class TenantWorkQuotaCapabilities(Model):
    format: Literal["tensorus-tenant-staging-work-quota-v1"]
    authority: Literal["active_unscoped_tenant_admin"]
    coverage: Literal["requesting_tenant_pooled_work_reservations"]
    max_active_reservations: StrictInt
    max_response_bytes: StrictInt
    uniform_per_tenant_limit: Literal[True]
    catalog_snapshot: Literal[True]
    unpooled_work_covered: Literal[False]
    host_identity_checked: Literal[False]
    live_locks_checked: Literal[False]
    filesystem_capacity_checked: Literal[False]
    database_work_hard_bounded: Literal[False]
    other_tenant_usage_disclosed: Literal[False]
    mutates_catalog_or_objects: Literal[False]

    @field_validator(
        "uniform_per_tenant_limit",
        "catalog_snapshot",
        "unpooled_work_covered",
        "host_identity_checked",
        "live_locks_checked",
        "filesystem_capacity_checked",
        "database_work_hard_bounded",
        "other_tenant_usage_disclosed",
        "mutates_catalog_or_objects",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Staging quota flags must be booleans")
        return value

    @model_validator(mode="after")
    def supported_limits(self) -> Self:
        if (
            self.max_active_reservations != MAX_ACTIVE
            or self.max_response_bytes != MAX_RESPONSE_BYTES
        ):
            raise ValueError("Staging work quota limits do not match the advertised format")
        return self
class TenantInputQuotaCapabilities(Model):
    format: Literal["tensorus-tenant-staging-input-quota-v1"] = (
        "tensorus-tenant-staging-input-quota-v1"
    )
    authority: Literal["active_unscoped_tenant_admin"] = "active_unscoped_tenant_admin"
    coverage: Literal["requesting_tenant_pooled_upload_inputs"] = (
        "requesting_tenant_pooled_upload_inputs"
    )
    max_active_reservations: Literal[1024] = 1024
    max_response_bytes: Literal[8192] = 8192
    uniform_per_tenant_limit: Literal[True] = True
    catalog_snapshot: Literal[True] = True
    rounded_to_filesystem_fragment: Literal[True] = True
    allocation_can_temporarily_exceed_reservation: Literal[True] = True
    unpooled_inputs_covered: Literal[False] = False
    host_identity_checked: Literal[False] = False
    live_locks_checked: Literal[False] = False
    filesystem_capacity_checked: Literal[False] = False
    database_work_hard_bounded: Literal[False] = False
    other_tenant_usage_disclosed: Literal[False] = False
    mutates_catalog_or_objects: Literal[False] = False

    @field_validator(
        "uniform_per_tenant_limit",
        "catalog_snapshot",
        "rounded_to_filesystem_fragment",
        "allocation_can_temporarily_exceed_reservation",
        "unpooled_inputs_covered",
        "host_identity_checked",
        "live_locks_checked",
        "filesystem_capacity_checked",
        "database_work_hard_bounded",
        "other_tenant_usage_disclosed",
        "mutates_catalog_or_objects",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Staging input quota flags must be booleans")
        return value

    @field_validator("max_active_reservations", "max_response_bytes", mode="before")
    @classmethod
    def strict_integer_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Staging input quota limits must be integers")
        return value

    @model_validator(mode="after")
    def supported_limits(self) -> Self:
        if (
            self.max_active_reservations != MAX_ACTIVE
            or self.max_response_bytes != MAX_RESPONSE_BYTES
        ):
            raise ValueError("Staging input quota limits do not match the advertised format")
        return self
class TenantReservationCounts(Model):
    active_count: StrictInt = Field(ge=0, le=MAX_ACTIVE)
    reserved_input_bytes: StrictInt = Field(ge=0, le=MAX_TOTAL_BYTES)
    reserved_work_bytes: StrictInt = Field(ge=0, le=MAX_TOTAL_BYTES)

    @model_validator(mode="after")
    def consistent_counts(self) -> Self:
        if (
            self.active_count == 0 and (self.reserved_input_bytes or self.reserved_work_bytes)
        ) or self.reserved_work_bytes < self.active_count:
            raise ValueError("Staging reservation counts are inconsistent")
        return self
class TenantStagingStatus(Model):
    format: Literal["tensorus-tenant-staging-reservations-v1"]
    tenant_id: UUID
    coverage: Literal["requesting_tenant_active_catalog_records"]
    pool_initialized: StrictBool
    input_capacity_bytes: StrictInt | None = Field(ge=1024, le=2**50)
    work_capacity_bytes: StrictInt | None = Field(ge=1024, le=2**50)
    catalog_upload_chunks: StrictBool
    catalog_worker_output_chunks: StrictBool
    catalog_eager_ingest_chunks: StrictBool
    configuration_matches_catalog: StrictBool
    uploads: TenantReservationCounts
    worker_outputs: TenantReservationCounts
    eager_ingests: TenantReservationCounts
    total: TenantReservationCounts
    host_identity_checked: Literal[False]
    live_locks_checked: Literal[False]
    filesystem_capacity_checked: Literal[False]
    database_work_hard_bounded: Literal[False]
    other_tenant_usage_disclosed: Literal[False]
    mutates_catalog_or_objects: Literal[False]

    @field_validator(
        "host_identity_checked",
        "live_locks_checked",
        "filesystem_capacity_checked",
        "database_work_hard_bounded",
        "other_tenant_usage_disclosed",
        "mutates_catalog_or_objects",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Staging status flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        groups = (self.uploads, self.worker_outputs, self.eager_ingests)
        if (
            self.total.active_count != sum(group.active_count for group in groups)
            or self.total.reserved_input_bytes
            != sum(group.reserved_input_bytes for group in groups)
            or self.total.reserved_work_bytes != sum(group.reserved_work_bytes for group in groups)
            or self.uploads.reserved_input_bytes < self.uploads.active_count
            or self.worker_outputs.reserved_input_bytes != 0
            or self.eager_ingests.reserved_input_bytes != 0
            or (self.worker_outputs.active_count > 0 and not self.catalog_worker_output_chunks)
            or (self.eager_ingests.active_count > 0 and not self.catalog_eager_ingest_chunks)
            or (
                (self.catalog_worker_output_chunks or self.catalog_eager_ingest_chunks)
                and not self.catalog_upload_chunks
            )
        ):
            raise ValueError("Staging reservation totals or policies are inconsistent")
        capacities_present = self.input_capacity_bytes is not None and (
            self.work_capacity_bytes is not None
        )
        if (
            (self.input_capacity_bytes is None) != (self.work_capacity_bytes is None)
            or self.pool_initialized != capacities_present
            or (
                not self.pool_initialized
                and (
                    self.total.active_count
                    or self.catalog_upload_chunks
                    or self.catalog_worker_output_chunks
                    or self.catalog_eager_ingest_chunks
                )
            )
        ):
            raise ValueError("Staging pool initialization evidence is inconsistent")
        return self
class TenantWorkQuotaStatus(Model):
    format: Literal["tensorus-tenant-staging-work-quota-v1"]
    tenant_id: UUID
    coverage: Literal["requesting_tenant_pooled_work_reservations"]
    pool_initialized: StrictBool
    quota_enabled: StrictBool
    policy_generation: StrictInt | None = Field(ge=1, le=MAX_NUMBER)
    limit_bytes: StrictInt | None = Field(ge=1024, le=2**50)
    reserved_work_bytes: StrictInt = Field(ge=0, le=MAX_TOTAL_BYTES)
    remaining_work_bytes: StrictInt | None = Field(ge=0, le=2**50)
    over_quota_work_bytes: StrictInt | None = Field(ge=0, le=MAX_TOTAL_BYTES)
    configuration_matches_catalog: StrictBool
    unpooled_work_covered: Literal[False]
    host_identity_checked: Literal[False]
    live_locks_checked: Literal[False]
    filesystem_capacity_checked: Literal[False]
    database_work_hard_bounded: Literal[False]
    other_tenant_usage_disclosed: Literal[False]
    mutates_catalog_or_objects: Literal[False]

    @field_validator(
        "unpooled_work_covered",
        "host_identity_checked",
        "live_locks_checked",
        "filesystem_capacity_checked",
        "database_work_hard_bounded",
        "other_tenant_usage_disclosed",
        "mutates_catalog_or_objects",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Staging quota flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        if (
            self.pool_initialized != (self.policy_generation is not None)
            or self.quota_enabled != (self.limit_bytes is not None)
            or (self.quota_enabled and not self.pool_initialized)
            or (not self.pool_initialized and self.reserved_work_bytes != 0)
        ):
            raise ValueError("Staging work quota policy evidence is inconsistent")
        if self.limit_bytes is None:
            if self.remaining_work_bytes is not None or self.over_quota_work_bytes is not None:
                raise ValueError("Disabled staging work quota cannot report quota balances")
        elif self.remaining_work_bytes != max(
            0, self.limit_bytes - self.reserved_work_bytes
        ) or self.over_quota_work_bytes != max(0, self.reserved_work_bytes - self.limit_bytes):
            raise ValueError("Staging work quota totals are inconsistent")
        return self
class TenantInputQuotaStatus(Model):
    format: Literal["tensorus-tenant-staging-input-quota-v1"]
    tenant_id: UUID
    coverage: Literal["requesting_tenant_pooled_upload_inputs"]
    pool_initialized: StrictBool
    quota_enabled: StrictBool
    policy_generation: StrictInt | None = Field(ge=1, le=MAX_NUMBER)
    limit_bytes: StrictInt | None = Field(ge=1024, le=MAX_TENANT_STAGING_INPUT_BYTES)
    reserved_input_bytes: StrictInt = Field(ge=0, le=MAX_TOTAL_BYTES)
    remaining_input_bytes: StrictInt | None = Field(ge=0, le=MAX_TENANT_STAGING_INPUT_BYTES)
    over_quota_input_bytes: StrictInt | None = Field(ge=0, le=MAX_TOTAL_BYTES)
    configuration_matches_catalog: StrictBool
    unpooled_inputs_covered: Literal[False]
    host_identity_checked: Literal[False]
    live_locks_checked: Literal[False]
    filesystem_capacity_checked: Literal[False]
    database_work_hard_bounded: Literal[False]
    other_tenant_usage_disclosed: Literal[False]
    mutates_catalog_or_objects: Literal[False]

    @field_validator(
        "unpooled_inputs_covered",
        "host_identity_checked",
        "live_locks_checked",
        "filesystem_capacity_checked",
        "database_work_hard_bounded",
        "other_tenant_usage_disclosed",
        "mutates_catalog_or_objects",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Staging input quota flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        if (
            self.pool_initialized != (self.policy_generation is not None)
            or self.quota_enabled != (self.limit_bytes is not None)
            or (self.quota_enabled and not self.pool_initialized)
            or (not self.pool_initialized and self.reserved_input_bytes != 0)
        ):
            raise ValueError("Staging input quota policy evidence is inconsistent")
        if self.limit_bytes is None:
            if self.remaining_input_bytes is not None or self.over_quota_input_bytes is not None:
                raise ValueError("Disabled staging input quota cannot report quota balances")
        elif self.remaining_input_bytes != max(
            0, self.limit_bytes - self.reserved_input_bytes
        ) or self.over_quota_input_bytes != max(0, self.reserved_input_bytes - self.limit_bytes):
            raise ValueError("Staging input quota totals are inconsistent")
        return self
