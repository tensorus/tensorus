"""Catalog-authoritative tenant request admission and operational counters."""
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_validator
from tensorus.models import Model
MAX_EVENTS = 1000
MAX_NUMBER = 2**63 - 1
MAX_RESPONSE_BYTES = 8192
class TenantHttpPolicy(Model):
    generation: StrictInt = Field(ge=1, le=MAX_EVENTS)
    requests_per_second: StrictInt | None = Field(ge=1, le=100_000)
    burst: StrictInt | None = Field(ge=1, le=1_000_000)
    configured_at: str

    @model_validator(mode="after")
    def paired_limits(self) -> Self:
        if (self.requests_per_second is None) != (self.burst is None):
            raise ValueError("Tenant HTTP admission limits must be paired")
        return self
class TenantHttpAdmission(Model):
    format: Literal["tensorus-tenant-http-admission-v1"]
    tenant_id: UUID
    policy: TenantHttpPolicy
    admitted: StrictInt = Field(ge=0, le=MAX_NUMBER)
    denied: StrictInt = Field(ge=0, le=MAX_NUMBER)
    coverage: Literal["authenticated_nonexempt_requests"]
    admin_emergency_endpoints_exempt: Literal[True]
    preauthentication_covered: Literal[False]
    resource_use_measured: Literal[False]
    billing_authority: Literal[False]
    other_tenants_disclosed: Literal[False]

    @field_validator(
        "admin_emergency_endpoints_exempt",
        "preauthentication_covered",
        "resource_use_measured",
        "billing_authority",
        "other_tenants_disclosed",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Tenant HTTP admission flags must be booleans")
        return value
class TenantHttpAdmissionCapabilities(Model):
    format: Literal["tensorus-tenant-http-admission-v1"]
    authority: Literal["active_unscoped_tenant_admin"]
    coverage: Literal["authenticated_nonexempt_requests"]
    max_response_bytes: Literal[8192]
    catalog_policy: Literal[True]
    admin_emergency_endpoints_exempt: Literal[True]
    preauthentication_covered: Literal[False]
    resource_use_measured: Literal[False]
    billing_authority: Literal[False]

    @field_validator("max_response_bytes", mode="before")
    @classmethod
    def exact_limit(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Tenant HTTP admission response limit must be an integer")
        return value

    @field_validator(
        "catalog_policy",
        "admin_emergency_endpoints_exempt",
        "preauthentication_covered",
        "resource_use_measured",
        "billing_authority",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Tenant HTTP admission flags must be booleans")
        return value
