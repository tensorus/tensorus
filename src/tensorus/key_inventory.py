"""Bounded administrator views over the existing credential catalog."""
from typing import Literal
from pydantic import Field
from pydantic import field_validator
from tensorus.models import Model
from tensorus.scientific_read_admission import CREDENTIAL_INVENTORY_ROUTES
FORMAT: Literal["tensorus-credential-page-v1"] = "tensorus-credential-page-v1"
MAX_PAGE_SIZE: Literal[100] = 100
MAX_RESPONSE_BYTES: Literal[1048576] = 1_048_576
MAX_LEGACY_ITEMS: Literal[100] = 100
class CredentialPageCapabilities(Model):
    format: Literal["tensorus-credential-page-v1"] = FORMAT
    max_page_size: Literal[100] = MAX_PAGE_SIZE
    max_response_bytes: Literal[1048576] = MAX_RESPONSE_BYTES
    active_unscoped_administrator_required: Literal[True] = True
    legacy_inventory_unchanged: Literal[True] = True
    expired_membership_sessions_excluded: Literal[True] = True
    cross_page_snapshot: Literal[False] = False
    database_work_hard_bounded: Literal[False] = False
    physical_memory_reserved: Literal[False] = False

    @field_validator("max_page_size", "max_response_bytes", mode="before")
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Credential page limits must be integers")
        return value

    @field_validator(
        "active_unscoped_administrator_required",
        "legacy_inventory_unchanged",
        "expired_membership_sessions_excluded",
        "cross_page_snapshot",
        "database_work_hard_bounded",
        "physical_memory_reserved",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Credential page flags must be booleans")
        return value
class CredentialPageV2Capabilities(Model):
    format: Literal["tensorus-credential-page-capabilities-v2"] = (
        "tensorus-credential-page-capabilities-v2"
    )
    page_format: Literal["tensorus-credential-page-v1"] = FORMAT
    max_page_size: Literal[100] = MAX_PAGE_SIZE
    max_response_bytes: Literal[1048576] = MAX_RESPONSE_BYTES
    legacy_inventory_max_items: Literal[100] = MAX_LEGACY_ITEMS
    legacy_inventory_max_response_bytes: Literal[1048576] = MAX_RESPONSE_BYTES
    active_unscoped_administrator_required: Literal[True] = True
    legacy_inventory_unchanged: Literal[False] = False
    expired_membership_sessions_excluded: Literal[True] = True
    cross_page_snapshot: Literal[False] = False
    database_work_hard_bounded: Literal[False] = False
    physical_memory_reserved: Literal[False] = False

    @field_validator(
        "max_page_size",
        "max_response_bytes",
        "legacy_inventory_max_items",
        "legacy_inventory_max_response_bytes",
        mode="before",
    )
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Credential page limits must be integers")
        return value

    @field_validator(
        "active_unscoped_administrator_required",
        "legacy_inventory_unchanged",
        "expired_membership_sessions_excluded",
        "cross_page_snapshot",
        "database_work_hard_bounded",
        "physical_memory_reserved",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Credential page flags must be booleans")
        return value
class CredentialInventoryAdmissionCapabilities(Model):
    format: Literal["tensorus-credential-inventory-admission-v1"] = (
        "tensorus-credential-inventory-admission-v1"
    )
    routes: list[str] = Field(default_factory=lambda: list(CREDENTIAL_INVENTORY_ROUTES))
    after_authentication_and_admin_check: Literal[True] = True
    shared_scientific_read_pool: Literal[True] = True
    single_api_process: Literal[True] = True
    active_until_response_and_handler_finish: Literal[True] = True
    revocation_admitted: Literal[False] = False
    legacy_inventory_bounded: Literal[False] = False
    fair_queue: Literal[False] = False
    database_work_hard_bounded: Literal[False] = False
    physical_memory_reserved: Literal[False] = False

    @field_validator("routes")
    @classmethod
    def exact_routes(cls, value: list[str]) -> list[str]:
        if value != list(CREDENTIAL_INVENTORY_ROUTES):
            raise ValueError("Credential inventory admission routes are inconsistent")
        return value

    @field_validator(
        "after_authentication_and_admin_check",
        "shared_scientific_read_pool",
        "single_api_process",
        "active_until_response_and_handler_finish",
        "revocation_admitted",
        "legacy_inventory_bounded",
        "fair_queue",
        "database_work_hard_bounded",
        "physical_memory_reserved",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Credential inventory admission flags must be booleans")
        return value
class CredentialInventoryAdmissionV2Capabilities(Model):
    format: Literal["tensorus-credential-inventory-admission-v2"] = (
        "tensorus-credential-inventory-admission-v2"
    )
    routes: list[str] = Field(default_factory=lambda: list(CREDENTIAL_INVENTORY_ROUTES))
    after_authentication_and_admin_check: Literal[True] = True
    shared_scientific_read_pool: Literal[True] = True
    single_api_process: Literal[True] = True
    active_until_response_and_handler_finish: Literal[True] = True
    revocation_admitted: Literal[False] = False
    legacy_inventory_bounded: Literal[True] = True
    fair_queue: Literal[False] = False
    database_work_hard_bounded: Literal[False] = False
    physical_memory_reserved: Literal[False] = False

    @field_validator("routes")
    @classmethod
    def exact_routes(cls, value: list[str]) -> list[str]:
        if value != list(CREDENTIAL_INVENTORY_ROUTES):
            raise ValueError("Credential inventory admission routes are inconsistent")
        return value

    @field_validator(
        "after_authentication_and_admin_check",
        "shared_scientific_read_pool",
        "single_api_process",
        "active_until_response_and_handler_finish",
        "revocation_admitted",
        "legacy_inventory_bounded",
        "fair_queue",
        "database_work_hard_bounded",
        "physical_memory_reserved",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Credential inventory admission flags must be booleans")
        return value
