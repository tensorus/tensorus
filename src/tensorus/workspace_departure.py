"""A membership session can relinquish its own workspace access."""
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_validator
from tensorus.control_request_limits import MAX_CONTROL_REQUEST_BYTES
from tensorus.models import Model
from tensorus.models import Role
from tensorus.models import WithdrawalReason
from tensorus.organizations import _timestamp
from tensorus.receipt_secrets import validate_receipt_secret as validate_client_receipt_secret
FORMAT: Literal["tensorus-workspace-departure-v1"] = "tensorus-workspace-departure-v1"
RECEIPT_FORMAT: Literal["tensorus-workspace-departure-receipt-v2"] = (
    "tensorus-workspace-departure-receipt-v2"
)
RECEIPT_LIFETIME_SECONDS: Literal[300] = 300
MAX_SESSIONS: Literal[10000] = 10_000
MAX_GRANTS: Literal[20000] = 20_000
MAX_RESPONSE_BYTES: Literal[8192] = 8192
MAX_GENERATION = 2**63 - 1
class DepartureCapabilities(Model):
    format: Literal["tensorus-workspace-departure-v1"] = FORMAT
    max_request_bytes: Annotated[StrictInt, Field(ge=1, le=MAX_CONTROL_REQUEST_BYTES)]
    max_response_bytes: Literal[8192] = MAX_RESPONSE_BYTES
    max_unrevoked_sessions: Literal[10000] = MAX_SESSIONS
    max_issued_grants: Literal[20000] = MAX_GRANTS
    current_membership_session_only: Literal[True] = True
    all_membership_sessions_revoked: Literal[True] = True
    issued_agent_grants_revoked: Literal[True] = True
    last_owners_protected: Literal[True] = True
    same_credential_retry_supported: Literal[False] = False
    provider_session_revoked: Literal[False] = False
    ordinary_keys_revoked: Literal[False] = False
    running_processes_stopped: Literal[False] = False
    exported_copies_revoked: Literal[False] = False
    database_time_or_memory_bounded: Literal[False] = False

    @field_validator(
        "max_request_bytes",
        "max_response_bytes",
        "max_unrevoked_sessions",
        "max_issued_grants",
        mode="before",
    )
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Workspace departure limits must be integers")
        return value

    @field_validator(
        "current_membership_session_only",
        "all_membership_sessions_revoked",
        "issued_agent_grants_revoked",
        "last_owners_protected",
        "same_credential_retry_supported",
        "provider_session_revoked",
        "ordinary_keys_revoked",
        "running_processes_stopped",
        "exported_copies_revoked",
        "database_time_or_memory_bounded",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Workspace departure flags must be booleans")
        return value
def _trimmed_reason(value: str) -> str:
    if value != value.strip():
        raise ValueError("A trimmed reason is required")
    return value
class DepartureRequest(Model):
    request_id: UUID
    membership_id: UUID
    session_id: UUID
    reason: WithdrawalReason
    confirmation: Literal["LEAVE"]

    _reason = field_validator("reason")(_trimmed_reason)
class DepartureReceipt(Model):
    format: Literal["tensorus-workspace-departure-v1"] = FORMAT
    tenant_id: UUID
    membership_id: UUID
    session_id: UUID
    request_id: UUID
    role: Role
    previous_generation: Annotated[str, Field(strict=True, pattern=r"^[1-9][0-9]{0,18}$")]
    generation: Annotated[str, Field(strict=True, pattern=r"^[1-9][0-9]{0,18}$")]
    removed_at: str
    reason: WithdrawalReason
    sessions_revoked: Annotated[StrictInt, Field(ge=1, le=MAX_SESSIONS)]
    grants_revoked: Annotated[StrictInt, Field(ge=0, le=MAX_GRANTS)]

    _removed_at = field_validator("removed_at")(_timestamp)
    _reason = field_validator("reason")(_trimmed_reason)

    @model_validator(mode="after")
    def consistent_generation(self) -> Self:
        if (
            int(self.previous_generation) >= MAX_GENERATION
            or int(self.generation) != int(self.previous_generation) + 1
        ):
            raise ValueError("Workspace departure generation is invalid")
        return self
class DepartureReceiptRecoveryCapabilities(Model):
    format: Literal["tensorus-workspace-departure-receipt-v2"] = RECEIPT_FORMAT
    max_request_bytes: Annotated[StrictInt, Field(ge=1, le=MAX_CONTROL_REQUEST_BYTES)]
    max_response_bytes: Literal[8192] = MAX_RESPONSE_BYTES
    max_lifetime_seconds: Literal[300] = RECEIPT_LIFETIME_SECONDS
    client_held_secret: Literal[True] = True
    retained_departure_required: Literal[True] = True
    original_session_expiry_enforced: Literal[True] = True
    current_removed_membership_required: Literal[True] = True
    receipt_only: Literal[True] = True
    departure_reexecuted: Literal[False] = False
    grants_other_access: Literal[False] = False
    missing_receipt_proves_no_commit: Literal[False] = False

    @field_validator(
        "max_request_bytes", "max_response_bytes", "max_lifetime_seconds", mode="before"
    )
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Departure receipt limits must be integers")
        return value

    @field_validator(
        "client_held_secret",
        "retained_departure_required",
        "original_session_expiry_enforced",
        "current_removed_membership_required",
        "receipt_only",
        "departure_reexecuted",
        "grants_other_access",
        "missing_receipt_proves_no_commit",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Departure receipt flags must be booleans")
        return value
class RecoveredDepartureReceipt(Model):
    format: Literal["tensorus-workspace-departure-receipt-v2"] = RECEIPT_FORMAT
    receipt_id: UUID
    expires_at: Annotated[StrictInt, Field(ge=1, le=2**53 - 1)]
    receipt: DepartureReceipt
def validate_receipt_secret(secret: str) -> str:
    return validate_client_receipt_secret(secret, "tdr_", "departure")
