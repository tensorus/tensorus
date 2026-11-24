"""Short-lived, receipt-only observation of a committed workspace detachment."""
from typing import Annotated
from typing import Literal
from uuid import UUID
from pydantic import Field
from pydantic import StrictInt
from pydantic import field_validator
from tensorus.control_request_limits import MAX_CONTROL_REQUEST_BYTES
from tensorus.models import Model
from tensorus.organization_detachment import DetachmentReceipt
from tensorus.receipt_secrets import validate_receipt_secret as validate_client_receipt_secret
FORMAT: Literal["tensorus-organization-detachment-receipt-v1"] = (
    "tensorus-organization-detachment-receipt-v1"
)
MAX_LIFETIME_SECONDS: Literal[300] = 300
MAX_RESPONSE_BYTES: Literal[16384] = 16_384
class DetachmentReceiptCapabilities(Model):
    format: Literal["tensorus-organization-detachment-receipt-v1"] = FORMAT
    max_request_bytes: Annotated[StrictInt, Field(ge=1, le=MAX_CONTROL_REQUEST_BYTES)]
    max_response_bytes: Literal[16384] = MAX_RESPONSE_BYTES
    max_lifetime_seconds: Literal[300] = MAX_LIFETIME_SECONDS
    client_held_secret: Literal[True] = True
    current_detachment_required: Literal[True] = True
    original_session_expiry_enforced: Literal[True] = True
    workspace_and_owner_authority_rechecked: Literal[True] = True
    receipt_only: Literal[True] = True
    detachment_reexecuted: Literal[False] = False
    grants_other_access: Literal[False] = False
    missing_receipt_proves_no_commit: Literal[False] = False
    restored_authority_reconciled: Literal[False] = False

    @field_validator(
        "max_request_bytes", "max_response_bytes", "max_lifetime_seconds", mode="before"
    )
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Detachment receipt limits must be integers")
        return value

    @field_validator(
        "client_held_secret",
        "current_detachment_required",
        "original_session_expiry_enforced",
        "workspace_and_owner_authority_rechecked",
        "receipt_only",
        "detachment_reexecuted",
        "grants_other_access",
        "missing_receipt_proves_no_commit",
        "restored_authority_reconciled",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Detachment receipt flags must be booleans")
        return value
class RecoveredDetachmentReceipt(Model):
    format: Literal["tensorus-organization-detachment-receipt-v1"] = FORMAT
    receipt_id: UUID
    expires_at: Annotated[StrictInt, Field(ge=1, le=2**53 - 1)]
    actor_session_id: UUID
    actor_membership_id: UUID
    receipt: DetachmentReceipt
def validate_receipt_secret(secret: str) -> str:
    return validate_client_receipt_secret(secret, "tor_", "organization detachment")
