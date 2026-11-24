"""Catalog-backed, single-workspace provisioning invitations."""
import re
import secrets
from datetime import datetime
from datetime import timedelta
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import SecretStr
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_validator
from tensorus.credentials import valid_generated_token
from tensorus.models import Model
from tensorus.models import Name
from tensorus.workspace_provisioning import initial_key_id
FORMAT: Literal["tensorus-workspace-provisioning-invitation-v1"] = (
    "tensorus-workspace-provisioning-invitation-v1"
)
REVOCATION_FORMAT: Literal["tensorus-workspace-invitation-revocation-v1"] = (
    "tensorus-workspace-invitation-revocation-v1"
)
MAX_RESPONSE_BYTES = 16 * 1024
Epoch = Annotated[StrictInt, Field(ge=1, le=2**53 - 1)]
Reason = Annotated[str, Field(strict=True, min_length=1, max_length=512)]
def new_invitation_token() -> str:
    return f"tni_{secrets.token_urlsafe(32)}"
def valid_invitation_token(token: str) -> bool:
    return re.fullmatch(r"tni_[A-Za-z0-9_-]{42}[AEIMQUYcgkosw048]", token) is not None
def _reason(value: str) -> str:
    if value != value.strip() or not value.isprintable():
        raise ValueError("A trimmed, printable reason is required")
    return value
def _timestamp(value: str) -> str:
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() != timedelta(0) or parsed.isoformat(timespec="microseconds") != value:
        raise ValueError("A canonical UTC timestamp is required")
    return value
class InvitationIssue(Model):
    request_id: UUID
    invitation_token: SecretStr
    name: Name
    title: str = Field(default="", strict=True, max_length=256)
    reason: Reason
    expires_at: Epoch
    max_initial_expires_at: Epoch

    @field_validator("invitation_token")
    @classmethod
    def generated_token(cls, value: SecretStr) -> SecretStr:
        if not valid_invitation_token(value.get_secret_value()):
            raise ValueError("The invitation must be a generated Tensorus invitation token")
        return value

    @field_validator("reason")
    @classmethod
    def checked_reason(cls, value: str) -> str:
        return _reason(value)

    @model_validator(mode="after")
    def ordered_expiries(self) -> Self:
        if self.max_initial_expires_at < self.expires_at:
            raise ValueError("The initial key limit must not precede the invitation expiry")
        return self
class InvitationClaim(Model):
    invitation_id: UUID
    name: Name
    title: str = Field(default="", strict=True, max_length=256)
    initial_api_key: SecretStr
    expires_at: Epoch

    @field_validator("initial_api_key")
    @classmethod
    def generated_key(cls, value: SecretStr) -> SecretStr:
        if not valid_generated_token(value.get_secret_value()):
            raise ValueError("The initial credential must be a generated Tensorus key")
        return value
class InvitationRevoke(Model):
    request_id: UUID
    reason: Reason
    confirmation: Literal["REVOKE"]

    @field_validator("reason")
    @classmethod
    def checked_reason(cls, value: str) -> str:
        return _reason(value)
class InvitationReceipt(Model):
    format: Literal["tensorus-workspace-provisioning-invitation-v1"] = FORMAT
    id: UUID
    name: Name
    title: str = Field(strict=True, max_length=256)
    created_at: str
    expires_at: Epoch
    max_initial_expires_at: Epoch

    @field_validator("created_at")
    @classmethod
    def checked_timestamp(cls, value: str) -> str:
        return _timestamp(value)

    @model_validator(mode="after")
    def ordered_expiries(self) -> Self:
        if self.max_initial_expires_at < self.expires_at:
            raise ValueError("Invitation expiries are inconsistent")
        return self
class InvitationView(InvitationReceipt):
    state: Literal["available", "claimed"]
class InvitationRevocationReceipt(Model):
    format: Literal["tensorus-workspace-invitation-revocation-v1"] = REVOCATION_FORMAT
    invitation_id: UUID
    request_id: UUID
    reason: Reason
    revoked_at: str
    claimed: bool
    initial_key_revoked: Literal[False] = False

    @field_validator("reason")
    @classmethod
    def checked_reason(cls, value: str) -> str:
        return _reason(value)

    @field_validator("revoked_at")
    @classmethod
    def checked_timestamp(cls, value: str) -> str:
        return _timestamp(value)

    @field_validator("claimed", "initial_key_revoked", mode="before")
    @classmethod
    def exact_boolean(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Invitation flags must be booleans")
        return value
class InvitationOperatorStatus(InvitationReceipt):
    reason: Reason
    created_by: Literal["bootstrap"]
    state: Literal["available", "claimed", "expired", "revoked"]
    claimed_at: str | None
    claimed_tenant_id: UUID | None
    claimed_key_id: UUID | None
    initial_expires_at: Epoch | None
    revoked_at: str | None
    revoked_by: Literal["bootstrap"] | None
    revocation_request_id: UUID | None
    revocation_reason: Reason | None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        claim = (
            self.claimed_at,
            self.claimed_tenant_id,
            self.claimed_key_id,
            self.initial_expires_at,
        )
        revocation = (
            self.revoked_at,
            self.revoked_by,
            self.revocation_request_id,
            self.revocation_reason,
        )
        if (
            (any(value is None for value in claim) and any(value is not None for value in claim))
            or (
                any(value is None for value in revocation)
                and any(value is not None for value in revocation)
            )
            or (self.state == "revoked") != (self.revoked_at is not None)
            or (self.state == "available" and self.claimed_tenant_id is not None)
            or (self.state == "claimed" and self.claimed_tenant_id is None)
            or (
                self.claimed_tenant_id is not None
                and self.claimed_key_id != initial_key_id(self.claimed_tenant_id)
            )
            or (
                self.initial_expires_at is not None
                and self.initial_expires_at > self.max_initial_expires_at
            )
        ):
            raise ValueError("Invitation status is inconsistent")
        if self.claimed_at is not None:
            _timestamp(self.claimed_at)
        if self.revoked_at is not None:
            _timestamp(self.revoked_at)
        _reason(self.reason)
        if self.revocation_reason is not None:
            _reason(self.revocation_reason)
        return self
