"""Retry-safe preparation of ordinary credentials without revoking their source."""
import hashlib
from datetime import datetime
from datetime import timedelta
from typing import Annotated
from typing import Literal
from uuid import UUID
from uuid import uuid5
from pydantic import Field
from pydantic import SecretStr
from pydantic import StrictInt
from pydantic import field_validator
from tensorus.control_request_limits import MAX_CONTROL_REQUEST_BYTES
from tensorus.credentials import valid_generated_token
from tensorus.models import Model
from tensorus.models import Role
FORMAT: Literal["tensorus-key-rotation-prepare-v1"] = "tensorus-key-rotation-prepare-v1"
MAX_RESPONSE_BYTES = 4096
def replacement_key_id(source_key_id: UUID, token: str) -> UUID:
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    return uuid5(source_key_id, f"{FORMAT}:{digest}")
class KeyRotationRequest(Model):
    name: str = Field(min_length=1, max_length=128)
    replacement_api_key: SecretStr
    expires_at: Annotated[StrictInt, Field(ge=1, le=2**53 - 1)]

    @field_validator("replacement_api_key")
    @classmethod
    def generated_credential(cls, value: SecretStr) -> SecretStr:
        if not valid_generated_token(value.get_secret_value()):
            raise ValueError("The replacement credential must be a generated Tensorus key")
        return value
class KeyRotationReceipt(Model):
    format: Literal["tensorus-key-rotation-prepare-v1"]
    tenant_id: UUID
    source_key_id: UUID
    replacement_key_id: UUID
    name: str = Field(min_length=1, max_length=128)
    role: Role
    created_at: str
    expires_at: Annotated[StrictInt, Field(ge=1, le=2**53 - 1)]
    source_revoked_by_prepare: Literal[False]
    grants_transferred: Literal[False]

    @field_validator("created_at")
    @classmethod
    def canonical_timestamp(cls, value: str) -> str:
        timestamp = datetime.fromisoformat(value)
        if (
            timestamp.utcoffset() != timedelta(0)
            or timestamp.isoformat(timespec="microseconds") != value
        ):
            raise ValueError("The rotation timestamp must be canonical UTC")
        return value

    @field_validator("source_revoked_by_prepare", "grants_transferred", mode="before")
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Rotation receipt flags must be booleans")
        return value
class KeyRotationCapabilities(Model):
    format: Literal["tensorus-key-rotation-prepare-v1"]
    max_request_bytes: StrictInt = Field(ge=1024, le=MAX_CONTROL_REQUEST_BYTES)
    max_response_bytes: Literal[4096]
    max_lifetime_seconds: Literal[31536000]
    client_held_secret: Literal[True]
    exact_retry: Literal[True]
    replacement_role_preserved: Literal[True]
    source_revoked_by_prepare: Literal[False]
    grants_transferred: Literal[False]
    membership_sessions_supported: Literal[False]
    agent_grants_supported: Literal[False]

    @field_validator(
        "client_held_secret",
        "exact_retry",
        "replacement_role_preserved",
        "source_revoked_by_prepare",
        "grants_transferred",
        "membership_sessions_supported",
        "agent_grants_supported",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Rotation capability flags must be booleans")
        return value
