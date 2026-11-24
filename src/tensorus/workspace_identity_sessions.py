"""Coordinated workspace identity-session interruption and recovery."""
from datetime import datetime
from datetime import timedelta
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
MAX_MEMBERSHIPS = 10_000
MAX_SESSIONS = 10_000
MAX_GRANTS = 20_000
MAX_RESPONSE_BYTES = 16 * 1024
MAX_GENERATION = 2**53 - 1
class IdentitySessionsTransition(Model):
    expected_generation: Annotated[StrictInt, Field(ge=1, lt=MAX_GENERATION)]
    request_id: UUID
    reason: str = Field(strict=True, min_length=1, max_length=512)
    confirmation: Literal["PAUSE", "RESUME"]

    @field_validator("reason")
    @classmethod
    def trimmed_reason(cls, value: str) -> str:
        if not value.strip() or value != value.strip() or not value.isprintable():
            raise ValueError("A trimmed, printable reason is required")
        return value
class IdentitySessionsEventDetails(Model):
    format: Literal["tensorus-workspace-identity-sessions-v1"]
    state: Literal["active", "paused"]
    generation: Annotated[StrictInt, Field(ge=2, le=MAX_GENERATION)]
    expected_generation: Annotated[StrictInt, Field(ge=1, lt=MAX_GENERATION)]
    reason: str = Field(strict=True, min_length=1, max_length=512)
    memberships_advanced: Annotated[StrictInt, Field(ge=0, le=MAX_MEMBERSHIPS)]
    sessions_revoked: Annotated[StrictInt, Field(ge=0, le=MAX_SESSIONS)]
    grants_revoked: Annotated[StrictInt, Field(ge=0, le=MAX_GRANTS)]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (
            self.generation != self.expected_generation + 1
            or not self.reason.strip()
            or self.reason != self.reason.strip()
            or not self.reason.isprintable()
        ):
            raise ValueError("Identity-session event is inconsistent")
        return self
class IdentitySessionsEvent(IdentitySessionsEventDetails):
    id: UUID
    tenant_id: UUID
    actor_id: str
    created_at: str
class IdentitySessionsStatus(Model):
    format: Literal["tensorus-workspace-identity-sessions-v1"]
    tenant_id: UUID
    state: Literal["active", "paused"]
    generation: Annotated[StrictInt, Field(ge=1, le=MAX_GENERATION)]
    updated_at: str
    updated_by: str
    last_event: IdentitySessionsEvent | None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        event = self.last_event
        if self.generation == 1:
            if self.state != "active" or event is not None:
                raise ValueError("Initial identity-session state is inconsistent")
        elif (
            event is None
            or event.tenant_id != self.tenant_id
            or event.state != self.state
            or event.generation != self.generation
            or event.actor_id != self.updated_by
            or event.created_at != self.updated_at
        ):
            raise ValueError("Identity-session state is inconsistent")
        timestamp = datetime.fromisoformat(self.updated_at)
        if (
            timestamp.utcoffset() != timedelta(0)
            or timestamp.isoformat(timespec="microseconds") != self.updated_at
        ):
            raise ValueError("Identity-session timestamp is not canonical UTC")
        return self
class IdentitySessionsCapabilities(Model):
    format: Literal["tensorus-workspace-identity-sessions-v1"]
    max_request_bytes: Annotated[StrictInt, Field(ge=1024, le=MAX_CONTROL_REQUEST_BYTES)]
    max_response_bytes: Literal[16384]
    max_memberships: Literal[10000]
    max_sessions: Literal[10000]
    max_issued_grants: Literal[20000]
    max_events: Literal[10000]
    ordinary_administrator_required: Literal[True]
    current_transition_exact_retry: Literal[True]
    provider_sessions_revoked: Literal[False]
    running_processes_stopped: Literal[False]
    exported_copies_revoked: Literal[False]
    fleet_configuration_verified: Literal[False]

    @field_validator(
        "max_response_bytes",
        "max_memberships",
        "max_sessions",
        "max_issued_grants",
        "max_events",
        mode="before",
    )
    @classmethod
    def strict_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Identity-session capability limits must be integers")
        return value

    @field_validator(
        "ordinary_administrator_required",
        "current_transition_exact_retry",
        "provider_sessions_revoked",
        "running_processes_stopped",
        "exported_copies_revoked",
        "fleet_configuration_verified",
        mode="before",
    )
    @classmethod
    def strict_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Identity-session capability flags must be booleans")
        return value
