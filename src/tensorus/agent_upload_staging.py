"""Per-credential admission for existing resumable-upload reservations."""
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_validator
from tensorus.models import Model
from tensorus.serialization import NPY_MAX_OVERHEAD_BYTES
FORMAT: Literal["tensorus-agent-upload-staging-v1"] = "tensorus-agent-upload-staging-v1"
ADMIN_FORMAT: Literal["tensorus-agent-upload-staging-admin-v1"] = (
    "tensorus-agent-upload-staging-admin-v1"
)
MAX_BYTES = 2**53 - 1
MAX_SESSIONS = 10_000
MAX_UPLOAD_BYTES = 2 * 1024**3 + NPY_MAX_OVERHEAD_BYTES
MAX_RESPONSE_BYTES: Literal[4096] = 4096
class AgentUploadStagingCapabilities(Model):
    format: Literal["tensorus-agent-upload-staging-v1"] = FORMAT
    max_staged_bytes: Annotated[StrictInt, Field(ge=1024, le=MAX_BYTES)]
    max_open_sessions: Annotated[StrictInt, Field(ge=1, le=MAX_SESSIONS)]
    max_response_bytes: Literal[4096] = MAX_RESPONSE_BYTES
    applies_to: Literal["scoped_agent_resumable_uploads"] = "scoped_agent_resumable_uploads"
    declared_npy_bytes_counted: Literal[True] = True
    exact_creation_retries_preserved: Literal[True] = True
    physical_disk_reserved: Literal[False] = False
    other_ingestion_paths_included: Literal[False] = False
    fleet_configuration_verified: Literal[False] = False

    @field_validator(
        "declared_npy_bytes_counted",
        "exact_creation_retries_preserved",
        "physical_disk_reserved",
        "other_ingestion_paths_included",
        "fleet_configuration_verified",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Agent upload staging flags must be booleans")
        return value

    @field_validator("max_response_bytes", mode="before")
    @classmethod
    def exact_limit(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Agent upload staging response limit must be an integer")
        return value
class AgentUploadStagingStatus(Model):
    format: Literal["tensorus-agent-upload-staging-v1"] = FORMAT
    tenant_id: UUID
    key_id: UUID
    max_staged_bytes: Annotated[StrictInt, Field(ge=1024, le=MAX_BYTES)]
    max_open_sessions: Annotated[StrictInt, Field(ge=1, le=MAX_SESSIONS)]
    reserved_bytes: Annotated[StrictInt, Field(ge=0, le=MAX_BYTES)]
    open_sessions: Annotated[StrictInt, Field(ge=0, le=MAX_SESSIONS)]
    remaining_bytes: Annotated[StrictInt, Field(ge=0, le=MAX_BYTES)]
    remaining_sessions: Annotated[StrictInt, Field(ge=0, le=MAX_SESSIONS)]
    over_limit: bool

    @field_validator("over_limit", mode="before")
    @classmethod
    def exact_flag(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Agent upload staging over-limit flag must be a boolean")
        return value

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (
            self.remaining_bytes != max(0, self.max_staged_bytes - self.reserved_bytes)
            or self.remaining_sessions != max(0, self.max_open_sessions - self.open_sessions)
            or self.over_limit
            != (
                self.reserved_bytes > self.max_staged_bytes
                or self.open_sessions > self.max_open_sessions
            )
            or (self.open_sessions == 0) != (self.reserved_bytes == 0)
            or self.reserved_bytes < self.open_sessions
            or self.reserved_bytes > self.open_sessions * MAX_UPLOAD_BYTES
        ):
            raise ValueError("Agent upload staging counts are inconsistent")
        return self
class AgentUploadStagingAdminCapabilities(Model):
    format: Literal["tensorus-agent-upload-staging-admin-v1"] = ADMIN_FORMAT
    max_response_bytes: Literal[4096] = MAX_RESPONSE_BYTES
    requires_unscoped_administrator: Literal[True] = True
    target_is_agent_grant: Literal[True] = True
    includes_revoked_open_records: Literal[True] = True
    mutates_reservations: Literal[False] = False
    physical_disk_reserved: Literal[False] = False

    @field_validator("max_response_bytes", mode="before")
    @classmethod
    def exact_limit(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Agent staging review response limit must be an integer")
        return value

    @field_validator(
        "requires_unscoped_administrator",
        "target_is_agent_grant",
        "includes_revoked_open_records",
        "mutates_reservations",
        "physical_disk_reserved",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Agent staging review flags must be booleans")
        return value
class AgentUploadStagingAdminStatus(Model):
    format: Literal["tensorus-agent-upload-staging-admin-v1"] = ADMIN_FORMAT
    status: AgentUploadStagingStatus
    includes_revoked_open_records: Literal[True] = True
    mutates_reservations: Literal[False] = False
    physical_disk_reserved: Literal[False] = False

    @field_validator(
        "includes_revoked_open_records",
        "mutates_reservations",
        "physical_disk_reserved",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Agent staging review flags must be booleans")
        return value
