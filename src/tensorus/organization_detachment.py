"""One-way, audited detachment of a workspace from an organization."""
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
from tensorus.models import WithdrawalReason
from tensorus.organizations import MAX_MEMBERS
from tensorus.organizations import _timestamp
FORMAT: Literal["tensorus-organization-workspace-detachment-v1"] = (
    "tensorus-organization-workspace-detachment-v1"
)
MAX_SESSIONS: Literal[10000] = 10_000
MAX_GRANTS: Literal[20000] = 20_000
MAX_RESPONSE_BYTES: Literal[16384] = 16_384
class DetachmentCapabilities(Model):
    format: Literal["tensorus-organization-workspace-detachment-v1"] = FORMAT
    max_request_bytes: Annotated[StrictInt, Field(ge=1, le=MAX_CONTROL_REQUEST_BYTES)]
    max_response_bytes: Literal[16384] = MAX_RESPONSE_BYTES
    max_memberships: Literal[10000] = MAX_MEMBERS
    max_unrevoked_sessions: Literal[10000] = MAX_SESSIONS
    max_issued_grants: Literal[20000] = MAX_GRANTS
    owner_and_workspace_administrator_required: Literal[True] = True
    eligible_workspace_members_required: Literal[True] = True
    last_usable_organization_owner_protected: Literal[True] = True
    sessions_and_issued_grants_revoked: Literal[True] = True
    association_history_retained: Literal[True] = True
    reattachment_supported: Literal[False] = False
    ordinary_keys_revoked: Literal[False] = False
    provider_sessions_revoked: Literal[False] = False
    exported_copies_revoked: Literal[False] = False
    fleet_configuration_verified: Literal[False] = False
    database_time_or_memory_bounded: Literal[False] = False

    @field_validator(
        "max_request_bytes",
        "max_response_bytes",
        "max_memberships",
        "max_unrevoked_sessions",
        "max_issued_grants",
        mode="before",
    )
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Detachment limits must be integers")
        return value

    @field_validator(
        "owner_and_workspace_administrator_required",
        "eligible_workspace_members_required",
        "last_usable_organization_owner_protected",
        "sessions_and_issued_grants_revoked",
        "association_history_retained",
        "reattachment_supported",
        "ordinary_keys_revoked",
        "provider_sessions_revoked",
        "exported_copies_revoked",
        "fleet_configuration_verified",
        "database_time_or_memory_bounded",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Detachment flags must be booleans")
        return value
class DetachmentRequest(Model):
    request_id: UUID
    attachment_request_id: UUID
    reason: WithdrawalReason
    confirmation: Literal["DETACH"]

    @field_validator("reason")
    @classmethod
    def trimmed_reason(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("A trimmed reason is required")
        return value
class DetachmentReceipt(Model):
    format: Literal["tensorus-organization-workspace-detachment-v1"] = FORMAT
    tenant_id: UUID
    organization_id: UUID
    attachment_request_id: UUID
    request_id: UUID
    actor_user_id: UUID
    detached_at: str
    reason: WithdrawalReason
    memberships_advanced: Annotated[StrictInt, Field(ge=1, le=MAX_MEMBERS)]
    sessions_revoked: Annotated[StrictInt, Field(ge=1, le=MAX_SESSIONS)]
    grants_revoked: Annotated[StrictInt, Field(ge=0, le=MAX_GRANTS)]

    _detached_at = field_validator("detached_at")(_timestamp)

    @field_validator("reason")
    @classmethod
    def trimmed_reason(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("A trimmed reason is required")
        return value
class DetachmentStatus(Model):
    format: Literal["tensorus-organization-workspace-detachment-v1"] = FORMAT
    tenant_id: UUID
    state: Literal["unattached", "attached", "detached"]
    organization_id: UUID | None
    attachment_request_id: UUID | None
    detachment: DetachmentReceipt | None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.state == "unattached":
            if any((self.organization_id, self.attachment_request_id, self.detachment)):
                raise ValueError("Unattached workspace has association evidence")
        elif self.organization_id is None or self.attachment_request_id is None:
            raise ValueError("Workspace association evidence is missing")
        elif self.state == "attached" and self.detachment is not None:
            raise ValueError("Attached workspace has detachment evidence")
        elif self.state == "detached" and (
            self.detachment is None
            or self.detachment.tenant_id != self.tenant_id
            or self.detachment.organization_id != self.organization_id
            or self.detachment.attachment_request_id != self.attachment_request_id
        ):
            raise ValueError("Workspace detachment evidence is inconsistent")
        return self
