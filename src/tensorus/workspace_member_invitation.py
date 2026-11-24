"""Retryable human workspace invitations using the existing membership authority."""
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_validator
from tensorus.control_request_limits import MAX_CONTROL_REQUEST_BYTES
from tensorus.control_request_limits import WORKSPACE_MEMBER_INVITATION_PATH
from tensorus.models import Model
from tensorus.models import Role
from tensorus.models import WorkspaceMemberInvite
from tensorus.service_actors import ServiceActorMember
FORMAT: Literal["tensorus-workspace-member-invitation-v1"] = (
    "tensorus-workspace-member-invitation-v1"
)
PATH: Literal["/v1/workspace/member-invitations"] = WORKSPACE_MEMBER_INVITATION_PATH
MAX_RESPONSE_BYTES: Literal[16384] = 16384
PolicyHash = Annotated[
    str, Field(strict=True, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
]
Issuer = Annotated[str, Field(strict=True, min_length=1, max_length=1024)]
class MemberInvitation(WorkspaceMemberInvite):
    request_id: UUID
    expected_issuer: Issuer
    expected_policy_hash: PolicyHash
    confirmation: Literal["INVITE"]

    @field_validator("request_id")
    @classmethod
    def non_nil_request_id(cls, value: UUID) -> UUID:
        if value.int == 0:
            raise ValueError("A non-nil invitation request ID is required")
        return value
class WorkspaceMemberEvidence(ServiceActorMember):
    pass
class MemberInvitationReceipt(Model):
    format: Literal["tensorus-workspace-member-invitation-v1"] = FORMAT
    request_id: UUID
    invited_role: Role
    invited_policy_hash: PolicyHash
    member: WorkspaceMemberEvidence
    replayed: bool

    @field_validator("replayed", mode="before")
    @classmethod
    def exact_replayed(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Invitation replay evidence must be boolean")
        return value

    @model_validator(mode="after")
    def non_nil_request_id(self) -> Self:
        if self.request_id.int == 0:
            raise ValueError("Invitation receipt request ID is invalid")
        return self
class MemberInvitationCapabilities(Model):
    format: Literal["tensorus-workspace-member-invitation-v1"] = FORMAT
    path: Literal["/v1/workspace/member-invitations"] = PATH
    configured_issuer: Issuer
    identity_policy_hash: PolicyHash
    max_request_bytes: Annotated[StrictInt, Field(ge=1, le=MAX_CONTROL_REQUEST_BYTES)]
    max_response_bytes: Literal[16384] = MAX_RESPONSE_BYTES
    active_unscoped_administrator_required: Literal[True] = True
    exact_retry_same_actor: Literal[True] = True
    provider_policy_pinned: Literal[True] = True
    retry_returns_current_membership: Literal[True] = True
    existing_members_reclassified: Literal[False] = False
    retry_reactivates_membership: Literal[False] = False
    email_sent: Literal[False] = False
    provider_session_created: Literal[False] = False
    evidence_independently_authenticated: Literal[False] = False

    @field_validator("max_response_bytes", mode="before")
    @classmethod
    def exact_limit(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Invitation response limit must be an integer")
        return value

    @field_validator(
        "active_unscoped_administrator_required",
        "exact_retry_same_actor",
        "provider_policy_pinned",
        "retry_returns_current_membership",
        "existing_members_reclassified",
        "retry_reactivates_membership",
        "email_sent",
        "provider_session_created",
        "evidence_independently_authenticated",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Invitation capability flags must be booleans")
        return value
