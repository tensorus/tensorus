"""Provider-backed service identities using the existing workspace membership authority."""
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
from tensorus.models import WorkspaceMemberInvite
from tensorus.organizations import _timestamp
FORMAT: Literal["tensorus-workspace-service-actor-v1"] = "tensorus-workspace-service-actor-v1"
MAX_PAGE_SIZE: Literal[100] = 100
MAX_RESPONSE_BYTES: Literal[524288] = 524_288
MAX_SELF_RESPONSE_BYTES: Literal[4096] = 4_096
class ServiceActorInvitation(WorkspaceMemberInvite):
    request_id: UUID
    expected_issuer: Annotated[str, Field(strict=True, min_length=1, max_length=1024)]
    expected_policy_hash: Annotated[
        str, Field(strict=True, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ]
    display_name: Annotated[str, Field(strict=True, min_length=1, max_length=128)]
    confirmation: Literal["INVITE"]

    @field_validator("display_name")
    @classmethod
    def trimmed_display_name(cls, value: str) -> str:
        if value != value.strip() or not value.isprintable():
            raise ValueError("A trimmed, printable service actor name is required")
        return value
class ServiceActorMember(Model):
    id: UUID
    tenant_id: UUID
    user_id: UUID
    issuer: Annotated[str, Field(strict=True, min_length=1, max_length=1024)]
    subject: Annotated[str, Field(strict=True, min_length=1, max_length=255)]
    role: Role
    status: Literal["invited", "active", "suspended", "removed"]
    generation: Annotated[StrictInt, Field(ge=1, le=2**63 - 1)]
    invited_by: Annotated[str, Field(strict=True, min_length=1, max_length=36)]
    created_at: str
    updated_at: str
    accepted_at: str | None

    _created_at = field_validator("created_at")(_timestamp)
    _updated_at = field_validator("updated_at")(_timestamp)

    @field_validator("accepted_at")
    @classmethod
    def accepted_timestamp(cls, value: str | None) -> str | None:
        return _timestamp(value) if value is not None else None
class ServiceActorView(Model):
    format: Literal["tensorus-workspace-service-actor-v1"] = FORMAT
    actor_id: UUID
    invitation_request_id: UUID
    display_name: Annotated[str, Field(strict=True, min_length=1, max_length=128)]
    invited_role: Role
    invited_policy_hash: Annotated[
        str, Field(strict=True, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ]
    member: ServiceActorMember

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.actor_id != self.member.user_id:
            raise ValueError("Service actor identity differs from its workspace membership")
        if self.display_name != self.display_name.strip() or not self.display_name.isprintable():
            raise ValueError("Service actor name is invalid")
        return self
class ServiceActorInvitationReceipt(Model):
    format: Literal["tensorus-workspace-service-actor-v1"] = FORMAT
    request_id: UUID
    actor: ServiceActorView
    replayed: bool

    @field_validator("replayed", mode="before")
    @classmethod
    def exact_replayed(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Service actor retry evidence must be boolean")
        return value

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.request_id != self.actor.invitation_request_id:
            raise ValueError("Service actor receipt request is inconsistent")
        return self
class ServiceActorPage(Model):
    format: Literal["tensorus-workspace-service-actor-v1"] = FORMAT
    tenant_id: UUID
    configured_issuer: Annotated[str, Field(strict=True, min_length=1, max_length=1024)] | None
    identity_policy_hash: (
        Annotated[str, Field(strict=True, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]
        | None
    )
    limit: Annotated[StrictInt, Field(ge=1, le=MAX_PAGE_SIZE)]
    offset: Annotated[StrictInt, Field(ge=0, le=10_000)]
    total: Annotated[StrictInt, Field(ge=0, le=10_000)]
    items: list[ServiceActorView] = Field(max_length=MAX_PAGE_SIZE)
    cross_page_snapshot: Literal[False] = False

    @field_validator("cross_page_snapshot", mode="before")
    @classmethod
    def exact_snapshot(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Service actor snapshot evidence must be boolean")
        return value

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (
            (self.configured_issuer is None) != (self.identity_policy_hash is None)
            or len(self.items) > self.limit
            or len(self.items) > max(0, self.total - self.offset)
            or len({item.member.id for item in self.items}) != len(self.items)
            or len({item.invitation_request_id for item in self.items}) != len(self.items)
            or any(item.member.tenant_id != self.tenant_id for item in self.items)
        ):
            raise ValueError("Service actor page is inconsistent")
        return self
class SelfServiceActor(Model):
    actor_id: UUID
    membership_id: UUID
    display_name: Annotated[str, Field(strict=True, min_length=1, max_length=128)]

    @field_validator("display_name")
    @classmethod
    def trimmed_display_name(cls, value: str) -> str:
        if value != value.strip() or not value.isprintable():
            raise ValueError("A trimmed, printable service actor name is required")
        return value
class ServiceActorIdentity(Model):
    format: Literal["tensorus-workspace-service-actor-v1"] = FORMAT
    tenant_id: UUID
    key_id: UUID
    source: Literal["workspace_session", "delegated_grant", "other"]
    actor: SelfServiceActor | None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.source == "other" and self.actor is not None:
            raise ValueError("An unrelated credential cannot identify a service actor")
        return self
class ServiceActorCapabilities(Model):
    format: Literal["tensorus-workspace-service-actor-v1"] = FORMAT
    max_request_bytes: Annotated[StrictInt, Field(ge=1, le=MAX_CONTROL_REQUEST_BYTES)]
    max_page_size: Literal[100] = MAX_PAGE_SIZE
    max_response_bytes: Literal[524288] = MAX_RESPONSE_BYTES
    max_self_response_bytes: Literal[4096] = MAX_SELF_RESPONSE_BYTES
    workspace_membership_authority: Literal[True] = True
    verified_assertion_required: Literal[True] = True
    exact_invitation_retry: Literal[True] = True
    separate_scoped_delegation: Literal[True] = True
    provider_policy_pinned: Literal[True] = True
    self_issued_grant_requires_admin_role: Literal[True] = True
    other_issuer_actor_recipient_binding: Literal[False] = False
    membership_changes_revoke_sessions_and_grants: Literal[True] = True
    administrator_declares_actor_type: Literal[True] = True
    workload_identity_independently_proven: Literal[False] = False
    existing_members_reclassified: Literal[False] = False
    provider_sessions_revoked: Literal[False] = False
    exported_copies_revoked: Literal[False] = False
    cross_page_snapshot: Literal[False] = False

    @field_validator(
        "max_page_size", "max_response_bytes", "max_self_response_bytes", mode="before"
    )
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Service actor limits must be integers")
        return value

    @field_validator(
        "workspace_membership_authority",
        "verified_assertion_required",
        "exact_invitation_retry",
        "separate_scoped_delegation",
        "provider_policy_pinned",
        "self_issued_grant_requires_admin_role",
        "other_issuer_actor_recipient_binding",
        "membership_changes_revoke_sessions_and_grants",
        "administrator_declares_actor_type",
        "workload_identity_independently_proven",
        "existing_members_reclassified",
        "provider_sessions_revoked",
        "exported_copies_revoked",
        "cross_page_snapshot",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Service actor flags must be booleans")
        return value
