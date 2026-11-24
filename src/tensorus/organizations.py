"""Organization governance over existing verified users and workspace memberships."""
import re
from datetime import datetime
from datetime import timedelta
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictBool
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_validator
from tensorus.control_request_limits import MAX_CONTROL_REQUEST_BYTES
from tensorus.models import Model
from tensorus.models import Name
from tensorus.models import WithdrawalReason
FORMAT: Literal["tensorus-organization-governance-v1"] = "tensorus-organization-governance-v1"
CAPABILITIES_V2: Literal["tensorus-organization-governance-v2"] = (
    "tensorus-organization-governance-v2"
)
SESSION_FENCE_FORMAT: Literal["tensorus-organization-session-fence-v1"] = (
    "tensorus-organization-session-fence-v1"
)
SESSION_FENCE_FORMAT_V2: Literal["tensorus-organization-session-fence-v2"] = (
    "tensorus-organization-session-fence-v2"
)
MAX_WORKSPACES: Literal[16] = 16
MAX_MEMBERS: Literal[10000] = 10_000
MAX_ORGANIZATIONS_PER_USER: Literal[100] = 100
MAX_PAGE_SIZE: Literal[25] = 25
MAX_RESPONSE_BYTES: Literal[524288] = 524_288
MAX_GENERATION = 2**53 - 1
OrgRole = Literal["owner", "member"]
OrgStatus = Literal["invited", "active", "suspended", "removed"]
def _timestamp(value: str) -> str:
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Organization timestamp is invalid") from exc
    if (
        timestamp.utcoffset() != timedelta(0)
        or timestamp.isoformat(timespec="microseconds") != value
    ):
        raise ValueError("Organization timestamp must be canonical UTC")
    return value
class OrganizationCapabilities(Model):
    format: Literal[
        "tensorus-organization-governance-v1", "tensorus-organization-governance-v2"
    ] = FORMAT
    max_workspaces: Literal[16] = MAX_WORKSPACES
    max_members: Literal[10000] = MAX_MEMBERS
    max_organizations_per_user: Literal[100] = MAX_ORGANIZATIONS_PER_USER
    max_page_size: Literal[25] = MAX_PAGE_SIZE
    max_request_bytes: Annotated[StrictInt, Field(ge=1, le=MAX_CONTROL_REQUEST_BYTES)]
    max_response_bytes: Literal[524288] = MAX_RESPONSE_BYTES
    verified_membership_session_required: Literal[True] = True
    workspace_membership_still_required: Literal[True] = True
    member_changes_revoke_sessions_and_grants: Literal[True] = True
    membership_evidence_bound_to_audit: Literal[True] = True
    last_usable_owner_protected: Literal[True] = True
    existing_service_keys_affected: Literal[False] = False
    provider_sessions_revoked: Literal[False] = False
    exported_copies_revoked: Literal[False] = False
    workspace_detach_supported: StrictBool = False
    database_time_or_memory_bounded: Literal[False] = False

    @field_validator(
        "max_workspaces",
        "max_members",
        "max_organizations_per_user",
        "max_page_size",
        "max_request_bytes",
        "max_response_bytes",
        mode="before",
    )
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Organization limits must be integers")
        return value

    @field_validator(
        "verified_membership_session_required",
        "workspace_membership_still_required",
        "member_changes_revoke_sessions_and_grants",
        "membership_evidence_bound_to_audit",
        "last_usable_owner_protected",
        "existing_service_keys_affected",
        "provider_sessions_revoked",
        "exported_copies_revoked",
        "workspace_detach_supported",
        "database_time_or_memory_bounded",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Organization flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_version(self) -> Self:
        if self.workspace_detach_supported != (self.format == CAPABILITIES_V2):
            raise ValueError("Organization capability version is inconsistent")
        return self
class OrganizationSessionFenceCapabilities(Model):
    format: Literal["tensorus-organization-session-fence-v1"] = SESSION_FENCE_FORMAT
    max_organizations_per_session: Literal[100] = MAX_ORGANIZATIONS_PER_USER
    organization_access_requires_current_binding: Literal[True] = True
    organization_creation_binds_creator: Literal[True] = True
    legacy_existing_organization_access_requires_exchange: Literal[True] = True
    unrelated_workspace_credentials_revoked: Literal[False] = False
    assertion_issue_time_after_change_enforced: Literal[False] = False
    database_time_or_memory_bounded: Literal[False] = False

    @field_validator("max_organizations_per_session", mode="before")
    @classmethod
    def exact_limit(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Organization session limit must be an integer")
        return value

    @field_validator(
        "organization_access_requires_current_binding",
        "organization_creation_binds_creator",
        "legacy_existing_organization_access_requires_exchange",
        "unrelated_workspace_credentials_revoked",
        "assertion_issue_time_after_change_enforced",
        "database_time_or_memory_bounded",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Organization session flags must be booleans")
        return value
class OrganizationSessionFenceV2Capabilities(Model):
    format: Literal["tensorus-organization-session-fence-v2"] = SESSION_FENCE_FORMAT_V2
    max_organizations_per_session: Literal[100] = MAX_ORGANIZATIONS_PER_USER
    organization_access_requires_current_binding: Literal[True] = True
    organization_creation_binds_creator: Literal[True] = True
    legacy_existing_organization_access_requires_exchange: Literal[True] = True
    unrelated_workspace_credentials_revoked: Literal[False] = False
    assertion_issue_time_after_change_enforced: Literal[True] = True
    workspace_membership_issue_time_enforced: Literal[True] = True
    associated_workspace_access_requires_current_binding: Literal[True] = True
    same_second_assertions_denied: Literal[True] = True
    previous_policy_sessions_invalidated: Literal[True] = True
    rollback_can_reactivate_previous_sessions: Literal[True] = True
    assertion_issue_time_independently_verified: Literal[False] = False
    provider_sessions_revoked: Literal[False] = False
    database_time_or_memory_bounded: Literal[False] = False

    @field_validator("max_organizations_per_session", mode="before")
    @classmethod
    def exact_limit(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Organization session limit must be an integer")
        return value

    @field_validator(
        "organization_access_requires_current_binding",
        "organization_creation_binds_creator",
        "legacy_existing_organization_access_requires_exchange",
        "unrelated_workspace_credentials_revoked",
        "assertion_issue_time_after_change_enforced",
        "workspace_membership_issue_time_enforced",
        "associated_workspace_access_requires_current_binding",
        "same_second_assertions_denied",
        "previous_policy_sessions_invalidated",
        "rollback_can_reactivate_previous_sessions",
        "assertion_issue_time_independently_verified",
        "provider_sessions_revoked",
        "database_time_or_memory_bounded",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Organization session flags must be booleans")
        return value
class OrganizationCreate(Model):
    request_id: UUID
    name: Name
    title: Annotated[str, Field(strict=True, max_length=256)] = ""
    confirmation: Literal["CREATE"]
class OrganizationAttach(Model):
    request_id: UUID
    confirmation: Literal["ATTACH"]
class OrganizationInvite(Model):
    request_id: UUID
    subject: Annotated[str, Field(strict=True, min_length=1, max_length=255)]
    role: OrgRole

    @field_validator("subject")
    @classmethod
    def exact_subject(cls, value: str) -> str:
        if not re.fullmatch(r"[!-~]{1,255}", value):
            raise ValueError("OIDC subject must be printable ASCII without whitespace")
        return value
class OrganizationMemberUpdate(Model):
    request_id: UUID
    expected_generation: Annotated[StrictInt, Field(ge=1, lt=MAX_GENERATION)]
    role: OrgRole
    status: OrgStatus
    reason: WithdrawalReason
    confirmation: Literal["UPDATE"]

    @field_validator("reason")
    @classmethod
    def trimmed_reason(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("A trimmed reason is required")
        return value
class OrganizationView(Model):
    format: Literal["tensorus-organization-governance-v1"] = FORMAT
    id: UUID
    name: Name
    title: Annotated[str, Field(strict=True, max_length=256)]
    created_at: str
    membership_id: UUID
    role: OrgRole
    status: Literal["active", "invited"]
    current_workspace_attached: StrictBool

    _created_at = field_validator("created_at")(_timestamp)
class OrganizationDirectory(Model):
    format: Literal["tensorus-organization-governance-v1"] = FORMAT
    current_tenant_id: UUID
    current_membership_id: UUID
    items: list[OrganizationView] = Field(max_length=MAX_ORGANIZATIONS_PER_USER)

    @model_validator(mode="after")
    def distinct_items(self) -> Self:
        if (
            len({item.id for item in self.items}) != len(self.items)
            or len({item.membership_id for item in self.items}) != len(self.items)
            or len({item.name for item in self.items}) != len(self.items)
            or [item.name for item in self.items] != sorted(item.name for item in self.items)
            or sum(item.current_workspace_attached for item in self.items) > 1
            or any(
                item.current_workspace_attached and item.status != "active" for item in self.items
            )
        ):
            raise ValueError("Organization directory is inconsistent")
        return self
class OrganizationMember(Model):
    id: UUID
    organization_id: UUID
    user_id: UUID
    issuer: Annotated[str, Field(strict=True, min_length=1, max_length=1024)]
    subject: Annotated[str, Field(strict=True, min_length=1, max_length=255)]
    role: OrgRole
    status: OrgStatus
    generation: Annotated[StrictInt, Field(ge=1, le=MAX_GENERATION)]
    created_at: str
    updated_at: str
    accepted_at: str | None

    _times = field_validator("created_at", "updated_at")(_timestamp)

    @field_validator("accepted_at")
    @classmethod
    def accepted_timestamp(cls, value: str | None) -> str | None:
        return _timestamp(value) if value is not None else None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (
            not re.fullmatch(r"[!-~]{1,255}", self.subject)
            or (self.status == "active" and self.accepted_at is None)
            or (self.status == "invited" and self.accepted_at is not None)
        ):
            raise ValueError("Organization membership is inconsistent")
        return self
class OrganizationMemberPage(Model):
    format: Literal["tensorus-organization-governance-v1"] = FORMAT
    organization_id: UUID
    items: list[OrganizationMember] = Field(max_length=MAX_PAGE_SIZE)
    total: Annotated[StrictInt, Field(ge=0, le=MAX_MEMBERS)]
    limit: Annotated[StrictInt, Field(ge=1, le=MAX_PAGE_SIZE)]
    offset: Annotated[StrictInt, Field(ge=0, le=MAX_MEMBERS)]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (
            len(self.items) > self.limit
            or (self.items and self.offset + len(self.items) > self.total)
            or len({item.id for item in self.items}) != len(self.items)
            or len({item.user_id for item in self.items}) != len(self.items)
            or any(item.organization_id != self.organization_id for item in self.items)
        ):
            raise ValueError("Organization member page is inconsistent")
        return self
class OrganizationWorkspace(Model):
    organization_id: UUID
    tenant_id: UUID
    name: Name
    title: Annotated[str, Field(strict=True, max_length=256)]
    request_id: UUID
    attached_by_user_id: UUID
    attached_at: str

    _attached_at = field_validator("attached_at")(_timestamp)
class OrganizationWorkspaceList(Model):
    format: Literal["tensorus-organization-governance-v1"] = FORMAT
    organization_id: UUID
    items: list[OrganizationWorkspace] = Field(max_length=MAX_WORKSPACES)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (
            len({item.tenant_id for item in self.items}) != len(self.items)
            or len({item.request_id for item in self.items}) != len(self.items)
            or len({item.name for item in self.items}) != len(self.items)
            or any(item.organization_id != self.organization_id for item in self.items)
        ):
            raise ValueError("Organization workspace list is inconsistent")
        return self
