"""Self-scoped workspace discovery for verified membership sessions."""
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_validator
from tensorus.models import Model
from tensorus.models import Role
MAX_PAGE_SIZE = 100
MAX_RESPONSE_BYTES = 512 * 1024
class WorkspaceDirectoryCapabilities(Model):
    format: Literal["tensorus-workspace-directory-v1"]
    max_page_size: Literal[100]
    max_response_bytes: Literal[524288]
    membership_session_required: Literal[True]
    active_or_invited_only: Literal[True]
    unavailable_workspaces_excluded: Literal[True]
    separate_assertion_exchange_required: Literal[True]
    transfers_authority: Literal[False]
    cross_page_snapshot: Literal[False]

    @field_validator("max_page_size", "max_response_bytes", mode="before")
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Workspace directory limits must be integers")
        return value

    @field_validator(
        "membership_session_required",
        "active_or_invited_only",
        "unavailable_workspaces_excluded",
        "separate_assertion_exchange_required",
        "transfers_authority",
        "cross_page_snapshot",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Workspace directory flags must be booleans")
        return value
class WorkspaceDirectoryItem(Model):
    tenant_id: UUID
    name: str = Field(
        strict=True, min_length=1, max_length=96, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$"
    )
    title: str = Field(strict=True, max_length=256)
    membership_id: UUID
    role: Role
    status: Literal["active", "invited"]
class WorkspaceDirectoryPage(Model):
    format: Literal["tensorus-workspace-directory-v1"]
    current_tenant_id: UUID
    current_membership_id: UUID
    limit: Annotated[StrictInt, Field(ge=1, le=MAX_PAGE_SIZE)]
    after: UUID | None
    items: list[WorkspaceDirectoryItem] = Field(max_length=MAX_PAGE_SIZE)
    next_after: UUID | None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        ids = [item.tenant_id for item in self.items]
        if (
            len(ids) > self.limit
            or ids != sorted(set(ids))
            or len({item.membership_id for item in self.items}) != len(ids)
            or len({item.name for item in self.items}) != len(ids)
            or (self.after is not None and any(item <= self.after for item in ids))
            or (
                self.next_after is not None
                and (len(ids) != self.limit or not ids or self.next_after != ids[-1])
            )
        ):
            raise ValueError("Workspace directory page or cursor is inconsistent")
        for item in self.items:
            if item.tenant_id == self.current_tenant_id and (
                item.membership_id != self.current_membership_id or item.status != "active"
            ):
                raise ValueError("Current workspace membership is inconsistent")
        return self
