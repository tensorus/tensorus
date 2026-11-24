"""Audited, explicit holds on supported catalog retirement operations."""
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictInt
from pydantic import model_validator
from tensorus.models import Model
from tensorus.models import WithdrawalReason
MAX_HOLDS_PER_RESOURCE = 64
RetentionResource = Literal["snapshot", "product_release"]
class RetentionHoldCreate(Model):
    request_id: UUID
    resource_type: RetentionResource
    resource_id: UUID
    reason: WithdrawalReason
class RetentionHoldRelease(Model):
    request_id: UUID
    reason: WithdrawalReason
class RetentionHold(Model):
    format: Literal["tensorus-retention-hold-v1"] = "tensorus-retention-hold-v1"
    id: UUID
    tenant_id: UUID
    resource_type: RetentionResource
    resource_id: UUID
    reason: WithdrawalReason
    created_at: str
    created_by: str
    state: Literal["active", "released"]
    release_request_id: UUID | None
    release_reason: WithdrawalReason | None
    released_at: str | None
    released_by: str | None

    @model_validator(mode="after")
    def consistent_release(self) -> Self:
        release = (
            self.release_request_id,
            self.release_reason,
            self.released_at,
            self.released_by,
        )
        if (self.state == "active" and any(item is not None for item in release)) or (
            self.state == "released" and any(item is None for item in release)
        ):
            raise ValueError("Retention hold release evidence is incomplete")
        return self
class RetentionHoldHistory(Model):
    format: Literal["tensorus-retention-hold-history-v1"] = "tensorus-retention-hold-history-v1"
    tenant_id: UUID
    resource_type: RetentionResource
    resource_id: UUID
    resource_present: bool
    active_count: Annotated[StrictInt, Field(ge=0, le=MAX_HOLDS_PER_RESOURCE)]
    holds: Annotated[list[RetentionHold], Field(max_length=MAX_HOLDS_PER_RESOURCE)]

    @model_validator(mode="after")
    def consistent_history(self) -> Self:
        if (
            self.active_count != sum(hold.state == "active" for hold in self.holds)
            or len({hold.id for hold in self.holds}) != len(self.holds)
            or any(
                hold.tenant_id != self.tenant_id
                or hold.resource_type != self.resource_type
                or hold.resource_id != self.resource_id
                for hold in self.holds
            )
            or (not self.resource_present and self.active_count != 0)
        ):
            raise ValueError("Retention hold history is inconsistent")
        return self
