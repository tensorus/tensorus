"""Audited holds on exact data-product revisions."""
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictInt
from pydantic import model_validator
from tensorus.models import Model
from tensorus.models import Name
from tensorus.models import WithdrawalReason
from tensorus.products import Revision
from tensorus.retention import MAX_HOLDS_PER_RESOURCE
class ProductRevisionHoldCreate(Model):
    request_id: UUID
    reason: WithdrawalReason
class ProductRevisionHold(Model):
    format: Literal["tensorus-product-revision-hold-v1"] = "tensorus-product-revision-hold-v1"
    id: UUID
    tenant_id: UUID
    product_id: UUID
    revision: Revision
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
            raise ValueError("Product-revision hold release evidence is incomplete")
        return self
class ProductRevisionHoldHistory(Model):
    format: Literal["tensorus-product-revision-hold-history-v1"] = (
        "tensorus-product-revision-hold-history-v1"
    )
    tenant_id: UUID
    product_id: UUID
    product_name: Name
    revision: Revision
    retired: bool
    active_count: Annotated[StrictInt, Field(ge=0, le=MAX_HOLDS_PER_RESOURCE)]
    holds: Annotated[list[ProductRevisionHold], Field(max_length=MAX_HOLDS_PER_RESOURCE)]

    @model_validator(mode="after")
    def consistent_history(self) -> Self:
        if (
            self.active_count != sum(hold.state == "active" for hold in self.holds)
            or len({hold.id for hold in self.holds}) != len(self.holds)
            or any(
                hold.tenant_id != self.tenant_id
                or hold.product_id != self.product_id
                or hold.revision != self.revision
                for hold in self.holds
            )
            or (self.retired and self.active_count != 0)
        ):
            raise ValueError("Product-revision hold history is inconsistent")
        return self
