"""Catalog-backed public service declarations, not enforcement or legal approval."""
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
from tensorus.catalog_numbers import MAX_SERVICE_DISCLOSURE_EVENTS
from tensorus.models import Model
FORMAT: Literal["tensorus-service-disclosures-v1"] = "tensorus-service-disclosures-v1"
EVENT_FORMAT: Literal["tensorus-service-disclosure-event-v1"] = (
    "tensorus-service-disclosure-event-v1"
)
RECEIPT_FORMAT: Literal["tensorus-service-disclosure-update-v1"] = (
    "tensorus-service-disclosure-update-v1"
)
MAX_RESPONSE_BYTES = 32 * 1024
FIELDS = ("usage", "residency", "retention", "privacy", "support")
Declaration = Annotated[str, Field(strict=True, min_length=1, max_length=512)]
Reason = Annotated[str, Field(strict=True, min_length=1, max_length=512)]
Generation = Annotated[StrictInt, Field(ge=1, le=MAX_SERVICE_DISCLOSURE_EVENTS)]
def _printable(value: str) -> str:
    if value != value.strip() or not value.isprintable():
        raise ValueError("A trimmed, printable single-line value is required")
    return value
def _timestamp(value: str) -> str:
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() != timedelta(0) or parsed.isoformat(timespec="microseconds") != value:
        raise ValueError("A canonical UTC timestamp is required")
    return value
class DisclosureFields(Model):
    usage: Declaration | None
    residency: Declaration | None
    retention: Declaration | None
    privacy: Declaration | None
    support: Declaration | None

    @field_validator(*FIELDS)
    @classmethod
    def checked_declaration(cls, value: str | None) -> str | None:
        return _printable(value) if value is not None else None
class DisclosureUpdate(DisclosureFields):
    expected_generation: Generation
    request_id: UUID
    reason: Reason
    confirmation: Literal["PUBLISH"]

    @field_validator("reason")
    @classmethod
    def checked_reason(cls, value: str) -> str:
        return _printable(value)
class DisclosureEvent(DisclosureFields):
    format: Literal["tensorus-service-disclosure-event-v1"] = EVENT_FORMAT
    generation: Generation
    expected_generation: Annotated[StrictInt, Field(ge=0)]
    request_id: UUID | None
    updated_at: str
    updated_by: Literal["system", "bootstrap"]
    reason: Reason

    @field_validator("updated_at")
    @classmethod
    def checked_timestamp(cls, value: str) -> str:
        return _timestamp(value)

    @field_validator("reason")
    @classmethod
    def checked_reason(cls, value: str) -> str:
        return _printable(value)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.expected_generation != self.generation - 1:
            raise ValueError("Disclosure generation is inconsistent")
        if self.generation == 1:
            if (
                self.request_id is not None
                or self.updated_by != "system"
                or self.reason != "initial unconfigured disclosure"
                or any(getattr(self, field) is not None for field in FIELDS)
            ):
                raise ValueError("Initial disclosure evidence is inconsistent")
        elif self.request_id is None or self.updated_by != "bootstrap":
            raise ValueError("Published disclosure evidence is inconsistent")
        return self
class DisclosureView(DisclosureFields):
    format: Literal["tensorus-service-disclosures-v1"] = FORMAT
    generation: Generation
    updated_at: str
    operator_declared: StrictBool
    independently_verified: Literal[False] = False
    enforces_policy: Literal[False] = False
    legal_approval_established: Literal[False] = False

    @field_validator("updated_at")
    @classmethod
    def checked_timestamp(cls, value: str) -> str:
        return _timestamp(value)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.operator_declared != (self.generation > 1) or (
            self.generation == 1 and any(getattr(self, field) is not None for field in FIELDS)
        ):
            raise ValueError("Disclosure attribution is inconsistent")
        return self
class DisclosureUpdateReceipt(Model):
    format: Literal["tensorus-service-disclosure-update-v1"] = RECEIPT_FORMAT
    replayed: StrictBool
    event: DisclosureEvent
    current: DisclosureView

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.current.generation < self.event.generation:
            raise ValueError("Current disclosure precedes the retained event")
        if self.current.generation == self.event.generation and self.current != view(self.event):
            raise ValueError("Current disclosure does not match the retained event")
        if self.current.generation > self.event.generation and not self.replayed:
            raise ValueError("A new disclosure cannot precede the current declaration")
        return self
def view(event: DisclosureEvent) -> DisclosureView:
    return DisclosureView(
        **event.model_dump(include=set(FIELDS)),
        generation=event.generation,
        updated_at=event.updated_at,
        operator_declared=event.generation > 1,
    )
