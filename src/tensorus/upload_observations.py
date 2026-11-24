"""Creator-scoped observations of resumable upload attempts in catalog audit."""
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
from tensorus.models import Model
from tensorus.models import Name
FORMAT: Literal["tensorus-upload-attempt-observations-v1"] = (
    "tensorus-upload-attempt-observations-v1"
)
IDENTITY_FORMAT: Literal["tensorus-upload-observation-identity-v1"] = (
    "tensorus-upload-observation-identity-v1"
)
ERROR_FORMAT: Literal["tensorus-upload-finalizer-error-v1"] = "tensorus-upload-finalizer-error-v1"
MAX_PAGE_SIZE: Literal[50] = 50
MAX_RESPONSE_BYTES: Literal[32768] = 32768
Generation = Annotated[StrictInt, Field(ge=1, le=2**63 - 1)]
Size = Annotated[StrictInt, Field(ge=1, le=2**63 - 1)]
Timestamp = Annotated[str, Field(strict=True, min_length=1, max_length=40)]
ErrorCategory = Literal["authorization", "capacity", "conflict", "request", "storage", "unexpected"]
def _catalog_timestamp(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Upload observation timestamp must be a string")
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() != timedelta(0) or parsed.isoformat(timespec="microseconds") != value:
        raise ValueError("Upload observation timestamp must be canonical UTC")
    return value
class UploadAttemptEvent(Model):
    event_id: UUID
    recorded_at: Timestamp
    kind: Literal["claimed", "released", "error", "finalized", "cancelled"]
    generation: Generation | None
    expired_takeover: bool | None
    error_category: ErrorCategory | None
    version_id: UUID | None

    @field_validator("recorded_at")
    @classmethod
    def canonical_recorded_at(cls, value: str) -> str:
        return _catalog_timestamp(value)

    @field_validator("expired_takeover", mode="before")
    @classmethod
    def exact_takeover(cls, value: object) -> object:
        if value is not None and type(value) is not bool:
            raise ValueError("Takeover must be a boolean")
        return value

    @model_validator(mode="after")
    def consistent_kind(self) -> Self:
        if (
            (self.generation is not None) != (self.kind in {"claimed", "released", "error"})
            or (self.expired_takeover is not None) != (self.kind == "claimed")
            or (self.error_category is not None) != (self.kind == "error")
            or (self.version_id is not None) != (self.kind == "finalized")
        ):
            raise ValueError("Upload attempt event fields do not match its kind")
        return self
class UploadAttemptObservations(Model):
    format: Literal["tensorus-upload-attempt-observations-v1"] = FORMAT
    tenant_id: UUID
    upload_id: UUID
    created_by: UUID
    dataset_id: UUID
    dataset: Name
    name: Name
    declared_npy_bytes: Size
    creation_event_id: UUID
    identity_source: Literal["pinned_audit", "retained_session"]
    identity_event_id: UUID | None
    limit: Annotated[StrictInt, Field(ge=1, le=MAX_PAGE_SIZE)]
    after: UUID | None
    items: list[UploadAttemptEvent] = Field(max_length=MAX_PAGE_SIZE)
    next_after: UUID | None
    history_complete: Literal[False] = False
    pre_claim_rejections_included: Literal[False] = False
    expiry_and_revocation_included: Literal[False] = False
    failure_outcomes_complete: Literal[False] = False
    current_session_state_inferred: Literal[False] = False
    physical_usage_measured: Literal[False] = False
    billing_authority: Literal[False] = False
    source_independently_authenticated: Literal[False] = False

    @field_validator(
        "history_complete",
        "pre_claim_rejections_included",
        "expiry_and_revocation_included",
        "failure_outcomes_complete",
        "current_session_state_inferred",
        "physical_usage_measured",
        "billing_authority",
        "source_independently_authenticated",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Upload observation flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_page(self) -> Self:
        keys = [(item.recorded_at, str(item.event_id)) for item in self.items]
        if (
            (self.identity_source == "pinned_audit") != (self.identity_event_id is not None)
            or len(self.items) > self.limit
            or keys != sorted(set(keys))
            or len({item.event_id for item in self.items}) != len(self.items)
            or any(item.event_id == self.after for item in self.items)
            or (
                self.next_after is not None
                and (
                    len(self.items) != self.limit
                    or not self.items
                    or self.next_after != self.items[-1].event_id
                )
            )
        ):
            raise ValueError("Upload observation page is inconsistent")
        return self
class UploadAttemptCapabilities(Model):
    format: Literal["tensorus-upload-attempt-observations-v1"] = FORMAT
    identity_format: Literal["tensorus-upload-observation-identity-v1"] = IDENTITY_FORMAT
    error_format: Literal["tensorus-upload-finalizer-error-v1"] = ERROR_FORMAT
    max_page_size: Literal[50] = MAX_PAGE_SIZE
    max_response_bytes: Literal[32768] = MAX_RESPONSE_BYTES
    shared_scientific_read_admission_enabled: bool
    creator_only: Literal[True] = True
    dataset_ingest_permission_required: Literal[True] = True
    legacy_post_cleanup_readable: Literal[False] = False
    history_complete: Literal[False] = False
    pre_claim_rejections_included: Literal[False] = False
    expiry_and_revocation_included: Literal[False] = False
    failure_outcomes_complete: Literal[False] = False
    physical_usage_measured: Literal[False] = False
    billing_authority: Literal[False] = False
    source_independently_authenticated: Literal[False] = False
    delivery_atomic: Literal[False] = False

    @field_validator("max_page_size", "max_response_bytes", mode="before")
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Upload observation limits must be integers")
        return value

    @field_validator(
        "shared_scientific_read_admission_enabled",
        "creator_only",
        "dataset_ingest_permission_required",
        "legacy_post_cleanup_readable",
        "history_complete",
        "pre_claim_rejections_included",
        "expiry_and_revocation_included",
        "failure_outcomes_complete",
        "physical_usage_measured",
        "billing_authority",
        "source_independently_authenticated",
        "delivery_atomic",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Upload observation flags must be booleans")
        return value
