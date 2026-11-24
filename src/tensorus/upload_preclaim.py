"""Keyed, creator-scoped observations of selected upload pre-claim rejections."""
from typing import Literal
from typing import Self
from uuid import UUID
from uuid import uuid5
from pydantic import field_validator
from pydantic import model_validator
from tensorus.models import Model
from tensorus.models import Name
from tensorus.upload_observations import Timestamp
from tensorus.upload_observations import _catalog_timestamp
FORMAT: Literal["tensorus-upload-preclaim-rejection-v1"] = "tensorus-upload-preclaim-rejection-v1"
HEADER: Literal["X-Tensorus-Finalization-Attempt-ID"] = "X-Tensorus-Finalization-Attempt-ID"
MAX_PER_UPLOAD: Literal[100] = 100
MAX_RESPONSE_BYTES: Literal[4096] = 4096
Reason = Literal["incomplete", "closed", "busy", "generation_exhausted"]
def parse_attempt_id(value: str) -> UUID:
    attempt_id = UUID(value)
    if str(attempt_id) != value or attempt_id.int == 0:
        raise ValueError("A non-nil lowercase canonical attempt UUID is required")
    return attempt_id
class PreclaimRejection(Model):
    format: Literal["tensorus-upload-preclaim-rejection-v1"] = FORMAT
    event_id: UUID
    tenant_id: UUID
    created_by: UUID
    upload_id: UUID
    attempt_id: UUID
    dataset_id: UUID
    dataset: Name
    name: Name
    creation_event_id: UUID
    identity_source: Literal["pinned_audit", "retained_session"]
    identity_event_id: UUID | None
    recorded_at: Timestamp
    reason: Reason
    first_recorded_rejection_for_attempt: Literal[True] = True
    history_complete: Literal[False] = False
    recording_atomic_with_rejection: Literal[False] = False
    retry_replays_response: Literal[False] = False
    physical_usage_measured: Literal[False] = False
    billing_authority: Literal[False] = False
    source_independently_authenticated: Literal[False] = False

    @field_validator("recorded_at")
    @classmethod
    def canonical_recorded_at(cls, value: str) -> str:
        return _catalog_timestamp(value)

    @field_validator(
        "first_recorded_rejection_for_attempt",
        "history_complete",
        "recording_atomic_with_rejection",
        "retry_replays_response",
        "physical_usage_measured",
        "billing_authority",
        "source_independently_authenticated",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Preclaim observation flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_identity(self) -> Self:
        if (self.identity_source == "pinned_audit") != (
            self.identity_event_id is not None
        ) or self.event_id != rejection_id(
            self.tenant_id, self.created_by, self.upload_id, self.attempt_id
        ):
            raise ValueError("Preclaim observation identity is inconsistent")
        return self
class PreclaimCapabilities(Model):
    format: Literal["tensorus-upload-preclaim-rejection-v1"] = FORMAT
    attempt_header: Literal["X-Tensorus-Finalization-Attempt-ID"] = HEADER
    max_rejections_per_upload: Literal[100] = MAX_PER_UPLOAD
    max_response_bytes: Literal[4096] = MAX_RESPONSE_BYTES
    shared_scientific_read_admission_enabled: bool
    creator_only: Literal[True] = True
    dataset_ingest_permission_required: Literal[True] = True
    attempt_identity_client_supplied: Literal[True] = True
    first_recorded_rejection_for_attempt_retained: Literal[True] = True
    keyed_requests_only: Literal[True] = True
    legacy_post_cleanup_readable: Literal[False] = False
    all_preclaim_failures_recorded: Literal[False] = False
    preauthentication_failures_recorded: Literal[False] = False
    recording_atomic_with_rejection: Literal[False] = False
    retry_replays_response: Literal[False] = False
    physical_usage_measured: Literal[False] = False
    billing_authority: Literal[False] = False
    source_independently_authenticated: Literal[False] = False
    delivery_atomic: Literal[False] = False

    @field_validator("max_rejections_per_upload", "max_response_bytes", mode="before")
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Preclaim observation limits must be integers")
        return value

    @field_validator(
        "shared_scientific_read_admission_enabled",
        "creator_only",
        "dataset_ingest_permission_required",
        "attempt_identity_client_supplied",
        "first_recorded_rejection_for_attempt_retained",
        "keyed_requests_only",
        "legacy_post_cleanup_readable",
        "all_preclaim_failures_recorded",
        "preauthentication_failures_recorded",
        "recording_atomic_with_rejection",
        "retry_replays_response",
        "physical_usage_measured",
        "billing_authority",
        "source_independently_authenticated",
        "delivery_atomic",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Preclaim capability flags must be booleans")
        return value
def rejection_id(
    tenant_id: str | UUID, actor_id: str | UUID, upload_id: str | UUID, attempt_id: UUID
) -> UUID:
    tenant = UUID(str(tenant_id))
    actor = UUID(str(actor_id))
    upload = UUID(str(upload_id))
    return uuid5(tenant, f"{FORMAT}:{actor}:{upload}:{attempt_id}")
