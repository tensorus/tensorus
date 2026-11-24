"""Provider-neutral copies of existing plan usage observations."""
import hashlib
from typing import Annotated
from typing import Literal
from typing import Self
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from tensorus.arrays import Record
from tensorus.arrays import canonical_hash
from tensorus.arrays import canonical_json
from tensorus.arrays import decode_unique_json
from tensorus.contracts import ReleaseHash
from tensorus.errors import TensorusError
from tensorus.models import Model
from tensorus.plan_process_usage import PlanProcessUsage
from tensorus.plan_usage import PlanUsageObservations
FORMAT: Literal["tensorus-plan-usage-handoff-v1"] = "tensorus-plan-usage-handoff-v1"
DOCUMENT_FORMAT: Literal["tensorus-plan-usage-handoff-document-v1"] = (
    "tensorus-plan-usage-handoff-document-v1"
)
EVENT_ID_FORMAT: Literal["tensorus-plan-usage-event-identity-v1"] = (
    "tensorus-plan-usage-event-identity-v1"
)
MAX_DOCUMENT_BYTES: Literal[65536] = 65536
MAX_RESPONSE_BYTES: Literal[131072] = 131072
MAX_EVENTS: Literal[36] = 36
EventKind = Literal["logical_output", "wall_attempt", "process_attempt"]
class HandoffEvent(Model):
    kind: EventKind
    identity: Annotated[str, Field(strict=True, min_length=1, max_length=32)]
    event_id: ReleaseHash
    payload_sha256: ReleaseHash
    observation_json: Annotated[str, Field(strict=True, min_length=1, max_length=2048)]
def _event(
    usage: PlanUsageObservations, kind: EventKind, identity: str, observation: Record
) -> HandoffEvent:
    observation_json = canonical_json(observation)
    return HandoffEvent(
        kind=kind,
        identity=identity,
        event_id=canonical_hash(
            {
                "format": EVENT_ID_FORMAT,
                "tenant_id": str(usage.tenant_id),
                "plan_id": str(usage.plan_id),
                "plan_hash": usage.plan_hash,
                "kind": kind,
                "identity": identity,
            }
        ),
        payload_sha256=hashlib.sha256(observation_json.encode("utf-8")).hexdigest(),
        observation_json=observation_json,
    )
def _events(usage: PlanUsageObservations, process: PlanProcessUsage) -> list[HandoffEvent]:
    result = [
        _event(usage, "logical_output", item.step_id, item.model_dump(mode="json"))
        for item in usage.outputs
    ]
    result.extend(
        _event(usage, "wall_attempt", str(item.attempt), item.model_dump(mode="json"))
        for item in usage.attempts
    )
    result.extend(
        _event(usage, "process_attempt", str(item.attempt), item.model_dump(mode="json"))
        for item in process.attempts
    )
    return result
class HandoffDocument(Model):
    format: Literal["tensorus-plan-usage-handoff-document-v1"] = DOCUMENT_FORMAT
    usage: PlanUsageObservations
    process: PlanProcessUsage
    events: list[HandoffEvent] = Field(max_length=MAX_EVENTS)
    snapshots_atomic: Literal[False] = False
    observations_independently_authenticated: Literal[False] = False
    automatic_reconciliation: Literal[False] = False
    provider_submission: Literal[False] = False
    charges_created: Literal[False] = False
    billing_authority: Literal[False] = False
    restored_sources_fenced: Literal[False] = False
    missing_observations_are_zero: Literal[False] = False

    @field_validator(
        "snapshots_atomic",
        "observations_independently_authenticated",
        "automatic_reconciliation",
        "provider_submission",
        "charges_created",
        "billing_authority",
        "restored_sources_fenced",
        "missing_observations_are_zero",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Usage handoff flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        if any(
            getattr(self.usage, field) != getattr(self.process, field)
            for field in ("tenant_id", "plan_id", "plan_hash", "plan_status", "plan_attempts")
        ):
            raise ValueError("Usage handoff observations identify different plans")
        wall = {item.attempt: item.outcome for item in self.usage.attempts}
        if any(wall.get(item.attempt) != item.outcome for item in self.process.attempts):
            raise ValueError("Process observations do not match terminal wall observations")
        expected = _events(self.usage, self.process)
        if self.events != expected or len({event.event_id for event in self.events}) != len(
            self.events
        ):
            raise ValueError("Usage handoff event identities or payloads are inconsistent")
        return self
class PlanUsageHandoff(Model):
    format: Literal["tensorus-plan-usage-handoff-v1"] = FORMAT
    document_json: Annotated[str, Field(strict=True, min_length=1, max_length=MAX_DOCUMENT_BYTES)]
    document_sha256: ReleaseHash

    @model_validator(mode="after")
    def checked_document(self) -> Self:
        raw = self.document_json.encode("utf-8")
        if len(raw) > MAX_DOCUMENT_BYTES:
            raise ValueError("Usage handoff document exceeds its byte limit")
        try:
            decoded = decode_unique_json(raw)
            checked = HandoffDocument.model_validate_json(raw, strict=True)
            if (
                canonical_json(decoded) != self.document_json
                or canonical_json(checked.model_dump(mode="json")) != self.document_json
                or hashlib.sha256(raw).hexdigest() != self.document_sha256
            ):
                raise ValueError("Usage handoff document is not exact")
        except (TensorusError, TypeError, ValueError, UnicodeError, RecursionError) as exc:
            raise ValueError("Usage handoff document is invalid") from exc
        return self
class PlanUsageHandoffCapabilities(Model):
    format: Literal["tensorus-plan-usage-handoff-v1"] = FORMAT
    document_format: Literal["tensorus-plan-usage-handoff-document-v1"] = DOCUMENT_FORMAT
    event_identity_format: Literal["tensorus-plan-usage-event-identity-v1"] = EVENT_ID_FORMAT
    max_document_bytes: Literal[65536] = MAX_DOCUMENT_BYTES
    max_response_bytes: Literal[131072] = MAX_RESPONSE_BYTES
    max_events: Literal[36] = MAX_EVENTS
    snapshots_atomic: Literal[False] = False
    observations_independently_authenticated: Literal[False] = False
    automatic_reconciliation: Literal[False] = False
    provider_submission: Literal[False] = False
    charges_created: Literal[False] = False
    billing_authority: Literal[False] = False
    restored_sources_fenced: Literal[False] = False
    missing_observations_are_zero: Literal[False] = False

    @field_validator("max_document_bytes", "max_response_bytes", "max_events", mode="before")
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Usage handoff limits must be integers")
        return value

    @field_validator(
        "snapshots_atomic",
        "observations_independently_authenticated",
        "automatic_reconciliation",
        "provider_submission",
        "charges_created",
        "billing_authority",
        "restored_sources_fenced",
        "missing_observations_are_zero",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Usage handoff flags must be booleans")
        return value
