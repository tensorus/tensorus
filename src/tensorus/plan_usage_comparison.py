"""Compare compact, independently retained claims about plan usage evidence."""

from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, StrictInt, ValidationError, model_validator

from tensorus.arrays import Record, canonical_json, decode_unique_json
from tensorus.contracts import ReleaseHash
from tensorus.errors import Conflict, LimitExceeded, TensorusError
from tensorus.models import Model, PlanStatus
from tensorus.plan_usage_adjusted_handoff import MAX_ASSERTIONS, AdjustedHandoffDocument
from tensorus.plan_usage_handoff import MAX_EVENTS

REFERENCE_FORMAT: Literal["tensorus-plan-usage-reference-v1"] = "tensorus-plan-usage-reference-v1"
COMPARISON_FORMAT: Literal["tensorus-plan-usage-comparison-v1"] = (
    "tensorus-plan-usage-comparison-v1"
)
MAX_REFERENCE_BYTES: Literal[16384] = 16384
_TERMINAL = {PlanStatus.succeeded, PlanStatus.failed, PlanStatus.cancelled}


class EvidenceIdentity(Model):
    event_id: ReleaseHash
    payload_sha256: ReleaseHash


class UsageEvidenceReference(Model):
    format: Literal["tensorus-plan-usage-reference-v1"] = REFERENCE_FORMAT
    tenant_id: UUID
    plan_id: UUID
    plan_hash: ReleaseHash
    plan_status: PlanStatus
    plan_attempts: Annotated[StrictInt, Field(ge=0, le=10)]
    document_sha256: ReleaseHash
    observations: list[EvidenceIdentity] = Field(max_length=MAX_EVENTS)
    assertions: list[EvidenceIdentity] = Field(max_length=MAX_ASSERTIONS)

    @model_validator(mode="after")
    def sorted_identities(self) -> Self:
        if self.plan_status not in _TERMINAL:
            raise ValueError("A usage evidence reference requires a terminal plan")
        for entries in (self.observations, self.assertions):
            ids = [item.event_id for item in entries]
            if ids != sorted(set(ids)):
                raise ValueError("Usage evidence identities must be sorted and unique")
        return self


def read_reference(data: bytes) -> UsageEvidenceReference:
    if not isinstance(data, bytes):
        raise TypeError("Usage evidence reference must be bytes")
    if len(data) > MAX_REFERENCE_BYTES:
        raise LimitExceeded("Usage evidence reference exceeds its byte limit")
    try:
        decoded = decode_unique_json(data)
        checked = UsageEvidenceReference.model_validate_json(data, strict=True)
        if not isinstance(decoded, dict) or canonical_json(decoded) != canonical_json(
            checked.model_dump(mode="json")
        ):
            raise ValueError("Usage evidence reference is not exact")
        return checked
    except (
        ValidationError,
        TensorusError,
        TypeError,
        ValueError,
        UnicodeError,
        RecursionError,
    ) as exc:
        raise TensorusError(
            "The usage evidence reference is invalid", code="invalid_usage_reference"
        ) from exc


def reference_from_document(
    document: AdjustedHandoffDocument, document_sha256: str
) -> UsageEvidenceReference:
    usage = document.original.usage
    return UsageEvidenceReference(
        tenant_id=usage.tenant_id,
        plan_id=usage.plan_id,
        plan_hash=usage.plan_hash,
        plan_status=usage.plan_status,
        plan_attempts=usage.plan_attempts,
        document_sha256=document_sha256,
        observations=sorted(
            (
                EvidenceIdentity(event_id=item.event_id, payload_sha256=item.payload_sha256)
                for item in document.original.events
            ),
            key=lambda item: item.event_id,
        ),
        assertions=sorted(
            (
                EvidenceIdentity(event_id=item.event_id, payload_sha256=item.payload_sha256)
                for item in document.assertion_identities
            ),
            key=lambda item: item.event_id,
        ),
    )


def _changes(previous: list[EvidenceIdentity], current: list[EvidenceIdentity]) -> list[Record]:
    old = {item.event_id: item.payload_sha256 for item in previous}
    new = {item.event_id: item.payload_sha256 for item in current}
    result: list[Record] = []
    for event_id in sorted(old.keys() | new.keys()):
        before, after = old.get(event_id), new.get(event_id)
        if before is None:
            classification = "new"
        elif after is None:
            classification = "missing"
        elif before != after:
            classification = "changed"
        else:
            classification = "unchanged"
        result.append(
            {
                "event_id": event_id,
                "classification": classification,
                "previous_payload_sha256": before,
                "current_payload_sha256": after,
            }
        )
    return result


def compare_references(previous: UsageEvidenceReference, current: UsageEvidenceReference) -> Record:
    if any(
        getattr(previous, field) != getattr(current, field)
        for field in ("tenant_id", "plan_id", "plan_hash")
    ):
        raise Conflict("Usage evidence references identify different tenants or plans")
    observations = _changes(previous.observations, current.observations)
    assertions = _changes(previous.assertions, current.assertions)
    return {
        "format": COMPARISON_FORMAT,
        "tenant_id": str(current.tenant_id),
        "plan_id": str(current.plan_id),
        "plan_hash": current.plan_hash,
        "previous_document_sha256": previous.document_sha256,
        "current_document_sha256": current.document_sha256,
        "previous_plan_status": previous.plan_status.value,
        "current_plan_status": current.plan_status.value,
        "previous_plan_attempts": previous.plan_attempts,
        "current_plan_attempts": current.plan_attempts,
        "plan_status_changed": previous.plan_status != current.plan_status,
        "plan_attempts_changed": previous.plan_attempts != current.plan_attempts,
        "previous_entries_missing_or_changed": any(
            item["classification"] in {"missing", "changed"} for item in observations + assertions
        ),
        "observations": observations,
        "assertions": assertions,
        "snapshots_atomic": False,
        "prior_reference_authenticated": False,
        "current_reference_authenticated": False,
        "current_observations_independently_authenticated": False,
        "current_assertions_independently_verified": False,
        "automatic_reconciliation": False,
        "provider_submission": False,
        "charges_created": False,
        "billing_authority": False,
        "restored_sources_fenced": False,
        "missing_observations_are_zero": False,
    }
