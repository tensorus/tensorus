"""Audited administrator assertions alongside, never in place of, measured usage."""
import re
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
from tensorus.contracts import ReleaseHash
from tensorus.models import Model
from tensorus.plan_usage import MAX_ELAPSED_MS
from tensorus.plan_usage import PlanUsageObservations
from tensorus.plan_usage_handoff import HandoffEvent
from tensorus.plan_usage_handoff import _event
FORMAT: Literal["tensorus-plan-usage-adjustments-v1"] = "tensorus-plan-usage-adjustments-v1"
EVENT_FORMAT: Literal["tensorus-plan-usage-adjustment-event-v1"] = (
    "tensorus-plan-usage-adjustment-event-v1"
)
RECEIPT_FORMAT: Literal["tensorus-plan-usage-adjustment-receipt-v1"] = (
    "tensorus-plan-usage-adjustment-receipt-v1"
)
MAX_ADJUSTMENTS: Literal[32] = 32
MAX_RESPONSE_BYTES: Literal[262144] = 262144
_TERMINAL = {"succeeded", "failed", "cancelled"}
DecimalValue = Annotated[
    str, Field(strict=True, min_length=1, max_length=19, pattern=r"^(0|[1-9][0-9]*)$")
]
def _decimal(value: str) -> str:
    if not re.fullmatch(r"(?:0|[1-9][0-9]{0,18})", value) or int(value) > 2**63 - 1:
        raise ValueError("An asserted value must be a canonical supported decimal integer")
    return value
def _timestamp(value: str) -> str:
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() != timedelta(0) or parsed.isoformat(timespec="microseconds") != value:
        raise ValueError("A canonical UTC timestamp is required")
    return value
class AdjustmentRequest(Model):
    request_id: UUID
    expected_plan_hash: ReleaseHash
    source_event_id: ReleaseHash
    source_payload_sha256: ReleaseHash
    expected_generation: Annotated[StrictInt, Field(ge=0, lt=MAX_ADJUSTMENTS)]
    expected_value: DecimalValue
    asserted_value: DecimalValue
    reason: Annotated[str, Field(strict=True, min_length=1, max_length=512)]
    confirmation: Literal["ASSERT"]

    _values = field_validator("expected_value", "asserted_value")(_decimal)

    @field_validator("reason")
    @classmethod
    def checked_reason(cls, value: str) -> str:
        if value != value.strip() or not value.isprintable():
            raise ValueError("A trimmed, printable reason is required")
        return value
class AdjustmentEvent(Model):
    format: Literal["tensorus-plan-usage-adjustment-event-v1"] = EVENT_FORMAT
    tenant_id: UUID
    plan_id: UUID
    actor_id: Annotated[str, Field(strict=True, min_length=1, max_length=36)]
    created_at: str
    generation: Annotated[StrictInt, Field(ge=1, le=MAX_ADJUSTMENTS)]
    previous_request_id: UUID | None
    request: AdjustmentRequest

    _created_at = field_validator("created_at")(_timestamp)

    @field_validator("actor_id")
    @classmethod
    def checked_actor(cls, value: str) -> str:
        if value != "embedded" and str(UUID(value)) != value:
            raise ValueError("Actor ID must be canonical")
        return value

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (
            self.generation != self.request.expected_generation + 1
            or (self.generation == 1) != (self.previous_request_id is None)
            or self.request.expected_value == self.request.asserted_value
        ):
            raise ValueError("Adjustment event is inconsistent")
        return self
class AdjustmentObservation(Model):
    source: HandoffEvent
    observed_value: DecimalValue
    asserted_value: DecimalValue
    generation: Annotated[StrictInt, Field(ge=0, le=MAX_ADJUSTMENTS)]
    latest_request_id: UUID | None

    _values = field_validator("observed_value", "asserted_value")(_decimal)
def _sources(usage: PlanUsageObservations) -> list[tuple[HandoffEvent, str]]:
    result = [
        (
            _event(usage, "logical_output", item.step_id, item.model_dump(mode="json")),
            str(item.logical_bytes),
        )
        for item in usage.outputs
    ]
    result.extend(
        (
            _event(usage, "wall_attempt", str(item.attempt), item.model_dump(mode="json")),
            str(item.elapsed_ms),
        )
        for item in usage.attempts
    )
    return result
def _replay(
    usage: PlanUsageObservations, adjustments: list[AdjustmentEvent]
) -> list[AdjustmentObservation]:
    sources = _sources(usage)
    states = {
        source.event_id: AdjustmentObservation(
            source=source,
            observed_value=value,
            asserted_value=value,
            generation=0,
            latest_request_id=None,
        )
        for source, value in sources
    }
    if len(states) != len(sources) or len({event.request.request_id for event in adjustments}) != (
        len(adjustments)
    ):
        raise ValueError("Adjustment identities are duplicated")
    if adjustments and usage.plan_status not in _TERMINAL:
        raise ValueError("Adjusted plan is not terminal")
    ordered = sorted(
        adjustments, key=lambda event: (event.request.source_event_id, event.generation)
    )
    if ordered != adjustments:
        raise ValueError("Adjustment history is not ordered")
    for event in adjustments:
        request = event.request
        state = states.get(request.source_event_id)
        if (
            event.tenant_id != usage.tenant_id
            or event.plan_id != usage.plan_id
            or request.expected_plan_hash != usage.plan_hash
            or state is None
            or request.source_payload_sha256 != state.source.payload_sha256
            or request.expected_generation != state.generation
            or request.expected_value != state.asserted_value
            or event.previous_request_id != state.latest_request_id
            or (
                state.source.kind == "wall_attempt" and int(request.asserted_value) > MAX_ELAPSED_MS
            )
        ):
            raise ValueError("Adjustment does not match its observed source or prior assertion")
        states[request.source_event_id] = state.model_copy(
            update={
                "asserted_value": request.asserted_value,
                "generation": event.generation,
                "latest_request_id": request.request_id,
            }
        )
    return [states[source.event_id] for source, _ in sources]
class AdjustmentReport(Model):
    format: Literal["tensorus-plan-usage-adjustments-v1"] = FORMAT
    usage: PlanUsageObservations
    observations: list[AdjustmentObservation] = Field(max_length=26)
    adjustments: list[AdjustmentEvent] = Field(max_length=MAX_ADJUSTMENTS)
    assertions_independently_verified: Literal[False] = False
    source_measurements_changed: Literal[False] = False
    provider_submission: Literal[False] = False
    charges_created: Literal[False] = False
    billing_authority: Literal[False] = False
    restored_sources_fenced: Literal[False] = False
    missing_observations_are_zero: Literal[False] = False

    @field_validator(
        "assertions_independently_verified",
        "source_measurements_changed",
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
            raise ValueError("Adjustment flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.observations != _replay(self.usage, self.adjustments):
            raise ValueError("Adjustment report history is inconsistent")
        return self
class AdjustmentReceipt(Model):
    format: Literal["tensorus-plan-usage-adjustment-receipt-v1"] = RECEIPT_FORMAT
    replayed: bool = Field(strict=True)
    event: AdjustmentEvent
    current: AdjustmentReport

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.event not in self.current.adjustments or (
            not self.replayed
            and not any(
                item.latest_request_id == self.event.request.request_id
                for item in self.current.observations
            )
        ):
            raise ValueError("Adjustment receipt does not match current history")
        return self
class AdjustmentCapabilities(Model):
    format: Literal["tensorus-plan-usage-adjustments-v1"] = FORMAT
    event_format: Literal["tensorus-plan-usage-adjustment-event-v1"] = EVENT_FORMAT
    receipt_format: Literal["tensorus-plan-usage-adjustment-receipt-v1"] = RECEIPT_FORMAT
    authority: Literal["active_unscoped_tenant_admin"] = "active_unscoped_tenant_admin"
    max_adjustments_per_plan: Literal[32] = MAX_ADJUSTMENTS
    max_response_bytes: Literal[262144] = MAX_RESPONSE_BYTES
    terminal_plans_only: Literal[True] = True
    exact_source_and_prior_required: Literal[True] = True
    assertions_independently_verified: Literal[False] = False
    source_measurements_changed: Literal[False] = False
    provider_submission: Literal[False] = False
    charges_created: Literal[False] = False
    billing_authority: Literal[False] = False
    restored_sources_fenced: Literal[False] = False
    missing_observations_are_zero: Literal[False] = False

    @field_validator("max_adjustments_per_plan", "max_response_bytes", mode="before")
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Adjustment limits must be integers")
        return value

    @field_validator(
        "terminal_plans_only",
        "exact_source_and_prior_required",
        "assertions_independently_verified",
        "source_measurements_changed",
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
            raise ValueError("Adjustment flags must be booleans")
        return value
