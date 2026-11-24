"""Plan-scoped views of the existing delegated execution allowance evidence."""
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictBool
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_validator
from tensorus.agent_execution_budget import ESTIMATOR
from tensorus.agent_execution_budget import MAX_UNITS
from tensorus.arrays import canonical_hash
from tensorus.contracts import ReleaseHash
from tensorus.models import Model
from tensorus.models import PlanStatus
FORMAT: Literal["tensorus-plan-estimated-attempts-v1"] = "tensorus-plan-estimated-attempts-v1"
IDENTITY_FORMAT: Literal["tensorus-plan-estimated-event-identity-v1"] = (
    "tensorus-plan-estimated-event-identity-v1"
)
MAX_RESPONSE_BYTES: Literal[16384] = 16384
Attempt = Annotated[StrictInt, Field(ge=1, le=10)]
Units = Annotated[StrictInt, Field(ge=1, le=MAX_UNITS)]
Timestamp = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
def event_identity(
    tenant_id: UUID, plan_id: UUID, plan_hash: str, kind: Literal["estimate", "attempt"], slot: str
) -> str:
    return canonical_hash(
        {
            "format": IDENTITY_FORMAT,
            "tenant_id": str(tenant_id),
            "plan_id": str(plan_id),
            "plan_hash": plan_hash,
            "kind": kind,
            "identity": slot,
        }
    )
class PlanEstimate(Model):
    estimator: Literal["tensorus-plan-preflight-estimate-v1"] = ESTIMATOR
    estimated_units: Units
    created_at: Annotated[str, Field(strict=True, min_length=1, max_length=40)]
    event_id: ReleaseHash
class EstimatedAttempt(Model):
    attempt: Attempt
    estimated_units: Units
    claimed_at: Timestamp
    ended_at: Timestamp | None
    outcome: Literal["running", "succeeded", "failed", "cancelled", "lease_expired"]
    event_id: ReleaseHash

    @model_validator(mode="after")
    def consistent_outcome(self) -> Self:
        if (self.outcome == "running") != (self.ended_at is None) or (
            self.ended_at is not None and self.ended_at < self.claimed_at
        ):
            raise ValueError("Estimated attempt outcome is inconsistent")
        return self
class PlanEstimatedAttempts(Model):
    format: Literal["tensorus-plan-estimated-attempts-v1"] = FORMAT
    tenant_id: UUID
    plan_id: UUID
    plan_hash: ReleaseHash
    plan_status: PlanStatus
    plan_attempts: Annotated[StrictInt, Field(ge=0, le=10)]
    estimate: PlanEstimate | None
    attempts: list[EstimatedAttempt] = Field(max_length=10)
    unrecorded_attempts: list[Attempt] = Field(max_length=10)
    attempted_units: Annotated[StrictInt, Field(ge=0, le=MAX_UNITS)] | None
    missing_estimate_is_zero: Literal[False] = False
    failure_refunds_estimate: Literal[False] = False
    claims_prove_execution: Literal[False] = False
    resource_use_measured: Literal[False] = False
    billing_authority: Literal[False] = False
    independently_authenticated: Literal[False] = False
    restored_sources_fenced: Literal[False] = False
    response_delivery_atomic: Literal[False] = False

    @field_validator(
        "missing_estimate_is_zero",
        "failure_refunds_estimate",
        "claims_prove_execution",
        "resource_use_measured",
        "billing_authority",
        "independently_authenticated",
        "restored_sources_fenced",
        "response_delivery_atomic",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Estimated-attempt flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        recorded = [item.attempt for item in self.attempts]
        if (
            recorded != sorted(set(recorded))
            or any(number > self.plan_attempts for number in recorded)
            or any(
                item.outcome == "running"
                and (self.plan_status != PlanStatus.running or item.attempt != self.plan_attempts)
                for item in self.attempts
            )
            or self.unrecorded_attempts
            != sorted(set(range(1, self.plan_attempts + 1)) - set(recorded))
        ):
            raise ValueError("Estimated attempt coverage is inconsistent")
        if self.estimate is None:
            if self.attempts or self.attempted_units is not None:
                raise ValueError("Missing estimate is not zero")
            return self
        if (
            self.unrecorded_attempts
            or self.estimate.event_id
            != event_identity(self.tenant_id, self.plan_id, self.plan_hash, "estimate", "estimate")
            or self.attempted_units != sum(item.estimated_units for item in self.attempts)
            or any(
                item.estimated_units != self.estimate.estimated_units
                or item.event_id
                != event_identity(
                    self.tenant_id, self.plan_id, self.plan_hash, "attempt", str(item.attempt)
                )
                for item in self.attempts
            )
        ):
            raise ValueError("Estimated attempt identities or units are inconsistent")
        return self
class PlanEstimatedAttemptsCapabilities(Model):
    format: Literal["tensorus-plan-estimated-attempts-v1"] = FORMAT
    estimator: Literal["tensorus-plan-preflight-estimate-v1"] = ESTIMATOR
    event_identity_format: Literal["tensorus-plan-estimated-event-identity-v1"] = IDENTITY_FORMAT
    max_plan_attempts: Literal[10] = 10
    max_response_bytes: Literal[16384] = MAX_RESPONSE_BYTES
    missing_estimate_is_zero: Literal[False] = False
    failure_refunds_estimate: Literal[False] = False
    claims_prove_execution: Literal[False] = False
    resource_use_measured: Literal[False] = False
    billing_authority: Literal[False] = False
    independently_authenticated: Literal[False] = False
    restored_sources_fenced: Literal[False] = False
    response_delivery_atomic: Literal[False] = False
    scientific_read_admission_v11_covered: Literal[False] = False
    separate_scientific_read_admission_enabled: StrictBool = False

    @field_validator("max_plan_attempts", "max_response_bytes", mode="before")
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Estimated-attempt limits must be integers")
        return value

    @field_validator(
        "missing_estimate_is_zero",
        "failure_refunds_estimate",
        "claims_prove_execution",
        "resource_use_measured",
        "billing_authority",
        "independently_authenticated",
        "restored_sources_fenced",
        "response_delivery_atomic",
        "scientific_read_admission_v11_covered",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Estimated-attempt capability flags must be booleans")
        return value
