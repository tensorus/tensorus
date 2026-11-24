"""Catalog evidence for plan-attempt claims and outcomes, not metered billing."""

from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, StrictBool, StrictInt, field_validator, model_validator

from tensorus.models import Model

FORMAT: Literal["tensorus-worker-claim-accounting-v1"] = "tensorus-worker-claim-accounting-v1"
MAX_RESPONSE_BYTES = 16 * 1024
PlanHash = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
Attempt = Annotated[StrictInt, Field(ge=1, le=10)]
Timestamp = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class WorkerClaim(Model):
    attempt: Attempt
    claimed_at: Timestamp
    ended_at: Timestamp | None
    outcome: Literal["running", "succeeded", "failed", "cancelled", "lease_expired"]
    current_lease: StrictBool

    @model_validator(mode="after")
    def consistent_outcome(self) -> Self:
        if (
            (self.outcome == "running") != (self.ended_at is None)
            or (self.ended_at is not None and self.ended_at < self.claimed_at)
            or (self.current_lease and self.outcome != "running")
        ):
            raise ValueError("Worker claim outcome is inconsistent")
        return self


class WorkerClaimAccounting(Model):
    format: Literal["tensorus-worker-claim-accounting-v1"]
    plan_id: UUID
    plan_hash: PlanHash
    plan_attempts: Annotated[StrictInt, Field(ge=0, le=10)]
    claims: list[WorkerClaim] = Field(max_length=10)
    unrecorded_attempts: list[Attempt] = Field(max_length=10)
    claims_prove_execution: Literal[False]
    actual_resource_use_measured: Literal[False]
    billing_authority: Literal[False]

    @field_validator(
        "claims_prove_execution", "actual_resource_use_measured", "billing_authority", mode="before"
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Worker claim accounting flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_attempts(self) -> Self:
        recorded = [item.attempt for item in self.claims]
        if (
            recorded != sorted(set(recorded))
            or any(attempt > self.plan_attempts for attempt in recorded)
            or self.unrecorded_attempts
            != sorted(set(range(1, self.plan_attempts + 1)) - set(recorded))
            or sum(item.current_lease for item in self.claims) > 1
            or any(
                item.current_lease and item.attempt != self.plan_attempts for item in self.claims
            )
        ):
            raise ValueError("Worker claim accounting is inconsistent")
        return self


class WorkerClaimCapabilities(Model):
    format: Literal["tensorus-worker-claim-accounting-v1"]
    max_plan_attempts: Literal[10]
    legacy_attempts_may_be_unrecorded: Literal[True]
    resource_use_measured: Literal[False]
    billing_authority: Literal[False]

    @field_validator("max_plan_attempts", mode="before")
    @classmethod
    def exact_limit(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Worker claim attempt limit must be an integer")
        return value

    @field_validator(
        "legacy_attempts_may_be_unrecorded",
        "resource_use_measured",
        "billing_authority",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Worker claim capability flags must be booleans")
        return value


class WorkerCancellationRecoveryCapabilities(Model):
    format: Literal["tensorus-worker-cancellation-recovery-v1"] = (
        "tensorus-worker-cancellation-recovery-v1"
    )
    implementation: Literal["bundled_worker_code"] = "bundled_worker_code"
    expired_cancelled_claim_retried: Literal[False] = False
    expired_recorded_claim_outcome: Literal["lease_expired"] = "lease_expired"
    estimated_allowance_refunded: Literal[False] = False
    legacy_claims_invented: Literal[False] = False
    stale_process_termination_proven: Literal[False] = False
    external_workers_verified: Literal[False] = False

    @field_validator(
        "expired_cancelled_claim_retried",
        "estimated_allowance_refunded",
        "legacy_claims_invented",
        "stale_process_termination_proven",
        "external_workers_verified",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Worker cancellation recovery flags must be booleans")
        return value


class WorkerCancellationRecoveryV2Capabilities(Model):
    format: Literal["tensorus-worker-cancellation-recovery-v2"] = (
        "tensorus-worker-cancellation-recovery-v2"
    )
    implementation: Literal["bundled_worker_code"] = "bundled_worker_code"
    expired_cancellation_eligible_at_lease_limit: Literal[True] = True
    configured_fair_limit_applies_to_new_attempts: Literal[True] = True
    immediate_settlement_guaranteed: Literal[False] = False
    stale_process_termination_proven: Literal[False] = False
    external_workers_verified: Literal[False] = False

    @field_validator(
        "expired_cancellation_eligible_at_lease_limit",
        "configured_fair_limit_applies_to_new_attempts",
        "immediate_settlement_guaranteed",
        "stale_process_termination_proven",
        "external_workers_verified",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Worker cancellation recovery flags must be booleans")
        return value
