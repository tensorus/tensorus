"""Catalog-recorded, unauthenticated evidence of bounded local worker claims."""

from datetime import datetime, timedelta
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, StrictBool, StrictInt, field_validator, model_validator

from tensorus.models import Model

ATTEMPT_ACTION = "plan.local_worker_attempt"
ATTEMPT_FORMAT: Literal["tensorus-local-worker-attempt-v1"] = "tensorus-local-worker-attempt-v1"
STATUS_FORMAT: Literal["tensorus-worker-attempt-evidence-v1"] = (
    "tensorus-worker-attempt-evidence-v1"
)
MAX_ATTEMPTS = 10
MAX_EVENT_TEXT_CHARS = 2048
MAX_RESPONSE_BYTES = 16 * 1024

PlanHash = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]


class LocalWorkerLimits(Model):
    worker_address_space_bytes: Annotated[StrictInt, Field(ge=128 * 1024**2, le=4 * 1024**3)]
    worker_cpu_seconds: Annotated[StrictInt, Field(ge=2, le=3601)]
    reported_manager_deadline_seconds: Annotated[StrictInt, Field(ge=1, le=3600)]
    fallback_wall_seconds: Annotated[StrictInt, Field(ge=2, le=3601)]
    kernel_address_space_bytes: Annotated[StrictInt, Field(ge=128 * 1024**2, le=4 * 1024**3)]
    local_eager_reads_enabled: StrictBool

    @model_validator(mode="after")
    def consistent_limits(self) -> Self:
        if (
            self.worker_cpu_seconds != self.reported_manager_deadline_seconds + 1
            or self.fallback_wall_seconds != self.reported_manager_deadline_seconds + 1
            or self.kernel_address_space_bytes > self.worker_address_space_bytes
        ):
            raise ValueError("Local worker limits are inconsistent")
        return self


class LocalWorkerAttemptEvent(Model):
    format: Literal["tensorus-local-worker-attempt-v1"]
    plan_hash: PlanHash
    attempt: Annotated[StrictInt, Field(ge=1, le=MAX_ATTEMPTS)]
    limits: LocalWorkerLimits


class RecordedWorkerAttempt(Model):
    attempt: Annotated[StrictInt, Field(ge=1, le=MAX_ATTEMPTS)]
    recorded_at: Annotated[str, Field(strict=True, min_length=1, max_length=40)]
    limits: LocalWorkerLimits

    @field_validator("recorded_at")
    @classmethod
    def utc_timestamp(cls, value: str) -> str:
        try:
            timestamp = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("Worker attempt timestamp is invalid") from exc
        if (
            timestamp.utcoffset() != timedelta(0)
            or timestamp.isoformat(timespec="microseconds") != value
        ):
            raise ValueError("Worker attempt timestamp must be canonical UTC")
        return value


class WorkerAttemptEvidence(Model):
    format: Literal["tensorus-worker-attempt-evidence-v1"]
    plan_id: UUID
    plan_hash: PlanHash
    plan_attempts: Annotated[StrictInt, Field(ge=0, le=MAX_ATTEMPTS)]
    recorded_attempts: list[RecordedWorkerAttempt] = Field(max_length=MAX_ATTEMPTS)
    unrecorded_attempts: list[Annotated[StrictInt, Field(ge=1, le=MAX_ATTEMPTS)]] = Field(
        max_length=MAX_ATTEMPTS
    )
    authentication: Literal["none"]

    @model_validator(mode="after")
    def consistent_attempts(self) -> Self:
        recorded = [item.attempt for item in self.recorded_attempts]
        if (
            recorded != sorted(set(recorded))
            or any(attempt > self.plan_attempts for attempt in recorded)
            or self.unrecorded_attempts
            != sorted(set(range(1, self.plan_attempts + 1)) - set(recorded))
        ):
            raise ValueError("Worker attempt evidence is inconsistent")
        return self


class WorkerAttemptEvidenceCapabilities(Model):
    format: Literal["tensorus-worker-attempt-evidence-v1"]
    record_format: Literal["tensorus-local-worker-attempt-v1"]
    max_plan_attempts: Literal[10]
    local_bounded_claims_only: StrictBool
    independently_authenticated: StrictBool

    @field_validator("max_plan_attempts", mode="before")
    @classmethod
    def exact_attempt_limit(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Worker attempt limit must be an integer")
        return value

    @model_validator(mode="after")
    def truthful_scope(self) -> Self:
        if not self.local_bounded_claims_only or self.independently_authenticated:
            raise ValueError("Worker attempt evidence scope is inconsistent")
        return self
