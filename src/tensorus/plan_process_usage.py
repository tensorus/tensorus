"""Opt-in local worker process observations, not complete resource metering."""
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
from tensorus.models import PlanStatus
FORMAT: Literal["tensorus-plan-process-usage-v1"] = "tensorus-plan-process-usage-v1"
EVENT_FORMAT: Literal["tensorus-local-process-usage-observation-v1"] = (
    "tensorus-local-process-usage-observation-v1"
)
MAX_RESPONSE_BYTES: Literal[16384] = 16384
MAX_VALUE = 2**53 - 1
Attempt = Annotated[StrictInt, Field(ge=1, le=10)]
Measured = Annotated[StrictInt, Field(ge=0, le=MAX_VALUE)]
Outcome = Literal["succeeded", "failed", "cancelled"]
class ProcessObservation(Model):
    attempt: Attempt
    outcome: Outcome
    worker_cpu_ms: Measured
    reaped_child_cpu_ms: Measured
    worker_peak_rss_kib_since_start: Measured
    reaped_child_max_rss_kib_since_start: Measured
class PlanProcessUsage(Model):
    format: Literal["tensorus-plan-process-usage-v1"] = FORMAT
    tenant_id: UUID
    plan_id: UUID
    plan_hash: ReleaseHash
    observed_at: str = Field(min_length=1, max_length=40)
    plan_status: PlanStatus
    plan_attempts: Annotated[StrictInt, Field(ge=0, le=10)]
    attempts: list[ProcessObservation] = Field(max_length=10)
    unobserved_attempts: list[Attempt] = Field(max_length=10)
    linux_bounded_worker_only: Literal[True] = True
    child_cpu_reaped_in_window_only: Literal[True] = True
    rss_peaks_since_process_start: Literal[True] = True
    aggregate_rss_measured: Literal[False] = False
    manager_gpu_and_unreaped_children_measured: Literal[False] = False
    independently_authenticated: Literal[False] = False
    billing_authority: Literal[False] = False

    @field_validator(
        "linux_bounded_worker_only",
        "child_cpu_reaped_in_window_only",
        "rss_peaks_since_process_start",
        "aggregate_rss_measured",
        "manager_gpu_and_unreaped_children_measured",
        "independently_authenticated",
        "billing_authority",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Process usage flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_coverage(self) -> Self:
        recorded = [item.attempt for item in self.attempts]
        if (
            recorded != sorted(set(recorded))
            or any(item > self.plan_attempts for item in recorded)
            or self.unobserved_attempts
            != sorted(set(range(1, self.plan_attempts + 1)) - set(recorded))
        ):
            raise ValueError("Process usage coverage is inconsistent")
        return self
class PlanProcessUsageCapabilities(Model):
    format: Literal["tensorus-plan-process-usage-v1"] = FORMAT
    event_format: Literal["tensorus-local-process-usage-observation-v1"] = EVENT_FORMAT
    max_attempts: Literal[10] = 10
    max_response_bytes: Literal[16384] = MAX_RESPONSE_BYTES
    opt_in_linux_bounded_worker_only: Literal[True] = True
    complete_coverage: Literal[False] = False
    aggregate_rss_measured: Literal[False] = False
    independently_authenticated: Literal[False] = False
    billing_authority: Literal[False] = False

    @field_validator("max_attempts", "max_response_bytes", mode="before")
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Process usage limits must be integers")
        return value

    @field_validator(
        "opt_in_linux_bounded_worker_only",
        "complete_coverage",
        "aggregate_rss_measured",
        "independently_authenticated",
        "billing_authority",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Process usage capability flags must be booleans")
        return value
