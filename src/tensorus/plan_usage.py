"""Plan-scoped observations tied to existing worker claims and checkpoints."""
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
from tensorus.models import StepName
FORMAT: Literal["tensorus-plan-usage-observations-v1"] = "tensorus-plan-usage-observations-v1"
OUTPUT_FORMAT: Literal["tensorus-plan-output-observation-v1"] = (
    "tensorus-plan-output-observation-v1"
)
ATTEMPT_FORMAT: Literal["tensorus-plan-attempt-observation-v1"] = (
    "tensorus-plan-attempt-observation-v1"
)
MAX_RESPONSE_BYTES: Literal[16384] = 16384
MAX_ELAPSED_MS = 2**53 - 1
Attempt = Annotated[StrictInt, Field(ge=1, le=10)]
LogicalBytes = Annotated[StrictInt, Field(ge=0, le=2**63 - 1)]
ElapsedMilliseconds = Annotated[StrictInt, Field(ge=0, le=MAX_ELAPSED_MS)]
TerminalOutcome = Literal["succeeded", "failed", "cancelled"]
class OutputObservation(Model):
    step_id: StepName
    attempt: Attempt
    version_id: UUID
    logical_bytes: LogicalBytes
class AttemptObservation(Model):
    attempt: Attempt
    outcome: TerminalOutcome
    elapsed_ms: ElapsedMilliseconds
class PlanUsageObservations(Model):
    format: Literal["tensorus-plan-usage-observations-v1"] = FORMAT
    tenant_id: UUID
    plan_id: UUID
    plan_hash: ReleaseHash
    observed_at: str = Field(min_length=1, max_length=40)
    plan_status: PlanStatus
    plan_attempts: Annotated[StrictInt, Field(ge=0, le=10)]
    outputs: list[OutputObservation] = Field(max_length=16)
    unobserved_output_steps: list[StepName] = Field(max_length=16)
    attempts: list[AttemptObservation] = Field(max_length=10)
    unobserved_attempts: list[Attempt] = Field(max_length=10)
    observed_logical_output_bytes: Annotated[StrictInt, Field(ge=0)]
    logical_output_bytes_measured: Literal[True] = True
    elapsed_wall_time_only: Literal[True] = True
    physical_usage_measured: Literal[False] = False
    billing_authority: Literal[False] = False

    @field_validator(
        "logical_output_bytes_measured",
        "elapsed_wall_time_only",
        "physical_usage_measured",
        "billing_authority",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Usage observation flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_coverage(self) -> Self:
        steps = [item.step_id for item in self.outputs]
        versions = [item.version_id for item in self.outputs]
        recorded_attempts = [item.attempt for item in self.attempts]
        if (
            len(set(steps)) != len(steps)
            or len(set(versions)) != len(versions)
            or len(set(self.unobserved_output_steps)) != len(self.unobserved_output_steps)
            or set(steps) & set(self.unobserved_output_steps)
            or any(item.attempt > self.plan_attempts for item in self.outputs)
            or recorded_attempts != sorted(set(recorded_attempts))
            or any(attempt > self.plan_attempts for attempt in recorded_attempts)
            or self.unobserved_attempts
            != sorted(set(range(1, self.plan_attempts + 1)) - set(recorded_attempts))
            or self.observed_logical_output_bytes
            != sum(item.logical_bytes for item in self.outputs)
        ):
            raise ValueError("Plan usage coverage is inconsistent")
        return self
class PlanUsageCapabilities(Model):
    format: Literal["tensorus-plan-usage-observations-v1"] = FORMAT
    output_format: Literal["tensorus-plan-output-observation-v1"] = OUTPUT_FORMAT
    attempt_format: Literal["tensorus-plan-attempt-observation-v1"] = ATTEMPT_FORMAT
    max_steps: Literal[16] = 16
    max_attempts: Literal[10] = 10
    max_response_bytes: Literal[16384] = MAX_RESPONSE_BYTES
    logical_output_bytes_measured: Literal[True] = True
    elapsed_wall_time_only: Literal[True] = True
    physical_usage_measured: Literal[False] = False
    billing_authority: Literal[False] = False

    @field_validator("max_steps", "max_attempts", "max_response_bytes", mode="before")
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Usage observation limits must be integers")
        return value

    @field_validator(
        "logical_output_bytes_measured",
        "elapsed_wall_time_only",
        "physical_usage_measured",
        "billing_authority",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Usage observation flags must be booleans")
        return value
