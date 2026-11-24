"""Version-bound direct-ingestion observations in the existing catalog audit."""
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
FORMAT: Literal["tensorus-ingest-usage-observation-v1"] = "tensorus-ingest-usage-observation-v1"
EVENT_FORMAT: Literal["tensorus-ingest-commit-observation-v1"] = (
    "tensorus-ingest-commit-observation-v1"
)
MAX_RESPONSE_BYTES: Literal[8192] = 8192
LogicalBytes = Annotated[StrictInt, Field(ge=0, le=2**63 - 1)]
Timestamp = Annotated[str, Field(min_length=1, max_length=40)]
ActorId = Annotated[str, Field(min_length=1, max_length=36)]
class IngestCommitObservation(Model):
    event_id: UUID
    committed_by: ActorId
    recorded_at: Timestamp
    logical_bytes: LogicalBytes
class IngestUsageObservation(Model):
    format: Literal["tensorus-ingest-usage-observation-v1"] = FORMAT
    tenant_id: UUID
    version_id: UUID
    checksum: ReleaseHash
    observed_at: Timestamp
    coverage: Literal["recorded", "unobserved"]
    commit: IngestCommitObservation | None
    attempted_work_measured: Literal[False] = False
    physical_usage_measured: Literal[False] = False
    billing_authority: Literal[False] = False

    @field_validator(
        "attempted_work_measured", "physical_usage_measured", "billing_authority", mode="before"
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Ingestion observation flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_coverage(self) -> Self:
        if (self.coverage == "recorded") != (self.commit is not None):
            raise ValueError("Ingestion observation coverage is inconsistent")
        return self
class IngestUsageCapabilities(Model):
    format: Literal["tensorus-ingest-usage-observation-v1"] = FORMAT
    event_format: Literal["tensorus-ingest-commit-observation-v1"] = EVENT_FORMAT
    max_response_bytes: Literal[8192] = MAX_RESPONSE_BYTES
    historical_coverage_complete: Literal[False] = False
    attempted_work_measured: Literal[False] = False
    physical_usage_measured: Literal[False] = False
    billing_authority: Literal[False] = False

    @field_validator("max_response_bytes", mode="before")
    @classmethod
    def exact_limit(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Ingestion observation limit must be an integer")
        return value

    @field_validator(
        "historical_coverage_complete",
        "attempted_work_measured",
        "physical_usage_measured",
        "billing_authority",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Ingestion observation flags must be booleans")
        return value
