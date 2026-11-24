"""Tenant-local catalog capacity and advisory plan consequences."""
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictInt
from pydantic import model_validator
from tensorus.contracts import ReleaseHash
from tensorus.models import Model
CAPACITY_FORMAT: Literal["tensorus-workspace-capacity-v1"] = "tensorus-workspace-capacity-v1"
PLAN_FORMAT: Literal["tensorus-plan-capacity-review-v1"] = "tensorus-plan-capacity-review-v1"
MAX_RESPONSE_BYTES: Literal[16384] = 16384
Nonnegative = Annotated[StrictInt, Field(ge=0)]
Positive = Annotated[StrictInt, Field(ge=1)]
class QuotaBalance(Model):
    used: Nonnegative
    limit: Positive
    remaining: Nonnegative
    over_limit: Nonnegative

    @model_validator(mode="after")
    def consistent_balance(self) -> Self:
        if self.remaining != max(0, self.limit - self.used) or self.over_limit != max(
            0, self.used - self.limit
        ):
            raise ValueError("Quota balance is inconsistent")
        return self
class ClaimTotals(Model):
    plan_attempts: Nonnegative
    recorded: Nonnegative
    unrecorded: Nonnegative
    running: Nonnegative
    succeeded: Nonnegative
    failed: Nonnegative
    cancelled: Nonnegative
    lease_expired: Nonnegative

    @model_validator(mode="after")
    def consistent_claims(self) -> Self:
        if (
            self.recorded
            != sum((self.running, self.succeeded, self.failed, self.cancelled, self.lease_expired))
            or self.plan_attempts != self.recorded + self.unrecorded
        ):
            raise ValueError("Worker claim totals are inconsistent")
        return self
class WorkspaceCapacity(Model):
    format: Literal["tensorus-workspace-capacity-v1"] = CAPACITY_FORMAT
    tenant_id: UUID
    observed_at: str
    datasets: QuotaBalance
    archived_datasets: Nonnegative
    retained_versions: QuotaBalance
    retained_logical_bytes: QuotaBalance
    plan_history: QuotaBalance
    worker_claims: ClaimTotals
    catalog_snapshot: Literal[True] = True
    limits_from_this_process: Literal[True] = True
    quota_reserved: Literal[False] = False
    physical_usage_measured: Literal[False] = False
    billing_authority: Literal[False] = False

    @model_validator(mode="after")
    def consistent_datasets(self) -> Self:
        if self.archived_datasets > self.datasets.used:
            raise ValueError("Archived dataset count exceeds retained datasets")
        return self
class PlanCapacityReview(Model):
    format: Literal["tensorus-plan-capacity-review-v1"] = PLAN_FORMAT
    tenant_id: UUID
    plan_id: UUID
    plan_hash: ReleaseHash
    observed_at: str
    predicted_output_versions: Annotated[StrictInt, Field(ge=0, le=16)]
    predicted_output_bytes: Nonnegative
    estimated_compute: Nonnegative
    retained_versions: QuotaBalance
    retained_logical_bytes: QuotaBalance
    predicted_outputs_fit: bool
    quota_reserved: Literal[False] = False
    approval_granted: Literal[False] = False
    partial_commits_possible: Literal[True] = True
    physical_usage_measured: Literal[False] = False
    billing_authority: Literal[False] = False

    @model_validator(mode="after")
    def consistent_projection(self) -> Self:
        if not self.predicted_output_versions and self.predicted_output_bytes:
            raise ValueError("A read-only plan cannot predict retained output bytes")
        fits = self.predicted_output_versions == 0 or (
            self.retained_versions.used + self.predicted_output_versions
            <= self.retained_versions.limit
            and self.retained_logical_bytes.used + self.predicted_output_bytes
            <= self.retained_logical_bytes.limit
        )
        if self.predicted_outputs_fit != fits:
            raise ValueError("Predicted plan capacity is inconsistent")
        return self
class WorkspaceCapacityCapabilities(Model):
    format: Literal["tensorus-workspace-capacity-v1"] = CAPACITY_FORMAT
    plan_review_format: Literal["tensorus-plan-capacity-review-v1"] = PLAN_FORMAT
    authority: Literal["active_unscoped_tenant_admin"] = "active_unscoped_tenant_admin"
    max_response_bytes: Literal[16384] = MAX_RESPONSE_BYTES
    limits_from_this_process: Literal[True] = True
    quota_reserved: Literal[False] = False
    physical_usage_measured: Literal[False] = False
    billing_authority: Literal[False] = False
