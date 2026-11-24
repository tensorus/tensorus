"""Validated public HTTP contracts; private catalog columns are never implicit API fields."""

from datetime import datetime, timedelta
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import (
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from tensorus.advisory_pricing import AdvisoryPricingCapabilities
from tensorus.agent_grant_retry import AgentGrantRetryCapabilities
from tensorus.agent_upload_staging import (
    AgentUploadStagingAdminCapabilities,
    AgentUploadStagingCapabilities,
)
from tensorus.arrays import canonical_hash, canonical_json
from tensorus.authenticated_read_fence import AuthenticatedReadFenceCapabilities
from tensorus.bundle_inspection_contract import (
    BundleHeadPreflightCapabilities,
    BundleInspectionCapabilities,
    BundlePayloadObservationCapabilities,
    BundleRestoreApplyCapabilities,
    BundleRestorePreparationCapabilities,
    HistoryIndependentObservationCapabilities,
    HistoryIndependentSelectionCapabilities,
    HistoryPayloadSelectionCapabilities,
    HistoryReleaseObservationCapabilities,
    HistoryRestoreSelectionCapabilities,
    NameOccupancyCapabilities,
    NamePreflightCapabilities,
    ProductPayloadSelectionCapabilities,
)
from tensorus.contracts import ContractDefinition, ReleaseReport, ReleaseRequest
from tensorus.control_request_limits import CONTROL_REQUEST_ROUTES, MAX_CONTROL_REQUEST_BYTES
from tensorus.dataset_options import DatasetOptionsCapabilities
from tensorus.dataset_read_fence import DatasetReadFenceCapabilities
from tensorus.errors import TensorusError
from tensorus.history_read_admission import (
    HistoryListAdmissionCapabilities,
    HistoryReadAdmissionCapabilities,
)
from tensorus.history_summary_policy import ScopedHistorySummaryPolicyCapabilities
from tensorus.http_body_admission import HttpBodyAdmissionCapabilities
from tensorus.identity_admission import IdentityProviderAdmissionCapabilities
from tensorus.ingest_usage import IngestUsageCapabilities
from tensorus.investigation import InvestigationCapabilities
from tensorus.key_inventory import (
    CredentialInventoryAdmissionCapabilities,
    CredentialInventoryAdmissionV2Capabilities,
    CredentialPageCapabilities,
    CredentialPageV2Capabilities,
)
from tensorus.key_rotation import KeyRotationCapabilities
from tensorus.local_usage_delivery import (
    FixtureCapabilities,
    FixtureFileInspectionCapabilities,
    FixtureNamespaceGuardCapabilities,
    FixtureReceiptReadCapabilities,
)
from tensorus.local_usage_intent import FixtureIntentCapabilities
from tensorus.model_admission import (
    ModelProviderAdmissionCapabilities,
    ModelProviderTenantAdmissionCapabilities,
)
from tensorus.model_destination import DirectTransportCapabilities, ModelDestinationCapabilities
from tensorus.model_provider_http import ModelProviderTimeoutCapabilities
from tensorus.models import (
    AgentScopeValue,
    AxisCoordinateMode,
    CoordinateRange,
    DType,
    Model,
    MultiDatasetAgentScope,
    NpyWriteCondition,
    Operation,
    PlanDefinition,
    PlanStatus,
    ReindexFill,
    Role,
    SimilarityEngine,
    StepName,
    TensorSpec,
    WorkflowDefinition,
    WorkflowOrigin,
    WriteContractBinding,
    validate_agent_actions,
)
from tensorus.observation_scope import (
    OBSERVATION_SCOPE_V2_FORMAT,
    OBSERVATION_SCOPE_V3_FORMAT,
    OBSERVATION_SCOPE_V4_FORMAT,
    V4_COORDINATE_OPERATIONS,
    V4_FITTED_OPERATIONS,
    StepObservationScope,
    validate_arithmetic_input_count,
    validate_step_observation_scope,
    validate_step_observation_scope_v2,
    validate_step_observation_scope_v3,
    validate_step_observation_scope_v4,
)
from tensorus.operations import ARITHMETIC, parameters
from tensorus.organization_detachment import DetachmentCapabilities
from tensorus.organization_detachment_receipt import DetachmentReceiptCapabilities
from tensorus.organizations import (
    OrganizationCapabilities,
    OrganizationSessionFenceCapabilities,
    OrganizationSessionFenceV2Capabilities,
)
from tensorus.plan_estimated_attempts import PlanEstimatedAttemptsCapabilities
from tensorus.plan_process_usage import PlanProcessUsageCapabilities
from tensorus.plan_proposer_authority import PlanProposerAuthorityCapabilities
from tensorus.plan_usage import PlanUsageCapabilities
from tensorus.plan_usage_adjusted_handoff import AdjustedHandoffCapabilities
from tensorus.plan_usage_adjustments import AdjustmentCapabilities
from tensorus.plan_usage_handoff import PlanUsageHandoffCapabilities
from tensorus.product_descriptor_admission import ProductDescriptorAdmissionCapabilities
from tensorus.recovery_inventory import RecoveryInventoryCapabilities, RecoveryPartitionCapabilities
from tensorus.recovery_verification import RecoveryVerificationCapabilities
from tensorus.scientific_read_admission import MAX_SLOTS as MAX_SCIENTIFIC_READ_SLOTS
from tensorus.scientific_read_admission import PRODUCT_DESCRIPTOR_READ_ROUTES
from tensorus.scientific_read_admission import ROUTES as SCIENTIFIC_READ_ROUTES
from tensorus.scientific_read_admission import ROUTES_V2 as SCIENTIFIC_READ_ROUTES_V2
from tensorus.scientific_read_admission import ROUTES_V3 as SCIENTIFIC_READ_ROUTES_V3
from tensorus.scientific_read_admission import ROUTES_V4 as SCIENTIFIC_READ_ROUTES_V4
from tensorus.scientific_read_admission import ROUTES_V5 as SCIENTIFIC_READ_ROUTES_V5
from tensorus.scientific_read_admission import ROUTES_V6 as SCIENTIFIC_READ_ROUTES_V6
from tensorus.scientific_read_admission import ROUTES_V7 as SCIENTIFIC_READ_ROUTES_V7
from tensorus.scientific_read_admission import ROUTES_V8 as SCIENTIFIC_READ_ROUTES_V8
from tensorus.scientific_read_admission import ROUTES_V9 as SCIENTIFIC_READ_ROUTES_V9
from tensorus.scientific_read_admission import ROUTES_V10 as SCIENTIFIC_READ_ROUTES_V10
from tensorus.scientific_read_admission import ROUTES_V11 as SCIENTIFIC_READ_ROUTES_V11
from tensorus.semantics import CoordinateFillAxis, CoordinateOverlap, TensorSemantics
from tensorus.service_actor_grants import (
    RecipientBindingReviewCapabilities,
    RecipientCapabilities,
    RecipientCapacityCapabilities,
)
from tensorus.service_actors import ServiceActorCapabilities
from tensorus.similarity_admission import MAX_SLOTS as MAX_SIMILARITY_ADMISSION_SLOTS
from tensorus.similarity_admission import ROUTES as SIMILARITY_ADMISSION_ROUTES
from tensorus.snapshot_read_fence import SnapshotReadFenceCapabilities
from tensorus.staging_status import (
    TenantInputQuotaCapabilities,
    TenantStagingCapabilities,
    TenantStagingReadAdmissionCapabilities,
    TenantWorkQuotaCapabilities,
)
from tensorus.tenant_admission import TenantHttpAdmissionCapabilities
from tensorus.upload_chunk_admission import UploadChunkAdmissionCapabilities
from tensorus.upload_observations import UploadAttemptCapabilities
from tensorus.upload_preclaim import PreclaimCapabilities
from tensorus.validity import (
    FittedPreprocessingState,
    ReviewedImportedFittedState,
    ValidityBinding,
    _paired_definition,
    _v2_fitted_apply_definition,
    _v2_fitted_fit_definition,
)
from tensorus.worker_attempts import WorkerAttemptEvidenceCapabilities
from tensorus.worker_usage import (
    WorkerCancellationRecoveryCapabilities,
    WorkerCancellationRecoveryV2Capabilities,
    WorkerClaimCapabilities,
)
from tensorus.workflow_inventory_admission import WorkflowInventoryAdmissionCapabilities
from tensorus.workspace_capacity import WorkspaceCapacityCapabilities
from tensorus.workspace_departure import (
    DepartureCapabilities,
    DepartureReceiptRecoveryCapabilities,
)
from tensorus.workspace_directory import WorkspaceDirectoryCapabilities
from tensorus.workspace_identity_sessions import IdentitySessionsCapabilities
from tensorus.workspace_lockdown_receipt import LockdownReceiptCapabilities
from tensorus.workspace_member_invitation import MemberInvitationCapabilities
from tensorus.workspace_member_transition import MemberTransitionCapabilities
from tensorus.workspace_overview import (
    WorkspaceOverviewAdmissionCapabilities,
    WorkspaceOverviewResponseFenceCapabilities,
)
from tensorus.workspace_usage_handoff import WorkspaceUsageHandoffCapabilities
from tensorus.workspace_usage_handoff_v2 import MixedWorkspaceUsageHandoffCapabilities

MAX_CAPABILITIES_RESPONSE_BYTES = 2 * 1024**2


class Page[T](Model):
    items: list[T]
    total: int
    limit: int
    offset: int


class Problem(Model):
    code: str
    message: str
    request_id: str
    details: JsonValue = None


class ErrorEnvelope(Model):
    error: Problem


class Health(Model):
    status: Literal["alive", "ready"]
    version: str


class Tenant(Model):
    id: UUID
    name: str
    title: str
    created_at: str


class TenantIssued(Tenant):
    key_id: UUID
    api_key: str


class WorkspaceSessionIdentity(Model):
    membership_id: UUID
    expires_at: float


class WorkspaceLifecycleEvent(Model):
    id: UUID
    tenant_id: UUID
    generation: StrictInt = Field(ge=2, le=2**63 - 1)
    expected_generation: StrictInt = Field(ge=1, le=2**63 - 2)
    action: Literal["locked", "recovered"]
    actor_id: str
    created_at: str
    reason: str = Field(min_length=1, max_length=512)
    revoked_keys: StrictInt = Field(ge=0)
    suspended_members: StrictInt = Field(ge=0)
    cancelled_plans: StrictInt = Field(ge=0)
    recovery_key_id: UUID | None
    recovery_lifetime_seconds: StrictInt | None = Field(ge=3600, le=604_800)

    @model_validator(mode="after")
    def consistent_event(self) -> Self:
        if self.generation != self.expected_generation + 1 or self.reason != self.reason.strip():
            raise ValueError("Workspace lifecycle event is inconsistent")
        if self.action == "locked":
            if self.recovery_key_id is not None or self.recovery_lifetime_seconds is not None:
                raise ValueError("A lockdown cannot issue a recovery key")
        elif (
            self.recovery_key_id is None
            or self.recovery_lifetime_seconds is None
            or self.revoked_keys
            or self.suspended_members
            or self.cancelled_plans
        ):
            raise ValueError("Workspace recovery evidence is inconsistent")
        return self


class WorkspaceLifecycle(Model):
    tenant_id: UUID
    state: Literal["active", "locked"]
    generation: StrictInt = Field(ge=1, le=2**63 - 1)
    updated_at: str
    updated_by: str
    last_event: WorkspaceLifecycleEvent | None

    @model_validator(mode="after")
    def consistent_state(self) -> Self:
        event = self.last_event
        if self.generation == 1:
            if self.state != "active" or event is not None:
                raise ValueError("Initial workspace lifecycle state is inconsistent")
        elif (
            event is None
            or event.tenant_id != self.tenant_id
            or event.generation != self.generation
            or event.action != ("locked" if self.state == "locked" else "recovered")
            or event.actor_id != self.updated_by
            or event.created_at != self.updated_at
        ):
            raise ValueError("Workspace lifecycle state is inconsistent")
        return self


class WorkspaceLockdownExtended(Model):
    format: Literal["tensorus-workspace-lockdown-set-v2"]
    lifecycle: WorkspaceLifecycle

    @model_validator(mode="after")
    def consistent_lockdown(self) -> Self:
        event = self.lifecycle.last_event
        if (
            self.lifecycle.state != "locked"
            or event is None
            or event.revoked_keys > 1_001_000
            or event.suspended_members > 10_000
            or event.cancelled_plans > 10_000
        ):
            raise ValueError("Extended lockdown evidence is inconsistent")
        return self


class WorkspaceLockdownReceipt(Model):
    format: Literal["tensorus-workspace-lockdown-receipt-v1"] = (
        "tensorus-workspace-lockdown-receipt-v1"
    )
    receipt_id: UUID
    expires_at: StrictInt = Field(ge=1, le=2**53 - 1)
    mode: Literal["standard", "extended"]
    lifecycle: WorkspaceLifecycle

    @model_validator(mode="after")
    def consistent_lockdown(self) -> Self:
        event = self.lifecycle.last_event
        maximum_keys = 20_000 if self.mode == "standard" else 1_001_000
        if (
            self.lifecycle.state != "locked"
            or event is None
            or event.revoked_keys < 1
            or event.revoked_keys > maximum_keys
            or event.suspended_members > 10_000
            or event.cancelled_plans > 10_000
        ):
            raise ValueError("Lockdown receipt evidence is inconsistent")
        return self


class AgentProposalBudget(Model):
    format: Literal["tensorus-agent-proposal-budget-v1"]
    limit: StrictInt = Field(ge=1, le=100_000)
    committed: StrictInt = Field(ge=0, le=100_000)
    remaining: StrictInt = Field(ge=0, le=100_000)
    execution_resources_measured: Literal[False]
    billing_authority: Literal[False]

    @field_validator("execution_resources_measured", "billing_authority", mode="before")
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Proposal allowance flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_budget(self) -> Self:
        if self.committed + self.remaining != self.limit:
            raise ValueError("Proposal allowance counts are inconsistent")
        return self


class AgentProposalBudgetStatus(Model):
    format: Literal["tensorus-agent-proposal-budget-status-v1"]
    budget: AgentProposalBudget | None


class AgentExecutionBudget(Model):
    format: Literal["tensorus-agent-execution-budget-v1"]
    estimator: Literal["tensorus-plan-preflight-estimate-v1"]
    limit_units: StrictInt = Field(ge=1, le=1_000_000_000_000)
    attempted_units: StrictInt = Field(ge=0, le=1_000_000_000_000)
    remaining_units: StrictInt = Field(ge=0, le=1_000_000_000_000)
    attempts: StrictInt = Field(ge=0, le=10_000)
    max_attempts: Literal[10_000]
    refunded_on_failure: Literal[False]
    resource_use_measured: Literal[False]
    billing_authority: Literal[False]

    @field_validator(
        "refunded_on_failure", "resource_use_measured", "billing_authority", mode="before"
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Execution allowance flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_budget(self) -> Self:
        if (
            self.attempted_units + self.remaining_units != self.limit_units
            or self.attempted_units < self.attempts
            or (self.attempts == 0 and self.attempted_units != 0)
        ):
            raise ValueError("Execution allowance counts are inconsistent")
        return self


class AgentExecutionBudgetStatus(Model):
    format: Literal["tensorus-agent-execution-budget-status-v1"]
    budget: AgentExecutionBudget | None


class AgentIngestBudget(Model):
    format: Literal["tensorus-agent-ingest-budget-v1"]
    limit_bytes: StrictInt = Field(ge=1, le=1_000_000_000_000)
    committed_bytes: StrictInt = Field(ge=0, le=1_000_000_000_000)
    remaining_bytes: StrictInt = Field(ge=0, le=1_000_000_000_000)
    committed_versions: StrictInt = Field(ge=0, le=10_000)
    max_versions: Literal[10_000]
    applies_to: Literal["direct_ingestion"]
    attempted_work_measured: Literal[False]
    physical_storage_measured: Literal[False]
    billing_authority: Literal[False]

    @field_validator(
        "attempted_work_measured", "physical_storage_measured", "billing_authority", mode="before"
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Ingestion allowance flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_budget(self) -> Self:
        if self.committed_bytes + self.remaining_bytes != self.limit_bytes or (
            self.committed_versions == 0 and self.committed_bytes != 0
        ):
            raise ValueError("Ingestion allowance counts are inconsistent")
        return self


class AgentIngestBudgetStatus(Model):
    format: Literal["tensorus-agent-ingest-budget-status-v1"]
    budget: AgentIngestBudget | None


class Identity(Tenant):
    role: Role
    key_id: UUID
    scope: AgentScopeValue | None = None
    workspace_session: WorkspaceSessionIdentity | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class Key(Model):
    id: UUID
    name: str
    role: Role
    created_at: str
    expires_at: float | None
    revoked_at: str | None = None
    scope: AgentScopeValue | None = None
    proposal_budget: AgentProposalBudget | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    execution_budget: AgentExecutionBudget | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    ingest_budget: AgentIngestBudget | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class CredentialPage(Model):
    format: Literal["tensorus-credential-page-v1"] = "tensorus-credential-page-v1"
    tenant_id: UUID
    limit: Annotated[StrictInt, Field(ge=1, le=100)]
    after: UUID | None
    items: list[Key] = Field(max_length=100)
    next_after: UUID | None
    cross_page_snapshot: Literal[False] = False

    @field_validator("cross_page_snapshot", mode="before")
    @classmethod
    def exact_snapshot(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Credential page snapshot evidence must be boolean")
        return value

    @model_validator(mode="after")
    def consistent(self) -> Self:
        from tensorus.organizations import _timestamp

        order = []
        for item in self.items:
            order.append((_timestamp(item.created_at), str(item.id)))
            if not item.name or len(item.name) > 128:
                raise ValueError("Credential page entry is invalid")
            if item.revoked_at is not None:
                _timestamp(item.revoked_at)
            if item.scope is None:
                if any((item.proposal_budget, item.execution_budget, item.ingest_budget)):
                    raise ValueError("An unscoped credential cannot have agent allowances")
                continue
            if (
                item.role != Role.reader
                or item.scope.name != item.name
                or item.scope.expires_at != item.expires_at
            ):
                raise ValueError("Credential scope is inconsistent")
            resources = (
                item.scope.datasets
                if isinstance(item.scope, MultiDatasetAgentScope)
                else [item.scope]
            )
            for resource in resources:
                if str(UUID(resource.dataset_id)) != resource.dataset_id:
                    raise ValueError("Credential dataset identity is invalid")
                validate_agent_actions(resource.actions)
            if (item.proposal_budget is not None or item.execution_budget is not None) and not any(
                "propose" in resource.actions for resource in resources
            ):
                raise ValueError("Agent proposal allowance lacks proposal authority")
            if item.ingest_budget is not None and not any(
                "ingest" in resource.actions for resource in resources
            ):
                raise ValueError("Agent ingestion allowance lacks ingestion authority")
        ids = [item.id for item in self.items]
        if (
            len(ids) > self.limit
            or len(set(ids)) != len(ids)
            or order != sorted(order, reverse=True)
            or self.after in ids
            or (
                self.next_after is not None
                and (len(ids) != self.limit or not ids or self.next_after != ids[-1])
            )
        ):
            raise ValueError("Credential page or cursor is inconsistent")
        return self


class KeyIssued(Key):
    tenant_id: UUID
    api_key: str


class WorkspaceMember(Model):
    id: UUID
    tenant_id: UUID
    user_id: UUID
    issuer: str
    subject: str
    role: Role
    status: Literal["invited", "active", "suspended", "removed"]
    generation: int
    invited_by: str
    created_at: str
    updated_at: str
    accepted_at: str | None


class WorkspaceSessionQuota(Model):
    unexpired_unrevoked: int
    active_limit: int
    retained: int
    retained_limit: int


class WorkspaceMemberPage(Model):
    items: list[WorkspaceMember]
    total: int
    limit: int
    offset: int
    session_quota: WorkspaceSessionQuota


class WorkspaceMemberListResponseFenceCapabilities(Model):
    format: Literal["tensorus-workspace-member-list-response-fence-v1"] = (
        "tensorus-workspace-member-list-response-fence-v1"
    )
    route: Literal["GET /v1/workspace/members"] = "GET /v1/workspace/members"
    principal_rechecked_after_serialization: Literal[True] = True
    captured_page_compared: Literal[False] = False
    atomic_with_delivery: Literal[False] = False
    cross_page_snapshot: Literal[False] = False
    other_identity_routes_covered: Literal[False] = False

    @field_validator(
        "principal_rechecked_after_serialization",
        "captured_page_compared",
        "atomic_with_delivery",
        "cross_page_snapshot",
        "other_identity_routes_covered",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Membership list response fence flags must be booleans")
        return value


class IdentitySession(KeyIssued):
    membership_id: UUID


class WorkspaceSessionRevocation(Model):
    format: Literal["tensorus-workspace-session-revocation-v1"]
    tenant_id: UUID
    key_id: UUID
    membership_id: UUID
    revoked_at: str = Field(strict=True, min_length=1, max_length=64)

    @field_validator("revoked_at")
    @classmethod
    def canonical_timestamp(cls, value: str) -> str:
        timestamp = datetime.fromisoformat(value)
        if (
            timestamp.utcoffset() != timedelta(0)
            or timestamp.isoformat(timespec="microseconds") != value
        ):
            raise ValueError("Session revocation time must be canonical UTC")
        return value


class WorkspaceSessionRevocationCapabilities(Model):
    format: Literal["tensorus-workspace-session-revocation-v1"]
    max_request_bytes: StrictInt = Field(ge=1024, le=MAX_CONTROL_REQUEST_BYTES)
    max_response_bytes: Literal[4096]
    current_membership_session_only: Literal[True]
    issued_agent_grants_revoked: Literal[True]
    same_credential_retry_supported: Literal[False]
    provider_session_revoked: Literal[False]
    running_processes_stopped: Literal[False]
    exported_copies_revoked: Literal[False]

    @field_validator("max_response_bytes", mode="before")
    @classmethod
    def exact_limit(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Session revocation response limit must be an integer")
        return value

    @field_validator(
        "current_membership_session_only",
        "issued_agent_grants_revoked",
        "same_credential_retry_supported",
        "provider_session_revoked",
        "running_processes_stopped",
        "exported_copies_revoked",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Session revocation capability flags must be booleans")
        return value


class Dataset(Model):
    id: UUID
    tenant_id: UUID
    name: str
    title: str
    description: str
    spec: TensorSpec
    created_at: str
    archived_at: str | None


class DatasetSummary(Dataset):
    tensor_count: int


class Statistics(Model):
    count: int
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    norm: float | None
    overflow: list[str]


class DatasetAccessRequirement(Model):
    dataset: str
    dataset_id: UUID


class VersionAccess(Model):
    policy: Literal["source_datasets_v1", "grant_scope_datasets_v1"]
    required_datasets: list[DatasetAccessRequirement]


class TensorVersion(Model):
    id: UUID
    tenant_id: UUID
    tensor_id: UUID
    version: int
    dtype: DType
    shape: list[int]
    nbytes: int
    checksum: str
    metadata: dict[str, JsonValue]
    statistics: Statistics = Field(
        description=(
            "Approximate raw summaries of all stored values; no validity assertion is applied."
        )
    )
    statistics_scope: Literal["raw_stored_values"] = "raw_stored_values"
    operation: dict[str, JsonValue] | None
    created_at: str
    created_by: str
    name: str
    head_version: int
    deleted_at: str | None
    dataset: str
    dataset_id: UUID
    parents: list[UUID]
    write_contract: WriteContractBinding | None = None
    access: VersionAccess | None = Field(default=None, exclude_if=lambda value: value is None)


class ConditionalNpyReceipt(Model):
    format: Literal["tensorus-conditional-npy-receipt-v1"]
    tenant_id: UUID
    actor_key_id: UUID
    condition: NpyWriteCondition
    request_hash: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    status: Literal["not_observed", "committed"]
    version: TensorVersion | None
    payload_verified: Literal[False]
    retry_guaranteed: Literal[False]

    @model_validator(mode="after")
    def consistent_status(self) -> Self:
        if (self.version is None) != (self.status == "not_observed"):
            raise ValueError("Conditional ingestion receipt status is inconsistent")
        return self


class ConditionalNpyHistoricalReceipt(Model):
    format: Literal["tensorus-conditional-npy-historical-receipt-v1"]
    tenant_id: UUID
    actor_key_id: UUID
    condition: NpyWriteCondition
    current_write_contract: WriteContractBinding | None
    request_hash: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    status: Literal["not_observed", "committed"]
    version: TensorVersion | None
    review_policy_current: StrictBool
    retained_policy_matches_review: StrictBool | None
    retained_policy_current: StrictBool | None
    continuation: Literal["not_assessed", "blocked_policy_changed"]
    next_action: Literal["review_before_retry", "review_partial_and_select_new_destinations"]
    payload_verified: Literal[False]
    retry_guaranteed: Literal[False]

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        current = self.current_write_contract.generation if self.current_write_contract else 0
        retained = (
            self.version.write_contract.generation
            if self.version is not None and self.version.write_contract is not None
            else 0
        )
        observed = self.version is not None
        blocked = self.condition.write_contract_generation != current or (
            observed and retained != current
        )
        if (
            (self.status == "committed") != observed
            or self.condition.write_contract_generation > current
            or (observed and retained > current)
            or self.review_policy_current != (self.condition.write_contract_generation == current)
            or self.retained_policy_matches_review
            != (retained == self.condition.write_contract_generation if observed else None)
            or self.retained_policy_current != (retained == current if observed else None)
            or self.continuation != ("blocked_policy_changed" if blocked else "not_assessed")
            or self.next_action
            != ("review_partial_and_select_new_destinations" if blocked else "review_before_retry")
            or (
                observed
                and retained == current
                and self.version is not None
                and self.version.write_contract != self.current_write_contract
            )
        ):
            raise ValueError("Historical conditional ingestion receipt is inconsistent")
        return self


class SharedAxisQueryEvidence(Model):
    format: Literal["tensorus-exact-shared-axis-query-v1", "tensorus-overlap-shared-axis-query-v1"]
    version_id: UUID
    axes: list[str] = Field(min_length=1, max_length=16)
    candidates_evaluated: int = Field(ge=0)
    coordinate_values: int = Field(ge=0)
    kind: (
        Literal[
            "exact_cross_snapshot_shared_axis_v1",
            "overlap_shared_axis_v1",
            "overlap_cross_snapshot_shared_axis_v1",
        ]
        | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    anchor_snapshot_id: UUID | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def cross_snapshot_evidence(self) -> Self:
        if (
            self.kind
            in (
                "exact_cross_snapshot_shared_axis_v1",
                "overlap_cross_snapshot_shared_axis_v1",
            )
        ) != (self.anchor_snapshot_id is not None):
            raise ValueError("Cross-snapshot evidence requires its kind and anchor snapshot")
        if (self.format == "tensorus-overlap-shared-axis-query-v1") != (
            self.kind in ("overlap_shared_axis_v1", "overlap_cross_snapshot_shared_axis_v1")
        ):
            raise ValueError("Shared-axis evidence format does not match its kind")
        return self


class StreamedIndexEvaluationEvidence(Model):
    format: Literal["tensorus-indexed-stream-evaluation-v1"]
    candidates_evaluated: StrictInt = Field(ge=0)
    descriptor_bytes: StrictInt = Field(ge=0)


class SemanticIndexEvidence(Model):
    format: Literal["tensorus-semantic-property-index-v1"]
    generation: int = Field(ge=1)
    watermark: int = Field(ge=0)
    reconciled_candidates: int = Field(ge=0)
    query_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_id: UUID | None
    shared_axis: SharedAxisQueryEvidence | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    streamed_evaluation: StreamedIndexEvaluationEvidence | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class IndexedTensorPage(Page[TensorVersion]):
    index: SemanticIndexEvidence


class IndexedSnapshotPage(IndexedTensorPage):
    format: Literal["tensorus-indexed-snapshot-page-v1"]
    query_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    next_cursor: str | None = Field(max_length=2048)


class SemanticIndexStatus(Model):
    format: Literal["tensorus-semantic-property-index-v1"]
    generation: int | None
    watermark: int | None
    versions: int
    index_entries: int
    counts_match: bool
    contents_reconciled: Literal[False]
    rebuild_version_limit: int
    rebuild_descriptor_bytes: int


class SemanticIndexRebuild(Model):
    format: Literal["tensorus-semantic-property-index-v1"]
    generation: int
    watermark: int
    versions: int
    descriptor_bytes: int


class UploadInputReservationState(Model):
    format: Literal["tensorus-upload-input-reservation-v1"]
    reserved_input_bytes: StrictInt = Field(ge=1)
    observed_allocated_bytes: StrictInt | None = Field(ge=1)
    reserved_work_bytes: StrictInt = Field(ge=1)


class UploadFinalizerState(Model):
    format: Literal["tensorus-upload-finalizer-v1"]
    generation: StrictInt = Field(ge=1)
    lease_expires: float | None
    input_reservation: UploadInputReservationState | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class Upload(Model):
    id: UUID
    dataset: str
    name: str
    size: int
    checksum: str
    metadata: dict[str, JsonValue]
    expected_version: int
    write_contract: WriteContractBinding | None
    offset: int
    chunk_size: int
    status: Literal["open", "completed", "cancelled", "expired", "revoked"]
    created_at: str
    expires_at: float
    retain_until: float
    version_id: UUID | None
    finalization: UploadFinalizerState | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_input_reservation(self) -> Self:
        reservation = self.finalization.input_reservation if self.finalization else None
        if reservation is not None:
            allocated = reservation.observed_allocated_bytes
            if reservation.reserved_input_bytes < self.size or (
                allocated is not None
                and (allocated < self.size or allocated > reservation.reserved_input_bytes)
            ):
                raise ValueError("Upload input reservation contradicts the staged file size")
        return self


class Preview(Model):
    version_id: UUID
    shape: list[int]
    dtype: DType
    selection: list[list[int | None]]
    data: JsonValue
    truncated: bool


class SearchHit(Model):
    tensor: TensorVersion
    score: float


class SearchResults(Model):
    metric: Literal["cosine", "l2"]
    exact: Literal[True]
    candidates: int
    items: list[SearchHit]
    engine: SimilarityEngine
    score_unit: str | None
    warnings: list[str]

    @model_validator(mode="after")
    def consistent_engine(self) -> Self:
        self.engine.require_metric(self.metric)
        return self


class Snapshot(Model):
    id: UUID
    tenant_id: UUID
    dataset_id: UUID
    name: str
    created_at: str
    created_by: str


class SnapshotCursorCapabilities(Model):
    format: Literal["tensorus-snapshot-cursor-list-v1"] = "tensorus-snapshot-cursor-list-v1"
    max_page: Literal[200] = 200
    ordering: Literal["created_at_id_descending_binary"] = "created_at_id_descending_binary"
    retained_visible_anchor_required: Literal[True] = True
    authority_rechecked: Literal[True] = True
    cross_page_snapshot: Literal[False] = False
    total_included: Literal[False] = False
    cursor_authorizes_read: Literal[False] = False
    atomic_with_delivery: Literal[False] = False

    @field_validator("max_page", mode="before")
    @classmethod
    def exact_limit(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Snapshot cursor page limit must be an integer")
        return value

    @field_validator(
        "retained_visible_anchor_required",
        "authority_rechecked",
        "cross_page_snapshot",
        "total_included",
        "cursor_authorizes_read",
        "atomic_with_delivery",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Snapshot cursor flags must be booleans")
        return value


class SnapshotCursorPage(Model):
    format: Literal["tensorus-snapshot-cursor-page-v1"]
    tenant_id: UUID
    dataset_id: UUID
    limit: Annotated[StrictInt, Field(ge=1, le=200)]
    after: UUID | None
    before: UUID | None
    items: list[Snapshot] = Field(max_length=200)
    has_newer: StrictBool
    has_older: StrictBool

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if any(not item.created_at.isascii() or len(item.created_at) > 40 for item in self.items):
            raise ValueError("Snapshot cursor page timestamps are invalid")
        try:
            timestamps = [datetime.fromisoformat(item.created_at) for item in self.items]
        except ValueError as exc:
            raise ValueError("Snapshot cursor page timestamps are invalid") from exc
        keys = [(item.created_at, str(item.id)) for item in self.items]
        if (
            (self.after is not None and self.before is not None)
            or len(self.items) > self.limit
            or len({item.id for item in self.items}) != len(self.items)
            or any(
                item.tenant_id != self.tenant_id or item.dataset_id != self.dataset_id
                for item in self.items
            )
            or any(
                stamp.tzinfo is None or stamp.utcoffset() != timedelta(0) for stamp in timestamps
            )
            or keys != sorted(keys, reverse=True)
            or (not self.items and (self.after is not None or self.before is not None))
            or (not self.items and (self.has_newer or self.has_older))
            or (self.after is not None and not self.has_newer)
            or (self.before is not None and not self.has_older)
            or (self.after is None and self.before is None and self.has_newer)
            or any(item.id in {self.after, self.before} for item in self.items)
        ):
            raise ValueError("Snapshot cursor page is inconsistent")
        return self


class SnapshotDetail(Snapshot):
    tensor_count: int


class SnapshotDiff(Model):
    added: list[str]
    removed: list[str]
    changed: list[str]


class SnapshotRetirementPreview(Model):
    format: Literal["tensorus-snapshot-retirement-preview-v1"]
    tenant_id: UUID
    snapshot_id: UUID
    dataset_id: UUID
    dataset: str
    name: str
    tensor_count: int
    release_referenced: bool
    imported_history_referenced: bool
    evaluation_referenced: bool
    receipt_budget_available: bool
    can_retire: bool
    tensor_data_deleted: Literal[False]


class SnapshotRetirementReview(Model):
    format: Literal["tensorus-snapshot-retirement-review-v2"]
    preview: SnapshotRetirementPreview
    active_hold_count: Annotated[StrictInt, Field(ge=0, le=64)]

    @model_validator(mode="after")
    def consistent_blockers(self) -> Self:
        preview = self.preview
        eligible = not (
            preview.release_referenced
            or preview.imported_history_referenced
            or preview.evaluation_referenced
            or not preview.receipt_budget_available
            or self.active_hold_count
        )
        if preview.can_retire != eligible:
            raise ValueError("Snapshot retirement blockers are inconsistent")
        return self


class ContractRevision(Model):
    dataset_id: UUID
    revision: int
    definition: ContractDefinition
    definition_hash: str
    created_at: str
    created_by: str | None
    origin: Literal["legacy_dataset", "dataset_creation", "revision"]


class WriteContractEvent(Model):
    dataset_id: UUID
    generation: int
    revision: int
    definition_hash: str
    activated_at: str
    activated_by: str


class WriteContractState(Model):
    dataset_id: UUID
    generation: int
    revision: int | None
    definition_hash: str | None
    definition: ContractDefinition | None
    activated_at: str | None
    activated_by: str | None


class WriteContractActivationResult(WriteContractState):
    current_generation: int
    is_current: bool


class ReleaseSummary(Model):
    id: UUID
    tenant_id: UUID
    dataset_id: UUID
    name: str
    snapshot_id: UUID
    contract_revision: int
    report_hash: str
    status: Literal["failed", "proposed", "approved"]
    created_at: str
    created_by: str
    approved_at: str | None
    approved_by: str | None


class Release(ReleaseSummary):
    dataset: str
    request: ReleaseRequest
    report: ReleaseReport


class LineageEdge(Model):
    parent: UUID
    child: UUID


class Lineage(Model):
    root: UUID
    nodes: list[TensorVersion]
    edges: list[LineageEdge]
    truncated: bool


class DescendantNode(Model):
    id: UUID
    dataset: str
    dataset_id: UUID
    name: str
    version: int
    head_version: int
    deleted_at: str | None
    dtype: DType
    shape: list[int]
    nbytes: int
    created_at: str


class DescendantLineage(Model):
    root: UUID
    nodes: list[DescendantNode]
    edges: list[LineageEdge]
    truncated: bool
    max_depth: int


class ProfileResult(Model):
    version_id: UUID
    shape: list[int]
    dtype: DType
    statistics: Statistics = Field(
        description=(
            "Approximate raw summaries of all stored values; no validity assertion is applied."
        )
    )
    statistics_scope: Literal["raw_stored_values"] = "raw_stored_values"


class TransformResult(Model):
    version_id: UUID
    nbytes: int


class ExecutionError(Model):
    code: str
    message: str


class Plan(Model):
    id: UUID
    tenant_id: UUID
    definition: PlanDefinition
    plan_hash: str
    planner: Literal["manual", "builtin", "llm", "workflow", "delegated"]
    writes: bool
    status: PlanStatus
    workflow: WorkflowOrigin | None = None
    results: dict[str, ProfileResult | TransformResult]
    error: ExecutionError | None
    created_at: str
    created_by: str
    approved_by: str | None
    approved_at: str | None
    started_at: float | None
    finished_at: str | None
    cancel_requested: bool
    attempts: int

    @model_validator(mode="after")
    def consistent_proposer(self) -> Self:
        if not self.definition.matches_recorded_proposer(str(self.tenant_id), self.created_by):
            raise ValueError("Plan origin does not match its pinned proposer")
        return self


class WorkerOutputReservationEvidence(Model):
    step_id: StepName
    attempt: StrictInt = Field(ge=1, le=10)
    state: Literal["active", "released"]
    reserved_work_bytes: StrictInt = Field(ge=1)


class WorkerOutputReservationStatus(Model):
    format: Literal["tensorus-worker-output-reservations-v1"]
    plan_id: UUID
    plan_attempts: StrictInt = Field(ge=0, le=10)
    writable_steps: list[StepName] = Field(max_length=16)
    catalog_policy_enabled: StrictBool
    reservations: list[WorkerOutputReservationEvidence] = Field(max_length=160)
    active_work_bytes: StrictInt = Field(ge=0)
    metadata_reserved: Literal[False]
    whole_host_reserved: Literal[False]
    policy: str = Field(min_length=1)

    @field_validator("metadata_reserved", "whole_host_reserved", mode="before")
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Reservation scope flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_reservations(self) -> Self:
        identities = [(item.attempt, item.step_id) for item in self.reservations]
        active = [item for item in self.reservations if item.state == "active"]
        if (
            len(self.writable_steps) != len(set(self.writable_steps))
            or identities != sorted(set(identities))
            or self.active_work_bytes != sum(item.reserved_work_bytes for item in active)
            or (active and not self.catalog_policy_enabled)
            or any(
                item.attempt > self.plan_attempts or item.step_id not in self.writable_steps
                for item in self.reservations
            )
        ):
            raise ValueError("Worker output reservation evidence is inconsistent")
        return self


class PredictedTensor(Model):
    shape: list[int]
    dtype: DType
    nbytes: int
    semantics: TensorSemantics | None = None
    access: VersionAccess | None = Field(default=None, exclude_if=lambda value: value is None)


class ExplainedInput(PredictedTensor):
    reference: str
    name: str
    dataset: str


class ExplainedStep(Model):
    id: str
    operation: Operation
    inputs: list[ExplainedInput]
    output: PredictedTensor | None
    output_name: str | None
    parameters: dict[str, JsonValue]
    estimated_compute: int
    observation_scope: StepObservationScope
    observation_scope_v2: StepObservationScope | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    observation_scope_v3: StepObservationScope | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    observation_scope_v4: StepObservationScope | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def scope_matches_operation(self) -> Self:
        validate_step_observation_scope(self.operation, self.observation_scope, len(self.inputs))
        return self


class PlanExplanation(Model):
    definition: PlanDefinition
    plan_hash: str
    observation_scope_format: Literal["tensorus-step-observation-scope-v1"]
    observation_scope_v2_format: Literal["tensorus-step-observation-scope-v2"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    observation_scope_v3_format: Literal["tensorus-step-observation-scope-v3"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    observation_scope_v4_format: Literal["tensorus-step-observation-scope-v4"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    workflow: WorkflowOrigin | None = None
    writes: bool
    required_role: Role
    can_approve: bool
    output_bytes: int
    estimated_compute: int
    output_byte_limit: int
    compute_limit: int
    steps: list[ExplainedStep]
    warnings: list[str]

    @model_validator(mode="after")
    def extended_scope_matches_plan(self) -> Self:
        versions = (
            (
                "V2",
                self.observation_scope_v2_format,
                OBSERVATION_SCOPE_V2_FORMAT,
                [step.observation_scope_v2 for step in self.steps],
                validate_step_observation_scope_v2,
            ),
            (
                "V3",
                self.observation_scope_v3_format,
                OBSERVATION_SCOPE_V3_FORMAT,
                [step.observation_scope_v3 for step in self.steps],
                validate_step_observation_scope_v3,
            ),
            (
                "V4",
                self.observation_scope_v4_format,
                OBSERVATION_SCOPE_V4_FORMAT,
                [step.observation_scope_v4 for step in self.steps],
                validate_step_observation_scope_v4,
            ),
        )
        for version, marker, expected, scopes, validate_scope in versions:
            if marker is None:
                if any(scope is not None for scope in scopes):
                    raise ValueError(f"{version} observation scope requires its format marker")
                continue
            if marker != expected or any(scope is None for scope in scopes):
                raise ValueError(f"{version} observation scope requires complete step evidence")
            if len(self.steps) != len(self.definition.steps):
                raise ValueError(f"{version} observation scope requires the complete plan")
            for definition_step, explained, scope in zip(
                self.definition.steps, self.steps, scopes, strict=True
            ):
                if (
                    definition_step.id != explained.id
                    or definition_step.operation != explained.operation
                    or definition_step.inputs != [source.reference for source in explained.inputs]
                ):
                    raise ValueError(f"{version} observation scope does not match the plan step")
                assert scope is not None
                validate_scope(explained.operation, scope, len(explained.inputs))
                validate_arithmetic_input_count(
                    explained.operation, len(explained.inputs), explained.parameters
                )
                if explained.operation in ARITHMETIC or (
                    version == "V4"
                    and explained.operation in V4_COORDINATE_OPERATIONS | V4_FITTED_OPERATIONS
                ):
                    try:
                        same_parameters = canonical_json(explained.parameters) == canonical_json(
                            parameters(definition_step)
                        )
                    except TensorusError as exc:
                        raise ValueError("Invalid scoped operation parameters in the plan") from exc
                    if not same_parameters:
                        raise ValueError(
                            f"{version} observation scope does not match the plan's parameters"
                        )
        return self


class CoordinateFill(Model):
    anchor: Literal["left", "right"]
    fill_value: ReindexFill
    requires_left_reindex: bool
    requires_right_reindex: bool
    axes: list[CoordinateFillAxis]
    coverage_output: str | None = Field(default=None, exclude_if=lambda value: value is None)
    coverage_step: str | None = Field(default=None, exclude_if=lambda value: value is None)


class ValidityTransformExplanation(Model):
    binding: ValidityBinding
    other_binding: ValidityBinding | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    coordinate_intersection: list[CoordinateOverlap] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    coordinate_fill: CoordinateFill | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    right_permutation: list[int] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    source_permutation: list[int] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    transposed_side: Literal["left", "right"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    requires_transpose: bool | None = Field(default=None, exclude_if=lambda value: value is None)
    explanation: PlanExplanation


class ValidityCoordinateWindowExplanation(Model):
    format: Literal["tensorus-validity-coordinate-window-v1"]
    binding: ValidityBinding
    coordinate_selection: dict[str, CoordinateRange]
    positional_selection: list[list[int | None]]
    explanation: PlanExplanation
    limitations: list[str]


class FittedPreprocessingFitExplanation(Model):
    binding: ValidityBinding
    explanation: PlanExplanation

    @model_validator(mode="after")
    def matches_fit(self) -> Self:
        definition = self.explanation.definition
        if (
            not _v2_fitted_fit_definition(definition)
            or not _paired_definition(
                definition,
                str(self.binding.data_version_id),
                str(self.binding.mask_version_id),
                str(self.binding.id),
                self.binding.descriptor_hash,
            )
            or self.explanation.observation_scope_v4_format != OBSERVATION_SCOPE_V4_FORMAT
            or canonical_hash(definition.model_dump(mode="json")) != self.explanation.plan_hash
        ):
            raise ValueError("Fitted fit explanation does not match its binding and plan")
        return self


class FittedPreprocessingApplyExplanation(Model):
    binding: ValidityBinding
    state: FittedPreprocessingState | ReviewedImportedFittedState
    explanation: PlanExplanation

    @model_validator(mode="after")
    def matches_apply(self) -> Self:
        definition = self.explanation.definition
        if (
            not _v2_fitted_apply_definition(definition)
            or not _paired_definition(
                definition,
                str(self.binding.data_version_id),
                str(self.binding.mask_version_id),
                str(self.binding.id),
                self.binding.descriptor_hash,
            )
            or self.explanation.observation_scope_v4_format != OBSERVATION_SCOPE_V4_FORMAT
            or canonical_hash(definition.model_dump(mode="json")) != self.explanation.plan_hash
        ):
            raise ValueError("Fitted transform explanation does not match its binding and plan")
        step = definition.steps[0]
        normalized = parameters(step)
        if (
            step.inputs[1] != str(self.state.state_version_id)
            or normalized["state_checksum"] != self.state.state_checksum
            or normalized["algorithm"] != self.state.algorithm
            or normalized["algorithm_version"] != self.state.algorithm_version
            or normalized["axis"] != self.state.axis
            or normalized.get("imported_state")
            != (
                self.state.selection.model_dump(mode="json")
                if isinstance(self.state, ReviewedImportedFittedState)
                else None
            )
        ):
            raise ValueError("Fitted transform explanation does not match its state")
        return self


class AxisAlignment(Model):
    format: Literal["tensorus-axis-alignment-v1", "tensorus-axis-alignment-v2"]
    left_version_id: UUID
    right_version_id: UUID
    coordinate_mode: AxisCoordinateMode
    value_scope: Literal["raw_unmasked"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    right_permutation: list[int]
    requires_transpose: bool
    requires_left_slice: bool
    requires_right_slice: bool
    left_selection: list[list[int] | None] | None
    right_selection: list[list[int] | None] | None
    coordinate_overlaps: list[CoordinateOverlap]
    coordinate_fill: CoordinateFill | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    explanation: PlanExplanation
    limitations: list[str]

    @model_validator(mode="after")
    def validate_value_scope(self) -> Self:
        if (self.format == "tensorus-axis-alignment-v2") != (self.value_scope == "raw_unmasked"):
            raise ValueError("V2 alignment must explicitly identify raw, unmasked values")
        return self


class Workflow(Model):
    id: UUID
    tenant_id: UUID
    name: str
    version: int
    definition: WorkflowDefinition
    created_at: str
    updated_at: str
    updated_by: str
    archived_at: str | None


class WorkflowRevision(Model):
    workflow_id: UUID
    version: int
    definition: WorkflowDefinition
    created_at: str
    created_by: str


class Limits(Model):
    tensor_bytes: int
    tenant_bytes: int
    preview_elements: int
    search_candidates: int
    retained_versions: int
    datasets: int
    keys: int
    snapshots: int
    snapshot_retirements: int
    contract_revisions: int
    releases: int
    release_tensors: int
    release_bytes: int
    plans: int
    workflows: int
    workflow_versions: int


class ScopedMetadataDiscoveryCapabilities(Model):
    format: Literal["tensorus-scoped-metadata-discovery-v1"]
    max_protected_versions: StrictInt = Field(ge=1)
    dataset_counts: Literal["readable_heads"]
    legacy_query_and_history: Literal["reject_unreadable_direct_lineage_before_count"]
    snapshot_list: Literal["omit_unreadable_memberships"]
    response_delivery_atomic: Literal[False]


class WorkspaceStats(Model):
    datasets: int
    tensors: int
    versions: int
    logical_bytes: int
    unique_objects: int
    plans: dict[str, int]
    workflows: int
    workflow_versions: int
    limits: Limits


class NearestCoordinateCapabilities(Model):
    format: Literal["tensorus-nearest-integer-coordinate-v1"]
    algorithm: Literal["nearest_integer_v1"]
    coordinates: Literal["signed_int64_changed_axes"]
    distance: Literal["absolute_integer_offset"]
    tie_break: Literal["lower_coordinate"]
    max_distance: Literal[9007199254740991]
    max_axis_coordinates: Literal[8192]
    max_target_positions: Literal[16384]
    max_changed_axis_positions: Literal[32768]
    max_request_bytes: Literal[1048576]
    semantic_policy: Literal["v2"]
    anchor: Literal["required_same_axis_order"]
    units_and_references: Literal["exact"]
    reference_conversion: Literal[False]
    mapped_values: Literal["source_values_or_explicit_fill"]
    mapped_validity: Literal["source_bits_or_false"]
    preprocessing_identity: Literal["cleared"]
    execution: Literal["eager_full_source_reads"]
    output_commits_atomic: Literal[False]
    independently_authenticated: Literal[False]


class ReferenceSemanticsCapabilities(Model):
    plan_operations: list[str]
    alignment: Literal["shared_axes_exact_positional"]
    guided_alignment: Literal[True]
    guided_alignment_modes: list[AxisCoordinateMode]
    guided_alignment_value_scope: Literal["raw_unmasked"]
    paired_validity_transforms: Literal[True]
    paired_validity_coordinate_window: Literal[True]
    paired_validity_coordinate_window_execution_modes: list[Literal["eager", "chunked_verified"]]
    paired_validity_operations: list[
        Literal[
            "select_indices",
            "slice",
            "chunked_slice",
            "transpose",
            "reindex_coordinates",
            "convert_axis_unit",
            "convert_value_unit",
            "convert_temperature_unit",
            "cast",
            "scalar_arithmetic",
            "normalize",
            "standardize",
            "sum",
            "mean",
            "min",
            "max",
            "variance",
            "std",
            "chunked_count_nonzero_by_axis",
            "chunked_sum_by_axis",
            "chunked_mean_by_axis",
            "chunked_variance_by_axis",
            "chunked_std_by_axis",
            "chunked_min_by_axis",
            "chunked_max_by_axis",
            "add",
            "subtract",
            "multiply",
            "divide",
        ]
    ]
    paired_validity_coordinate_modes: list[
        Literal[
            "exact",
            "inner_indexed",
            "inner_indexed_permuted",
            "left_filled",
            "right_filled",
            "left_filled_permuted",
            "right_filled_permuted",
        ]
    ]
    paired_validity_selected_preprocessing_policies: list[Literal["bound_selected_v1"]] = Field(
        default_factory=list
    )
    paired_validity_pinned_reductions: list[
        Literal[
            "sum",
            "mean",
            "min",
            "max",
            "variance",
            "std",
            "chunked_count_nonzero_by_axis",
            "chunked_sum_by_axis",
            "chunked_mean_by_axis",
            "chunked_variance_by_axis",
            "chunked_std_by_axis",
            "chunked_min_by_axis",
            "chunked_max_by_axis",
        ]
    ] = Field(default_factory=list)
    paired_validity_pinned_positionals: list[
        Literal["transpose", "slice", "chunked_slice", "select_indices"]
    ] = Field(default_factory=list)
    paired_validity_pinned_conversions: list[
        Literal["convert_axis_unit", "convert_value_unit", "convert_temperature_unit"]
    ] = Field(default_factory=list)
    paired_validity_pinned_scaling: list[Literal["normalize", "standardize"]] = Field(
        default_factory=list
    )
    paired_validity_pinned_cast: list[Literal["cast"]] = Field(default_factory=list)
    paired_validity_pinned_scalar: list[Literal["scalar_arithmetic"]] = Field(default_factory=list)
    paired_validity_nearest: NearestCoordinateCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    reference_conversion: Literal[False]
    xarray_projection: Literal[False]
    policy: str


class SemanticsCapabilities(Model):
    metadata_key: Literal["tensorus_semantics"]
    descriptor_versions: list[int]
    plan_policy: Literal["v1"]
    plan_policies: list[Literal["v1", "v2"]]
    descriptor_schema: dict[str, JsonValue]
    reference_semantics: ReferenceSemanticsCapabilities
    legacy_policy: str


class TensorEnvelopeCapabilities(Model):
    format: Literal["tensorus-tensor-envelope-v1"]
    maximum_rank: Literal[16]
    scalar: Literal[True]
    zero_sized_dimensions: Literal[True]
    finite_only: Literal[True]
    canonical_order: Literal["C"]
    canonical_byte_order: Literal["little"]
    explicit_validity_only: Literal[True]
    masked_array_input: Literal[False]
    sparse: Literal[False]
    ragged: Literal[False]
    complex: Literal[False]
    gpu: Literal[False]
    eager_source_byte_limit: int
    conversion_policy: str
    policy: str


class DataProductCapabilities(Model):
    descriptor_format: Literal["tensorus-data-product-v1"]
    relationship_descriptor_format: Literal["tensorus-data-product-v2"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    request_schema: dict[str, JsonValue]
    member_limit: int
    relationship_schema: dict[str, JsonValue] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    relationship_limit: int | None = Field(default=None, exclude_if=lambda value: value is None)
    product_limit: int
    revision_limit: int
    tenant_revision_limit: int
    descriptor_byte_limit: int
    release_report_format: Literal["tensorus-product-release-report-v1"]
    release_reference_report_format: Literal["tensorus-product-release-report-v2"]
    release_outcome_report_format: Literal["tensorus-product-release-report-v3"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    release_scoped_outcome_report_format: Literal["tensorus-product-release-report-v4"] | None = (
        Field(default=None, exclude_if=lambda value: value is None)
    )
    release_coordinate_outcome_report_format: (
        Literal["tensorus-product-release-report-v5"] | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    release_relationship_report_format: Literal["tensorus-product-release-report-v6"] | None = (
        Field(default=None, exclude_if=lambda value: value is None)
    )
    release_outcome_plan_limit: int | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    release_outcome_byte_limit: int | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    release_request_schema: dict[str, JsonValue]
    portable_release_format: Literal["tensorus-portable-product-release-v1"]
    portable_reference_release_format: Literal["tensorus-portable-product-release-v2"]
    portable_outcome_release_format: Literal["tensorus-portable-product-release-v3"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    portable_scoped_outcome_release_format: (
        Literal["tensorus-portable-product-release-v4"] | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    portable_coordinate_outcome_release_format: (
        Literal["tensorus-portable-product-release-v5"] | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    portable_relationship_release_format: Literal["tensorus-portable-product-release-v6"] | None = (
        Field(default=None, exclude_if=lambda value: value is None)
    )
    release_import_schema: dict[str, JsonValue]
    release_limit: int
    tenant_release_limit: int
    release_import_limit: int
    tenant_release_import_limit: int
    portable_release_byte_limit: int
    policy: str


class ImportedHistoryNameLookupCapabilities(Model):
    format: Literal["tensorus-imported-history-name-lookup-v1"]
    path: Literal["/v1/datasets/{dataset}/release-history/imports/by-name/{name}"]
    checks_membership_and_access: Literal[True]
    verifies_payload: Literal[False]


class PortableExportResponseFenceCapabilities(Model):
    format: Literal["tensorus-portable-export-response-fence-v1"]
    release_path: Literal["/v1/releases/{release_id}/portable"]
    history_path: Literal["/v1/datasets/{dataset}/release-history/export"]
    product_release_path: Literal["/v1/product-releases/{release_id}/portable"]
    rechecks_principal_and_selected_evidence: Literal[True]
    rereads_payloads: Literal[False]
    atomic_with_delivery: Literal[False]

    @field_validator(
        "rechecks_principal_and_selected_evidence",
        "rereads_payloads",
        "atomic_with_delivery",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Portable export response fence flags must be booleans")
        return value


PLAN_STATUS_RESPONSE_ROUTES = (
    "GET /v1/plans/{plan_id}",
    "GET /v1/plans/{plan_id}/capacity-review",
    "GET /v1/plans/{plan_id}/price-scenario",
    "GET /v1/plans/{plan_id}/output-reservations",
    "GET /v1/plans/{plan_id}/worker-attempts",
    "GET /v1/plans/{plan_id}/worker-claims",
    "GET /v1/plans/{plan_id}/estimated-attempts",
    "GET /v1/plans/{plan_id}/usage-observations",
    "GET /v1/plans/{plan_id}/process-usage-observations",
    "GET /v1/plans/{plan_id}/usage-handoff",
    "GET /v1/plans/{plan_id}/adjusted-usage-handoff",
    "GET /v1/plans/{plan_id}/usage-adjustments",
)


class PlanStatusResponseFenceCapabilities(Model):
    format: Literal["tensorus-plan-status-response-fence-v1"]
    routes: list[str] = Field(
        min_length=len(PLAN_STATUS_RESPONSE_ROUTES), max_length=len(PLAN_STATUS_RESPONSE_ROUTES)
    )
    authorized_read_repeated_after_serialization: Literal[True]
    captured_response_compared: Literal[False]
    rereads_payloads: Literal[False]
    atomic_with_delivery: Literal[False]
    all_plan_routes_covered: Literal[False]

    @field_validator(
        "authorized_read_repeated_after_serialization",
        "captured_response_compared",
        "rereads_payloads",
        "atomic_with_delivery",
        "all_plan_routes_covered",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Plan status response fence flags must be booleans")
        return value

    @model_validator(mode="after")
    def checked_routes(self) -> Self:
        if tuple(self.routes) != PLAN_STATUS_RESPONSE_ROUTES:
            raise ValueError("Plan status response fence routes are inconsistent")
        return self


PRODUCT_READ_RESPONSE_ROUTES = (
    *PRODUCT_DESCRIPTOR_READ_ROUTES,
    "GET /v1/data-products/{name}/releases",
    "GET /v1/product-releases/{release_id}",
    "GET /v1/data-products/{name}/release-imports",
    "GET /v1/product-release-imports/{import_id}",
)


class _ProductReadResponseFenceBase(Model):
    routes: list[str] = Field(
        min_length=len(PRODUCT_READ_RESPONSE_ROUTES),
        max_length=len(PRODUCT_READ_RESPONSE_ROUTES),
    )
    rechecks_authority_and_captured_response: Literal[True]
    rereads_payloads: Literal[False]
    atomic_with_delivery: Literal[False]
    all_product_routes_covered: Literal[False]

    @field_validator(
        "rechecks_authority_and_captured_response",
        "rereads_payloads",
        "atomic_with_delivery",
        "all_product_routes_covered",
        "all_routes_admitted",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Product read response fence flags must be booleans")
        return value

    @model_validator(mode="after")
    def checked_routes(self) -> Self:
        if tuple(self.routes) != PRODUCT_READ_RESPONSE_ROUTES:
            raise ValueError("Product read response fence routes are inconsistent")
        return self


class ProductReadResponseFenceCapabilities(_ProductReadResponseFenceBase):
    format: Literal["tensorus-product-read-response-fence-v1"]
    all_routes_admitted: Literal[False]


class ProductReadResponseFenceV2Capabilities(_ProductReadResponseFenceBase):
    format: Literal["tensorus-product-read-response-fence-v2"]
    all_routes_admitted: Literal[True]


class ReleaseCapabilities(Model):
    contract_schema: dict[str, JsonValue]
    quality_schema: dict[str, JsonValue]
    report_format: Literal["tensorus-release-report-v1"]
    portable_format: Literal["tensorus-portable-release-v1"]
    portable_lineage_format: Literal["tensorus-portable-release-v2"]
    portable_lineage_version_limit: int
    portable_lineage_edge_limit: int
    portable_lineage_depth_limit: int
    portable_validity_format: Literal["tensorus-portable-release-v3"]
    portable_multi_validity_format: Literal["tensorus-portable-release-v5"]
    portable_aligned_validity_format: Literal["tensorus-portable-release-v6"]
    portable_validity_selection_limit: int
    portable_validity_binding_limit: int
    portable_plan_format: Literal["tensorus-portable-release-v4"]
    portable_plan_selection_limit: int
    portable_import_schema: dict[str, JsonValue]
    portable_byte_limit: int
    portable_descriptor_byte_limit: int
    portable_history_format: Literal["tensorus-portable-release-history-v1"]
    portable_history_export_schema: dict[str, JsonValue]
    portable_history_import_schema: dict[str, JsonValue]
    portable_history_release_limit: int
    portable_history_byte_limit: int
    portable_history_import_byte_limit: int
    portable_history_inspection_format: Literal["tensorus-imported-history-inspection-v1"]
    portable_history_inspection_schema: dict[str, JsonValue]
    portable_history_inspection_input_byte_limit: int
    portable_history_inspection_record_byte_limit: int
    portable_history_inspection_source_byte_limit: int
    portable_history_inspection_mapping_limit: int
    portable_history_inspection_conflict_limit: int
    portable_history_inspection_report_byte_limit: int
    portable_history_inspection_policy: str
    portable_history_index_lookup_format: Literal["tensorus-imported-history-index-lookup-v1"]
    portable_history_index_lookup_schema: dict[str, JsonValue]
    portable_history_index_backfill_format: Literal["tensorus-imported-history-index-backfill-v1"]
    portable_history_index_backfill_schema: dict[str, JsonValue]
    portable_history_index_review_format: Literal["tensorus-imported-history-index-review-v1"]
    portable_history_index_review_schema: dict[str, JsonValue]
    portable_history_index_reconciliation_format: Literal[
        "tensorus-imported-history-index-reconciliation-v1"
    ]
    portable_history_index_reconciliation_schema: dict[str, JsonValue]
    portable_history_index_page_limit: int
    portable_history_index_mapping_limit: int
    portable_history_index_input_byte_limit: int
    portable_history_index_policy: str
    portable_history_scan_format: Literal["tensorus-imported-history-scan-v1"]
    portable_history_scan_schema: dict[str, JsonValue]
    portable_history_scan_policy: str
    portable_history_readable_scan_format: Literal["tensorus-imported-history-readable-scan-v1"]
    portable_history_readable_scan_policy: str
    portable_history_name_lookup: ImportedHistoryNameLookupCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    admission_policy: str


class ConditionalNpyIngestCapabilities(Model):
    format: Literal["tensorus-conditional-npy-ingest-v1"]
    path: Literal["/v1/datasets/{dataset}/tensors/{name}/conditional-npy"]
    requires_dataset_identity: Literal[True]
    requires_write_generation: Literal[True]
    requires_expected_version: Literal[True]
    requires_idempotency_key: Literal[True]
    checks_current_generation_on_retry: Literal[True]
    whole_restore_atomic: Literal[False]


class ConditionalNpyReceiptCapabilities(Model):
    format: Literal["tensorus-conditional-npy-receipt-v1"]
    path: Literal["/v1/datasets/{dataset}/tensors/{name}/conditional-npy/receipt"]
    max_request_bytes: StrictInt = Field(ge=1, le=16384)
    requires_unscoped_writer: Literal[True]
    payload_verified: Literal[False]
    retry_guaranteed: Literal[False]


class ConditionalNpyHistoricalReceiptCapabilities(Model):
    format: Literal["tensorus-conditional-npy-historical-receipt-v1"]
    path: Literal["/v1/datasets/{dataset}/tensors/{name}/conditional-npy/receipt/history"]
    max_request_bytes: StrictInt = Field(ge=1, le=16384)
    requires_unscoped_writer: Literal[True]
    payload_verified: Literal[False]
    retry_guaranteed: Literal[False]
    permits_policy_change_observation: Literal[True]


class WriteContractCapabilities(Model):
    activation_schema: dict[str, JsonValue]
    policy: str
    conditional_npy_ingest: ConditionalNpyIngestCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    conditional_npy_receipt: ConditionalNpyReceiptCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    conditional_npy_historical_receipt: ConditionalNpyHistoricalReceiptCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class EvaluationCapabilities(Model):
    request_schema: dict[str, JsonValue]
    report_format: Literal["tensorus-plan-evaluation-v1"]
    policy: str


class PlanOutcomeCapabilities(Model):
    format: Literal["tensorus-plan-outcome-v1"]
    scoped_format: Literal["tensorus-plan-outcome-v2"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scoped_query: Literal["outcome_version=v2"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scoped_observation_format: Literal["tensorus-step-observation-scope-v3"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    coordinate_format: Literal["tensorus-plan-outcome-v3"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    coordinate_query: Literal["outcome_version=v3"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    coordinate_observation_format: Literal["tensorus-step-observation-scope-v4"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    byte_limit: int
    policy: str


class TensorEvaluationCapabilities(Model):
    request_schema: dict[str, JsonValue]
    report_format: Literal["tensorus-tensor-evaluation-v1"]
    policy: str


class AlignmentCapabilities(Model):
    request_schema: dict[str, JsonValue]
    policy: str


class DescendantCapabilities(Model):
    max_depth: int
    max_nodes: int
    max_edges: int
    policy: str


class SnapshotRetirementCapabilities(Model):
    preview: Literal[True]
    requires_unscoped_writer: Literal[True]
    only_unreferenced: Literal[True]
    tensor_data_deleted: Literal[False]
    receipt_format: Literal["tensorus-snapshot-retirement-v1"]
    archive_format: Literal["tensorus-snapshot-retirement-archive-v1"]
    archive_requires_unscoped_admin: Literal[True]
    archive_selection_schema: dict[str, JsonValue]
    archive_receipt_limit: int
    archive_byte_limit: int
    archive_file_byte_limit: int
    archive_policy: str
    policy: str


class RetentionLifecycleCapabilities(Model):
    hold_format: Literal["tensorus-retention-hold-v1"]
    snapshot_review_format: Literal["tensorus-snapshot-retirement-review-v2"]
    product_release_review_format: Literal["tensorus-product-release-retirement-review-v1"]
    product_release_retirement_format: Literal["tensorus-product-release-retirement-v1"]
    supported_hold_resources: list[Literal["snapshot", "product_release"]]
    requires_unscoped_admin: Literal[True]
    holds_per_resource: Annotated[StrictInt, Field(ge=1)]
    holds_per_workspace: Annotated[StrictInt, Field(ge=1)]
    product_release_retirements_per_workspace: Annotated[StrictInt, Field(ge=1)]
    product_release_retirement_page_limit: Annotated[StrictInt, Field(ge=1)]
    policy: str


class ProductRevisionRetirementCapabilities(Model):
    format: Literal[
        "tensorus-product-revision-retirement-v1", "tensorus-product-revision-retirement-v2"
    ] = "tensorus-product-revision-retirement-v2"
    review_format: Literal[
        "tensorus-product-revision-retirement-review-v1",
        "tensorus-product-revision-retirement-review-v2",
    ] = "tensorus-product-revision-retirement-review-v2"
    authority: Literal["active_unscoped_tenant_admin"] = "active_unscoped_tenant_admin"
    max_receipts: Literal[1000] = 1000
    max_page: Literal[16] = 16
    descriptor_retained: Literal[True] = True
    tensor_data_deleted: Literal[False] = False
    quota_released: Literal[False] = False

    @model_validator(mode="after")
    def matching_formats(self) -> Self:
        if self.format.rsplit("-", 1)[-1] != self.review_format.rsplit("-", 1)[-1]:
            raise ValueError("Product-revision retirement capability formats disagree")
        return self


class ProductRetirementCapabilities(Model):
    format: Literal["tensorus-product-retirement-v1"] = "tensorus-product-retirement-v1"
    review_format: Literal["tensorus-product-retirement-review-v1"] = (
        "tensorus-product-retirement-review-v1"
    )
    authority: Literal["active_unscoped_tenant_admin"] = "active_unscoped_tenant_admin"
    max_products: Literal[128] = 128
    max_revisions: Literal[64] = 64
    max_page: Literal[16] = 16
    max_request_bytes: Annotated[StrictInt, Field(ge=1024, le=MAX_CONTROL_REQUEST_BYTES)]
    historical_revisions_must_be_retired: Literal[True] = True
    descriptors_retained: Literal[True] = True
    tensor_data_deleted: Literal[False] = False
    quota_released: Literal[False] = False
    exported_copies_revoked: Literal[False] = False


class QualityReleaseRetirementCapabilities(Model):
    format: Literal["tensorus-quality-release-retirement-v1"] = (
        "tensorus-quality-release-retirement-v1"
    )
    review_format: Literal["tensorus-quality-release-retirement-review-v1"] = (
        "tensorus-quality-release-retirement-review-v1"
    )
    authority: Literal["active_unscoped_tenant_admin"] = "active_unscoped_tenant_admin"
    max_receipts: Literal[1000] = 1000
    max_page: Literal[16] = 16
    max_request_bytes: Annotated[StrictInt, Field(ge=1024, le=MAX_CONTROL_REQUEST_BYTES)]
    source_evidence_retained: Literal[True] = True
    product_references_block: Literal[True] = True
    snapshot_holds_block: Literal[True] = True
    snapshot_deleted: Literal[False] = False
    tensor_data_deleted: Literal[False] = False
    exported_copies_revoked: Literal[False] = False
    source_authenticated: Literal[False] = False
    payloads_reverified: Literal[False] = False

    @field_validator(
        "source_evidence_retained",
        "product_references_block",
        "snapshot_holds_block",
        "snapshot_deleted",
        "tensor_data_deleted",
        "exported_copies_revoked",
        "source_authenticated",
        "payloads_reverified",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Quality-release retirement capability flags must be booleans")
        return value


class ProductRevisionHoldCapabilities(Model):
    format: Literal["tensorus-product-revision-hold-v1"] = "tensorus-product-revision-hold-v1"
    history_format: Literal["tensorus-product-revision-hold-history-v1"] = (
        "tensorus-product-revision-hold-history-v1"
    )
    authority: Literal["active_unscoped_tenant_admin"] = "active_unscoped_tenant_admin"
    holds_per_revision: Literal[64] = 64
    workspace_hold_limit: Literal[10000] = 10000
    retirement_only: Literal[True] = True
    exports_revoked: Literal[False] = False


class UploadPreallocatedChunksCapabilities(Model):
    format: Literal["tensorus-upload-preallocated-chunks-v1"]
    enabled: StrictBool
    catalog_work_reservation_backed: Literal[True]
    max_chunks: StrictInt = Field(ge=1)
    max_filesystem_fragment_bytes: StrictInt = Field(ge=1)
    slot_rule: Literal["twice-decoded-plus-4096-v1"]
    metadata_reserved: Literal[False]
    eager_output_reserved: Literal[False]
    whole_host_reserved: Literal[False]
    policy: str = Field(min_length=1)

    @field_validator(
        "catalog_work_reservation_backed",
        "metadata_reserved",
        "eager_output_reserved",
        "whole_host_reserved",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Chunk reservation flags must be booleans")
        return value


class ControlRequestBodyCapabilities(Model):
    format: Literal["tensorus-control-request-body-limit-v1"]
    max_body_bytes: StrictInt = Field(ge=1024, le=MAX_CONTROL_REQUEST_BYTES)
    routes: dict[str, list[str]]
    before_authentication: Literal[True]
    streamed_bodies_bounded: Literal[True]
    proxy_buffers_bounded: Literal[False]
    asgi_buffers_bounded: Literal[False]
    aggregate_memory_bounded: Literal[False]

    @field_validator(
        "before_authentication",
        "streamed_bodies_bounded",
        "proxy_buffers_bounded",
        "asgi_buffers_bounded",
        "aggregate_memory_bounded",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Control request limit flags must be booleans")
        return value

    @model_validator(mode="after")
    def exact_routes(self) -> Self:
        if self.routes != {
            method: list(routes) for method, routes in CONTROL_REQUEST_ROUTES.items()
        }:
            raise ValueError("Control request routes are inconsistent")
        return self


class JsonRequestBodyCapabilities(Model):
    format: Literal["tensorus-json-request-body-limit-v1"]
    max_body_bytes: StrictInt = Field(ge=MAX_CONTROL_REQUEST_BYTES, le=3 * 1024**3)
    global_body_bytes: StrictInt = Field(ge=MAX_CONTROL_REQUEST_BYTES, le=3 * 1024**3)
    methods: list[Literal["POST", "PUT", "PATCH"]]
    media_types: Literal["application-json-plus-json-or-absent"]
    before_authentication: Literal[True]
    duplicate_content_type_rejected: Literal[True]
    streamed_bodies_bounded: Literal[True]
    proxy_buffers_bounded: Literal[False]
    asgi_buffers_bounded: Literal[False]
    aggregate_memory_bounded: Literal[False]

    @field_validator(
        "before_authentication",
        "duplicate_content_type_rejected",
        "streamed_bodies_bounded",
        "proxy_buffers_bounded",
        "asgi_buffers_bounded",
        "aggregate_memory_bounded",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("JSON request limit flags must be booleans")
        return value

    @model_validator(mode="after")
    def validate_limit(self) -> Self:
        if self.max_body_bytes > self.global_body_bytes:
            raise ValueError("JSON request limit exceeds the global body limit")
        if self.methods != ["POST", "PUT", "PATCH"]:
            raise ValueError("JSON request limit methods are inconsistent")
        return self


class RequestBodyStagingBase(Model):
    configured: Literal[True]
    concurrent_slots: StrictInt = Field(ge=1, le=64)
    max_content_bytes_per_request: StrictInt = Field(ge=1024, le=3 * 1024**3)
    max_inflight_content_bytes: StrictInt = Field(ge=1024, le=64 * 3 * 1024**3)
    max_filesystem_fragment_bytes: Literal[1048576]
    preallocated_before_receive: Literal[True]
    anonymous_files: Literal[True]
    exact_content_length: Literal[True]
    single_api_process: Literal[True]
    filesystem_metadata_reserved: Literal[False]
    other_host_usage_reserved: Literal[False]
    policy: str = Field(min_length=1)

    @field_validator(
        "configured",
        "preallocated_before_receive",
        "anonymous_files",
        "exact_content_length",
        "single_api_process",
        "filesystem_metadata_reserved",
        "other_host_usage_reserved",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Request staging capability flags must be booleans")
        return value

    @model_validator(mode="after")
    def validate_capacity(self) -> Self:
        if self.max_inflight_content_bytes != (
            self.concurrent_slots * self.max_content_bytes_per_request
        ):
            raise ValueError("Request staging content capacity is inconsistent")
        return self


class RequestBodyStagingCapabilities(RequestBodyStagingBase):
    format: Literal["tensorus-reserved-http-body-v1"]
    participating_unreserved_servers_blocked: Literal[True]

    @field_validator("participating_unreserved_servers_blocked", mode="before")
    @classmethod
    def strict_fence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Request staging fence must be a boolean")
        return value


class RemoteRequestBodyStagingCapabilities(RequestBodyStagingBase):
    format: Literal["tensorus-reserved-http-body-local-fence-v2"]
    same_staging_parent_processes_blocked: Literal[True]
    other_staging_directories_blocked: Literal[False]
    fleet_capacity_bounded: Literal[False]

    @field_validator(
        "same_staging_parent_processes_blocked",
        "other_staging_directories_blocked",
        "fleet_capacity_bounded",
        mode="before",
    )
    @classmethod
    def strict_fence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Request staging fence flags must be booleans")
        return value


class RequestBodyStagingDeadlineCapabilities(Model):
    format: Literal["tensorus-http-staging-deadline-v1"]
    staging_timeout_seconds: StrictInt = Field(ge=1, le=3600)
    includes_file_allocation: Literal[True]
    includes_receiving: Literal[True]
    includes_finish: Literal[True]
    covers_endpoint_processing: Literal[False]
    interrupts_running_io: Literal[False]
    immediate_slot_release: Literal[False]

    @field_validator(
        "includes_file_allocation",
        "includes_receiving",
        "includes_finish",
        "covers_endpoint_processing",
        "interrupts_running_io",
        "immediate_slot_release",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Request staging deadline flags must be booleans")
        return value


class EagerIngestStagingCapabilities(Model):
    format: Literal["tensorus-eager-ingest-preallocated-chunks-v1"]
    configured: StrictBool
    work_token_bytes: StrictInt = Field(ge=1024, le=16 * 1024**3)
    work_pool_capacity_bytes: StrictInt | None = Field(ge=1024, le=2**50)
    max_chunks: StrictInt = Field(ge=1)
    max_filesystem_fragment_bytes: StrictInt = Field(ge=1)
    slot_rule: Literal["twice-decoded-plus-4096-v1"]
    shares_upload_work_pool: Literal[True]
    catalog_bound_scratch: Literal[True]
    filesystem_metadata_reserved: Literal[False]
    eager_input_memory_reserved: Literal[False]
    request_body_staging_reserved: Literal[False]
    whole_host_reserved: Literal[False]
    policy: str = Field(min_length=1)

    @field_validator(
        "shares_upload_work_pool",
        "catalog_bound_scratch",
        "filesystem_metadata_reserved",
        "eager_input_memory_reserved",
        "request_body_staging_reserved",
        "whole_host_reserved",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Eager ingestion staging flags must be booleans")
        return value

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        if self.configured and self.work_pool_capacity_bytes is None:
            raise ValueError("Eager ingestion staging requires a configured work pool")
        return self


class UploadBoundScratchCapabilities(Model):
    format: Literal["tensorus-upload-reservation-scratch-v1"]
    enabled: StrictBool
    catalog_owner_pinned: Literal[True]
    cleanup_entries: StrictInt = Field(ge=1)
    cleanup_bytes: StrictInt = Field(ge=1)
    cleanup_depth: StrictInt = Field(ge=1)
    prior_boot_recovery: Literal[False]
    staging_capacity_reserved: Literal[False]
    preallocated_chunks: UploadPreallocatedChunksCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    policy: str = Field(min_length=1)

    @field_validator(
        "catalog_owner_pinned", "prior_boot_recovery", "staging_capacity_reserved", mode="before"
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Scratch capability flags must be booleans")
        return value

    @model_validator(mode="after")
    def validate_chunk_configuration(self) -> Self:
        if self.preallocated_chunks is not None and (
            self.preallocated_chunks.enabled and not self.enabled
        ):
            raise ValueError("Chunk preallocation requires configured scratch ownership")
        return self


class UploadInputReservationCapabilities(Model):
    format: Literal["tensorus-upload-input-reservation-v1"]
    configured: bool
    input_capacity_bytes: StrictInt | None = Field(ge=1024, le=2**50)
    work_capacity_bytes: StrictInt | None = Field(ge=1024, le=2**50)
    global_staging_capacity_reserved: Literal[False]
    catalog_bound_scratch: UploadBoundScratchCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    policy: str = Field(min_length=1)

    @field_validator("global_staging_capacity_reserved", mode="before")
    @classmethod
    def strict_global_capacity_flag(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("The global capacity flag must be a boolean")
        return value

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        if self.configured != (
            self.input_capacity_bytes is not None and self.work_capacity_bytes is not None
        ) or (self.input_capacity_bytes is None) != (self.work_capacity_bytes is None):
            raise ValueError("Upload input pool configuration is inconsistent")
        if self.catalog_bound_scratch is not None and (
            self.catalog_bound_scratch.enabled != self.configured
        ):
            raise ValueError("Catalog-bound scratch disagrees with input pool configuration")
        return self


class UploadFinalizerCapabilities(Model):
    format: Literal["tensorus-upload-finalizer-v1"]
    lease_seconds: int
    wait_seconds: float
    catalog_publication_fenced: Literal[True]
    in_flight_storage_writes_fenced: Literal[False]
    physical_capacity_reserved: Literal[False]
    input_file_preallocation: bool
    input_reservation: UploadInputReservationCapabilities

    @model_validator(mode="after")
    def validate_reservation_preallocation(self) -> Self:
        if self.input_reservation.configured and not self.input_file_preallocation:
            raise ValueError("Upload input pool requires input preallocation")
        return self


class NpyDownloadCapabilities(Model):
    byte_ranges: bool
    resumable_uploads: bool
    streaming_uploads: bool
    streaming_upload_tensor_bytes: int
    streaming_upload_work_bytes: int
    streaming_scratch_auto_recovery: bool
    finalizer_ownership: UploadFinalizerCapabilities
    policy: str


class NpyDownloadAdmissionCapabilities(Model):
    format: Literal["tensorus-npy-download-admission-v1"]
    pool_bytes: StrictInt = Field(ge=1024, le=2**50)
    tenant_bytes: StrictInt = Field(ge=1024, le=2**50)
    single_api_process: Literal[True]
    ranges_account_full_tensor: Literal[True]
    active_until_response_and_handler_finish: Literal[True]
    inflight_catalog_checks_hold_admission: Literal[True]
    physical_memory_reserved: Literal[False]
    process_memory_bounded: Literal[False]
    other_endpoints_covered: Literal[False]
    policy: str = Field(min_length=1)

    @field_validator(
        "single_api_process",
        "ranges_account_full_tensor",
        "active_until_response_and_handler_finish",
        "inflight_catalog_checks_hold_admission",
        "physical_memory_reserved",
        "process_memory_bounded",
        "other_endpoints_covered",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("NPY download admission flags must be booleans")
        return value

    @model_validator(mode="after")
    def validate_capacity(self) -> Self:
        if self.tenant_bytes > self.pool_bytes:
            raise ValueError("NPY download tenant capacity exceeds the shared pool")
        return self


class SimilarityAdmissionCapabilities(Model):
    format: Literal["tensorus-similarity-admission-v1"]
    slots: StrictInt = Field(ge=1, le=MAX_SIMILARITY_ADMISSION_SLOTS)
    tenant_slots: StrictInt = Field(ge=1, le=MAX_SIMILARITY_ADMISSION_SLOTS)
    routes: list[str] = Field(min_length=3, max_length=3)
    single_api_process: Literal[True]
    active_until_response_and_handler_finish: Literal[True]
    fair_queue: Literal[False]
    physical_memory_reserved: Literal[False]
    process_memory_bounded: Literal[False]
    other_endpoints_covered: Literal[False]
    policy: str = Field(min_length=1)

    @field_validator(
        "single_api_process",
        "active_until_response_and_handler_finish",
        "fair_queue",
        "physical_memory_reserved",
        "process_memory_bounded",
        "other_endpoints_covered",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Similarity admission flags must be booleans")
        return value

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.tenant_slots > self.slots or tuple(self.routes) != SIMILARITY_ADMISSION_ROUTES:
            raise ValueError("Similarity admission capacities or routes are inconsistent")
        return self


class _ScientificReadAdmissionCapabilities(Model):
    slots: StrictInt = Field(ge=1, le=MAX_SCIENTIFIC_READ_SLOTS)
    tenant_slots: StrictInt = Field(ge=1, le=MAX_SCIENTIFIC_READ_SLOTS)
    single_api_process: Literal[True]
    active_until_response_and_handler_finish: Literal[True]
    inflight_catalog_checks_hold_admission: Literal[True]
    fair_queue: Literal[False]
    physical_memory_reserved: Literal[False]
    process_memory_bounded: Literal[False]
    other_endpoints_covered: Literal[False]
    policy: str = Field(min_length=1)

    @field_validator(
        "single_api_process",
        "active_until_response_and_handler_finish",
        "inflight_catalog_checks_hold_admission",
        "fair_queue",
        "physical_memory_reserved",
        "process_memory_bounded",
        "other_endpoints_covered",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Scientific read admission flags must be booleans")
        return value

    @model_validator(mode="after")
    def validate_capacities(self) -> Self:
        if self.tenant_slots > self.slots:
            raise ValueError("Scientific read admission capacities are inconsistent")
        return self


class ScientificReadAdmissionCapabilities(_ScientificReadAdmissionCapabilities):
    format: Literal["tensorus-scientific-read-admission-v1"]
    routes: list[str] = Field(
        min_length=len(SCIENTIFIC_READ_ROUTES), max_length=len(SCIENTIFIC_READ_ROUTES)
    )

    @model_validator(mode="after")
    def validate_routes(self) -> Self:
        if tuple(self.routes) != SCIENTIFIC_READ_ROUTES:
            raise ValueError("Scientific read admission routes are inconsistent")
        return self


class ScientificReadAdmissionV2Capabilities(_ScientificReadAdmissionCapabilities):
    format: Literal["tensorus-scientific-read-admission-v2"]
    routes: list[str] = Field(
        min_length=len(SCIENTIFIC_READ_ROUTES_V2), max_length=len(SCIENTIFIC_READ_ROUTES_V2)
    )

    @model_validator(mode="after")
    def validate_routes(self) -> Self:
        if tuple(self.routes) != SCIENTIFIC_READ_ROUTES_V2:
            raise ValueError("Scientific read admission routes are inconsistent")
        return self


class ScientificReadAdmissionV3Capabilities(_ScientificReadAdmissionCapabilities):
    format: Literal["tensorus-scientific-read-admission-v3"]
    routes: list[str] = Field(
        min_length=len(SCIENTIFIC_READ_ROUTES_V3), max_length=len(SCIENTIFIC_READ_ROUTES_V3)
    )

    @model_validator(mode="after")
    def validate_routes(self) -> Self:
        if tuple(self.routes) != SCIENTIFIC_READ_ROUTES_V3:
            raise ValueError("Scientific read admission routes are inconsistent")
        return self


class ScientificReadAdmissionV4Capabilities(_ScientificReadAdmissionCapabilities):
    format: Literal["tensorus-scientific-read-admission-v4"]
    routes: list[str] = Field(
        min_length=len(SCIENTIFIC_READ_ROUTES_V4), max_length=len(SCIENTIFIC_READ_ROUTES_V4)
    )

    @model_validator(mode="after")
    def validate_routes(self) -> Self:
        if tuple(self.routes) != SCIENTIFIC_READ_ROUTES_V4:
            raise ValueError("Scientific read admission routes are inconsistent")
        return self


class ScientificReadAdmissionV5Capabilities(_ScientificReadAdmissionCapabilities):
    format: Literal["tensorus-scientific-read-admission-v5"]
    routes: list[str] = Field(
        min_length=len(SCIENTIFIC_READ_ROUTES_V5), max_length=len(SCIENTIFIC_READ_ROUTES_V5)
    )

    @model_validator(mode="after")
    def validate_routes(self) -> Self:
        if tuple(self.routes) != SCIENTIFIC_READ_ROUTES_V5:
            raise ValueError("Scientific read admission routes are inconsistent")
        return self


class ScientificReadAdmissionV6Capabilities(_ScientificReadAdmissionCapabilities):
    format: Literal["tensorus-scientific-read-admission-v6"]
    routes: list[str] = Field(
        min_length=len(SCIENTIFIC_READ_ROUTES_V6), max_length=len(SCIENTIFIC_READ_ROUTES_V6)
    )

    @model_validator(mode="after")
    def validate_routes(self) -> Self:
        if tuple(self.routes) != SCIENTIFIC_READ_ROUTES_V6:
            raise ValueError("Scientific read admission routes are inconsistent")
        return self


class ScientificReadAdmissionV7Capabilities(_ScientificReadAdmissionCapabilities):
    format: Literal["tensorus-scientific-read-admission-v7"]
    routes: list[str] = Field(
        min_length=len(SCIENTIFIC_READ_ROUTES_V7), max_length=len(SCIENTIFIC_READ_ROUTES_V7)
    )

    @model_validator(mode="after")
    def validate_routes(self) -> Self:
        if tuple(self.routes) != SCIENTIFIC_READ_ROUTES_V7:
            raise ValueError("Scientific read admission routes are inconsistent")
        return self


class ScientificReadAdmissionV8Capabilities(_ScientificReadAdmissionCapabilities):
    format: Literal["tensorus-scientific-read-admission-v8"]
    routes: list[str] = Field(
        min_length=len(SCIENTIFIC_READ_ROUTES_V8), max_length=len(SCIENTIFIC_READ_ROUTES_V8)
    )

    @model_validator(mode="after")
    def validate_routes(self) -> Self:
        if tuple(self.routes) != SCIENTIFIC_READ_ROUTES_V8:
            raise ValueError("Scientific read admission routes are inconsistent")
        return self


class ScientificReadAdmissionV9Capabilities(_ScientificReadAdmissionCapabilities):
    format: Literal["tensorus-scientific-read-admission-v9"]
    routes: list[str] = Field(
        min_length=len(SCIENTIFIC_READ_ROUTES_V9), max_length=len(SCIENTIFIC_READ_ROUTES_V9)
    )

    @model_validator(mode="after")
    def validate_routes(self) -> Self:
        if tuple(self.routes) != SCIENTIFIC_READ_ROUTES_V9:
            raise ValueError("Scientific read admission routes are inconsistent")
        return self


class ScientificReadAdmissionV10Capabilities(_ScientificReadAdmissionCapabilities):
    format: Literal["tensorus-scientific-read-admission-v10"]
    routes: list[str] = Field(
        min_length=len(SCIENTIFIC_READ_ROUTES_V10), max_length=len(SCIENTIFIC_READ_ROUTES_V10)
    )

    @model_validator(mode="after")
    def validate_routes(self) -> Self:
        if tuple(self.routes) != SCIENTIFIC_READ_ROUTES_V10:
            raise ValueError("Scientific read admission routes are inconsistent")
        return self


class ScientificReadAdmissionV11Capabilities(_ScientificReadAdmissionCapabilities):
    format: Literal["tensorus-scientific-read-admission-v11"]
    routes: list[str] = Field(
        min_length=len(SCIENTIFIC_READ_ROUTES_V11), max_length=len(SCIENTIFIC_READ_ROUTES_V11)
    )

    @model_validator(mode="after")
    def validate_routes(self) -> Self:
        if tuple(self.routes) != SCIENTIFIC_READ_ROUTES_V11:
            raise ValueError("Scientific read admission routes are inconsistent")
        return self


class ChunkedSupervisionCapabilities(Model):
    format: Literal["tensorus-supervised-chunked-local-v1"]
    enabled: bool
    supported_platform: Literal["linux"]
    scope: Literal["verified_local_chunked_operations"]
    child_address_space_bytes: int = Field(gt=0)
    output_bytes: int = Field(gt=0)
    whole_source_checksum: Literal[True]
    parent_process_memory_bounded: Literal[False]
    output_publication_in_child: Literal[False]
    remote_objects_supported: Literal[False]
    filesystem_network_isolation: Literal[False]
    policy: str


class LocalEagerSupervisionCapabilities(Model):
    format: Literal["tensorus-supervised-eager-local-v1"]
    enabled: StrictBool
    supported_platform: Literal["linux"]
    scope: Literal["verified_local_eager_operations"]
    child_address_space_bytes: StrictInt = Field(gt=0)
    input_bytes: StrictInt = Field(gt=0)
    output_bytes: StrictInt = Field(gt=0)
    decoded_work_bytes: StrictInt = Field(gt=0)
    decoded_work_policy: Literal["equal_per_source"]
    max_blocks_per_source: Literal[8192]
    max_chunk_visits_per_source: Literal[65536]
    max_scanned_positions_per_source: Literal[131072]
    max_chunks_per_block: Literal[256]
    max_decoded_bytes_per_block: Literal[16777216]
    whole_source_checksum: Literal[True]
    canonical_codecs_required: Literal[True]
    remote_objects_supported: Literal[False]
    imported_state_review_in_parent: Literal[True]
    parent_process_memory_bounded: Literal[False]
    output_publication_in_child: Literal[False]
    filesystem_network_isolation: Literal[False]
    policy: str = Field(min_length=1)

    @field_validator(
        "whole_source_checksum",
        "canonical_codecs_required",
        "remote_objects_supported",
        "imported_state_review_in_parent",
        "parent_process_memory_bounded",
        "output_publication_in_child",
        "filesystem_network_isolation",
        mode="before",
    )
    @classmethod
    def strict_boolean_evidence(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Supervision scope flags must be booleans")
        return value

    @field_validator(
        "max_blocks_per_source",
        "max_chunk_visits_per_source",
        "max_scanned_positions_per_source",
        "max_chunks_per_block",
        "max_decoded_bytes_per_block",
        mode="before",
    )
    @classmethod
    def strict_integer_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Supervision work limits must be integers")
        return value

    @field_validator("policy")
    @classmethod
    def nonblank_policy(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Supervision policy must not be blank")
        return value


class EagerSupervisionCapabilities(Model):
    format: Literal["tensorus-supervised-eager-v1"]
    enabled: bool
    supported_platform: Literal["linux"]
    scope: Literal["eager_operations_except_profile_and_chunked"]
    child_address_space_bytes: int
    input_bytes: int
    output_bytes: int
    parent_process_memory_bounded: Literal[False]
    io_and_publication_hard_stopped: Literal[False]
    filesystem_network_isolation: Literal[False]
    policy: str
    eager_local_reads: LocalEagerSupervisionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    chunked_local: ChunkedSupervisionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class SharedAxisQueryCapabilities(Model):
    format: Literal["tensorus-exact-shared-axis-query-v1"]
    kind: Literal["exact_shared_axis_v1"]
    max_axes: Literal[16]
    coordinate_values: int
    snapshot_member_required: Literal[True]


class CrossSnapshotSharedAxisCapabilities(Model):
    format: Literal["tensorus-exact-cross-snapshot-shared-axis-query-v1"]
    kind: Literal["exact_cross_snapshot_shared_axis_v1"]
    max_axes: Literal[16]
    coordinate_values: int
    candidate_and_anchor_snapshots_required: Literal[True]
    same_tenant_required: Literal[True]


class OverlapSharedAxisCapabilities(Model):
    format: Literal["tensorus-overlap-shared-axis-query-v1"]
    kind: Literal["overlap_shared_axis_v1"]
    max_axes: Literal[16]
    coordinate_values: int
    snapshot_member_required: Literal[True]


class CrossSnapshotOverlapSharedAxisCapabilities(Model):
    format: Literal["tensorus-overlap-cross-snapshot-shared-axis-query-v1"]
    kind: Literal["overlap_cross_snapshot_shared_axis_v1"]
    max_axes: Literal[16]
    coordinate_values: int
    candidate_and_anchor_snapshots_required: Literal[True]
    same_tenant_required: Literal[True]


class StreamedIndexEvaluationCapabilities(Model):
    format: Literal["tensorus-indexed-stream-evaluation-v1"]
    execution: Literal["streamed_exact_v2"]
    max_candidates: StrictInt = Field(ge=1)
    max_descriptor_bytes: StrictInt = Field(ge=1)
    fetch_batch: StrictInt = Field(ge=1)
    exact_evaluation_passes: StrictInt = Field(ge=1, le=1)
    policy: str = Field(min_length=1)


class IndexWatermarkChecksCapabilities(Model):
    format: Literal["tensorus-index-watermark-checks-v1"]
    candidate_watermarks: Literal["positive_integer_at_most_state"]
    candidate_generation: Literal["equals_state"]
    filtered_candidates_checked: Literal[True]
    zero_state_requires_no_entries: Literal[True]
    missing_state_with_entries: Literal["rejected"]
    invalid_state_blocks: Literal["query_status_rebuild_publication"]
    sqlite_numeric_storage: Literal["integer"]
    independently_authenticated: Literal[False]


class SemanticIndexCapabilities(Model):
    format: Literal["tensorus-semantic-property-index-v1"]
    properties: list[str]
    reconciliation_candidates: int
    exact_match_candidates: int
    rebuild_versions: int
    rebuild_descriptor_bytes: int
    query_reconciles_source: Literal[True]
    automatic_legacy_backfill: Literal[False]
    snapshot_cursor_format: Literal["tensorus-indexed-snapshot-cursor-v1"]
    watermark_checks: IndexWatermarkChecksCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    shared_axis: SharedAxisQueryCapabilities
    cross_snapshot_shared_axis: CrossSnapshotSharedAxisCapabilities
    overlap_shared_axis: OverlapSharedAxisCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    cross_snapshot_overlap_shared_axis: CrossSnapshotOverlapSharedAxisCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    streamed_evaluation: StreamedIndexEvaluationCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    policy: str


class RawSimilarityAlgorithms(Model):
    cosine: Literal["scaled-longdouble-v1"]
    l2: Literal["difference-scaled-longdouble-v2"]


class SnapshotSimilarityCapabilities(Model):
    report_format: Literal["tensorus-snapshot-similarity-v2"]
    request_schema: dict[str, JsonValue]
    report_byte_limit: int
    score_algorithms: RawSimilarityAlgorithms
    policy: str


class ValiditySnapshotSimilarityCapabilities(Model):
    report_format: Literal["tensorus-validity-snapshot-similarity-v1"]
    request_schema: dict[str, JsonValue]
    report_byte_limit: int
    candidate_limit: int
    element_limit: int
    intersection_work_limit: int
    policy: str


class ReferenceEvaluationCapabilities(Model):
    format: Literal["tensorus-reference-evaluation-v1"]
    request_schema: dict[str, JsonValue]
    request_byte_limit: int
    report_byte_limit: int
    tenant_limit: int
    archive_format: Literal["tensorus-reference-evaluation-archive-v1"]
    archive_tenant_limit: int
    archive_page_limit: int
    retirement_review_format: Literal["tensorus-reference-retirement-review-v1"]
    policy: str


class ProductSimilarityEvaluationCapabilities(Model):
    report_format: Literal["tensorus-product-similarity-evaluation-v1"]
    request_schema: dict[str, JsonValue]
    request_byte_limit: int
    report_byte_limit: int
    product_limit: int
    tenant_limit: int
    withdrawal_format: Literal["tensorus-product-similarity-withdrawal-v1"]
    withdrawal_request_schema: dict[str, JsonValue]
    withdrawal_tenant_limit: int
    withdrawal_page_limit: int
    withdrawn_evaluation_byte_limit: int
    withdrawal_receipt_byte_limit: int
    retirement_format: Literal["tensorus-product-similarity-retirement-v1"]
    retirement_review_format: Literal["tensorus-product-similarity-retirement-review-v1"]
    retirement_review_request_schema: dict[str, JsonValue]
    retirement_request_schema: dict[str, JsonValue]
    retirement_tenant_limit: int
    retirement_page_limit: int
    retired_evaluation_byte_limit: int
    retirement_receipt_byte_limit: int
    source_binding_review_format: Literal["tensorus-product-source-evaluation-review-v1"]
    source_binding_review_request_schema: dict[str, JsonValue]
    source_binding_request_schema: dict[str, JsonValue]
    source_binding_review_byte_limit: int
    source_binding_product_limit: int
    source_binding_tenant_limit: int
    source_binding_policy: str
    policy: str


class WorkspaceIdentityCapabilities(Model):
    format: Literal["tensorus-workspace-identity-v1"]
    enabled: bool
    session_seconds: int
    max_members: int
    assertion_single_use: Literal[True]
    identity_provider_login_included: Literal[False]


class WorkspaceLifecycleCapabilities(Model):
    format: Literal["tensorus-workspace-lockdown-v1"]
    max_unrevoked_keys: Literal[20_000]
    max_active_members: Literal[10_000]
    max_nonterminal_plans: Literal[10_000]
    max_retained_events: Literal[10_000]
    operator_recovery_only: Literal[True]
    processes_hard_stopped: Literal[False]
    restored_authority_reconciled: Literal[False]

    @field_validator(
        "max_unrevoked_keys",
        "max_active_members",
        "max_nonterminal_plans",
        "max_retained_events",
        mode="before",
    )
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Workspace lifecycle limits must be integers")
        return value

    @field_validator(
        "operator_recovery_only",
        "processes_hard_stopped",
        "restored_authority_reconciled",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Workspace lifecycle flags must be booleans")
        return value


class WorkspaceLifecycleExtendedCapabilities(Model):
    format: Literal["tensorus-workspace-lockdown-set-v2"]
    max_unrevoked_keys: Literal[1_001_000]
    max_active_members: Literal[10_000]
    max_nonterminal_plans: Literal[10_000]
    max_retained_events: Literal[10_000]
    max_request_bytes: StrictInt = Field(ge=1, le=MAX_CONTROL_REQUEST_BYTES)
    max_response_bytes: Literal[16_384]
    tenant_lock_held: Literal[True]
    catalog_transaction_atomic: Literal[True]
    database_memory_bounded: Literal[False]
    transaction_time_bounded: Literal[False]
    operator_recovery_only: Literal[True]
    processes_hard_stopped: Literal[False]
    restored_authority_reconciled: Literal[False]

    @field_validator(
        "max_unrevoked_keys",
        "max_active_members",
        "max_nonterminal_plans",
        "max_retained_events",
        "max_request_bytes",
        "max_response_bytes",
        mode="before",
    )
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Extended workspace lockdown limits must be integers")
        return value

    @field_validator(
        "tenant_lock_held",
        "catalog_transaction_atomic",
        "database_memory_bounded",
        "transaction_time_bounded",
        "operator_recovery_only",
        "processes_hard_stopped",
        "restored_authority_reconciled",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Extended workspace lockdown flags must be booleans")
        return value


class TenantFairSchedulingCapabilities(Model):
    format: Literal["tensorus-tenant-fair-scheduling-v1"]
    api_process_enabled: bool
    api_process_max_active_leases: StrictInt | None = Field(ge=1, le=1024)
    scope: Literal["participating_workers_with_matching_configuration"]
    fleet_configuration_verified: Literal[False]
    expired_attempts_hard_stopped: Literal[False]
    weighted_or_billing_usage: Literal[False]
    claim_events_recorded: Literal[True]

    @field_validator(
        "api_process_enabled",
        "fleet_configuration_verified",
        "expired_attempts_hard_stopped",
        "weighted_or_billing_usage",
        "claim_events_recorded",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Scheduling capability flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_limit(self) -> Self:
        if self.api_process_max_active_leases is not None and not self.api_process_enabled:
            raise ValueError("A tenant lease limit requires fair scheduling")
        return self


class HttpRateLimitCapabilities(Model):
    format: Literal["tensorus-process-http-rate-limit-v1"]
    enabled: bool
    requests_per_second: StrictInt | None = Field(ge=1, le=100_000)
    burst: StrictInt | None = Field(ge=1, le=1_000_000)
    scope: Literal["this_api_process_global"]
    tenant_isolation: Literal[False]
    fleet_limit: Literal[False]
    denied_requests_in_process_metrics_and_logs: Literal[True]

    @field_validator(
        "enabled",
        "tenant_isolation",
        "fleet_limit",
        "denied_requests_in_process_metrics_and_logs",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Rate-limit capability flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_limit(self) -> Self:
        if self.enabled != (self.requests_per_second is not None and self.burst is not None):
            raise ValueError("Rate-limit configuration is inconsistent")
        if (self.requests_per_second is None) != (self.burst is None):
            raise ValueError("Rate-limit configuration is inconsistent")
        return self


class HttpPeerRateLimitCapabilities(Model):
    format: Literal["tensorus-process-peer-rate-limit-v1"]
    enabled: bool
    requests_per_second: StrictInt | None = Field(ge=1, le=100_000)
    burst: StrictInt | None = Field(ge=1, le=1_000_000)
    buckets: StrictInt = Field(ge=1, le=16_384)
    peer_source: Literal["asgi_client_address"]
    forwarded_headers_used_by_middleware: Literal[False]
    unknown_peers_shared: Literal[True]
    hash_collisions_share_limits: Literal[True]
    before_authentication_and_body_staging: Literal[True]
    live_health_exempt: Literal[True]
    fleet_limit: Literal[False]
    peer_addresses_in_metrics: Literal[False]
    denials_in_process_metrics: Literal[True]
    counters_durable: Literal[False]

    @field_validator(
        "enabled",
        "forwarded_headers_used_by_middleware",
        "unknown_peers_shared",
        "hash_collisions_share_limits",
        "before_authentication_and_body_staging",
        "live_health_exempt",
        "fleet_limit",
        "peer_addresses_in_metrics",
        "denials_in_process_metrics",
        "counters_durable",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Peer rate-limit capability flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_limit(self) -> Self:
        if self.enabled != (self.requests_per_second is not None and self.burst is not None):
            raise ValueError("Peer rate-limit configuration is inconsistent")
        if (self.requests_per_second is None) != (self.burst is None):
            raise ValueError("Peer rate-limit configuration is inconsistent")
        return self


class AgentProposalBudgetCapabilities(Model):
    format: Literal["tensorus-agent-proposal-budget-v1"]
    grant_option: Literal["max_plan_proposals"]
    max_proposals: Literal[100_000]
    applies_to: Literal["committed_plan_proposals"]
    execution_resources_measured: Literal[False]
    billing_authority: Literal[False]


class AgentExecutionBudgetCapabilities(Model):
    format: Literal["tensorus-agent-execution-budget-v1"]
    grant_option: Literal["max_estimated_execution_units"]
    estimator: Literal["tensorus-plan-preflight-estimate-v1"]
    max_units: Literal[1_000_000_000_000]
    max_attempts: Literal[10_000]
    applies_to: Literal["worker_attempts"]
    refunded_on_failure: Literal[False]
    resource_use_measured: Literal[False]
    billing_authority: Literal[False]


class AgentIngestBudgetCapabilities(Model):
    format: Literal["tensorus-agent-ingest-budget-v1"]
    grant_option: Literal["max_direct_ingest_bytes"]
    max_bytes: Literal[1_000_000_000_000]
    max_versions: Literal[10_000]
    applies_to: Literal["direct_ingestion"]
    attempted_work_measured: Literal[False]
    physical_storage_measured: Literal[False]
    billing_authority: Literal[False]

    @field_validator(
        "attempted_work_measured", "physical_storage_measured", "billing_authority", mode="before"
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Ingestion allowance flags must be booleans")
        return value


class Capabilities(Model):
    version: str
    role: Role
    scope: AgentScopeValue | None = None
    storage_format: str
    snapshot_cursor_list: SnapshotCursorCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    portable_export_response_fence: PortableExportResponseFenceCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    plan_status_response_fence: PlanStatusResponseFenceCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    product_read_response_fence: ProductReadResponseFenceCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    product_read_response_fence_v2: ProductReadResponseFenceV2Capabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scoped_metadata_discovery: ScopedMetadataDiscoveryCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    plan_proposer_authority: PlanProposerAuthorityCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    control_request_body_limit: ControlRequestBodyCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    json_request_body_limit: JsonRequestBodyCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    http_body_admission: HttpBodyAdmissionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    bundle_restore_inspection: BundleInspectionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    bundle_restore_head_preflight: BundleHeadPreflightCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    bundle_restore_payload_observation: BundlePayloadObservationCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    history_release_observation: HistoryReleaseObservationCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    history_restore_selection: HistoryRestoreSelectionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    history_independent_selection: HistoryIndependentSelectionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    history_independent_observation: HistoryIndependentObservationCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    history_payload_selection: HistoryPayloadSelectionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    product_payload_selection: ProductPayloadSelectionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    bundle_restore_name_occupancy: NameOccupancyCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    bundle_restore_name_preflight: NamePreflightCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    bundle_restore_apply: BundleRestoreApplyCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    bundle_restore_preparation: BundleRestorePreparationCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    workspace_identity: WorkspaceIdentityCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    workspace_member_list_response_fence: WorkspaceMemberListResponseFenceCapabilities | None = (
        Field(default=None, exclude_if=lambda value: value is None)
    )
    workspace_member_invitation: MemberInvitationCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    workspace_member_transition: MemberTransitionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    identity_provider_admission: IdentityProviderAdmissionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    model_provider_admission: ModelProviderAdmissionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    model_provider_tenant_admission: ModelProviderTenantAdmissionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    model_provider_timeout: ModelProviderTimeoutCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    workspace_directory: WorkspaceDirectoryCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    credential_pages: CredentialPageCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    credential_pages_v2: CredentialPageV2Capabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    credential_inventory_admission: CredentialInventoryAdmissionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    credential_inventory_admission_v2: CredentialInventoryAdmissionV2Capabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    workflow_inventory_admission: WorkflowInventoryAdmissionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    history_read_admission: HistoryReadAdmissionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    product_descriptor_admission: ProductDescriptorAdmissionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    workspace_overview_response_fence: WorkspaceOverviewResponseFenceCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    workspace_overview_admission: WorkspaceOverviewAdmissionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    authenticated_get_caller_fence: AuthenticatedReadFenceCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    dataset_read_response_fence: DatasetReadFenceCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    dataset_options: DatasetOptionsCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    snapshot_read_response_fence: SnapshotReadFenceCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    history_list_admission: HistoryListAdmissionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scoped_history_summary_policy: ScopedHistorySummaryPolicyCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    service_actor_recipient_review: RecipientBindingReviewCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    workspace_service_actors: ServiceActorCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    service_actor_recipient_grants: RecipientCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    service_actor_recipient_capacity: RecipientCapacityCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    organization_governance: OrganizationCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    organization_workspace_detachment: DetachmentCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    organization_detachment_receipt_v1: DetachmentReceiptCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    organization_session_fence: (
        OrganizationSessionFenceCapabilities | OrganizationSessionFenceV2Capabilities | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    workspace_session_revocation: WorkspaceSessionRevocationCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    workspace_departure: DepartureCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    workspace_departure_receipt_v2: DepartureReceiptRecoveryCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    workspace_identity_sessions: IdentitySessionsCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    key_rotation: KeyRotationCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    workspace_lifecycle: WorkspaceLifecycleCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    workspace_lifecycle_extended: WorkspaceLifecycleExtendedCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    workspace_lockdown_receipt_v1: LockdownReceiptCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    agent_proposal_budget: AgentProposalBudgetCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    agent_execution_budget: AgentExecutionBudgetCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    agent_ingest_budget: AgentIngestBudgetCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    agent_upload_staging: AgentUploadStagingCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    agent_upload_staging_admin: AgentUploadStagingAdminCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    agent_grant_retry: AgentGrantRetryCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    tenant_fair_scheduling: TenantFairSchedulingCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    http_rate_limit: HttpRateLimitCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    http_peer_rate_limit: HttpPeerRateLimitCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    tenant_http_admission: TenantHttpAdmissionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    workspace_capacity: WorkspaceCapacityCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    advisory_pricing: AdvisoryPricingCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    operations: list[str]
    dtypes: list[DType]
    tensor_envelope: TensorEnvelopeCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    llm_planning: bool
    model_investigation: InvestigationCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    model_destination: ModelDestinationCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    model_direct_transport: DirectTransportCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    plan_schema: dict[str, JsonValue]
    tensor_semantics: SemanticsCapabilities
    snapshot_similarity: SnapshotSimilarityCapabilities
    validity_snapshot_similarity: ValiditySnapshotSimilarityCapabilities
    reference_evaluations: ReferenceEvaluationCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    product_similarity_evaluations: ProductSimilarityEvaluationCapabilities
    data_products: DataProductCapabilities
    release_contracts: ReleaseCapabilities
    write_contracts: WriteContractCapabilities
    plan_evaluation: EvaluationCapabilities
    plan_outcomes: PlanOutcomeCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    tensor_evaluation: TensorEvaluationCapabilities
    axis_alignment: AlignmentCapabilities
    descendant_lineage: DescendantCapabilities
    snapshot_retirement: SnapshotRetirementCapabilities
    retention_lifecycle: RetentionLifecycleCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    product_revision_retirement: ProductRevisionRetirementCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    product_retirement: ProductRetirementCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    quality_release_retirement: QualityReleaseRetirementCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    product_revision_holds: ProductRevisionHoldCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    npy_downloads: NpyDownloadCapabilities
    npy_download_admission: NpyDownloadAdmissionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    similarity_admission: SimilarityAdmissionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scientific_read_admission: ScientificReadAdmissionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scientific_read_admission_v2: ScientificReadAdmissionV2Capabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scientific_read_admission_v3: ScientificReadAdmissionV3Capabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scientific_read_admission_v4: ScientificReadAdmissionV4Capabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scientific_read_admission_v5: ScientificReadAdmissionV5Capabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scientific_read_admission_v6: ScientificReadAdmissionV6Capabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scientific_read_admission_v7: ScientificReadAdmissionV7Capabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scientific_read_admission_v8: ScientificReadAdmissionV8Capabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scientific_read_admission_v9: ScientificReadAdmissionV9Capabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scientific_read_admission_v10: ScientificReadAdmissionV10Capabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scientific_read_admission_v11: ScientificReadAdmissionV11Capabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    execution_supervision: EagerSupervisionCapabilities
    semantic_discovery_index: SemanticIndexCapabilities
    eager_ingest_staging: EagerIngestStagingCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    request_body_staging: RequestBodyStagingCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    request_body_staging_v2: RemoteRequestBodyStagingCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    request_body_staging_deadline: RequestBodyStagingDeadlineCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    recovery_verification: RecoveryVerificationCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    recovery_inventory: RecoveryInventoryCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    recovery_inventory_partition: RecoveryPartitionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    tenant_staging_reservations: TenantStagingCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    tenant_staging_read_admission: TenantStagingReadAdmissionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    upload_chunk_admission: UploadChunkAdmissionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    tenant_staging_work_quota: TenantWorkQuotaCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    tenant_staging_input_quota: TenantInputQuotaCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    worker_attempt_evidence: WorkerAttemptEvidenceCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    worker_claim_accounting: WorkerClaimCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    worker_cancellation_recovery: WorkerCancellationRecoveryCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    worker_cancellation_recovery_v2: WorkerCancellationRecoveryV2Capabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    plan_estimated_attempts: PlanEstimatedAttemptsCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    plan_usage_observations: PlanUsageCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    ingest_usage_observations: IngestUsageCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    upload_attempt_observations: UploadAttemptCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    upload_preclaim_rejections: PreclaimCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    plan_process_usage_observations: PlanProcessUsageCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    plan_usage_handoff: PlanUsageHandoffCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    plan_usage_adjustments: AdjustmentCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    plan_adjusted_usage_handoff: AdjustedHandoffCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    workspace_usage_handoff: WorkspaceUsageHandoffCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    workspace_usage_handoff_v2: MixedWorkspaceUsageHandoffCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    local_usage_fixture: FixtureCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    local_usage_fixture_inspection: FixtureFileInspectionCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    local_usage_fixture_intents: FixtureIntentCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    local_usage_fixture_namespace_guard: FixtureNamespaceGuardCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    local_usage_fixture_receipt_reads: FixtureReceiptReadCapabilities | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    limits: Limits


class AuditEvent(Model):
    id: UUID
    tenant_id: UUID
    actor_id: str
    action: str
    resource_id: str
    details: dict[str, JsonValue]
    created_at: str


class DailyActivity(Model):
    date: str
    versions: int
    logical_bytes: int
    proposed: int
    succeeded: int
    failed: int
    cancelled: int


class DatasetUsage(Model):
    name: str
    title: str
    archived: bool
    versions: int
    logical_bytes: int


class WorkspaceInsights(Model):
    generated_at: str
    days: int
    daily: list[DailyActivity]
    datasets: list[DatasetUsage]
    oldest_queued_at: str | None
    recoverable_runs: int
