"""Selected plan and direct-ingestion usage evidence, without billing authority."""
import hashlib
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from tensorus.arrays import canonical_hash
from tensorus.arrays import canonical_json
from tensorus.arrays import decode_unique_json
from tensorus.contracts import ReleaseHash
from tensorus.errors import TensorusError
from tensorus.ingest_usage import IngestUsageObservation
from tensorus.models import Model
from tensorus.plan_usage_adjusted_handoff import AdjustedHandoffDocument
from tensorus.workspace_usage_handoff import BundleDocument
from tensorus.workspace_usage_handoff import PlanSelection
from tensorus.workspace_usage_handoff import WorkspaceUsageHandoff as PlanBundle
FORMAT: Literal["tensorus-workspace-usage-handoff-v2"] = "tensorus-workspace-usage-handoff-v2"
DOCUMENT_FORMAT: Literal["tensorus-workspace-usage-handoff-document-v2"] = (
    "tensorus-workspace-usage-handoff-document-v2"
)
INGEST_ID_FORMAT: Literal["tensorus-ingest-usage-event-identity-v1"] = (
    "tensorus-ingest-usage-event-identity-v1"
)
INGEST_PAYLOAD_FORMAT: Literal["tensorus-ingest-usage-event-payload-v1"] = (
    "tensorus-ingest-usage-event-payload-v1"
)
MAX_PLANS: Literal[4] = 4
MAX_INGESTIONS: Literal[16] = 16
MAX_DOCUMENT_BYTES: Literal[1048576] = 1048576
MAX_RESPONSE_BYTES: Literal[2097152] = 2097152
class IngestSelection(Model):
    version_id: UUID
    expected_checksum: ReleaseHash
class MixedBundleRequest(Model):
    plans: list[PlanSelection] = Field(default_factory=list, max_length=MAX_PLANS)
    ingestions: list[IngestSelection] = Field(default_factory=list, max_length=MAX_INGESTIONS)

    @model_validator(mode="after")
    def unique_selections(self) -> Self:
        if not self.plans and not self.ingestions:
            raise ValueError("Select at least one plan or ingestion version")
        if len({item.plan_id for item in self.plans}) != len(self.plans):
            raise ValueError("A plan may appear only once in a usage bundle")
        if len({item.version_id for item in self.ingestions}) != len(self.ingestions):
            raise ValueError("An ingestion version may appear only once in a usage bundle")
        return self
def _ingest_event_id(tenant_id: UUID, version_id: UUID) -> str:
    return canonical_hash(
        {
            "format": INGEST_ID_FORMAT,
            "tenant_id": str(tenant_id),
            "version_id": str(version_id),
        }
    )
def _ingest_payload(observation: IngestUsageObservation) -> dict[str, object] | None:
    if observation.commit is None:
        return None
    return {
        "format": INGEST_PAYLOAD_FORMAT,
        "tenant_id": str(observation.tenant_id),
        "version_id": str(observation.version_id),
        "checksum": observation.checksum,
        "commit": observation.commit.model_dump(mode="json"),
    }
def _ingest_identity(observation: IngestUsageObservation) -> tuple[str | None, str | None]:
    payload = _ingest_payload(observation)
    if payload is None:
        return None, None
    event_id = _ingest_event_id(observation.tenant_id, observation.version_id)
    return event_id, hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
class IngestBundleItem(Model):
    observation: IngestUsageObservation
    event_id: ReleaseHash | None
    payload_sha256: ReleaseHash | None

    @model_validator(mode="after")
    def consistent_identity(self) -> Self:
        if (self.event_id, self.payload_sha256) != _ingest_identity(self.observation):
            raise ValueError("Ingestion observation identity or payload is inconsistent")
        return self
def _plan_output_evidence(plans: PlanBundle, tenant_id: UUID) -> tuple[set[UUID], bool]:
    document = BundleDocument.model_validate_json(plans.document_json, strict=True)
    if document.tenant_id != tenant_id:
        raise ValueError("Plan evidence identifies a different workspace")
    versions: set[UUID] = set()
    unobserved = False
    for item in document.items:
        adjusted = AdjustedHandoffDocument.model_validate_json(
            item.handoff.document_json, strict=True
        )
        unobserved = unobserved or bool(adjusted.original.usage.unobserved_output_steps)
        for output in adjusted.original.usage.outputs:
            if output.version_id in versions:
                raise ValueError("Plan output version appears more than once")
            versions.add(output.version_id)
    return versions, unobserved
class MixedBundleDocument(Model):
    format: Literal["tensorus-workspace-usage-handoff-document-v2"] = DOCUMENT_FORMAT
    tenant_id: UUID
    plans: PlanBundle | None
    ingestions: list[IngestBundleItem] = Field(max_length=MAX_INGESTIONS)
    unobserved_plan_outputs_may_overlap: bool
    snapshots_atomic: Literal[False] = False
    source_authenticity_verified: Literal[False] = False
    automatic_reconciliation: Literal[False] = False
    provider_submission: Literal[False] = False
    charges_created: Literal[False] = False
    billing_authority: Literal[False] = False
    restored_sources_fenced: Literal[False] = False
    missing_observations_are_zero: Literal[False] = False
    complete_workspace_coverage: Literal[False] = False

    @field_validator(
        "snapshots_atomic",
        "source_authenticity_verified",
        "automatic_reconciliation",
        "provider_submission",
        "charges_created",
        "billing_authority",
        "restored_sources_fenced",
        "missing_observations_are_zero",
        "complete_workspace_coverage",
        "unobserved_plan_outputs_may_overlap",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Usage bundle flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_items(self) -> Self:
        if self.plans is None and not self.ingestions:
            raise ValueError("A usage bundle must contain selected evidence")
        versions, unobserved = (
            _plan_output_evidence(self.plans, self.tenant_id)
            if self.plans is not None
            else (set(), False)
        )
        if self.unobserved_plan_outputs_may_overlap != (unobserved and bool(self.ingestions)):
            raise ValueError("Unobserved plan output overlap coverage is inconsistent")
        event_ids: set[str] = set()
        for item in self.ingestions:
            observation = item.observation
            if observation.tenant_id != self.tenant_id or observation.version_id in versions:
                raise ValueError("Ingestion evidence identifies a different or duplicate version")
            versions.add(observation.version_id)
            if item.event_id is not None:
                if item.event_id in event_ids:
                    raise ValueError("Ingestion event identity is duplicated")
                event_ids.add(item.event_id)
        return self
class MixedWorkspaceUsageHandoff(Model):
    format: Literal["tensorus-workspace-usage-handoff-v2"] = FORMAT
    document_json: Annotated[str, Field(strict=True, min_length=1, max_length=MAX_DOCUMENT_BYTES)]
    document_sha256: ReleaseHash

    @model_validator(mode="after")
    def checked_document(self) -> Self:
        raw = self.document_json.encode("utf-8")
        if len(raw) > MAX_DOCUMENT_BYTES:
            raise ValueError("Usage bundle document exceeds its byte limit")
        try:
            decoded = decode_unique_json(raw)
            checked = MixedBundleDocument.model_validate_json(raw, strict=True)
            if (
                canonical_json(decoded) != self.document_json
                or canonical_json(checked.model_dump(mode="json")) != self.document_json
                or hashlib.sha256(raw).hexdigest() != self.document_sha256
            ):
                raise ValueError("Usage bundle document is not exact")
        except (TensorusError, TypeError, ValueError, UnicodeError, RecursionError) as exc:
            raise ValueError("Usage bundle document is invalid") from exc
        return self
class MixedWorkspaceUsageHandoffCapabilities(Model):
    format: Literal["tensorus-workspace-usage-handoff-v2"] = FORMAT
    document_format: Literal["tensorus-workspace-usage-handoff-document-v2"] = DOCUMENT_FORMAT
    ingestion_identity_format: Literal["tensorus-ingest-usage-event-identity-v1"] = INGEST_ID_FORMAT
    ingestion_payload_format: Literal["tensorus-ingest-usage-event-payload-v1"] = (
        INGEST_PAYLOAD_FORMAT
    )
    authority: Literal["active_unscoped_tenant_admin"] = "active_unscoped_tenant_admin"
    max_request_bytes: Annotated[int, Field(strict=True, ge=1024, le=16384)]
    max_plans: Literal[4] = MAX_PLANS
    max_ingestions: Literal[16] = MAX_INGESTIONS
    max_document_bytes: Literal[1048576] = MAX_DOCUMENT_BYTES
    max_response_bytes: Literal[2097152] = MAX_RESPONSE_BYTES
    separate_scientific_read_admission_enabled: bool
    terminal_plans_only: Literal[True] = True
    snapshots_atomic: Literal[False] = False
    source_authenticity_verified: Literal[False] = False
    automatic_reconciliation: Literal[False] = False
    provider_submission: Literal[False] = False
    charges_created: Literal[False] = False
    billing_authority: Literal[False] = False
    restored_sources_fenced: Literal[False] = False
    missing_observations_are_zero: Literal[False] = False
    complete_workspace_coverage: Literal[False] = False

    @field_validator(
        "max_plans", "max_ingestions", "max_document_bytes", "max_response_bytes", mode="before"
    )
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Usage bundle limits must be integers")
        return value

    @field_validator(
        "separate_scientific_read_admission_enabled",
        "terminal_plans_only",
        "snapshots_atomic",
        "source_authenticity_verified",
        "automatic_reconciliation",
        "provider_submission",
        "charges_created",
        "billing_authority",
        "restored_sources_fenced",
        "missing_observations_are_zero",
        "complete_workspace_coverage",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Usage bundle capability flags must be booleans")
        return value
