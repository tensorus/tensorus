"""Reviewed, catalog-audited delivery to the optional local usage fixture."""
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictBool
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_validator
from tensorus.arrays import canonical_hash
from tensorus.contracts import ReleaseHash
from tensorus.local_usage_provider import MAX_BATCH_EVENTS
from tensorus.local_usage_provider import MAX_EVENT_BYTES
from tensorus.local_usage_provider import MAX_INSPECTION_EVENTS
from tensorus.local_usage_provider import MAX_TENANT_ENTRIES
from tensorus.local_usage_provider import FixtureFileObservation
from tensorus.local_usage_provider import FixtureManifestEntry
from tensorus.models import Model
from tensorus.models import PlanStatus
from tensorus.models import StepName
from tensorus.plan_usage import Attempt
from tensorus.plan_usage_adjustments import _timestamp
from tensorus.plan_usage_comparison import EvidenceIdentity
from tensorus.workspace_usage_comparison_v2 import IngestReference
from tensorus.workspace_usage_handoff_v2 import MAX_INGESTIONS
from tensorus.workspace_usage_handoff_v2 import MAX_PLANS
from tensorus.workspace_usage_handoff_v2 import MixedBundleRequest
from tensorus.workspace_usage_handoff_v2 import _ingest_event_id
CAPABILITY_FORMAT: Literal["tensorus-local-usage-fixture-v1"] = "tensorus-local-usage-fixture-v1"
REVIEW_FORMAT: Literal["tensorus-local-usage-fixture-review-v1"] = (
    "tensorus-local-usage-fixture-review-v1"
)
RECEIPT_FORMAT: Literal["tensorus-local-usage-fixture-receipt-v1"] = (
    "tensorus-local-usage-fixture-receipt-v1"
)
RECEIPT_READ_FORMAT: Literal["tensorus-local-usage-fixture-receipt-read-v1"] = (
    "tensorus-local-usage-fixture-receipt-read-v1"
)
RECOVERY_FORMAT: Literal["tensorus-local-usage-fixture-recovery-v1"] = (
    "tensorus-local-usage-fixture-recovery-v1"
)
MAX_RECEIPTS: Literal[1000] = 1000
MAX_RESPONSE_BYTES: Literal[262144] = 262144
INSPECTION_FORMAT: Literal["tensorus-local-usage-fixture-inspection-v1"] = (
    "tensorus-local-usage-fixture-inspection-v1"
)
INSPECTION_RESPONSE_FORMAT: Literal["tensorus-local-usage-fixture-file-observation-v1"] = (
    "tensorus-local-usage-fixture-file-observation-v1"
)
MAX_INSPECTION_RESPONSE_BYTES: Literal[65536] = 65536
NAMESPACE_GUARD_FORMAT: Literal["tensorus-local-usage-namespace-guard-v1"] = (
    "tensorus-local-usage-namespace-guard-v1"
)
class FixturePlanReview(Model):
    plan_id: UUID
    plan_hash: ReleaseHash
    plan_status: PlanStatus
    plan_attempts: Annotated[StrictInt, Field(ge=0, le=10)]
    observations: list[EvidenceIdentity] = Field(max_length=36)
    assertions: list[EvidenceIdentity] = Field(max_length=32)
    unobserved_output_steps: list[StepName] = Field(max_length=16)
    unobserved_wall_attempts: list[Attempt] = Field(max_length=10)
    unobserved_process_attempts: list[Attempt] = Field(max_length=10)

    @model_validator(mode="after")
    def checked_items(self) -> Self:
        if self.plan_status not in {PlanStatus.succeeded, PlanStatus.failed, PlanStatus.cancelled}:
            raise ValueError("Fixture review requires a terminal plan")
        for entries in (self.observations, self.assertions):
            ids = [item.event_id for item in entries]
            if ids != sorted(set(ids)):
                raise ValueError("Fixture review event identities must be sorted and unique")
        if self.unobserved_output_steps != sorted(set(self.unobserved_output_steps)):
            raise ValueError("Fixture review coverage must be sorted and unique")
        for attempts in (self.unobserved_wall_attempts, self.unobserved_process_attempts):
            if attempts != sorted(set(attempts)):
                raise ValueError("Fixture review coverage must be sorted and unique")
        if any(
            attempt > self.plan_attempts
            for attempt in self.unobserved_wall_attempts + self.unobserved_process_attempts
        ):
            raise ValueError("Fixture review attempts exceed the plan attempt count")
        return self
class FixtureReviewDocument(Model):
    format: Literal["tensorus-local-usage-fixture-review-v1"] = REVIEW_FORMAT
    tenant_id: UUID
    source_namespace: UUID
    plans: list[FixturePlanReview] = Field(max_length=MAX_PLANS)
    ingestions: list[IngestReference] = Field(max_length=MAX_INGESTIONS)
    unobserved_plan_outputs_may_overlap: StrictBool
    unknown_coverage: StrictBool
    event_count: Annotated[StrictInt, Field(ge=0, le=MAX_BATCH_EVENTS)]
    snapshots_atomic: Literal[False] = False
    source_authenticity_verified: Literal[False] = False
    restored_sources_fenced: Literal[False] = False
    missing_observations_are_zero: Literal[False] = False
    complete_workspace_coverage: Literal[False] = False
    billing_authority: Literal[False] = False

    @field_validator(
        "snapshots_atomic",
        "source_authenticity_verified",
        "restored_sources_fenced",
        "missing_observations_are_zero",
        "complete_workspace_coverage",
        "billing_authority",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Fixture review flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.source_namespace.int == 0 or (not self.plans and not self.ingestions):
            raise ValueError("Fixture review requires a namespace and a selection")
        plan_ids = [item.plan_id for item in self.plans]
        version_ids = [item.version_id for item in self.ingestions]
        if plan_ids != sorted(set(plan_ids)) or version_ids != sorted(set(version_ids)):
            raise ValueError("Fixture review selections must be sorted and unique")
        ids = [item.event_id for plan in self.plans for item in plan.observations + plan.assertions]
        ids.extend(item.event_id for item in self.ingestions if item.event_id is not None)
        if len(set(ids)) != len(ids) or self.event_count != len(ids):
            raise ValueError("Fixture review event count or identities are inconsistent")
        for item in self.ingestions:
            if item.event_id is not None and item.event_id != _ingest_event_id(
                self.tenant_id, item.version_id
            ):
                raise ValueError("Fixture review ingestion identity is inconsistent")
        overlap = bool(self.ingestions and any(item.unobserved_output_steps for item in self.plans))
        unknown = overlap or any(
            item.unobserved_output_steps
            or item.unobserved_wall_attempts
            or item.unobserved_process_attempts
            for item in self.plans
        )
        unknown = bool(unknown or any(item.coverage == "unobserved" for item in self.ingestions))
        if self.unobserved_plan_outputs_may_overlap != overlap or self.unknown_coverage != unknown:
            raise ValueError("Fixture review unknown coverage is inconsistent")
        return self
class FixtureReview(Model):
    review: FixtureReviewDocument
    review_sha256: ReleaseHash

    @model_validator(mode="after")
    def checked_hash(self) -> Self:
        if self.review_sha256 != canonical_hash(self.review.model_dump(mode="json")):
            raise ValueError("Fixture review hash is inconsistent")
        return self
class FixtureReviewRequest(Model):
    source_namespace: UUID
    selection: MixedBundleRequest

    @field_validator("source_namespace")
    @classmethod
    def non_nil_namespace(cls, value: UUID) -> UUID:
        if value.int == 0:
            raise ValueError("A non-nil source namespace is required")
        return value
class FixtureDeliveryRequest(FixtureReviewRequest):
    request_id: UUID
    expected_review_sha256: ReleaseHash
    acknowledge_unknown: StrictBool
    confirmation: Literal["DELIVER"]

    @field_validator("request_id")
    @classmethod
    def non_nil_request_id(cls, value: UUID) -> UUID:
        if value.int == 0:
            raise ValueError("A non-nil request ID is required")
        return value
class FixtureFileInspectionRequest(Model):
    source_namespace: UUID
    events: list[EvidenceIdentity] = Field(min_length=1, max_length=MAX_INSPECTION_EVENTS)

    @model_validator(mode="after")
    def checked_selection(self) -> Self:
        ids = [item.event_id for item in self.events]
        if self.source_namespace.int == 0 or ids != sorted(set(ids)):
            raise ValueError("Fixture inspection requires a namespace and sorted unique events")
        return self
class FixtureFileInspection(Model):
    format: Literal["tensorus-local-usage-fixture-file-observation-v1"] = INSPECTION_RESPONSE_FORMAT
    tenant_id: UUID
    request: FixtureFileInspectionRequest
    observations: list[FixtureFileObservation] = Field(
        min_length=1, max_length=MAX_INSPECTION_EVENTS
    )
    observed_at: str
    fixture_only: Literal[True] = True
    read_only: Literal[True] = True
    snapshot_atomic: Literal[False] = False
    receipt_verified: Literal[False] = False
    delivery_attributed: Literal[False] = False
    source_authenticity_verified: Literal[False] = False
    file_contents_authenticated: Literal[False] = False
    restored_sources_fenced: Literal[False] = False
    billing_authority: Literal[False] = False
    copies_revocable: Literal[False] = False

    _observed_at = field_validator("observed_at")(_timestamp)

    @field_validator(
        "fixture_only",
        "read_only",
        "snapshot_atomic",
        "receipt_verified",
        "delivery_attributed",
        "source_authenticity_verified",
        "file_contents_authenticated",
        "restored_sources_fenced",
        "billing_authority",
        "copies_revocable",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Fixture file observation flags must be booleans")
        return value

    @model_validator(mode="after")
    def checked_items(self) -> Self:
        if [(item.event_id, item.expected_payload_sha256) for item in self.observations] != [
            (item.event_id, item.payload_sha256) for item in self.request.events
        ]:
            raise ValueError("Fixture file observations do not match the selected claims")
        return self
class FixtureFileInspectionCapabilities(Model):
    format: Literal["tensorus-local-usage-fixture-inspection-v1"] = INSPECTION_FORMAT
    observation_format: Literal["tensorus-local-usage-fixture-file-observation-v1"] = (
        INSPECTION_RESPONSE_FORMAT
    )
    authority: Literal["active_unscoped_tenant_admin"] = "active_unscoped_tenant_admin"
    max_request_bytes: Annotated[int, Field(strict=True, ge=1024, le=16384)]
    max_events: Literal[64] = MAX_INSPECTION_EVENTS
    max_response_bytes: Literal[65536] = MAX_INSPECTION_RESPONSE_BYTES
    separate_scientific_read_admission_enabled: StrictBool
    fixture_only: Literal[True] = True
    read_only: Literal[True] = True
    snapshot_atomic: Literal[False] = False
    receipt_verified: Literal[False] = False
    delivery_attributed: Literal[False] = False
    source_authenticity_verified: Literal[False] = False
    billing_authority: Literal[False] = False

    @field_validator("max_events", "max_response_bytes", mode="before")
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Fixture inspection limits must be integers")
        return value

    @field_validator(
        "fixture_only",
        "read_only",
        "snapshot_atomic",
        "receipt_verified",
        "delivery_attributed",
        "source_authenticity_verified",
        "billing_authority",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Fixture inspection flags must be booleans")
        return value
def review_matches_request(
    review: FixtureReviewDocument, request: FixtureReviewRequest, tenant_id: UUID
) -> bool:
    expected_plans = sorted(
        (item.plan_id, item.expected_plan_hash) for item in request.selection.plans
    )
    expected_ingestions = sorted(
        (item.version_id, item.expected_checksum) for item in request.selection.ingestions
    )
    return (
        review.tenant_id == tenant_id
        and review.source_namespace == request.source_namespace
        and expected_plans == [(item.plan_id, item.plan_hash) for item in review.plans]
        and expected_ingestions == [(item.version_id, item.checksum) for item in review.ingestions]
    )
class FixtureReceipt(Model):
    format: Literal["tensorus-local-usage-fixture-receipt-v1"] = RECEIPT_FORMAT
    tenant_id: UUID
    actor_id: Annotated[str, Field(strict=True, min_length=1, max_length=36)]
    request: FixtureDeliveryRequest
    review: FixtureReviewDocument
    manifest: list[FixtureManifestEntry] = Field(min_length=1, max_length=MAX_BATCH_EVENTS)
    initially_created: Annotated[StrictInt, Field(ge=0, le=MAX_BATCH_EVENTS)]
    initially_existing: Annotated[StrictInt, Field(ge=0, le=MAX_BATCH_EVENTS)]
    recorded_at: str
    replayed: StrictBool
    fixture_only: Literal[True] = True
    external_provider_contacted: Literal[False] = False
    charges_created: Literal[False] = False
    billing_authority: Literal[False] = False
    source_authenticity_verified: Literal[False] = False
    restored_sources_fenced: Literal[False] = False
    copies_revocable: Literal[False] = False

    _recorded_at = field_validator("recorded_at")(_timestamp)

    @field_validator("actor_id")
    @classmethod
    def checked_actor(cls, value: str) -> str:
        if value != "embedded" and str(UUID(value)) != value:
            raise ValueError("Actor ID must be canonical")
        return value

    @field_validator(
        "fixture_only",
        "external_provider_contacted",
        "charges_created",
        "billing_authority",
        "source_authenticity_verified",
        "restored_sources_fenced",
        "copies_revocable",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Fixture receipt flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent(self) -> Self:
        request, review = self.request, self.review
        ids = sorted(
            [item.event_id for plan in review.plans for item in plan.observations + plan.assertions]
            + [item.event_id for item in review.ingestions if item.event_id is not None]
        )
        if (
            not review_matches_request(review, request, self.tenant_id)
            or request.expected_review_sha256 != canonical_hash(review.model_dump(mode="json"))
            or (review.unknown_coverage and not request.acknowledge_unknown)
            or ids != [item.event_id for item in self.manifest]
            or self.initially_created + self.initially_existing != review.event_count
        ):
            raise ValueError("Fixture receipt does not match the reviewed selection")
        return self
class FixtureRecovery(Model):
    format: Literal["tensorus-local-usage-fixture-recovery-v1"] = RECOVERY_FORMAT
    tenant_id: UUID
    actor_id: UUID
    request: FixtureDeliveryRequest
    review: FixtureReviewDocument
    source_authenticity_verified: Literal[False] = False
    authorizes_delivery: Literal[False] = False
    recovery_guaranteed: Literal[False] = False
    billing_authority: Literal[False] = False

    @field_validator(
        "source_authenticity_verified",
        "authorizes_delivery",
        "recovery_guaranteed",
        "billing_authority",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Fixture recovery flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (
            self.actor_id.int == 0
            or not review_matches_request(self.review, self.request, self.tenant_id)
            or self.request.expected_review_sha256
            != canonical_hash(self.review.model_dump(mode="json"))
            or self.review.event_count == 0
            or (self.review.unknown_coverage and not self.request.acknowledge_unknown)
        ):
            raise ValueError("Fixture recovery does not match the reviewed delivery")
        return self
class FixtureNamespaceGuardCapabilities(Model):
    format: Literal["tensorus-local-usage-namespace-guard-v1"] = NAMESPACE_GUARD_FORMAT
    max_tenant_entries: Literal[4096] = MAX_TENANT_ENTRIES
    max_event_bytes: Literal[8192] = MAX_EVENT_BYTES
    selected_cross_namespace_conflicts_blocked: Literal[True] = True
    invalid_or_unrecognized_entries_blocked: Literal[True] = True
    existing_receipt_retries_unchanged: Literal[True] = True
    catalog_intent_precedes_inspection: Literal[True] = True
    single_local_root_only: Literal[True] = True
    other_roots_fenced: Literal[False] = False
    nonparticipating_writers_fenced: Literal[False] = False
    restored_sources_fenced: Literal[False] = False
    source_authenticity_verified: Literal[False] = False
    billing_authority: Literal[False] = False

    @field_validator("max_tenant_entries", "max_event_bytes", mode="before")
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Fixture namespace guard limits must be integers")
        return value

    @field_validator(
        "selected_cross_namespace_conflicts_blocked",
        "invalid_or_unrecognized_entries_blocked",
        "existing_receipt_retries_unchanged",
        "catalog_intent_precedes_inspection",
        "single_local_root_only",
        "other_roots_fenced",
        "nonparticipating_writers_fenced",
        "restored_sources_fenced",
        "source_authenticity_verified",
        "billing_authority",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Fixture namespace guard flags must be booleans")
        return value
class FixtureCapabilities(Model):
    format: Literal["tensorus-local-usage-fixture-v1"] = CAPABILITY_FORMAT
    review_format: Literal["tensorus-local-usage-fixture-review-v1"] = REVIEW_FORMAT
    receipt_format: Literal["tensorus-local-usage-fixture-receipt-v1"] = RECEIPT_FORMAT
    authority: Literal["active_unscoped_tenant_admin"] = "active_unscoped_tenant_admin"
    max_request_bytes: Annotated[int, Field(strict=True, ge=1024, le=16384)]
    max_events: Literal[288] = MAX_BATCH_EVENTS
    max_event_bytes: Literal[8192] = MAX_EVENT_BYTES
    max_tenant_entries: Literal[4096] = MAX_TENANT_ENTRIES
    max_receipts_per_tenant: Literal[1000] = MAX_RECEIPTS
    max_response_bytes: Literal[262144] = MAX_RESPONSE_BYTES
    separate_scientific_read_admission_enabled: StrictBool
    single_host_nonblocking_lock: Literal[True] = True
    fixture_only: Literal[True] = True
    external_provider_contacted: Literal[False] = False
    charges_created: Literal[False] = False
    billing_authority: Literal[False] = False
    restored_sources_fenced: Literal[False] = False

    @field_validator(
        "max_events",
        "max_event_bytes",
        "max_tenant_entries",
        "max_receipts_per_tenant",
        "max_response_bytes",
        mode="before",
    )
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Fixture capability limits must be integers")
        return value

    @field_validator(
        "fixture_only",
        "single_host_nonblocking_lock",
        "external_provider_contacted",
        "charges_created",
        "billing_authority",
        "restored_sources_fenced",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Fixture capability flags must be booleans")
        return value
class FixtureReceiptReadCapabilities(Model):
    format: Literal["tensorus-local-usage-fixture-receipt-read-v1"] = RECEIPT_READ_FORMAT
    receipt_format: Literal["tensorus-local-usage-fixture-receipt-v1"] = RECEIPT_FORMAT
    authority: Literal["active_unscoped_tenant_admin"] = "active_unscoped_tenant_admin"
    max_response_bytes: Literal[262144] = MAX_RESPONSE_BYTES
    separate_scientific_read_admission_enabled: StrictBool
    verified_against_local_files: Literal[True] = True
    current_authority_rechecked_after_serialization: Literal[True] = True
    read_only: Literal[True] = True

    @field_validator("max_response_bytes", mode="before")
    @classmethod
    def exact_limit(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Fixture receipt read limit must be an integer")
        return value

    @field_validator(
        "verified_against_local_files",
        "current_authority_rechecked_after_serialization",
        "read_only",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Fixture receipt read flags must be booleans")
        return value
