"""Bounded, copy-only bundles of existing adjusted plan usage evidence."""
import hashlib
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from tensorus.arrays import canonical_json
from tensorus.arrays import decode_unique_json
from tensorus.contracts import ReleaseHash
from tensorus.errors import TensorusError
from tensorus.models import Model
from tensorus.plan_usage_adjusted_handoff import AdjustedHandoffDocument
from tensorus.plan_usage_adjusted_handoff import AdjustedUsageHandoff
FORMAT: Literal["tensorus-workspace-usage-handoff-v1"] = "tensorus-workspace-usage-handoff-v1"
DOCUMENT_FORMAT: Literal["tensorus-workspace-usage-handoff-document-v1"] = (
    "tensorus-workspace-usage-handoff-document-v1"
)
MAX_PLANS: Literal[4] = 4
MAX_DOCUMENT_BYTES: Literal[524288] = 524288
MAX_RESPONSE_BYTES: Literal[1048576] = 1048576
class PlanSelection(Model):
    plan_id: UUID
    expected_plan_hash: ReleaseHash
class BundleRequest(Model):
    plans: list[PlanSelection] = Field(min_length=1, max_length=MAX_PLANS)

    @model_validator(mode="after")
    def unique_plans(self) -> Self:
        if len({item.plan_id for item in self.plans}) != len(self.plans):
            raise ValueError("A plan may appear only once in a usage bundle")
        return self
class BundleItem(Model):
    plan_id: UUID
    plan_hash: ReleaseHash
    handoff: AdjustedUsageHandoff
class BundleDocument(Model):
    format: Literal["tensorus-workspace-usage-handoff-document-v1"] = DOCUMENT_FORMAT
    tenant_id: UUID
    items: list[BundleItem] = Field(min_length=1, max_length=MAX_PLANS)
    snapshots_atomic: Literal[False] = False
    source_authenticity_verified: Literal[False] = False
    automatic_reconciliation: Literal[False] = False
    provider_submission: Literal[False] = False
    charges_created: Literal[False] = False
    billing_authority: Literal[False] = False
    restored_sources_fenced: Literal[False] = False
    missing_observations_are_zero: Literal[False] = False

    @field_validator(
        "snapshots_atomic",
        "source_authenticity_verified",
        "automatic_reconciliation",
        "provider_submission",
        "charges_created",
        "billing_authority",
        "restored_sources_fenced",
        "missing_observations_are_zero",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Usage bundle flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_items(self) -> Self:
        plan_ids: set[UUID] = set()
        event_ids: set[str] = set()
        for item in self.items:
            if item.plan_id in plan_ids:
                raise ValueError("Usage bundle contains a duplicate plan")
            plan_ids.add(item.plan_id)
            document = AdjustedHandoffDocument.model_validate_json(
                item.handoff.document_json, strict=True
            )
            usage = document.original.usage
            if (
                usage.tenant_id != self.tenant_id
                or usage.plan_id != item.plan_id
                or usage.plan_hash != item.plan_hash
            ):
                raise ValueError("Usage bundle item identifies a different plan or workspace")
            identities = [
                *(event.event_id for event in document.original.events),
                *(event.event_id for event in document.assertion_identities),
            ]
            if len(set(identities)) != len(identities) or event_ids.intersection(identities):
                raise ValueError("Usage bundle contains duplicate event identities")
            event_ids.update(identities)
        return self
class WorkspaceUsageHandoff(Model):
    format: Literal["tensorus-workspace-usage-handoff-v1"] = FORMAT
    document_json: Annotated[str, Field(strict=True, min_length=1, max_length=MAX_DOCUMENT_BYTES)]
    document_sha256: ReleaseHash

    @model_validator(mode="after")
    def checked_document(self) -> Self:
        raw = self.document_json.encode("utf-8")
        if len(raw) > MAX_DOCUMENT_BYTES:
            raise ValueError("Usage bundle document exceeds its byte limit")
        try:
            decoded = decode_unique_json(raw)
            checked = BundleDocument.model_validate_json(raw, strict=True)
            if (
                canonical_json(decoded) != self.document_json
                or canonical_json(checked.model_dump(mode="json")) != self.document_json
                or hashlib.sha256(raw).hexdigest() != self.document_sha256
            ):
                raise ValueError("Usage bundle document is not exact")
        except (TensorusError, TypeError, ValueError, UnicodeError, RecursionError) as exc:
            raise ValueError("Usage bundle document is invalid") from exc
        return self
class WorkspaceUsageHandoffCapabilities(Model):
    format: Literal["tensorus-workspace-usage-handoff-v1"] = FORMAT
    document_format: Literal["tensorus-workspace-usage-handoff-document-v1"] = DOCUMENT_FORMAT
    authority: Literal["active_unscoped_tenant_admin"] = "active_unscoped_tenant_admin"
    max_request_bytes: Annotated[int, Field(strict=True, ge=1024, le=16384)]
    max_plans: Literal[4] = MAX_PLANS
    max_document_bytes: Literal[524288] = MAX_DOCUMENT_BYTES
    max_response_bytes: Literal[1048576] = MAX_RESPONSE_BYTES
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

    @field_validator("max_plans", "max_document_bytes", "max_response_bytes", mode="before")
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
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Usage bundle capability flags must be booleans")
        return value
