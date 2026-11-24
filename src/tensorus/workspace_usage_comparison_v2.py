"""Compare retained selected plan and ingestion usage identities."""

from typing import Literal, Self
from uuid import UUID

from pydantic import Field, ValidationError, field_validator, model_validator

from tensorus.arrays import Record, canonical_json, decode_unique_json
from tensorus.contracts import ReleaseHash
from tensorus.errors import Conflict, LimitExceeded, TensorusError
from tensorus.models import Model
from tensorus.workspace_usage_comparison import WorkspaceUsageReference
from tensorus.workspace_usage_comparison import (
    compare_references_for_workspace as compare_plan_references,
)
from tensorus.workspace_usage_comparison import reference_from_bundle as plan_reference
from tensorus.workspace_usage_handoff import BundleDocument
from tensorus.workspace_usage_handoff_v2 import (
    MAX_INGESTIONS,
    MixedBundleDocument,
    _ingest_event_id,
)

REFERENCE_FORMAT: Literal["tensorus-workspace-usage-reference-v2"] = (
    "tensorus-workspace-usage-reference-v2"
)
COMPARISON_FORMAT: Literal["tensorus-workspace-usage-comparison-v2"] = (
    "tensorus-workspace-usage-comparison-v2"
)
MAX_REFERENCE_BYTES: Literal[81920] = 81920


class IngestReference(Model):
    version_id: UUID
    checksum: ReleaseHash
    coverage: Literal["recorded", "unobserved"]
    event_id: ReleaseHash | None
    payload_sha256: ReleaseHash | None

    @model_validator(mode="after")
    def consistent_coverage(self) -> Self:
        if self.coverage == "recorded" and (self.event_id is None or self.payload_sha256 is None):
            raise ValueError("Ingestion reference coverage is inconsistent")
        if self.coverage == "unobserved" and (
            self.event_id is not None or self.payload_sha256 is not None
        ):
            raise ValueError("Ingestion reference coverage is inconsistent")
        return self


class MixedWorkspaceUsageReference(Model):
    format: Literal["tensorus-workspace-usage-reference-v2"] = REFERENCE_FORMAT
    source_namespace: UUID
    tenant_id: UUID
    plans: WorkspaceUsageReference | None
    ingestions: list[IngestReference] = Field(max_length=MAX_INGESTIONS)
    unobserved_plan_outputs_may_overlap: bool

    @field_validator("unobserved_plan_outputs_may_overlap", mode="before")
    @classmethod
    def exact_flag(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Overlap coverage must be a boolean")
        return value

    @model_validator(mode="after")
    def consistent_selection(self) -> Self:
        if self.source_namespace.int == 0 or (self.plans is None and not self.ingestions):
            raise ValueError("A source namespace and selected evidence are required")
        unobserved_plan_outputs = False
        if self.plans is not None:
            if (
                self.plans.source_namespace != self.source_namespace
                or self.plans.tenant_id != self.tenant_id
            ):
                raise ValueError("Plan references identify a different source")
            unobserved_plan_outputs = any(item.unobserved_output_steps for item in self.plans.plans)
        if self.unobserved_plan_outputs_may_overlap != (
            unobserved_plan_outputs and bool(self.ingestions)
        ):
            raise ValueError("Unobserved plan output overlap coverage is inconsistent")
        ids = [item.version_id for item in self.ingestions]
        if ids != sorted(set(ids)):
            raise ValueError("Ingestion versions must be sorted and unique")
        for item in self.ingestions:
            if item.event_id is not None and item.event_id != _ingest_event_id(
                self.tenant_id, item.version_id
            ):
                raise ValueError("Ingestion event identity is inconsistent")
        return self


def read_reference(data: bytes) -> MixedWorkspaceUsageReference:
    if not isinstance(data, bytes):
        raise TypeError("Workspace usage reference must be bytes")
    if len(data) > MAX_REFERENCE_BYTES:
        raise LimitExceeded("Workspace usage reference exceeds its byte limit")
    try:
        decoded = decode_unique_json(data)
        checked = MixedWorkspaceUsageReference.model_validate_json(data, strict=True)
        if not isinstance(decoded, dict) or canonical_json(decoded) != canonical_json(
            checked.model_dump(mode="json")
        ):
            raise ValueError("Workspace usage reference is not exact")
        return checked
    except (
        ValidationError,
        TensorusError,
        TypeError,
        ValueError,
        UnicodeError,
        RecursionError,
    ) as exc:
        raise TensorusError(
            "The workspace usage reference is invalid", code="invalid_workspace_usage_reference"
        ) from exc


def reference_from_bundle(
    bundle: MixedBundleDocument, source_namespace: UUID
) -> MixedWorkspaceUsageReference:
    plans = (
        plan_reference(
            BundleDocument.model_validate_json(bundle.plans.document_json, strict=True),
            source_namespace,
        )
        if bundle.plans is not None
        else None
    )
    ingestions = [
        IngestReference(
            version_id=item.observation.version_id,
            checksum=item.observation.checksum,
            coverage=item.observation.coverage,
            event_id=item.event_id,
            payload_sha256=item.payload_sha256,
        )
        for item in bundle.ingestions
    ]
    return MixedWorkspaceUsageReference(
        source_namespace=source_namespace,
        tenant_id=bundle.tenant_id,
        plans=plans,
        ingestions=sorted(ingestions, key=lambda item: item.version_id),
        unobserved_plan_outputs_may_overlap=bundle.unobserved_plan_outputs_may_overlap,
    )


def compare_references(
    previous: MixedWorkspaceUsageReference, current: MixedWorkspaceUsageReference
) -> Record:
    if (
        previous.source_namespace != current.source_namespace
        or previous.tenant_id != current.tenant_id
        or (previous.plans is None) != (current.plans is None)
        or [item.version_id for item in previous.ingestions]
        != [item.version_id for item in current.ingestions]
    ):
        raise Conflict("Workspace usage references identify different sources or selections")
    plans = (
        compare_plan_references(previous.plans, current.plans)
        if previous.plans is not None and current.plans is not None
        else None
    )
    ingestions: list[Record] = []
    for old, new in zip(previous.ingestions, current.ingestions, strict=True):
        checksum_changed = old.checksum != new.checksum
        if old.event_id is None and new.event_id is None:
            classification = "unobserved"
        elif old.event_id is None:
            classification = "new"
        elif new.event_id is None:
            classification = "missing"
        elif old.payload_sha256 != new.payload_sha256 or checksum_changed:
            classification = "changed"
        else:
            classification = "unchanged"
        ingestions.append(
            {
                "version_id": str(new.version_id),
                "event_id": new.event_id or old.event_id,
                "classification": classification,
                "previous_checksum": old.checksum,
                "current_checksum": new.checksum,
                "checksum_changed": checksum_changed,
                "previous_payload_sha256": old.payload_sha256,
                "current_payload_sha256": new.payload_sha256,
            }
        )
    missing_or_changed = bool(
        (plans is not None and plans["previous_entries_missing_or_changed"])
        or any(
            item["classification"] in {"missing", "changed"} or item["checksum_changed"]
            for item in ingestions
        )
    )
    new_entries = bool(
        (plans is not None and plans["new_entries_observed"])
        or any(item["classification"] == "new" for item in ingestions)
    )
    state_changed = bool(plans is not None and plans["plan_state_changed"])
    coverage_changed = bool(
        (plans is not None and plans["unobserved_coverage_changed"])
        or previous.unobserved_plan_outputs_may_overlap
        != current.unobserved_plan_outputs_may_overlap
        or any(
            old.coverage != new.coverage
            for old, new in zip(previous.ingestions, current.ingestions, strict=True)
        )
    )
    coverage_present = bool(
        (plans is not None and plans["unobserved_coverage_present"])
        or previous.unobserved_plan_outputs_may_overlap
        or current.unobserved_plan_outputs_may_overlap
        or any(item.coverage == "unobserved" for item in previous.ingestions + current.ingestions)
    )
    return {
        "format": COMPARISON_FORMAT,
        "source_namespace": str(current.source_namespace),
        "tenant_id": str(current.tenant_id),
        "plans": plans,
        "ingestions": ingestions,
        "previous_entries_missing_or_changed": missing_or_changed,
        "new_entries_observed": new_entries,
        "plan_state_changed": state_changed,
        "unobserved_coverage_changed": coverage_changed,
        "unobserved_coverage_present": coverage_present,
        "requires_review": (
            missing_or_changed
            or new_entries
            or state_changed
            or coverage_changed
            or coverage_present
        ),
        "selection_exhaustive": False,
        "snapshots_atomic": False,
        "prior_reference_authenticated": False,
        "current_reference_authenticated": False,
        "source_namespace_independently_verified": False,
        "automatic_reconciliation": False,
        "provider_submission": False,
        "charges_created": False,
        "billing_authority": False,
        "restored_sources_fenced": False,
        "missing_observations_are_zero": False,
    }
