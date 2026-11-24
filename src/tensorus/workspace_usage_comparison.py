"""Compare retained identity references for selected workspace usage evidence."""

from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, ValidationError, model_validator

from tensorus.arrays import Record, canonical_json, decode_unique_json
from tensorus.errors import Conflict, LimitExceeded, TensorusError
from tensorus.models import Model, StepName
from tensorus.plan_usage import Attempt
from tensorus.plan_usage_adjusted_handoff import AdjustedHandoffDocument
from tensorus.plan_usage_comparison import (
    UsageEvidenceReference,
    compare_references,
    reference_from_document,
)
from tensorus.workspace_usage_handoff import MAX_PLANS, BundleDocument

REFERENCE_FORMAT: Literal["tensorus-workspace-usage-reference-v1"] = (
    "tensorus-workspace-usage-reference-v1"
)
COMPARISON_FORMAT: Literal["tensorus-workspace-usage-comparison-v1"] = (
    "tensorus-workspace-usage-comparison-v1"
)
MAX_REFERENCE_BYTES: Literal[69632] = 69632


class PlanReference(Model):
    reference: UsageEvidenceReference
    unobserved_output_steps: list[StepName] = Field(max_length=16)
    unobserved_wall_attempts: list[Attempt] = Field(max_length=10)
    unobserved_process_attempts: list[Attempt] = Field(max_length=10)

    @model_validator(mode="after")
    def sorted_coverage(self) -> Self:
        if self.unobserved_output_steps != sorted(set(self.unobserved_output_steps)) or any(
            values != sorted(set(values))
            for values in (self.unobserved_wall_attempts, self.unobserved_process_attempts)
        ):
            raise ValueError("Unobserved usage coverage must be sorted and unique")
        if any(
            attempt > self.reference.plan_attempts
            for attempt in self.unobserved_wall_attempts + self.unobserved_process_attempts
        ):
            raise ValueError("Unobserved attempts exceed the recorded attempt count")
        return self


class WorkspaceUsageReference(Model):
    format: Literal["tensorus-workspace-usage-reference-v1"] = REFERENCE_FORMAT
    source_namespace: UUID
    tenant_id: UUID
    plans: Annotated[list[PlanReference], Field(min_length=1, max_length=MAX_PLANS)]

    @model_validator(mode="after")
    def consistent_selection(self) -> Self:
        if self.source_namespace.int == 0:
            raise ValueError("A non-nil source namespace is required")
        plan_ids = [item.reference.plan_id for item in self.plans]
        if plan_ids != sorted(set(plan_ids)):
            raise ValueError("Selected plans must be sorted and unique")
        identities: set[str] = set()
        for item in self.plans:
            reference = item.reference
            if reference.tenant_id != self.tenant_id:
                raise ValueError("A selected plan identifies a different workspace")
            for identity in reference.observations + reference.assertions:
                if identity.event_id in identities:
                    raise ValueError("Selected plans contain duplicate event identities")
                identities.add(identity.event_id)
        return self


def parse_namespace(value: str) -> UUID:
    if not isinstance(value, str):
        raise TypeError("Source namespace must be a canonical UUID string")
    try:
        namespace = UUID(value)
    except ValueError as exc:
        raise ValueError("Source namespace must be a canonical UUID string") from exc
    if str(namespace) != value or namespace.int == 0:
        raise ValueError("Source namespace must be a non-nil canonical UUID")
    return namespace


def read_reference(data: bytes) -> WorkspaceUsageReference:
    if not isinstance(data, bytes):
        raise TypeError("Workspace usage reference must be bytes")
    if len(data) > MAX_REFERENCE_BYTES:
        raise LimitExceeded("Workspace usage reference exceeds its byte limit")
    try:
        decoded = decode_unique_json(data)
        checked = WorkspaceUsageReference.model_validate_json(data, strict=True)
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
    bundle: BundleDocument, source_namespace: UUID
) -> WorkspaceUsageReference:
    plans = []
    for item in bundle.items:
        document = AdjustedHandoffDocument.model_validate_json(
            item.handoff.document_json, strict=True
        )
        plans.append(
            PlanReference(
                reference=reference_from_document(document, item.handoff.document_sha256),
                unobserved_output_steps=sorted(document.original.usage.unobserved_output_steps),
                unobserved_wall_attempts=document.original.usage.unobserved_attempts,
                unobserved_process_attempts=document.original.process.unobserved_attempts,
            )
        )
    return WorkspaceUsageReference(
        source_namespace=source_namespace,
        tenant_id=bundle.tenant_id,
        plans=sorted(plans, key=lambda item: item.reference.plan_id),
    )


def compare_references_for_workspace(
    previous: WorkspaceUsageReference, current: WorkspaceUsageReference
) -> Record:
    if (
        previous.source_namespace != current.source_namespace
        or previous.tenant_id != current.tenant_id
        or [(item.reference.plan_id, item.reference.plan_hash) for item in previous.plans]
        != [(item.reference.plan_id, item.reference.plan_hash) for item in current.plans]
    ):
        raise Conflict("Workspace usage references identify different sources or selections")
    plans: list[Record] = []
    for old, new in zip(previous.plans, current.plans, strict=True):
        comparison = compare_references(old.reference, new.reference)
        previous_coverage = {
            "unobserved_output_steps": old.unobserved_output_steps,
            "unobserved_wall_attempts": old.unobserved_wall_attempts,
            "unobserved_process_attempts": old.unobserved_process_attempts,
        }
        current_coverage = {
            "unobserved_output_steps": new.unobserved_output_steps,
            "unobserved_wall_attempts": new.unobserved_wall_attempts,
            "unobserved_process_attempts": new.unobserved_process_attempts,
        }
        plans.append(
            {
                "comparison": comparison,
                "previous_unobserved": previous_coverage,
                "current_unobserved": current_coverage,
                "unobserved_coverage_changed": previous_coverage != current_coverage,
            }
        )
    prior_entries_changed = any(
        item["comparison"]["previous_entries_missing_or_changed"] for item in plans
    )
    new_entries = any(
        event["classification"] == "new"
        for item in plans
        for event in item["comparison"]["observations"] + item["comparison"]["assertions"]
    )
    state_changed = any(
        item["comparison"]["plan_status_changed"] or item["comparison"]["plan_attempts_changed"]
        for item in plans
    )
    coverage_changed = any(item["unobserved_coverage_changed"] for item in plans)
    coverage_present = any(
        values
        for item in plans
        for coverage in (item["previous_unobserved"], item["current_unobserved"])
        for values in coverage.values()
    )
    return {
        "format": COMPARISON_FORMAT,
        "source_namespace": str(current.source_namespace),
        "tenant_id": str(current.tenant_id),
        "plans": plans,
        "previous_entries_missing_or_changed": prior_entries_changed,
        "new_entries_observed": new_entries,
        "plan_state_changed": state_changed,
        "unobserved_coverage_changed": coverage_changed,
        "unobserved_coverage_present": coverage_present,
        "requires_review": (
            prior_entries_changed
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
