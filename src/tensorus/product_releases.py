"""Separately reviewed releases of pinned cross-dataset product revisions."""
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictInt
from pydantic import model_validator
from tensorus.arrays import canonical_json
from tensorus.arrays import version_integrity_descriptor
from tensorus.contracts import ReleaseHash
from tensorus.errors import Conflict
from tensorus.errors import LimitExceeded
from tensorus.errors import TensorusError
from tensorus.models import Model
from tensorus.models import Name
from tensorus.models import PlanDefinition
from tensorus.models import StepName
from tensorus.models import TensorSpec
from tensorus.operations import parameters
from tensorus.plan_outcomes import COORDINATE_PLAN_OUTCOME_FORMAT
from tensorus.plan_outcomes import SCOPED_PLAN_OUTCOME_FORMAT
from tensorus.plan_outcomes import PlanOutcomeSnapshot
from tensorus.plan_outcomes import plan_outcome_format
from tensorus.products import MAX_PRODUCT_MEMBERS
from tensorus.products import MAX_PRODUCT_RELATIONSHIPS
from tensorus.products import DataProductRevision
from tensorus.products import ProductAxisRelationship
from tensorus.products import ProductMember
from tensorus.products import Revision
from tensorus.products import _check_relationship_selections
from tensorus.products import _hash
from tensorus.reference_evaluations import ReferenceEvaluation
from tensorus.semantics import read_semantics
from tensorus.semantics import require_fitted_domain
from tensorus.validity import TransformedValidityOrigin
from tensorus.validity import ValidityBinding
PRODUCT_RELEASE_FORMAT: Literal["tensorus-product-release-report-v1"] = (
    "tensorus-product-release-report-v1"
)
PRODUCT_RELEASE_REFERENCE_FORMAT: Literal["tensorus-product-release-report-v2"] = (
    "tensorus-product-release-report-v2"
)
PRODUCT_RELEASE_OUTCOME_FORMAT: Literal["tensorus-product-release-report-v3"] = (
    "tensorus-product-release-report-v3"
)
PRODUCT_RELEASE_SCOPED_OUTCOME_FORMAT: Literal["tensorus-product-release-report-v4"] = (
    "tensorus-product-release-report-v4"
)
PRODUCT_RELEASE_COORDINATE_OUTCOME_FORMAT: Literal["tensorus-product-release-report-v5"] = (
    "tensorus-product-release-report-v5"
)
PRODUCT_RELEASE_RELATIONSHIP_FORMAT: Literal["tensorus-product-release-report-v6"] = (
    "tensorus-product-release-report-v6"
)
MAX_PRODUCT_OUTCOME_PLANS = 8
MAX_PRODUCT_OUTCOME_BYTES = 512 * 1024
MAX_PRODUCT_RELEASE_IMPORTS_PER_PRODUCT = 64
MAX_PORTABLE_PRODUCT_BYTES = 8 * 1024**2
class ProductReferenceSelection(Model):
    output_role: Name
    reference_role: Name
    evaluation_id: UUID
    report_hash: ReleaseHash
    policy: Literal["record", "require_pass"]

    @model_validator(mode="after")
    def distinct_roles(self) -> Self:
        if self.output_role == self.reference_role:
            raise ValueError("Output and reference must name distinct product roles")
        return self
class ProductOutcomeSelection(Model):
    role: Name
    plan_id: UUID
    outcome_hash: ReleaseHash
    step_id: StepName
    policy: Literal["require_success", "record_terminal"]
    outcome_version: Literal["v2", "v3"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
class ProductReleaseRequest(Model):
    name: Name
    product_revision: Revision
    descriptor_hash: ReleaseHash
    quality_coverage: Literal["require_all", "selected_only"]
    references: list[ProductReferenceSelection] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_PRODUCT_MEMBERS,
        exclude_if=lambda value: value is None,
    )
    outcomes: list[ProductOutcomeSelection] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_PRODUCT_MEMBERS,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def distinct_reference_outputs(self) -> Self:
        if self.references is not None and len({item.output_role for item in self.references}) != (
            len(self.references)
        ):
            raise ValueError("A product output role can select only one reference evaluation")
        if self.outcomes is not None:
            if len({item.role for item in self.outcomes}) != len(self.outcomes):
                raise ValueError("A product role can select only one plan outcome")
            plans: dict[UUID, tuple[str, str | None]] = {}
            for item in self.outcomes:
                identity = (item.outcome_hash, item.outcome_version)
                if item.plan_id in plans and plans[item.plan_id] != identity:
                    raise ValueError("A selected plan must have one outcome hash and version")
                plans[item.plan_id] = identity
            if len(plans) > MAX_PRODUCT_OUTCOME_PLANS:
                raise ValueError("A product release can select at most eight distinct plans")
        return self
class ProductReleaseMember(Model):
    role: Name
    version_id: UUID
    checksum: ReleaseHash
    nbytes: Annotated[StrictInt, Field(ge=0, le=2**63 - 1)]
    write_spec: TensorSpec
    write_spec_hash: ReleaseHash
    contract_hash: ReleaseHash
    validity_selected: bool
    quality_basis: Literal["not_attached", "raw_values", "selected_validity", "other_validity"]
    quality_report_hash: ReleaseHash | None
class ProductReleaseReference(Model):
    output_role: Name
    reference_role: Name
    evaluation_id: UUID
    evaluation_hash: ReleaseHash
    report_hash: ReleaseHash
    policy: Literal["record", "require_pass"]
    passed: bool
    included_count: Annotated[StrictInt, Field(ge=0)]
    excluded_count: Annotated[StrictInt, Field(ge=0)]
class ProductReleaseOutcome(Model):
    role: Name
    plan_id: UUID
    outcome_hash: ReleaseHash
    snapshot_hash: ReleaseHash
    step_id: StepName
    version_id: UUID
    policy: Literal["require_success", "record_terminal"]
    status: Literal["succeeded", "failed", "cancelled"]
    checkpoint_state: Literal["none", "partial", "all"]
class ProductReleaseReport(Model):
    format: Literal[
        "tensorus-product-release-report-v1",
        "tensorus-product-release-report-v2",
        "tensorus-product-release-report-v3",
        "tensorus-product-release-report-v4",
        "tensorus-product-release-report-v5",
        "tensorus-product-release-report-v6",
    ] = PRODUCT_RELEASE_FORMAT
    tenant_id: UUID
    product_id: UUID
    product_name: Name
    product_revision: Revision
    descriptor_hash: ReleaseHash
    release_name: Name
    proposed_by: str
    quality_coverage: Literal["require_all", "selected_only"]
    verification: Literal["full_checksum_of_distinct_role_versions"] = (
        "full_checksum_of_distinct_role_versions"
    )
    checked_bytes: Annotated[StrictInt, Field(ge=0)]
    checked_versions: Annotated[StrictInt, Field(ge=1, le=MAX_PRODUCT_MEMBERS)]
    members: list[ProductReleaseMember] = Field(min_length=1, max_length=MAX_PRODUCT_MEMBERS)
    relationships: list[ProductAxisRelationship] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_PRODUCT_RELATIONSHIPS,
        exclude_if=lambda value: value is None,
    )
    references: list[ProductReleaseReference] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_PRODUCT_MEMBERS,
        exclude_if=lambda value: value is None,
    )
    outcomes: list[ProductReleaseOutcome] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_PRODUCT_MEMBERS,
        exclude_if=lambda value: value is None,
    )
    outcome_snapshots: list[PlanOutcomeSnapshot] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_PRODUCT_OUTCOME_PLANS,
        exclude_if=lambda value: value is None,
    )
    outcome_checked_bytes: Annotated[StrictInt, Field(ge=0)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def consistent_format(self) -> Self:
        versions: dict[UUID, tuple[str, int]] = {}
        for member in self.members:
            identity = (member.checksum, member.nbytes)
            if member.version_id in versions and versions[member.version_id] != identity:
                raise ValueError("Repeated product role versions have conflicting descriptors")
            versions[member.version_id] = identity
            if (
                (member.quality_basis == "not_attached") != (member.quality_report_hash is None)
                or (member.quality_basis == "selected_validity" and not member.validity_selected)
                or (self.quality_coverage == "require_all" and member.quality_report_hash is None)
            ):
                raise ValueError("Product role quality evidence does not match its coverage")
        if self.checked_versions != len(versions) or self.checked_bytes != sum(
            nbytes for _, nbytes in versions.values()
        ):
            raise ValueError("Product verification totals do not match distinct role versions")
        if self.format == PRODUCT_RELEASE_RELATIONSHIP_FORMAT:
            if self.relationships is None:
                raise ValueError("Relationship report requires its declared relationships")
            _check_relationship_selections(
                self.relationships, {member.role for member in self.members}
            )
            parts = (self.outcomes, self.outcome_snapshots, self.outcome_checked_bytes)
            if any(part is None for part in parts) and not all(part is None for part in parts):
                raise ValueError("Product outcome report requires its selections and snapshots")
            return self
        if self.relationships is not None:
            raise ValueError("Legacy product release formats cannot contain relationships")
        if self.format in (
            PRODUCT_RELEASE_OUTCOME_FORMAT,
            PRODUCT_RELEASE_SCOPED_OUTCOME_FORMAT,
            PRODUCT_RELEASE_COORDINATE_OUTCOME_FORMAT,
        ):
            if (
                self.outcomes is None
                or self.outcome_snapshots is None
                or self.outcome_checked_bytes is None
            ):
                raise ValueError("Product outcome report requires its selections and snapshots")
            has_scoped_outcome = any(
                item.evidence.outcome.format == SCOPED_PLAN_OUTCOME_FORMAT
                for item in self.outcome_snapshots
            )
            has_coordinate_outcome = any(
                item.evidence.outcome.format == COORDINATE_PLAN_OUTCOME_FORMAT
                for item in self.outcome_snapshots
            )
            expected = (
                PRODUCT_RELEASE_COORDINATE_OUTCOME_FORMAT
                if has_coordinate_outcome
                else PRODUCT_RELEASE_SCOPED_OUTCOME_FORMAT
                if has_scoped_outcome
                else PRODUCT_RELEASE_OUTCOME_FORMAT
            )
            if self.format != expected:
                raise ValueError("Product release report format does not match scoped outcomes")
        elif (
            self.outcomes is not None
            or self.outcome_snapshots is not None
            or self.outcome_checked_bytes is not None
            or (self.format == PRODUCT_RELEASE_REFERENCE_FORMAT) != (self.references is not None)
        ):
            raise ValueError("Product release report format does not match selected evidence")
        return self
class ProductRelease(Model):
    id: UUID
    tenant_id: UUID
    product_id: UUID
    product_name: Name
    product_revision: Revision
    name: Name
    request: ProductReleaseRequest
    report: ProductReleaseReport
    report_hash: ReleaseHash
    status: Literal["proposed", "approved"]
    created_at: str
    created_by: str
    approved_at: str | None
    approved_by: str | None

    @model_validator(mode="after")
    def selected_evidence_matches_request(self) -> Self:
        report, request = self.report, self.request
        if report.quality_coverage != request.quality_coverage:
            raise ValueError("Product report quality coverage does not match its request")
        members = {member.role: member for member in report.members}
        if len(members) != len(report.members):
            raise ValueError("Product release report repeats a role")

        if (request.references is None) != (report.references is None):
            raise ValueError("Product release reference selections are missing from the report")
        for selected_reference, recorded_reference in zip(
            request.references or [], report.references or [], strict=True
        ):
            if (
                recorded_reference.output_role != selected_reference.output_role
                or recorded_reference.reference_role != selected_reference.reference_role
                or recorded_reference.evaluation_id != selected_reference.evaluation_id
                or recorded_reference.report_hash != selected_reference.report_hash
                or recorded_reference.policy != selected_reference.policy
                or selected_reference.output_role not in members
                or selected_reference.reference_role not in members
                or (selected_reference.policy == "require_pass" and not recorded_reference.passed)
            ):
                raise ValueError("Product release reference does not match its selection")

        if (request.outcomes is None) != (report.outcomes is None):
            raise ValueError("Product release outcome selections are missing from the report")
        if request.outcomes is None:
            return self
        if report.outcomes is None or report.outcome_snapshots is None:
            raise ValueError("Product release outcome evidence is incomplete")
        snapshots = {
            UUID(snapshot.evidence.outcome.receipt.plan_id): snapshot
            for snapshot in report.outcome_snapshots
        }
        selected_plans = {selection.plan_id for selection in request.outcomes}
        if (
            len(snapshots) != len(report.outcome_snapshots)
            or set(snapshots) != selected_plans
            or list(snapshots) != sorted(snapshots, key=str)
            or report.outcome_checked_bytes
            != sum(item.evidence.outcome.checked_bytes for item in snapshots.values())
        ):
            raise ValueError("Product release outcome snapshots do not match its selections")
        for selected, recorded in zip(request.outcomes, report.outcomes, strict=True):
            snapshot = snapshots[selected.plan_id]
            outcome = snapshot.evidence.outcome
            member = members.get(selected.role)
            output = next(
                (item for item in outcome.materialized_outputs if item.step_id == selected.step_id),
                None,
            )
            if (
                member is None
                or output is None
                or outcome.tenant_id != self.tenant_id
                or outcome.receipt.plan_id != str(selected.plan_id)
                or outcome.format != plan_outcome_format(selected.outcome_version or "v1")
                or snapshot.evidence.outcome_hash != selected.outcome_hash
                or recorded.role != selected.role
                or recorded.plan_id != selected.plan_id
                or recorded.outcome_hash != selected.outcome_hash
                or recorded.snapshot_hash != _hash(snapshot.model_dump(mode="json"))
                or recorded.step_id != selected.step_id
                or recorded.policy != selected.policy
                or recorded.status != outcome.receipt.status
                or recorded.checkpoint_state != outcome.checkpoint_state
                or recorded.version_id != output.version_id
                or member.version_id != output.version_id
                or member.checksum != output.checksum
                or member.nbytes != output.nbytes
                or (selected.policy == "require_success" and recorded.status != "succeeded")
            ):
                raise ValueError("Product release outcome does not match its selection and role")
        return self
def _quality_basis(
    member: ProductMember,
) -> Literal["not_attached", "raw_values", "selected_validity", "other_validity"]:
    evidence = member.quality_release
    if evidence is None:
        return "not_attached"
    if evidence.quality_mask is None:
        return "raw_values"
    if (
        member.validity is not None
        and evidence.quality_mask.binding_id == member.validity.binding.id
        and evidence.quality_mask.descriptor_hash == member.validity.binding.descriptor_hash
    ):
        return "selected_validity"
    return "other_validity"
def _reference_evidence(
    product: DataProductRevision,
    selection: ProductReferenceSelection,
    evaluation: ReferenceEvaluation,
) -> ProductReleaseReference:
    members = {member.role: member for member in product.definition.members}
    output = members.get(selection.output_role)
    reference = members.get(selection.reference_role)
    if output is None or reference is None or output.validity is None or reference.validity is None:
        raise Conflict("Reference evidence requires both named roles and their selected validity")
    output_mask = members.get(output.validity.mask_role)
    reference_mask = members.get(reference.validity.mask_role)
    result = evaluation.report.result
    comparison = result.reference
    intersection = comparison.validity_intersection if comparison is not None else None
    if output_mask is None or reference_mask is None or comparison is None or intersection is None:
        raise Conflict("Reference evidence lacks the product's selected mask roles")
    output_binding = output.validity.binding
    reference_binding = reference.validity.binding
    masks_match = all(
        (
            evidence.version_id == mask.version.id
            and evidence.checksum == mask.version.checksum
            and evidence.shape == mask.version.shape
            and evidence.nbytes == mask.version.nbytes
            and evidence.validity_binding.id == binding.id
            and evidence.validity_binding.descriptor_hash == binding.descriptor_hash
            and evidence.validity_binding.data_version_id == member.version.id
            and evidence.validity_binding.data_checksum == member.version.checksum
            and evidence.validity_binding.mask_version_id == mask.version.id
            and evidence.validity_binding.mask_checksum == mask.version.checksum
            and evidence.validity_binding.origin_kind == binding.origin.kind
        )
        for evidence, mask, member, binding in (
            (intersection.output_mask, output_mask, output, output_binding),
            (intersection.reference_mask, reference_mask, reference, reference_binding),
        )
    )
    if (
        evaluation.status != "approved"
        or evaluation.tenant_id != product.tenant_id
        or evaluation.id != selection.evaluation_id
        or evaluation.report_hash != selection.report_hash
        or evaluation.report.dataset != output.version.dataset
        or evaluation.report.name != output.version.name
        or result.version_id != output.version.id
        or result.checksum != output.version.checksum
        or result.dtype != output.version.dtype
        or result.shape != output.version.shape
        or result.nbytes != output.version.nbytes
        or comparison.version_id != reference.version.id
        or comparison.checksum != reference.version.checksum
        or comparison.dtype != reference.version.dtype
        or comparison.shape != reference.version.shape
        or comparison.nbytes != reference.version.nbytes
        or not masks_match
    ):
        raise Conflict("Reference evaluation does not match the exact product roles and validity")
    if selection.policy == "require_pass" and not evaluation.report.passed:
        raise Conflict("Reference evaluation did not pass the required comparison")
    return ProductReleaseReference(
        output_role=selection.output_role,
        reference_role=selection.reference_role,
        evaluation_id=evaluation.id,
        evaluation_hash=_hash(evaluation.model_dump(mode="json")),
        report_hash=evaluation.report_hash,
        policy=selection.policy,
        passed=evaluation.report.passed,
        included_count=intersection.included_count,
        excluded_count=intersection.excluded_count,
    )
def _outcome_evidence(
    product: DataProductRevision,
    selection: ProductOutcomeSelection,
    snapshot: PlanOutcomeSnapshot,
) -> ProductReleaseOutcome:
    member = next(
        (member for member in product.definition.members if member.role == selection.role), None
    )
    outcome = snapshot.evidence.outcome
    output = next(
        (item for item in outcome.materialized_outputs if item.step_id == selection.step_id), None
    )
    if (
        member is None
        or output is None
        or outcome.tenant_id != product.tenant_id
        or outcome.receipt.plan_id != str(selection.plan_id)
        or outcome.format != plan_outcome_format(selection.outcome_version or "v1")
        or snapshot.evidence.outcome_hash != selection.outcome_hash
        or output.version_id != member.version.id
        or output.checksum != member.version.checksum
        or output.dtype != member.version.dtype
        or output.shape != member.version.shape
        or output.nbytes != member.version.nbytes
        or output.dataset != member.version.dataset
        or output.name != member.version.name
    ):
        raise Conflict("Plan outcome does not match the exact product role and producing step")
    if outcome.format == COORDINATE_PLAN_OUTCOME_FORMAT:
        descriptor = next(
            (item for item in outcome.version_descriptors or [] if item.id == member.version.id),
            None,
        )
        if descriptor is None or canonical_json(
            descriptor.model_dump(mode="json")
        ) != canonical_json(version_integrity_descriptor(member.version.model_dump(mode="json"))):
            raise Conflict("Plan outcome descriptor does not match the exact product role")
    if selection.policy == "require_success" and outcome.receipt.status != "succeeded":
        raise Conflict("The selected plan outcome did not succeed")
    return ProductReleaseOutcome.model_validate(
        {
            "role": selection.role,
            "plan_id": selection.plan_id,
            "outcome_hash": selection.outcome_hash,
            "snapshot_hash": _hash(snapshot.model_dump(mode="json")),
            "step_id": selection.step_id,
            "version_id": member.version.id,
            "policy": selection.policy,
            "status": outcome.receipt.status,
            "checkpoint_state": outcome.checkpoint_state,
        }
    )
def _check_fitted_release_evidence(
    product: DataProductRevision,
    selections: list[ProductOutcomeSelection],
    snapshots: dict[UUID, PlanOutcomeSnapshot],
) -> None:
    members = {member.role: member for member in product.definition.members}
    for selection in selections:
        outcome = snapshots[selection.plan_id].evidence.outcome
        definition = PlanDefinition.model_validate(outcome.definition)
        if definition.paired_validity_policy != "pinned_fitted_apply_v1" or (
            selection.step_id != "values"
        ):
            continue
        if outcome.format != COORDINATE_PLAN_OUTCOME_FORMAT or not outcome.source_bindings:
            raise Conflict("Fitted transform releases require v3 target-binding evidence")
        target_binding = outcome.source_bindings[0]
        result_member = members[selection.role]
        if outcome.receipt.status == "succeeded":
            result_validity = result_member.validity
            if (
                result_validity is None
                or not isinstance(result_validity.binding.origin, TransformedValidityOrigin)
                or str(result_validity.binding.origin.plan_id) != outcome.receipt.plan_id
                or result_validity.binding.origin.plan_hash != outcome.receipt.plan_hash
                or result_validity.binding.origin.binding_id != target_binding.id
                or result_validity.binding.origin.binding_hash != target_binding.descriptor_hash
            ):
                raise Conflict("A succeeded fitted result requires its completed validity binding")
        elif selection.policy != "record_terminal" or result_member.validity is not None:
            raise Conflict("An incomplete fitted result must remain explicitly unbound")
        state_id = definition.steps[0].inputs[1]
        apply_descriptors = {str(item.id): item for item in outcome.version_descriptors or []}
        state_descriptor = apply_descriptors.get(state_id)
        if state_descriptor is None or not isinstance(state_descriptor.operation, dict):
            raise Conflict("Fitted transform state descriptor is unavailable")
        fit_plan_id = state_descriptor.operation.get("plan_id")
        matching = [
            item
            for item in selections
            if str(members[item.role].version.id) == state_id
            and str(item.plan_id) == fit_plan_id
            and item.step_id == "state"
            and item.outcome_version == "v3"
            and item.policy == "require_success"
        ]
        if not matching:
            raise Conflict("Fitted result release requires its exact state role and v3 fit outcome")
        fit_selection = matching[0]
        fit_outcome = snapshots[fit_selection.plan_id].evidence.outcome
        fit_definition = PlanDefinition.model_validate(fit_outcome.definition)
        if (
            fit_outcome.format != COORDINATE_PLAN_OUTCOME_FORMAT
            or fit_outcome.receipt.status != "succeeded"
            or fit_definition.paired_validity_policy != "pinned_fitted_fit_v1"
            or not fit_outcome.source_bindings
        ):
            raise Conflict("Fitted state requires a succeeded, source-pinned fit outcome")
        fit_descriptors = {str(item.id): item for item in fit_outcome.version_descriptors or []}
        if (
            state_id not in fit_descriptors
            or fit_descriptors[state_id] != state_descriptor
            or canonical_json(state_descriptor.model_dump(mode="json"))
            != canonical_json(
                version_integrity_descriptor(
                    members[fit_selection.role].version.model_dump(mode="json")
                )
            )
        ):
            raise Conflict("Fitted state differs between the product and its selected outcomes")
        training_binding = fit_outcome.source_bindings[0]

        def contains_binding(binding: ValidityBinding) -> bool:
            return any(
                member.version.id == binding.data_version_id
                and member.validity is not None
                and member.validity.binding == binding
                and members[member.validity.mask_role].version.id == binding.mask_version_id
                for member in members.values()
            )

        if not contains_binding(training_binding) or not contains_binding(target_binding):
            raise Conflict("Fitted release requires exact training and target data/mask roles")
        try:
            training = fit_descriptors[str(training_binding.data_version_id)]
            target = apply_descriptors[str(target_binding.data_version_id)]
            training_semantics = read_semantics(
                training.metadata, tuple(training.shape), training.operation
            )
            target_semantics = read_semantics(
                target.metadata, tuple(target.shape), target.operation
            )
            if training_semantics is None or target_semantics is None:
                raise TensorusError("Fitted source semantics are unavailable")
            require_fitted_domain(
                training_semantics,
                target_semantics,
                tuple(training.shape),
                tuple(target.shape),
                parameters(fit_definition.steps[0])["axis"],
            )
        except (TensorusError, KeyError, TypeError, ValueError) as exc:
            raise Conflict("Fitted training and target domains are inconsistent") from exc
def product_release_report(
    product: DataProductRevision,
    request: ProductReleaseRequest,
    proposed_by: str,
    specs: dict[str, TensorSpec],
    references: dict[UUID, ReferenceEvaluation] | None = None,
    snapshots: dict[UUID, PlanOutcomeSnapshot] | None = None,
) -> ProductReleaseReport:
    if request.product_revision != product.revision or request.descriptor_hash != (
        product.descriptor_hash
    ):
        raise Conflict("The selected product revision or descriptor hash changed")
    members: list[ProductReleaseMember] = []
    versions: dict[UUID, int] = {}
    for member in product.definition.members:
        quality = member.quality_release
        if request.quality_coverage == "require_all" and quality is None:
            raise Conflict("Every product role must have approved quality evidence")
        versions[member.version.id] = member.version.nbytes
        spec = specs[str(member.version.dataset_id)]
        members.append(
            ProductReleaseMember(
                role=member.role,
                version_id=member.version.id,
                checksum=member.version.checksum,
                nbytes=member.version.nbytes,
                write_spec=spec,
                write_spec_hash=_hash(spec.model_dump(mode="json")),
                contract_hash=member.contract.definition_hash,
                validity_selected=member.validity is not None,
                quality_basis=_quality_basis(member),
                quality_report_hash=quality.report_hash if quality else None,
            )
        )
    reference_evidence = None
    if request.references is not None:
        if references is None or set(references) != {
            selection.evaluation_id for selection in request.references
        }:
            raise Conflict("Selected reference evaluations are unavailable")
        reference_evidence = [
            _reference_evidence(product, selection, references[selection.evaluation_id])
            for selection in request.references
        ]
    outcome_evidence = None
    ordered_snapshots = None
    outcome_checked_bytes = None
    if request.outcomes is not None:
        if snapshots is None or set(snapshots) != {
            selection.plan_id for selection in request.outcomes
        }:
            raise Conflict("Selected terminal plan outcomes are unavailable")
        ordered_snapshots = [snapshots[plan_id] for plan_id in sorted(snapshots, key=str)]
        if len(
            canonical_json([item.model_dump(mode="json") for item in ordered_snapshots]).encode()
        ) > (MAX_PRODUCT_OUTCOME_BYTES):
            raise LimitExceeded("Product release plan outcome evidence exceeds 512 KiB")
        outcome_evidence = [
            _outcome_evidence(product, selection, snapshots[selection.plan_id])
            for selection in request.outcomes
        ]
        _check_fitted_release_evidence(product, request.outcomes, snapshots)
        outcome_checked_bytes = sum(
            item.evidence.outcome.checked_bytes for item in ordered_snapshots
        )
    return ProductReleaseReport(
        format=(
            PRODUCT_RELEASE_RELATIONSHIP_FORMAT
            if product.definition.relationships is not None
            else PRODUCT_RELEASE_COORDINATE_OUTCOME_FORMAT
            if ordered_snapshots
            and any(
                item.evidence.outcome.format == COORDINATE_PLAN_OUTCOME_FORMAT
                for item in ordered_snapshots
            )
            else PRODUCT_RELEASE_SCOPED_OUTCOME_FORMAT
            if ordered_snapshots
            and any(
                item.evidence.outcome.format == SCOPED_PLAN_OUTCOME_FORMAT
                for item in ordered_snapshots
            )
            else PRODUCT_RELEASE_OUTCOME_FORMAT
            if outcome_evidence
            else PRODUCT_RELEASE_REFERENCE_FORMAT
            if reference_evidence
            else PRODUCT_RELEASE_FORMAT
        ),
        tenant_id=product.tenant_id,
        product_id=product.id,
        product_name=product.name,
        product_revision=product.revision,
        descriptor_hash=product.descriptor_hash,
        release_name=request.name,
        proposed_by=proposed_by,
        quality_coverage=request.quality_coverage,
        checked_bytes=sum(versions.values()),
        checked_versions=len(versions),
        members=members,
        relationships=product.definition.relationships,
        references=reference_evidence,
        outcomes=outcome_evidence,
        outcome_snapshots=ordered_snapshots,
        outcome_checked_bytes=outcome_checked_bytes,
    )
