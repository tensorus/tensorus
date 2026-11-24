"""Use-specific review of an imported fitted payload without transferring authority."""
from datetime import datetime
from typing import Any
from tensorus.arrays import canonical_hash
from tensorus.arrays import canonical_json
from tensorus.errors import Conflict
from tensorus.errors import StorageError
from tensorus.models import Operation
from tensorus.models import PlanDefinition
from tensorus.models import Step
from tensorus.operations import FittedScalingParameters
from tensorus.operations import ImportedFittedStateSelection
from tensorus.operations import TensorShape
from tensorus.operations import parameters
from tensorus.semantics import derive_semantics
from tensorus.semantics import read_semantics
def derive_reviewed_imported_state(
    imported: dict[str, Any],
    release: dict[str, Any],
    product: "DataProductRevision",
    state_id: str,
    expected_checksum: str,
    selection: ImportedFittedStateSelection,
) -> "ReviewedImportedFittedState":
    from tensorus.arrays import version_integrity_descriptor
    from tensorus.product_portability import ProductReleaseImportRequest
    from tensorus.product_releases import ProductRelease
    from tensorus.validity import ReviewedImportedFittedState, _v2_fitted_fit_definition

    local_release = ProductRelease.model_validate(release)
    if imported["source_hash"] != selection.source_hash:
        raise Conflict("Expected imported product source hash does not match")
    if (
        imported["id"] != str(selection.import_id)
        or str(local_release.id) != str(selection.local_release_id)
        or local_release.status != "approved"
        or local_release.report_hash != selection.local_release_report_hash
        or local_release.approved_by is None
        or local_release.approved_at is None
        or local_release.approved_by == local_release.created_by
        or str(product.id) != imported["product_id"]
        or product.revision != imported["product_revision"]
        or local_release.product_id != product.id
        or local_release.product_revision != product.revision
        or local_release.request.descriptor_hash != product.descriptor_hash
        or local_release.request.references is not None
        or local_release.request.outcomes is not None
        or canonical_hash(local_release.report.model_dump(mode="json")) != local_release.report_hash
    ):
        raise Conflict("The selected local product release is not approved or changed")
    try:
        imported_at = datetime.fromisoformat(imported["created_at"])
        reviewed_at = datetime.fromisoformat(release["created_at"])
        if imported_at.utcoffset() is None or reviewed_at.utcoffset() is None:
            raise ValueError("Naive review timestamp")
    except (TypeError, ValueError) as exc:
        raise StorageError("Imported product review timestamps are invalid") from exc
    if reviewed_at < imported_at:
        raise Conflict("The local product review must be prepared after the source import")

    request = ProductReleaseImportRequest.model_validate(imported["request"])
    source = request.source
    original = {member.role: member for member in source.product.definition.members}
    local = {member.role: member for member in product.definition.members}
    if (
        request.descriptor_hash != product.descriptor_hash
        or set(original) != set(local)
        or source.product.definition.relationships != product.definition.relationships
    ):
        raise Conflict("The destination product does not match the source role mapping")
    source_reports = {member.role: member for member in source.release.report.members}
    local_reports = {member.role: member for member in local_release.report.members}
    if set(source_reports) != set(original) or set(local_reports) != set(local):
        raise Conflict("The selected product release role reports are incomplete")
    forward: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for role, member in original.items():
        target = local[role]
        source_id, target_id = str(member.version.id), str(target.version.id)
        if (
            member.version.dtype != target.version.dtype
            or member.version.shape != target.version.shape
            or member.version.nbytes != target.version.nbytes
            or member.version.checksum != target.version.checksum
            or canonical_json(member.version.metadata) != canonical_json(target.version.metadata)
            or member.contract.definition_hash != target.contract.definition_hash
            or source_reports[role].write_spec_hash != local_reports[role].write_spec_hash
            or local_reports[role].version_id != target.version.id
            or local_reports[role].checksum != target.version.checksum
            or local_reports[role].nbytes != target.version.nbytes
            or local_reports[role].contract_hash != target.contract.definition_hash
            or local_reports[role].validity_selected != (target.validity is not None)
            or (member.validity is None) != (target.validity is None)
            or (
                member.validity is not None
                and target.validity is not None
                and member.validity.mask_role != target.validity.mask_role
            )
            or target.quality_release is not None
            or (target.validity is not None and target.validity.binding.origin.kind != "asserted")
            or (source_id in forward and forward[source_id] != target_id)
            or (target_id in reverse and reverse[target_id] != source_id)
        ):
            raise Conflict("The destination product does not match the source role mapping")
        forward[source_id], reverse[target_id] = target_id, source_id
    state_member = original.get(selection.state_role)
    destination = local.get(selection.state_role)
    if state_member is None or destination is None:
        raise Conflict("The selected imported state role is unavailable")
    if (
        str(destination.version.id) != state_id
        or destination.version.checksum != expected_checksum
        or destination.version.dtype != "float64"
        or destination.version.operation is not None
        or destination.version.parents
    ):
        raise Conflict("The imported state must be the exact unmodified destination payload")
    source_operation = state_member.version.operation
    if (
        not isinstance(source_operation, dict)
        or source_operation.get("name") != Operation.fit_selected_preprocessing
        or source_operation.get("step_id") != "state"
        or not isinstance(source_operation.get("plan_id"), str)
    ):
        raise Conflict("The source role does not claim a fitted state")
    selections = [
        item
        for item in source.release.request.outcomes or []
        if item.role == selection.state_role
        and item.step_id == "state"
        and item.policy == "require_success"
        and item.outcome_version == "v3"
        and str(item.plan_id) == source_operation["plan_id"]
    ]
    snapshots = [
        item
        for item in source.release.report.outcome_snapshots or []
        if item.evidence.outcome.receipt.plan_id == source_operation["plan_id"]
    ]
    if len(selections) != 1 or len(snapshots) != 1:
        raise Conflict("The source claim must include one exact succeeded v3 fit outcome")
    selected, snapshot = selections[0], snapshots[0]
    outcome = snapshot.evidence.outcome
    definition = PlanDefinition.model_validate(outcome.definition)
    if (
        snapshot.evidence.outcome_hash != selected.outcome_hash
        or outcome.receipt.status != "succeeded"
        or not _v2_fitted_fit_definition(definition)
        or not outcome.source_bindings
        or len(outcome.source_bindings) != 1
    ):
        raise Conflict("The selected source fit outcome is inconsistent")
    descriptors = {str(item.id): item for item in outcome.version_descriptors or []}
    claimed_state = descriptors.get(str(state_member.version.id))
    if claimed_state is None or canonical_json(claimed_state.model_dump(mode="json")) != (
        canonical_json(version_integrity_descriptor(state_member.version.model_dump(mode="json")))
    ):
        raise Conflict("The source fit outcome does not pin the selected state")
    source_binding = outcome.source_bindings[0]
    training_roles = [
        member.role
        for member in original.values()
        if member.validity is not None and member.validity.binding == source_binding
    ]
    if len(training_roles) != 1:
        raise Conflict("The source claim must identify one exact training data/mask role")
    training = local[training_roles[0]]
    if training.validity is None:
        raise Conflict("The local product must bind the copied training data and mask")
    mask = local[training.validity.mask_role]
    fit = FittedScalingParameters.model_validate(parameters(definition.steps[0]))
    training_shape = TensorShape(tuple(training.version.shape), training.version.dtype)
    mask_shape = TensorShape(tuple(mask.version.shape), mask.version.dtype)
    state_shape = TensorShape(tuple(destination.version.shape), destination.version.dtype)
    training_semantics = read_semantics(
        training.version.metadata, training_shape.shape, training.version.operation
    )
    mask_semantics = read_semantics(mask.version.metadata, mask_shape.shape, mask.version.operation)
    state_semantics = read_semantics(
        destination.version.metadata, state_shape.shape, destination.version.operation
    )
    if training_semantics is None or mask_semantics is None or state_semantics is None:
        raise Conflict("Imported fitted-state semantic declarations are unavailable")
    semantic_step = Step(
        id="state",
        operation=Operation.fit_selected_preprocessing,
        inputs=[str(training.version.id), str(mask.version.id)],
        output="state",
        parameters=fit.model_dump(mode="json"),
    )
    derived = derive_semantics(
        semantic_step,
        [training_shape, mask_shape],
        [training_semantics, mask_semantics],
        state_shape,
        "v2",
    )
    if derived is None or canonical_json(derived.model_dump(mode="json")) != canonical_json(
        state_semantics.model_dump(mode="json")
    ):
        raise Conflict("Imported fitted-state semantics do not match the reviewed training domain")
    reviewed = ReviewedImportedFittedState(
        state_version_id=destination.version.id,
        state_checksum=destination.version.checksum,
        state_shape=destination.version.shape,
        algorithm=fit.algorithm,
        algorithm_version=fit.algorithm_version,
        axis=fit.axis,
        epsilon=fit.epsilon,
        training_binding=training.validity.binding,
        training_shape=training.version.shape,
        training_semantics=training_semantics,
        selection=selection,
        claimed_fit_plan_id=selected.plan_id,
        claimed_fit_plan_hash=outcome.receipt.plan_hash,
        claimed_fit_outcome_hash=selected.outcome_hash,
        local_reviewed_by=release["approved_by"],
        local_reviewed_at=release["approved_at"],
        limitations=[
            "The source fit, source approval and training designation are unauthenticated claims.",
            "The local product review verifies selected payloads but does not recompute the fit.",
            "A distinct local actor must approve each imported-state application plan.",
            "The local training binding is an assertion, not proof of observation truth.",
        ],
    )
    return reviewed
