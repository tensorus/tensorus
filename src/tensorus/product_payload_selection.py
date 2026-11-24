"""Reviewed preparation of one product role and its explicitly selected mask."""

import re
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import ValidationError

from tensorus import responses as dto
from tensorus.arrays import canonical_hash, canonical_json
from tensorus.bundle_inspection_contract import validate_progress
from tensorus.bundle_restore_heads import HeadExpectation, _read_head, _selected_assessment
from tensorus.errors import Conflict, TensorusError
from tensorus.models import NpyWriteCondition, WriteContractBinding
from tensorus.mutation_outcome import verify_after_mutation
from tensorus.product_bundle import (
    MAX_BUNDLE_VERSION_RESPONSE_BYTES,
    ProductBundleManifest,
    _capture_write_credential,
    _checked_reviewed_identity,
    _checked_write_identity,
    _destination_state,
    _inspect_copied_payload,
    _inspection_review,
    _product_payload_key,
    _read_bundle,
    _read_model,
    _require_conditional_ingestion,
    _require_write_credential,
    import_product_bundle,
)
from tensorus.product_bundle_restore import (
    ProductBundleRestoreRequest,
    _checked_asserted_binding,
    _checked_request,
    _find_asserted_binding,
    _review,
)
from tensorus.validity import ValidityBinding, ValidityBindingCreate

if TYPE_CHECKING:
    from tensorus.client import Client


SELECTION_FORMAT = "tensorus-product-payload-selection-v1"
REVIEW_FORMAT = "tensorus-product-payload-selection-review-v1"


def _selection(
    manifest: ProductBundleManifest,
    request: ProductBundleRestoreRequest,
    review: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    members = {member.role: member for member in manifest.source.product.definition.members}
    member = members.get(role)
    if member is None:
        raise TensorusError("The selected role is not in this reviewed product bundle")
    selected_roles = {role}
    assertions: list[dict[str, str]] = []
    if member.validity is not None:
        mask_role = member.validity.mask_role
        mask = members.get(mask_role)
        binding = member.validity.binding
        if (
            mask is None
            or binding.data_version_id != member.version.id
            or binding.mask_version_id != mask.version.id
        ):
            raise TensorusError("The selected product validity pair is inconsistent")
        selected_roles.add(mask_role)
        assertions.append(
            {
                "data_role": role,
                "mask_role": mask_role,
                "source_binding_id": str(binding.id),
                "source_descriptor_hash": binding.descriptor_hash,
                "data_version_id": str(binding.data_version_id),
                "mask_version_id": str(binding.mask_version_id),
            }
        )
    reviewed_roles = {item["role"]: item for item in review["roles"]}
    source_ids = {members[name].version.id for name in selected_roles}
    if not source_ids <= {entry.source_version_id for entry in manifest.payloads}:
        raise TensorusError("The selected role requires payloads absent from the product bundle")
    payloads = []
    for source_id in sorted(source_ids, key=str):
        names = sorted(name for name in selected_roles if members[name].version.id == source_id)
        target = request.destinations[source_id]
        payloads.append(
            {
                "source_version_id": str(source_id),
                "destination_dataset": target.dataset,
                "destination_dataset_id": reviewed_roles[names[0]]["destination_dataset_id"],
                "destination_name": target.name,
                "expected_version": 0,
                "roles": names,
            }
        )
    selection = {
        "format": SELECTION_FORMAT,
        "restore_review_hash": review["review_hash"],
        "bundle_hash": manifest.content_hash,
        "tenant_id": review["tenant_id"],
        "actor_key_id": review["actor_key_id"],
        "actor_role": review["actor_role"],
        "role": role,
        "payloads": payloads,
        "selected_validity": assertions,
        "validity_assertions_required": bool(assertions),
    }
    return {**selection, "selection_hash": canonical_hash(selection)}


def _subset_review(review: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    ids = {item["source_version_id"] for item in selection["payloads"]}
    roles = {role for item in selection["payloads"] for role in item["roles"]}
    return {
        **review,
        "request": {
            **review["request"],
            "destinations": {
                key: value for key, value in review["request"]["destinations"].items() if key in ids
            },
        },
        "roles": [
            {
                **item,
                "validity_mask_role": (
                    item["validity_mask_role"] if item["role"] == selection["role"] else None
                ),
            }
            for item in review["roles"]
            if item["role"] in roles
        ],
    }


def _validate_result(
    result: dict[str, Any],
    selection: dict[str, Any],
    review: dict[str, Any],
    manifest: ProductBundleManifest,
) -> None:
    try:
        fields = {
            "format",
            "review_hash",
            "bundle_hash",
            "selection_hash",
            "selection",
            "payloads",
            "validity_bindings",
            "heads",
            "review_policy_current",
            "status",
            "complete",
            "authorizes_restore",
            "point_in_time_atomic",
            "retry_guaranteed",
            "source_claims_authenticated",
            "approval_transferred",
            "local_product_created",
            "local_release_created",
        }
        if (
            not isinstance(result, dict)
            or set(result) != fields
            or result["format"] != REVIEW_FORMAT
            or result["review_hash"] != review["review_hash"]
            or result["bundle_hash"] != manifest.content_hash
            or result["selection_hash"] != selection["selection_hash"]
            or canonical_json(result["selection"]) != canonical_json(selection)
            or result["review_policy_current"] is not True
            or type(result["complete"]) is not bool
            or any(
                result[field] is not False
                for field in (
                    "authorizes_restore",
                    "point_in_time_atomic",
                    "retry_guaranteed",
                    "source_claims_authenticated",
                    "approval_transferred",
                    "local_product_created",
                    "local_release_created",
                )
            )
        ):
            raise ValueError("Invalid selected-product review identity")
        ids = {item["source_version_id"] for item in selection["payloads"]}
        if (
            not isinstance(result["payloads"], dict)
            or set(result["payloads"]) != ids
            or not isinstance(result["heads"], dict)
            or set(result["heads"]) != ids
            or not isinstance(result["validity_bindings"], dict)
        ):
            raise ValueError("Selected-product review coverage is incomplete")
        copied = any(item["status"] == "committed" for item in result["payloads"].values())
        progress = {
            "format": "tensorus-product-bundle-restore-payload-observation-v1",
            "review_hash": review["review_hash"],
            "current_review_hash": review["review_hash"],
            "bundle_hash": review["bundle_hash"],
            "source_claims_authenticated": False,
            "payloads": result["payloads"],
            "validity_bindings": result["validity_bindings"],
            "product": {"status": "not_checked", "record": None},
            "source_import": {"status": "not_checked", "record": None},
            "complete": None,
            "status": "payloads_observed" if copied else "no_payloads_observed",
            "review_policy_current": True,
            "continuation": "not_assessed",
            "next_action": "inspect_complete_progress_before_retry",
            "point_in_time_atomic": False,
            "retry_guaranteed": False,
            "approval_transferred": False,
            "local_release_created": False,
            "retained_records_checked": False,
            "completion_assessed": False,
            "heads_checked": False,
            "names_checked": False,
            "authorizes_restore": False,
        }
        validate_progress(
            "product", progress, _subset_review(review, selection), payloads_only=True
        )
        sources = {
            str(member.version.id): member.version
            for member in manifest.source.product.definition.members
        }
        dataset_ids = {
            item["source_version_id"]: item["destination_dataset_id"]
            for item in selection["payloads"]
        }
        blocked = False
        version_ids: set[UUID] = set()
        tensor_ids: set[UUID] = set()
        for source_id in ids:
            evidence = result["payloads"][source_id]
            head = result["heads"][source_id]
            if not isinstance(head, dict) or set(head) != {
                "expected_version",
                "head",
                "assessment",
            }:
                raise ValueError("Invalid selected-product head evidence")
            value = head["head"]
            if value is not None and (
                not isinstance(value, dict)
                or set(value) != {"tensor_id", "version_id", "version", "deleted"}
                or str(UUID(value["tensor_id"])) != value["tensor_id"]
                or str(UUID(value["version_id"])) != value["version_id"]
                or type(value["version"]) is not int
                or value["version"] < 1
                or type(value["deleted"]) is not bool
            ):
                raise ValueError("Invalid selected-product head")
            if type(head["expected_version"]) is not int or head["expected_version"] != 0:
                raise ValueError("Invalid selected-product expected version")
            assessment = _selected_assessment(0, value, evidence)
            if head["assessment"] != assessment:
                raise ValueError("Selected-product head contradicts its receipt")
            blocked |= assessment in ("head_advanced", "tombstoned", "receipt_tombstoned")
            if evidence["version"] is not None:
                version = dto.TensorVersion.model_validate_json(
                    canonical_json(evidence["version"]), strict=True
                )
                source = sources[source_id]
                if (
                    str(version.dataset_id) != dataset_ids[source_id]
                    or version.version != 1
                    or version.head_version < 1
                    or version.shape != source.shape
                    or version.dtype != source.dtype
                    or version.nbytes != source.nbytes
                    or version.checksum != source.checksum
                    or canonical_json(version.metadata) != canonical_json(source.metadata)
                    or version.operation is not None
                    or version.parents
                    or version.id in version_ids
                    or version.tensor_id in tensor_ids
                ):
                    raise ValueError("Selected-product payload contradicts its source")
                version_ids.add(version.id)
                tensor_ids.add(version.tensor_id)
        complete = (
            not blocked
            and all(item["status"] == "committed" for item in result["payloads"].values())
            and all(item["status"] == "committed" for item in result["validity_bindings"].values())
        )
        status = "blocked_head_conflict" if blocked else "complete" if complete else "ready"
        if result["complete"] is not complete or result["status"] != status:
            raise ValueError("Selected-product summary contradicts its observations")
    except (
        AttributeError,
        Conflict,
        KeyError,
        RecursionError,
        StopIteration,
        TypeError,
        ValueError,
        ValidationError,
    ) as exc:
        raise TensorusError("Selected product payload evidence is inconsistent") from exc


def validate_product_payload_selection_review(
    result: dict[str, Any],
    manifest: ProductBundleManifest,
    request: ProductBundleRestoreRequest,
    review: dict[str, Any],
    role: str,
) -> None:
    """Validate a response against the original archive, mapping, and selected role."""
    try:
        _validate_result(result, _selection(manifest, request, review, role), review, manifest)
    except (AttributeError, KeyError, TypeError, ValueError, TensorusError) as exc:
        raise ValueError("Invalid selected-product payload evidence") from exc


def inspect_product_payload_selection(
    client: "Client",
    payload: bytes,
    request: ProductBundleRestoreRequest,
    *,
    reviewed: dict[str, Any],
    role: str,
    _credential: str | None = None,
) -> dict[str, Any]:
    """Review one role's payloads without observing unrelated destinations or names."""
    if not isinstance(reviewed, dict) or not isinstance(reviewed.get("review_hash"), str):
        raise TensorusError("The complete original restore review is required")
    manifest, arrays = _read_bundle(client, payload)
    del arrays
    checked = _checked_request(request)
    credential = _capture_write_credential(client) if _credential is None else _credential
    _require_write_credential(client, credential)
    _require_conditional_ingestion(client, credential=credential)
    client._require_conditional_npy_historical_receipt(_credential=credential)
    current = _review(client, manifest, checked, _credential=credential)
    review = _inspection_review(current, reviewed["review_hash"], reviewed, "roles")
    identity = _checked_reviewed_identity(client, credential, review)
    if current["review_hash"] != review["review_hash"]:
        raise Conflict("Restore policy changed; selected product preparation needs a new review")
    selection = _selection(manifest, checked, review, role)
    roles = {item["role"]: item for item in review["roles"]}
    sources = {
        member.version.id: member.version for member in manifest.source.product.definition.members
    }
    payloads: dict[str, Any] = {}
    heads: dict[str, Any] = {}
    for item in selection["payloads"]:
        source_id = UUID(item["source_version_id"])
        target = checked.destinations[source_id]
        role_review = roles[item["roles"][0]]
        active = role_review["active_write_contract"]
        condition = NpyWriteCondition(
            dataset_id=role_review["destination_dataset_id"],
            write_contract_generation=active["generation"],
        )
        contract = WriteContractBinding.model_validate(active) if active["generation"] else None
        evidence = _inspect_copied_payload(
            client,
            identity,
            sources[source_id].model_dump(mode="json"),
            target,
            0,
            condition,
            contract,
            contract,
            _product_payload_key(manifest.content_hash, source_id, target),
            credential=credential,
            allow_tombstoned=True,
        )
        payloads[str(source_id)] = evidence
        target_head = HeadExpectation(
            str(source_id), target.dataset, item["destination_dataset_id"], target.name, 0
        )
        head = _read_head(client, target_head, str(identity.id), credential=credential)
        heads[str(source_id)] = {
            "expected_version": 0,
            "head": head,
            "assessment": _selected_assessment(0, head, evidence),
        }
    bindings: dict[str, Any] = {}
    for item in selection["selected_validity"]:
        data = payloads[item["data_version_id"]]["version"]
        mask = payloads[item["mask_version_id"]]["version"]
        if data is None or mask is None:
            bindings[item["data_role"]] = {"status": "not_checked", "binding": None}
            continue
        binding = _find_asserted_binding(client, data, mask, identity.id, credential=credential)
        bindings[item["data_role"]] = {
            "status": "committed" if binding is not None else "not_observed",
            "binding": binding,
        }
    if _review(client, manifest, checked, _credential=credential) != current:
        raise Conflict("Restore policy changed during selected product inspection")
    blocked = any(
        item["assessment"] in ("head_advanced", "tombstoned", "receipt_tombstoned")
        for item in heads.values()
    )
    complete = (
        not blocked
        and all(item["status"] == "committed" for item in payloads.values())
        and all(item["status"] == "committed" for item in bindings.values())
    )
    result = {
        "format": REVIEW_FORMAT,
        "review_hash": review["review_hash"],
        "bundle_hash": manifest.content_hash,
        "selection_hash": selection["selection_hash"],
        "selection": selection,
        "payloads": payloads,
        "validity_bindings": bindings,
        "heads": heads,
        "review_policy_current": True,
        "status": "blocked_head_conflict" if blocked else "complete" if complete else "ready",
        "complete": complete,
        "authorizes_restore": False,
        "point_in_time_atomic": False,
        "retry_guaranteed": False,
        "source_claims_authenticated": False,
        "approval_transferred": False,
        "local_product_created": False,
        "local_release_created": False,
    }
    _validate_result(result, selection, review, manifest)
    _checked_write_identity(client, credential, identity)
    return result


def prepare_product_payload_selection(
    client: "Client",
    payload: bytes,
    request: ProductBundleRestoreRequest,
    *,
    expected_review_hash: str,
    role: str,
    expected_selection_hash: str,
    confirm_reviewed_selection: bool = False,
    confirm_unauthenticated_claims: bool = False,
    confirm_validity_assertions: bool = False,
    _credential: str | None = None,
) -> dict[str, Any]:
    """Prepare only the explicitly reviewed role and selected validity mask."""
    if confirm_reviewed_selection is not True or confirm_unauthenticated_claims is not True:
        raise TensorusError("Selected preparation requires both explicit confirmations")
    if type(confirm_validity_assertions) is not bool:
        raise TensorusError("Validity assertion confirmation must be a boolean")
    if not isinstance(expected_selection_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_selection_hash
    ):
        raise TensorusError("Invalid selected-product review hash")
    manifest, arrays = _read_bundle(client, payload)
    del arrays
    checked = _checked_request(request)
    credential = _capture_write_credential(client) if _credential is None else _credential
    identity = _checked_write_identity(client, credential)
    review = _review(client, manifest, checked, _credential=credential)
    if review["review_hash"] != expected_review_hash:
        raise Conflict("Restore review changed; review the destination again")
    selection = _selection(manifest, checked, review, role)
    if selection["selection_hash"] != expected_selection_hash:
        raise Conflict("Selected product review changed; review the selection again")
    if selection["validity_assertions_required"] and not confirm_validity_assertions:
        raise TensorusError(
            "Selected preparation requires confirmation of local validity assertions"
        )
    before = inspect_product_payload_selection(
        client, payload, checked, reviewed=review, role=role, _credential=credential
    )
    if before["status"] == "blocked_head_conflict":
        raise Conflict("A selected product destination head conflicts with the reviewed mapping")
    if before["complete"]:
        return before
    selected_ids = {UUID(item["source_version_id"]) for item in selection["payloads"]}
    states: dict[str, dict[str, Any]] = {}
    contracts: dict[str, WriteContractBinding | None] = {}
    for dataset in sorted({item["destination_dataset"] for item in selection["payloads"]}):
        _, state = _destination_state(client, dataset, identity.id, credential=credential)
        reviewed_role = next(
            item for item in review["roles"] if item["destination_dataset"] == dataset
        )
        if (
            state["dataset_id"] != reviewed_role["destination_dataset_id"]
            or state["write_spec_hash"] != reviewed_role["write_spec_hash"]
            or {
                key: state["active_write_contract"][key]
                for key in ("generation", "revision", "definition_hash")
            }
            != reviewed_role["active_write_contract"]
        ):
            raise Conflict("Destination policy changed during selected product preparation")
        states[dataset] = state
        active = state["active_write_contract"]
        contracts[dataset] = (
            WriteContractBinding.model_validate(
                {key: active[key] for key in ("generation", "revision", "definition_hash")}
            )
            if active["generation"]
            else None
        )

    def guard(dataset: str) -> None:
        _checked_write_identity(client, credential, identity)
        _, current = _destination_state(client, dataset, identity.id, credential=credential)
        if current != states[dataset]:
            raise Conflict("Destination policy changed during selected product preparation")
        _require_write_credential(client, credential)

    conditions = {
        dataset: NpyWriteCondition(
            dataset_id=state["dataset_id"],
            write_contract_generation=state["active_write_contract"]["generation"],
        )
        for dataset, state in states.items()
    }

    def perform() -> dict[str, Any]:
        copied = import_product_bundle(
            client,
            payload,
            checked.model_dump(mode="json")["destinations"],
            _conditions=conditions,
            _expected_contracts=contracts,
            _credential=credential,
            _expected_identity=identity,
            _guard=guard,
            _source_ids=selected_ids,
        )
        versions = copied["destination_versions"]
        for item in selection["selected_validity"]:
            data, mask = versions[item["data_version_id"]], versions[item["mask_version_id"]]
            guard(data["dataset"])
            guard(mask["dataset"])
            existing = _find_asserted_binding(
                client, data, mask, identity.id, credential=credential
            )
            if existing is None:
                response = client.create_validity_binding(
                    ValidityBindingCreate(data_version_id=data["id"], mask_version_id=mask["id"]),
                    _credential=credential,
                )
                binding = _checked_asserted_binding(response, data, mask, identity.id)
                stored = _read_model(
                    client,
                    f"/v1/validity-bindings/{binding['id']}",
                    ValidityBinding,
                    "Validity binding",
                    MAX_BUNDLE_VERSION_RESPONSE_BYTES,
                    credential=credential,
                )
                if canonical_json(stored.model_dump(mode="json")) != canonical_json(binding):
                    raise TensorusError(
                        "Destination returned contradictory retained validity evidence"
                    )
            guard(data["dataset"])
            guard(mask["dataset"])
        after = inspect_product_payload_selection(
            client, payload, checked, reviewed=review, role=role, _credential=credential
        )
        if after["selection_hash"] != expected_selection_hash or not after["complete"]:
            raise Conflict("Selected product preparation did not retain all reviewed payloads")
        _checked_write_identity(client, credential, identity)
        return after

    return verify_after_mutation(perform, "Selected product payload preparation")
