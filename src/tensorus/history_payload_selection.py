"""Reviewed preparation of one release's required history payloads."""

import re
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import ValidationError

from tensorus import responses as dto
from tensorus.arrays import canonical_hash, canonical_json
from tensorus.bundle_inspection_contract import validate_progress
from tensorus.bundle_restore_heads import HeadExpectation, _read_head, _selected_assessment
from tensorus.errors import Conflict, TensorusError
from tensorus.history_payload_bundle import (
    HistoryPayloadManifest,
    _history_payload_key,
    _read_history_payload_bundle,
    copy_history_payload_bundle,
)
from tensorus.history_payload_restore import (
    HistoryPayloadRestoreRequest,
    _checked_request,
    _review,
)
from tensorus.history_restore_selection import _release_id
from tensorus.models import NpyWriteCondition, WriteContractBinding
from tensorus.product_bundle import (
    MAX_BUNDLE_VERSION_RESPONSE_BYTES,
    _capture_write_credential,
    _checked_reviewed_identity,
    _checked_write_identity,
    _destination_state,
    _inspect_copied_payload,
    _inspection_review,
    _read_model,
    _require_conditional_ingestion,
    _require_write_credential,
)
from tensorus.product_bundle_restore import _checked_asserted_binding, _find_asserted_binding
from tensorus.validity import ValidityBinding, ValidityBindingCreate

if TYPE_CHECKING:
    from tensorus.client import Client


SELECTION_FORMAT = "tensorus-history-payload-selection-v1"
REVIEW_FORMAT = "tensorus-history-payload-selection-review-v1"


def _selection(
    manifest: HistoryPayloadManifest,
    request: HistoryPayloadRestoreRequest,
    review: dict[str, Any],
    release_id: UUID,
) -> dict[str, Any]:
    if release_id not in {item.release_id for item in request.releases}:
        raise TensorusError("The selected release is not in the reviewed restore request")
    release = next(
        (item for item in manifest.source.releases if item.release.id == release_id), None
    )
    if release is None:
        raise TensorusError("The selected release is not in this history bundle")
    required = {item.version_id for item in release.release.report.members}
    if release.lineage is not None:
        required.update(item.id for item in release.lineage.versions)
    selected_bindings: list[ValidityBinding] = []
    if request.assert_selected_validity and release.validity is not None:
        bindings = {item.id: item for item in release.validity.bindings}
        selected_bindings = [
            bindings[item] for item in sorted(release.validity.selected_binding_ids, key=str)
        ]
        for binding in selected_bindings:
            required.update((binding.data_version_id, binding.mask_version_id))
    sources = [entry.source for entry in manifest.payloads]
    if not required <= {source.id for source in sources}:
        raise TensorusError("The selected release requires payloads absent from the history bundle")
    if not required:
        raise TensorusError("The selected release needs no payload preparation; review its import")
    ordered = sorted(sources, key=lambda item: (str(item.tensor_id), item.version, str(item.id)))
    last_required: dict[UUID, int] = {}
    positions: dict[UUID, int] = {}
    for source in ordered:
        position = positions.get(source.tensor_id, 0)
        positions[source.tensor_id] = position + 1
        if source.id in required:
            last_required[source.tensor_id] = position
    datasets = {item["dataset"]: item["dataset_id"] for item in review["datasets"]}
    payloads: list[dict[str, Any]] = []
    positions.clear()
    for source in ordered:
        position = positions.get(source.tensor_id, 0)
        positions[source.tensor_id] = position + 1
        if position > last_required.get(source.tensor_id, -1):
            continue
        target = request.destinations[source.id]
        payloads.append(
            {
                "source_version_id": str(source.id),
                "destination_dataset": target.dataset,
                "destination_dataset_id": datasets[target.dataset],
                "destination_name": target.name,
                "expected_version": position,
                "required_directly": source.id in required,
            }
        )
    return {
        "format": SELECTION_FORMAT,
        "restore_review_hash": review["review_hash"],
        "bundle_hash": manifest.content_hash,
        "tenant_id": review["tenant_id"],
        "actor_key_id": review["actor_key_id"],
        "actor_role": review["actor_role"],
        "release_id": str(release_id),
        "payloads": payloads,
        "selected_validity": [
            {
                "source_binding_id": str(item.id),
                "source_descriptor_hash": item.descriptor_hash,
                "data_version_id": str(item.data_version_id),
                "mask_version_id": str(item.mask_version_id),
            }
            for item in selected_bindings
        ],
        "validity_assertions_required": bool(selected_bindings),
    }


def _contract(state: dict[str, Any]) -> WriteContractBinding | None:
    active = state["active_write_contract"]
    return (
        WriteContractBinding.model_validate(
            {field: active[field] for field in ("generation", "revision", "definition_hash")}
        )
        if active["generation"]
        else None
    )


def validate_history_payload_selection_review(
    result: object,
    manifest: HistoryPayloadManifest,
    request: HistoryPayloadRestoreRequest,
    review: dict[str, Any],
    release_id: UUID,
) -> None:
    """Validate selected observations without treating them as write authority."""
    try:
        selection = _selection(manifest, request, review, release_id)
        if (
            not isinstance(result, dict)
            or set(result)
            != {
                "format",
                "selection",
                "selection_hash",
                "payloads",
                "validity_bindings",
                "heads",
                "blocked_observed",
                "complete",
                "point_in_time_atomic",
                "retry_guaranteed",
                "authorizes_restore",
                "source_claims_authenticated",
            }
            or result["format"] != REVIEW_FORMAT
            or canonical_json(result["selection"]) != canonical_json(selection)
            or result["selection_hash"] != canonical_hash(selection)
            or any(
                result[field] is not False
                for field in (
                    "point_in_time_atomic",
                    "retry_guaranteed",
                    "authorizes_restore",
                    "source_claims_authenticated",
                )
            )
            or type(result["blocked_observed"]) is not bool
            or type(result["complete"]) is not bool
        ):
            raise ValueError("Selected payload review identity is inconsistent")
        selected_ids = {item["source_version_id"] for item in selection["payloads"]}
        binding_ids = {item["source_binding_id"] for item in selection["selected_validity"]}
        payloads = result["payloads"]
        bindings = result["validity_bindings"]
        heads = result["heads"]
        if (
            not isinstance(payloads, dict)
            or set(payloads) != selected_ids
            or not isinstance(bindings, dict)
            or set(bindings) != binding_ids
            or not isinstance(heads, dict)
            or set(heads) != selected_ids
        ):
            raise ValueError("Selected payload coverage is inconsistent")

        subset_review = {
            **review,
            "request": {
                **review["request"],
                "destinations": {
                    key: value
                    for key, value in review["request"]["destinations"].items()
                    if key in selected_ids
                },
            },
            "selected_validity": [
                item
                for item in review["selected_validity"]
                if item["source_binding_id"] in binding_ids
            ],
        }
        policy_blocked = any(
            item["continuation"] == "blocked_policy_changed" for item in payloads.values()
        )
        copied = any(item["status"] == "committed" for item in payloads.values())
        progress = {
            "format": "tensorus-history-payload-restore-payload-observation-v1",
            "review_hash": review["review_hash"],
            "current_review_hash": review["review_hash"],
            "bundle_hash": review["bundle_hash"],
            "source_claims_authenticated": False,
            "payloads": payloads,
            "validity_bindings": bindings,
            "history_imports": {
                item["release_id"]: {"status": "not_checked", "record": None}
                for item in review["request"]["releases"]
            },
            "complete": None,
            "status": (
                "blocked_policy_changed"
                if policy_blocked
                else "payloads_observed"
                if copied
                else "no_payloads_observed"
            ),
            "review_policy_current": all(
                item["review_policy_current"] for item in payloads.values()
            ),
            "continuation": "blocked_policy_changed" if policy_blocked else "not_assessed",
            "next_action": (
                "review_partial_and_select_new_destinations"
                if policy_blocked
                else "inspect_complete_progress_before_retry"
            ),
            "point_in_time_atomic": False,
            "retry_guaranteed": False,
            "source_lineage_recreated": False,
            "source_plans_recreated": False,
            "approval_transferred": False,
            "local_release_created": False,
            "retained_records_checked": False,
            "completion_assessed": False,
            "heads_checked": False,
            "names_checked": False,
            "authorizes_restore": False,
        }
        validate_progress("history", progress, subset_review, payloads_only=True)
        sources = {str(entry.source.id): entry.source for entry in manifest.payloads}
        tensor_ids: dict[UUID, str] = {}
        source_tensors: dict[str, UUID] = {}
        version_ids: set[str] = set()
        blocked = policy_blocked
        for item in selection["payloads"]:
            identifier = item["source_version_id"]
            evidence = payloads[identifier]
            head = heads[identifier]
            if not isinstance(head, dict) or set(head) != {"head", "assessment"}:
                raise ValueError("Selected destination head evidence is invalid")
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
                raise ValueError("Selected destination head is invalid")
            assessment = _selected_assessment(item["expected_version"], value, evidence)
            if head["assessment"] != assessment:
                raise ValueError("Selected destination head assessment is inconsistent")
            blocked |= assessment in ("head_advanced", "tombstoned", "receipt_tombstoned")
            version = evidence["version"]
            if version is not None:
                checked = dto.TensorVersion.model_validate_json(
                    canonical_json(version), strict=True
                )
                source = sources[identifier]
                if (
                    canonical_json(version) != canonical_json(checked.model_dump(mode="json"))
                    or str(checked.dataset_id) != item["destination_dataset_id"]
                    or checked.version != item["expected_version"] + 1
                    or checked.dtype != source.dtype
                    or checked.shape != source.shape
                    or checked.nbytes != source.nbytes
                    or checked.checksum != source.checksum
                    or canonical_json(checked.metadata) != canonical_json(source.metadata)
                    or checked.operation is not None
                    or checked.parents
                    or str(checked.id) in version_ids
                    or tensor_ids.setdefault(source.tensor_id, str(checked.tensor_id))
                    != str(checked.tensor_id)
                    or source_tensors.setdefault(str(checked.tensor_id), source.tensor_id)
                    != source.tensor_id
                ):
                    raise ValueError("Selected payload contradicts its source or destination")
                version_ids.add(str(checked.id))
        complete = (
            not blocked
            and all(item["status"] == "committed" for item in payloads.values())
            and all(item["status"] == "committed" for item in bindings.values())
        )
        if result["blocked_observed"] is not blocked or result["complete"] is not complete:
            raise ValueError("Selected payload review summary is inconsistent")
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        ValidationError,
        RecursionError,
        TensorusError,
    ) as exc:
        raise ValueError("Invalid or contradictory selected history payload review") from exc


def review_history_payload_selection(
    client: "Client",
    payload: bytes,
    request: HistoryPayloadRestoreRequest,
    *,
    expected_review_hash: str,
    release_id: str | UUID,
    reviewed: dict[str, Any] | None = None,
    _credential: str | None = None,
) -> dict[str, Any]:
    """Observe only one release's required payloads and conditional-write prefixes."""
    credential = _capture_write_credential(client) if _credential is None else _credential
    _require_write_credential(client, credential)
    selected_id = _release_id(release_id)
    manifest, arrays = _read_history_payload_bundle(client, payload)
    del arrays
    request = _checked_request(request)
    _require_conditional_ingestion(client, credential=credential)
    client._require_conditional_npy_historical_receipt(_credential=credential)
    current = _review(client, manifest, request, _credential=credential)
    review = _inspection_review(current, expected_review_hash, reviewed, "datasets")
    if current["review_hash"] != review["review_hash"]:
        raise Conflict("History restore policy changed; review the destination mapping again")
    identity = _checked_reviewed_identity(client, credential, review)
    selection = _selection(manifest, request, review, selected_id)
    states = {item["dataset"]: item for item in review["datasets"]}
    sources = {str(entry.source.id): entry.source for entry in manifest.payloads}
    payloads: dict[str, dict[str, Any]] = {}
    heads: dict[str, dict[str, Any]] = {}
    head_cache: dict[tuple[str, str], dict[str, Any] | None] = {}
    for item in selection["payloads"]:
        identifier = item["source_version_id"]
        source = sources[identifier]
        target = request.destinations[source.id]
        state = states[target.dataset]
        contract = _contract(state)
        condition = NpyWriteCondition(
            dataset_id=state["dataset_id"],
            write_contract_generation=state["active_write_contract"]["generation"],
        )
        evidence = _inspect_copied_payload(
            client,
            identity,
            source.model_dump(mode="json"),
            target,
            item["expected_version"],
            condition,
            contract,
            contract,
            _history_payload_key(
                manifest.content_hash, source.id, target, item["expected_version"]
            ),
            credential=credential,
            allow_tombstoned=True,
        )
        payloads[identifier] = evidence
        expectation = HeadExpectation(
            identifier,
            target.dataset,
            state["dataset_id"],
            target.name,
            item["expected_version"],
        )
        key = (target.dataset, target.name)
        if key not in head_cache:
            head_cache[key] = _read_head(
                client, expectation, str(identity.id), credential=credential
            )
        head = head_cache[key]
        heads[identifier] = {
            "head": head,
            "assessment": _selected_assessment(item["expected_version"], head, evidence),
        }
    bindings: dict[str, dict[str, Any]] = {}
    for item in selection["selected_validity"]:
        data = payloads[item["data_version_id"]]["version"]
        mask = payloads[item["mask_version_id"]]["version"]
        binding = (
            _find_asserted_binding(client, data, mask, identity.id, credential=credential)
            if data is not None and mask is not None
            else None
        )
        bindings[item["source_binding_id"]] = {
            "status": (
                "not_checked"
                if data is None or mask is None
                else "committed"
                if binding is not None
                else "not_observed"
            ),
            "binding": binding,
        }
    if _review(client, manifest, request, _credential=credential) != current:
        raise Conflict("History restore review changed during selected payload inspection")
    blocked = any(
        item["continuation"] == "blocked_policy_changed" for item in payloads.values()
    ) or any(
        item["assessment"] in ("head_advanced", "tombstoned", "receipt_tombstoned")
        for item in heads.values()
    )
    result = {
        "format": REVIEW_FORMAT,
        "selection": selection,
        "selection_hash": canonical_hash(selection),
        "payloads": payloads,
        "validity_bindings": bindings,
        "heads": heads,
        "blocked_observed": blocked,
        "complete": not blocked
        and all(item["status"] == "committed" for item in payloads.values())
        and all(item["status"] == "committed" for item in bindings.values()),
        "point_in_time_atomic": False,
        "retry_guaranteed": False,
        "authorizes_restore": False,
        "source_claims_authenticated": False,
    }
    try:
        validate_history_payload_selection_review(result, manifest, request, review, selected_id)
    except ValueError as exc:
        raise TensorusError(
            "Selected payload observation is inconsistent", code="invalid_response"
        ) from exc
    _checked_write_identity(client, credential, identity)
    return result


def prepare_history_payload_selection(
    client: "Client",
    payload: bytes,
    request: HistoryPayloadRestoreRequest,
    *,
    expected_review_hash: str,
    release_id: str | UUID,
    expected_selection_hash: str,
    confirm_reviewed_selection: bool,
    confirm_unauthenticated_claims: bool,
    confirm_validity_assertions: bool = False,
    reviewed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy only reviewed prefixes and assert selected local validity pairs."""
    if confirm_reviewed_selection is not True:
        raise TensorusError("Explicit confirmation of the reviewed payload selection is required")
    if confirm_unauthenticated_claims is not True:
        raise TensorusError("Explicit confirmation of unauthenticated source claims is required")
    if type(confirm_validity_assertions) is not bool:
        raise TensorusError("Validity assertion confirmation must be a boolean")
    if (
        not isinstance(expected_selection_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_selection_hash) is None
    ):
        raise TensorusError("A valid selected payload review hash is required")
    credential = _capture_write_credential(client)
    request = _checked_request(request)
    before = review_history_payload_selection(
        client,
        payload,
        request,
        expected_review_hash=expected_review_hash,
        release_id=release_id,
        reviewed=reviewed,
        _credential=credential,
    )
    selection = before["selection"]
    if before["selection_hash"] != expected_selection_hash:
        raise Conflict("Selected payload review changed; inspect it again")
    if before["blocked_observed"]:
        raise Conflict("Selected payload destinations or policies conflict; inspect them again")
    if selection["validity_assertions_required"] and not confirm_validity_assertions:
        raise TensorusError("Explicit confirmation of local validity assertions is required")
    if before["complete"]:
        return before

    manifest, arrays = _read_history_payload_bundle(client, payload)
    del arrays
    review = _review(client, manifest, request, _credential=credential)
    if review["review_hash"] != expected_review_hash:
        raise Conflict("History restore policy changed; review the destination mapping again")
    actor = _checked_reviewed_identity(client, credential, review)
    states = {item["dataset"]: item for item in review["datasets"]}
    contracts = {name: _contract(state) for name, state in states.items()}
    conditions = {
        name: NpyWriteCondition(
            dataset_id=state["dataset_id"],
            write_contract_generation=state["active_write_contract"]["generation"],
        )
        for name, state in states.items()
    }

    def guard(dataset: str) -> None:
        identity = _checked_write_identity(client, credential, actor)
        _, state = _destination_state(client, dataset, identity.id, credential=credential)
        if state != states[dataset]:
            raise Conflict("History restore policy changed during selected payload preparation")
        _require_write_credential(client, credential)

    try:
        copied = copy_history_payload_bundle(
            client,
            payload,
            request.dataset,
            request.model_dump(mode="json")["destinations"],
            _guard=guard,
            _expected_contracts=contracts,
            _conditions=conditions,
            _credential=credential,
            _expected_identity=actor,
            _source_ids={UUID(item["source_version_id"]) for item in selection["payloads"]},
        )["destination_versions"]
        for item in selection["selected_validity"]:
            data, mask = copied[item["data_version_id"]], copied[item["mask_version_id"]]
            guard(data["dataset"])
            guard(mask["dataset"])
            response = client.create_validity_binding(
                ValidityBindingCreate(data_version_id=data["id"], mask_version_id=mask["id"]),
                _credential=credential,
            )
            binding = _checked_asserted_binding(response, data, mask, actor.id)
            stored = _read_model(
                client,
                f"/v1/validity-bindings/{binding['id']}",
                ValidityBinding,
                "Validity binding",
                MAX_BUNDLE_VERSION_RESPONSE_BYTES,
                credential=credential,
            )
            if canonical_json(stored.model_dump(mode="json")) != canonical_json(binding):
                raise TensorusError("Destination returned contradictory retained validity evidence")
            guard(data["dataset"])
            guard(mask["dataset"])
        after = review_history_payload_selection(
            client,
            payload,
            request,
            expected_review_hash=expected_review_hash,
            release_id=release_id,
            reviewed=reviewed,
            _credential=credential,
        )
        if (
            after["selection_hash"] != expected_selection_hash
            or after["blocked_observed"]
            or not after["complete"]
        ):
            raise TensorusError("Selected payload preparation is not completely observed")
        return after
    except TensorusError as exc:
        raise TensorusError(
            f"Selected payload preparation may have committed; {exc}",
            code=exc.code,
            status=exc.status,
            details=exc.details,
        ) from exc
