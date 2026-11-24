"""Reviewed continuation of one already-copied history release."""

import re
from typing import TYPE_CHECKING, Any
from urllib.parse import quote
from uuid import UUID

from pydantic import ValidationError

from tensorus.arrays import canonical_hash, canonical_json, decode_unique_json
from tensorus.bundle_restore_names import (
    HistoryNames,
    HistoryNamesEntry,
    NameOccupancy,
    NameOccupancyRequest,
    observe_destination_names,
)
from tensorus.errors import Conflict, LimitExceeded, TensorusError
from tensorus.history_payload_bundle import HistoryPayloadManifest, _read_history_payload_bundle
from tensorus.history_payload_restore import (
    HistoryPayloadRestoreRequest,
    _checked_request,
    _history_destination,
    _review,
    inspect_history_payload_restore,
    validate_history_release_observation,
)
from tensorus.product_bundle import (
    MAX_BUNDLE_VERSION_RESPONSE_BYTES,
    _capture_write_credential,
    _checked_reviewed_identity,
    _checked_write_identity,
    _inspection_review,
    _require_write_credential,
)
from tensorus.release_portability import MAX_PORTABLE_BYTES, ImportedRelease, ReleaseHistoryImport

if TYPE_CHECKING:
    from tensorus.client import Client


SELECTION_FORMAT = "tensorus-history-restore-selection-v1"
REVIEW_FORMAT = "tensorus-history-restore-selection-review-v1"


def _release_id(value: str | UUID) -> UUID:
    try:
        result = UUID(str(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise TensorusError("A valid source release ID is required") from exc
    if str(result) != str(value):
        raise TensorusError("The source release ID must be canonical")
    return result


def _selection(
    manifest: HistoryPayloadManifest,
    request: HistoryPayloadRestoreRequest,
    review: dict[str, Any],
    observation: dict[str, Any],
    release_id: UUID,
) -> tuple[dict[str, Any], NameOccupancyRequest]:
    release = next(item for item in manifest.source.releases if item.release.id == release_id)
    choice = next(item for item in request.releases if item.release_id == release_id)
    selected = observation["releases"][str(release_id)]
    copied = {
        key: value["version"]
        for key, value in observation["payload_observation"]["payloads"].items()
        if value["status"] == "committed"
    }
    wanted = _history_destination(release, choice, copied).import_request(
        release, manifest.source.manifest
    )
    encoded = wanted.model_dump(mode="json")
    if len(canonical_json(encoded).encode()) > MAX_PORTABLE_BYTES:
        raise LimitExceeded("The selected history import exceeds its byte limit")
    selection = {
        "format": SELECTION_FORMAT,
        "restore_review_hash": review["review_hash"],
        "bundle_hash": manifest.content_hash,
        "tenant_id": review["tenant_id"],
        "actor_key_id": review["actor_key_id"],
        "actor_role": review["actor_role"],
        "dataset": request.dataset,
        "release_id": str(release_id),
        "validity_assertions_required": (
            request.assert_selected_validity and bool(selected["requirements"]["validity_bindings"])
        ),
        "selected_validity_bindings": {
            identifier: observation["payload_observation"]["validity_bindings"][identifier]
            for identifier in selected["requirements"]["validity_bindings"]
        },
        "import_request": encoded,
    }
    names = NameOccupancyRequest(
        kind="history",
        history=HistoryNames(
            dataset=request.dataset,
            releases=[
                HistoryNamesEntry(
                    release_id=release_id, name=choice.name, snapshot_name=choice.snapshot_name
                )
            ],
        ),
    )
    return selection, names


def _matching_names(observation: dict[str, Any], names: NameOccupancy, release_id: UUID) -> bool:
    if names.history is None:
        return False
    selected = observation["releases"][str(release_id)]
    occupant = names.history[release_id]
    if selected["status"] == "not_observed":
        return (
            occupant.import_id is None
            and occupant.import_snapshot_id is None
            and occupant.snapshot_id is None
        )
    if selected["status"] == "committed":
        retained = selected["record"]
        return (
            str(occupant.import_id) == retained["id"]
            and str(occupant.import_snapshot_id) == retained["snapshot_id"]
            and str(occupant.snapshot_id) == retained["snapshot_id"]
        )
    return False


def validate_history_selection_review(
    result: object,
    manifest: HistoryPayloadManifest,
    request: HistoryPayloadRestoreRequest,
    review: dict[str, Any],
    release_id: UUID,
) -> None:
    """Validate a selection against its original archive, review and observed receipts."""
    try:
        if (
            not isinstance(result, dict)
            or set(result)
            != {
                "format",
                "selection",
                "selection_hash",
                "release_observation",
                "name_observation",
                "names_reserved",
                "point_in_time_atomic",
                "authorizes_restore",
            }
            or result["format"] != REVIEW_FORMAT
            or any(
                result[field] is not False
                for field in ("names_reserved", "point_in_time_atomic", "authorizes_restore")
            )
        ):
            raise ValueError("Invalid selected history review")
        observed = result["release_observation"]
        validate_history_release_observation(observed, manifest, request, review)
        if (
            observed["current_review_hash"] != review["review_hash"]
            or observed["payload_observation"]["continuation"] == "blocked_policy_changed"
            or observed["releases"][str(release_id)]["status"] not in ("not_observed", "committed")
        ):
            raise ValueError("Selected history is not eligible for continuation")
        expected, names_request = _selection(manifest, request, review, observed, release_id)
        names_raw = result["name_observation"]
        names = NameOccupancy.model_validate_json(canonical_json(names_raw), strict=True)
        dataset_id = next(
            item["dataset_id"] for item in review["datasets"] if item["dataset"] == request.dataset
        )
        if (
            canonical_json(names_raw) != canonical_json(names.model_dump(mode="json"))
            or names.request != names_request
            or str(names.tenant_id) != review["tenant_id"]
            or str(names.actor_key_id) != review["actor_key_id"]
            or names.actor_role != review["actor_role"]
            or str(names.dataset_id) != dataset_id
            or not _matching_names(observed, names, release_id)
            or canonical_json(result["selection"]) != canonical_json(expected)
            or result["selection_hash"] != canonical_hash(expected)
        ):
            raise ValueError("Selected history review contradicts its evidence")
    except (
        AttributeError,
        KeyError,
        StopIteration,
        TensorusError,
        TypeError,
        ValueError,
        ValidationError,
        RecursionError,
    ) as exc:
        raise ValueError("Invalid or contradictory selected history review") from exc


def review_history_restore_selection(
    client: "Client",
    payload: bytes,
    request: HistoryPayloadRestoreRequest,
    *,
    expected_review_hash: str,
    release_id: str | UUID,
    reviewed: dict[str, Any] | None = None,
    _credential: str | None = None,
) -> dict[str, Any]:
    """Review one ready release without reserving names or performing writes."""
    if not isinstance(payload, bytes):
        raise TensorusError("A history bundle archive is required")
    selected_id = _release_id(release_id)
    if reviewed is not None:
        reviewed = _inspection_review(reviewed, expected_review_hash, reviewed, "datasets")
    credential = _capture_write_credential(client) if _credential is None else _credential
    _require_write_credential(client, credential)
    request = _checked_request(request)
    manifest, arrays = _read_history_payload_bundle(client, payload)
    del arrays
    if selected_id not in {item.release.id for item in manifest.source.releases}:
        raise TensorusError("The selected release is not in this history bundle")
    current = _review(client, manifest, request, _credential=credential)
    if current["review_hash"] != expected_review_hash:
        raise Conflict("History payload restore review changed; inspect it again")
    observed = inspect_history_payload_restore(
        client,
        payload,
        request,
        expected_review_hash=expected_review_hash,
        reviewed=reviewed,
        _release_observation=True,
        _credential=credential,
    )
    if (
        observed["current_review_hash"] != expected_review_hash
        or observed["payload_observation"]["continuation"] == "blocked_policy_changed"
    ):
        raise Conflict("History payload restore policy changed; select new destinations")
    status = observed["releases"][str(selected_id)]["status"]
    if status == "not_checked":
        raise Conflict("Selected release payloads or validity bindings are not ready")
    if status == "conflicting":
        raise Conflict("Selected release name contains a different retained import")
    selection, names_request = _selection(manifest, request, current, observed, selected_id)
    names_raw = observe_destination_names(client, names_request, _credential=credential)
    names = NameOccupancy.model_validate(names_raw)
    if not _matching_names(observed, names, selected_id):
        raise Conflict("Selected release or snapshot name is occupied or changed")
    if _review(client, manifest, request, _credential=credential) != current:
        raise Conflict("History payload restore review changed during selection")
    result = {
        "format": REVIEW_FORMAT,
        "selection": selection,
        "selection_hash": canonical_hash(selection),
        "release_observation": observed,
        "name_observation": names_raw,
        "names_reserved": False,
        "point_in_time_atomic": False,
        "authorizes_restore": False,
    }
    try:
        validate_history_selection_review(result, manifest, request, current, selected_id)
    except ValueError as exc:
        raise TensorusError(
            "Selected history review is inconsistent", code="invalid_response"
        ) from exc
    _checked_reviewed_identity(client, credential, current)
    return result


def _write_selected_import(
    client: "Client",
    request: HistoryPayloadRestoreRequest,
    selection: dict[str, Any],
    credential: str,
) -> dict[str, Any]:
    wanted = ReleaseHistoryImport.model_validate(selection["import_request"])
    try:
        data = client._bounded_report(
            "POST",
            f"/v1/datasets/{quote(request.dataset, safe='')}/release-history/imports",
            MAX_PORTABLE_BYTES + MAX_BUNDLE_VERSION_RESPONSE_BYTES,
            "Selected imported history",
            headers={"Authorization": credential},
            json=wanted.model_dump(mode="json"),
        )
    except TensorusError as exc:
        raise TensorusError(
            f"Selected import may have committed; {exc}",
            code=exc.code,
            status=exc.status,
            details=exc.details,
        ) from exc
    try:
        raw = decode_unique_json(data)
        imported = ImportedRelease.model_validate_json(data, strict=True)
        if (
            canonical_json(raw) != canonical_json(imported.model_dump(mode="json"))
            or canonical_json(imported.request.model_dump(mode="json"))
            != canonical_json(wanted.model_dump(mode="json"))
            or imported.request_hash != canonical_hash(wanted.model_dump(mode="json"))
            or str(imported.tenant_id) != selection["tenant_id"]
            or imported.dataset != selection["dataset"]
            or imported.name != wanted.name
        ):
            raise ValueError("Selected imported history contradicts the request")
    except (UnicodeDecodeError, ValueError, RecursionError, TensorusError) as exc:
        raise TensorusError(
            "Selected import may have committed; destination returned invalid import evidence"
        ) from exc
    return raw


def apply_history_restore_selection(
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
    """Apply one exact selection; the original bundle restore remains non-atomic."""
    if confirm_reviewed_selection is not True:
        raise TensorusError("Explicit confirmation of the reviewed selection is required")
    if confirm_unauthenticated_claims is not True:
        raise TensorusError("Explicit confirmation of unauthenticated source claims is required")
    if type(confirm_validity_assertions) is not bool:
        raise TensorusError("Validity assertion confirmation must be a boolean")
    if (
        not isinstance(expected_selection_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_selection_hash) is None
    ):
        raise TensorusError("A valid selected history review hash is required")
    if reviewed is not None:
        reviewed = _inspection_review(reviewed, expected_review_hash, reviewed, "datasets")
    credential = _capture_write_credential(client)
    request = _checked_request(request)
    before = review_history_restore_selection(
        client,
        payload,
        request,
        expected_review_hash=expected_review_hash,
        release_id=release_id,
        reviewed=reviewed,
        _credential=credential,
    )
    if before["selection_hash"] != expected_selection_hash:
        raise Conflict("Selected history review changed; inspect it again")
    selection = before["selection"]
    identity = _checked_reviewed_identity(client, credential, selection)
    if selection["validity_assertions_required"] and confirm_validity_assertions is not True:
        raise TensorusError("Explicit confirmation of local validity assertions is required")
    selected_id = _release_id(release_id)
    if before["release_observation"]["releases"][str(selected_id)]["status"] == "committed":
        return before
    raw = _write_selected_import(client, request, selection, credential)
    try:
        _checked_write_identity(client, credential, identity)
        after = review_history_restore_selection(
            client,
            payload,
            request,
            expected_review_hash=expected_review_hash,
            release_id=selected_id,
            reviewed=reviewed,
            _credential=credential,
        )
    except TensorusError as exc:
        raise TensorusError(
            f"Selected import may have committed; {exc}",
            code=exc.code,
            status=exc.status,
            details=exc.details,
        ) from exc
    retained = after["release_observation"]["releases"][str(selected_id)]
    if (
        after["selection_hash"] != expected_selection_hash
        or retained["status"] != "committed"
        or canonical_json(retained["record"]) != canonical_json(raw)
    ):
        raise TensorusError("Selected import may have committed; inspect it again")
    return after
